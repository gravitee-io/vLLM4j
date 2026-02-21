package io.gravitee.vllm.runtime;

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.platform.VllmBackend;

import org.vllm.python.CPython;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Manages the embedded CPython interpreter lifecycle.
 *
 * <p>Handles:
 * <ul>
 *   <li>Setting backend-specific environment variables before init</li>
 *   <li>Calling {@code Py_InitializeEx(0)}</li>
 *   <li>Fixing {@code sys.path} to include the venv's site-packages</li>
 *   <li>Fixing {@code sys.executable} to the venv's Python binary</li>
 *   <li>Registering a shutdown hook to hard-exit the process on close,
 *       bypassing native atexit handlers that would otherwise SIGABRT</li>
 * </ul>
 *
 * <h2>GIL contract</h2>
 * <p>The GIL is acquired during {@code Py_InitializeEx}. After initialization
 * completes (sys.path setup, sys.executable fix), the GIL is <em>released</em>
 * via {@code PyEval_SaveThread()} so that any Java thread can acquire it using
 * {@link GIL#acquire()}. All subsequent CPython calls from any thread must be
 * wrapped in a {@code try (var gil = GIL.acquire()) { ... }} block.
 *
 * @see VllmBackend#envVars()
 */
public final class PythonRuntime implements AutoCloseable {

    /** Tracks whether CPython has been initialized (and GIL released). */
    private static final AtomicBoolean INITIALIZED = new AtomicBoolean(false);

    private volatile boolean closed = false;

    /**
     * Saved thread state from {@code PyEval_SaveThread()}.
     * Must be restored before finalization.
     */
    private MemorySegment savedThreadState;

    /**
     * Initializes the CPython interpreter for the given venv and backend.
     *
     * <p>If CPython was already initialized by a prior {@code PythonRuntime}
     * (which released the GIL via {@code PyEval_SaveThread}), we re-acquire
     * the GIL before performing sys.path setup and release it again on exit.
     *
     * @param venvPath absolute path to the {@code .venv} directory
     * @param backend  the compute backend (determines env vars)
     */
    public PythonRuntime(Path venvPath, VllmBackend backend) {
        String venv = venvPath.toAbsolutePath().toString();

        // Load libpython before touching CPython class
        PythonLibLoader.ensureLoaded(venvPath.toAbsolutePath().getParent());

        // PYTHONHOME must point to the *base* Python installation (where the stdlib
        // lives), NOT the venv directory.
        String pythonHome = resolvePythonHome(venv);
        setEnv("PYTHONHOME", pythonHome);

        // Set backend-specific env vars before Py_Initialize
        for (Map.Entry<String, String> entry : backend.envVars().entrySet()) {
            setEnv(entry.getKey(), entry.getValue());
        }

        // Check if Python is already initialized (a prior PythonRuntime released the GIL).
        boolean alreadyInitialized = INITIALIZED.get();

        if (!alreadyInitialized) {
            // First init — Py_InitializeEx implicitly acquires the GIL on this thread.
            CPython.Py_InitializeEx(0);
            INITIALIZED.set(true);
        }

        // If already initialized, a prior PythonRuntime released the GIL.
        // We must acquire it before calling any CPython API.
        // If freshly initialized, PyGILState_Ensure is a no-op (we already hold it).
        try (var gil = GIL.acquire()) {
            // Prepend the venv's site-packages to sys.path
            String pyVer = resolvePythonVersionTag();
            prependSysPath(venv + "/lib/" + pyVer + "/site-packages");
            prependSysPath(venv + "/lib/" + pyVer);

            // Fix sys.executable — CPython inherits the JVM's argv[0]
            fixSysExecutable(venv);
        }

        // Release the GIL so any Java thread can acquire it via GIL.acquire().
        // On first init this transitions from "GIL held by this thread" to "GIL free".
        // On subsequent inits this is redundant but harmless (GIL.acquire/close above
        // already released it; PyEval_SaveThread will save the current thread state).
        if (!alreadyInitialized) {
            savedThreadState = CPython.PyEval_SaveThread();
        }
    }

    /** Returns true if the interpreter has been finalized. */
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // We deliberately skip Py_FinalizeEx() here.
        //
        // vLLM / PyTorch / MLX native extensions call abort() inside their
        // atexit handlers or background-thread teardown, which sends SIGABRT
        // to the process (exit code 134).  macOS abort() resets the SIGABRT
        // disposition to SIG_DFL before raising it, so signal(SIGABRT, SIG_IGN)
        // cannot prevent it.
        //
        // Instead we register a JVM shutdown hook that calls _exit(0) to
        // hard-terminate the process before Python's daemon threads or atexit
        // handlers get a chance to abort().  This is safe because:
        //   - All test results have already been reported to Surefire
        //   - The OS reclaims all process resources on _exit
        //   - _exit skips C atexit handlers entirely (including the problematic ones)
        //
        // See: https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
        Runtime.getRuntime().addShutdownHook(new Thread(() -> hardExit(0),
                "vllm4j-hard-exit"));
    }

    // ── Hard exit helper ────────────────────────────────────────────────────

    /**
     * Calls POSIX {@code _exit(status)} to terminate the process immediately,
     * bypassing C atexit handlers and Python finalizers.
     */
    private static void hardExit(int status) {
        try {
            var linker  = Linker.nativeLinker();
            var lookup  = linker.defaultLookup();
            var exitAddr = lookup.find("_exit").orElseThrow();
            var desc = FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT);
            var exitHandle = linker.downcallHandle(exitAddr, desc);
            exitHandle.invokeExact(status);
        } catch (Throwable t) {
            // Last resort — Runtime.halt also skips shutdown hooks (we're
            // already inside one) and C atexit handlers on most JVMs.
            Runtime.getRuntime().halt(0);
        }
    }

    // ── sys.path / sys.executable fixes ────────────────────────────────────

    /**
     * Prepends a path string to {@code sys.path} at index 0.
     */
    private void prependSysPath(String path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sysModule = CPython.PyImport_ImportModule(arena.allocateFrom("sys"));
            if (PythonTypes.isNull(sysModule)) { CPython.PyErr_Clear(); return; }

            MemorySegment sysPath    = PythonTypes.getAttr(arena, sysModule, "path");
            MemorySegment insertName = PythonTypes.pyStr(arena, "insert");
            MemorySegment pyIndex    = CPython.PyLong_FromLong(0L);
            MemorySegment pyPath     = PythonTypes.pyStr(arena, path);
            MemorySegment result     = PythonCall.callMethodObjArgs(sysPath, insertName, pyIndex, pyPath);
            PythonTypes.decref(result);
            PythonTypes.decref(pyPath);
            PythonTypes.decref(pyIndex);
            PythonTypes.decref(insertName);
            PythonTypes.decref(sysPath);
            PythonTypes.decref(sysModule);
        }
    }

    /**
     * Sets {@code sys.executable} to the venv's Python binary.
     */
    private void fixSysExecutable(String venvPath) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment sysModule = CPython.PyImport_ImportModule(tmp.allocateFrom("sys"));
            if (PythonTypes.isNull(sysModule)) { CPython.PyErr_Clear(); return; }

            String pyBin = venvPath + "/bin/python";
            if (!new java.io.File(pyBin).exists()) {
                pyBin = venvPath + "/bin/python3";
            }

            MemorySegment pyExec = CPython.PyUnicode_FromString(tmp.allocateFrom(pyBin));
            CPython.PyObject_SetAttrString(sysModule, tmp.allocateFrom("executable"), pyExec);
            CPython.Py_DecRef(pyExec);
            CPython.Py_DecRef(sysModule);
        }
    }

    // ── Python version resolution ──────────────────────────────────────────

    /**
     * Returns the Python version tag for the running interpreter, e.g. {@code "python3.12"}.
     */
    private String resolvePythonVersionTag() {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment sysModule = CPython.PyImport_ImportModule(tmp.allocateFrom("sys"));
            if (PythonTypes.isNull(sysModule)) return "python3.12";
            MemorySegment versionInfo = CPython.PyObject_GetAttrString(sysModule, tmp.allocateFrom("version_info"));
            PythonTypes.decref(sysModule);
            if (PythonTypes.isNull(versionInfo)) return "python3.12";
            MemorySegment major = CPython.PySequence_GetItem(versionInfo, 0);
            MemorySegment minor = CPython.PySequence_GetItem(versionInfo, 1);
            PythonTypes.decref(versionInfo);
            long maj = PythonTypes.isNull(major) ? 3 : CPython.PyLong_AsLong(major);
            long min = PythonTypes.isNull(minor) ? 12 : CPython.PyLong_AsLong(minor);
            PythonTypes.decref(major);
            PythonTypes.decref(minor);
            return "python" + maj + "." + min;
        }
    }

    // ── PYTHONHOME resolution ──────────────────────────────────────────────

    /**
     * Resolves the base Python prefix to use as {@code PYTHONHOME}.
     *
     * <p>Reads {@code .python-home} from the project directory, then falls back
     * to parsing {@code pyvenv.cfg}.
     */
    static String resolvePythonHome(String venvPath) {
        java.nio.file.Path venv = java.nio.file.Path.of(venvPath).toAbsolutePath();
        java.nio.file.Path projectDir = venv.getParent();

        // 1. Try .python-home
        if (projectDir != null) {
            java.nio.file.Path dotFile = projectDir.resolve(".python-home");
            if (java.nio.file.Files.exists(dotFile)) {
                try {
                    String home = java.nio.file.Files.readString(dotFile).strip();
                    if (!home.isEmpty() && java.nio.file.Files.isDirectory(java.nio.file.Path.of(home))) {
                        return home;
                    }
                } catch (java.io.IOException ignored) {}
            }
        }

        // 2. Fallback: parse pyvenv.cfg
        java.nio.file.Path pyvenvCfg = venv.resolve("pyvenv.cfg");
        if (java.nio.file.Files.exists(pyvenvCfg)) {
            try {
                for (String line : java.nio.file.Files.readAllLines(pyvenvCfg)) {
                    if (line.startsWith("home")) {
                        String[] parts = line.split("=", 2);
                        if (parts.length == 2) {
                            java.nio.file.Path home = java.nio.file.Path.of(parts[1].strip());
                            java.nio.file.Path prefix = home.getParent();
                            if (prefix != null && java.nio.file.Files.isDirectory(prefix)) {
                                return prefix.toString();
                            }
                        }
                    }
                }
            } catch (java.io.IOException ignored) {}
        }

        throw new VllmException(
                "Cannot determine PYTHONHOME (base Python prefix). " +
                "Run 'mvn generate-sources -P <profile>' to regenerate .python-home, " +
                "or set the PYTHONHOME environment variable before launching the JVM.");
    }

    // ── Environment variable helper ────────────────────────────────────────

    /**
     * Sets an environment variable in-process via {@code setenv(3)}.
     * Must be called before {@code Py_InitializeEx} for CPython to pick up
     * variables like {@code PYTHONHOME}.
     */
    static void setEnv(String name, String value) {
        var lookup = Linker.nativeLinker().defaultLookup();
        var setenvAddr = lookup.find("setenv").orElseThrow(() ->
                new VllmException("Cannot find setenv() in libc"));
        var desc = FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT
        );
        var setenvHandle = Linker.nativeLinker().downcallHandle(setenvAddr, desc);
        try (Arena tmp = Arena.ofConfined()) {
            int rc = (int) setenvHandle.invokeExact(
                    tmp.allocateFrom(name),
                    tmp.allocateFrom(value),
                    1   // overwrite = true
            );
            if (rc != 0) {
                throw new VllmException("setenv(" + name + ") returned " + rc);
            }
        } catch (VllmException ex) {
            throw ex;
        } catch (Throwable t) {
            throw new VllmException("setenv(" + name + ") failed", t);
        }
    }
}
