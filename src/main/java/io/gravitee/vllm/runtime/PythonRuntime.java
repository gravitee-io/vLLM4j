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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Manages the embedded CPython interpreter lifecycle.
 *
 * <p>Handles:
 * <ul>
 *   <li>Setting backend-specific environment variables before init</li>
 *   <li>Calling {@code Py_InitializeEx(0)}</li>
 *   <li>Fixing {@code sys.path} to include the venv's site-packages</li>
 *   <li>Fixing {@code sys.executable} to the venv's Python binary</li>
 * </ul>
 *
 * <h2>Interpreter lifecycle</h2>
 * <p>The CPython interpreter is a <em>process-wide singleton</em>. Once
 * initialized via {@code Py_InitializeEx}, it remains alive for the lifetime
 * of the JVM. {@link #close()} is intentionally a no-op — calling
 * {@code Py_FinalizeEx} would trigger SIGABRT from PyTorch/vLLM atexit
 * handlers, and the interpreter cannot be safely re-initialized afterward.
 *
 * <p>GPU memory is released by {@code VllmEngine.close()}, which decrefs all
 * Python objects, runs {@code gc.collect()}, and flushes the CUDA/MPS caching
 * allocator. The ~100-300 MB CUDA context overhead persists but is reused by
 * subsequent model loads.
 *
 * <h2>Keepalive thread</h2>
 * <p>A dedicated daemon thread periodically acquires the GIL and performs a
 * trivial CPython operation ({@code id(None)}). This prevents the CUDA context
 * and Python thread state from going stale during long idle periods — without
 * it, the JVM may reap the thread that called {@code Py_InitializeEx},
 * causing {@code PyGILState_Ensure} to operate on a dead thread state, which
 * leads to segfaults on the next engine call.
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

    /** Keepalive interval — how often to ping CPython (30 seconds). */
    private static final long KEEPALIVE_INTERVAL_NS = 30_000_000_000L;

    /** Keepalive daemon thread. Started once on first initialization. */
    private static volatile Thread keepaliveThread;

    /** Flag to stop the keepalive thread (set when last engine closes). */
    private static final AtomicBoolean KEEPALIVE_STOPPED = new AtomicBoolean(false);

    /** Number of live VllmEngine instances. Keepalive runs while > 0. */
    private static final AtomicInteger ENGINE_COUNT = new AtomicInteger(0);

    /**
     * Registers a new VllmEngine instance. Starts the keepalive thread
     * if this is the first live engine.
     *
     * <p>Called from the {@code VllmEngine} constructor.
     */
    public static void registerEngine() {
        if (ENGINE_COUNT.incrementAndGet() == 1) {
            startKeepalive();
        }
    }

    /**
     * Unregisters a VllmEngine instance. Stops the keepalive thread
     * when the last engine closes.
     *
     * <p>Called from {@code VllmEngine.close()} <em>before</em> any
     * Python object teardown — this ensures the keepalive thread is
     * no longer touching CPython when {@code shutdownEngineCore()} and
     * {@code decref()} run, which may tear down vLLM's background
     * threads and invalidate Python thread states.
     */
    public static void unregisterEngine() {
        if (ENGINE_COUNT.decrementAndGet() <= 0) {
            ENGINE_COUNT.set(0); // clamp to 0
            stopKeepalive();
        }
    }

    /**
     * Returns {@code true} if CPython has been initialized via
     * {@code Py_InitializeEx} and the GIL has been released.
     *
     * <p>Callers that need CPython but cannot guarantee initialization
     * ordering (e.g. pre-flight memory checks) should test this before
     * calling {@link GIL#acquire()}.
     */
    public static boolean isInitialized() {
        return INITIALIZED.get();
    }

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
            // Keepalive is NOT started here — it is started by registerEngine()
            // when the first VllmEngine is created, and stopped by unregisterEngine()
            // when the last VllmEngine closes.
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

        // The CPython interpreter is a process-wide singleton — it stays alive
        // for the lifetime of the JVM.  Individual VllmEngine instances release
        // their Python objects and GPU memory in VllmEngine.close(); the
        // interpreter itself is never finalized because:
        //
        //   1. Py_FinalizeEx() triggers SIGABRT from PyTorch/vLLM atexit
        //      handlers and native thread teardown.
        //   2. Py_InitializeEx() cannot be safely called again after finalize.
        //   3. The CUDA context (~100-300 MB) persists anyway and is reused
        //      by the next model load, so keeping the interpreter alive is
        //      effectively free.
        //
        // See: https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
    }

    // ── Keepalive thread ───────────────────────────────────────────────────

    /**
     * Starts the keepalive daemon thread if not already running.
     *
     * <p>The thread periodically acquires the GIL and calls {@code id(None)} —
     * a trivial CPython operation that keeps the Python thread state machinery
     * and CUDA context warm. Without this, long idle periods cause the JVM to
     * reap the thread that called {@code Py_InitializeEx}, leading to segfaults
     * when {@code PyGILState_Ensure} is later invoked from a different thread.
     *
     * <p>Restartable: if the keepalive was previously stopped (last engine
     * closed), calling this again will start a fresh keepalive thread.
     */
    private static synchronized void startKeepalive() {
        if (keepaliveThread != null && keepaliveThread.isAlive()) return;
        KEEPALIVE_STOPPED.set(false);
        keepaliveThread = new Thread(PythonRuntime::keepaliveLoop, "vllm4j-keepalive");
        keepaliveThread.setDaemon(true);
        keepaliveThread.start();
    }

    /**
     * Stops the keepalive daemon thread and waits for it to exit.
     *
     * <p>Called when the last {@code VllmEngine} closes. The thread must be
     * fully stopped <em>before</em> the engine tears down Python objects,
     * because vLLM's shutdown may invalidate Python thread states that the
     * keepalive thread uses via {@code PyGILState_Ensure}.
     */
    private static synchronized void stopKeepalive() {
        KEEPALIVE_STOPPED.set(true);
        Thread t = keepaliveThread;
        if (t != null) {
            LockSupport.unpark(t); // wake it if parked
            try {
                t.join(5_000); // wait up to 5 seconds
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            keepaliveThread = null;
        }
    }

    /**
     * Keepalive loop: parks for {@link #KEEPALIVE_INTERVAL_NS}, then pings
     * CPython. Runs until {@link #KEEPALIVE_STOPPED} is set.
     */
    private static void keepaliveLoop() {
        while (!KEEPALIVE_STOPPED.get()) {
            LockSupport.parkNanos(KEEPALIVE_INTERVAL_NS);
            if (KEEPALIVE_STOPPED.get()) break;
            pingInterpreter();
        }
    }

    /**
     * Acquires the GIL and calls {@code id(None)} — a trivial, side-effect-free
     * CPython operation. This keeps the interpreter thread state valid and
     * prevents the CUDA context from going stale.
     *
     * <p>Best-effort — silently ignores any errors.
     */
    private static void pingInterpreter() {
        try (var gil = GIL.acquire(); Arena tmp = Arena.ofConfined()) {
            // Import builtins and call id(None) — trivial, side-effect-free.
            // This touches the interpreter state and validates the thread state.
            MemorySegment builtins = CPython.PyImport_ImportModule(tmp.allocateFrom("builtins"));
            if (!PythonTypes.isNull(builtins)) {
                PythonTypes.decref(builtins);
            }
            CPython.PyErr_Clear();
        } catch (Exception e) {
            // Ignore — best-effort keepalive
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
     *
     * <p>Also useful to inject credentials such as {@code HF_TOKEN} before
     * the vLLM Python engine initialises and downloads gated models.
     */
    public static void setEnv(String name, String value) {
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
