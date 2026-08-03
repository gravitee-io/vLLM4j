/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.vllm.runtime;

import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.platform.VllmBackend;
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
  private static final AtomicBoolean KEEPALIVE_STOPPED = new AtomicBoolean(
    false
  );

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
    PythonLibLoader.ensureLoaded(venvPath.toAbsolutePath());

    // PYTHONHOME must point to the *base* Python installation (where the stdlib
    // lives), NOT the venv directory.
    String pythonHome = resolvePythonHome(venv);
    setEnv("PYTHONHOME", pythonHome);

    // Wire up the venv's build toolchain (PATH + CUDA_HOME) so flashinfer can
    // JIT-compile CUDA kernels at engine init. Must run before Py_InitializeEx
    // so the os module snapshots the updated environment into os.environ, which
    // subprocess consults for executable resolution.
    configureBuildToolchain(venv);

    // Set backend-specific env vars before Py_Initialize.
    //
    // Defaults, not overrides: a value already in the environment was put there
    // deliberately, and silently replacing it leaves no way to re-test a setting
    // we default off because a backend is buggy.
    for (Map.Entry<String, String> entry : backend.envVars().entrySet()) {
      String existing = System.getenv(entry.getKey());
      if (existing != null && !existing.isBlank()) {
        continue;
      }
      setEnv(entry.getKey(), entry.getValue());
    }

    // Check if Python is already initialized (a prior PythonRuntime released the GIL).
    boolean alreadyInitialized = INITIALIZED.get();

    if (!alreadyInitialized) {
      // First init — Py_InitializeEx implicitly acquires the GIL on this thread.
      CPythonBinding.Py_InitializeEx(0);
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
      savedThreadState = CPythonBinding.PyEval_SaveThread();
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
    keepaliveThread = new Thread(
      PythonRuntime::keepaliveLoop,
      "vllm4j-keepalive"
    );
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
   * CPythonBinding. Runs until {@link #KEEPALIVE_STOPPED} is set.
   */
  private static void keepaliveLoop() {
    while (!KEEPALIVE_STOPPED.get()) {
      LockSupport.parkNanos(KEEPALIVE_INTERVAL_NS);
      if (KEEPALIVE_STOPPED.get()) break;
      pingInterpreter();
    }
  }

  /**
   * Acquires the GIL and touches {@code Py_None} — the cheapest possible
   * CPython operation. This keeps the interpreter thread state valid and
   * prevents the CUDA context from going stale.
   *
   * <p>Best-effort — silently ignores any errors.
   */
  private static void pingInterpreter() {
    try (var gil = GIL.acquire()) {
      MemorySegment none = PythonTypes.pyNone();
      CPythonBinding.Py_IncRef(none);
      CPythonBinding.Py_DecRef(none);
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
      MemorySegment sysModule = CPythonBinding.PyImport_ImportModule(
        arena.allocateFrom("sys")
      );
      if (PythonTypes.isNull(sysModule)) {
        CPythonBinding.PyErr_Clear();
        return;
      }

      MemorySegment sysPath = PythonTypes.getAttr(arena, sysModule, "path");
      MemorySegment insertName = PythonTypes.pyStr(arena, "insert");
      MemorySegment pyIndex = CPythonBinding.PyLong_FromLong(0L);
      MemorySegment pyPath = PythonTypes.pyStr(arena, path);
      MemorySegment result = PythonCall.callMethodObjArgs(
        sysPath,
        insertName,
        pyIndex,
        pyPath
      );
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
      MemorySegment sysModule = CPythonBinding.PyImport_ImportModule(
        tmp.allocateFrom("sys")
      );
      if (PythonTypes.isNull(sysModule)) {
        CPythonBinding.PyErr_Clear();
        return;
      }

      String pyBin = venvPath + "/bin/python";
      if (!new java.io.File(pyBin).exists()) {
        pyBin = venvPath + "/bin/python3";
      }

      MemorySegment pyExec = CPythonBinding.PyUnicode_FromString(
        tmp.allocateFrom(pyBin)
      );
      CPythonBinding.PyObject_SetAttrString(
        sysModule,
        tmp.allocateFrom("executable"),
        pyExec
      );
      CPythonBinding.Py_DecRef(pyExec);
      CPythonBinding.Py_DecRef(sysModule);
    }
  }

  // ── Python version resolution ──────────────────────────────────────────

  /**
   * Returns the Python version tag for the running interpreter, e.g. {@code "python3.12"}.
   */
  private String resolvePythonVersionTag() {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment sysModule = CPythonBinding.PyImport_ImportModule(
        tmp.allocateFrom("sys")
      );
      if (PythonTypes.isNull(sysModule)) return "python3.12";
      MemorySegment versionInfo = CPythonBinding.PyObject_GetAttrString(
        sysModule,
        tmp.allocateFrom("version_info")
      );
      PythonTypes.decref(sysModule);
      if (PythonTypes.isNull(versionInfo)) return "python3.12";
      MemorySegment major = CPythonBinding.PySequence_GetItem(versionInfo, 0);
      MemorySegment minor = CPythonBinding.PySequence_GetItem(versionInfo, 1);
      PythonTypes.decref(versionInfo);
      long maj = PythonTypes.isNull(major)
        ? 3
        : CPythonBinding.PyLong_AsLong(major);
      long min = PythonTypes.isNull(minor)
        ? 12
        : CPythonBinding.PyLong_AsLong(minor);
      PythonTypes.decref(major);
      PythonTypes.decref(minor);
      return "python" + maj + "." + min;
    }
  }

  // ── PYTHONHOME resolution ──────────────────────────────────────────────

  /**
   * Resolves the base Python prefix to use as {@code PYTHONHOME} by parsing
   * the venv's {@code pyvenv.cfg} ({@code home} points at the base
   * interpreter's {@code bin} directory).
   *
   * <p>The parent of {@code home} is not always the prefix that holds the
   * standard library. Homebrew's macOS pythons are framework builds: they
   * expose {@code <opt>/bin/python3.X} as a shim, so a venv created through it
   * records {@code home = <opt>/bin}, whose parent has no {@code lib/pythonX.Y}
   * stdlib at all — the real one lives under
   * {@code <opt>/Frameworks/Python.framework/Versions/X.Y}. Setting PYTHONHOME
   * to the shim's parent makes CPython abort during {@code Py_InitializeEx}
   * with "No module named 'encodings'", long before any of our code runs.
   *
   * <p>So the candidate is verified to actually contain the stdlib, and the
   * framework layout is tried before giving up.
   */
  static String resolvePythonHome(String venvPath) {
    java.nio.file.Path venv = java.nio.file.Path.of(venvPath).toAbsolutePath();

    java.nio.file.Path prefix = PythonLibLoader.basePrefix(venv);
    if (prefix != null) {
      java.nio.file.Path verified = withStdlib(prefix);
      if (verified != null) {
        return verified.toString();
      }
      throw new VllmException(
        "Resolved PYTHONHOME '" +
          prefix +
          "' does not contain a Python standard library (no lib/pythonX.Y/encodings). " +
          "For Homebrew framework builds the stdlib lives under " +
          "<prefix>/Frameworks/Python.framework/Versions/X.Y — recreate the venv with that " +
          "interpreter, or set PYTHONHOME explicitly before launching the JVM."
      );
    }

    throw new VllmException(
      "Cannot determine PYTHONHOME (base Python prefix) from " +
        venv.resolve("pyvenv.cfg") +
        ". Ensure the venv exists (scripts/setup-venv.sh) " +
        "or set the PYTHONHOME environment variable before launching the JVM."
    );
  }

  /**
   * Returns {@code prefix} if it holds a Python standard library, otherwise the
   * framework prefix nested under it, otherwise {@code null}.
   */
  private static java.nio.file.Path withStdlib(java.nio.file.Path prefix) {
    if (hasStdlib(prefix)) {
      return prefix;
    }
    // Homebrew framework layout: <prefix>/Frameworks/Python.framework/Versions/X.Y
    java.nio.file.Path versions = prefix.resolve(
      "Frameworks/Python.framework/Versions"
    );
    if (java.nio.file.Files.isDirectory(versions)) {
      try (var entries = java.nio.file.Files.list(versions)) {
        return entries
          .filter(java.nio.file.Files::isDirectory)
          .filter(PythonRuntime::hasStdlib)
          .findFirst()
          .orElse(null);
      } catch (java.io.IOException ignored) {}
    }
    return null;
  }

  /** True when {@code prefix/lib/pythonX.Y/encodings} exists. */
  private static boolean hasStdlib(java.nio.file.Path prefix) {
    java.nio.file.Path lib = prefix.resolve("lib");
    if (!java.nio.file.Files.isDirectory(lib)) {
      return false;
    }
    try (var entries = java.nio.file.Files.list(lib)) {
      return entries.anyMatch(
        p ->
          p.getFileName().toString().startsWith("python") &&
          java.nio.file.Files.isDirectory(p.resolve("encodings"))
      );
    } catch (java.io.IOException e) {
      return false;
    }
  }

  // ── Environment variable helper ────────────────────────────────────────

  /**
   * Wires up the build toolchain bundled in the venv so flashinfer can
   * JIT-compile CUDA kernels during engine initialization:
   *
   * <ul>
   *   <li>Prepends the venv's {@code bin} directory to {@code PATH} so
   *       {@code ninja} (bundled by vllm&gt;=0.23.0) resolves — without it
   *       engine init fails with
   *       {@code "[Errno 2] No such file or directory: 'ninja'"}.</li>
   *   <li>If the venv bundles an NVIDIA CUDA toolkit (the {@code nvidia/cuNN}
   *       tree shipped by the {@code nvidia-cuda-*} wheels on the CUDA
   *       backend), exports {@code CUDA_HOME} and prepends its {@code bin} so
   *       flashinfer compiles with that toolkit rather than a stray/older
   *       system CUDA. flashinfer resolves {@code nvcc} from {@code CUDA_HOME}
   *       (falling back to {@code which nvcc}); the bundled {@code nvcc} is not
   *       otherwise on {@code PATH}, so without this it would silently pick up
   *       the system toolkit, which is generally a different version than the
   *       torch CUDA runtime the wheels target and fails to build the
   *       kernels.</li>
   * </ul>
   *
   * <p>{@code PATH} is assembled and set in a single {@code setEnv} call:
   * {@link System#getenv} returns the JVM's startup snapshot and never
   * reflects our {@code setenv(3)} writes, so successive prepends each
   * reading {@code getenv("PATH")} would clobber one another.
   *
   * <p>The CUDA portion is a no-op on backends without a bundled toolkit
   * (metal, cpu).
   *
   * <p>Must be called before {@code Py_InitializeEx}: CPython's {@code os}
   * module snapshots the C environment into {@code os.environ} at import time,
   * and {@code subprocess} resolves bare executable names against that snapshot.
   */
  private static void configureBuildToolchain(String venvPath) {
    String sep = java.io.File.pathSeparator;
    StringBuilder pathPrefix = new StringBuilder();

    // CUDA toolkit (if bundled) takes precedence on PATH, and sets CUDA_HOME.
    java.nio.file.Path cudaHome = findBundledCudaToolkit(venvPath);
    if (cudaHome != null) {
      setEnv("CUDA_HOME", cudaHome.toString());
      pathPrefix.append(cudaHome.resolve("bin")).append(sep);
    }

    // venv bin (ninja and other bundled executables).
    pathPrefix.append(venvPath).append("/bin");

    String current = System.getenv("PATH");
    String updated = (current == null || current.isEmpty())
      ? pathPrefix.toString()
      : pathPrefix + sep + current;
    setEnv("PATH", updated);
  }

  /**
   * Locates a CUDA toolkit bundled in the venv at
   * {@code <venv>/lib/python*\/site-packages/nvidia/cu*\/bin/nvcc}, returning
   * the toolkit root (the {@code cuNN} directory) or {@code null} if none is
   * present (e.g. metal/cpu backends).
   */
  private static java.nio.file.Path findBundledCudaToolkit(String venvPath) {
    java.nio.file.Path libDir = java.nio.file.Path.of(venvPath, "lib");
    try (
      var pyDirs = java.nio.file.Files.newDirectoryStream(libDir, "python*")
    ) {
      for (java.nio.file.Path pyDir : pyDirs) {
        java.nio.file.Path nvidiaDir = pyDir.resolve("site-packages/nvidia");
        if (!java.nio.file.Files.isDirectory(nvidiaDir)) continue;
        try (
          var cuDirs = java.nio.file.Files.newDirectoryStream(nvidiaDir, "cu*")
        ) {
          for (java.nio.file.Path cuDir : cuDirs) {
            if (java.nio.file.Files.isExecutable(cuDir.resolve("bin/nvcc"))) {
              return cuDir.toAbsolutePath();
            }
          }
        }
      }
    } catch (java.io.IOException ignored) {
      // venv/lib missing or unreadable — no bundled toolkit to configure.
    }
    return null;
  }

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
    var setenvAddr = lookup
      .find("setenv")
      .orElseThrow(() -> new VllmException("Cannot find setenv() in libc"));
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
        1 // overwrite = true
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
