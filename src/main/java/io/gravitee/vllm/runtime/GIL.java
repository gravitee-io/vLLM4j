package io.gravitee.vllm.runtime;

import org.vllm.python.CPython;

/**
 * RAII wrapper for CPython's Global Interpreter Lock (GIL).
 *
 * <p>Ensures that any Java thread can safely call CPython C API functions
 * by acquiring the GIL before the call and releasing it after. This is
 * critical because CPython's internal data structures (object allocator,
 * reference counts, etc.) are not thread-safe — concurrent access without
 * the GIL causes segfaults ({@code SIGSEGV in _PyObject_Malloc}).
 *
 * <h2>Threading model</h2>
 * <p>After {@link PythonRuntime} initializes CPython, it <em>releases</em>
 * the GIL via {@code PyEval_SaveThread()}, making it available for any
 * Java thread to acquire. Each thread that needs to call CPython must:
 * <ol>
 *   <li>Call {@link #acquire()} to get a {@code GIL} instance (acquires the GIL)</li>
 *   <li>Perform CPython operations</li>
 *   <li>Call {@link #close()} (or use try-with-resources) to release the GIL</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (var gil = GIL.acquire()) {
 *     MemorySegment result = CPython.PyObject_GetAttrString(obj, name);
 *     // ... more CPython calls ...
 * }
 * // GIL is released here — other threads can now acquire it
 * }</pre>
 *
 * <h2>Re-entrancy</h2>
 * <p>{@code PyGILState_Ensure()} is re-entrant: if the current thread
 * already holds the GIL, it increments an internal counter and
 * {@code PyGILState_Release()} decrements it. Nesting is safe.
 *
 * @see <a href="https://docs.python.org/3/c-api/init.html#c.PyGILState_Ensure">
 *     PyGILState_Ensure</a>
 */
public final class GIL implements AutoCloseable {

    private final int gilState;
    private boolean released = false;

    private GIL(int gilState) {
        this.gilState = gilState;
    }

    /**
     * Acquires the GIL for the calling thread.
     *
     * <p>If the calling thread already holds the GIL, this is a no-op
     * (the internal reference count is incremented). Otherwise, the
     * thread blocks until the GIL becomes available.
     *
     * @return a {@code GIL} instance that must be {@link #close() closed}
     *         to release the GIL
     */
    public static GIL acquire() {
        int state = CPython.PyGILState_Ensure();
        return new GIL(state);
    }

    /**
     * Releases the GIL. Must be called exactly once per {@link #acquire()}.
     *
     * <p>After this call, the current thread no longer holds the GIL and
     * must not call CPython API functions until it re-acquires it.
     */
    @Override
    public void close() {
        if (!released) {
            released = true;
            CPython.PyGILState_Release(gilState);
        }
    }
}
