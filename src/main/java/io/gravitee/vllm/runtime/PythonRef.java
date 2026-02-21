package io.gravitee.vllm.runtime;

import io.gravitee.vllm.Freeable;
import io.gravitee.vllm.binding.PythonTypes;

import java.lang.foreign.MemorySegment;

/**
 * A safe, reference-counted wrapper around a Python {@code PyObject*}.
 *
 * <p>Implements {@link AutoCloseable} and {@link Freeable} so it can be used
 * in try-with-resources blocks. On {@link #close()}, the Python reference count
 * is decremented via {@code Py_DecRef}.
 *
 * <h2>Ownership semantics</h2>
 * <ul>
 *   <li>{@link #steal(MemorySegment)} — Takes ownership of a <em>new</em> reference
 *       (the caller's reference is "stolen"). No incref.</li>
 *   <li>{@link #of(MemorySegment)} — Creates a new owning reference from a
 *       <em>borrowed</em> reference. Calls {@code Py_IncRef}.</li>
 * </ul>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * try (PythonRef result = PythonRef.steal(CPython.PyImport_ImportModule(...))) {
 *     // use result.get() ...
 * } // Py_DecRef called automatically
 * }</pre>
 */
public final class PythonRef implements AutoCloseable, Freeable {

    private final MemorySegment ptr;
    private volatile boolean closed = false;

    private PythonRef(MemorySegment ptr) {
        this.ptr = ptr;
    }

    /**
     * Wraps a <em>new</em> reference (steals ownership). The caller must not
     * decref the pointer after calling this method.
     *
     * @param newRef a new Python reference (from a C API call that returns a new ref)
     * @return a new {@code PythonRef} owning the reference
     */
    public static PythonRef steal(MemorySegment newRef) {
        return new PythonRef(newRef);
    }

    /**
     * Wraps a <em>borrowed</em> reference by incrementing its reference count.
     * The original borrowed reference remains valid independently.
     *
     * @param borrowed a borrowed Python reference
     * @return a new {@code PythonRef} owning a new reference to the same object
     */
    public static PythonRef of(MemorySegment borrowed) {
        PythonTypes.incref(borrowed);
        return new PythonRef(borrowed);
    }

    /**
     * Returns the raw {@code PyObject*} pointer.
     *
     * @throws IllegalStateException if this ref has been closed
     */
    public MemorySegment get() {
        if (closed) throw new IllegalStateException("PythonRef has been closed");
        return ptr;
    }

    /** Returns {@code true} if the underlying pointer is null/NULL. */
    public boolean isNull() {
        return PythonTypes.isNull(ptr);
    }

    @Override
    public void close() {
        free();
    }

    @Override
    public void free() {
        if (!closed) {
            closed = true;
            PythonTypes.decref(ptr);
        }
    }

    @Override
    public boolean isFree() {
        return closed;
    }
}
