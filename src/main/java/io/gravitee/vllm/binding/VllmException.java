package io.gravitee.vllm.binding;

/**
 * Thrown when a CPython exception propagates across the FFM boundary.
 *
 * <p>The message is the string representation of the Python exception that was
 * active at the time of the call ({@code PyErr_Fetch} + {@code PyObject_Str}),
 * so it matches what you would see in a Python traceback.
 */
public class VllmException extends RuntimeException {

    public VllmException(String message) {
        super(message);
    }

    public VllmException(String message, Throwable cause) {
        super(message, cause);
    }
}
