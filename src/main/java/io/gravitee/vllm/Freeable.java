package io.gravitee.vllm;

/**
 * Contract for objects that hold native resources requiring explicit release.
 *
 * <p>Mirrors the pattern from llamaj.cpp. Implementors should call
 * {@link #free()} when the resource is no longer needed, and {@link #isFree()}
 * should return {@code true} after that point.
 */
public interface Freeable {

    /** Releases the underlying native resource. Idempotent. */
    void free();

    /** Returns {@code true} if {@link #free()} has been called. */
    default boolean isFree() {
        return false;
    }
}
