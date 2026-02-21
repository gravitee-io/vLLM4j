package io.gravitee.vllm.platform;

/**
 * Singleton that resolves the current platform (OS + architecture) and the
 * vLLM compute backend at class-load time.
 *
 * <p>The resolved values are cached and immutable for the lifetime of the JVM.
 */
public final class PlatformResolver {

    private static final OperatingSystem OS = OperatingSystem.fromSystem();
    private static final Architecture ARCH = Architecture.fromSystem();
    private static final VllmBackend BACKEND = VllmBackend.detect();

    private PlatformResolver() {}

    /** Returns the detected operating system. */
    public static OperatingSystem os() {
        return OS;
    }

    /** Returns the detected CPU architecture. */
    public static Architecture architecture() {
        return ARCH;
    }

    /** Returns the detected (or explicitly configured) vLLM backend. */
    public static VllmBackend backend() {
        return BACKEND;
    }
}
