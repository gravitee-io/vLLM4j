package io.gravitee.vllm.platform;

/**
 * Supported CPU architectures for vLLM4j.
 *
 * <p>Auto-detected from the {@code os.arch} system property.
 */
public enum Architecture {
    X86_64("x86_64"),
    AARCH64("aarch64");

    private final String arch;

    Architecture(String arch) {
        this.arch = arch;
    }

    /**
     * Detects the current architecture from {@code os.arch}.
     *
     * @throws IllegalArgumentException if the architecture is not supported
     */
    public static Architecture fromSystem() {
        String osArch = System.getProperty("os.arch").toLowerCase();
        if (osArch.contains("x86_64") || osArch.contains("amd64")) {
            return X86_64;
        }
        if (osArch.contains("aarch64") || osArch.contains("arm64")) {
            return AARCH64;
        }
        throw new IllegalArgumentException("Unsupported architecture: " + osArch);
    }

    public String getArch() {
        return arch;
    }
}
