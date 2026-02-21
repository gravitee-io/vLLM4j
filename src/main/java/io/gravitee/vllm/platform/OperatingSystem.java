package io.gravitee.vllm.platform;

/**
 * Supported operating systems for vLLM4j.
 *
 * <p>Auto-detected from the {@code os.name} system property.
 */
public enum OperatingSystem {
    MAC_OS_X("macosx"),
    LINUX("linux");

    private final String osName;

    OperatingSystem(String osName) {
        this.osName = osName;
    }

    /**
     * Detects the current operating system from {@code os.name}.
     *
     * @throws IllegalArgumentException if the OS is not supported
     */
    public static OperatingSystem fromSystem() {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("mac")) {
            return MAC_OS_X;
        }
        if (osName.contains("linux")) {
            return LINUX;
        }
        throw new IllegalArgumentException("Unsupported operating system: " + osName);
    }

    public String getOsName() {
        return osName;
    }
}
