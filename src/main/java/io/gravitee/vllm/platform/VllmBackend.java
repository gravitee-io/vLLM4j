package io.gravitee.vllm.platform;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * vLLM compute backend — determines which Python packages are installed and
 * which environment variables are set before CPython initialization.
 *
 * <p>The backend can be:
 * <ul>
 *   <li>{@link #METAL} — Apple Silicon GPU via MLX (macOS only)</li>
 *   <li>{@link #CUDA} — NVIDIA GPU via CUDA (Linux)</li>
 *   <li>{@link #CPU} — CPU-only inference</li>
 * </ul>
 *
 * <p>Auto-detection:
 * <ol>
 *   <li>{@code VLLM4J_BACKEND} environment variable</li>
 *   <li>{@code vllm4j.backend} system property</li>
 *   <li>macOS + aarch64 → {@code METAL}</li>
 *   <li>Linux + NVIDIA GPU detected → {@code CUDA}</li>
 *   <li>Else → {@code CPU}</li>
 * </ol>
 */
public enum VllmBackend {

    METAL(Map.of(
            "VLLM_ENABLE_V1_MULTIPROCESSING", "0",
            "VLLM_METAL_USE_MLX", "1",
            "VLLM_MLX_DEVICE", "gpu",
            "GLOO_SOCKET_IFNAME", "lo0"
    )),

    CUDA(Map.of(
            "VLLM_ENABLE_V1_MULTIPROCESSING", "0"
    )),

    CPU(Map.of(
            "VLLM_ENABLE_V1_MULTIPROCESSING", "0",
            "VLLM_TARGET_DEVICE", "cpu"
    ));

    private final Map<String, String> envVars;

    VllmBackend(Map<String, String> envVars) {
        this.envVars = envVars;
    }

    /**
     * Returns the environment variables that must be set before
     * {@code Py_InitializeEx} for this backend.
     */
    public Map<String, String> envVars() {
        return envVars;
    }

    /**
     * Detects the appropriate backend for the current platform.
     *
     * @see VllmBackend class-level javadoc for detection order
     */
    public static VllmBackend detect() {
        // 1. Explicit env var
        String envBackend = System.getenv("VLLM4J_BACKEND");
        if (envBackend != null && !envBackend.isBlank()) {
            return fromString(envBackend);
        }

        // 2. System property
        String propBackend = System.getProperty("vllm4j.backend");
        if (propBackend != null && !propBackend.isBlank()) {
            return fromString(propBackend);
        }

        // 3. Auto-detect from platform
        OperatingSystem os = OperatingSystem.fromSystem();
        Architecture arch = Architecture.fromSystem();

        if (os == OperatingSystem.MAC_OS_X && arch == Architecture.AARCH64) {
            return METAL;
        }

        if (os == OperatingSystem.LINUX && hasNvidiaGpu()) {
            return CUDA;
        }

        return CPU;
    }

    private static VllmBackend fromString(String value) {
        return switch (value.strip().toLowerCase()) {
            case "metal" -> METAL;
            case "cuda" -> CUDA;
            case "cpu" -> CPU;
            default -> throw new IllegalArgumentException(
                    "Unknown vLLM backend: '" + value + "'. Supported: metal, cuda, cpu");
        };
    }

    /**
     * Probes for an NVIDIA GPU by checking {@code /dev/nvidia0} or running
     * {@code nvidia-smi}. This is a best-effort check — CUDA availability
     * ultimately depends on the installed Python packages.
     */
    private static boolean hasNvidiaGpu() {
        // Fast path: check device node
        if (Files.exists(Path.of("/dev/nvidia0"))) {
            return true;
        }

        // Fallback: try nvidia-smi
        try {
            Process p = new ProcessBuilder("nvidia-smi")
                    .redirectErrorStream(true)
                    .start();
            int exitCode = p.waitFor();
            return exitCode == 0;
        } catch (IOException | InterruptedException _) {
            return false;
        }
    }
}
