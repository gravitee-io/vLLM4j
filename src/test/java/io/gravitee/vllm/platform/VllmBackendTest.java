package io.gravitee.vllm.platform;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class VllmBackendTest {

    @Test
    void detect_shouldReturnNonNull() {
        VllmBackend backend = VllmBackend.detect();
        assertThat(backend).isNotNull();
    }

    @Test
    void metal_shouldHaveExpectedEnvVars() {
        assertThat(VllmBackend.METAL.envVars())
                .containsEntry("VLLM_METAL_USE_MLX", "1")
                .containsEntry("VLLM_MLX_DEVICE", "gpu")
                .containsEntry("GLOO_SOCKET_IFNAME", "lo0")
                .containsEntry("VLLM_ENABLE_V1_MULTIPROCESSING", "0");
    }

    @Test
    void cuda_shouldHaveMultiprocessingDisabled() {
        assertThat(VllmBackend.CUDA.envVars())
                .containsEntry("VLLM_ENABLE_V1_MULTIPROCESSING", "0");
    }

    @Test
    void cpu_shouldHaveExpectedEnvVars() {
        assertThat(VllmBackend.CPU.envVars())
                .containsEntry("VLLM_TARGET_DEVICE", "cpu")
                .containsEntry("VLLM_ENABLE_V1_MULTIPROCESSING", "0");
    }

    @Test
    void allBackends_shouldHaveMultiprocessingDisabled() {
        for (VllmBackend backend : VllmBackend.values()) {
            assertThat(backend.envVars())
                    .as("Backend %s should disable V1 multiprocessing", backend)
                    .containsEntry("VLLM_ENABLE_V1_MULTIPROCESSING", "0");
        }
    }
}
