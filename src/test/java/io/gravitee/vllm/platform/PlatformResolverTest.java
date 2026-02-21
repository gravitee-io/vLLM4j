package io.gravitee.vllm.platform;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PlatformResolverTest {

    @Test
    void os_shouldReturnNonNull() {
        assertThat(PlatformResolver.os()).isNotNull();
    }

    @Test
    void architecture_shouldReturnNonNull() {
        assertThat(PlatformResolver.architecture()).isNotNull();
    }

    @Test
    void backend_shouldReturnNonNull() {
        assertThat(PlatformResolver.backend()).isNotNull();
    }

    @Test
    void onMacOs_shouldDetectMetal() {
        if (System.getProperty("os.name").toLowerCase().contains("mac")
                && System.getProperty("os.arch").toLowerCase().contains("aarch64")) {
            assertThat(PlatformResolver.backend()).isEqualTo(VllmBackend.METAL);
        }
    }
}
