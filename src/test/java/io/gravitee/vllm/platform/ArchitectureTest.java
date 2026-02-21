package io.gravitee.vllm.platform;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ArchitectureTest {

    @Test
    void fromSystem_shouldReturnNonNull() {
        Architecture arch = Architecture.fromSystem();
        assertThat(arch).isNotNull();
    }

    @Test
    void archName_shouldBeNonEmpty() {
        Architecture arch = Architecture.fromSystem();
        assertThat(arch.getArch()).isNotBlank();
    }

    @Test
    void x86_64_shouldHaveCorrectName() {
        assertThat(Architecture.X86_64.getArch()).isEqualTo("x86_64");
    }

    @Test
    void aarch64_shouldHaveCorrectName() {
        assertThat(Architecture.AARCH64.getArch()).isEqualTo("aarch64");
    }
}
