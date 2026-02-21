package io.gravitee.vllm.platform;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class OperatingSystemTest {

    @Test
    void fromSystem_shouldReturnNonNull() {
        OperatingSystem os = OperatingSystem.fromSystem();
        assertThat(os).isNotNull();
    }

    @Test
    void osName_shouldBeNonEmpty() {
        OperatingSystem os = OperatingSystem.fromSystem();
        assertThat(os.getOsName()).isNotBlank();
    }

    @Test
    void macOsX_shouldHaveCorrectName() {
        assertThat(OperatingSystem.MAC_OS_X.getOsName()).isEqualTo("macosx");
    }

    @Test
    void linux_shouldHaveCorrectName() {
        assertThat(OperatingSystem.LINUX.getOsName()).isEqualTo("linux");
    }
}
