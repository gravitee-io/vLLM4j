/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.vllm.platform;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

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
