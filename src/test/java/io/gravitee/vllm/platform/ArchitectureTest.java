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
