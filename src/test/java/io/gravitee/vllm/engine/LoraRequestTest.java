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
package io.gravitee.vllm.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class LoraRequestTest {

  @Test
  void valid_construction() {
    var lora = new LoraRequest("sql-lora", 1, "org/adapter-repo");
    assertThat(lora.loraName()).isEqualTo("sql-lora");
    assertThat(lora.loraIntId()).isEqualTo(1);
    assertThat(lora.loraPath()).isEqualTo("org/adapter-repo");
  }

  @Test
  void nullLoraName_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest(null, 1, "path"))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraName");
  }

  @Test
  void blankLoraName_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest("  ", 1, "path"))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraName");
  }

  @Test
  void loraIntIdZero_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest("name", 0, "path"))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraIntId");
  }

  @Test
  void loraIntIdNegative_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest("name", -1, "path"))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraIntId");
  }

  @Test
  void nullLoraPath_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest("name", 1, null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraPath");
  }

  @Test
  void blankLoraPath_shouldThrow() {
    assertThatThrownBy(() -> new LoraRequest("name", 1, ""))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("loraPath");
  }

  @Test
  void loraIntId_one_is_valid() {
    var lora = new LoraRequest("adapter", 1, "/local/path");
    assertThat(lora.loraIntId()).isEqualTo(1);
  }

  @Test
  void loraIntId_large_is_valid() {
    var lora = new LoraRequest("adapter", 999, "some/path");
    assertThat(lora.loraIntId()).isEqualTo(999);
  }
}
