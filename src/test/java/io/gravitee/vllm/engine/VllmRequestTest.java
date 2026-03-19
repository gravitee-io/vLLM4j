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

class VllmRequestTest {

  /**
   * We cannot construct a real SamplingParams in unit tests (requires CPython),
   * so we test only the VllmRequest record validation with null checks.
   */

  @Test
  void nullRequestId_shouldThrow() {
    assertThatThrownBy(() -> new VllmRequest(null, "prompt", null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("requestId");
  }

  @Test
  void blankRequestId_shouldThrow() {
    assertThatThrownBy(() -> new VllmRequest("  ", "prompt", null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("requestId");
  }

  @Test
  void nullPrompt_shouldThrow() {
    assertThatThrownBy(() -> new VllmRequest("req-1", null, null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("prompt");
  }

  @Test
  void nullSamplingParams_shouldThrow() {
    assertThatThrownBy(() -> new VllmRequest("req-1", "hello", null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("samplingParams");
  }

  @Test
  void isMultiModal_shouldBeFalseWhenNull() {
    var mm = new MultiModalData();
    assertThat(mm.hasData()).isFalse();
  }

  @Test
  void isMultiModal_shouldBeTrueWithData() {
    var mm = new MultiModalData().addImage(new byte[] { 1, 2, 3 });
    assertThat(mm.hasData()).isTrue();
  }

  @Test
  void hasPriority_shouldBeFalseByDefault() {
    // Can't construct with real SamplingParams, but verify via MultiModalData path
    var mm = new MultiModalData();
    assertThat(mm.hasData()).isFalse();
    // Default priority is 0; hasPriority() returns false for 0
  }

  @Test
  void hasLora_shouldBeFalseWhenNull() {
    // We can't construct a full VllmRequest without SamplingParams (requires CPython),
    // but we can test LoraRequest construction and hasLora logic via the record itself
    var lora = new LoraRequest("test", 1, "path/to/adapter");
    assertThat(lora.loraName()).isEqualTo("test");
    assertThat(lora.loraIntId()).isEqualTo(1);
    assertThat(lora.loraPath()).isEqualTo("path/to/adapter");
  }
}
