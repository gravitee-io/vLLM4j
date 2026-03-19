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

import org.junit.jupiter.api.Test;

class EngineStatsTest {

  @Test
  void record_shouldHaveExpectedFields() {
    var stats = new EngineStats(5, "Qwen/Qwen3-0.6B", "auto", 40960);

    assertThat(stats.numUnfinishedRequests()).isEqualTo(5);
    assertThat(stats.model()).isEqualTo("Qwen/Qwen3-0.6B");
    assertThat(stats.dtype()).isEqualTo("auto");
    assertThat(stats.maxModelLen()).isEqualTo(40960);
  }

  @Test
  void equality_shouldWork() {
    var a = new EngineStats(0, "model", "float16", 2048);
    var b = new EngineStats(0, "model", "float16", 2048);
    assertThat(a).isEqualTo(b);
    assertThat(a.hashCode()).isEqualTo(b.hashCode());
  }

  @Test
  void differentValues_shouldNotBeEqual() {
    var a = new EngineStats(0, "model-a", "float16", 2048);
    var b = new EngineStats(0, "model-b", "float16", 2048);
    assertThat(a).isNotEqualTo(b);
  }

  @Test
  void toString_shouldContainFields() {
    var stats = new EngineStats(2, "test-model", "auto", 4096);
    String str = stats.toString();
    assertThat(str).contains("test-model", "auto", "4096");
  }
}
