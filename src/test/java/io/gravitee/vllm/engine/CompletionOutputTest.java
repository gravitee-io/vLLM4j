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

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CompletionOutputTest {

  @Test
  void record_shouldHaveExpectedFields() {
    var output = new CompletionOutput(
      0,
      "Paris",
      List.of(42, 99),
      FinishReason.STOP
    );

    assertThat(output.index()).isEqualTo(0);
    assertThat(output.text()).isEqualTo("Paris");
    assertThat(output.tokenIds()).containsExactly(42, 99);
    assertThat(output.finishReason()).isEqualTo(FinishReason.STOP);
    assertThat(output.finished()).isTrue();
    assertThat(output.numGeneratedTokens()).isEqualTo(2);
  }

  @Test
  void unfinished_shouldHaveNullFinishReason() {
    var output = new CompletionOutput(0, "partial", List.of(1, 2, 3), null);

    assertThat(output.finished()).isFalse();
    assertThat(output.finishReason()).isNull();
    assertThat(output.numGeneratedTokens()).isEqualTo(3);
  }

  @Test
  void emptyTokenIds_shouldReturnZeroCount() {
    var output = new CompletionOutput(0, "", List.of(), null);

    assertThat(output.numGeneratedTokens()).isEqualTo(0);
  }

  @Test
  void nullTokenIds_shouldReturnZeroCount() {
    var output = new CompletionOutput(0, "", null, null);

    assertThat(output.numGeneratedTokens()).isEqualTo(0);
  }

  @Test
  void equality_shouldWork() {
    var a = new CompletionOutput(1, "text", List.of(10), FinishReason.LENGTH);
    var b = new CompletionOutput(1, "text", List.of(10), FinishReason.LENGTH);
    assertThat(a).isEqualTo(b);
  }

  @Test
  void convenienceConstructor_shouldDefaultLogprobsToNull() {
    var output = new CompletionOutput(0, "text", List.of(1), FinishReason.STOP);
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void fullConstructor_shouldPopulateLogprobs() {
    var entry = new LogprobEntry(42, -1.5, 1, "hello");
    var logprobs = List.of(Map.of(42, entry));
    var output = new CompletionOutput(
      0,
      "hello",
      List.of(42),
      FinishReason.STOP,
      logprobs
    );

    assertThat(output.logprobs()).hasSize(1);
    assertThat(output.logprobs().getFirst()).containsKey(42);
    assertThat(output.logprobs().getFirst().get(42).logprob()).isEqualTo(-1.5);
  }

  @Test
  void logprobs_multiplePositions() {
    var entry1 = new LogprobEntry(10, -0.5, 1, "A");
    var entry2 = new LogprobEntry(20, -1.0, 1, "B");
    var altEntry2 = new LogprobEntry(21, -2.0, 2, "C");
    var logprobs = List.of(
      Map.of(10, entry1),
      Map.of(20, entry2, 21, altEntry2)
    );
    var output = new CompletionOutput(
      0,
      "AB",
      List.of(10, 20),
      FinishReason.STOP,
      logprobs
    );

    assertThat(output.logprobs()).hasSize(2);
    assertThat(output.logprobs().get(0)).hasSize(1);
    assertThat(output.logprobs().get(1)).hasSize(2);
    assertThat(output.logprobs().get(1).get(21).rank()).isEqualTo(2);
  }
}
