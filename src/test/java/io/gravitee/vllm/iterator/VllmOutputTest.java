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
package io.gravitee.vllm.iterator;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.vllm.engine.LogprobEntry;
import io.gravitee.vllm.state.GenerationState;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class VllmOutputTest {

  @Test
  void record_shouldHaveExpectedFields() {
    var output = new VllmOutput(
      "req-1",
      "hello world",
      "world",
      false,
      null,
      null,
      List.of(1, 2)
    );

    assertThat(output.requestId()).isEqualTo("req-1");
    assertThat(output.text()).isEqualTo("hello world");
    assertThat(output.delta()).isEqualTo("world");
    assertThat(output.finished()).isFalse();
    assertThat(output.finishReason()).isNull();
    assertThat(output.state()).isNull();
    assertThat(output.tokenIds()).containsExactly(1, 2);
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void convenienceConstructor_shouldDefaultDeltaAndState() {
    var output = new VllmOutput("req-1", "hello", false, null);

    assertThat(output.delta()).isEmpty();
    assertThat(output.state()).isNull();
    assertThat(output.tokenIds()).isEmpty();
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void sixArgConstructor_shouldDefaultTokenIds() {
    var output = new VllmOutput(
      "req-1",
      "hello",
      "hello",
      true,
      "stop",
      GenerationState.ANSWER
    );

    assertThat(output.tokenIds()).isEmpty();
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void finishedOutput_shouldHaveReason() {
    var output = new VllmOutput(
      "req-2",
      "done",
      "done",
      true,
      "stop",
      GenerationState.ANSWER,
      List.of(10)
    );

    assertThat(output.finished()).isTrue();
    assertThat(output.finishReason()).isEqualTo("stop");
    assertThat(output.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(output.tokenIds()).containsExactly(10);
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void logprobs_shouldBeNullWhenNotProvided() {
    var output = new VllmOutput(
      "req-1",
      "hello",
      "hello",
      true,
      "stop",
      GenerationState.ANSWER,
      List.of(1),
      null
    );

    assertThat(output.logprobs()).isNull();
  }

  @Test
  void logprobs_shouldBeAccessibleWhenProvided() {
    var entry = new LogprobEntry(42, -0.5, 1, "hello");
    var logprobs = List.of(Map.of(42, entry));

    var output = new VllmOutput(
      "req-1",
      "hello",
      "hello",
      true,
      "stop",
      null,
      List.of(42),
      logprobs
    );

    assertThat(output.logprobs()).hasSize(1);
    assertThat(output.logprobs().getFirst()).containsKey(42);
    assertThat(output.logprobs().getFirst().get(42).decodedToken()).isEqualTo(
      "hello"
    );
    assertThat(output.logprobs().getFirst().get(42).logprob()).isEqualTo(-0.5);
    assertThat(output.logprobs().getFirst().get(42).rank()).isEqualTo(1);
  }

  @Test
  void equality_shouldWork() {
    var a = new VllmOutput(
      "req-1",
      "text",
      "text",
      true,
      "length",
      GenerationState.REASONING,
      List.of(5)
    );
    var b = new VllmOutput(
      "req-1",
      "text",
      "text",
      true,
      "length",
      GenerationState.REASONING,
      List.of(5)
    );
    assertThat(a).isEqualTo(b);
  }

  @Test
  void differentRequests_shouldNotBeEqual() {
    var a = new VllmOutput("req-1", "text", "text", false, null, null);
    var b = new VllmOutput("req-2", "text", "text", false, null, null);
    assertThat(a).isNotEqualTo(b);
  }
}
