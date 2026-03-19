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
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the lazy tokenIds/logprobs emission contract in {@link VllmIterator}.
 *
 * <p>During streaming, tokenIds and logprobs are empty/null to avoid accumulation.
 * On the final output (finished=true), the full tokenIds and logprobs are emitted.
 */
class VllmIteratorLazyEmissionTest {

  private static VllmOutput streamingOutput(
    String requestId,
    String text,
    String delta
  ) {
    return new VllmOutput(
      requestId,
      text,
      delta,
      false,
      null,
      null,
      List.of(),
      null
    );
  }

  private static VllmOutput finishedOutput(
    String requestId,
    String text,
    String delta,
    List<Integer> tokenIds,
    List<Map<Integer, LogprobEntry>> logprobs
  ) {
    return new VllmOutput(
      requestId,
      text,
      delta,
      true,
      "stop",
      null,
      tokenIds,
      logprobs
    );
  }

  // ═══════════════════════════════════════════════════════════════════
  //  tokenIds — empty during streaming, full on finish
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void tokenIds_shouldBeEmptyDuringStreaming() {
    var output = streamingOutput("req-1", "hel", "hel");

    assertThat(output.finished()).isFalse();
    assertThat(output.tokenIds()).isEmpty();
  }

  @Test
  void tokenIds_shouldBeEmittedOnFinish() {
    var output = finishedOutput(
      "req-1",
      "hello",
      "o",
      List.of(10, 20, 30),
      null
    );

    assertThat(output.finished()).isTrue();
    assertThat(output.tokenIds()).containsExactly(10, 20, 30);
  }

  @Test
  void tokenIds_onlyLastOutputHasTokens() {
    // Simulate a 3-step generation
    var step1 = streamingOutput("req-1", "hel", "hel");
    var step2 = streamingOutput("req-1", "hell", "l");
    var step3 = finishedOutput(
      "req-1",
      "hello",
      "o",
      List.of(10, 20, 30),
      null
    );

    var allOutputs = List.of(step1, step2, step3);

    // Only the finished output has token IDs
    var withTokens = allOutputs
      .stream()
      .filter(o -> !o.tokenIds().isEmpty())
      .collect(Collectors.toList());

    assertThat(withTokens).hasSize(1);
    assertThat(withTokens.getFirst().finished()).isTrue();
    assertThat(withTokens.getFirst().tokenIds()).containsExactly(10, 20, 30);
  }

  // ═══════════════════════════════════════════════════════════════════
  //  logprobs — null during streaming, emitted on finish if requested
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void logprobs_shouldBeNullDuringStreaming() {
    var output = streamingOutput("req-1", "hel", "hel");

    assertThat(output.logprobs()).isNull();
  }

  @Test
  void logprobs_shouldBeNullOnFinishWhenNotRequested() {
    // logprobs=null means SamplingParams.logprobs() was not set (n=0)
    var output = finishedOutput(
      "req-1",
      "hello",
      "o",
      List.of(10, 20, 30),
      null
    );

    assertThat(output.finished()).isTrue();
    assertThat(output.logprobs()).isNull();
  }

  @Test
  void logprobs_shouldBeEmittedOnFinishWhenRequested() {
    var entry1 = new LogprobEntry(10, -0.1, 1, "hel");
    var entry2 = new LogprobEntry(20, -0.2, 1, "l");
    var entry3 = new LogprobEntry(30, -0.3, 1, "o");
    var logprobs = List.of(
      Map.of(10, entry1),
      Map.of(20, entry2),
      Map.of(30, entry3)
    );

    var output = finishedOutput(
      "req-1",
      "hello",
      "o",
      List.of(10, 20, 30),
      logprobs
    );

    assertThat(output.finished()).isTrue();
    assertThat(output.logprobs()).hasSize(3);
    assertThat(output.logprobs().get(0).get(10).decodedToken()).isEqualTo(
      "hel"
    );
    assertThat(output.logprobs().get(1).get(20).decodedToken()).isEqualTo("l");
    assertThat(output.logprobs().get(2).get(30).decodedToken()).isEqualTo("o");
  }

  @Test
  void logprobs_onlyLastOutputHasLogprobs() {
    var entry = new LogprobEntry(30, -0.3, 1, "o");
    var logprobs = List.of(Map.of(30, entry));

    var step1 = streamingOutput("req-1", "hel", "hel");
    var step2 = streamingOutput("req-1", "hell", "l");
    var step3 = finishedOutput(
      "req-1",
      "hello",
      "o",
      List.of(10, 20, 30),
      logprobs
    );

    var allOutputs = List.of(step1, step2, step3);

    var withLogprobs = allOutputs
      .stream()
      .filter(o -> o.logprobs() != null)
      .collect(Collectors.toList());

    assertThat(withLogprobs).hasSize(1);
    assertThat(withLogprobs.getFirst().finished()).isTrue();
  }

  // ═══════════════════════════════════════════════════════════════════
  //  tokenIds count matches logprobs positions when both present
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void tokenIds_countShouldMatchLogprobsPositions() {
    var entry1 = new LogprobEntry(10, -0.1, 1, "hel");
    var entry2 = new LogprobEntry(20, -0.2, 1, "lo");
    var logprobs = List.of(Map.of(10, entry1), Map.of(20, entry2));

    var output = finishedOutput(
      "req-1",
      "hello",
      "lo",
      List.of(10, 20),
      logprobs
    );

    assertThat(output.tokenIds()).hasSameSizeAs(output.logprobs());
  }
}
