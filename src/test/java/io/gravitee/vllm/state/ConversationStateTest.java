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
package io.gravitee.vllm.state;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.vllm.engine.FinishReason;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ConversationStateTest {

  private ConversationState state;

  @BeforeEach
  void setUp() {
    state = new ConversationState()
      .reasoning("<think>", "</think>")
      .toolCall("<tool_call>", "</tool_call>");
  }

  @Test
  void initialize_shouldResetState() {
    state.initialize(50);

    assertThat(state.currentState()).isEqualTo(GenerationState.ANSWER);
    assertThat(state.finishReason()).isNull();
    assertThat(state.inputTokens()).isEqualTo(50);
    assertThat(state.answerTokens()).isEqualTo(0);
    assertThat(state.reasoningTokens()).isEqualTo(0);
    assertThat(state.toolsTokens()).isEqualTo(0);
  }

  @Test
  void isClassificationEnabled_shouldBeTrueWhenTagsConfigured() {
    state.initialize(10);
    assertThat(state.isClassificationEnabled()).isTrue();
  }

  @Test
  void isClassificationEnabled_shouldBeFalseWithNoTags() {
    var empty = new ConversationState();
    empty.initialize(10);
    assertThat(empty.isClassificationEnabled()).isFalse();
  }

  // ── End-to-end reasoning ────────────────────────────────────────────

  @Test
  void shouldClassifyReasoningTokens() {
    state.initialize(10);

    // Reasoning phase
    var s1 = state.evaluate("<think>", 1);
    assertThat(s1).isEqualTo(GenerationState.REASONING);

    var s2 = state.evaluate("I need to think about this...", 8);
    assertThat(s2).isEqualTo(GenerationState.REASONING);

    var s3 = state.evaluate("</think>", 1);
    assertThat(s3).isEqualTo(GenerationState.ANSWER);

    // Answer phase
    var s4 = state.evaluate("The answer is 42.", 5);
    assertThat(s4).isEqualTo(GenerationState.ANSWER);

    // Verify counters
    assertThat(state.inputTokens()).isEqualTo(10);
    assertThat(state.reasoningTokens()).isEqualTo(9); // <think>(1) + thinking(8)
    assertThat(state.answerTokens()).isEqualTo(6); // </think>(1, transitions to ANSWER) + answer(5)
    assertThat(state.toolsTokens()).isEqualTo(0);
    assertThat(state.totalOutputTokens()).isEqualTo(15);
    assertThat(state.totalTokens()).isEqualTo(25);
  }

  // ── End-to-end tool call ────────────────────────────────────────────

  @Test
  void shouldClassifyToolCallTokens() {
    state.initialize(5);

    state.evaluate("<tool_call>", 1);
    state.evaluate("{\"name\":\"get_weather\"}", 5);
    state.evaluate("</tool_call>", 1);

    assertThat(state.toolsTokens()).isEqualTo(6); // <tool_call>(1) + json(5)
    assertThat(state.answerTokens()).isEqualTo(1); // </tool_call>(1, transitions back to ANSWER)
  }

  @Test
  void toolCallTransition_shouldSetToolCallFinishReason() {
    state.initialize(5);

    state.evaluate("<tool_call>", 1);
    state.evaluate("{}", 1);
    state.evaluate("</tool_call>", 1);

    assertThat(state.finishReason()).isEqualTo(FinishReason.TOOL_CALL);
  }

  // ── Finish reason priority ──────────────────────────────────────────

  @Test
  void finishReason_toolCallShouldNotBeOverwrittenByStop() {
    state.initialize(5);

    state.evaluate("<tool_call>", 1);
    state.evaluate("</tool_call>", 1);

    // TOOL_CALL set from transition
    assertThat(state.finishReason()).isEqualTo(FinishReason.TOOL_CALL);

    // STOP should NOT overwrite TOOL_CALL
    state.setFinishReason(FinishReason.STOP);
    assertThat(state.finishReason()).isEqualTo(FinishReason.TOOL_CALL);
  }

  @Test
  void finishReason_lengthShouldAlwaysWin() {
    state.initialize(5);

    state.evaluate("<tool_call>", 1);
    state.evaluate("</tool_call>", 1);

    // LENGTH always wins over TOOL_CALL
    state.setFinishReason(FinishReason.LENGTH);
    assertThat(state.finishReason()).isEqualTo(FinishReason.LENGTH);
  }

  @Test
  void finishReason_nullShouldBeIgnored() {
    state.initialize(5);
    state.setFinishReason(FinishReason.STOP);
    state.setFinishReason(null);
    assertThat(state.finishReason()).isEqualTo(FinishReason.STOP);
  }

  // ── No classification ───────────────────────────────────────────────

  @Test
  void noTags_shouldCountAllAsAnswer() {
    var noTags = new ConversationState();
    noTags.initialize(10);

    noTags.evaluate("hello", 1);
    noTags.evaluate(" world", 1);

    assertThat(noTags.answerTokens()).isEqualTo(2);
    assertThat(noTags.reasoningTokens()).isEqualTo(0);
    assertThat(noTags.currentState()).isEqualTo(GenerationState.ANSWER);
  }

  // ── Token tracking accessor ─────────────────────────────────────────

  @Test
  void tokenTracking_shouldExposeUnderlyingTracker() {
    state.initialize(10);
    state.evaluate("hello", 5);

    assertThat(state.tokenTracking()).isNotNull();
    assertThat(state.tokenTracking().inputTokens()).isEqualTo(10);
    assertThat(state.tokenTracking().totalOutputTokens()).isEqualTo(5);
  }
}
