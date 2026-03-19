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

import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class StateEvaluationTest {

  private StateEvaluation fsm;

  @BeforeEach
  void setUp() {
    fsm = new StateEvaluation();
  }

  @Test
  void uninitializedFsm_shouldReturnCurrentState() {
    assertThat(fsm.isInitialized()).isFalse();
    assertThat(fsm.evaluate(GenerationState.ANSWER, "hello")).isEqualTo(
      GenerationState.ANSWER
    );
  }

  @Test
  void nullDelta_shouldReturnCurrentState() {
    initWithReasoningTags();
    assertThat(fsm.evaluate(GenerationState.ANSWER, null)).isEqualTo(
      GenerationState.ANSWER
    );
  }

  @Test
  void emptyDelta_shouldReturnCurrentState() {
    initWithReasoningTags();
    assertThat(fsm.evaluate(GenerationState.ANSWER, "")).isEqualTo(
      GenerationState.ANSWER
    );
  }

  // ── Reasoning tag transitions ──────────────────────────────────────

  @Test
  void shouldTransitionToReasoningOnOpenTag() {
    initWithReasoningTags();

    var state = fsm.evaluate(GenerationState.ANSWER, "<think>");
    assertThat(state).isEqualTo(GenerationState.REASONING);
  }

  @Test
  void shouldTransitionBackToAnswerOnCloseTag() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>");
    var state = fsm.evaluate(GenerationState.REASONING, "</think>");
    assertThat(state).isEqualTo(GenerationState.ANSWER);
  }

  @Test
  void shouldDetectTagsSplitAcrossDeltas() {
    initWithReasoningTags();

    // Tag arrives across two deltas: "<thi" + "nk>"
    var state1 = fsm.evaluate(GenerationState.ANSWER, "Hello <thi");
    assertThat(state1).isEqualTo(GenerationState.ANSWER); // not yet

    var state2 = fsm.evaluate(GenerationState.ANSWER, "nk>");
    assertThat(state2).isEqualTo(GenerationState.REASONING); // buffer accumulated
  }

  @Test
  void reasoningShouldNotReenter() {
    initWithReasoningTags();

    // First reasoning block
    fsm.evaluate(GenerationState.ANSWER, "<think>");
    fsm.evaluate(GenerationState.REASONING, "thinking...");
    fsm.evaluate(GenerationState.REASONING, "</think>");

    // Second <think> should NOT re-enter reasoning
    var state = fsm.evaluate(GenerationState.ANSWER, "<think>");
    assertThat(state).isEqualTo(GenerationState.ANSWER);
  }

  // ── Tools tag transitions ──────────────────────────────────────────

  @Test
  void shouldTransitionToToolsOnOpenTag() {
    initWithToolTags();

    var state = fsm.evaluate(GenerationState.ANSWER, "<tool_call>");
    assertThat(state).isEqualTo(GenerationState.TOOLS);
  }

  @Test
  void toolsShouldAllowReentry() {
    initWithToolTags();

    // First tool call
    fsm.evaluate(GenerationState.ANSWER, "<tool_call>");
    fsm.evaluate(GenerationState.TOOLS, "{\"name\":\"foo\"}");
    fsm.evaluate(GenerationState.TOOLS, "</tool_call>");

    // Second tool call should work
    var state = fsm.evaluate(GenerationState.ANSWER, "<tool_call>");
    assertThat(state).isEqualTo(GenerationState.TOOLS);
  }

  // ── Both reasoning + tools ─────────────────────────────────────────

  @Test
  void shouldHandleBothReasoningAndTools() {
    initWithBothTags();

    // Reasoning phase
    var s1 = fsm.evaluate(GenerationState.ANSWER, "<think>");
    assertThat(s1).isEqualTo(GenerationState.REASONING);

    var s2 = fsm.evaluate(GenerationState.REASONING, "let me think");
    assertThat(s2).isEqualTo(GenerationState.REASONING);

    var s3 = fsm.evaluate(GenerationState.REASONING, "</think>");
    assertThat(s3).isEqualTo(GenerationState.ANSWER);

    // Tool phase
    var s4 = fsm.evaluate(GenerationState.ANSWER, "<tool_call>");
    assertThat(s4).isEqualTo(GenerationState.TOOLS);

    var s5 = fsm.evaluate(GenerationState.TOOLS, "</tool_call>");
    assertThat(s5).isEqualTo(GenerationState.ANSWER);
  }

  // ── Reset ──────────────────────────────────────────────────────────

  @Test
  void reset_shouldAllowReasoningAgain() {
    initWithReasoningTags();

    // Use reasoning
    fsm.evaluate(GenerationState.ANSWER, "<think>");
    fsm.evaluate(GenerationState.REASONING, "</think>");

    // After reset, reasoning should work again
    fsm.reset();
    var state = fsm.evaluate(GenerationState.ANSWER, "<think>");
    assertThat(state).isEqualTo(GenerationState.REASONING);
  }

  @Test
  void noTagsInText_shouldStayInAnswer() {
    initWithReasoningTags();

    var state = fsm.evaluate(
      GenerationState.ANSWER,
      "just normal text without tags"
    );
    assertThat(state).isEqualTo(GenerationState.ANSWER);
  }

  // ── Helpers ────────────────────────────────────────────────────────

  private void initWithReasoningTags() {
    fsm.initialize(
      List.of(new TagBounds(GenerationState.REASONING, "<think>", "</think>"))
    );
  }

  private void initWithToolTags() {
    fsm.initialize(
      List.of(
        new TagBounds(GenerationState.TOOLS, "<tool_call>", "</tool_call>")
      )
    );
  }

  private void initWithBothTags() {
    fsm.initialize(
      List.of(
        new TagBounds(GenerationState.REASONING, "<think>", "</think>"),
        new TagBounds(GenerationState.TOOLS, "<tool_call>", "</tool_call>")
      )
    );
  }
}
