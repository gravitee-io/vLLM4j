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

/**
 * The tag FSM, which decides both what channel text belongs to and what text
 * the client is allowed to see.
 *
 * <p>Emission is the half that is easy to forget: markers are syntax, so they
 * must never reach the client, and a marker split across deltas must not leak
 * its fragments while it is still unconfirmed. Both are asserted here rather
 * than only the resulting state.
 */
class StateEvaluationTest {

  private StateEvaluation fsm;

  @BeforeEach
  void setUp() {
    fsm = new StateEvaluation();
  }

  @Test
  void uninitializedFsm_shouldEmitTextUnchanged() {
    assertThat(fsm.isInitialized()).isFalse();

    var emission = fsm.evaluate(GenerationState.ANSWER, "hello", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("hello");
    assertThat(emission.emitTokens()).isEqualTo(1);
  }

  @Test
  void nullDelta_shouldStillCountItsToken() {
    initWithReasoningTags();

    // A token that decodes to nothing (half a multi-byte character) is still a
    // generated token; dropping it under-reports completion_tokens.
    var emission = fsm.evaluate(GenerationState.ANSWER, null, 1);

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEmpty();
    assertThat(emission.emitTokens()).isEqualTo(1);
  }

  @Test
  void emptyDelta_shouldStillCountItsToken() {
    initWithReasoningTags();

    var emission = fsm.evaluate(GenerationState.ANSWER, "", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emitTokens()).isEqualTo(1);
  }

  // ── Reasoning tag transitions ──────────────────────────────────────

  @Test
  void shouldTransitionToReasoningOnOpenTag_andSuppressTheTag() {
    initWithReasoningTags();

    var emission = fsm.evaluate(GenerationState.ANSWER, "<think>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.REASONING);
    // The tag is syntax: the client must never see it in its content stream.
    assertThat(emission.emit()).isEmpty();
    // ...but the token it cost is billed, to the channel it opened.
    assertThat(emission.emitTokens()).isEqualTo(1);
  }

  @Test
  void shouldTransitionBackToAnswerOnCloseTag() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    var emission = fsm.evaluate(GenerationState.REASONING, "</think>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEmpty();
  }

  @Test
  void aDeltaSpanningTheCloseTag_shouldEmitOnlyTheRemainder() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    var emission = fsm.evaluate(
      GenerationState.REASONING,
      "</think>The answer",
      1
    );

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("The answer");
  }

  // ── Split markers: the leak this FSM exists to prevent ─────────────

  @Test
  void aTagSplitAcrossDeltas_shouldNotLeakItsFragments() {
    initWithReasoningTags();

    // "<thi" is a strict prefix of "<think>": buffered, nothing emitted, and
    // crucially not billed yet either.
    var first = fsm.evaluate(GenerationState.ANSWER, "<thi", 1);
    assertThat(first.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(first.emit()).isEmpty();
    assertThat(first.emitTokens()).isZero();
    assertThat(fsm.hasPending()).isTrue();

    var second = fsm.evaluate(GenerationState.ANSWER, "nk>", 1);
    assertThat(second.state()).isEqualTo(GenerationState.REASONING);
    assertThat(second.emit()).isEmpty();
    // Both buffered tokens resolve here, so nothing is lost.
    assertThat(second.emitTokens()).isEqualTo(2);
    assertThat(fsm.hasPending()).isFalse();
  }

  @Test
  void textThatMerelyLooksLikeATag_shouldBeEmittedWhenRefuted() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<thi", 1);
    var emission = fsm.evaluate(GenerationState.ANSWER, "s is prose", 1);

    // Refutation: buffer flushed to the current channel, nothing swallowed.
    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("<this is prose");
    assertThat(emission.emitTokens()).isEqualTo(2);
  }

  @Test
  void aRefutingDeltaThatStartsANewTag_shouldBeRescanned() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<thi", 1);
    // "x" refutes "<thi"; the rest of this delta opens a fresh candidate.
    var emission = fsm.evaluate(GenerationState.ANSWER, "<th", 1);

    assertThat(emission.emit()).isEqualTo("<thi");
    assertThat(emission.emitTokens()).isEqualTo(1);
    // The refuting delta itself is now buffered as a candidate prefix.
    assertThat(fsm.hasPending()).isTrue();
  }

  @Test
  void flushPending_shouldReleaseAnUnconfirmedTagAtEndOfGeneration() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<thi", 1);
    var flushed = fsm.flushPending(GenerationState.ANSWER);

    // Generation stopped mid-marker; without the flush this text and its token
    // would simply vanish.
    assertThat(flushed.emit()).isEqualTo("<thi");
    assertThat(flushed.emitTokens()).isEqualTo(1);
    assertThat(fsm.hasPending()).isFalse();
  }

  @Test
  void flushPending_shouldBeANoopWhenNothingIsBuffered() {
    initWithReasoningTags();

    var flushed = fsm.flushPending(GenerationState.REASONING);

    assertThat(flushed.state()).isEqualTo(GenerationState.REASONING);
    assertThat(flushed.emit()).isEmpty();
    assertThat(flushed.emitTokens()).isZero();
  }

  @Test
  void reasoningShouldNotReenter() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    fsm.evaluate(GenerationState.REASONING, "thinking...", 1);
    fsm.evaluate(GenerationState.REASONING, "</think>", 1);

    // A second <think> is no longer a marker, so it is plain content — and is
    // emitted rather than silently swallowed.
    var emission = fsm.evaluate(GenerationState.ANSWER, "<think>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("<think>");
  }

  // ── Tools tag transitions ──────────────────────────────────────────

  @Test
  void shouldTransitionToToolsOnOpenTag() {
    initWithToolTags();

    var emission = fsm.evaluate(GenerationState.ANSWER, "<tool_call>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.TOOLS);
    assertThat(emission.emit()).isEmpty();
  }

  @Test
  void toolsShouldAllowReentry() {
    initWithToolTags();

    fsm.evaluate(GenerationState.ANSWER, "<tool_call>", 1);
    fsm.evaluate(GenerationState.TOOLS, "{\"name\":\"foo\"}", 1);
    fsm.evaluate(GenerationState.TOOLS, "</tool_call>", 1);

    var emission = fsm.evaluate(GenerationState.ANSWER, "<tool_call>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.TOOLS);
  }

  @Test
  void aChannelWithSeveralOpeningMarkers_shouldEnterOnAnyOfThem() {
    // Harmony opens its tool channel as both commentary and analysis;
    // configuring only one leaks the other into the answer as raw text.
    fsm.initialize(
      List.of(
        new TagBounds(
          GenerationState.TOOLS,
          List.of("<tool_call>", "<function_call>"),
          "</tool_call>"
        )
      )
    );

    var emission = fsm.evaluate(GenerationState.ANSWER, "<function_call>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.TOOLS);
    assertThat(emission.emit()).isEmpty();
  }

  @Test
  void whenMarkersSharePrefixes_theLongestShouldWin() {
    // Both markers match "<tool_call_json>"; taking the shorter one would emit
    // "_json>" as content.
    fsm.initialize(
      List.of(
        new TagBounds(
          GenerationState.TOOLS,
          List.of("<tool_call>", "<tool_call_json>"),
          "</tool_call>"
        )
      )
    );

    var emission = fsm.evaluate(GenerationState.ANSWER, "<tool_call_json>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.TOOLS);
    assertThat(emission.emit()).isEmpty();
  }

  // ── Both reasoning + tools ─────────────────────────────────────────

  @Test
  void shouldHandleBothReasoningAndTools() {
    initWithBothTags();

    assertThat(
      fsm.evaluate(GenerationState.ANSWER, "<think>", 1).state()
    ).isEqualTo(GenerationState.REASONING);
    assertThat(
      fsm.evaluate(GenerationState.REASONING, "let me think", 1).state()
    ).isEqualTo(GenerationState.REASONING);
    assertThat(
      fsm.evaluate(GenerationState.REASONING, "</think>", 1).state()
    ).isEqualTo(GenerationState.ANSWER);
    assertThat(
      fsm.evaluate(GenerationState.ANSWER, "<tool_call>", 1).state()
    ).isEqualTo(GenerationState.TOOLS);
    assertThat(
      fsm.evaluate(GenerationState.TOOLS, "</tool_call>", 1).state()
    ).isEqualTo(GenerationState.ANSWER);
  }

  @Test
  void channelsShouldChainRatherThanNest() {
    initWithBothTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    // A tool call opening straight out of the reasoning span closes it
    // implicitly — models do not always emit the close tag first.
    var emission = fsm.evaluate(GenerationState.REASONING, "<tool_call>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.TOOLS);
    assertThat(emission.emit()).isEmpty();
  }

  // ── Prompt-seeded initial state ────────────────────────────────────

  @Test
  void aPromptEndingInsideAnOpenSpan_shouldSeedThatState() {
    initWithReasoningTags();

    // Templates that pre-fill <think> mean the model never emits it, so
    // without the seed the whole reasoning block reads as answer.
    assertThat(fsm.initialState("<|im_start|>assistant\n<think>\n")).isEqualTo(
      GenerationState.REASONING
    );
  }

  @Test
  void aPromptWhoseSpanIsClosed_shouldStartInAnswer() {
    initWithReasoningTags();

    assertThat(fsm.initialState("<think>earlier turn</think>done\n")).isEqualTo(
      GenerationState.ANSWER
    );
  }

  @Test
  void anAbsentOrUnknownPrompt_shouldStartInAnswer() {
    initWithReasoningTags();

    assertThat(fsm.initialState(null)).isEqualTo(GenerationState.ANSWER);
    assertThat(fsm.initialState("just a question")).isEqualTo(
      GenerationState.ANSWER
    );
  }

  // ── Reset ──────────────────────────────────────────────────────────

  @Test
  void reset_shouldAllowReasoningAgain() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    fsm.evaluate(GenerationState.REASONING, "</think>", 1);

    fsm.reset();
    var emission = fsm.evaluate(GenerationState.ANSWER, "<think>", 1);

    assertThat(emission.state()).isEqualTo(GenerationState.REASONING);
  }

  @Test
  void reset_shouldDropBufferedMarkerText() {
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<thi", 1);
    fsm.reset();

    assertThat(fsm.hasPending()).isFalse();
  }

  @Test
  void noTagsInText_shouldStayInAnswer() {
    initWithReasoningTags();

    var emission = fsm.evaluate(
      GenerationState.ANSWER,
      "just normal text without tags",
      1
    );

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("just normal text without tags");
  }

  // ── Stray syntax in the answer channel ─────────────────────────────

  @Test
  void aCloseMarkerArrivingInAnswer_shouldBeSuppressed() {
    // The reported bug. After a tool call, generation restarts in ANSWER and
    // Harmony still emits its final-channel header first — no span is open for
    // it to close, so before this it reached the client as
    // "<|channel|>final<|message|>DONE".
    fsm.initialize(
      List.of(
        new TagBounds(
          GenerationState.REASONING,
          List.of("<|channel|>analysis<|message|>"),
          List.of("<|channel|>final<|message|>")
        )
      )
    );

    var emission = fsm.evaluate(
      GenerationState.ANSWER,
      "<|channel|>final<|message|>DONE",
      1
    );

    assertThat(emission.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(emission.emit()).isEqualTo("DONE");
  }

  @Test
  void aLiteralCloseMarkerInAnswerProse_isSuppressed_byDesign() {
    // The deliberate cost of the Harmony fix above: a close marker is treated
    // as stray syntax in ANSWER even when the model meant it as prose, so
    // "Wrap it in </think> like so." loses the marker text. Markers are
    // special tokens in every supported dialect, so a model producing one as
    // content is already off-template; suppressing it is the safer failure.
    initWithReasoningTags();

    var before = fsm.evaluate(GenerationState.ANSWER, "Wrap it in ", 1);
    var marker = fsm.evaluate(GenerationState.ANSWER, "</think>", 1);
    var after = fsm.evaluate(GenerationState.ANSWER, " like so.", 1);

    assertThat(before.emit()).isEqualTo("Wrap it in ");
    assertThat(marker.emit()).isEmpty();
    assertThat(marker.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(after.emit()).isEqualTo(" like so.");
  }

  @Test
  void aStrayCloseMarkerSplitAcrossDeltas_shouldNotLeakEither() {
    initWithReasoningTags();

    var first = fsm.evaluate(GenerationState.ANSWER, "</thi", 1);
    assertThat(first.emit()).isEmpty();

    var second = fsm.evaluate(GenerationState.ANSWER, "nk>Answer", 1);
    assertThat(second.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(second.emit()).isEqualTo("Answer");
  }

  @Test
  void ordinaryAnswerText_shouldStillPassThroughUntouched() {
    // The guard on the above: suppressing stray closes must not start eating
    // content that merely begins like one.
    initWithReasoningTags();

    var emission = fsm.evaluate(
      GenerationState.ANSWER,
      "</thinking about it",
      1
    );

    assertThat(emission.emit()).isEqualTo("</thinking about it");
  }

  // ── Re-entering a channel within one generation ────────────────────

  @Test
  void aRepeatableChannelCanBeReEnteredInOneGeneration() {
    // Harmony chains channels: analysis, back to final, then commentary — all in
    // ONE generation. Non-repeatable, that second opening stops matching and its
    // header reaches the client as raw text, with the prose billed as answer.
    fsm.initialize(
      List.of(
        new TagBounds(
          GenerationState.REASONING,
          List.of(
            "<|channel|>analysis<|message|>",
            "<|channel|>commentary<|message|>"
          ),
          List.of("<|channel|>final<|message|>"),
          true
        )
      )
    );

    assertThat(
      fsm
        .evaluate(GenerationState.ANSWER, "<|channel|>analysis<|message|>", 1)
        .state()
    ).isEqualTo(GenerationState.REASONING);
    assertThat(
      fsm
        .evaluate(GenerationState.REASONING, "<|channel|>final<|message|>", 1)
        .state()
    ).isEqualTo(GenerationState.ANSWER);

    var reopened = fsm.evaluate(
      GenerationState.ANSWER,
      "<|channel|>commentary<|message|>",
      1
    );

    assertThat(reopened.state()).isEqualTo(GenerationState.REASONING);
    assertThat(reopened.emit()).isEmpty();
  }

  @Test
  void aNonRepeatableChannelStillOccursOnce() {
    // The guard this preserves: a model that types "<think>" in its answer must
    // not re-open reasoning after the real block closed.
    initWithReasoningTags();

    fsm.evaluate(GenerationState.ANSWER, "<think>", 1);
    fsm.evaluate(GenerationState.REASONING, "</think>", 1);
    var second = fsm.evaluate(GenerationState.ANSWER, "<think>", 1);

    assertThat(second.state()).isEqualTo(GenerationState.ANSWER);
    assertThat(second.emit()).isEqualTo("<think>");
  }

  @Test
  void aRepeatableChannelAbsorbsItsOwnOpenerWhileInsideIt() {
    // Models re-announce the channel they are already in: Harmony emits a second
    // <|channel|>analysis<|message|> mid-thought. A state's own openers were
    // skipped as candidates, leaving that header in the reasoning text as raw
    // protocol.
    fsm.initialize(
      List.of(
        new TagBounds(
          GenerationState.REASONING,
          List.of("<|channel|>analysis<|message|>"),
          List.of("<|channel|>final<|message|>"),
          true
        )
      )
    );

    fsm.evaluate(GenerationState.ANSWER, "<|channel|>analysis<|message|>", 1);
    fsm.evaluate(GenerationState.REASONING, "thinking... ", 1);
    var again = fsm.evaluate(
      GenerationState.REASONING,
      "<|channel|>analysis<|message|>",
      1
    );

    assertThat(again.state()).isEqualTo(GenerationState.REASONING);
    assertThat(again.emit()).isEmpty();
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
