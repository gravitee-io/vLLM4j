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
import org.junit.jupiter.api.Test;

/**
 * Leaving a channel by more than one marker.
 *
 * <p>Harmony forces this. Reasoning ends into the final channel through
 * {@code <|end|><|start|>assistant<|channel|>final<|message|>} when the model
 * answers directly, but through {@code <|call|>} when a tool call intervenes —
 * and a tool call itself either ends generation (the agent flow, where
 * {@code <|call|>} is an EOS token) or is followed immediately by the
 * final-channel header. A single close marker covers one of those and leaks the
 * other into the answer as visible syntax.
 *
 * <p>The subtle case is a complete match that may still grow: {@code <|call|>}
 * matches while {@code <|call|><|start|>assistant…} may still be arriving.
 * Committing immediately makes the longer marker unreachable; waiting without
 * remembering the match strands the state machine when it never comes.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class CloseAlternativesTest {

  private static final String ANALYSIS_OPEN = "<|channel|>analysis<|message|>";
  private static final String FINAL_HEADER =
    "<|start|>assistant<|channel|>final<|message|>";
  private static final String TOOL_OPEN = "<|start|>assistant to=functions.";

  private static ConversationState harmony() {
    ConversationState state = new ConversationState()
      .reasoning(
        List.of(ANALYSIS_OPEN),
        List.of("<|end|>" + FINAL_HEADER, "<|call|>" + FINAL_HEADER)
      )
      .toolCall(
        // Both alignments: the run may begin at <|start|> or carry the <|end|>
        // that terminated the analysis message, and a close marker sharing that
        // prefix buffers it away from the start of the run.
        List.of("<|end|>" + TOOL_OPEN, TOOL_OPEN),
        List.of("<|call|>" + FINAL_HEADER, "<|call|>")
      );
    state.initialize(0);
    return state;
  }

  /** Feeds per-token deltas, as vLLM streams them, and collects each channel. */
  private static Segments feed(ConversationState state, String... tokens) {
    Segments out = new Segments();
    for (String token : tokens) {
      StateEvaluation.Emission emission = state.evaluate(token, 1);
      out.append(emission.state(), emission.emit());
    }
    StateEvaluation.Emission flushed = state.flush();
    if (flushed != null) {
      out.append(flushed.state(), flushed.emit());
    }
    out.finalState = state.currentState();
    return out;
  }

  @Test
  void answering_directly_leaves_reasoning_by_the_end_marker() {
    Segments s = feed(
      harmony(),
      "<|channel|>",
      "analysis",
      "<|message|>",
      "Think",
      ".",
      "<|end|>",
      "<|start|>",
      "assistant",
      "<|channel|>",
      "final",
      "<|message|>",
      "Hello",
      "!"
    );

    assertThat(s.reasoning).isEqualTo("Think.");
    assertThat(s.answer).isEqualTo("Hello!");
  }

  @Test
  void a_tool_call_that_ends_generation_still_closes_its_span() {
    // The agent flow: <|call|> is an EOS token, so nothing follows it. The
    // longer alternative never arrives and the span must settle on flush —
    // otherwise the turn ends stuck in TOOLS with a stray <|call|> emitted.
    Segments s = feed(
      harmony(),
      "<|channel|>",
      "analysis",
      "<|message|>",
      "Need",
      " a",
      " file",
      "<|end|>",
      "<|start|>",
      "assistant",
      " to=functions.",
      "write",
      "<|channel|>",
      "commentary",
      "<|message|>",
      "{\"p\":1}",
      "<|call|>"
    );

    assertThat(s.tools).isEqualTo(
      "write<|channel|>commentary<|message|>{\"p\":1}"
    );
    assertThat(s.answer).doesNotContain("<|call|>");
    assertThat(s.finalState).isEqualTo(GenerationState.ANSWER);
  }

  @Test
  void a_tool_call_the_model_continues_past_hides_the_final_header() {
    // Here the longer alternative does arrive, and taking it is what keeps
    // "<|channel|>final<|message|>" out of the answer.
    Segments s = feed(
      harmony(),
      "<|channel|>",
      "analysis",
      "<|message|>",
      "Need",
      " a",
      " file",
      "<|end|>",
      "<|start|>",
      "assistant",
      " to=functions.",
      "write",
      "<|channel|>",
      "commentary",
      "<|message|>",
      "{\"p\":1}",
      "<|call|>",
      "<|start|>",
      "assistant",
      "<|channel|>",
      "final",
      "<|message|>",
      "Created",
      "!"
    );

    assertThat(s.tools).isEqualTo(
      "write<|channel|>commentary<|message|>{\"p\":1}"
    );
    assertThat(s.answer).isEqualTo("Created!");
    assertThat(s.answer).doesNotContain("<|channel|>", "<|start|>");
  }

  @Test
  void the_longest_matching_close_wins() {
    ConversationState state = new ConversationState().reasoning(
      List.of("<open>"),
      List.of("<close>", "<close>extra")
    );
    state.initialize(0);

    Segments s = feed(state, "<open>", "body", "<close>", "extra", "tail");

    // "<close>extra" is the longer match, so "extra" is syntax, not content.
    assertThat(s.reasoning).isEqualTo("body");
    assertThat(s.answer).isEqualTo("tail");
  }

  private static final class Segments {

    String reasoning = "";
    String tools = "";
    String answer = "";
    GenerationState finalState;

    void append(GenerationState state, String text) {
      if (text == null || text.isEmpty()) {
        return;
      }
      switch (state) {
        case REASONING -> reasoning += text;
        case TOOLS -> tools += text;
        default -> answer += text;
      }
    }
  }
}
