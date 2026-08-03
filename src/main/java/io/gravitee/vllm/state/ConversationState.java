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

import io.gravitee.vllm.engine.FinishReason;
import io.gravitee.vllm.state.StateEvaluation.Emission;
import java.util.ArrayList;
import java.util.List;

/**
 * Tracks the generation state of a conversation turn, classifying
 * output tokens into semantic categories (answer, reasoning, tools)
 * and maintaining per-category counters.
 *
 * <p>Mirrors llamaj.cpp's {@code ConversationState} pattern — a fluent
 * configuration holder that combines a {@link StateEvaluation} FSM with
 * {@link TokenTracking} counters.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var state = new ConversationState()
 *     .reasoning("<think>", "</think>")
 *     .toolCall("<tool_call>", "</tool_call>");
 *
 * state.initialize(promptTokenCount);
 *
 * // On each generated delta:
 * GenerationState classified = state.evaluate(delta);
 * // Counters and finish reason are updated automatically.
 *
 * // Query:
 * state.answerTokens();
 * state.reasoningTokens();
 * state.toolsTokens();
 * state.finishReason();
 * }</pre>
 */
public final class ConversationState {

  private final List<TagBounds> tagBounds = new ArrayList<>();
  private final StateEvaluation stateEvaluation = new StateEvaluation();
  private final TokenTracking tokenTracking = new TokenTracking();

  private GenerationState currentState = GenerationState.ANSWER;
  private FinishReason finishReason;

  // ── Fluent configuration (before initialize) ────────────────────────

  /**
   * Configures reasoning tag boundaries.
   *
   * @param openTag  e.g. {@code "<think>"}
   * @param closeTag e.g. {@code "</think>"}
   * @return this
   */
  public ConversationState reasoning(String openTag, String closeTag) {
    tagBounds.add(new TagBounds(GenerationState.REASONING, openTag, closeTag));
    return this;
  }

  /**
   * Configures tool-call tag boundaries.
   *
   * @param openTag  e.g. {@code "<tool_call>"}
   * @param closeTag e.g. {@code "</tool_call>"}
   * @return this
   */
  public ConversationState toolCall(String openTag, String closeTag) {
    tagBounds.add(new TagBounds(GenerationState.TOOLS, openTag, closeTag));
    return this;
  }

  /**
   * Configures tool-call tag boundaries where the channel can be opened more
   * than one way — Harmony uses both {@code commentary} and {@code analysis}.
   * Each alternative must be complete from the start of a run; with only one
   * configured, the other leaks into the answer as raw text.
   *
   * @param openTags the opening markers, any of which enters the tool channel
   * @param closeTag the closing marker
   * @return this
   */
  public ConversationState toolCall(List<String> openTags, String closeTag) {
    tagBounds.add(new TagBounds(GenerationState.TOOLS, openTags, closeTag));
    return this;
  }

  /**
   * Configures reasoning boundaries where the channel can be left more than one
   * way.
   *
   * <p>Harmony needs this: reasoning ends into the final channel via
   * {@code <|end|><|start|>assistant<|channel|>final<|message|>} when the model
   * answers directly, but the run that precedes the final channel is terminated
   * by {@code <|call|>} when a tool call intervenes. One marker cannot cover
   * both, and the one that misses leaks its header into the answer.
   *
   * @param openTags  the opening markers, any of which enters reasoning
   * @param closeTags the closing markers, any of which leaves it
   * @return this
   */
  public ConversationState reasoning(
    List<String> openTags,
    List<String> closeTags
  ) {
    tagBounds.add(
      new TagBounds(GenerationState.REASONING, openTags, closeTags)
    );
    return this;
  }

  /**
   * Configures tool-call boundaries where the channel can be left more than one
   * way.
   *
   * <p>{@code <|call|>} ends a tool call, but when the model continues into an
   * answer rather than stopping, the final-channel header follows immediately
   * and belongs to the marker — otherwise it is stranded in the answer as a
   * visible {@code <|channel|>final<|message|>}. Listing both lets the longer
   * one win when it arrives and the shorter one settle the span when generation
   * stops at the call, which is the normal agent flow.
   *
   * @param openTags  the opening markers, any of which enters the tool channel
   * @param closeTags the closing markers, any of which leaves it
   * @return this
   */
  public ConversationState toolCall(
    List<String> openTags,
    List<String> closeTags
  ) {
    tagBounds.add(new TagBounds(GenerationState.TOOLS, openTags, closeTags));
    return this;
  }

  // ── Lifecycle ───────────────────────────────────────────────────────

  /**
   * Initializes the state for a new generation turn.
   *
   * @param promptTokenCount number of prompt tokens
   */
  public void initialize(int promptTokenCount) {
    initialize(promptTokenCount, null);
  }

  /**
   * Initializes the state for a new generation turn, seeding the starting state
   * from the rendered prompt.
   *
   * <p>A chat template may end inside a span it opened itself (a template ending
   * with {@code <think>}), and a continuation prompt may end anywhere inside an
   * unfinished one. The model never re-emits that open tag, so without the seed
   * the whole span is misclassified as answer.
   *
   * @param promptTokenCount number of prompt tokens
   * @param prompt           the rendered prompt, or {@code null} if unavailable
   */
  public void initialize(int promptTokenCount, String prompt) {
    tokenTracking.initialize(promptTokenCount);
    if (!tagBounds.isEmpty()) {
      stateEvaluation.initialize(tagBounds);
    }
    stateEvaluation.reset();
    currentState = isClassificationEnabled()
      ? stateEvaluation.initialState(prompt)
      : GenerationState.ANSWER;
    finishReason = null;
  }

  /** Whether tag-based classification is configured. */
  public boolean isClassificationEnabled() {
    return !tagBounds.isEmpty() && stateEvaluation.isInitialized();
  }

  // ── Evaluation (called per delta) ───────────────────────────────────

  /**
   * Evaluates a text delta, updating the generation state and token counters.
   *
   * <p>The returned {@link Emission} carries the text that should actually be
   * emitted for this step: tag markers are syntax and are suppressed, and while
   * a partial marker is buffered nothing is emitted at all. Callers must emit
   * {@link Emission#emit()} rather than the raw delta, or the markers reach the
   * client verbatim.
   *
   * @param delta      the new text fragment
   * @param tokenCount number of tokens this delta represents
   * @return the state, the text to emit, and the tokens it accounts for
   */
  public Emission evaluate(String delta, int tokenCount) {
    if (!isClassificationEnabled()) {
      tokenTracking.consume(currentState, tokenCount);
      return new Emission(currentState, delta == null ? "" : delta, tokenCount);
    }

    GenerationState previousState = currentState;
    Emission emission = stateEvaluation.evaluate(
      currentState,
      delta,
      tokenCount
    );
    currentState = emission.state();

    // Leaving the tool channel — by its close marker or by another channel's
    // open marker — is what makes this generation a tool call.
    if (
      previousState == GenerationState.TOOLS &&
      currentState != GenerationState.TOOLS
    ) {
      setFinishReason(FinishReason.TOOL_CALL);
    }

    // Buffered deltas count 0 now and their full weight when they resolve, so
    // no token is lost or double-counted.
    tokenTracking.consume(currentState, emission.emitTokens());
    return emission;
  }

  /**
   * Flushes text buffered behind an unconfirmed marker at the end of
   * generation, attributing it to the current channel.
   *
   * <p>Callers must append {@link Emission#emit()} to the final delta: a
   * generation that stops mid-marker (a truncated {@code </thin}) otherwise
   * drops that text and its tokens on the floor.
   */
  public Emission flush() {
    if (!isClassificationEnabled()) {
      return new Emission(currentState, "", 0);
    }
    GenerationState previousState = currentState;
    Emission emission = stateEvaluation.flushPending(currentState);
    // A flush can now close a span: when generation stops on a marker that is
    // also the prefix of a longer alternative — <|call|> ending an agent turn —
    // the buffered match is settled here. Adopting the emitted state keeps
    // currentState() truthful, so a turn does not end reported as still inside
    // a tool call.
    currentState = emission.state();

    // ...and a turn that ENDS in the tool channel is a tool call, whether the
    // close marker settled here or never arrived at all. evaluate() only stamps
    // TOOL_CALL when it sees the transition mid-stream, so an agent turn whose
    // <|call|> is the EOS token — the normal case — reached the executor as
    // finish_reason=stop, and the span was rendered to the user as content
    // instead of being extracted and run.
    //
    // Only when something was actually captured in there. Markers that share a
    // prefix — Harmony's reasoning-close and tool-open agree for 34 characters —
    // let the machine enter TOOLS provisionally and resolve straight back out,
    // emitting nothing. Reporting a tool call for an empty span invites callers
    // to hunt for one in the plain answer and manufacture it.
    if (
      previousState == GenerationState.TOOLS &&
      tokenTracking.outputTokens(GenerationState.TOOLS) > 0
    ) {
      setFinishReason(FinishReason.TOOL_CALL);
    }

    tokenTracking.consume(emission.state(), emission.emitTokens());
    return emission;
  }

  // ── Finish reason (with priority logic from llamaj.cpp) ─────────────

  /**
   * Sets the finish reason, respecting priority rules:
   * <ul>
   *   <li>{@code TOOL_CALL} is not overwritten by {@code STOP} or {@code ABORT}.</li>
   *   <li>{@code LENGTH} always wins.</li>
   * </ul>
   */
  public void setFinishReason(FinishReason reason) {
    if (reason == null) return;
    if (reason == FinishReason.LENGTH) {
      // LENGTH always wins
      this.finishReason = reason;
    } else if (this.finishReason != FinishReason.TOOL_CALL) {
      // TOOL_CALL is preserved over STOP/ABORT
      this.finishReason = reason;
    }
  }

  // ── Accessors ───────────────────────────────────────────────────────

  public GenerationState currentState() {
    return currentState;
  }

  public FinishReason finishReason() {
    return finishReason;
  }

  public int inputTokens() {
    return tokenTracking.inputTokens();
  }

  public int answerTokens() {
    return tokenTracking.outputTokens(GenerationState.ANSWER);
  }

  public int reasoningTokens() {
    return tokenTracking.outputTokens(GenerationState.REASONING);
  }

  public int toolsTokens() {
    return tokenTracking.outputTokens(GenerationState.TOOLS);
  }

  public int totalOutputTokens() {
    return tokenTracking.totalOutputTokens();
  }

  public int totalTokens() {
    return tokenTracking.totalTokens();
  }

  public TokenTracking tokenTracking() {
    return tokenTracking;
  }
}
