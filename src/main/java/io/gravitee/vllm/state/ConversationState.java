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

  // ── Lifecycle ───────────────────────────────────────────────────────

  /**
   * Initializes the state for a new generation turn.
   *
   * @param promptTokenCount number of prompt tokens
   */
  public void initialize(int promptTokenCount) {
    tokenTracking.initialize(promptTokenCount);
    if (!tagBounds.isEmpty()) {
      stateEvaluation.initialize(tagBounds);
    }
    stateEvaluation.reset();
    currentState = GenerationState.ANSWER;
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
   * @param delta      the new text fragment
   * @param tokenCount number of tokens this delta represents
   * @return the generation state after this delta
   */
  public GenerationState evaluate(String delta, int tokenCount) {
    GenerationState previousState = currentState;

    if (isClassificationEnabled()) {
      currentState = stateEvaluation.evaluate(currentState, delta);
    }

    // Detect TOOLS → ANSWER transition → TOOL_CALL finish reason
    if (
      previousState == GenerationState.TOOLS &&
      currentState == GenerationState.ANSWER
    ) {
      setFinishReason(FinishReason.TOOL_CALL);
    }

    tokenTracking.consume(currentState, tokenCount);
    return currentState;
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
