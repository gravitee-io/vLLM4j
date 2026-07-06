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

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Finite state machine that detects tag boundaries in generated text
 * and transitions between {@link GenerationState}s.
 *
 * <p>Mirrors llamaj.cpp's {@code StateEvaluation} module. The FSM operates
 * on text deltas (substrings) accumulated into a buffer, detecting open/close
 * tags across token boundaries.
 *
 * <h2>Transition rules</h2>
 * <ul>
 *   <li>From {@code ANSWER}: if the accumulated text contains any configured
 *       {@code openTag}, transition to that state.</li>
 *   <li>From {@code REASONING} or {@code TOOLS}: if the accumulated text
 *       contains the matching {@code closeTag}, transition back to {@code ANSWER}.</li>
 * </ul>
 *
 * <h2>Re-entry rules</h2>
 * <ul>
 *   <li>{@code REASONING} can occur at most once per generation.</li>
 *   <li>{@code TOOLS} can occur multiple times (models may produce
 *       several tool calls).</li>
 * </ul>
 */
public final class StateEvaluation {

  private Map<GenerationState, TagBounds> tagsByState;
  private Map<GenerationState, Boolean> occurred;
  private final StringBuilder buffer = new StringBuilder();

  /**
   * Longest configured tag. Bounds how much history a tag could still span,
   * and so how much of the buffer is worth keeping.
   */
  private int longestTag;

  /**
   * Initializes the FSM with the given tag configurations.
   * Must be called before {@link #evaluate}.
   */
  public void initialize(List<TagBounds> tags) {
    tagsByState = new EnumMap<>(GenerationState.class);
    occurred = new EnumMap<>(GenerationState.class);
    longestTag = 0;
    for (TagBounds tb : tags) {
      tagsByState.put(tb.state(), tb);
      occurred.put(tb.state(), false);
      longestTag = Math.max(
        longestTag,
        Math.max(length(tb.openTag()), length(tb.closeTag()))
      );
    }
    buffer.setLength(0);
  }

  /** Whether this FSM has been initialized with at least one tag config. */
  public boolean isInitialized() {
    return tagsByState != null && !tagsByState.isEmpty();
  }

  /**
   * Feeds a text delta into the FSM and returns the new generation state.
   *
   * @param currentState the current state before this delta
   * @param delta        the new text fragment
   * @return the state after evaluating this delta
   */
  public GenerationState evaluate(GenerationState currentState, String delta) {
    if (!isInitialized() || delta == null || delta.isEmpty()) {
      return currentState != null ? currentState : GenerationState.ANSWER;
    }

    buffer.append(delta);

    GenerationState next = switch (currentState) {
      case ANSWER -> detectOpenTag();
      case REASONING, TOOLS -> detectCloseTag(currentState);
      case null -> GenerationState.ANSWER;
    };

    trimBuffer();
    return next;
  }

  /**
   * Keeps the buffer bounded to the longest tag minus one character.
   *
   * <p>Without this the buffer grows for the whole generation — it is only
   * cleared on a tag transition, and a plain answer contains none — while every
   * delta calls {@code toString()} and {@code lastIndexOf} over all of it. That
   * is quadratic in the response length, and on a fast model it costs more than
   * decoding the token does.
   *
   * <p>Discarding the older text is safe because the buffer is scanned after
   * <em>every</em> delta: a tag is detected in the step its final character
   * arrives, so the only history that can still matter is a partial tag, which
   * is at most {@code longestTag - 1} characters.
   */
  private void trimBuffer() {
    int keep = longestTag > 0 ? longestTag - 1 : 0;
    if (buffer.length() > keep) {
      buffer.delete(0, buffer.length() - keep);
    }
  }

  private static int length(String tag) {
    return tag == null ? 0 : tag.length();
  }

  /**
   * Resets the internal text buffer. Call this when starting a new
   * generation turn.
   */
  public void reset() {
    buffer.setLength(0);
    if (occurred != null) {
      occurred.replaceAll((k, v) -> false);
    }
  }

  // ── Internal ────────────────────────────────────────────────────────

  private GenerationState detectOpenTag() {
    String text = buffer.toString();
    for (var entry : tagsByState.entrySet()) {
      GenerationState state = entry.getKey();
      TagBounds bounds = entry.getValue();

      // Skip states that already occurred (except TOOLS which can repeat)
      if (
        state != GenerationState.TOOLS &&
        Boolean.TRUE.equals(occurred.get(state))
      ) {
        continue;
      }

      int idx = text.lastIndexOf(bounds.openTag());
      if (idx >= 0) {
        // Transition — clear buffer up to after the open tag
        buffer.setLength(0);
        return state;
      }
    }
    return GenerationState.ANSWER;
  }

  private GenerationState detectCloseTag(GenerationState currentState) {
    TagBounds bounds = tagsByState.get(currentState);
    if (bounds == null) return GenerationState.ANSWER;

    String text = buffer.toString();
    int idx = text.lastIndexOf(bounds.closeTag());
    if (idx >= 0) {
      // Mark occurred (REASONING once, TOOLS can repeat)
      if (currentState != GenerationState.TOOLS) {
        occurred.put(currentState, true);
      }
      buffer.setLength(0);
      return GenerationState.ANSWER;
    }
    return currentState;
  }
}
