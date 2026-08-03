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
 * State machine over generation channels (ANSWER / REASONING / TOOLS) driven by
 * tag markers, matching llamaj.cpp's {@code StateEvaluation}.
 *
 * <p>Matching is TEXT-based and streaming, so it is robust to arbitrary
 * tokenizations of the same marker: fused pieces, subword splits, or a delta
 * spanning the marker boundary such as {@code "thought\nThe"}. A delta whose
 * accumulated text is a strict prefix of a candidate marker is buffered — and
 * <em>emitted as an empty delta</em> — until either:
 *
 * <ul>
 *   <li><b>confirmation</b>: the marker text is fully covered. The state flips,
 *   the marker text is pure syntax and is suppressed (never emitted to any
 *   channel; its tokens are counted in the post-flip state), and the remainder
 *   of a boundary-spanning delta is emitted as the first text of the post-flip
 *   channel; or</li>
 *   <li><b>refutation</b>: the text diverges from every candidate. The buffered
 *   text is emitted in the current channel and the refuting delta is re-scanned,
 *   since it may itself start a marker prefix.</li>
 * </ul>
 *
 * <p>Suppressing markers is the whole point: without it the client sees raw
 * {@code <think>} / {@code </tool_call>} in its content stream, and a tokenizer
 * that splits a marker across deltas leaks the fragments before the FSM has
 * enough text to recognise them.
 *
 * <h2>Transition rules</h2>
 * Channels CHAIN rather than nest: in ANY state the candidates are the current
 * state's close marker (&rarr; ANSWER) and the open markers of the OTHER states
 * (&rarr; that state directly, the current span implicitly closing). Longest
 * match wins, since markers may share prefixes.
 *
 * <h2>Re-entry rules</h2>
 * <ul>
 *   <li>{@code REASONING} can occur at most once per generation.</li>
 *   <li>{@code TOOLS} can occur multiple times (models may produce several
 *       tool calls).</li>
 * </ul>
 */
public final class StateEvaluation {

  private Map<GenerationState, TagBounds> tagsByState;
  private Map<GenerationState, Boolean> occurred;

  /**
   * Streaming text buffer: while {@code pendingTokens > 0}, {@code pending}
   * holds the last deltas' text, all of it a strict prefix of at least one
   * candidate marker — so it is inherently bounded by the longest marker.
   */
  private final StringBuilder pending = new StringBuilder();

  private int pendingTokens;

  /**
   * A close/open marker that has matched completely while a LONGER candidate
   * sharing its prefix is still possible. Held rather than acted on, and
   * settled when the stream diverges or ends. {@code null} when no such match
   * is outstanding.
   */
  private String provisionalMarker;

  private GenerationState provisionalTarget;

  /**
   * Initializes the FSM with the given tag configurations.
   * Must be called before {@link #evaluate}.
   */
  public void initialize(List<TagBounds> tags) {
    tagsByState = new EnumMap<>(GenerationState.class);
    occurred = new EnumMap<>(GenerationState.class);
    for (TagBounds tb : tags) {
      tagsByState.put(tb.state(), tb);
      occurred.put(tb.state(), false);
    }
    resetBuffer();
  }

  /** Whether this FSM has been initialized with at least one tag config. */
  public boolean isInitialized() {
    return tagsByState != null && !tagsByState.isEmpty();
  }

  /**
   * Feeds a text delta into the FSM.
   *
   * @param currentState the current state before this delta
   * @param delta        the new text fragment
   * @param tokenCount   how many generated tokens this delta covers
   * @return the resulting state, the text to emit, and the tokens that text
   *         accounts for
   */
  public Emission evaluate(
    GenerationState currentState,
    String delta,
    int tokenCount
  ) {
    String piece = delta == null ? "" : delta;
    if (!isInitialized() || currentState == null) {
      return new Emission(GenerationState.ANSWER, piece, tokenCount);
    }
    if (pendingTokens == 0 && piece.isEmpty()) {
      // Nothing buffered and nothing to match (e.g. a delta that decodes to
      // "" because it completes a multi-byte character): plain emit.
      return new Emission(currentState, piece, tokenCount);
    }
    return matchText(currentState, piece, tokenCount, true);
  }

  /**
   * Flushes any buffered marker-prefix text — generation ended while a candidate
   * marker was still unconfirmed. The flushed text belongs to the current
   * channel.
   *
   * <p>Without this the trailing fragment of a marker that never completed is
   * silently dropped, and its tokens with it.
   */
  public Emission flushPending(GenerationState currentState) {
    if (pendingTokens == 0) {
      return new Emission(currentState, "", 0);
    }
    if (provisionalMarker != null) {
      // Generation ended while a longer close marker was still possible — the
      // agent case, where <|call|> IS the last token. Settle for the match that
      // did complete, so the span closes and the state machine does not end
      // stuck inside it.
      String marker = provisionalMarker;
      GenerationState target = provisionalTarget;
      if (currentState != GenerationState.ANSWER) {
        markOccurred(currentState);
      }
      String remainder = pending.toString().substring(marker.length());
      int tokens = pendingTokens;
      resetBuffer();
      return new Emission(target, remainder, tokens);
    }
    String text = pending.toString();
    int tokens = pendingTokens;
    resetBuffer();
    return new Emission(currentState, text, tokens);
  }

  /** Whether marker-prefix text is currently buffered. */
  public boolean hasPending() {
    return pendingTokens > 0;
  }

  /**
   * Resolves the state generation should start in for the given prompt.
   *
   * <p>Chat templates may pre-fill a state's open tag at the end of the prompt
   * (e.g. a template ending with {@code <think>}), and continuation prompts may
   * end anywhere inside an unfinished span. In both cases the model never
   * (re-)emits the open tag itself and generation must begin inside that state,
   * so a prompt whose LAST open-tag occurrence is not followed by the
   * corresponding close tag seeds that state. Matching is string-based on the
   * rendered prompt, so it is independent of how the markers tokenize.
   */
  public GenerationState initialState(String prompt) {
    if (!isInitialized() || prompt == null) {
      return GenerationState.ANSWER;
    }
    String trimmed = prompt.stripTrailing();
    for (TagBounds bounds : tagsByState.values()) {
      if (insideOpenSpan(trimmed, bounds)) {
        return bounds.state();
      }
    }
    return GenerationState.ANSWER;
  }

  /**
   * Resets the buffer and the occurrence flags. Call this when starting a new
   * generation turn.
   */
  public void reset() {
    resetBuffer();
    if (occurred != null) {
      occurred.replaceAll((k, v) -> false);
    }
  }

  // ── Internal ────────────────────────────────────────────────────────

  /**
   * Matches the accumulated text (buffer + delta) against the candidate markers.
   * {@code allowRestart} guards the single-level refutation re-scan.
   */
  private Emission matchText(
    GenerationState currentState,
    String piece,
    int tokenCount,
    boolean allowRestart
  ) {
    if (
      currentState != GenerationState.ANSWER &&
      alreadyOccurred(tagsByState.get(currentState))
    ) {
      // A state whose close already occurred resolves to ANSWER.
      return new Emission(GenerationState.ANSWER, piece, tokenCount);
    }

    String accumulated = pendingTokens == 0
      ? piece
      : pending.toString() + piece;

    String bestMarker = null;
    GenerationState bestTarget = null;
    boolean anyPrefix = false;

    // (a) the current state's close markers
    if (currentState != GenerationState.ANSWER) {
      for (String marker : tagsByState.get(currentState).closeTags()) {
        if (accumulated.startsWith(marker)) {
          // Longest wins: "<|call|>" and "<|call|><|start|>assistant..." can
          // both match, and the longer one suppresses more of the header.
          if (bestMarker == null || marker.length() > bestMarker.length()) {
            bestMarker = marker;
            bestTarget = GenerationState.ANSWER;
          }
        } else if (marker.startsWith(accumulated)) {
          anyPrefix = true;
        }
      }
    }

    // (b) the other states' open markers — cross-transitions when not in ANSWER
    for (TagBounds bounds : tagsByState.values()) {
      if (bounds.state() == currentState || alreadyOccurred(bounds)) {
        continue;
      }
      for (String marker : bounds.openTags()) {
        if (accumulated.startsWith(marker)) {
          if (bestMarker == null || marker.length() > bestMarker.length()) {
            bestMarker = marker;
            bestTarget = bounds.state();
          }
        } else if (marker.startsWith(accumulated)) {
          anyPrefix = true;
        }
      }
    }

    if (bestMarker != null && anyPrefix) {
      // Matched, but a LONGER candidate is still viable: "<|call|>" is complete
      // while "<|call|><|start|>assistant<|channel|>final<|message|>" may still
      // be arriving. Committing now would make the longer marker unreachable
      // forever; waiting without remembering this match would strand the state
      // machine when it never arrives. So hold the match provisionally and keep
      // buffering — it is settled on divergence or at end of stream.
      provisionalMarker = bestMarker;
      provisionalTarget = bestTarget;
      pending.append(piece);
      pendingTokens += tokenCount;
      return new Emission(currentState, "", 0);
    }

    if (bestMarker != null) {
      // Confirmed: marker text suppressed; the boundary-spanning remainder (if
      // any) is the first text of the post-flip channel; all covered tokens are
      // counted post-flip.
      if (currentState != GenerationState.ANSWER) {
        markOccurred(currentState);
      }
      String remainder = accumulated.substring(bestMarker.length());
      int tokens = pendingTokens + tokenCount;
      resetBuffer();
      return new Emission(bestTarget, remainder, tokens);
    }

    if (anyPrefix) {
      // Still a strict prefix of at least one candidate: buffer, emit nothing.
      pending.append(piece);
      pendingTokens += tokenCount;
      return new Emission(currentState, "", 0);
    }

    if (provisionalMarker != null) {
      // The longer candidate never came. Settle for the match we held: suppress
      // it, flip, and hand the rest of the accumulated text to the new channel.
      String marker = provisionalMarker;
      GenerationState target = provisionalTarget;
      if (currentState != GenerationState.ANSWER) {
        markOccurred(currentState);
      }
      String remainder = accumulated.substring(marker.length());
      int tokens = pendingTokens + tokenCount;
      resetBuffer();
      return new Emission(target, remainder, tokens);
    }

    if (pendingTokens == 0) {
      // No buffer, no match: plain content.
      return new Emission(currentState, piece, tokenCount);
    }

    // Refutation: the accumulated text diverged from every candidate. Flush the
    // buffered text to the current channel and re-scan the refuting delta alone
    // (it may itself start a marker prefix).
    String flushed = pending.toString();
    int flushedTokens = pendingTokens;
    resetBuffer();
    if (!allowRestart) {
      return new Emission(
        currentState,
        flushed + piece,
        flushedTokens + tokenCount
      );
    }
    Emission restart = matchText(currentState, piece, tokenCount, false);
    return new Emission(
      restart.state(),
      flushed + restart.emit(),
      flushedTokens + restart.emitTokens()
    );
  }

  private static boolean insideOpenSpan(String prompt, TagBounds bounds) {
    // Any opening marker may be the unclosed one.
    int lastStart = -1;
    for (String marker : bounds.openTags()) {
      lastStart = Math.max(lastStart, prompt.lastIndexOf(marker));
    }
    if (lastStart < 0) {
      return false;
    }
    int lastClose = -1;
    for (String marker : bounds.closeTags()) {
      lastClose = Math.max(lastClose, prompt.lastIndexOf(marker));
    }
    return lastStart > lastClose;
  }

  private void resetBuffer() {
    pending.setLength(0);
    pendingTokens = 0;
    provisionalMarker = null;
    provisionalTarget = null;
  }

  /** TOOLS may repeat; every other state closes for good. */
  private void markOccurred(GenerationState state) {
    occurred.put(state, state != GenerationState.TOOLS);
  }

  private boolean alreadyOccurred(TagBounds bounds) {
    if (bounds == null) {
      return true;
    }
    if (bounds.state() == GenerationState.TOOLS) {
      return false;
    }
    return Boolean.TRUE.equals(occurred.get(bounds.state()));
  }

  /**
   * Result of evaluating one delta: the resulting generation state, the text to
   * emit for this step, and how many generated tokens that text accounts for
   * (0 while buffering a marker prefix; N when buffered deltas resolve — marker
   * text itself is always suppressed). Emitted text always belongs to
   * {@code state}.
   */
  public record Emission(GenerationState state, String emit, int emitTokens) {}
}
