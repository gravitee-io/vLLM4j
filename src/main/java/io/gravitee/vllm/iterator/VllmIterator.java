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

import io.gravitee.vllm.engine.CompletionOutput;
import io.gravitee.vllm.engine.RequestOutput;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.state.ConversationState;
import io.gravitee.vllm.state.GenerationState;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Continuous-batching iterator for vLLM generation with optional
 * per-request token classification (reasoning / tool-call detection).
 *
 * <p>Each request registered via {@link #addRequest} gets its own
 * {@link SequenceState} with an optional {@link ConversationState}
 * (for tag-based classification and token counting). Requests are submitted
 * with {@code RequestOutputKind.DELTA} so each step carries only the new
 * fragment; the cumulative text is accumulated on this side. This allows
 * multiple concurrent requests to be classified independently.
 *
 * <h2>With classification</h2>
 * <pre>{@code
 * var state = new ConversationState()
 *     .reasoning("<think>", "</think>");
 * var iter = new VllmIterator(engine);
 * iter.addRequest(request, state);
 * iter.stream().forEach(out ->
 *     System.out.printf("[%s] %s", out.state(), out.delta()));
 * System.out.println("Reasoning tokens: " + iter.conversationState("req-1").reasoningTokens());
 * }</pre>
 *
 * <h2>Without classification (simple)</h2>
 * <pre>{@code
 * var iter = new VllmIterator(engine);
 * iter.addRequest(request);
 * iter.stream().forEach(out -> System.out.print(out.delta()));
 * }</pre>
 */
public final class VllmIterator
  implements Iterator<VllmOutput>, Iterable<VllmOutput> {

  /**
   * Per-request tracking state: previous text length and optional classification.
   */
  private static final class SequenceState {

    /**
     * Cumulative text, accumulated here rather than re-read from Python each
     * step — appending is amortised O(1), re-marshalling is O(n) per token.
     */
    final StringBuilder text = new StringBuilder();

    final ConversationState conversationState;

    /**
     * The rendered prompt, kept to seed the initial generation state: a chat
     * template may end inside a span it opened itself (e.g. with
     * {@code <think>}), which the model never re-emits.
     */
    final String prompt;

    SequenceState(ConversationState conversationState, String prompt) {
      this.conversationState = conversationState;
      this.prompt = prompt;
    }
  }

  private final VllmEngine engine;

  /** Per-request state, keyed by request ID. */
  private final Map<String, SequenceState> sequences = new HashMap<>();

  /** Buffered outputs from the last step(), consumed one at a time by next(). */
  private final List<VllmOutput> buffer = new ArrayList<>();
  private int bufferIndex = 0;

  private volatile boolean stopped = false;

  /**
   * Creates a new iterator.
   */
  public VllmIterator(VllmEngine engine) {
    this.engine = engine;
  }

  /**
   * Submits a request to the engine without token classification.
   *
   * @param request the request to submit
   * @return this iterator for chaining
   */
  public VllmIterator addRequest(VllmRequest request) {
    return addRequest(request, null);
  }

  /**
   * Submits a request to the engine with optional token classification.
   *
   * <p>Each request gets its own {@link ConversationState}, so multiple
   * concurrent requests can be classified independently with separate
   * counters and FSM state.
   *
   * @param request           the request to submit
   * @param conversationState optional conversation state for this request's
   *                          token classification. May be {@code null}.
   * @return this iterator for chaining
   */
  public VllmIterator addRequest(
    VllmRequest request,
    ConversationState conversationState
  ) {
    // The streaming path tracks one text buffer and one ConversationState per
    // request id, and the packed step reads only the first completion — with
    // n > 1 the candidates would be interleaved through both. Refuse rather
    // than silently return one candidate of n.
    if (request.samplingParams().n() > 1) {
      throw new UnsupportedOperationException(
        "VllmIterator does not support SamplingParams.n > 1; use one request per candidate"
      );
    }
    sequences.put(
      request.requestId(),
      new SequenceState(conversationState, request.prompt())
    );
    // Ask vLLM for deltas: see VllmEngine.useDeltaOutput — without this every
    // step ships the whole response so far across the FFI boundary.
    engine.useDeltaOutput(request.samplingParams().get());
    engine.addRequest(request);
    return this;
  }

  /**
   * Aborts a request mid-generation.
   *
   * @param requestId the request to abort
   * @return this iterator for chaining
   */
  public VllmIterator abortRequest(String requestId) {
    engine.abortRequest(requestId);
    return this;
  }

  /**
   * Stops the iterator.
   */
  public void stop() {
    stopped = true;
  }

  /**
   * Returns the conversation state for a given request, or {@code null}
   * if classification was not configured for that request.
   *
   * @param requestId the request ID
   * @return the conversation state, or {@code null}
   */
  public ConversationState conversationState(String requestId) {
    SequenceState seq = sequences.get(requestId);
    return seq != null ? seq.conversationState : null;
  }

  @Override
  public boolean hasNext() {
    if (stopped) return false;

    if (bufferIndex < buffer.size()) {
      return true;
    }

    while (!stopped && engine.hasUnfinishedRequests()) {
      buffer.clear();
      bufferIndex = 0;

      // One FFI crossing for the whole step: still-generating sequences come
      // back as flat primitives, and only those that finished are mapped in
      // full (once per request, so its cost does not matter).
      var batch = engine.stepPacked();

      for (var fast : batch.streaming()) {
        SequenceState seq = sequences.computeIfAbsent(fast.requestId(), k ->
          new SequenceState(null, null)
        );
        if (
          seq.conversationState != null &&
          fast.promptTokens() > 0 &&
          seq.conversationState.inputTokens() == 0
        ) {
          seq.conversationState.initialize(fast.promptTokens(), seq.prompt);
        }

        // Counted even when the delta is empty: a step is a generated token
        // whether or not it produced printable text. Tokens that complete a
        // multi-byte character, or that the detokenizer holds back, decode to
        // "" — skipping them under-reports completion_tokens (and made the
        // throughput figure divide real time by an undercount).
        String delta = fast.delta() != null ? fast.delta() : "";
        GenerationState state = null;
        if (seq.conversationState != null) {
          // The FSM decides what is emitted, not just how it is labelled: tag
          // markers are syntax and never reach the client, and a delta holding
          // only part of a marker emits nothing until the marker is confirmed
          // or refuted.
          var emission = seq.conversationState.evaluate(delta, 1);
          state = emission.state();
          delta = emission.emit();
        }
        seq.text.append(delta);

        buffer.add(
          new VllmOutput(
            fast.requestId(),
            // Deliberately not the cumulative text: materialising it here would
            // copy the whole response on every token, which is quadratic in the
            // response length — the exact cost that asking vLLM for deltas was
            // meant to remove. It is populated on the final output; a consumer
            // that wants it mid-stream joins the deltas, as the tests do.
            "",
            delta,
            false,
            null,
            state,
            List.of(),
            null
          )
        );
      }

      for (RequestOutput reqOut : batch.finished()) {
        String requestId = reqOut.requestId();
        SequenceState seq = sequences.computeIfAbsent(requestId, k ->
          new SequenceState(null, null)
        );
        // Unknown request — create a bare tracker

        // Initialize conversation state with prompt token count on first output
        if (
          seq.conversationState != null &&
          reqOut.numPromptTokens() > 0 &&
          seq.conversationState.inputTokens() == 0
        ) {
          seq.conversationState.initialize(
            reqOut.numPromptTokens(),
            seq.prompt
          );
        }

        for (CompletionOutput comp : reqOut.outputs()) {
          String finishReasonStr = comp.finishReason() != null
            ? comp.finishReason().label()
            : null;

          // With RequestOutputKind.DELTA the engine hands us only the new
          // fragment, so the cumulative text is built up on this side.
          String delta = comp.text() != null ? comp.text() : "";

          // Classify through FSM if configured. Counted even for an empty
          // delta — see the streaming branch above.
          GenerationState state = null;
          if (seq.conversationState != null) {
            var emission = seq.conversationState.evaluate(delta, 1);
            state = emission.state();
            delta = emission.emit();
            if (comp.finished()) {
              // Generation ended: anything still buffered behind an
              // unconfirmed marker belongs to the current channel and would
              // otherwise be dropped along with its tokens.
              var flushed = seq.conversationState.flush();
              state = flushed.emitTokens() > 0 ? flushed.state() : state;
              delta = delta + flushed.emit();
            }
          }
          seq.text.append(delta);
          String fullText = seq.text.toString();

          // Set finish reason on conversation state
          if (
            comp.finished() &&
            seq.conversationState != null &&
            comp.finishReason() != null
          ) {
            seq.conversationState.setFinishReason(comp.finishReason());
          }

          buffer.add(
            new VllmOutput(
              requestId,
              fullText,
              delta,
              comp.finished(),
              finishReasonStr,
              state,
              comp.finished() && comp.tokenIds() != null
                ? comp.tokenIds()
                : List.of(),
              comp.finished() ? comp.logprobs() : null
            )
          );
        }
      }

      if (!buffer.isEmpty()) {
        return true;
      }
    }

    return false;
  }

  @Override
  public VllmOutput next() {
    if (!hasNext()) throw new NoSuchElementException("No more outputs");
    return buffer.get(bufferIndex++);
  }

  /**
   * Returns a sequential {@link Stream} over the generated outputs.
   */
  public Stream<VllmOutput> stream() {
    Spliterator<VllmOutput> spliterator = Spliterators.spliteratorUnknownSize(
      this,
      Spliterator.ORDERED | Spliterator.NONNULL
    );
    return StreamSupport.stream(spliterator, false);
  }

  @Override
  public Iterator<VllmOutput> iterator() {
    return this;
  }
}
