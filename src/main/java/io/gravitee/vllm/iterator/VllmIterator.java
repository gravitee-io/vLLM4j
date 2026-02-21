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
 * (for tag-based classification and token counting). Only the previous
 * text length is tracked (not the full string) to extract the delta
 * efficiently. This allows multiple concurrent requests to be classified
 * independently.
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
public final class VllmIterator implements Iterator<VllmOutput>, Iterable<VllmOutput> {

    /**
     * Per-request tracking state: previous text length and optional classification.
     */
    private static final class SequenceState {
        int previousTextLength = 0;
        final ConversationState conversationState;

        SequenceState(ConversationState conversationState) {
            this.conversationState = conversationState;
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
    public VllmIterator addRequest(VllmRequest request, ConversationState conversationState) {
        sequences.put(request.requestId(), new SequenceState(conversationState));
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

            List<RequestOutput> stepOutputs = engine.step();
            for (RequestOutput reqOut : stepOutputs) {
                String requestId = reqOut.requestId();
                SequenceState seq = sequences.computeIfAbsent(requestId, k -> new SequenceState(null));
                // Unknown request — create a bare tracker

                // Initialize conversation state with prompt token count on first output
                if (seq.conversationState != null && reqOut.numPromptTokens() > 0
                        && seq.conversationState.inputTokens() == 0) {
                    seq.conversationState.initialize(reqOut.numPromptTokens());
                }

                for (CompletionOutput comp : reqOut.outputs()) {
                    String finishReasonStr = comp.finishReason() != null
                            ? comp.finishReason().label() : null;

                    // Compute delta using only the current text and the previous length
                    String fullText = comp.text();
                    String delta = fullText.length() > seq.previousTextLength
                            ? fullText.substring(seq.previousTextLength) : "";
                    seq.previousTextLength = fullText.length();

                    // Classify through FSM if configured
                    GenerationState state = null;
                    if (seq.conversationState != null && !delta.isEmpty()) {
                        state = seq.conversationState.evaluate(delta, 1);
                    }

                    // Set finish reason on conversation state
                    if (comp.finished() && seq.conversationState != null && comp.finishReason() != null) {
                        seq.conversationState.setFinishReason(comp.finishReason());
                    }

                    buffer.add(new VllmOutput(
                            requestId,
                            fullText,
                            delta,
                            comp.finished(),
                            finishReasonStr,
                            state,
                            comp.finished() && comp.tokenIds() != null ? comp.tokenIds() : List.of(),
                            comp.finished() ? comp.logprobs() : null));
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
                this, Spliterator.ORDERED | Spliterator.NONNULL);
        return StreamSupport.stream(spliterator, false);
    }

    @Override
    public Iterator<VllmOutput> iterator() {
        return this;
    }
}
