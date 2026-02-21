package io.gravitee.vllm.iterator;

import io.gravitee.vllm.engine.LogprobEntry;
import io.gravitee.vllm.state.GenerationState;

import java.util.List;
import java.util.Map;

/**
 * A single output from one generation step, tagged with its request ID
 * and semantic classification.
 *
 * @param requestId    the request that produced this output
 * @param text         the cumulative generated text so far for this completion
 * @param delta        the new text fragment since the last output for this request
 * @param finished     whether generation is complete for this request
 * @param finishReason why generation stopped (e.g. {@code "stop"}, {@code "length"}),
 *                     or {@code null} if not finished
 * @param state        the semantic category of this delta ({@code ANSWER},
 *                     {@code REASONING}, or {@code TOOLS}), or {@code null}
 *                     if classification is not configured
 * @param tokenIds     all generated token IDs so far (cumulative), may be empty
 * @param logprobs     per-token logprob data (one map per generated token position),
 *                     or {@code null} if logprobs were not requested
 */
public record VllmOutput(
        String requestId,
        String text,
        String delta,
        boolean finished,
        String finishReason,
        GenerationState state,
        List<Integer> tokenIds,
        List<Map<Integer, LogprobEntry>> logprobs) {

    /**
     * Convenience constructor without delta, state classification, token IDs, or logprobs.
     */
    public VllmOutput(String requestId, String text, boolean finished, String finishReason) {
        this(requestId, text, "", finished, finishReason, null, List.of(), null);
    }

    /**
     * Convenience constructor without token IDs or logprobs (backward-compatible).
     */
    public VllmOutput(String requestId, String text, String delta,
                      boolean finished, String finishReason, GenerationState state) {
        this(requestId, text, delta, finished, finishReason, state, List.of(), null);
    }

    /**
     * Convenience constructor without logprobs (backward-compatible).
     */
    public VllmOutput(String requestId, String text, String delta,
                      boolean finished, String finishReason, GenerationState state,
                      List<Integer> tokenIds) {
        this(requestId, text, delta, finished, finishReason, state, tokenIds, null);
    }
}
