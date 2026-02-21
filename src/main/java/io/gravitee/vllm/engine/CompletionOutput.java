package io.gravitee.vllm.engine;

import java.util.List;
import java.util.Map;

/**
 * Java-side projection of {@code vllm.outputs.CompletionOutput}.
 *
 * @param index        index of this completion within the request
 * @param text         the generated text so far
 * @param tokenIds     generated token IDs (may be empty, never null)
 * @param finishReason why generation stopped, or {@code null} if still generating
 * @param logprobs     per-token logprob data (one map per generated token position),
 *                     or {@code null} if logprobs were not requested. Each map keys
 *                     token ID to its {@link LogprobEntry}.
 */
public record CompletionOutput(
        int index,
        String text,
        List<Integer> tokenIds,
        FinishReason finishReason,
        List<Map<Integer, LogprobEntry>> logprobs) {

    /**
     * Convenience constructor without logprobs (backward-compatible).
     */
    public CompletionOutput(int index, String text, List<Integer> tokenIds, FinishReason finishReason) {
        this(index, text, tokenIds, finishReason, null);
    }

    /** Whether this sequence has finished generating. */
    public boolean finished() {
        return finishReason != null;
    }

    /** Number of generated tokens so far. */
    public int numGeneratedTokens() {
        return tokenIds != null ? tokenIds.size() : 0;
    }
}
