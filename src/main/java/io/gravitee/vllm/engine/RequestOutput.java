package io.gravitee.vllm.engine;

import java.util.List;
import java.util.Map;

/**
 * Java-side projection of {@code vllm.outputs.RequestOutput}.
 *
 * @param requestId       unique identifier of the request
 * @param outputs         list of completion candidates (usually 1 for greedy)
 * @param finished        whether all sequences in this request have finished
 * @param promptTokenIds  prompt token IDs (may be null if not available)
 * @param numCachedTokens number of prompt tokens that hit the prefix cache
 * @param metrics         timing and stats (may be null)
 * @param promptLogprobs  per-prompt-token logprob data, or {@code null} if not requested.
 *                        Each map keys token ID to its {@link LogprobEntry}.
 */
public record RequestOutput(
        String requestId,
        List<CompletionOutput> outputs,
        boolean finished,
        List<Integer> promptTokenIds,
        int numCachedTokens,
        RequestMetrics metrics,
        List<Map<Integer, LogprobEntry>> promptLogprobs) {

    /** Number of prompt tokens. */
    public int numPromptTokens() {
        return promptTokenIds != null ? promptTokenIds.size() : 0;
    }

    /** Total generated tokens across all completions. */
    public int numGeneratedTokens() {
        return outputs.stream().mapToInt(CompletionOutput::numGeneratedTokens).sum();
    }

    /**
     * Convenience constructor for cases where only basic fields are needed.
     */
    public RequestOutput(String requestId, List<CompletionOutput> outputs, boolean finished) {
        this(requestId, outputs, finished, null, 0, null, null);
    }

    /**
     * Convenience constructor without prompt logprobs (backward-compatible).
     */
    public RequestOutput(String requestId, List<CompletionOutput> outputs, boolean finished,
                         List<Integer> promptTokenIds, int numCachedTokens, RequestMetrics metrics) {
        this(requestId, outputs, finished, promptTokenIds, numCachedTokens, metrics, null);
    }
}
