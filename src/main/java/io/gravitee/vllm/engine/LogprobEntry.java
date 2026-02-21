package io.gravitee.vllm.engine;

/**
 * A single logprob entry for one token at one position.
 *
 * <p>vLLM returns per-position logprob data as a dict mapping token IDs to
 * {@code Logprob(logprob=float, rank=int, decoded_token=str)}. This record
 * captures one such entry.
 *
 * @param tokenId      the token ID
 * @param logprob      the log probability (natural log)
 * @param rank         the rank of this token among all candidates (1-based)
 * @param decodedToken the decoded text of this token
 */
public record LogprobEntry(
        int tokenId,
        double logprob,
        int rank,
        String decodedToken) {
}
