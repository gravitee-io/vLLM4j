package io.gravitee.vllm.engine;

/**
 * A snapshot of the engine's current state.
 *
 * <p>Obtained via {@link VllmEngine#getStats()}.
 *
 * @param numUnfinishedRequests number of requests still being processed
 * @param model                the model name/path (e.g. {@code "Qwen/Qwen3-0.6B"})
 * @param dtype                the torch dtype string (e.g. {@code "auto"})
 * @param maxModelLen          the maximum sequence length the engine supports
 */
public record EngineStats(
        int numUnfinishedRequests,
        String model,
        String dtype,
        int maxModelLen) {
}
