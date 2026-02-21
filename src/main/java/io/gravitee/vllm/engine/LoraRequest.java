package io.gravitee.vllm.engine;

/**
 * A LoRA adapter request to attach to a {@link VllmRequest}.
 *
 * <p>Mirrors Python's {@code vllm.lora.request.LoRARequest}. When attached
 * to a request, the engine applies the specified LoRA adapter weights during
 * generation.
 *
 * <p>The {@code loraPath} accepts either a <strong>local filesystem path</strong>
 * to an adapter directory or a <strong>HuggingFace repo ID</strong>
 * (e.g. {@code "gauravprasadgp/Qwen3-0.6B_nlp_to_sql"}). If a repo ID is
 * given, the engine will automatically download the adapter via
 * {@code vllm.lora.utils.get_adapter_absolute_path()} on first use.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var lora = new LoraRequest("sql-lora", 1, "gauravprasadgp/Qwen3-0.6B_nlp_to_sql");
 * var request = new VllmRequest("req-1", prompt, sp, lora);
 * engine.addRequest(request);  // auto-downloads adapter if needed
 * }</pre>
 *
 * @param loraName  human-readable name (used for caching and identification)
 * @param loraIntId globally unique integer ID (must be &ge; 1)
 * @param loraPath  local path or HuggingFace repo ID for the adapter weights
 */
public record LoraRequest(String loraName, int loraIntId, String loraPath) {

    /**
     * Compact constructor — validates required fields.
     */
    public LoraRequest {
        if (loraName == null || loraName.isBlank()) {
            throw new IllegalArgumentException("loraName must not be null or blank");
        }
        if (loraIntId < 1) {
            throw new IllegalArgumentException("loraIntId must be >= 1, got " + loraIntId);
        }
        if (loraPath == null || loraPath.isBlank()) {
            throw new IllegalArgumentException("loraPath must not be null or blank");
        }
    }
}
