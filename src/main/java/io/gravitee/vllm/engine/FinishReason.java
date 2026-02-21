package io.gravitee.vllm.engine;

/**
 * Why generation stopped for a given completion.
 *
 * <p>Maps to vLLM's Python-side {@code finish_reason} strings and is
 * extended with {@link #TOOL_CALL} for Java-side tool-call detection
 * via the {@link io.gravitee.vllm.state.ConversationState} FSM.
 */
public enum FinishReason {

    /** End-of-sequence or stop token/string matched. */
    STOP("stop"),

    /** Maximum token limit reached. */
    LENGTH("length"),

    /** Request was aborted. */
    ABORT("abort"),

    /** Model produced a tool call (detected by tag-boundary FSM). */
    TOOL_CALL("tool_calls");

    private final String label;

    FinishReason(String label) {
        this.label = label;
    }

    /** Returns the OpenAI-compatible label (e.g. {@code "stop"}, {@code "length"}). */
    public String label() {
        return label;
    }

    /**
     * Parses vLLM's Python-side finish_reason string.
     *
     * @param value the Python string, or {@code null}
     * @return the matching enum value, or {@code null} if input is null/empty
     */
    public static FinishReason fromVllmString(String value) {
        if (value == null || value.isEmpty()) return null;
        return switch (value) {
            case "stop"       -> STOP;
            case "length"     -> LENGTH;
            case "abort"      -> ABORT;
            case "tool_calls" -> TOOL_CALL;
            default           -> STOP; // unknown reasons treated as stop
        };
    }
}
