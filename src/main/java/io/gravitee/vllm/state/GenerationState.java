package io.gravitee.vllm.state;

/**
 * The semantic category of generated tokens.
 *
 * <p>Used by the {@link ConversationState} FSM to classify output text
 * into distinct streams: regular answer content, reasoning/thinking
 * traces, and tool-call markup.
 *
 * @see ConversationState
 * @see TagBounds
 */
public enum GenerationState {
    /** Regular answer content visible to the user. */
    ANSWER,
    /** Reasoning/thinking trace (e.g. {@code <think>...</think>}). */
    REASONING,
    /** Tool-call markup (e.g. {@code <tool_call>...</tool_call>}). */
    TOOLS
}
