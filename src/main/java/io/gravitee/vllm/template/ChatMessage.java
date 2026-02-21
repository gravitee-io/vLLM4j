package io.gravitee.vllm.template;

import java.util.List;
import java.util.Map;

/**
 * A single message in a chat conversation.
 *
 * <p>Maps to the Python dict that vLLM's chat-template rendering expects.
 * Basic messages have {@code role} and {@code content}. Tool-use conversations
 * add optional fields:
 *
 * <ul>
 *   <li><b>Assistant with tool calls:</b> {@code tool_calls} list, {@code content} may be null</li>
 *   <li><b>Tool result:</b> {@code tool_call_id} and {@code name} identify which call this answers</li>
 * </ul>
 *
 * <p>For multimodal messages (images, audio), set {@code contentParts} to a list of
 * OpenAI-format content part maps. When present, the Jinja2 template receives
 * {@code content} as a list instead of a string, allowing VLM templates (e.g. Qwen-VL)
 * to insert vision placeholder tokens.
 *
 * @param role         e.g. {@code "system"}, {@code "user"}, {@code "assistant"}, {@code "tool"}
 * @param content      the text of the message (may be {@code null} for assistant tool-call messages)
 * @param contentParts multimodal content parts in OpenAI format, or {@code null} for text-only
 * @param toolCalls    tool calls made by the assistant (non-null only for assistant messages), or {@code null}
 * @param toolCallId   the ID of the tool call this message responds to (tool messages only), or {@code null}
 * @param name         the function name for tool result messages, or {@code null}
 */
public record ChatMessage(
        String role,
        String content,
        List<Map<String, Object>> contentParts,
        List<ToolCall> toolCalls,
        String toolCallId,
        String name) {

    /**
     * Simple two-field constructor (backward-compatible).
     */
    public ChatMessage(String role, String content) {
        this(role, content, null, null, null, null);
    }

    // ── Convenience factories ───────────────────────────────────────────

    /** Creates a user message. */
    public static ChatMessage user(String content) {
        return new ChatMessage("user", content);
    }

    /** Creates a system message. */
    public static ChatMessage system(String content) {
        return new ChatMessage("system", content);
    }

    /** Creates an assistant message (plain text, no tool calls). */
    public static ChatMessage assistant(String content) {
        return new ChatMessage("assistant", content);
    }

    /**
     * Creates a user message with multimodal content parts.
     *
     * <p>The content parts follow the OpenAI format:
     * <pre>{@code
     * [{"type": "text", "text": "Describe this image"},
     *  {"type": "image"}]
     * }</pre>
     *
     * @param textContent  the text portion (also stored in {@code content} for fallback)
     * @param contentParts the full multimodal content parts list
     */
    public static ChatMessage userWithParts(String textContent, List<Map<String, Object>> contentParts) {
        return new ChatMessage("user", textContent, contentParts, null, null, null);
    }

    /**
     * Creates an assistant message that contains tool calls.
     *
     * <p>The content may be {@code null} (model emitted no text before calling tools)
     * or may contain reasoning/text the model produced alongside the calls.
     *
     * @param content   optional text content (may be null)
     * @param toolCalls the tool calls the assistant wants to make
     */
    public static ChatMessage assistantWithToolCalls(String content, List<ToolCall> toolCalls) {
        return new ChatMessage("assistant", content, null, toolCalls, null, null);
    }

    /**
     * Creates a tool result message that answers a specific tool call.
     *
     * @param content    the tool's output (e.g. {@code "22°C, sunny"})
     * @param toolCallId the ID of the tool call this responds to
     * @param name       the function name
     */
    public static ChatMessage toolResult(String content, String toolCallId, String name) {
        return new ChatMessage("tool", content, null, null, toolCallId, name);
    }

    /** Creates a plain tool message (backward-compatible, no tool_call_id). */
    public static ChatMessage tool(String content) {
        return new ChatMessage("tool", content);
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /** Returns {@code true} if this message contains tool calls. */
    public boolean hasToolCalls() {
        return toolCalls != null && !toolCalls.isEmpty();
    }

    /** Returns {@code true} if this is a tool result message with a call ID. */
    public boolean isToolResult() {
        return toolCallId != null;
    }

    /** Returns {@code true} if this message has multimodal content parts. */
    public boolean hasContentParts() {
        return contentParts != null && !contentParts.isEmpty();
    }
}
