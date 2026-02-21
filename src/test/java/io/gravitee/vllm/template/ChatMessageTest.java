package io.gravitee.vllm.template;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class ChatMessageTest {

    @Test
    void user_shouldCreateUserRole() {
        ChatMessage msg = ChatMessage.user("Hello");
        assertThat(msg.role()).isEqualTo("user");
        assertThat(msg.content()).isEqualTo("Hello");
    }

    @Test
    void system_shouldCreateSystemRole() {
        ChatMessage msg = ChatMessage.system("You are helpful");
        assertThat(msg.role()).isEqualTo("system");
        assertThat(msg.content()).isEqualTo("You are helpful");
    }

    @Test
    void assistant_shouldCreateAssistantRole() {
        ChatMessage msg = ChatMessage.assistant("Sure!");
        assertThat(msg.role()).isEqualTo("assistant");
        assertThat(msg.content()).isEqualTo("Sure!");
    }

    @Test
    void equality_shouldWork() {
        ChatMessage a = ChatMessage.user("Hello");
        ChatMessage b = ChatMessage.user("Hello");
        assertThat(a).isEqualTo(b);
        assertThat(a.hashCode()).isEqualTo(b.hashCode());
    }

    @Test
    void inequality_shouldWork() {
        ChatMessage a = ChatMessage.user("Hello");
        ChatMessage b = ChatMessage.assistant("Hello");
        assertThat(a).isNotEqualTo(b);
    }

    // ── Tool-use fields ─────────────────────────────────────────────────

    @Test
    void simpleConstructor_shouldDefaultToolFieldsToNull() {
        ChatMessage msg = ChatMessage.user("test");
        assertThat(msg.toolCalls()).isNull();
        assertThat(msg.toolCallId()).isNull();
        assertThat(msg.name()).isNull();
        assertThat(msg.hasToolCalls()).isFalse();
        assertThat(msg.isToolResult()).isFalse();
    }

    @Test
    void assistantWithToolCalls_shouldPopulateToolCalls() {
        var tc = ToolCall.function("call_1", "get_weather", "{\"city\": \"Paris\"}");
        ChatMessage msg = ChatMessage.assistantWithToolCalls(null, List.of(tc));

        assertThat(msg.role()).isEqualTo("assistant");
        assertThat(msg.content()).isNull();
        assertThat(msg.hasToolCalls()).isTrue();
        assertThat(msg.toolCalls()).hasSize(1);
        assertThat(msg.toolCalls().getFirst().function().name()).isEqualTo("get_weather");
        assertThat(msg.isToolResult()).isFalse();
    }

    @Test
    void assistantWithToolCalls_canHaveContent() {
        var tc = ToolCall.function("call_1", "search", "{}");
        ChatMessage msg = ChatMessage.assistantWithToolCalls("Let me search for that", List.of(tc));

        assertThat(msg.content()).isEqualTo("Let me search for that");
        assertThat(msg.hasToolCalls()).isTrue();
    }

    @Test
    void assistantWithToolCalls_multipleTools() {
        var tc1 = ToolCall.function("call_1", "get_weather", "{\"city\": \"Paris\"}");
        var tc2 = ToolCall.function("call_2", "get_time", "{\"timezone\": \"CET\"}");
        ChatMessage msg = ChatMessage.assistantWithToolCalls(null, List.of(tc1, tc2));

        assertThat(msg.toolCalls()).hasSize(2);
    }

    @Test
    void toolResult_shouldPopulateCallIdAndName() {
        ChatMessage msg = ChatMessage.toolResult("22°C, sunny", "call_1", "get_weather");

        assertThat(msg.role()).isEqualTo("tool");
        assertThat(msg.content()).isEqualTo("22°C, sunny");
        assertThat(msg.toolCallId()).isEqualTo("call_1");
        assertThat(msg.name()).isEqualTo("get_weather");
        assertThat(msg.isToolResult()).isTrue();
        assertThat(msg.hasToolCalls()).isFalse();
    }

    @Test
    void tool_backwardCompatible_shouldHaveNoCallId() {
        ChatMessage msg = ChatMessage.tool("some result");

        assertThat(msg.role()).isEqualTo("tool");
        assertThat(msg.content()).isEqualTo("some result");
        assertThat(msg.toolCallId()).isNull();
        assertThat(msg.name()).isNull();
        assertThat(msg.isToolResult()).isFalse();
    }

    @Test
    void fullConstructor_shouldPopulateAll() {
        var tc = ToolCall.function("call_abc", "fn", "{}");
        ChatMessage msg = new ChatMessage("assistant", "thinking...", null, List.of(tc), null, null);

        assertThat(msg.role()).isEqualTo("assistant");
        assertThat(msg.content()).isEqualTo("thinking...");
        assertThat(msg.hasToolCalls()).isTrue();
        assertThat(msg.toolCallId()).isNull();
        assertThat(msg.name()).isNull();
    }
}
