package io.gravitee.vllm.template;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class ToolCallTest {

    @Test
    void factory_shouldCreateFunctionToolCall() {
        var tc = ToolCall.function("call_123", "get_weather", "{\"location\": \"Paris\"}");

        assertThat(tc.id()).isEqualTo("call_123");
        assertThat(tc.type()).isEqualTo("function");
        assertThat(tc.function().name()).isEqualTo("get_weather");
        assertThat(tc.function().arguments()).isEqualTo("{\"location\": \"Paris\"}");
    }

    @Test
    void equality_shouldWork() {
        var a = ToolCall.function("call_1", "fn", "{}");
        var b = ToolCall.function("call_1", "fn", "{}");
        assertThat(a).isEqualTo(b);
        assertThat(a.hashCode()).isEqualTo(b.hashCode());
    }

    @Test
    void differentId_shouldNotBeEqual() {
        var a = ToolCall.function("call_1", "fn", "{}");
        var b = ToolCall.function("call_2", "fn", "{}");
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void functionRecord_shouldWork() {
        var fn = new ToolCall.Function("search", "{\"query\": \"test\"}");
        assertThat(fn.name()).isEqualTo("search");
        assertThat(fn.arguments()).isEqualTo("{\"query\": \"test\"}");
    }

    @Test
    void toString_shouldContainFields() {
        var tc = ToolCall.function("call_abc", "my_func", "{\"x\": 1}");
        String str = tc.toString();
        assertThat(str).contains("call_abc", "function", "my_func");
    }
}
