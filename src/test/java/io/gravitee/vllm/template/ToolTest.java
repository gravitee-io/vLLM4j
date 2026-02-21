package io.gravitee.vllm.template;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class ToolTest {

    @Test
    void factory_shouldCreateFunctionTool() {
        var tool = Tool.function("get_weather", "Get current weather", Map.of(
                "type", "object",
                "properties", Map.of(
                        "location", Map.of("type", "string", "description", "City name")
                ),
                "required", java.util.List.of("location")
        ));

        assertThat(tool.type()).isEqualTo("function");
        assertThat(tool.function().name()).isEqualTo("get_weather");
        assertThat(tool.function().description()).isEqualTo("Get current weather");
        assertThat(tool.function().parameters()).containsKey("type");
        assertThat(tool.function().parameters()).containsKey("properties");
    }

    @Test
    void factory_shouldAcceptNullParameters() {
        var tool = Tool.function("no_args", "A tool with no args", null);

        assertThat(tool.function().parameters()).isNull();
    }

    @Test
    void equality_shouldWork() {
        var a = Tool.function("foo", "desc", Map.of("type", "object"));
        var b = Tool.function("foo", "desc", Map.of("type", "object"));
        assertThat(a).isEqualTo(b);
    }
}
