package io.gravitee.vllm.template;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class ToolFunctionTest {

    @Test
    void shouldStoreAllFields() {
        var params = Map.<String, Object>of("type", "object");
        var fn = new ToolFunction("search", "Search the web", params);

        assertThat(fn.name()).isEqualTo("search");
        assertThat(fn.description()).isEqualTo("Search the web");
        assertThat(fn.parameters()).isEqualTo(params);
    }

    @Test
    void shouldAcceptNullParameters() {
        var fn = new ToolFunction("noop", "Does nothing", null);

        assertThat(fn.parameters()).isNull();
    }

    @Test
    void equality_shouldWork() {
        var a = new ToolFunction("fn", "desc", Map.of("type", "object"));
        var b = new ToolFunction("fn", "desc", Map.of("type", "object"));
        assertThat(a).isEqualTo(b);
    }
}
