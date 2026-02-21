package io.gravitee.vllm.state;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TagBoundsTest {

    @Test
    void shouldCreateValidTagBounds() {
        var bounds = new TagBounds(GenerationState.REASONING, "<think>", "</think>");

        assertThat(bounds.state()).isEqualTo(GenerationState.REASONING);
        assertThat(bounds.openTag()).isEqualTo("<think>");
        assertThat(bounds.closeTag()).isEqualTo("</think>");
    }

    @Test
    void shouldRejectNullState() {
        assertThatThrownBy(() -> new TagBounds(null, "<tag>", "</tag>"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("state");
    }

    @Test
    void shouldRejectNullOpenTag() {
        assertThatThrownBy(() -> new TagBounds(GenerationState.REASONING, null, "</think>"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("openTag");
    }

    @Test
    void shouldRejectEmptyOpenTag() {
        assertThatThrownBy(() -> new TagBounds(GenerationState.REASONING, "", "</think>"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("openTag");
    }

    @Test
    void shouldRejectNullCloseTag() {
        assertThatThrownBy(() -> new TagBounds(GenerationState.REASONING, "<think>", null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("closeTag");
    }

    @Test
    void shouldRejectEmptyCloseTag() {
        assertThatThrownBy(() -> new TagBounds(GenerationState.REASONING, "<think>", ""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("closeTag");
    }

    @Test
    void equality_shouldWork() {
        var a = new TagBounds(GenerationState.TOOLS, "<tool_call>", "</tool_call>");
        var b = new TagBounds(GenerationState.TOOLS, "<tool_call>", "</tool_call>");
        assertThat(a).isEqualTo(b);
    }
}
