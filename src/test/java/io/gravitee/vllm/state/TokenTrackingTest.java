package io.gravitee.vllm.state;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class TokenTrackingTest {

    private TokenTracking tracking;

    @BeforeEach
    void setUp() {
        tracking = new TokenTracking();
    }

    @Test
    void initialize_shouldSetPromptAndResetCounters() {
        tracking.consume(GenerationState.ANSWER, 5);
        tracking.initialize(100);

        assertThat(tracking.inputTokens()).isEqualTo(100);
        assertThat(tracking.outputTokens(GenerationState.ANSWER)).isEqualTo(0);
        assertThat(tracking.outputTokens(GenerationState.REASONING)).isEqualTo(0);
        assertThat(tracking.outputTokens(GenerationState.TOOLS)).isEqualTo(0);
    }

    @Test
    void consume_shouldIncrementAnswerCounter() {
        tracking.consume(GenerationState.ANSWER, 10);
        assertThat(tracking.outputTokens(GenerationState.ANSWER)).isEqualTo(10);
    }

    @Test
    void consume_shouldIncrementReasoningCounter() {
        tracking.consume(GenerationState.REASONING, 7);
        assertThat(tracking.outputTokens(GenerationState.REASONING)).isEqualTo(7);
    }

    @Test
    void consume_shouldIncrementToolsCounter() {
        tracking.consume(GenerationState.TOOLS, 3);
        assertThat(tracking.outputTokens(GenerationState.TOOLS)).isEqualTo(3);
    }

    @Test
    void consume_shouldAccumulate() {
        tracking.consume(GenerationState.ANSWER, 5);
        tracking.consume(GenerationState.ANSWER, 3);
        assertThat(tracking.outputTokens(GenerationState.ANSWER)).isEqualTo(8);
    }

    @Test
    void totalOutputTokens_shouldSumAllStates() {
        tracking.consume(GenerationState.ANSWER, 10);
        tracking.consume(GenerationState.REASONING, 20);
        tracking.consume(GenerationState.TOOLS, 5);

        assertThat(tracking.totalOutputTokens()).isEqualTo(35);
    }

    @Test
    void totalTokens_shouldIncludeInput() {
        tracking.initialize(50);
        tracking.consume(GenerationState.ANSWER, 10);
        tracking.consume(GenerationState.REASONING, 20);

        assertThat(tracking.totalTokens()).isEqualTo(80);
    }

    @Test
    void freshInstance_shouldHaveZeroCounts() {
        assertThat(tracking.inputTokens()).isEqualTo(0);
        assertThat(tracking.totalOutputTokens()).isEqualTo(0);
        assertThat(tracking.totalTokens()).isEqualTo(0);
    }
}
