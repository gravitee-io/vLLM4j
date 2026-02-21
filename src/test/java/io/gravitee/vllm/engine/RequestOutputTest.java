package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class RequestOutputTest {

    @Test
    void record_shouldHaveExpectedFields() {
        var completion = new CompletionOutput(0, "hello", List.of(1, 2), null);
        var output = new RequestOutput("req-1", List.of(completion), false);

        assertThat(output.requestId()).isEqualTo("req-1");
        assertThat(output.outputs()).hasSize(1);
        assertThat(output.finished()).isFalse();
        assertThat(output.promptTokenIds()).isNull();
        assertThat(output.numCachedTokens()).isEqualTo(0);
        assertThat(output.metrics()).isNull();
    }

    @Test
    void fullConstructor_shouldPopulateAllFields() {
        var completion = new CompletionOutput(0, "done", List.of(10, 20, 30), FinishReason.STOP);
        var metrics = new RequestMetrics(1000.0, 0.05, 3);
        var output = new RequestOutput(
                "req-2", List.of(completion), true,
                List.of(100, 200), 1, metrics);

        assertThat(output.finished()).isTrue();
        assertThat(output.promptTokenIds()).containsExactly(100, 200);
        assertThat(output.numPromptTokens()).isEqualTo(2);
        assertThat(output.numCachedTokens()).isEqualTo(1);
        assertThat(output.metrics()).isNotNull();
        assertThat(output.numGeneratedTokens()).isEqualTo(3);
    }

    @Test
    void numPromptTokens_shouldReturnZeroWhenNull() {
        var output = new RequestOutput("req-1", List.of(), true);
        assertThat(output.numPromptTokens()).isEqualTo(0);
    }

    @Test
    void equality_shouldWork() {
        var a = new RequestOutput("req-1", List.of(), true);
        var b = new RequestOutput("req-1", List.of(), true);
        assertThat(a).isEqualTo(b);
    }

    @Test
    void convenienceConstructor_shouldDefaultPromptLogprobsToNull() {
        var output = new RequestOutput("req-1", List.of(), true);
        assertThat(output.promptLogprobs()).isNull();
    }

    @Test
    void sixArgConstructor_shouldDefaultPromptLogprobsToNull() {
        var metrics = new RequestMetrics(1000.0, 0.05, 0);
        var output = new RequestOutput("req-1", List.of(), true, List.of(1, 2), 0, metrics);
        assertThat(output.promptLogprobs()).isNull();
    }

    @Test
    void fullConstructor_shouldPopulatePromptLogprobs() {
        var entry = new LogprobEntry(100, -0.3, 1, "Hello");
        var promptLogprobs = new ArrayList<Map<Integer, LogprobEntry>>();
        promptLogprobs.add(null);  // first token has no logprobs
        promptLogprobs.add(Map.of(100, entry));
        var metrics = new RequestMetrics(1000.0, 0.05, 0);
        var output = new RequestOutput("req-1", List.of(), true,
                List.of(1, 100), 0, metrics, promptLogprobs);

        assertThat(output.promptLogprobs()).hasSize(2);
        assertThat(output.promptLogprobs().get(0)).isNull();
        assertThat(output.promptLogprobs().get(1)).containsKey(100);
    }
}
