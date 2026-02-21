package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class EngineStatsTest {

    @Test
    void record_shouldHaveExpectedFields() {
        var stats = new EngineStats(5, "Qwen/Qwen3-0.6B", "auto", 40960);

        assertThat(stats.numUnfinishedRequests()).isEqualTo(5);
        assertThat(stats.model()).isEqualTo("Qwen/Qwen3-0.6B");
        assertThat(stats.dtype()).isEqualTo("auto");
        assertThat(stats.maxModelLen()).isEqualTo(40960);
    }

    @Test
    void equality_shouldWork() {
        var a = new EngineStats(0, "model", "float16", 2048);
        var b = new EngineStats(0, "model", "float16", 2048);
        assertThat(a).isEqualTo(b);
        assertThat(a.hashCode()).isEqualTo(b.hashCode());
    }

    @Test
    void differentValues_shouldNotBeEqual() {
        var a = new EngineStats(0, "model-a", "float16", 2048);
        var b = new EngineStats(0, "model-b", "float16", 2048);
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void toString_shouldContainFields() {
        var stats = new EngineStats(2, "test-model", "auto", 4096);
        String str = stats.toString();
        assertThat(str).contains("test-model", "auto", "4096");
    }
}
