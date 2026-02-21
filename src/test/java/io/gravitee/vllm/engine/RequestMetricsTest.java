package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class RequestMetricsTest {

    @Test
    void ttftMs_shouldConvertSecondsToMilliseconds() {
        // arrival=1000.0, firstTokenTime=1000.05 → latency=0.05s → 50ms
        var metrics = new RequestMetrics(1000.0, -1, -1, 1000.05, -1, -1);

        assertThat(metrics.firstTokenLatency()).isCloseTo(0.05, org.assertj.core.data.Offset.offset(0.001));
        assertThat(metrics.ttftMs()).isCloseTo(50.0, org.assertj.core.data.Offset.offset(1.0));
    }

    @Test
    void ttftMs_shouldReturnMinusOneWhenUnavailable() {
        var metrics = new RequestMetrics(1000.0, -1, -1, -1, -1, -1);

        assertThat(metrics.firstTokenLatency()).isEqualTo(-1.0);
        assertThat(metrics.ttftMs()).isEqualTo(-1.0);
    }

    @Test
    void empty_shouldHaveSentinelValues() {
        assertThat(RequestMetrics.EMPTY.arrivalTime()).isEqualTo(0.0);
        assertThat(RequestMetrics.EMPTY.lastTokenTime()).isEqualTo(-1.0);
        assertThat(RequestMetrics.EMPTY.firstScheduledTime()).isEqualTo(-1.0);
        assertThat(RequestMetrics.EMPTY.firstTokenTime()).isEqualTo(-1.0);
        assertThat(RequestMetrics.EMPTY.timeInQueue()).isEqualTo(-1.0);
        assertThat(RequestMetrics.EMPTY.finishedTime()).isEqualTo(-1.0);
        assertThat(RequestMetrics.EMPTY.ttftMs()).isEqualTo(-1.0);
    }

    @Test
    void equality_shouldWork() {
        var a = new RequestMetrics(100.0, 101.0, 100.1, 100.2, 0.1, 101.5);
        var b = new RequestMetrics(100.0, 101.0, 100.1, 100.2, 0.1, 101.5);
        assertThat(a).isEqualTo(b);
    }

    @Test
    void totalTimeMs_shouldComputeCorrectly() {
        var metrics = new RequestMetrics(1000.0, -1, -1, -1, -1, 1002.5);
        assertThat(metrics.totalTimeSeconds()).isCloseTo(2.5, org.assertj.core.data.Offset.offset(0.001));
        assertThat(metrics.totalTimeMs()).isCloseTo(2500.0, org.assertj.core.data.Offset.offset(1.0));
    }

    @Test
    void totalTimeMs_shouldReturnMinusOneWhenUnfinished() {
        var metrics = new RequestMetrics(1000.0, -1, -1, -1, -1, -1);
        assertThat(metrics.totalTimeSeconds()).isEqualTo(-1.0);
        assertThat(metrics.totalTimeMs()).isEqualTo(-1.0);
    }

    @Test
    void backwardCompatibleConstructor_shouldWork() {
        // Legacy: arrivalTime=1000.0, firstTokenLatency=0.05, numGenerationTokens=10
        var metrics = new RequestMetrics(1000.0, 0.05, 10);
        assertThat(metrics.arrivalTime()).isEqualTo(1000.0);
        // firstTokenTime should be arrivalTime + firstTokenLatency = 1000.05
        assertThat(metrics.firstTokenTime()).isCloseTo(1000.05, org.assertj.core.data.Offset.offset(0.001));
        assertThat(metrics.firstTokenLatency()).isCloseTo(0.05, org.assertj.core.data.Offset.offset(0.001));
    }
}
