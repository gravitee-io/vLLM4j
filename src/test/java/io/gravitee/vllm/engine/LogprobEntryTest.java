package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LogprobEntryTest {

    @Test
    void record_shouldHaveExpectedFields() {
        var entry = new LogprobEntry(42, -1.5, 1, "hello");

        assertThat(entry.tokenId()).isEqualTo(42);
        assertThat(entry.logprob()).isEqualTo(-1.5);
        assertThat(entry.rank()).isEqualTo(1);
        assertThat(entry.decodedToken()).isEqualTo("hello");
    }

    @Test
    void equality_shouldWork() {
        var a = new LogprobEntry(10, -0.5, 2, "world");
        var b = new LogprobEntry(10, -0.5, 2, "world");
        assertThat(a).isEqualTo(b);
        assertThat(a.hashCode()).isEqualTo(b.hashCode());
    }

    @Test
    void differentTokenId_shouldNotBeEqual() {
        var a = new LogprobEntry(10, -0.5, 2, "world");
        var b = new LogprobEntry(11, -0.5, 2, "world");
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void differentLogprob_shouldNotBeEqual() {
        var a = new LogprobEntry(10, -0.5, 2, "world");
        var b = new LogprobEntry(10, -0.3, 2, "world");
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void differentRank_shouldNotBeEqual() {
        var a = new LogprobEntry(10, -0.5, 2, "world");
        var b = new LogprobEntry(10, -0.5, 3, "world");
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void zeroLogprob_shouldBeValid() {
        var entry = new LogprobEntry(0, 0.0, 1, "");
        assertThat(entry.logprob()).isEqualTo(0.0);
        assertThat(entry.decodedToken()).isEmpty();
    }

    @Test
    void negativeInfinityLogprob_shouldBeValid() {
        var entry = new LogprobEntry(99, Double.NEGATIVE_INFINITY, 50000, "<unk>");
        assertThat(entry.logprob()).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(entry.rank()).isEqualTo(50000);
    }

    @Test
    void toString_shouldContainFields() {
        var entry = new LogprobEntry(42, -1.5, 1, "hello");
        String str = entry.toString();
        assertThat(str).contains("42", "-1.5", "1", "hello");
    }
}
