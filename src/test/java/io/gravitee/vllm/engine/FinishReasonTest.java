package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;

import static org.assertj.core.api.Assertions.assertThat;

class FinishReasonTest {

    @ParameterizedTest
    @CsvSource({
            "stop,    STOP",
            "length,  LENGTH",
            "abort,   ABORT",
            "tool_calls, TOOL_CALL"
    })
    void fromVllmString_shouldMapKnownValues(String input, FinishReason expected) {
        assertThat(FinishReason.fromVllmString(input)).isEqualTo(expected);
    }

    @ParameterizedTest
    @NullAndEmptySource
    void fromVllmString_shouldReturnNullForNullOrEmpty(String input) {
        assertThat(FinishReason.fromVllmString(input)).isNull();
    }

    @Test
    void fromVllmString_shouldDefaultToStopForUnknown() {
        assertThat(FinishReason.fromVllmString("some_unknown_reason")).isEqualTo(FinishReason.STOP);
    }

    @Test
    void label_shouldReturnOpenAICompatibleString() {
        assertThat(FinishReason.STOP.label()).isEqualTo("stop");
        assertThat(FinishReason.LENGTH.label()).isEqualTo("length");
        assertThat(FinishReason.ABORT.label()).isEqualTo("abort");
        assertThat(FinishReason.TOOL_CALL.label()).isEqualTo("tool_calls");
    }

    @Test
    void roundTrip_shouldPreserveIdentity() {
        for (FinishReason reason : FinishReason.values()) {
            assertThat(FinishReason.fromVllmString(reason.label())).isEqualTo(reason);
        }
    }
}
