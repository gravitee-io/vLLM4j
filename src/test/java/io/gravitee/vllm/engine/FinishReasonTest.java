/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.vllm.engine;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;

class FinishReasonTest {

  @ParameterizedTest
  @CsvSource(
    {
      "stop,    STOP",
      "length,  LENGTH",
      "abort,   ABORT",
      "tool_calls, TOOL_CALL",
    }
  )
  void fromVllmString_shouldMapKnownValues(
    String input,
    FinishReason expected
  ) {
    assertThat(FinishReason.fromVllmString(input)).isEqualTo(expected);
  }

  @ParameterizedTest
  @NullAndEmptySource
  void fromVllmString_shouldReturnNullForNullOrEmpty(String input) {
    assertThat(FinishReason.fromVllmString(input)).isNull();
  }

  @Test
  void fromVllmString_shouldDefaultToStopForUnknown() {
    assertThat(FinishReason.fromVllmString("some_unknown_reason")).isEqualTo(
      FinishReason.STOP
    );
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
