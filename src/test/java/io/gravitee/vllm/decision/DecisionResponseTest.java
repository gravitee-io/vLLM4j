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
package io.gravitee.vllm.decision;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DecisionResponseTest {

  /** A response as the runtime returns it, after Python → Java conversion. */
  private static Map<String, Object> body() {
    Map<String, Object> answers = new LinkedHashMap<>();
    answers.put(
      "intent",
      Map.of(
        "type",
        "choice",
        "choice",
        "billing",
        "probabilities",
        Map.of("billing", 0.99, "account", 0.01),
        "confidence",
        0.94
      )
    );
    answers.put("upset", Map.of("type", "noul", "noul", 0.74));
    answers.put(
      "urgency",
      Map.of(
        "type",
        "score",
        "score",
        0.74,
        "probabilities",
        Map.of("0", 0.47, "1", 0.32, "2", 0.21),
        "confidence",
        0.05,
        "legend",
        Map.of("0", "Not urgent", "1", "Somewhat urgent", "2", "Urgent")
      )
    );
    Map<String, Object> failed = new LinkedHashMap<>();
    failed.put("type", null);
    failed.put("error", "max_length_exceeded");
    failed.put("message", "the state is too long");
    answers.put("broken", failed);
    answers.put(
      "labels.pii",
      Map.of("type", "set", "selected", List.of("email"))
    );

    Map<String, Object> body = new LinkedHashMap<>();
    body.put("model", "Decision-2.0-Kai-0.6B");
    body.put("answers", answers);
    body.put("usage", Map.of("input_tokens", 253L, "output_tokens", 0L));
    body.put("meta", Map.of("device", "cpu"));
    return body;
  }

  @Test
  void maps_every_answer_type_in_order() {
    var response = DecisionResponse.fromMap(body());

    assertThat(response.model()).isEqualTo("Decision-2.0-Kai-0.6B");
    assertThat(response.inputTokens()).isEqualTo(253);
    assertThat(response.meta()).containsEntry("device", "cpu");
    assertThat(List.copyOf(response.answers().keySet())).containsExactly(
      "intent",
      "upset",
      "urgency",
      "broken",
      "labels.pii"
    );

    assertThat(response.answer("intent")).isEqualTo(
      new Answer.Choice(
        "billing",
        Map.of("billing", 0.99, "account", 0.01),
        0.94
      )
    );
    assertThat(response.answer("upset")).isEqualTo(new Answer.Noul(0.74));
    var urgency = (Answer.Score) response.answer("urgency");
    assertThat(urgency.score()).isEqualTo(0.74);
    assertThat(urgency.probabilities()).containsEntry("2", 0.21);
    assertThat(urgency.legend()).containsEntry("0", "Not urgent");
    assertThat(response.answer("broken")).isEqualTo(
      new Answer.Failed("max_length_exceeded", "the state is too long")
    );
    assertThat(response.answer("labels.pii")).isInstanceOfSatisfying(
      Answer.Other.class,
      other -> assertThat(other.type()).isEqualTo("set")
    );
  }

  @Test
  void meta_is_empty_when_not_requested() {
    var body = body();
    body.remove("meta");
    assertThat(DecisionResponse.fromMap(body).meta()).isEmpty();
  }

  @Test
  void request_errors_become_decision_exceptions() {
    var e = VllmDecider.toException(
      400,
      Map.of(
        "error",
        Map.of("code", "invalid_request", "message", "no question is valid")
      )
    );
    assertThat(e.status()).isEqualTo(400);
    assertThat(e.code()).isEqualTo("invalid_request");
    assertThat(e).hasMessage("invalid_request (400): no question is valid");
  }
}
