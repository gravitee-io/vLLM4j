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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Runs {@code vllm-sr/Decision-2.0-Kai-0.6B} in-process through {@code vllm_srun}.
 *
 * <p>Needs {@code vllm-srun} in the venv ({@code setup-venv.sh -r}).
 * Usage: {@code mvn test -P decision-integration,macosx-aarch64}
 */
@Tag("decision-integration")
class VllmDeciderTest {

  private static VllmDecider decider;

  @BeforeAll
  static void setUp() {
    decider = VllmDecider.builder()
      .model("vllm-sr/Decision-2.0-Kai-0.6B")
      .device("cpu")
      .build();
  }

  @AfterAll
  static void tearDown() {
    if (decider != null) decider.close();
  }

  private static DecisionRequest billingRequest() {
    Map<String, String> intents = new LinkedHashMap<>();
    intents.put("billing", "Payment or invoice issue");
    intents.put("account", "Login or access issue");
    return DecisionRequest.builder(
      Map.of("request", "I was charged twice for my subscription this month.")
    )
      .choice("intent", "What does the user want?", intents)
      .noul("upset", "Is the user upset?")
      .score(
        "urgency",
        "How urgent is it?",
        List.of("Not urgent", "Somewhat urgent", "Urgent")
      )
      .options(DecisionOptions.withMeta())
      .build();
  }

  @Test
  void decider_is_ready_after_build() {
    assertThat(decider.state()).isEqualTo("ready");
    assertThat(decider.reason()).isNull();
  }

  @Test
  void answers_every_question_type() {
    var response = decider.decide(billingRequest());

    assertThat(response.model()).isEqualTo("Decision-2.0-Kai-0.6B");
    assertThat(response.inputTokens()).isPositive();
    assertThat(response.meta()).containsKey("revision");

    var intent = (Answer.Choice) response.answer("intent");
    assertThat(intent.choice()).isEqualTo("billing");
    assertThat(intent.probabilities()).containsOnlyKeys("billing", "account");
    assertThat(
      intent
        .probabilities()
        .values()
        .stream()
        .mapToDouble(d -> d)
        .sum()
    ).isCloseTo(1.0, within(1e-6));

    var upset = (Answer.Noul) response.answer("upset");
    assertThat(upset.probability()).isBetween(0.0, 1.0);

    var urgency = (Answer.Score) response.answer("urgency");
    assertThat(urgency.score()).isBetween(0.0, 2.0);
    assertThat(urgency.legend()).containsEntry("2", "Urgent");
  }

  @Test
  void raw_bodies_round_trip() {
    Map<String, Object> response = decider.decide(billingRequest().toMap());
    assertThat(response).containsKeys("model", "answers", "usage");
  }

  @Test
  void an_invalid_question_fails_alone() {
    Map<String, Object> body = billingRequest().toMap();
    @SuppressWarnings("unchecked")
    var questions = (Map<String, Object>) body.get("questions");
    questions.put("bad", Map.of("type", "choice", "instructions", "Pick"));

    @SuppressWarnings("unchecked")
    var answers = (Map<String, Object>) decider.decide(body).get("answers");
    @SuppressWarnings("unchecked")
    var bad = (Map<String, Object>) answers.get("bad");
    assertThat(bad).containsEntry("error", "invalid_question");
    assertThat(answers.get("intent")).isNotNull();
  }

  @Test
  void a_request_without_valid_questions_throws() {
    Map<String, Object> body = Map.of(
      "state",
      "text",
      "questions",
      Map.of("q", Map.of("type", "nope"))
    );
    assertThatThrownBy(() -> decider.decide(body)).isInstanceOfSatisfying(
      DecisionException.class,
      e -> {
        assertThat(e.status()).isEqualTo(400);
        assertThat(e.code()).isEqualTo("invalid_request");
      }
    );
  }

  @Test
  void concurrent_requests_are_all_answered() throws Exception {
    try (var pool = Executors.newFixedThreadPool(4)) {
      List<Future<DecisionResponse>> futures = new java.util.ArrayList<>();
      for (int i = 0; i < 8; i++) {
        futures.add(pool.submit(() -> decider.decide(billingRequest())));
      }
      for (var future : futures) {
        var intent = (Answer.Choice) future.get().answer("intent");
        assertThat(intent.choice()).isEqualTo("billing");
      }
    }
  }
}
