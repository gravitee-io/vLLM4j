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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DecisionRequestTest {

  private static Map<String, String> criteria(String... keysAndDescriptions) {
    Map<String, String> map = new LinkedHashMap<>();
    for (int i = 0; i < keysAndDescriptions.length; i += 2) {
      map.put(keysAndDescriptions[i], keysAndDescriptions[i + 1]);
    }
    return map;
  }

  @Test
  @SuppressWarnings("unchecked")
  void request_maps_to_the_decisions_body_in_question_order() {
    var request = DecisionRequest.builder(Map.of("request", "charged twice"))
      .choice(
        "intent",
        "What does the user want?",
        criteria("billing", "Payment issue", "account", "Login issue")
      )
      .noul("upset", "Is the user upset?")
      .score("urgency", "How urgent?", List.of("Low", "Medium", "High"))
      .options(new DecisionOptions(250.0, "exact", true))
      .build();

    Map<String, Object> body = request.toMap();

    assertThat(body).containsOnlyKeys("state", "questions", "options");
    assertThat(body.get("state")).isEqualTo(Map.of("request", "charged twice"));
    var questions = (Map<String, Object>) body.get("questions");
    assertThat(questions).containsKeys("intent", "upset", "urgency");
    assertThat(List.copyOf(questions.keySet())).containsExactly(
      "intent",
      "upset",
      "urgency"
    );
    assertThat(questions.get("intent")).isEqualTo(
      Map.of(
        "type",
        "choice",
        "instructions",
        "What does the user want?",
        "criteria",
        criteria("billing", "Payment issue", "account", "Login issue")
      )
    );
    assertThat(questions.get("upset")).isEqualTo(
      Map.of("type", "noul", "instructions", "Is the user upset?")
    );
    assertThat(questions.get("urgency")).isEqualTo(
      Map.of(
        "type",
        "score",
        "instructions",
        "How urgent?",
        "criteria",
        List.of("Low", "Medium", "High")
      )
    );
    assertThat(body.get("options")).isEqualTo(
      Map.of("deadline_ms", 250.0, "profile", "exact", "return_meta", true)
    );
  }

  @Test
  void default_options_are_left_out() {
    var body = DecisionRequest.builder("text")
      .noul("q", "Is it?")
      .options(new DecisionOptions(null, null, false))
      .build()
      .toMap();
    assertThat(body).containsOnlyKeys("state", "questions");
  }

  @Test
  void choice_option_descriptions_may_be_null() {
    var question = new Question.Choice("Pick", criteria("a", null, "b", null));
    assertThat(question.criteria()).containsEntry("a", null);
  }

  @Test
  void a_request_needs_a_question() {
    assertThatThrownBy(() -> DecisionRequest.builder("text").build())
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("question");
  }

  @Test
  void a_request_needs_a_state() {
    assertThatThrownBy(() ->
      DecisionRequest.builder(null).noul("q", "Is it?").build()
    )
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("state");
  }

  @Test
  void question_names_are_unique() {
    var builder = DecisionRequest.builder("text").noul("q", "Is it?");
    assertThatThrownBy(() -> builder.noul("q", "Is it really?"))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("duplicate");
  }

  @Test
  void questions_check_their_option_counts() {
    assertThatThrownBy(() -> new Question.Choice("Pick", criteria("a", "A")))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("2 to 255");
    assertThatThrownBy(() -> new Question.Score("Rate", List.of("only")))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("2 to 10");
    assertThatThrownBy(() -> new Question.Noul(" "))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("instructions");
  }

  @Test
  void deadline_must_be_positive() {
    assertThatThrownBy(() -> new DecisionOptions(0.0, null, false))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("deadlineMs");
  }
}
