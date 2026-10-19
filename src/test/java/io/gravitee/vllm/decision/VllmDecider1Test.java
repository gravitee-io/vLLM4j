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
import static org.assertj.core.api.Assertions.within;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Runs Decision 1.0 ({@code vllm-sr/Decision-1.0-Kai-0.6B}) in-process. */
@Tag("decision-integration")
class VllmDecider1Test {

  @Test
  void answers_every_question_type() {
    try (
      var decider = VllmDecider.builder()
        .model("vllm-sr/Decision-1.0-Kai-0.6B")
        .device("cpu")
        .build()
    ) {
      Map<String, String> intents = new LinkedHashMap<>();
      intents.put("billing", "Payment or invoice issue");
      intents.put("account", "Login or access issue");
      var response = decider.decide(
        DecisionRequest.builder(
          Map.of(
            "request",
            "I was charged twice for my subscription this month."
          )
        )
          .choice("intent", "What does the user want?", intents)
          .noul("upset", "Is the user upset?")
          .score(
            "urgency",
            "How urgent is it?",
            List.of("Not urgent", "Somewhat urgent", "Urgent")
          )
          .build()
      );

      assertThat(response.model()).isEqualTo("Decision-1.0-Kai-0.6B");
      var intent = (Answer.Choice) response.answer("intent");
      assertThat(intent.choice()).isEqualTo("billing");
      assertThat(List.copyOf(intent.probabilities().keySet())).containsExactly(
        "billing",
        "account"
      );
      assertThat(
        intent
          .probabilities()
          .values()
          .stream()
          .mapToDouble(d -> d)
          .sum()
      ).isCloseTo(1.0, within(1e-6));
      assertThat(intent.confidence()).isBetween(0.0, 1.0);
      assertThat(
        ((Answer.Noul) response.answer("upset")).probability()
      ).isBetween(0.0, 1.0);
      assertThat(((Answer.Score) response.answer("urgency")).score()).isBetween(
        0.0,
        2.0
      );
    }
  }
}
