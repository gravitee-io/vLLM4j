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
package io.gravitee.vllm.iterator;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.vllm.engine.GuidedDecodingParams;
import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for {@link GuidedDecodingParams} end to end: each guide is
 * handed to vLLM's xgrammar backend and the generated text must satisfy it.
 *
 * <p>Uses Qwen/Qwen3-0.6B via {@link SharedEngine}.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class GuidedDecodingTest {

  private static VllmEngine engine;
  private static ChatTemplate chatTemplate;

  @BeforeAll
  static void initEngine() {
    engine = SharedEngine.baseEngine();
    chatTemplate = SharedEngine.chatTemplate();
  }

  @AfterAll
  static void closeEngine() {
    SharedEngine.close();
  }

  private static String prompt(String userMessage) {
    return chatTemplate.render(
      List.of(ChatMessage.user(userMessage + " /no_think")),
      true
    );
  }

  private static String generate(
    String id,
    String userMessage,
    GuidedDecodingParams guide
  ) {
    try (
      var sp = new SamplingParams(engine.arena())
        .temperature(0.0)
        .maxTokens(128)
        .guidedDecoding(guide)
    ) {
      var output = engine.generate(
        new VllmRequest(id, prompt(userMessage), sp)
      );
      assertThat(output.finished()).isTrue();
      String text = output.outputs().getFirst().text();
      System.out.println("[" + id + "] " + text);
      return text;
    }
  }

  @Test
  void json_schema_guide_produces_matching_object() {
    String text = generate(
      "guided-json",
      "Give the capital of France and its population in millions as JSON.",
      GuidedDecodingParams.json(
        """
        {
          "type": "object",
          "properties": {
            "city": {"type": "string", "enum": ["Paris", "Lyon"]},
            "population_millions": {"type": "integer"}
          },
          "required": ["city", "population_millions"],
          "additionalProperties": false
        }
        """
      )
    );

    assertThat(text.strip()).matches(
      "(?s)\\{\\s*\"city\"\\s*:\\s*\"(Paris|Lyon)\"\\s*,\\s*\"population_millions\"\\s*:\\s*-?\\d+\\s*}"
    );
  }

  @Test
  void regex_guide_produces_matching_text() {
    String text = generate(
      "guided-regex",
      "Invent a product code.",
      GuidedDecodingParams.regex("[A-Z]{2}-\\d{4}")
    );

    assertThat(text).matches("[A-Z]{2}-\\d{4}");
  }

  @Test
  void choice_guide_produces_one_of_the_choices() {
    String text = generate(
      "guided-choice",
      "Classify the sentiment of: I love this library.",
      GuidedDecodingParams.choice(List.of("positive", "negative", "neutral"))
    );

    assertThat(text).isIn("positive", "negative", "neutral");
  }

  @Test
  void grammar_guide_produces_text_in_the_language() {
    String text = generate(
      "guided-grammar",
      "Is Paris the capital of France? Answer yes or no.",
      GuidedDecodingParams.grammar("root ::= \"yes\" | \"no\"")
    );

    assertThat(text).isIn("yes", "no");
  }
}
