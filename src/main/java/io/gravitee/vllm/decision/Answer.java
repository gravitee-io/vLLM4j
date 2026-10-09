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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The answer to one {@link Question}. A question that could not be answered
 * is a {@link Failed}; its siblings are still answered.
 */
public sealed interface Answer {
  /**
   * The chosen option.
   *
   * @param choice        the key of the most probable option
   * @param probabilities option key → probability, in option order
   * @param confidence    the model's confidence in {@code choice}
   */
  record Choice(
    String choice,
    Map<String, Double> probabilities,
    double confidence
  ) implements Answer {}

  /**
   * The probability that the condition holds.
   *
   * @param probability P(true)
   */
  record Noul(double probability) implements Answer {}

  /**
   * The expected level.
   *
   * @param score         expected zero-based level
   * @param probabilities level index (as a string) → probability
   * @param confidence    the model's confidence
   * @param legend        level index (as a string) → level description
   */
  record Score(
    double score,
    Map<String, Double> probabilities,
    double confidence,
    Map<String, String> legend
  ) implements Answer {}

  /**
   * A question the model did not answer.
   *
   * @param error   {@code invalid_question}, {@code invalid_input}, {@code max_length_exceeded},
   *                {@code scan_budget_exceeded}, {@code invalid_model_output},
   *                {@code deadline_exceeded} or {@code unavailable}
   * @param message detail, or {@code null}
   */
  record Failed(String error, String message) implements Answer {}

  /**
   * An answer of a type this binding does not map (e.g. a {@code set} label);
   * {@code fields} holds it as returned.
   */
  record Other(String type, Map<String, Object> fields) implements Answer {}

  /** Maps one entry of the response's {@code answers}. */
  static Answer fromMap(Map<String, Object> map) {
    if (map.get("error") != null) {
      return new Failed(
        String.valueOf(map.get("error")),
        (String) map.get("message")
      );
    }
    String type = (String) map.get("type");
    return switch (type == null ? "" : type) {
      case "choice" -> new Choice(
        (String) map.get("choice"),
        doubles(map.get("probabilities")),
        number(map.get("confidence"))
      );
      case "noul" -> new Noul(number(map.get("noul")));
      case "score" -> new Score(
        number(map.get("score")),
        doubles(map.get("probabilities")),
        number(map.get("confidence")),
        strings(map.get("legend"))
      );
      default -> new Other(type, Collections.unmodifiableMap(map));
    };
  }

  private static double number(Object value) {
    return value instanceof Number n ? n.doubleValue() : Double.NaN;
  }

  private static Map<String, Double> doubles(Object value) {
    Map<String, Double> out = new LinkedHashMap<>();
    if (value instanceof Map<?, ?> m) {
      m.forEach((k, v) -> out.put(String.valueOf(k), number(v)));
    }
    return Collections.unmodifiableMap(out);
  }

  private static Map<String, String> strings(Object value) {
    Map<String, String> out = new LinkedHashMap<>();
    if (value instanceof Map<?, ?> m) {
      m.forEach((k, v) -> out.put(String.valueOf(k), String.valueOf(v)));
    }
    return Collections.unmodifiableMap(out);
  }
}
