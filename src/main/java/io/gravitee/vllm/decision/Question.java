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
import java.util.List;
import java.util.Map;

/**
 * A typed question asked of a decision model, as in the System One contract.
 *
 * <ul>
 *   <li>{@link Choice} — pick one of 2–255 options</li>
 *   <li>{@link Noul} — the probability that a condition holds</li>
 *   <li>{@link Score} — the expected level on an ordered rubric of 2–10 levels</li>
 * </ul>
 */
public sealed interface Question {
  /** The question text. */
  String instructions();

  /** The question as a {@code /v1/decisions} request entry. */
  Map<String, Object> toMap();

  /**
   * Picks one option.
   *
   * @param instructions the question text
   * @param criteria     option key → description (may be {@code null}), in order
   */
  record Choice(String instructions, Map<String, String> criteria) implements
    Question {
    public Choice {
      requireInstructions(instructions);
      if (criteria == null || criteria.size() < 2 || criteria.size() > 255) {
        throw new IllegalArgumentException(
          "a choice question needs 2 to 255 options"
        );
      }
      criteria = Collections.unmodifiableMap(new LinkedHashMap<>(criteria));
    }

    @Override
    public Map<String, Object> toMap() {
      Map<String, Object> map = new LinkedHashMap<>();
      map.put("type", "choice");
      map.put("instructions", instructions);
      map.put("criteria", criteria);
      return map;
    }
  }

  /**
   * The probability that a condition holds.
   *
   * @param instructions the condition to judge
   */
  record Noul(String instructions) implements Question {
    public Noul {
      requireInstructions(instructions);
    }

    @Override
    public Map<String, Object> toMap() {
      Map<String, Object> map = new LinkedHashMap<>();
      map.put("type", "noul");
      map.put("instructions", instructions);
      return map;
    }
  }

  /**
   * The expected level on an ordered rubric.
   *
   * @param instructions the question text
   * @param levels       level descriptions, lowest first
   */
  record Score(String instructions, List<String> levels) implements Question {
    public Score {
      requireInstructions(instructions);
      if (levels == null || levels.size() < 2 || levels.size() > 10) {
        throw new IllegalArgumentException(
          "a score question needs 2 to 10 levels"
        );
      }
      levels = List.copyOf(levels);
    }

    @Override
    public Map<String, Object> toMap() {
      Map<String, Object> map = new LinkedHashMap<>();
      map.put("type", "score");
      map.put("instructions", instructions);
      map.put("criteria", levels);
      return map;
    }
  }

  private static void requireInstructions(String instructions) {
    if (instructions == null || instructions.isBlank()) {
      throw new IllegalArgumentException(
        "instructions must not be null or blank"
      );
    }
  }
}
