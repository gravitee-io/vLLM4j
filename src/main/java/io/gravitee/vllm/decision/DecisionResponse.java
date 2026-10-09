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
 * The answers of one decisions request.
 *
 * @param model       the served model id
 * @param answers     question name → answer, in request order
 * @param inputTokens tokens the request was rendered to
 * @param meta        revision, engine, device, timings…; empty unless
 *                    {@link DecisionOptions#returnMeta()} was set
 */
public record DecisionResponse(
  String model,
  Map<String, Answer> answers,
  long inputTokens,
  Map<String, Object> meta
) {
  /** The answer named {@code name}, or {@code null}. */
  public Answer answer(String name) {
    return answers.get(name);
  }

  /** Maps a {@code /v1/decisions} response body. */
  @SuppressWarnings("unchecked")
  static DecisionResponse fromMap(Map<String, Object> body) {
    Map<String, Answer> answers = new LinkedHashMap<>();
    if (body.get("answers") instanceof Map<?, ?> raw) {
      raw.forEach((name, answer) ->
        answers.put(
          String.valueOf(name),
          Answer.fromMap((Map<String, Object>) answer)
        )
      );
    }
    long inputTokens = 0;
    if (
      body.get("usage") instanceof Map<?, ?> usage &&
      usage.get("input_tokens") instanceof Number n
    ) {
      inputTokens = n.longValue();
    }
    Map<String, Object> meta = body.get("meta") instanceof Map<?, ?> m
      ? Collections.unmodifiableMap((Map<String, Object>) m)
      : Map.of();
    return new DecisionResponse(
      (String) body.get("model"),
      Collections.unmodifiableMap(answers),
      inputTokens,
      meta
    );
  }
}
