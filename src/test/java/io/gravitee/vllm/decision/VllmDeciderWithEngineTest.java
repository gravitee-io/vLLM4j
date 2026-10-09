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

import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A decider and a {@link VllmEngine} in one JVM: one interpreter, one torch.
 * The engine is built first, as an application serving both would.
 */
@Tag("decision-integration")
class VllmDeciderWithEngineTest {

  @Test
  void decider_and_engine_share_the_interpreter() {
    try (
      var engine = VllmEngine.builder()
        .model("Qwen/Qwen3-0.6B")
        .enforceEager(true)
        .maxModelLen(1024)
        .maxNumSeqs(2)
        .gpuMemoryUtilization(0.5)
        .build();
      var decider = VllmDecider.builder()
        .model("vllm-sr/Decision-2.0-Kai-0.6B")
        .device("cpu")
        .build()
    ) {
      Map<String, String> routes = new LinkedHashMap<>();
      routes.put("chat", "Small talk or a general question");
      routes.put("code", "A programming task");
      var response = decider.decide(
        DecisionRequest.builder(Map.of("request", "Write a Java hello world."))
          .choice("route", "Which kind of request is this?", routes)
          .build()
      );
      assertThat(((Answer.Choice) response.answer("route")).choice()).isEqualTo(
        "code"
      );

      try (
        var sp = new SamplingParams(engine.arena())
          .temperature(0.0)
          .maxTokens(8)
      ) {
        var output = engine.generate(
          new VllmRequest("after-decision", "Hello, my name is", sp)
        );
        assertThat(output.finished()).isTrue();
        assertThat(output.outputs().getFirst().text()).isNotBlank();
      }
    }
  }
}
