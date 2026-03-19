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

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link GuidedDecodingParams} factory methods.
 *
 * <p>The {@code toPython()} method is package-private and requires CPython,
 * so it is tested at the integration level. These tests verify the Java-side
 * construction and factory method behavior.
 */
class GuidedDecodingParamsTest {

  @Test
  void json_shouldCreateInstance() {
    String schema = """
      {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}""";
    var params = GuidedDecodingParams.json(schema);
    assertThat(params).isNotNull();
  }

  @Test
  void regex_shouldCreateInstance() {
    var params = GuidedDecodingParams.regex("[A-Z][a-z]+");
    assertThat(params).isNotNull();
  }

  @Test
  void choice_shouldCreateInstance() {
    var params = GuidedDecodingParams.choice(List.of("yes", "no", "maybe"));
    assertThat(params).isNotNull();
  }

  @Test
  void grammar_shouldCreateInstance() {
    var params = GuidedDecodingParams.grammar("root ::= 'hello' | 'world'");
    assertThat(params).isNotNull();
  }

  @Test
  void jsonObject_shouldCreateInstance() {
    var params = GuidedDecodingParams.jsonObject();
    assertThat(params).isNotNull();
  }

  @Test
  void differentFactories_shouldCreateDistinctInstances() {
    var json = GuidedDecodingParams.json("{}");
    var regex = GuidedDecodingParams.regex(".*");
    var choice = GuidedDecodingParams.choice(List.of("a"));
    var grammar = GuidedDecodingParams.grammar("root ::= 'x'");
    var jsonObject = GuidedDecodingParams.jsonObject();

    // All are distinct instances
    assertThat(json).isNotSameAs(regex);
    assertThat(regex).isNotSameAs(choice);
    assertThat(choice).isNotSameAs(grammar);
    assertThat(grammar).isNotSameAs(jsonObject);
  }
}
