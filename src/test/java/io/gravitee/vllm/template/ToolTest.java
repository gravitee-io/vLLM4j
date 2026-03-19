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
package io.gravitee.vllm.template;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

class ToolTest {

  @Test
  void factory_shouldCreateFunctionTool() {
    var tool = Tool.function(
      "get_weather",
      "Get current weather",
      Map.of(
        "type",
        "object",
        "properties",
        Map.of(
          "location",
          Map.of("type", "string", "description", "City name")
        ),
        "required",
        java.util.List.of("location")
      )
    );

    assertThat(tool.type()).isEqualTo("function");
    assertThat(tool.function().name()).isEqualTo("get_weather");
    assertThat(tool.function().description()).isEqualTo("Get current weather");
    assertThat(tool.function().parameters()).containsKey("type");
    assertThat(tool.function().parameters()).containsKey("properties");
  }

  @Test
  void factory_shouldAcceptNullParameters() {
    var tool = Tool.function("no_args", "A tool with no args", null);

    assertThat(tool.function().parameters()).isNull();
  }

  @Test
  void equality_shouldWork() {
    var a = Tool.function("foo", "desc", Map.of("type", "object"));
    var b = Tool.function("foo", "desc", Map.of("type", "object"));
    assertThat(a).isEqualTo(b);
  }
}
