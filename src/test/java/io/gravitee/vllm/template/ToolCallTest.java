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

import java.util.List;
import org.junit.jupiter.api.Test;

class ToolCallTest {

  @Test
  void factory_shouldCreateFunctionToolCall() {
    var tc = ToolCall.function(
      "call_123",
      "get_weather",
      "{\"location\": \"Paris\"}"
    );

    assertThat(tc.id()).isEqualTo("call_123");
    assertThat(tc.type()).isEqualTo("function");
    assertThat(tc.function().name()).isEqualTo("get_weather");
    assertThat(tc.function().arguments()).isEqualTo(
      "{\"location\": \"Paris\"}"
    );
  }

  @Test
  void equality_shouldWork() {
    var a = ToolCall.function("call_1", "fn", "{}");
    var b = ToolCall.function("call_1", "fn", "{}");
    assertThat(a).isEqualTo(b);
    assertThat(a.hashCode()).isEqualTo(b.hashCode());
  }

  @Test
  void differentId_shouldNotBeEqual() {
    var a = ToolCall.function("call_1", "fn", "{}");
    var b = ToolCall.function("call_2", "fn", "{}");
    assertThat(a).isNotEqualTo(b);
  }

  @Test
  void functionRecord_shouldWork() {
    var fn = new ToolCall.Function("search", "{\"query\": \"test\"}");
    assertThat(fn.name()).isEqualTo("search");
    assertThat(fn.arguments()).isEqualTo("{\"query\": \"test\"}");
  }

  @Test
  void toString_shouldContainFields() {
    var tc = ToolCall.function("call_abc", "my_func", "{\"x\": 1}");
    String str = tc.toString();
    assertThat(str).contains("call_abc", "function", "my_func");
  }
}
