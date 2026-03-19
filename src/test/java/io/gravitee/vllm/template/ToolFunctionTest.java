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

class ToolFunctionTest {

  @Test
  void shouldStoreAllFields() {
    var params = Map.<String, Object>of("type", "object");
    var fn = new ToolFunction("search", "Search the web", params);

    assertThat(fn.name()).isEqualTo("search");
    assertThat(fn.description()).isEqualTo("Search the web");
    assertThat(fn.parameters()).isEqualTo(params);
  }

  @Test
  void shouldAcceptNullParameters() {
    var fn = new ToolFunction("noop", "Does nothing", null);

    assertThat(fn.parameters()).isNull();
  }

  @Test
  void equality_shouldWork() {
    var a = new ToolFunction("fn", "desc", Map.of("type", "object"));
    var b = new ToolFunction("fn", "desc", Map.of("type", "object"));
    assertThat(a).isEqualTo(b);
  }
}
