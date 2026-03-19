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

import java.util.Map;

/**
 * OpenAI-compatible tool definition for chat template rendering.
 *
 * <p>When passed to {@link ChatTemplate#render}, tools are serialized
 * to Python dicts matching the OpenAI format and provided as a
 * {@code tools} variable to the Jinja2 template. The model's template
 * handles formatting (Hermes-style, Llama-style, etc.).
 *
 * @param type     always {@code "function"}
 * @param function the function definition
 */
public record Tool(String type, ToolFunction function) {
  /**
   * Convenience factory for a function tool.
   *
   * @param name        function name
   * @param description what the function does
   * @param parameters  JSON Schema object describing the function's parameters
   */
  public static Tool function(
    String name,
    String description,
    Map<String, Object> parameters
  ) {
    return new Tool(
      "function",
      new ToolFunction(name, description, parameters)
    );
  }
}
