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

/**
 * Represents a tool call made by the assistant.
 *
 * <p>Maps to the OpenAI-compatible tool call structure:
 * <pre>{@code
 * {
 *   "id": "call_abc123",
 *   "type": "function",
 *   "function": {
 *     "name": "get_weather",
 *     "arguments": "{\"location\": \"Paris\"}"
 *   }
 * }
 * }</pre>
 *
 * @param id       unique identifier for this tool call (e.g. {@code "call_abc123"})
 * @param type     always {@code "function"}
 * @param function the function call details
 */
public record ToolCall(String id, String type, Function function) {
  /**
   * The function name and arguments within a tool call.
   *
   * @param name      the function name to invoke
   * @param arguments the arguments as a JSON string
   */
  public record Function(String name, String arguments) {}

  /**
   * Convenience factory for a function tool call.
   *
   * @param id        unique call identifier
   * @param name      the function name
   * @param arguments the arguments as a JSON string
   */
  public static ToolCall function(String id, String name, String arguments) {
    return new ToolCall(id, "function", new Function(name, arguments));
  }
}
