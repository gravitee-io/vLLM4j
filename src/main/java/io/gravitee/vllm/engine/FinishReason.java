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

/**
 * Why generation stopped for a given completion.
 *
 * <p>Maps to vLLM's Python-side {@code finish_reason} strings and is
 * extended with {@link #TOOL_CALL} for Java-side tool-call detection
 * via the {@link io.gravitee.vllm.state.ConversationState} FSM.
 */
public enum FinishReason {
  /** End-of-sequence or stop token/string matched. */
  STOP("stop"),

  /** Maximum token limit reached. */
  LENGTH("length"),

  /** Request was aborted. */
  ABORT("abort"),

  /** Model produced a tool call (detected by tag-boundary FSM). */
  TOOL_CALL("tool_calls"),

  /** Engine-side error during generation (V1 only). */
  ERROR("error"),

  /** Repetition detected by the engine (V1 only). */
  REPETITION("repetition");

  private final String label;

  FinishReason(String label) {
    this.label = label;
  }

  /** Returns the OpenAI-compatible label (e.g. {@code "stop"}, {@code "length"}). */
  public String label() {
    return label;
  }

  /**
   * Parses vLLM's Python-side finish_reason string.
   *
   * @param value the Python string, or {@code null}
   * @return the matching enum value, or {@code null} if input is null/empty
   */
  public static FinishReason fromVllmString(String value) {
    if (value == null || value.isEmpty()) return null;
    return switch (value) {
      case "stop" -> STOP;
      case "length" -> LENGTH;
      case "abort" -> ABORT;
      case "tool_calls" -> TOOL_CALL;
      case "error" -> ERROR;
      case "repetition" -> REPETITION;
      default -> STOP; // unknown reasons treated as stop
    };
  }
}
