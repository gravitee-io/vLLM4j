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
package io.gravitee.vllm.state;

/**
 * Defines the open/close tag boundaries for a {@link GenerationState}.
 *
 * <p>When the FSM detects {@code openTag} in the generated text, it
 * transitions into the associated state. When it detects {@code closeTag},
 * it transitions back to {@link GenerationState#ANSWER}.
 *
 * @param state    the generation state this tag pair activates
 * @param openTag  the opening marker (e.g. {@code "<think>"})
 * @param closeTag the closing marker (e.g. {@code "</think>"})
 */
public record TagBounds(
  GenerationState state,
  String openTag,
  String closeTag
) {
  public TagBounds {
    if (state == null) throw new IllegalArgumentException(
      "state must not be null"
    );
    if (
      openTag == null || openTag.isEmpty()
    ) throw new IllegalArgumentException("openTag must not be empty");
    if (
      closeTag == null || closeTag.isEmpty()
    ) throw new IllegalArgumentException("closeTag must not be empty");
  }
}
