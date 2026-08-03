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

import java.util.List;

/**
 * Defines the open/close tag boundaries for a {@link GenerationState}.
 *
 * <p>When the FSM detects one of {@code openTags} in the generated text, it
 * transitions into the associated state. When it detects {@code closeTag},
 * it transitions back to {@link GenerationState#ANSWER}.
 *
 * <p>A channel may be opened by more than one marker — Harmony opens its tool
 * channel as both {@code commentary} and {@code analysis}, and configuring only
 * one of them leaks the other into the answer as raw text. Mirrors llamaj.cpp's
 * {@code StateBounds}.
 *
 * @param state    the generation state these tags activate
 * @param openTags the opening markers, any of which enters the state
 * @param closeTag the closing marker (e.g. {@code "</think>"})
 */
public record TagBounds(
  GenerationState state,
  List<String> openTags,
  List<String> closeTags
) {
  public TagBounds {
    if (state == null) throw new IllegalArgumentException(
      "state must not be null"
    );
    if (
      openTags == null || openTags.isEmpty()
    ) throw new IllegalArgumentException("openTag must not be empty");
    for (String openTag : openTags) {
      if (
        openTag == null || openTag.isEmpty()
      ) throw new IllegalArgumentException("openTag must not be empty");
    }
    if (
      closeTags == null || closeTags.isEmpty()
    ) throw new IllegalArgumentException("closeTag must not be empty");
    for (String closeTag : closeTags) {
      if (
        closeTag == null || closeTag.isEmpty()
      ) throw new IllegalArgumentException("closeTag must not be empty");
    }
    openTags = List.copyOf(openTags);
    closeTags = List.copyOf(closeTags);
  }

  /** Single-marker form. */
  public TagBounds(GenerationState state, String openTag, String closeTag) {
    this(
      state,
      openTag == null ? List.of() : List.of(openTag),
      closeTag == null ? List.of() : List.of(closeTag)
    );
  }

  /** Many openings, one closing. */
  public TagBounds(
    GenerationState state,
    List<String> openTags,
    String closeTag
  ) {
    this(state, openTags, closeTag == null ? List.of() : List.of(closeTag));
  }

  /** The primary opening marker. */
  public String openTag() {
    return openTags.getFirst();
  }

  /**
   * The primary closing marker.
   *
   * <p>Kept for callers that only ever configured one, and for prompt scanning,
   * where the question is merely whether a span was closed at all.
   */
  public String closeTag() {
    return closeTags.getFirst();
  }
}
