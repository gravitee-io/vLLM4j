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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One {@code /v1/decisions} request: a state and the named questions to answer
 * about it. Answers come back under the same names, in the same order.
 *
 * <pre>{@code
 * Map<String, String> intents = new LinkedHashMap<>();   // options keep their order
 * intents.put("billing", "Payment or invoice issue");
 * intents.put("account", "Login or access issue");
 * var request = DecisionRequest.builder(Map.of("request", userMessage))
 *     .choice("intent", "What does the user want?", intents)
 *     .noul("upset", "Is the user upset?")
 *     .score("urgency", "How urgent is it?", List.of("Not urgent", "Somewhat urgent", "Urgent"))
 *     .image("data:image/png;base64,...") // Decision 3.0 only
 *     .build();
 * }</pre>
 *
 * @param state     text, or a JSON-like {@code Map}/{@code List} (typed-part models read
 *                  {@code request}/{@code user}/{@code prompt} and {@code answer}/{@code response})
 * @param questions question name → question, in answer order
 * @param images    base64 PNG, JPEG or WebP data URLs, for models that read
 *                  images (Decision 3.0); may be empty
 * @param videos    base64 MP4, WebM, QuickTime or Matroska data URLs, for models
 *                  that read videos (Decision 3.0); may be empty
 * @param options   request options, or {@code null} for the defaults
 */
public record DecisionRequest(
  Object state,
  Map<String, Question> questions,
  List<String> images,
  List<String> videos,
  DecisionOptions options
) {
  public DecisionRequest {
    if (state == null) {
      throw new IllegalArgumentException("state must not be null");
    }
    if (questions == null || questions.isEmpty()) {
      throw new IllegalArgumentException("at least one question is required");
    }
    questions = Collections.unmodifiableMap(new LinkedHashMap<>(questions));
    images = images == null ? List.of() : List.copyOf(images);
    videos = videos == null ? List.of() : List.copyOf(videos);
  }

  /** A text-only request. */
  public DecisionRequest(
    Object state,
    Map<String, Question> questions,
    DecisionOptions options
  ) {
    this(state, questions, List.of(), List.of(), options);
  }

  /** Starts a request about {@code state}. */
  public static Builder builder(Object state) {
    return new Builder(state);
  }

  /** The request as a {@code /v1/decisions} body. */
  public Map<String, Object> toMap() {
    Map<String, Object> body = new LinkedHashMap<>();
    body.put("state", state);
    Map<String, Object> qs = new LinkedHashMap<>();
    questions.forEach((name, question) -> qs.put(name, question.toMap()));
    body.put("questions", qs);
    if (!images.isEmpty()) {
      body.put("images", images);
    }
    if (!videos.isEmpty()) {
      body.put("videos", videos);
    }
    if (options != null) {
      Map<String, Object> opts = options.toMap();
      if (!opts.isEmpty()) {
        body.put("options", opts);
      }
    }
    return body;
  }

  /** Fluent builder; questions keep the order they are added in. */
  public static final class Builder {

    private final Object state;
    private final Map<String, Question> questions = new LinkedHashMap<>();
    private final List<String> images = new ArrayList<>();
    private final List<String> videos = new ArrayList<>();
    private DecisionOptions options;

    private Builder(Object state) {
      this.state = state;
    }

    public Builder question(String name, Question question) {
      if (questions.putIfAbsent(name, question) != null) {
        throw new IllegalArgumentException("duplicate question name: " + name);
      }
      return this;
    }

    public Builder choice(
      String name,
      String instructions,
      Map<String, String> criteria
    ) {
      return question(name, new Question.Choice(instructions, criteria));
    }

    public Builder noul(String name, String instructions) {
      return question(name, new Question.Noul(instructions));
    }

    public Builder score(
      String name,
      String instructions,
      List<String> levels
    ) {
      return question(name, new Question.Score(instructions, levels));
    }

    /** Adds an image, as a base64 {@code data:image/...} URL. */
    public Builder image(String dataUrl) {
      images.add(requireDataUrl(dataUrl, "image"));
      return this;
    }

    /** Adds a video, as a base64 {@code data:video/...} URL. */
    public Builder video(String dataUrl) {
      videos.add(requireDataUrl(dataUrl, "video"));
      return this;
    }

    private static String requireDataUrl(String dataUrl, String kind) {
      if (dataUrl == null || !dataUrl.startsWith("data:" + kind + "/")) {
        throw new IllegalArgumentException(
          kind + " must be a data:" + kind + "/... URL"
        );
      }
      return dataUrl;
    }

    public Builder options(DecisionOptions options) {
      this.options = options;
      return this;
    }

    public DecisionRequest build() {
      return new DecisionRequest(state, questions, images, videos, options);
    }
  }
}
