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

import java.util.List;

/**
 * A generation request to be submitted to a {@link VllmEngine}.
 *
 * <p>Bundles a unique request identifier, the text prompt, the
 * {@link SamplingParams} that control how generation proceeds, optional
 * {@link MultiModalData} for vision/audio models, an optional priority,
 * and an optional {@link LoraRequest} for LoRA adapter selection.
 *
 * <p>The caller retains ownership of the {@link SamplingParams} — closing
 * this request does <em>not</em> free the sampling params.
 *
 * <h2>Usage — text only</h2>
 * <pre>{@code
 * try (var sp = new SamplingParams().temperature(0.0).maxTokens(32)) {
 *     var request = new VllmRequest("req-1", "The capital of France is", sp);
 *     engine.addRequest(request);
 * }
 * }</pre>
 *
 * <h2>Usage — with LoRA adapter</h2>
 * <pre>{@code
 * var lora = new LoraRequest("sql-lora", 1, "gauravprasadgp/Qwen3-0.6B_nlp_to_sql");
 * var request = new VllmRequest("req-1", prompt, sp, lora);
 * engine.addRequest(request);  // auto-downloads adapter if needed
 * }</pre>
 *
 * @param requestId      unique identifier for this request
 * @param prompt         the text prompt to generate from
 * @param samplingParams the sampling parameters controlling generation
 * @param multiModalData optional multimodal data (images, audio), or {@code null}
 * @param priority       scheduling priority (lower = higher priority), default 0
 * @param loraRequest    optional LoRA adapter to apply, or {@code null} for base model
 */
public record VllmRequest(
  String requestId,
  String prompt,
  SamplingParams samplingParams,
  MultiModalData multiModalData,
  int priority,
  LoraRequest loraRequest,
  List<Integer> promptTokenIds
) {
  /**
   * Compact constructor — validates required fields.
   */
  public VllmRequest {
    if (requestId == null || requestId.isBlank()) {
      throw new IllegalArgumentException("requestId must not be null or blank");
    }
    if (prompt == null) {
      throw new IllegalArgumentException("prompt must not be null");
    }
    if (samplingParams == null) {
      throw new IllegalArgumentException("samplingParams must not be null");
    }
  }

  /** Full form without pre-tokenized input. */
  public VllmRequest(
    String requestId,
    String prompt,
    SamplingParams samplingParams,
    MultiModalData multiModalData,
    int priority,
    LoraRequest loraRequest
  ) {
    this(
      requestId,
      prompt,
      samplingParams,
      multiModalData,
      priority,
      loraRequest,
      null
    );
  }

  /**
   * Text-only constructor (backward-compatible, default priority 0, no LoRA).
   */
  public VllmRequest(
    String requestId,
    String prompt,
    SamplingParams samplingParams
  ) {
    this(requestId, prompt, samplingParams, null, 0, null, null);
  }

  /**
   * Constructor with multimodal data (default priority 0, no LoRA).
   */
  public VllmRequest(
    String requestId,
    String prompt,
    SamplingParams samplingParams,
    MultiModalData multiModalData
  ) {
    this(requestId, prompt, samplingParams, multiModalData, 0, null, null);
  }

  /**
   * Constructor with LoRA adapter (no multimodal, default priority 0).
   */
  public VllmRequest(
    String requestId,
    String prompt,
    SamplingParams samplingParams,
    LoraRequest loraRequest
  ) {
    this(requestId, prompt, samplingParams, null, 0, loraRequest, null);
  }

  /**
   * Submits pre-tokenized input instead of letting vLLM tokenize {@code prompt}.
   *
   * <p>The point is control over special tokens. vLLM tokenizes a text prompt with
   * the HuggingFace tokenizer's defaults, which match added tokens anywhere in the
   * string — so {@code <|channel|>} appearing inside a user message, a tool
   * argument or a tool result is promoted to the real control token (id 200005 for
   * gpt-oss) rather than staying data. That silently rewrites content, and lets
   * untrusted tool output inject protocol structure into the conversation. Encoding
   * each segment separately — template markup with specials parsed, interpolated
   * data with {@link VllmEngine#encode(String, boolean)} and
   * {@code parseSpecialTokens = false} — and submitting the resulting ids removes
   * the ambiguity entirely.
   *
   * <p>{@code prompt} is still carried: it remains the text used for logging and
   * for seeding the generation state from the rendered conversation.
   *
   * @param requestId      unique request identifier
   * @param prompt         the rendered prompt these ids were built from
   * @param promptTokenIds the exact tokens to feed the model
   * @param samplingParams sampling configuration
   */
  public static VllmRequest ofTokens(
    String requestId,
    String prompt,
    List<Integer> promptTokenIds,
    SamplingParams samplingParams
  ) {
    return new VllmRequest(
      requestId,
      prompt,
      samplingParams,
      null,
      0,
      null,
      promptTokenIds
    );
  }

  /** Returns {@code true} if this request carries pre-tokenized input. */
  public boolean hasPromptTokenIds() {
    return promptTokenIds != null && !promptTokenIds.isEmpty();
  }

  /** Returns {@code true} if this request includes multimodal data. */
  public boolean isMultiModal() {
    return multiModalData != null && multiModalData.hasData();
  }

  /** Returns {@code true} if this request has a non-default priority. */
  public boolean hasPriority() {
    return priority != 0;
  }

  /** Returns {@code true} if this request uses a LoRA adapter. */
  public boolean hasLora() {
    return loraRequest != null;
  }
}
