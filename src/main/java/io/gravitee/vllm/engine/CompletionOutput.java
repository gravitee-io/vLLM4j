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
import java.util.Map;

/**
 * Java-side projection of {@code vllm.outputs.CompletionOutput}.
 *
 * @param index        index of this completion within the request
 * @param text         the generated text so far
 * @param tokenIds     generated token IDs (may be empty, never null)
 * @param finishReason why generation stopped, or {@code null} if still generating
 * @param logprobs     per-token logprob data (one map per generated token position),
 *                     or {@code null} if logprobs were not requested. Each map keys
 *                     token ID to its {@link LogprobEntry}.
 */
public record CompletionOutput(
  int index,
  String text,
  List<Integer> tokenIds,
  FinishReason finishReason,
  List<Map<Integer, LogprobEntry>> logprobs
) {
  /**
   * Convenience constructor without logprobs (backward-compatible).
   */
  public CompletionOutput(
    int index,
    String text,
    List<Integer> tokenIds,
    FinishReason finishReason
  ) {
    this(index, text, tokenIds, finishReason, null);
  }

  /** Whether this sequence has finished generating. */
  public boolean finished() {
    return finishReason != null;
  }

  /** Number of generated tokens so far. */
  public int numGeneratedTokens() {
    return tokenIds != null ? tokenIds.size() : 0;
  }
}
