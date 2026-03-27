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
 * Java-side projection of {@code vllm.outputs.RequestOutput}.
 *
 * @param requestId       unique identifier of the request
 * @param outputs         list of completion candidates (usually 1 for greedy)
 * @param finished        whether all sequences in this request have finished
 * @param promptTokenIds  prompt token IDs (may be null if not available)
 * @param numCachedTokens number of prompt tokens that hit the prefix cache
 * @param metrics         timing and stats (may be null)
 * @param promptLogprobs  per-prompt-token logprob data, or {@code null} if not requested.
 *                        Each map keys token ID to its {@link LogprobEntry}.
 */
public record RequestOutput(
  String requestId,
  List<CompletionOutput> outputs,
  boolean finished,
  List<Integer> promptTokenIds,
  int numCachedTokens,
  RequestMetrics metrics,
  List<Map<Integer, LogprobEntry>> promptLogprobs
) {
  /** Number of prompt tokens, from metrics. */
  public int numPromptTokens() {
    return metrics != null ? metrics.numPromptTokens() : 0;
  }

  /** Total generated tokens across all completions. */
  public int numGeneratedTokens() {
    return outputs
      .stream()
      .mapToInt(CompletionOutput::numGeneratedTokens)
      .sum();
  }

  /**
   * Convenience constructor for cases where only basic fields are needed.
   */
  public RequestOutput(
    String requestId,
    List<CompletionOutput> outputs,
    boolean finished
  ) {
    this(requestId, outputs, finished, null, 0, null, null);
  }

  /**
   * Convenience constructor without prompt logprobs (backward-compatible).
   */
  public RequestOutput(
    String requestId,
    List<CompletionOutput> outputs,
    boolean finished,
    List<Integer> promptTokenIds,
    int numCachedTokens,
    RequestMetrics metrics
  ) {
    this(
      requestId,
      outputs,
      finished,
      promptTokenIds,
      numCachedTokens,
      metrics,
      null
    );
  }
}
