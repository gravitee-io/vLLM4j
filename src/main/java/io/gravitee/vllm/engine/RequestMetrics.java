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
 * Timing and stats for a single request, extracted from vLLM's
 * {@code RequestStateStats} Python object (v1 engine).
 *
 * <p>All time fields are in epoch seconds (double). A value of {@code -1}
 * indicates the field was unavailable for this request/vLLM version.
 *
 * @param arrivalTime        wall-clock time when the request arrived
 * @param lastTokenTime      monotonic time when the last token was generated, or -1
 * @param firstScheduledTime monotonic time when the request was first scheduled, or -1
 * @param firstTokenTime     monotonic time when the first token was generated, or -1
 * @param timeInQueue        monotonic time when the request was queued, or -1
 * @param finishedTime       reserved — always -1 in v1 (no finished_time in RequestStateStats)
 * @param firstTokenLatency  first token latency in seconds, or -1
 * @param numGenerationTokens number of tokens generated so far
 * @param numPromptTokens    number of prompt tokens (from prompt_token_ids length)
 */
public record RequestMetrics(
  double arrivalTime,
  double lastTokenTime,
  double firstScheduledTime,
  double firstTokenTime,
  double timeInQueue,
  double finishedTime,
  double firstTokenLatency,
  int numGenerationTokens,
  int numPromptTokens
) {
  /** A sentinel for when metrics are unavailable. */
  public static final RequestMetrics EMPTY = new RequestMetrics(
    0,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    0,
    0
  );

  /**
   * Backward-compatible constructor with the original six timing fields.
   */
  public RequestMetrics(
    double arrivalTime,
    double lastTokenTime,
    double firstScheduledTime,
    double firstTokenTime,
    double timeInQueue,
    double finishedTime
  ) {
    this(
      arrivalTime,
      lastTokenTime,
      firstScheduledTime,
      firstTokenTime,
      timeInQueue,
      finishedTime,
      -1,
      0,
      0
    );
  }

  /**
   * Backward-compatible constructor with the legacy three fields.
   *
   * @deprecated Use the full constructor.
   */
  public RequestMetrics(
    double arrivalTime,
    double firstTokenLatencyArg,
    int numGenerationTokensArg
  ) {
    this(
      arrivalTime,
      -1,
      -1,
      firstTokenLatencyArg >= 0 ? arrivalTime + firstTokenLatencyArg : -1,
      -1,
      -1,
      firstTokenLatencyArg,
      numGenerationTokensArg,
      0
    );
  }

  /** Time-to-first-token in seconds, or -1 if unavailable. */
  public double ttft() {
    if (firstTokenLatency >= 0) {
      return firstTokenLatency;
    }
    if (firstTokenTime >= 0 && arrivalTime > 0) {
      return firstTokenTime - arrivalTime;
    }
    return -1;
  }

  /** Time-to-first-token in milliseconds, or -1 if unavailable. */
  public double ttftMs() {
    double latency = ttft();
    return latency >= 0 ? latency * 1000.0 : -1;
  }

  /** Total generation time in seconds (arrival → finished), or -1 if unavailable. */
  public double totalTimeSeconds() {
    if (finishedTime >= 0 && arrivalTime > 0) {
      return finishedTime - arrivalTime;
    }
    return -1;
  }

  /** Total generation time in milliseconds, or -1 if unavailable. */
  public double totalTimeMs() {
    double total = totalTimeSeconds();
    return total >= 0 ? total * 1000.0 : -1;
  }
}
