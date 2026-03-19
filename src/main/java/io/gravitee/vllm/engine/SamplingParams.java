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

import io.gravitee.vllm.Freeable;
import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.runtime.GIL;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Wraps a {@code vllm.sampling_params.SamplingParams} Python object.
 *
 * <p>Follows a fluent self-returning pattern (like llamaj.cpp's
 * {@code LlamaSampler}). Each setter method mutates the underlying Python
 * kwargs dict and returns {@code this}. The actual Python {@code SamplingParams}
 * object is constructed lazily on the first call to {@link #get()}.
 *
 * <p>Implements {@link AutoCloseable} and {@link Freeable} — the Python
 * reference is decremented on {@link #close()}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (var sp = new SamplingParams()
 *         .temperature(0.8)
 *         .maxTokens(128)
 *         .topP(0.9)
 *         .stop("</s>", "<|endoftext|>")) {
 *     engine.addRequest(new VllmRequest("req-1", "Hello", sp));
 * }
 * }</pre>
 *
 * <h2>Defaults</h2>
 * <p>Any parameter not explicitly set inherits vLLM's Python-side default.
 */
public final class SamplingParams implements AutoCloseable, Freeable {

  // ── Lazy singleton for the Python SamplingParams class ──────────────

  private static volatile MemorySegment samplingParamsClass;

  /**
   * Lazily imports and caches the {@code vllm.sampling_params.SamplingParams}
   * Python class. Thread-safe (double-checked locking).
   * <p>Must be called with the GIL held (or from within a GIL-protected scope).
   *
   * @param arena arena for native string allocation during the import
   */
  private static MemorySegment ensureClass(Arena arena) {
    if (samplingParamsClass == null) {
      synchronized (SamplingParams.class) {
        if (samplingParamsClass == null) {
          samplingParamsClass = PythonCall.importClass(
            arena,
            "vllm.sampling_params",
            "SamplingParams"
          );
        }
      }
    }
    return samplingParamsClass;
  }

  // ── Instance state ──────────────────────────────────────────────────

  private final Arena arena;

  /** Accumulates kwargs until the Python object is built. */
  private MemorySegment kwargs;

  /** The built Python SamplingParams object. Null until {@link #get()} is called. */
  private MemorySegment pyObject;

  private volatile boolean freed = false;

  // ── Construction ────────────────────────────────────────────────────

  /**
   * Creates a new SamplingParams builder with no parameters set.
   * All values will inherit vLLM's Python-side defaults.
   *
   * @param arena shared arena for native memory allocation (must outlive this object)
   */
  public SamplingParams(Arena arena) {
    this.arena = arena;
    try (var gil = GIL.acquire()) {
      this.kwargs = CPythonBinding.PyDict_New();
    }
  }

  // ── Fluent setters ──────────────────────────────────────────────────

  /**
   * Sets the sampling temperature.
   *
   * @param temperature 0.0 for greedy, higher values increase randomness
   * @return this
   */
  public SamplingParams temperature(double temperature) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "temperature", temperature);
    }
    return this;
  }

  /**
   * Sets the maximum number of tokens to generate.
   *
   * @param maxTokens maximum output token count
   * @return this
   */
  public SamplingParams maxTokens(int maxTokens) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "max_tokens", maxTokens);
    }
    return this;
  }

  /**
   * Sets the nucleus (top-p) sampling threshold.
   *
   * @param topP cumulative probability cutoff (0.0–1.0)
   * @return this
   */
  public SamplingParams topP(double topP) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "top_p", topP);
    }
    return this;
  }

  /**
   * Sets the top-k sampling threshold.
   *
   * @param topK number of highest-probability tokens to consider (-1 for all)
   * @return this
   */
  public SamplingParams topK(int topK) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "top_k", topK);
    }
    return this;
  }

  /**
   * Sets the repetition penalty.
   *
   * @param penalty repetition penalty factor (1.0 = no penalty)
   * @return this
   */
  public SamplingParams repetitionPenalty(double penalty) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "repetition_penalty", penalty);
    }
    return this;
  }

  /**
   * Sets the frequency penalty.
   *
   * @param penalty frequency penalty (0.0 = no penalty)
   * @return this
   */
  public SamplingParams frequencyPenalty(double penalty) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "frequency_penalty", penalty);
    }
    return this;
  }

  /**
   * Sets the presence penalty.
   *
   * @param penalty presence penalty (0.0 = no penalty)
   * @return this
   */
  public SamplingParams presencePenalty(double penalty) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "presence_penalty", penalty);
    }
    return this;
  }

  /**
   * Sets the minimum probability threshold for min-p sampling.
   *
   * @param minP minimum probability cutoff (0.0–1.0)
   * @return this
   */
  public SamplingParams minP(double minP) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictFloat(arena, kwargs, "min_p", minP);
    }
    return this;
  }

  /**
   * Sets the random seed for sampling.
   *
   * @param seed random seed (null for non-deterministic)
   * @return this
   */
  public SamplingParams seed(Long seed) {
    checkNotBuilt();
    if (seed != null) {
      try (var gil = GIL.acquire()) {
        PythonTypes.putDictInt(arena, kwargs, "seed", seed);
      }
    }
    return this;
  }

  /**
   * Sets the number of output sequences to generate per prompt (beam width).
   *
   * @param n number of sequences
   * @return this
   */
  public SamplingParams n(int n) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "n", n);
    }
    return this;
  }

  /**
   * Sets the number of sequences to generate and return the best of.
   *
   * <p>When {@code bestOf > n}, vLLM generates {@code bestOf} sequences
   * internally and returns the top {@code n} by log probability.
   *
   * @param bestOf number of sequences to generate internally
   * @return this
   */
  public SamplingParams bestOf(int bestOf) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "best_of", bestOf);
    }
    return this;
  }

  /**
   * Sets the minimum number of tokens to generate before stop sequences
   * or EOS can fire.
   *
   * @param minTokens minimum output token count
   * @return this
   */
  public SamplingParams minTokens(int minTokens) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "min_tokens", minTokens);
    }
    return this;
  }

  /**
   * Sets the stop token IDs. Generation stops when any of these token IDs
   * is produced.
   *
   * @param tokenIds one or more token IDs
   * @return this
   */
  public SamplingParams stopTokenIds(int... tokenIds) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      MemorySegment pyList = CPythonBinding.PyList_New(0);
      for (int id : tokenIds) {
        MemorySegment pyInt = CPythonBinding.PyLong_FromLong(id);
        CPythonBinding.PyList_Append(pyList, pyInt);
        PythonTypes.decref(pyInt);
      }
      PythonTypes.putDictObj(arena, kwargs, "stop_token_ids", pyList);
      PythonTypes.decref(pyList);
    }
    return this;
  }

  /**
   * Sets the stop token IDs from a list.
   *
   * @param tokenIds list of token IDs that trigger stop
   * @return this
   */
  public SamplingParams stopTokenIds(List<Integer> tokenIds) {
    return stopTokenIds(
      tokenIds.stream().mapToInt(Integer::intValue).toArray()
    );
  }

  /**
   * When {@code true}, includes the stop string in the output text if
   * generation stopped due to a stop sequence.
   *
   * @param include whether to include the stop string
   * @return this
   */
  public SamplingParams includeStopStrInOutput(boolean include) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "include_stop_str_in_output",
        include ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
      );
    }
    return this;
  }

  /**
   * When {@code true} (default), strips special tokens from output text.
   *
   * @param skip whether to skip special tokens
   * @return this
   */
  public SamplingParams skipSpecialTokens(boolean skip) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "skip_special_tokens",
        skip ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
      );
    }
    return this;
  }

  /**
   * When {@code true} (default), adds spaces between special tokens.
   *
   * @param spaces whether to add spaces
   * @return this
   */
  public SamplingParams spacesBetweenSpecialTokens(boolean spaces) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "spaces_between_special_tokens",
        spaces ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
      );
    }
    return this;
  }

  /**
   * If the prompt exceeds this many tokens, truncate from the left.
   *
   * @param maxPromptTokens maximum prompt tokens before truncation
   * @return this
   */
  public SamplingParams truncatePromptTokens(int maxPromptTokens) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(
        arena,
        kwargs,
        "truncate_prompt_tokens",
        maxPromptTokens
      );
    }
    return this;
  }

  /**
   * Sets the number of top log probabilities to return per output token.
   *
   * @param n number of logprobs (0 to disable)
   * @return this
   */
  public SamplingParams logprobs(int n) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "logprobs", n);
    }
    return this;
  }

  /**
   * Sets the number of top log probabilities to return per prompt token.
   *
   * @param n number of prompt logprobs (0 to disable)
   * @return this
   */
  public SamplingParams promptLogprobs(int n) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      PythonTypes.putDictInt(arena, kwargs, "prompt_logprobs", n);
    }
    return this;
  }

  /**
   * Sets guided decoding constraints. Forces generation output to match
   * a specific structure (JSON schema, regex, choice, or grammar).
   *
   * @param params the guided decoding parameters
   * @return this
   * @see GuidedDecodingParams#json(String)
   * @see GuidedDecodingParams#regex(String)
   * @see GuidedDecodingParams#choice(java.util.List)
   * @see GuidedDecodingParams#grammar(String)
   */
  public SamplingParams guidedDecoding(GuidedDecodingParams params) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      MemorySegment pyGuided = params.toPython(arena);
      PythonTypes.putDictObj(arena, kwargs, "guided_decoding", pyGuided);
      PythonTypes.decref(pyGuided);
    }
    return this;
  }

  /**
   * Sets the stop sequences. Generation stops when any of these strings
   * is produced.
   *
   * @param sequences zero or more stop strings
   * @return this
   */
  public SamplingParams stop(String... sequences) {
    checkNotBuilt();
    try (var gil = GIL.acquire()) {
      MemorySegment pyStopList = CPythonBinding.PyList_New(0);
      for (String seq : sequences) {
        MemorySegment pyItem = PythonTypes.pyStr(arena, seq);
        CPythonBinding.PyList_Append(pyStopList, pyItem);
        PythonTypes.decref(pyItem);
      }
      PythonTypes.putDictObj(arena, kwargs, "stop", pyStopList);
      PythonTypes.decref(pyStopList);
    }
    return this;
  }

  /**
   * Sets the stop sequences from a list.
   *
   * @param sequences list of stop strings
   * @return this
   */
  public SamplingParams stop(List<String> sequences) {
    return stop(sequences.toArray(String[]::new));
  }

  // ── Build / access ──────────────────────────────────────────────────

  /**
   * Returns the underlying Python {@code SamplingParams} object, building it
   * on the first call. After this call, no more setter methods may be invoked.
   *
   * @return the raw {@code PyObject*} memory segment
   * @throws IllegalStateException if this SamplingParams has been freed
   */
  public MemorySegment get() {
    checkNotFreed();
    if (pyObject == null) {
      try (var gil = GIL.acquire()) {
        pyObject = PythonCall.callWithKwargs(ensureClass(arena), kwargs);
        PythonErrors.checkPythonError("SamplingParams() construction");
        PythonTypes.decref(kwargs);
        kwargs = null; // signal "built"
      }
    }
    return pyObject;
  }

  // ── Freeable / AutoCloseable ────────────────────────────────────────

  @Override
  public void close() {
    free();
  }

  @Override
  public void free() {
    if (!freed) {
      freed = true;
      try (var gil = GIL.acquire()) {
        if (pyObject != null) {
          PythonTypes.decref(pyObject);
          pyObject = null;
        }
        if (kwargs != null) {
          PythonTypes.decref(kwargs);
          kwargs = null;
        }
      }
    }
  }

  @Override
  public boolean isFree() {
    return freed;
  }

  // ── Guards ──────────────────────────────────────────────────────────

  private void checkNotBuilt() {
    checkNotFreed();
    if (kwargs == null) {
      throw new IllegalStateException(
        "SamplingParams has already been built — cannot modify after get() is called"
      );
    }
  }

  private void checkNotFreed() {
    if (freed) {
      throw new IllegalStateException("SamplingParams has been freed");
    }
  }
}
