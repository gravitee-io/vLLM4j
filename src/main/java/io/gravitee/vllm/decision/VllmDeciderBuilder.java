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

import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.platform.PlatformResolver;
import io.gravitee.vllm.platform.VllmBackend;
import io.gravitee.vllm.runtime.PythonRuntime;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Fluent builder for {@link VllmDecider}. Obtain via {@link VllmDecider#builder()}.
 *
 * <p>Options map to {@code vllm_srun}'s {@code ModelConfig} and
 * {@code ServeConfig}; unset options keep the runtime's defaults.
 */
public final class VllmDeciderBuilder {

  private Path venvPath;
  private VllmBackend backend;
  private Arena arena;

  // ModelConfig
  private String model;
  private String revision;
  private String device;
  private String profile;
  private String engine;
  private Double memoryBudgetGib;

  // ServeConfig
  private Integer threads;
  private String cacheDir;
  private Boolean offline;

  /** Package-private: obtain via {@link VllmDecider#builder()}. */
  VllmDeciderBuilder() {}

  /** The uv-managed {@code .venv} with {@code vllm-srun}; auto-detected when not set. */
  public VllmDeciderBuilder venvPath(Path venvPath) {
    this.venvPath = venvPath;
    return this;
  }

  /** The backend whose environment the interpreter starts with; detected when not set. */
  public VllmDeciderBuilder backend(VllmBackend backend) {
    this.backend = backend;
    return this;
  }

  /** Arena for native allocations; must outlive the decider. Defaults to {@link Arena#ofAuto()}. */
  public VllmDeciderBuilder arena(Arena arena) {
    this.arena = arena;
    return this;
  }

  /**
   * The decision model: a built-in Hugging Face id such as
   * {@code vllm-sr/Decision-2.0-Kai-0.6B} (resolved to its pinned revision),
   * or a local package directory. Required.
   */
  public VllmDeciderBuilder model(String model) {
    this.model = model;
    return this;
  }

  /** A Hub revision to load instead of the built-in pin. */
  public VllmDeciderBuilder revision(String revision) {
    this.revision = revision;
    return this;
  }

  /** {@code auto} (default), {@code cpu}, {@code cuda}, {@code cuda:1}, {@code mps}, {@code rocm}… */
  public VllmDeciderBuilder device(String device) {
    this.device = device;
    return this;
  }

  /** {@code exact} (default; matches the released numerics) or a faster profile the model enables. */
  public VllmDeciderBuilder profile(String profile) {
    this.profile = profile;
    return this;
  }

  /** {@code auto} (default), {@code native} or {@code onnxruntime}. */
  public VllmDeciderBuilder engine(String engine) {
    this.engine = engine;
    return this;
  }

  /** Device memory the model may use, in GiB; set it when sharing a GPU with a {@code VllmEngine}. */
  public VllmDeciderBuilder memoryBudgetGib(double memoryBudgetGib) {
    this.memoryBudgetGib = memoryBudgetGib;
    return this;
  }

  /** CPU threads the runtime may use. */
  public VllmDeciderBuilder threads(int threads) {
    this.threads = threads;
    return this;
  }

  /** Where model files are downloaded; the Hugging Face cache when not set. */
  public VllmDeciderBuilder cacheDir(String cacheDir) {
    this.cacheDir = cacheDir;
    return this;
  }

  /** Load from the cache only, without contacting the Hub. */
  public VllmDeciderBuilder offline(boolean offline) {
    this.offline = offline;
    return this;
  }

  /**
   * Initializes the CPython runtime if needed and loads the model.
   *
   * @throws VllmException if the venv cannot be found, {@code vllm-srun} is
   *                       missing, or the model fails to load or to pass its golden answers
   */
  public VllmDecider build() {
    if (model == null || model.isBlank()) {
      throw new IllegalArgumentException("model must be set");
    }
    initRuntime();
    return new VllmDecider(arena != null ? arena : Arena.ofAuto(), this);
  }

  private void initRuntime() {
    if (PythonRuntime.isInitialized()) {
      return;
    }
    // vLLM reads this at import time, from the environment the interpreter
    // starts with: keep the VllmEngine default for an engine built later.
    if (System.getenv("VLLM_ENABLE_V1_MULTIPROCESSING") == null) {
      PythonRuntime.setEnv("VLLM_ENABLE_V1_MULTIPROCESSING", "0");
    }
    new PythonRuntime(
      venvPath != null ? venvPath : PythonRuntime.resolveVenv(),
      backend != null ? backend : PlatformResolver.backend()
    );
  }

  // ── Package-private accessors for VllmDecider ───────────────────────────

  String model() {
    return model;
  }

  /** Keyword arguments of {@code ModelConfig}. */
  Map<String, Object> modelConfig() {
    Map<String, Object> config = new LinkedHashMap<>();
    config.put("model", model);
    if (revision != null) config.put("revision", revision);
    if (device != null) config.put("device", device);
    if (profile != null) config.put("profile", profile);
    if (engine != null) config.put("engine", engine);
    if (memoryBudgetGib != null) config.put(
      "memory_budget_gib",
      memoryBudgetGib
    );
    return config;
  }

  /** Keyword arguments of {@code ServeConfig}, without {@code models}. */
  Map<String, Object> serveConfig() {
    Map<String, Object> config = new LinkedHashMap<>();
    if (threads != null) config.put("threads", threads);
    if (cacheDir != null) config.put("cache_dir", cacheDir);
    if (offline != null) config.put("offline", offline);
    return config;
  }
}
