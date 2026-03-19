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
package io.gravitee.vllm.iterator;

import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmEngineBuilder;
import io.gravitee.vllm.template.ChatTemplate;

/**
 * Lazily initializes a single {@link VllmEngine} for the current JVM fork.
 *
 * <p>With {@code reuseForks=false} in the Surefire configuration, each test
 * class runs in its own JVM. The engine is created once per fork and closed
 * in {@code @AfterAll}, which registers the {@code _exit(0)} shutdown hook
 * in {@link io.gravitee.vllm.runtime.PythonRuntime} — the OS reclaims all
 * GPU memory when the process exits, leaving the next fork with a clean GPU.
 *
 * <p>Two engine configurations are available:
 * <ul>
 *   <li>{@link #baseEngine()} — no LoRA, higher GPU utilization
 *   <li>{@link #loraEngine()} — LoRA enabled, tuned for ≤ 8 GiB cards
 * </ul>
 */
final class SharedEngine {

  private static volatile VllmEngine engine;
  private static volatile ChatTemplate chatTemplate;

  private SharedEngine() {}

  /**
   * Base engine — no LoRA overhead, uses most of the GPU.
   */
  static VllmEngine baseEngine() {
    return getOrCreate(
      VllmEngine.builder()
        .model("Qwen/Qwen3-0.6B")
        .dtype("auto")
        .enforceEager(true)
        .maxModelLen(4096)
        .maxNumSeqs(4)
        .gpuMemoryUtilization(0.85)
    );
  }

  /**
   * LoRA-enabled engine — tuned for ≤ 8 GiB cards.
   * LoRA infrastructure adds ~2-3 GiB overhead (weight slots, profiling).
   */
  static VllmEngine loraEngine() {
    return getOrCreate(
      VllmEngine.builder()
        .model("Qwen/Qwen3-0.6B")
        .dtype("auto")
        .enableLora(true)
        .maxLoras(1)
        .maxLoraRank(64)
        .enforceEager(true)
        .maxModelLen(2048)
        .maxNumSeqs(4)
        .gpuMemoryUtilization(0.85)
    );
  }

  static ChatTemplate chatTemplate() {
    if (chatTemplate == null) {
      synchronized (SharedEngine.class) {
        if (chatTemplate == null) {
          if (engine == null) {
            throw new IllegalStateException(
              "Engine must be initialized before chatTemplate()"
            );
          }
          chatTemplate = new ChatTemplate(engine);
        }
      }
    }
    return chatTemplate;
  }

  /**
   * Closes the engine, triggering the {@code _exit(0)} shutdown hook.
   * Call from {@code @AfterAll} in each test class.
   */
  static void close() {
    if (engine != null) {
      engine.close();
    }
  }

  // ── internal ────────────────────────────────────────────────────────

  private static VllmEngine getOrCreate(VllmEngineBuilder builder) {
    if (engine == null) {
      synchronized (SharedEngine.class) {
        if (engine == null) {
          engine = builder.build();
        }
      }
    }
    return engine;
  }
}
