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
import io.gravitee.vllm.template.ChatTemplate;

/**
 * Lazily initializes a single VLM {@link VllmEngine} for the current JVM fork.
 *
 * <p>Loads {@code HuggingFaceTB/SmolVLM-256M-Instruct} (256M-parameter
 * vision-language model — tiny footprint, ideal for CI/test environments).
 * Requires CUDA (Linux).
 *
 * <p>With {@code reuseForks=false}, each test class gets its own JVM fork.
 * Call {@link #close()} in {@code @AfterAll} so the {@code _exit(0)} hook
 * reclaims all GPU memory for the next fork.
 *
 * <p>Run with:
 * <pre>{@code
 * mvn test -P vlm-integration,linux-x86_64,cuda
 * }</pre>
 */
final class SharedVlmEngine {

  private static volatile VllmEngine engine;
  private static volatile ChatTemplate chatTemplate;

  private SharedVlmEngine() {}

  static VllmEngine engine() {
    if (engine == null) {
      synchronized (SharedVlmEngine.class) {
        if (engine == null) {
          engine = VllmEngine.builder()
            .model("HuggingFaceTB/SmolVLM-256M-Instruct")
            .dtype("auto")
            .enforceEager(true)
            .maxModelLen(2048)
            .maxNumSeqs(2)
            .gpuMemoryUtilization(0.50)
            .build();
        }
      }
    }
    return engine;
  }

  static ChatTemplate chatTemplate() {
    if (chatTemplate == null) {
      synchronized (SharedVlmEngine.class) {
        if (chatTemplate == null) {
          chatTemplate = new ChatTemplate(engine());
        }
      }
    }
    return chatTemplate;
  }

  static void close() {
    if (engine != null) {
      engine.close();
    }
  }
}
