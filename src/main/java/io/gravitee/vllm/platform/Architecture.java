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
package io.gravitee.vllm.platform;

/**
 * Supported CPU architectures for vLLM4j.
 *
 * <p>Auto-detected from the {@code os.arch} system property.
 */
public enum Architecture {
  X86_64("x86_64"),
  AARCH64("aarch64");

  private final String arch;

  Architecture(String arch) {
    this.arch = arch;
  }

  /**
   * Detects the current architecture from {@code os.arch}.
   *
   * @throws IllegalArgumentException if the architecture is not supported
   */
  public static Architecture fromSystem() {
    String osArch = System.getProperty("os.arch").toLowerCase();
    if (osArch.contains("x86_64") || osArch.contains("amd64")) {
      return X86_64;
    }
    if (osArch.contains("aarch64") || osArch.contains("arm64")) {
      return AARCH64;
    }
    throw new IllegalArgumentException("Unsupported architecture: " + osArch);
  }

  public String getArch() {
    return arch;
  }
}
