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
 * Combines the detected operating system and CPU architecture into a single
 * value that can derive platform-specific package names and runtime identifiers.
 *
 * <p>Used by the reflection-based FFM dispatch in {@link io.gravitee.vllm.binding.CPythonBinding}
 * to locate the correct jextract-generated class for the current platform.
 *
 * @param os           the detected operating system
 * @param architecture the detected CPU architecture
 */
public record Platform(OperatingSystem os, Architecture architecture) {
  /**
   * Returns a runtime identifier string, e.g. {@code "macosx_aarch64"} or
   * {@code "linux_x86_64"}. Used for logging and error messages.
   */
  public String runtime() {
    return os.getOsName() + "_" + architecture.getArch();
  }

  /**
   * Returns the Java package suffix for jextract-generated classes, e.g.
   * {@code "macosx.aarch64"} or {@code "linux.x86_64"}.
   *
   * <p>The full generated class name is
   * {@code io.gravitee.vllm.<getPackage()>.CPython}.
   */
  public String getPackage() {
    return os.getOsName() + "." + architecture.getArch();
  }
}
