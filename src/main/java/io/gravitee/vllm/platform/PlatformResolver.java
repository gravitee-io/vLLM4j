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
 * Singleton that resolves the current platform (OS + architecture) and the
 * vLLM compute backend at class-load time.
 *
 * <p>The resolved values are cached and immutable for the lifetime of the JVM.
 */
public final class PlatformResolver {

  private static final Platform PLATFORM = new Platform(
    OperatingSystem.fromSystem(),
    Architecture.fromSystem()
  );

  private static final VllmBackend BACKEND = VllmBackend.detect();

  private PlatformResolver() {}

  /** Returns the detected platform (OS + architecture). */
  public static Platform platform() {
    return PLATFORM;
  }

  /** Returns the detected operating system. */
  public static OperatingSystem os() {
    return PLATFORM.os();
  }

  /** Returns the detected CPU architecture. */
  public static Architecture architecture() {
    return PLATFORM.architecture();
  }

  /** Returns the detected (or explicitly configured) vLLM backend. */
  public static VllmBackend backend() {
    return BACKEND;
  }
}
