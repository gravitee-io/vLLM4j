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
package io.gravitee.vllm;

/**
 * Contract for objects that hold native resources requiring explicit release.
 *
 * <p>Mirrors the pattern from llamaj.cpp. Implementors should call
 * {@link #free()} when the resource is no longer needed, and {@link #isFree()}
 * should return {@code true} after that point.
 */
public interface Freeable {
  /** Releases the underlying native resource. Idempotent. */
  void free();

  /** Returns {@code true} if {@link #free()} has been called. */
  default boolean isFree() {
    return false;
  }
}
