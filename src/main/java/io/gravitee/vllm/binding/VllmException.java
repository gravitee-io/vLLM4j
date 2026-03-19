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
package io.gravitee.vllm.binding;

/**
 * Thrown when a CPython exception propagates across the FFM boundary.
 *
 * <p>The message is the string representation of the Python exception that was
 * active at the time of the call ({@code PyErr_Fetch} + {@code PyObject_Str}),
 * so it matches what you would see in a Python traceback.
 */
public class VllmException extends RuntimeException {

  public VllmException(String message) {
    super(message);
  }

  public VllmException(String message, Throwable cause) {
    super(message, cause);
  }
}
