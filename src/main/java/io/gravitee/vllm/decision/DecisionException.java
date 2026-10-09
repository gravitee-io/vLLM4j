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

/**
 * A decisions request the runtime refused as a whole, with the contract's
 * status and error code.
 *
 * <p>Codes: {@code invalid_request}, {@code model_not_found},
 * {@code request_too_large}, {@code unsupported_surface}, {@code overloaded},
 * {@code internal_error}, {@code not_ready}. A single bad question does not
 * throw; it comes back as an {@link Answer.Failed}.
 */
public class DecisionException extends VllmException {

  private final int status;
  private final String code;

  public DecisionException(int status, String code, String message) {
    super("%s (%d): %s".formatted(code, status, message));
    this.status = status;
    this.code = code;
  }

  /** The HTTP status the runtime would have answered with. */
  public int status() {
    return status;
  }

  /** The contract's error code. */
  public String code() {
    return code;
  }
}
