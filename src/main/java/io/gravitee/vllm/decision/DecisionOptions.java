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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Options of one decisions request. {@code null} fields keep the runtime's defaults.
 *
 * @param deadlineMs work not started by then answers {@code deadline_exceeded}
 * @param profile    {@code exact} or a profile the model enables
 * @param returnMeta whether the response carries {@link DecisionResponse#meta()}
 */
public record DecisionOptions(
  Double deadlineMs,
  String profile,
  boolean returnMeta
) {
  public DecisionOptions {
    if (deadlineMs != null && !(deadlineMs > 0)) {
      throw new IllegalArgumentException("deadlineMs must be > 0");
    }
  }

  /** Options asking only for {@link DecisionResponse#meta()}. */
  public static DecisionOptions withMeta() {
    return new DecisionOptions(null, null, true);
  }

  Map<String, Object> toMap() {
    Map<String, Object> map = new LinkedHashMap<>();
    if (deadlineMs != null) map.put("deadline_ms", deadlineMs);
    if (profile != null) map.put("profile", profile);
    if (returnMeta) map.put("return_meta", true);
    return map;
  }
}
