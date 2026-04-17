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

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.vllm.engine.VllmEngine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for {@link VllmEngine} special token and chat template
 * accessors.
 *
 * <p>Verifies that BOS/EOS tokens and the Jinja2 chat template string can
 * be read from the HuggingFace tokenizer via the CPython bridge.
 *
 * <p>Uses Qwen/Qwen3-0.6B via {@link SharedEngine}.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@Tag("integration")
class VllmEngineTokenTest {

  private static VllmEngine engine;

  @BeforeAll
  static void initEngine() {
    engine = SharedEngine.baseEngine();
  }

  @AfterAll
  static void closeEngine() {
    SharedEngine.close();
  }

  @Test
  void engine_returns_eos_token() {
    String eos = engine.getEosToken();

    System.out.println("Qwen3 EOS token: '" + eos + "'");

    // Qwen3 uses <|im_end|> as EOS
    assertThat(eos).isNotNull();
    assertThat(eos).isNotEmpty();
    assertThat(eos).contains("im_end");
  }

  @Test
  void engine_returns_bos_token_or_empty() {
    String bos = engine.getBosToken();

    System.out.println("Qwen3 BOS token: '" + bos + "'");

    // Qwen3 has bos_token: null in tokenizer_config.json
    // getBosToken() should return empty string, not throw
    assertThat(bos).isNotNull();
  }

  @Test
  void engine_returns_chat_template_string() {
    String template = engine.getChatTemplate();

    System.out.println(
      "Qwen3 template length: " +
        (template != null ? template.length() : "null")
    );
    System.out.println(
      "Qwen3 template preview: " +
        (template != null
            ? template.substring(0, Math.min(200, template.length()))
            : "null")
    );

    assertThat(template).isNotNull();
    assertThat(template).isNotEmpty();
    // The template should contain Jinja2 syntax
    assertThat(template).contains("{%");
    assertThat(template).contains("messages");
    // Qwen3 templates reference im_start/im_end
    assertThat(template).contains("im_start");
  }
}
