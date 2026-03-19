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

import io.gravitee.vllm.engine.LoraRequest;
import io.gravitee.vllm.engine.RequestOutput;
import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for LoRA adapter support via {@link VllmEngine}.
 *
 * <p>Uses the base model {@code Qwen/Qwen3-0.6B} with the LoRA adapter
 * {@code gauravprasadgp/Qwen3-0.6B_nlp_to_sql} (an NLP-to-SQL fine-tune).
 * The adapter is automatically downloaded from HuggingFace on first run via
 * {@code vllm.lora.utils.get_adapter_absolute_path()}.
 *
 * <p>The engine is created with LoRA enabled and closed in {@code @AfterAll}.
 * With {@code reuseForks=false}, each test class gets its own JVM fork —
 * GPU memory is fully reclaimed between classes.
 *
 * <p>Tagged {@code "integration"} and excluded from the default
 * {@code mvn test} run. Execute with:
 * <pre>{@code
 * mvn test -P integration,linux-x86_64,cuda
 * }</pre>
 */
@Tag("integration")
class LoraTest {

  private static final String LORA_REPO =
    "gauravprasadgp/Qwen3-0.6B_nlp_to_sql";
  private static final String LORA_NAME = "sql-lora";
  private static final int LORA_INT_ID = 1;

  private static VllmEngine engine;
  private static ChatTemplate chatTemplate;

  @BeforeAll
  static void initEngine() {
    engine = SharedEngine.loraEngine();
    chatTemplate = SharedEngine.chatTemplate();
  }

  @AfterAll
  static void closeEngine() {
    SharedEngine.close();
  }

  // ═══════════════════════════════════════════════════════════════════
  //  LoRA path resolution — auto-download from HuggingFace
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void resolve_lora_path_should_return_local_path() {
    String resolved = engine.resolveLoraPath(LORA_REPO);

    assertThat(resolved).isNotNull();
    assertThat(resolved).isNotEmpty();
    // The resolved path should be a local directory (absolute path)
    assertThat(resolved).startsWith("/");

    System.out.println("\n=== LoRA Path Resolution ===");
    System.out.println("Input:    " + LORA_REPO);
    System.out.println("Resolved: " + resolved);
    System.out.println("============================");
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Blocking generate with LoRA — SQL adapter
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void generate_with_lora_adapter_from_huggingface() {
    var lora = new LoraRequest(LORA_NAME, LORA_INT_ID, LORA_REPO);

    String prompt = chatTemplate.render(
      List.of(
        ChatMessage.system(
          "You are a SQL expert. Convert natural language queries to SQL. /no_think"
        ),
        ChatMessage.user("Show all employees with salary greater than 50000")
      ),
      true
    );

    try (
      var sp = new SamplingParams(engine.arena())
        .temperature(0.0)
        .maxTokens(128)
    ) {
      var request = new VllmRequest("req-lora-sql", prompt, sp, lora);
      RequestOutput output = engine.generate(request);

      assertThat(output).isNotNull();
      assertThat(output.requestId()).isEqualTo("req-lora-sql");
      assertThat(output.finished()).isTrue();
      assertThat(output.outputs()).isNotEmpty();

      String text = output.outputs().getFirst().text();
      assertThat(text).isNotEmpty();

      System.out.println("\n=== LoRA Generate (SQL) ===");
      System.out.println(
        "Prompt: Show all employees with salary greater than 50000"
      );
      System.out.println("Output: " + text);
      System.out.println("Prompt tokens: " + output.numPromptTokens());
      System.out.println("Generated tokens: " + output.numGeneratedTokens());
      System.out.println("===========================");
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Generate without LoRA — base model on same engine
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void generate_without_lora_uses_base_model() {
    String prompt = chatTemplate.render(
      List.of(ChatMessage.user("What is the capital of France? /no_think")),
      true
    );

    try (
      var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(64)
    ) {
      var request = new VllmRequest("req-base", prompt, sp);
      RequestOutput output = engine.generate(request);

      assertThat(output).isNotNull();
      assertThat(output.requestId()).isEqualTo("req-base");
      assertThat(output.finished()).isTrue();
      assertThat(output.outputs().getFirst().text()).isNotEmpty();

      System.out.println("\n=== Base Model Generate (no LoRA) ===");
      System.out.println("Output: " + output.outputs().getFirst().text());
      System.out.println("======================================");
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Streaming with LoRA — VllmIterator path
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void stream_with_lora_adapter() {
    var lora = new LoraRequest(LORA_NAME, LORA_INT_ID, LORA_REPO);

    String prompt = chatTemplate.render(
      List.of(
        ChatMessage.system("Convert to SQL. /no_think"),
        ChatMessage.user("List all products with price less than 10")
      ),
      true
    );

    try (
      var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(64)
    ) {
      var request = new VllmRequest("req-lora-stream", prompt, sp, lora);
      var iterator = new VllmIterator(engine);
      iterator.addRequest(request);

      var output = new StringBuilder();
      iterator.stream().forEach(o -> output.append(o.delta()));

      assertThat(output.toString()).isNotEmpty();

      System.out.println("\n=== LoRA Stream ===");
      System.out.println("Output: " + output);
      System.out.println("====================");
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Parallel: LoRA + base model on same engine
  // ═══════════════════════════════════════════════════════════════════

  @Test
  void parallel_lora_and_base_model_requests() {
    var lora = new LoraRequest(LORA_NAME, LORA_INT_ID, LORA_REPO);

    String loraPrompt = chatTemplate.render(
      List.of(
        ChatMessage.system("Convert to SQL. /no_think"),
        ChatMessage.user("Get all orders from last month")
      ),
      true
    );

    String basePrompt = chatTemplate.render(
      List.of(ChatMessage.user("What is 2 + 2? /no_think")),
      true
    );

    var iterator = new VllmIterator(engine);

    try (
      var sp1 = new SamplingParams(engine.arena())
        .temperature(0.0)
        .maxTokens(64);
      var sp2 = new SamplingParams(engine.arena())
        .temperature(0.0)
        .maxTokens(64)
    ) {
      // Submit LoRA request and base model request in parallel
      iterator.addRequest(
        new VllmRequest("req-lora-parallel", loraPrompt, sp1, lora)
      );
      iterator.addRequest(
        new VllmRequest("req-base-parallel", basePrompt, sp2)
      );

      var loraOutput = new StringBuilder();
      var baseOutput = new StringBuilder();

      iterator
        .stream()
        .forEach(o -> {
          if ("req-lora-parallel".equals(o.requestId())) {
            loraOutput.append(o.delta());
          } else {
            baseOutput.append(o.delta());
          }
        });

      assertThat(loraOutput.toString()).isNotEmpty();
      assertThat(baseOutput.toString()).isNotEmpty();

      System.out.println("\n=== Parallel LoRA + Base ===");
      System.out.println("LoRA output: " + loraOutput);
      System.out.println("Base output: " + baseOutput);
      System.out.println("============================");
    }
  }
}
