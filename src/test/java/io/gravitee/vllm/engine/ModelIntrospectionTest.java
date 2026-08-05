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
package io.gravitee.vllm.engine;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.vllm.runtime.PythonRuntime;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Reading a model's shape from {@code config.json}, without loading weights.
 *
 * <p>Tagged {@code integration} because it needs the CPython runtime and the
 * Hub — but unlike the other integration tests it never builds an engine, so it
 * runs in seconds and needs no GPU.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@Tag("integration")
class ModelIntrospectionTest {

  private static Arena arena;

  @BeforeAll
  static void initRuntime() {
    VllmEngine.builder().initRuntime();
    arena = Arena.ofAuto();
  }

  @Test
  void reads_the_shape_of_a_dense_model() {
    var shape = ModelIntrospection.read(arena, "Qwen/Qwen3-0.6B", false);

    // Qwen3-0.6B: 28 layers, GQA with 8 KV heads over 16 attention heads,
    // head_dim 128, 40960-token positional encoding, ~0.75B params in bf16.
    assertThat(shape.numHiddenLayers()).isEqualTo(28);
    assertThat(shape.numKvHeads()).isEqualTo(8);
    assertThat(shape.headDim()).isEqualTo(128);
    assertThat(shape.maxPositionEmbeddings()).isEqualTo(40960);
    assertThat(shape.multimodal()).isFalse();
    assertThat(shape.bitsPerParam()).isEqualTo(16);
    assertThat(shape.totalParams()).isCloseTo(
      751_632_384L,
      org.assertj.core.data.Offset.offset(1L)
    );
    assertThat(shape.isUsable()).isTrue();
  }

  @Test
  void a_quantized_model_reports_its_quantized_width() {
    // The case a dtype read gets wrong: an AWQ checkpoint declares float16 for
    // its activations while the weights are 4-bit. Taking dtype at face value
    // would overstate the weights by 4x and reject a model that fits.
    var shape = ModelIntrospection.read(arena, "Qwen/Qwen3-4B-AWQ", false);

    assertThat(shape.bitsPerParam())
      .as("4-bit weights must not be read as 16-bit float16")
      .isEqualTo(4);
    assertThat(shape.numHiddenLayers()).isPositive();
  }

  @Test
  void an_unknown_model_degrades_instead_of_throwing() {
    // A pre-flight estimate must never be the reason a model fails to load.
    var shape = ModelIntrospection.read(
      arena,
      "definitely/not-a-real-model-xyz",
      false
    );

    assertThat(shape.isUsable()).isFalse();
  }

  @Test
  void a_blank_model_is_unknown() {
    assertThat(ModelIntrospection.read(arena, "", false)).isEqualTo(
      ModelShapeUnknown()
    );
    assertThat(ModelIntrospection.read(arena, null, false)).isEqualTo(
      ModelShapeUnknown()
    );
  }

  private static ModelIntrospection.ModelShape ModelShapeUnknown() {
    return ModelIntrospection.ModelShape.UNKNOWN;
  }

  @Test
  void runtime_is_initialised() {
    assertThat(PythonRuntime.isInitialized()).isTrue();
  }
}
