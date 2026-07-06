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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import io.gravitee.vllm.binding.VllmException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Local-model loading via {@link VllmEngineBuilder#modelPath(Path)}.
 *
 * <p>These cover the validation only — building an engine needs CPython and
 * weights, which is what the integration suites are for. Validation is worth
 * testing on its own because the failure it prevents is so badly mislocated:
 * HuggingFace treats any non-existent path as a repo id, so without it a typo'd
 * path surfaces as {@code HFValidationError: Repo id must be in the form …}
 * from deep inside the Python stack.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class ModelPathTest {

  private static Path modelDir(Path tmp, String name) throws IOException {
    Path dir = Files.createDirectories(tmp.resolve(name));
    Files.writeString(dir.resolve("config.json"), "{\"model_type\":\"bert\"}");
    return dir;
  }

  @Test
  void accepts_a_huggingface_model_directory(@TempDir Path tmp)
    throws IOException {
    Path dir = modelDir(tmp, "bge-m3");

    var builder = VllmEngine.builder().modelPath(dir);

    assertThat(builder.model()).isEqualTo(
      dir.toAbsolutePath().normalize().toString()
    );
  }

  @Test
  void normalises_to_an_absolute_path(@TempDir Path tmp) throws IOException {
    Path dir = modelDir(tmp, "bge-m3");
    // vLLM resolves a relative path against the Python process's CWD, which is
    // not necessarily the JVM's — so it has to be absolute by the time it lands.
    Path awkward = dir.resolve("..").resolve("bge-m3");

    var builder = VllmEngine.builder().modelPath(awkward);

    assertThat(builder.model())
      .isEqualTo(dir.toAbsolutePath().normalize().toString())
      .doesNotContain("..");
  }

  @Test
  void rejects_a_directory_without_config_json(@TempDir Path tmp)
    throws IOException {
    // The shape you get by pointing at a HF cache repo root instead of a snapshot.
    Path dir = Files.createDirectories(tmp.resolve("models--org--name"));

    assertThatThrownBy(() -> VllmEngine.builder().modelPath(dir))
      .isInstanceOf(VllmException.class)
      .hasMessageContaining("config.json")
      .hasMessageContaining("snapshots");
  }

  @Test
  void rejects_a_path_that_does_not_exist(@TempDir Path tmp) {
    assertThatThrownBy(() ->
      VllmEngine.builder().modelPath(tmp.resolve("nope"))
    )
      .isInstanceOf(VllmException.class)
      .hasMessageContaining("does not exist");
  }

  @Test
  void rejects_null(@TempDir Path tmp) {
    assertThatThrownBy(() -> VllmEngine.builder().modelPath(null)).isInstanceOf(
      VllmException.class
    );
  }

  @Test
  void accepts_a_single_file_model(@TempDir Path tmp) throws IOException {
    // GGUF and friends carry no config.json, so only existence is checked.
    Path gguf = tmp.resolve("model.gguf");
    Files.writeString(gguf, "not really a gguf");

    var builder = VllmEngine.builder().modelPath(gguf);

    assertThat(builder.model()).isEqualTo(
      gguf.toAbsolutePath().normalize().toString()
    );
  }

  @Test
  void model_id_and_model_path_write_to_the_same_setting(@TempDir Path tmp)
    throws IOException {
    Path dir = modelDir(tmp, "local");

    // Last writer wins — the two are alternatives, not additive.
    assertThat(
      VllmEngine.builder().model("BAAI/bge-m3").modelPath(dir).model()
    ).isEqualTo(dir.toAbsolutePath().normalize().toString());
    assertThat(
      VllmEngine.builder().modelPath(dir).model("BAAI/bge-m3").model()
    ).isEqualTo("BAAI/bge-m3");
  }
}
