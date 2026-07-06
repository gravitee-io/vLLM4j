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
package io.gravitee.vllm.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import io.gravitee.vllm.binding.VllmException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * PYTHONHOME resolution from a venv's {@code pyvenv.cfg}.
 *
 * <p>These are filesystem-shape tests, not CPython tests: the failure they
 * guard against happens inside {@code Py_InitializeEx}, before any Java code
 * can report it, and surfaces only as
 * {@code Fatal Python error: init_fs_encoding ... No module named 'encodings'}
 * with the JVM dying. That is unactionable in CI, so the shape is checked here
 * instead.
 *
 * <p>The Homebrew case is the one that actually broke: Homebrew's macOS pythons
 * are framework builds that expose {@code <opt>/bin/python3.X} as a shim, so
 * {@code pyvenv.cfg} records {@code home = <opt>/bin} whose parent holds no
 * stdlib at all.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class PythonHomeResolutionTest {

  /** Builds a venv directory whose pyvenv.cfg points {@code home} at {@code binDir}. */
  private static Path venvPointingAt(Path tmp, Path binDir) throws IOException {
    Path venv = Files.createDirectories(tmp.resolve(".venv"));
    Files.writeString(
      venv.resolve("pyvenv.cfg"),
      "home = " + binDir + "\nversion = 3.12.0\n"
    );
    return venv;
  }

  /** Lays out {@code <prefix>/lib/python3.12/encodings} — i.e. a real stdlib. */
  private static void withStdlib(Path prefix) throws IOException {
    Files.createDirectories(prefix.resolve("lib/python3.12/encodings"));
  }

  @Test
  void resolves_a_normal_prefix_from_the_parent_of_home(@TempDir Path tmp)
    throws IOException {
    Path prefix = tmp.resolve("usr");
    withStdlib(prefix);
    Path venv = venvPointingAt(
      tmp,
      Files.createDirectories(prefix.resolve("bin"))
    );

    assertThat(PythonRuntime.resolvePythonHome(venv.toString())).isEqualTo(
      prefix.toString()
    );
  }

  @Test
  void falls_back_to_the_framework_prefix_when_the_parent_has_no_stdlib(
    @TempDir Path tmp
  ) throws IOException {
    // The Homebrew layout: <opt>/bin is a shim, the stdlib is in the framework.
    Path opt = tmp.resolve("opt/python@3.12");
    Path shimBin = Files.createDirectories(opt.resolve("bin"));
    Path framework = opt.resolve("Frameworks/Python.framework/Versions/3.12");
    withStdlib(framework);

    Path venv = venvPointingAt(tmp, shimBin);

    // Without the fallback this returns <opt>, and CPython dies on startup.
    assertThat(PythonRuntime.resolvePythonHome(venv.toString())).isEqualTo(
      framework.toString()
    );
  }

  @Test
  void fails_loudly_when_no_stdlib_can_be_found(@TempDir Path tmp)
    throws IOException {
    Path prefix = tmp.resolve("empty");
    Path venv = venvPointingAt(
      tmp,
      Files.createDirectories(prefix.resolve("bin"))
    );

    // Better a clear exception here than a fatal CPython abort with no stack.
    assertThatThrownBy(() -> PythonRuntime.resolvePythonHome(venv.toString()))
      .isInstanceOf(VllmException.class)
      .hasMessageContaining("standard library");
  }

  @Test
  void fails_loudly_when_the_venv_has_no_pyvenv_cfg(@TempDir Path tmp)
    throws IOException {
    Path venv = Files.createDirectories(tmp.resolve(".venv"));

    assertThatThrownBy(() -> PythonRuntime.resolvePythonHome(venv.toString()))
      .isInstanceOf(VllmException.class)
      .hasMessageContaining("pyvenv.cfg");
  }
}
