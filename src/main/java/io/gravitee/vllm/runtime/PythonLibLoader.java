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

import io.gravitee.vllm.binding.VllmException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Resolves and loads the {@code libpython} shared library before the
 * jextract-generated {@code CPython} class is first referenced.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code VLLM4J_LIBPYTHON_PATH} environment variable</li>
 *   <li>{@code vllm4j.libpython.path} system property</li>
 *   <li>{@code .libpython-path} file in the project directory</li>
 *   <li>Fallback: let jextract's static initializer handle it (absolute path
 *       baked in at code-generation time)</li>
 * </ol>
 */
public final class PythonLibLoader {

  private static volatile boolean loaded = false;

  private PythonLibLoader() {}

  /**
   * Ensures libpython is loaded. Idempotent — subsequent calls are no-ops.
   *
   * @param projectDir the project root directory (for finding {@code .libpython-path})
   */
  public static void ensureLoaded(Path projectDir) {
    if (loaded) return;
    synchronized (PythonLibLoader.class) {
      if (loaded) return;
      String path = resolve(projectDir);
      if (path != null) {
        System.load(path);
      }
      // If path is null, we rely on CPython.<clinit> to load it
      // (absolute path baked in by jextract at generate-sources time)
      loaded = true;
    }
  }

  /**
   * Resolves the absolute path to libpython, or {@code null} if none found
   * (in which case jextract's baked-in path will be used).
   */
  private static String resolve(Path projectDir) {
    // 1. Environment variable
    String envPath = System.getenv("VLLM4J_LIBPYTHON_PATH");
    if (
      envPath != null &&
      !envPath.isBlank() &&
      Files.isRegularFile(Path.of(envPath))
    ) {
      return envPath;
    }

    // 2. System property
    String propPath = System.getProperty("vllm4j.libpython.path");
    if (
      propPath != null &&
      !propPath.isBlank() &&
      Files.isRegularFile(Path.of(propPath))
    ) {
      return propPath;
    }

    // 3. .libpython-path file
    if (projectDir != null) {
      Path dotFile = projectDir.resolve(".libpython-path");
      if (Files.exists(dotFile)) {
        try {
          String path = Files.readString(dotFile).strip();
          if (!path.isEmpty() && Files.isRegularFile(Path.of(path))) {
            return path;
          }
        } catch (IOException ignored) {}
      }
    }

    // 4. Fallback to jextract baked-in path
    return null;
  }
}
