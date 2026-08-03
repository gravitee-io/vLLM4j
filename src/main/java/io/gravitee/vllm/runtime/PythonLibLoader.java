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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

/**
 * Resolves and loads the {@code libpython} shared library before the
 * jextract-generated {@code CPython} class is first referenced.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code VLLM4J_LIBPYTHON_PATH} environment variable</li>
 *   <li>{@code vllm4j.libpython.path} system property</li>
 *   <li>{@code <venv>/lib/libpython*} — the stable symlink created by
 *       {@code generate-sources.sh}</li>
 *   <li>Derived from the venv: {@code pyvenv.cfg}'s {@code home} entry points
 *       at the base interpreter's {@code bin} directory; libpython lives in
 *       the sibling {@code lib} directory (or, for macOS framework builds,
 *       the {@code Python} binary in the prefix itself)</li>
 * </ol>
 *
 * <p>There is no fallback, by design. jextract can bake an absolute
 * {@code System.load(...)} into the generated class, but that path is the one
 * the <em>build</em> machine had: the jar then runs only there, and fails
 * everywhere else — in a container, in CI, on a colleague's laptop — with an
 * {@code UnsatisfiedLinkError} naming a directory that does not exist, however
 * correctly the venv is configured. {@code generate-sources.sh} strips that
 * block, so this class is the only thing that loads libpython, and it decides
 * from where the process is actually running.
 */
public final class PythonLibLoader {

  private static volatile boolean loaded = false;

  private PythonLibLoader() {}

  /**
   * Ensures libpython is loaded. Idempotent — subsequent calls are no-ops.
   *
   * @param venvDir the venv directory (for deriving libpython from {@code pyvenv.cfg})
   * @throws IllegalStateException if no libpython can be found. Failing here,
   *         naming every location that was tried, beats the
   *         {@code UnsatisfiedLinkError} the caller would otherwise hit on the
   *         first binding call — which names only one path and explains nothing.
   */
  public static void ensureLoaded(Path venvDir) {
    if (loaded) return;
    synchronized (PythonLibLoader.class) {
      if (loaded) return;
      String path = resolve(venvDir);
      if (path == null) {
        throw new IllegalStateException(notFoundMessage(venvDir));
      }
      System.load(path);
      loaded = true;
    }
  }

  private static String notFoundMessage(Path venvDir) {
    return (
      "Could not locate libpython. Tried, in order: the VLLM4J_LIBPYTHON_PATH " +
      "environment variable, the vllm4j.libpython.path system property, " +
      (venvDir == null
          ? "and no venv was supplied (set -Dvllm4j.venv or " +
          "VllmEngine.builder().venvPath(...))."
          : "'" +
          venvDir +
          "/lib/libpython*', and the base interpreter named by '" +
          venvDir +
          "/pyvenv.cfg'.")
    );
  }

  /**
   * Resolves the absolute path to libpython, or {@code null} if none found
   * (in which case jextract's baked-in path will be used).
   */
  private static String resolve(Path venvDir) {
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

    // 3. Symlink placed in <venv>/lib by generate-sources.sh
    if (venvDir != null) {
      String linked = findLibpython(venvDir);
      if (linked != null) {
        return linked;
      }
    }

    // 4. Derive from the venv's base interpreter
    if (venvDir != null) {
      Path prefix = basePrefix(venvDir);
      if (prefix != null) {
        String found = findLibpython(prefix);
        if (found != null) {
          return found;
        }
      }
    }

    // 5. Fallback to jextract baked-in path
    return null;
  }

  /**
   * Reads the base interpreter prefix from the venv's {@code pyvenv.cfg}
   * ({@code home} points at the base interpreter's {@code bin} directory).
   */
  static Path basePrefix(Path venvDir) {
    Path pyvenvCfg = venvDir.resolve("pyvenv.cfg");
    if (!Files.exists(pyvenvCfg)) {
      return null;
    }
    try {
      for (String line : Files.readAllLines(pyvenvCfg)) {
        if (line.startsWith("home")) {
          String[] parts = line.split("=", 2);
          if (parts.length == 2) {
            Path home = Path.of(parts[1].strip());
            Path prefix = home.getParent();
            if (prefix != null && Files.isDirectory(prefix)) {
              return prefix;
            }
          }
        }
      }
    } catch (IOException ignored) {}
    return null;
  }

  /**
   * Locates the libpython shared library under the given base prefix:
   * {@code <prefix>/lib/libpython3.x.{dylib,so[.1.0]}}, falling back to the
   * {@code <prefix>/Python} framework binary on macOS framework builds.
   */
  private static String findLibpython(Path prefix) {
    Path libDir = prefix.resolve("lib");
    if (Files.isDirectory(libDir)) {
      try (Stream<Path> entries = Files.list(libDir)) {
        Path lib = entries
          .filter(p -> {
            String name = p.getFileName().toString();
            return (
              name.startsWith("libpython") &&
              (name.endsWith(".dylib") || name.contains(".so")) &&
              Files.isRegularFile(p)
            );
          })
          .findFirst()
          .orElse(null);
        if (lib != null) {
          return lib.toRealPath().toString();
        }
      } catch (IOException ignored) {}
    }

    // macOS framework build: the shared library is the framework binary
    Path frameworkBinary = prefix.resolve("Python");
    if (Files.isRegularFile(frameworkBinary)) {
      try {
        return frameworkBinary.toRealPath().toString();
      } catch (IOException ignored) {}
    }
    return null;
  }
}
