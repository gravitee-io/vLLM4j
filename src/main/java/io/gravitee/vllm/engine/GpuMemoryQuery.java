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

import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.runtime.GIL;
import io.gravitee.vllm.runtime.PythonRuntime;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Queries GPU memory through the live CPython interpreter.
 *
 * <p>Supports two backends:
 * <ul>
 *   <li><b>CUDA</b> — via {@code torch.cuda.mem_get_info(0)}</li>
 *   <li><b>MPS</b> (Apple Silicon) — total memory from {@code os.sysconf}
 *       (unified memory), allocated from {@code torch.mps.current_allocated_memory()}</li>
 * </ul>
 *
 * <p>The detected backend is cached after the first call to {@link #detectBackend()}.
 *
 * <p>This class belongs in the vLLM4j binding layer because it requires the
 * CPython FFM infrastructure ({@link GIL}, {@link CPythonBinding}, {@link PythonTypes}).
 * Higher-level estimation logic that depends on {@code MemoryEstimate} or
 * {@code SafetensorsInfo} lives in the adapter modules.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class GpuMemoryQuery {

  /** The GPU compute backends we support. */
  public enum GpuBackend {
    CUDA,
    MPS,
    NONE,
  }

  private static volatile GpuBackend cachedBackend;

  private GpuMemoryQuery() {}

  /**
   * Detects the available GPU backend. Result is cached.
   *
   * <p>Checks CUDA first (higher priority for production), then MPS.
   * Returns {@link GpuBackend#NONE} if neither is available.
   *
   * <p>Never throws — returns {@code NONE} on any error.
   */
  public static GpuBackend detectBackend() {
    if (cachedBackend != null) return cachedBackend;
    if (!PythonRuntime.isInitialized()) return GpuBackend.NONE;

    try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
      // Try CUDA first
      if (checkTorchBackend(arena, "torch.cuda")) {
        cachedBackend = GpuBackend.CUDA;
        return cachedBackend;
      }
      // Then MPS
      if (checkTorchBackend(arena, "torch.backends.mps")) {
        cachedBackend = GpuBackend.MPS;
        return cachedBackend;
      }
    } catch (Exception e) {
      // fall through
    }
    cachedBackend = GpuBackend.NONE;
    return cachedBackend;
  }

  /**
   * Queries the GPU for free and total memory.
   *
   * <p>Dispatches to the appropriate backend (CUDA or MPS).
   *
   * <p><b>Important:</b> Returns {@code null} if the CPython runtime has not yet
   * been initialized. This is safe to call at any time; if initialization is
   * incomplete, it returns null rather than crashing.
   *
   * @return a {@link GpuMemoryInfo} with {@code (freeBytes, totalBytes)},
   *         or {@code null} if no GPU backend is available, CPython is not yet
   *         initialized, or any error occurs. Never throws.
   */
  public static GpuMemoryInfo query() {
    if (!PythonRuntime.isInitialized()) {
      return null;
    }
    return switch (detectBackend()) {
      case CUDA -> queryCuda();
      case MPS -> queryMps();
      case NONE -> null;
    };
  }

  /**
   * Releases cached GPU memory back to the system.
   *
   * <p>On CUDA: calls {@code torch.cuda.empty_cache()} to flush PyTorch's
   * caching allocator.
   * <p>On MPS: calls {@code torch.mps.empty_cache()} to flush the MPS
   * memory pool.
   * <p>On NONE: no-op.
   *
   * <p>Best-effort — silently ignores any errors.
   */
  public static void emptyCache() {
    switch (detectBackend()) {
      case CUDA -> callNoArgMethod("torch.cuda", "empty_cache");
      case MPS -> callNoArgMethod("torch.mps", "empty_cache");
      case NONE -> {}
    }
  }

  /**
   * Waits for all in-flight GPU operations to complete.
   *
   * <p>Calls {@code torch.cuda.synchronize()} (or the MPS equivalent).
   * This is a full GPU pipeline stall — it blocks until every queued kernel,
   * memcpy, and event completes.
   *
   * <p>Must be called <em>before</em> operations that modify the memory
   * allocator state (like {@code CuMemAllocator.sleep()}) to ensure no
   * pending CUDA captures or events reference memory that is about to be
   * unmapped.
   *
   * <p>Best-effort — silently ignores errors.
   */
  public static void callSynchronize() {
    switch (detectBackend()) {
      case CUDA -> callNoArgMethod("torch.cuda", "synchronize");
      case MPS -> callNoArgMethod("torch.mps", "synchronize");
      case NONE -> {}
    }
  }

  /**
   * Waits for all in-flight GPU operations to complete, then flushes
   * the caching allocator.
   *
   * <p>On CUDA: calls {@code torch.cuda.synchronize()} followed by
   * {@code torch.cuda.empty_cache()}. The synchronize ensures all
   * asynchronous kernels and memcpys finish before we release memory
   * blocks — without it, {@code empty_cache()} may skip blocks still
   * referenced by pending operations.
   * <p>On MPS: calls {@code torch.mps.synchronize()} followed by
   * {@code torch.mps.empty_cache()}.
   * <p>On CPU/NONE: no-op.
   *
   * <p>Best-effort — silently ignores any errors.
   */
  public static void synchronizeAndEmptyCache() {
    switch (detectBackend()) {
      case CUDA -> {
        System.out.println(
          "[vLLM4j] synchronizeAndEmptyCache: torch.cuda.synchronize()"
        );
        System.out.flush();
        callNoArgMethod("torch.cuda", "synchronize");
        System.out.println(
          "[vLLM4j] synchronizeAndEmptyCache: torch.cuda.empty_cache()"
        );
        System.out.flush();
        callNoArgMethod("torch.cuda", "empty_cache");
        System.out.println("[vLLM4j] synchronizeAndEmptyCache: done");
        System.out.flush();
      }
      case MPS -> {
        System.out.println(
          "[vLLM4j] synchronizeAndEmptyCache: torch.mps.synchronize()"
        );
        System.out.flush();
        callNoArgMethod("torch.mps", "synchronize");
        System.out.println(
          "[vLLM4j] synchronizeAndEmptyCache: torch.mps.empty_cache()"
        );
        System.out.flush();
        callNoArgMethod("torch.mps", "empty_cache");
        System.out.println("[vLLM4j] synchronizeAndEmptyCache: done");
        System.out.flush();
      }
      case NONE -> {}
    }
  }

  /**
   * Performs aggressive memory cleanup suitable for calling after each inference request.
   *
   * <p>Performs multiple passes of:
   * <ol>
   *   <li>GPU synchronization and cache flushing (via {@link #synchronizeAndEmptyCache()})</li>
   *   <li>Python garbage collection (multiple passes to break circular references)</li>
   * </ol>
   *
   * <p>This method does NOT restart the engine or release model weights. It only
   * cleans up temporary allocations and PyTorch-cached memory blocks.
   *
   * <p>Best-effort — silently ignores any errors.
   *
   * @see #synchronizeAndEmptyCache()
   */
  public static void aggressiveCacheCleanup() {
    synchronizeAndEmptyCache();
    forceGarbageCollection();
  }

  /**
   * Forces Python garbage collection through multiple passes.
   *
   * <p>Runs {@code gc.collect()} multiple times (typically 3-5 iterations) to
   * break circular references in the Python object graph that prevent
   * automatic deallocation via reference counting alone. This is particularly
   * effective for vLLM's scheduler ↔ model_runner ↔ cache_engine cycles.
   *
   * <p>Best-effort — silently ignores any errors.
   */
  private static void forceGarbageCollection() {
    for (int i = 0; i < 5; i++) {
      callNoArgMethod("gc", "collect");
    }
  }

  // ── CUDA backend ────────────────────────────────────────────────────────

  private static GpuMemoryInfo queryCuda() {
    try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
      MemorySegment torchCuda = importModule(arena, "torch.cuda");
      if (torchCuda == null) return null;
      try {
        MemorySegment memGetInfo = PythonTypes.getAttr(
          arena,
          torchCuda,
          "mem_get_info"
        );
        if (PythonTypes.isNull(memGetInfo)) {
          CPythonBinding.PyErr_Clear();
          return null;
        }
        try {
          MemorySegment deviceArg = CPythonBinding.PyLong_FromLong(0L);
          MemorySegment result = PythonCall.callOneArg(memGetInfo, deviceArg);
          PythonTypes.decref(deviceArg);
          if (PythonTypes.isNull(result)) {
            CPythonBinding.PyErr_Clear();
            return null;
          }
          try {
            MemorySegment pyFree = CPythonBinding.PySequence_GetItem(result, 0);
            MemorySegment pyTotal = CPythonBinding.PySequence_GetItem(
              result,
              1
            );
            long free = CPythonBinding.PyLong_AsLong(pyFree);
            long total = CPythonBinding.PyLong_AsLong(pyTotal);
            PythonTypes.decref(pyFree);
            PythonTypes.decref(pyTotal);
            return new GpuMemoryInfo(free, total);
          } finally {
            PythonTypes.decref(result);
          }
        } finally {
          PythonTypes.decref(memGetInfo);
        }
      } finally {
        PythonTypes.decref(torchCuda);
      }
    } catch (Exception e) {
      return null;
    }
  }

  // ── MPS backend (Apple Silicon) ─────────────────────────────────────────

  /**
   * Queries MPS memory on Apple Silicon.
   *
   * <p>Apple's MPS uses unified memory — the GPU shares physical RAM with
   * the CPU. Total memory is queried via {@code os.sysconf('SC_PAGE_SIZE') *
   * os.sysconf('SC_PHYS_PAGES')}. Current GPU allocation is read from
   * {@code torch.mps.current_allocated_memory()}.
   */
  private static GpuMemoryInfo queryMps() {
    try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
      // Total physical memory via os.sysconf
      long totalBytes = querySystemMemory(arena);
      if (totalBytes <= 0) return null;

      // Current MPS allocation
      MemorySegment torchMps = importModule(arena, "torch.mps");
      if (torchMps == null) return null;
      try {
        MemorySegment fnName = PythonTypes.pyStr(
          arena,
          "current_allocated_memory"
        );
        MemorySegment pyAllocated = PythonCall.callMethodObjArgs(
          torchMps,
          fnName
        );
        PythonTypes.decref(fnName);
        if (PythonTypes.isNull(pyAllocated)) {
          CPythonBinding.PyErr_Clear();
          return null;
        }
        long allocated = CPythonBinding.PyLong_AsLong(pyAllocated);
        PythonTypes.decref(pyAllocated);
        return new GpuMemoryInfo(totalBytes - allocated, totalBytes);
      } finally {
        PythonTypes.decref(torchMps);
      }
    } catch (Exception e) {
      return null;
    }
  }

  /**
   * Reads total physical RAM via {@code os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')}.
   */
  private static long querySystemMemory(Arena arena) {
    MemorySegment osModule = importModule(arena, "os");
    if (osModule == null) return -1;
    try {
      MemorySegment sysconfName = PythonTypes.pyStr(arena, "sysconf");

      MemorySegment pyPageSizeKey = PythonTypes.pyStr(arena, "SC_PAGE_SIZE");
      MemorySegment pyPageSize = PythonCall.callMethodObjArgs(
        osModule,
        sysconfName,
        pyPageSizeKey
      );
      PythonTypes.decref(pyPageSizeKey);
      if (PythonTypes.isNull(pyPageSize)) {
        PythonTypes.decref(sysconfName);
        CPythonBinding.PyErr_Clear();
        return -1;
      }
      long pageSize = CPythonBinding.PyLong_AsLong(pyPageSize);
      PythonTypes.decref(pyPageSize);

      MemorySegment pyPhysPagesKey = PythonTypes.pyStr(arena, "SC_PHYS_PAGES");
      MemorySegment pyPhysPages = PythonCall.callMethodObjArgs(
        osModule,
        sysconfName,
        pyPhysPagesKey
      );
      PythonTypes.decref(pyPhysPagesKey);
      PythonTypes.decref(sysconfName);
      if (PythonTypes.isNull(pyPhysPages)) {
        CPythonBinding.PyErr_Clear();
        return -1;
      }
      long physPages = CPythonBinding.PyLong_AsLong(pyPhysPages);
      PythonTypes.decref(pyPhysPages);

      return pageSize * physPages;
    } finally {
      PythonTypes.decref(osModule);
    }
  }

  // ── Shared helpers ──────────────────────────────────────────────────────

  /**
   * Imports a Python module, returning a <em>new reference</em>.
   * Returns {@code null} (never throws) if the import fails.
   *
   * <p>{@code PyImport_ImportModule} is backed by {@code sys.modules}, so
   * repeated imports of the same module are effectively a dict lookup — no
   * need to cache the result on the Java side.
   */
  private static MemorySegment importModule(Arena arena, String moduleName) {
    MemorySegment module = CPythonBinding.PyImport_ImportModule(
      arena.allocateFrom(moduleName)
    );
    if (PythonTypes.isNull(module)) {
      CPythonBinding.PyErr_Clear();
      return null;
    }
    return module;
  }

  /**
   * Checks {@code <module>.is_available()} for a torch backend module.
   * Must be called with the GIL held.
   */
  private static boolean checkTorchBackend(Arena arena, String moduleName) {
    MemorySegment module = importModule(arena, moduleName);
    if (module == null) return false;
    try {
      MemorySegment fnName = PythonTypes.pyStr(arena, "is_available");
      MemorySegment pyResult = PythonCall.callMethodObjArgs(module, fnName);
      PythonTypes.decref(fnName);
      boolean available =
        !PythonTypes.isNull(pyResult) &&
        CPythonBinding.PyObject_IsTrue(pyResult) != 0;
      PythonTypes.decref(pyResult);
      return available;
    } finally {
      PythonTypes.decref(module);
    }
  }

  /**
   * Calls a zero-argument method on a Python module.
   *
   * <p>Imports the module fresh each time (backed by {@code sys.modules},
   * so this is effectively a dict lookup). This avoids the dangling-pointer
   * issues of caching {@link MemorySegment} references to Python modules
   * in static fields.
   *
   * <p>Best-effort — silently ignores errors.
   */
  private static void callNoArgMethod(String moduleName, String methodName) {
    try (var gil = GIL.acquire(); Arena tmp = Arena.ofConfined()) {
      MemorySegment module = importModule(tmp, moduleName);
      if (module == null) return;
      try {
        MemorySegment fnName = PythonTypes.pyStr(tmp, methodName);
        MemorySegment result = PythonCall.callMethodObjArgs(module, fnName);
        PythonTypes.decref(result);
        PythonTypes.decref(fnName);
      } finally {
        PythonTypes.decref(module);
      }
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
    }
  }

  /**
   * Raw GPU memory info.
   *
   * @param freeBytes  free GPU memory in bytes
   * @param totalBytes total GPU memory in bytes
   */
  public record GpuMemoryInfo(long freeBytes, long totalBytes) {}
}
