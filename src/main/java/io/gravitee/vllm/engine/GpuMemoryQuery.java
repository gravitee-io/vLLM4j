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

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.runtime.GIL;
import io.gravitee.vllm.runtime.PythonRuntime;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.vllm.python.CPython;

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
 * CPython FFM infrastructure ({@link GIL}, {@link CPython}, {@link PythonTypes}).
 * Higher-level estimation logic that depends on {@code MemoryEstimate} or
 * {@code SafetensorsInfo} lives in the adapter modules.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class GpuMemoryQuery {

    /** The GPU compute backends we support. */
    public enum GpuBackend { CUDA, MPS, NONE }

    private static volatile GpuBackend cachedBackend;

    // Cached Python module references (borrowed from sys.modules after import)
    private static volatile MemorySegment cachedTorchCuda;
    private static volatile MemorySegment cachedTorchMps;
    private static volatile MemorySegment cachedOsModule;

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
     * @return a {@link GpuMemoryInfo} with {@code (freeBytes, totalBytes)},
     *         or {@code null} if no GPU backend is available or any error occurs.
     *         Never throws.
     */
    public static GpuMemoryInfo query() {
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
            case CUDA -> callEmptyCache("torch.cuda");
            case MPS -> callEmptyCache("torch.mps");
            case NONE -> {}
        }
    }

    // ── CUDA backend ────────────────────────────────────────────────────────

    private static GpuMemoryInfo queryCuda() {
        try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
            MemorySegment torchCuda = cachedTorchCuda;
            if (torchCuda == null) {
                torchCuda = CPython.PyImport_ImportModule(arena.allocateFrom("torch.cuda"));
                if (PythonTypes.isNull(torchCuda)) { CPython.PyErr_Clear(); return null; }
                cachedTorchCuda = torchCuda;
            }
            MemorySegment memGetInfo = PythonTypes.getAttr(arena, torchCuda, "mem_get_info");
            if (PythonTypes.isNull(memGetInfo)) { CPython.PyErr_Clear(); return null; }
            try {
                MemorySegment deviceArg = CPython.PyLong_FromLong(0L);
                MemorySegment result = PythonCall.callOneArg(memGetInfo, deviceArg);
                PythonTypes.decref(deviceArg);
                if (PythonTypes.isNull(result)) { CPython.PyErr_Clear(); return null; }
                try {
                    MemorySegment pyFree = CPython.PySequence_GetItem(result, 0);
                    MemorySegment pyTotal = CPython.PySequence_GetItem(result, 1);
                    long free = CPython.PyLong_AsLong(pyFree);
                    long total = CPython.PyLong_AsLong(pyTotal);
                    PythonTypes.decref(pyFree);
                    PythonTypes.decref(pyTotal);
                    return new GpuMemoryInfo(free, total);
                } finally {
                    PythonTypes.decref(result);
                }
            } finally {
                PythonTypes.decref(memGetInfo);
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
            MemorySegment torchMps = cachedTorchMps;
            if (torchMps == null) {
                torchMps = CPython.PyImport_ImportModule(arena.allocateFrom("torch.mps"));
                if (PythonTypes.isNull(torchMps)) { CPython.PyErr_Clear(); return null; }
                cachedTorchMps = torchMps;
            }
            MemorySegment fnName = PythonTypes.pyStr(arena, "current_allocated_memory");
            MemorySegment pyAllocated = PythonCall.callMethodObjArgs(torchMps, fnName);
            PythonTypes.decref(fnName);
            if (PythonTypes.isNull(pyAllocated)) { CPython.PyErr_Clear(); return null; }
            long allocated = CPython.PyLong_AsLong(pyAllocated);
            PythonTypes.decref(pyAllocated);
            return new GpuMemoryInfo(totalBytes - allocated, totalBytes);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Reads total physical RAM via {@code os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')}.
     */
    private static long querySystemMemory(Arena arena) {
        MemorySegment osModule = cachedOsModule;
        if (osModule == null) {
            osModule = CPython.PyImport_ImportModule(arena.allocateFrom("os"));
            if (PythonTypes.isNull(osModule)) { CPython.PyErr_Clear(); return -1; }
            cachedOsModule = osModule;
        }
        MemorySegment sysconfName = PythonTypes.pyStr(arena, "sysconf");

        MemorySegment pyPageSizeKey = PythonTypes.pyStr(arena, "SC_PAGE_SIZE");
        MemorySegment pyPageSize = PythonCall.callMethodObjArgs(osModule, sysconfName, pyPageSizeKey);
        PythonTypes.decref(pyPageSizeKey);
        if (PythonTypes.isNull(pyPageSize)) { PythonTypes.decref(sysconfName); CPython.PyErr_Clear(); return -1; }
        long pageSize = CPython.PyLong_AsLong(pyPageSize);
        PythonTypes.decref(pyPageSize);

        MemorySegment pyPhysPagesKey = PythonTypes.pyStr(arena, "SC_PHYS_PAGES");
        MemorySegment pyPhysPages = PythonCall.callMethodObjArgs(osModule, sysconfName, pyPhysPagesKey);
        PythonTypes.decref(pyPhysPagesKey);
        PythonTypes.decref(sysconfName);
        if (PythonTypes.isNull(pyPhysPages)) { CPython.PyErr_Clear(); return -1; }
        long physPages = CPython.PyLong_AsLong(pyPhysPages);
        PythonTypes.decref(pyPhysPages);

        return pageSize * physPages;
    }

    // ── Shared helpers ──────────────────────────────────────────────────────

    /**
     * Checks {@code <module>.is_available()} for a torch backend module.
     * Must be called with the GIL held.
     */
    private static boolean checkTorchBackend(Arena arena, String moduleName) {
        MemorySegment module = CPython.PyImport_ImportModule(arena.allocateFrom(moduleName));
        if (PythonTypes.isNull(module)) { CPython.PyErr_Clear(); return false; }
        try {
            MemorySegment fnName = PythonTypes.pyStr(arena, "is_available");
            MemorySegment pyResult = PythonCall.callMethodObjArgs(module, fnName);
            PythonTypes.decref(fnName);
            boolean available = !PythonTypes.isNull(pyResult)
                    && CPython.PyObject_IsTrue(pyResult) != 0;
            PythonTypes.decref(pyResult);
            return available;
        } finally {
            PythonTypes.decref(module);
        }
    }

    /**
     * Calls {@code <module>.empty_cache()} on the given torch backend module.
     * Best-effort — silently ignores errors.
     */
    private static void callEmptyCache(String moduleName) {
        try (var gil = GIL.acquire(); Arena tmp = Arena.ofConfined()) {
            MemorySegment module;
            if ("torch.cuda".equals(moduleName)) {
                module = cachedTorchCuda;
                if (module == null) {
                    module = CPython.PyImport_ImportModule(tmp.allocateFrom(moduleName));
                    if (PythonTypes.isNull(module)) { CPython.PyErr_Clear(); return; }
                    cachedTorchCuda = module;
                }
            } else {
                module = cachedTorchMps;
                if (module == null) {
                    module = CPython.PyImport_ImportModule(tmp.allocateFrom(moduleName));
                    if (PythonTypes.isNull(module)) { CPython.PyErr_Clear(); return; }
                    cachedTorchMps = module;
                }
            }
            MemorySegment fnName = PythonTypes.pyStr(tmp, "empty_cache");
            MemorySegment result = PythonCall.callMethodObjArgs(module, fnName);
            PythonTypes.decref(result);
            PythonTypes.decref(fnName);
        } catch (Exception e) {
            CPython.PyErr_Clear();
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
