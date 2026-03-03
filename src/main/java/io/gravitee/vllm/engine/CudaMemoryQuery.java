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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.vllm.python.CPython;

/**
 * Queries CUDA GPU memory via {@code torch.cuda.mem_get_info(0)} through the
 * live CPython interpreter. Returns raw {@code (freeBytes, totalBytes)} with
 * no external dependencies.
 *
 * <p>This class belongs in the vLLM4j binding layer because it requires the
 * CPython FFM infrastructure ({@link GIL}, {@link CPython}, {@link PythonTypes}).
 * Higher-level estimation logic that depends on {@code MemoryEstimate} or
 * {@code SafetensorsInfo} lives in the adapter modules.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class CudaMemoryQuery {

    private CudaMemoryQuery() {}

    /**
     * Queries CUDA device 0 for free and total memory.
     *
     * @return a {@link CudaMemoryInfo} with {@code (freeBytes, totalBytes)},
     *         or {@code null} if CUDA is unavailable, CPython is not initialised,
     *         or any error occurs. Never throws.
     */
    public static CudaMemoryInfo query() {
        try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
            MemorySegment torchCuda = CPython.PyImport_ImportModule(arena.allocateFrom("torch.cuda"));
            if (PythonTypes.isNull(torchCuda)) {
                CPython.PyErr_Clear();
                return null;
            }
            try {
                MemorySegment memGetInfo = PythonTypes.getAttr(arena, torchCuda, "mem_get_info");
                if (PythonTypes.isNull(memGetInfo)) {
                    CPython.PyErr_Clear();
                    return null;
                }
                try {
                    MemorySegment deviceArg = CPython.PyLong_FromLong(0L);
                    MemorySegment result = PythonCall.callOneArg(memGetInfo, deviceArg);
                    PythonTypes.decref(deviceArg);

                    if (PythonTypes.isNull(result)) {
                        CPython.PyErr_Clear();
                        return null;
                    }
                    try {
                        MemorySegment pyFree = CPython.PySequence_GetItem(result, 0);
                        MemorySegment pyTotal = CPython.PySequence_GetItem(result, 1);
                        long free = CPython.PyLong_AsLong(pyFree);
                        long total = CPython.PyLong_AsLong(pyTotal);
                        PythonTypes.decref(pyFree);
                        PythonTypes.decref(pyTotal);
                        return new CudaMemoryInfo(free, total);
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

    /**
     * Raw CUDA memory info for device 0.
     *
     * @param freeBytes  free GPU memory in bytes
     * @param totalBytes total GPU memory in bytes
     */
    public record CudaMemoryInfo(long freeBytes, long totalBytes) {}
}
