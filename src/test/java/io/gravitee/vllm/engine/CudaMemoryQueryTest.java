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

import io.gravitee.vllm.engine.CudaMemoryQuery.CudaMemoryInfo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class CudaMemoryQueryTest {

    @Test
    @DisplayName("CudaMemoryInfo record: accessors return correct values")
    void cudaMemoryInfo_accessors() {
        CudaMemoryInfo info = new CudaMemoryInfo(8_000_000_000L, 24_000_000_000L);
        assertThat(info.freeBytes()).isEqualTo(8_000_000_000L);
        assertThat(info.totalBytes()).isEqualTo(24_000_000_000L);
    }

    @Test
    @DisplayName("CudaMemoryInfo record: equality and hashCode")
    void cudaMemoryInfo_equality() {
        CudaMemoryInfo a = new CudaMemoryInfo(1024, 4096);
        CudaMemoryInfo b = new CudaMemoryInfo(1024, 4096);
        CudaMemoryInfo c = new CudaMemoryInfo(2048, 4096);

        assertThat(a).isEqualTo(b);
        assertThat(a).hasSameHashCodeAs(b);
        assertThat(a).isNotEqualTo(c);
    }

    @Test
    @DisplayName("CudaMemoryInfo record: toString includes field values")
    void cudaMemoryInfo_toString() {
        CudaMemoryInfo info = new CudaMemoryInfo(100, 200);
        assertThat(info.toString()).contains("100");
        assertThat(info.toString()).contains("200");
    }

    @Test
    @DisplayName("CudaMemoryInfo record: zero values are valid")
    void cudaMemoryInfo_zero_values() {
        CudaMemoryInfo info = new CudaMemoryInfo(0, 0);
        assertThat(info.freeBytes()).isEqualTo(0);
        assertThat(info.totalBytes()).isEqualTo(0);
    }

    @Test
    @DisplayName("CudaMemoryInfo record: large values (>32-bit) work correctly")
    void cudaMemoryInfo_large_values() {
        long free = 81_604_378_624L;  // ~76 GiB (A100 80GB typical free)
        long total = 85_899_345_920L; // ~80 GiB
        CudaMemoryInfo info = new CudaMemoryInfo(free, total);
        assertThat(info.freeBytes()).isEqualTo(free);
        assertThat(info.totalBytes()).isEqualTo(total);
    }
}
