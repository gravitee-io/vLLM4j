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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MultiModalDataTest {

  @Test
  void empty_shouldHaveNoData() {
    var mm = new MultiModalData();
    assertThat(mm.hasData()).isFalse();
    assertThat(mm.imageCount()).isEqualTo(0);
    assertThat(mm.audioCount()).isEqualTo(0);
  }

  @Test
  void addImage_bytes_shouldIncrementCount() {
    var mm = new MultiModalData().addImage(new byte[] { 1, 2, 3 });
    assertThat(mm.imageCount()).isEqualTo(1);
    assertThat(mm.hasData()).isTrue();
  }

  @Test
  void addImage_multipleImages() {
    var mm = new MultiModalData()
      .addImage(new byte[] { 1 })
      .addImage(new byte[] { 2 })
      .addImage(new byte[] { 3 });
    assertThat(mm.imageCount()).isEqualTo(3);
  }

  @Test
  void addImage_fromPath(@TempDir Path tempDir) throws IOException {
    Path imgFile = tempDir.resolve("test.png");
    Files.write(imgFile, new byte[] { (byte) 0x89, 'P', 'N', 'G' });

    var mm = new MultiModalData().addImage(imgFile);
    assertThat(mm.imageCount()).isEqualTo(1);
    assertThat(mm.hasData()).isTrue();
  }

  @Test
  void addImage_nullShouldThrow() {
    assertThatThrownBy(() -> new MultiModalData().addImage((byte[]) null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("imageData");
  }

  @Test
  void addImage_emptyShouldThrow() {
    assertThatThrownBy(() -> new MultiModalData().addImage(new byte[0]))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("imageData");
  }

  @Test
  void addAudio_bytes_shouldIncrementCount() {
    var mm = new MultiModalData().addAudio(new byte[] { 10, 20, 30 });
    assertThat(mm.audioCount()).isEqualTo(1);
    assertThat(mm.hasData()).isTrue();
  }

  @Test
  void addAudio_fromPath(@TempDir Path tempDir) throws IOException {
    Path audioFile = tempDir.resolve("test.wav");
    Files.write(audioFile, new byte[] { 0, 1, 2, 3 });

    var mm = new MultiModalData().addAudio(audioFile);
    assertThat(mm.audioCount()).isEqualTo(1);
  }

  @Test
  void addAudio_nullShouldThrow() {
    assertThatThrownBy(() -> new MultiModalData().addAudio((byte[]) null))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("audioData");
  }

  @Test
  void addAudio_emptyShouldThrow() {
    assertThatThrownBy(() -> new MultiModalData().addAudio(new byte[0]))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("audioData");
  }

  @Test
  void mixed_shouldTrackBoth() {
    var mm = new MultiModalData()
      .addImage(new byte[] { 1, 2 })
      .addAudio(new byte[] { 3, 4 });
    assertThat(mm.imageCount()).isEqualTo(1);
    assertThat(mm.audioCount()).isEqualTo(1);
    assertThat(mm.hasData()).isTrue();
  }

  @Test
  void fluent_shouldReturnSameInstance() {
    var mm = new MultiModalData();
    assertThat(mm.addImage(new byte[] { 1 })).isSameAs(mm);
    assertThat(mm.addAudio(new byte[] { 2 })).isSameAs(mm);
  }
}
