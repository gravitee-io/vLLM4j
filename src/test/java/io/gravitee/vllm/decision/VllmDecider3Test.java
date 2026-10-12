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
package io.gravitee.vllm.decision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Runs Decision 3.0 ({@code vllm-sr/d3-edge}) in-process, with text and an image.
 *
 * <p>Skipped on CPU: the model fails to load when its answers miss the golden ones.
 * Usage: {@code mvn test -P decision-integration,macosx-aarch64}, or
 * {@code -Dvllm4j.decision.device=cuda} on Linux
 */
@Tag("decision-integration")
class VllmDecider3Test {

  private static VllmDecider decider;

  @BeforeAll
  static void setUp() {
    boolean mac = System.getProperty("os.name").toLowerCase().contains("mac");
    String device = System.getProperty(
      "vllm4j.decision.device",
      mac ? "mps" : "cpu"
    );
    assumeFalse("cpu".equals(device), "d3 misses its golden answers on CPU");
    decider = VllmDecider.builder()
      .model("vllm-sr/d3-edge")
      .device(device)
      .build();
  }

  @AfterAll
  static void tearDown() {
    if (decider != null) decider.close();
  }

  private static String redPng() throws IOException {
    var image = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
    var g = image.createGraphics();
    g.setColor(new Color(220, 20, 20));
    g.fillRect(0, 0, 256, 256);
    g.dispose();
    var out = new ByteArrayOutputStream();
    ImageIO.write(image, "png", out);
    return (
      "data:image/png;base64," +
      Base64.getEncoder().encodeToString(out.toByteArray())
    );
  }

  @Test
  void answers_text_questions() {
    Map<String, String> intents = new LinkedHashMap<>();
    intents.put("billing", "Payment or invoice issue");
    intents.put("account", "Login or access issue");
    var response = decider.decide(
      DecisionRequest.builder(
        Map.of("request", "I was charged twice for my subscription.")
      )
        .choice("intent", "What does the user want?", intents)
        .noul("upset", "Is the user upset?")
        .build()
    );

    assertThat(response.model()).isEqualTo("d3-edge");
    assertThat(((Answer.Choice) response.answer("intent")).choice()).isEqualTo(
      "billing"
    );
    assertThat(
      ((Answer.Noul) response.answer("upset")).probability()
    ).isBetween(0.0, 1.0);
  }

  @Test
  void reads_an_image() throws IOException {
    Map<String, String> colours = new LinkedHashMap<>();
    colours.put("red", null);
    colours.put("green", null);
    colours.put("blue", null);
    var response = decider.decide(
      DecisionRequest.builder("What colour is the image?")
        .choice("colour", "Which colour dominates the image?", colours)
        .image(redPng())
        .build()
    );

    assertThat(((Answer.Choice) response.answer("colour")).choice()).isEqualTo(
      "red"
    );
  }
}
