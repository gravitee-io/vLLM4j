package io.gravitee.vllm.iterator;

import io.gravitee.vllm.engine.MultiModalData;
import io.gravitee.vllm.engine.RequestOutput;
import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.state.ConversationState;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for vision-language model (VLM) inference via
 * {@link VllmEngine} backed by CPython + vLLM + HuggingFaceTB/SmolVLM-256M-Instruct.
 *
 * <p>Tagged {@code "vlm-integration"} and excluded from the default
 * {@code mvn test} run. Requires <b>CUDA (Linux)</b> — vllm-metal does
 * not yet forward image data to the vision encoder.
 *
 * <p>Execute with:
 * <pre>{@code
 * mvn test -P vlm-integration,linux-x86_64,cuda
 * }</pre>
 *
 * <p>The engine is created per JVM fork via {@link SharedVlmEngine} and
 * closed in {@code @AfterAll}. With {@code reuseForks=false}, GPU memory
 * is fully reclaimed between test classes.
 */
@Tag("vlm-integration")
class VlmIteratorTest {

    private static VllmEngine engine;
    private static ChatTemplate chatTemplate;

    @BeforeAll
    static void initEngine() {
        engine = SharedVlmEngine.engine();
        chatTemplate = SharedVlmEngine.chatTemplate();
    }

    @AfterAll
    static void closeEngine() {
        SharedVlmEngine.close();
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════════════

    private static byte[] loadTestImage() throws IOException {
        try (var is = VlmIteratorTest.class.getResourceAsStream("/dog.jpg")) {
            if (is == null) {
                throw new IOException("Test resource /dog.jpg not found on classpath");
            }
            return is.readAllBytes();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Single image — model identifies a dog
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void vlm_describes_dog_image() throws IOException {
        byte[] imageBytes = loadTestImage();

        // Build multimodal content parts (OpenAI format)
        var contentParts = List.<Map<String, Object>>of(
                Map.of("type", "image"),
                Map.of("type", "text", "text", "What animal is in this picture? Answer in one word.")
        );

        String prompt = chatTemplate.render(List.of(
                ChatMessage.system("You are a helpful assistant. /no_think"),
                ChatMessage.userWithParts("What animal is in this picture?", contentParts)
        ), true);

        var mmData = new MultiModalData().addImage(imageBytes);

        try (var sp = new SamplingParams().temperature(0.0).maxTokens(32)) {
            var request = new VllmRequest("req-vlm-dog", prompt, sp, mmData);

            RequestOutput output = engine.generate(request);

            assertThat(output).isNotNull();
            assertThat(output.finished()).isTrue();
            assertThat(output.outputs()).isNotEmpty();

            String text = output.outputs().getFirst().text().toLowerCase();

            System.out.println("\n=== VLM Dog Image Result ===");
            System.out.println("Output: " + output.outputs().getFirst().text());
            System.out.println("Prompt tokens: " + output.numPromptTokens());
            System.out.println("Generated tokens: " + output.numGeneratedTokens());
            System.out.println("============================");

            assertThat(text).matches(".*\\b(dog|puppy|spaniel|canine)\\b.*");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Streaming VLM — deltas concatenate to full text
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void vlm_streaming_deltas_concatenate() throws IOException {
        byte[] imageBytes = loadTestImage();

        var contentParts = List.<Map<String, Object>>of(
                Map.of("type", "image"),
                Map.of("type", "text", "text", "Describe this image briefly.")
        );

        String prompt = chatTemplate.render(List.of(
                ChatMessage.system("You are a helpful assistant. /no_think"),
                ChatMessage.userWithParts("Describe this image briefly.", contentParts)
        ), true);

        var mmData = new MultiModalData().addImage(imageBytes);

        try (var sp = new SamplingParams().temperature(0.0).maxTokens(64)) {
            var request = new VllmRequest("req-vlm-stream", prompt, sp, mmData);
            var state = new ConversationState()
                    .reasoning("<think>", "</think>");

            var iterator = new VllmIterator(engine);
            iterator.addRequest(request, state);

            var allOutputs = iterator.stream().toList();

            // Deltas should concatenate to final text
            String fromDeltas = allOutputs.stream()
                    .map(VllmOutput::delta)
                    .collect(java.util.stream.Collectors.joining());
            String finalText = allOutputs.getLast().text();

            System.out.println("\n=== VLM Streaming Result ===");
            System.out.println("Output: " + finalText);
            System.out.println("Steps: " + allOutputs.size());
            System.out.println("Input tokens: " + state.inputTokens());
            System.out.println("Total output: " + state.totalOutputTokens());
            System.out.println("Finish reason: " + state.finishReason());
            System.out.println("============================");

            assertThat(fromDeltas).isEqualTo(finalText);
            assertThat(state.inputTokens()).isGreaterThan(0);
            assertThat(state.totalOutputTokens()).isGreaterThan(0);
            assertThat(state.finishReason()).isNotNull();
        }
    }
}
