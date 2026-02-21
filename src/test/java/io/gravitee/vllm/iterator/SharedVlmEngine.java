package io.gravitee.vllm.iterator;

import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.template.ChatTemplate;

/**
 * Lazily initializes a single VLM {@link VllmEngine} for the current JVM fork.
 *
 * <p>Loads {@code HuggingFaceTB/SmolVLM-256M-Instruct} (256M-parameter
 * vision-language model — tiny footprint, ideal for CI/test environments).
 * Requires CUDA (Linux).
 *
 * <p>With {@code reuseForks=false}, each test class gets its own JVM fork.
 * Call {@link #close()} in {@code @AfterAll} so the {@code _exit(0)} hook
 * reclaims all GPU memory for the next fork.
 *
 * <p>Run with:
 * <pre>{@code
 * mvn test -P vlm-integration,linux-x86_64,cuda
 * }</pre>
 */
final class SharedVlmEngine {

    private static volatile VllmEngine engine;
    private static volatile ChatTemplate chatTemplate;

    private SharedVlmEngine() {}

    static VllmEngine engine() {
        if (engine == null) {
            synchronized (SharedVlmEngine.class) {
                if (engine == null) {
                    engine = VllmEngine.builder()
                            .model("HuggingFaceTB/SmolVLM-256M-Instruct")
                            .dtype("auto")
                            .enforceEager(true)
                            .maxModelLen(2048)
                            .maxNumSeqs(2)
                            .gpuMemoryUtilization(0.50)
                            .build();
                }
            }
        }
        return engine;
    }

    static ChatTemplate chatTemplate() {
        if (chatTemplate == null) {
            synchronized (SharedVlmEngine.class) {
                if (chatTemplate == null) {
                    chatTemplate = new ChatTemplate(engine());
                }
            }
        }
        return chatTemplate;
    }

    static void close() {
        if (engine != null) {
            engine.close();
        }
    }
}
