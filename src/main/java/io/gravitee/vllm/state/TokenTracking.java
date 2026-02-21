package io.gravitee.vllm.state;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Token counters categorized by {@link GenerationState}.
 *
 * <p>Mirrors llamaj.cpp's {@code TokenTracking} module. Maintains four
 * atomic counters: input (prompt), answer, reasoning, and tools.
 *
 * <p>Thread-safe — all counters use {@link AtomicInteger}.
 */
public final class TokenTracking {

    private final AtomicInteger input = new AtomicInteger(0);
    private final AtomicInteger answer = new AtomicInteger(0);
    private final AtomicInteger reasoning = new AtomicInteger(0);
    private final AtomicInteger tools = new AtomicInteger(0);

    /**
     * Sets the initial prompt token count.
     */
    public void initialize(int promptTokenCount) {
        input.set(promptTokenCount);
        answer.set(0);
        reasoning.set(0);
        tools.set(0);
    }

    /**
     * Increments the counter for the given generation state.
     *
     * @param state the state category
     * @param count number of tokens to add (may be negative for corrections)
     */
    public void consume(GenerationState state, int count) {
        switch (state) {
            case ANSWER    -> answer.addAndGet(count);
            case REASONING -> reasoning.addAndGet(count);
            case TOOLS     -> tools.addAndGet(count);
        }
    }

    /** Returns the prompt (input) token count. */
    public int inputTokens() {
        return input.get();
    }

    /** Returns the token count for a specific generation state. */
    public int outputTokens(GenerationState state) {
        return switch (state) {
            case ANSWER    -> answer.get();
            case REASONING -> reasoning.get();
            case TOOLS     -> tools.get();
        };
    }

    /** Returns total output tokens across all states. */
    public int totalOutputTokens() {
        return answer.get() + reasoning.get() + tools.get();
    }

    /** Returns total tokens (input + all output). */
    public int totalTokens() {
        return inputTokens() + totalOutputTokens();
    }
}
