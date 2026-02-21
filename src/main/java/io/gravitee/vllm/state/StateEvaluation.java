package io.gravitee.vllm.state;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Finite state machine that detects tag boundaries in generated text
 * and transitions between {@link GenerationState}s.
 *
 * <p>Mirrors llamaj.cpp's {@code StateEvaluation} module. The FSM operates
 * on text deltas (substrings) accumulated into a buffer, detecting open/close
 * tags across token boundaries.
 *
 * <h2>Transition rules</h2>
 * <ul>
 *   <li>From {@code ANSWER}: if the accumulated text contains any configured
 *       {@code openTag}, transition to that state.</li>
 *   <li>From {@code REASONING} or {@code TOOLS}: if the accumulated text
 *       contains the matching {@code closeTag}, transition back to {@code ANSWER}.</li>
 * </ul>
 *
 * <h2>Re-entry rules</h2>
 * <ul>
 *   <li>{@code REASONING} can occur at most once per generation.</li>
 *   <li>{@code TOOLS} can occur multiple times (models may produce
 *       several tool calls).</li>
 * </ul>
 */
public final class StateEvaluation {

    private Map<GenerationState, TagBounds> tagsByState;
    private Map<GenerationState, Boolean> occurred;
    private final StringBuilder buffer = new StringBuilder();

    /**
     * Initializes the FSM with the given tag configurations.
     * Must be called before {@link #evaluate}.
     */
    public void initialize(List<TagBounds> tags) {
        tagsByState = new EnumMap<>(GenerationState.class);
        occurred = new EnumMap<>(GenerationState.class);
        for (TagBounds tb : tags) {
            tagsByState.put(tb.state(), tb);
            occurred.put(tb.state(), false);
        }
        buffer.setLength(0);
    }

    /** Whether this FSM has been initialized with at least one tag config. */
    public boolean isInitialized() {
        return tagsByState != null && !tagsByState.isEmpty();
    }

    /**
     * Feeds a text delta into the FSM and returns the new generation state.
     *
     * @param currentState the current state before this delta
     * @param delta        the new text fragment
     * @return the state after evaluating this delta
     */
    public GenerationState evaluate(GenerationState currentState, String delta) {
        if (!isInitialized() || delta == null || delta.isEmpty()) {
            return currentState != null ? currentState : GenerationState.ANSWER;
        }

        buffer.append(delta);

        return switch (currentState) {
            case ANSWER -> detectOpenTag();
            case REASONING, TOOLS -> detectCloseTag(currentState);
            case null -> GenerationState.ANSWER;
        };
    }

    /**
     * Resets the internal text buffer. Call this when starting a new
     * generation turn.
     */
    public void reset() {
        buffer.setLength(0);
        if (occurred != null) {
            occurred.replaceAll((k, v) -> false);
        }
    }

    // ── Internal ────────────────────────────────────────────────────────

    private GenerationState detectOpenTag() {
        String text = buffer.toString();
        for (var entry : tagsByState.entrySet()) {
            GenerationState state = entry.getKey();
            TagBounds bounds = entry.getValue();

            // Skip states that already occurred (except TOOLS which can repeat)
            if (state != GenerationState.TOOLS && Boolean.TRUE.equals(occurred.get(state))) {
                continue;
            }

            int idx = text.lastIndexOf(bounds.openTag());
            if (idx >= 0) {
                // Transition — clear buffer up to after the open tag
                buffer.setLength(0);
                return state;
            }
        }
        return GenerationState.ANSWER;
    }

    private GenerationState detectCloseTag(GenerationState currentState) {
        TagBounds bounds = tagsByState.get(currentState);
        if (bounds == null) return GenerationState.ANSWER;

        String text = buffer.toString();
        int idx = text.lastIndexOf(bounds.closeTag());
        if (idx >= 0) {
            // Mark occurred (REASONING once, TOOLS can repeat)
            if (currentState != GenerationState.TOOLS) {
                occurred.put(currentState, true);
            }
            buffer.setLength(0);
            return GenerationState.ANSWER;
        }
        return currentState;
    }
}
