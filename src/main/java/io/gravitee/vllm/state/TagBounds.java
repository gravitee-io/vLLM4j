package io.gravitee.vllm.state;

/**
 * Defines the open/close tag boundaries for a {@link GenerationState}.
 *
 * <p>When the FSM detects {@code openTag} in the generated text, it
 * transitions into the associated state. When it detects {@code closeTag},
 * it transitions back to {@link GenerationState#ANSWER}.
 *
 * @param state    the generation state this tag pair activates
 * @param openTag  the opening marker (e.g. {@code "<think>"})
 * @param closeTag the closing marker (e.g. {@code "</think>"})
 */
public record TagBounds(GenerationState state, String openTag, String closeTag) {

    public TagBounds {
        if (state == null) throw new IllegalArgumentException("state must not be null");
        if (openTag == null || openTag.isEmpty()) throw new IllegalArgumentException("openTag must not be empty");
        if (closeTag == null || closeTag.isEmpty()) throw new IllegalArgumentException("closeTag must not be empty");
    }
}
