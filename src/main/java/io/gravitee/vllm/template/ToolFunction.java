package io.gravitee.vllm.template;

import java.util.Map;

/**
 * A function definition within a {@link Tool}.
 *
 * @param name        the function name the model should call
 * @param description a natural-language description of what the function does
 * @param parameters  JSON Schema object describing the function's parameters,
 *                    or {@code null} if the function takes no parameters
 */
public record ToolFunction(
        String name,
        String description,
        Map<String, Object> parameters) {}
