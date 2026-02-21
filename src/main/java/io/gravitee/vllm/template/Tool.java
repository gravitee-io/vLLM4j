package io.gravitee.vllm.template;

import java.util.Map;

/**
 * OpenAI-compatible tool definition for chat template rendering.
 *
 * <p>When passed to {@link ChatTemplate#render}, tools are serialized
 * to Python dicts matching the OpenAI format and provided as a
 * {@code tools} variable to the Jinja2 template. The model's template
 * handles formatting (Hermes-style, Llama-style, etc.).
 *
 * @param type     always {@code "function"}
 * @param function the function definition
 */
public record Tool(String type, ToolFunction function) {

    /**
     * Convenience factory for a function tool.
     *
     * @param name        function name
     * @param description what the function does
     * @param parameters  JSON Schema object describing the function's parameters
     */
    public static Tool function(String name, String description, Map<String, Object> parameters) {
        return new Tool("function", new ToolFunction(name, description, parameters));
    }
}
