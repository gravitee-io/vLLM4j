package io.gravitee.vllm.engine;

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;

import org.vllm.python.CPython;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Guided decoding parameters that constrain generation output to match
 * a specific structure: JSON schema, regex, choice, or grammar.
 *
 * <p>Wraps {@code vllm.sampling_params.GuidedDecodingParams}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Force JSON schema output
 * var guided = GuidedDecodingParams.json("""
 *     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
 *     """);
 *
 * try (var sp = new SamplingParams()
 *         .temperature(0.0)
 *         .maxTokens(128)
 *         .guidedDecoding(guided)) {
 *     // ...
 * }
 * }</pre>
 */
public final class GuidedDecodingParams {

    private final String type;
    private final String value;
    private final List<String> choices;

    private GuidedDecodingParams(String type, String value, List<String> choices) {
        this.type = type;
        this.value = value;
        this.choices = choices;
    }

    /**
     * Constrains output to conform to a JSON schema.
     *
     * @param jsonSchema a valid JSON Schema string
     * @return guided decoding params
     */
    public static GuidedDecodingParams json(String jsonSchema) {
        return new GuidedDecodingParams("json", jsonSchema, null);
    }

    /**
     * Constrains output to match a regular expression.
     *
     * @param regex the regular expression pattern
     * @return guided decoding params
     */
    public static GuidedDecodingParams regex(String regex) {
        return new GuidedDecodingParams("regex", regex, null);
    }

    /**
     * Constrains output to be exactly one of the given choices.
     *
     * @param choices the allowed output strings
     * @return guided decoding params
     */
    public static GuidedDecodingParams choice(List<String> choices) {
        return new GuidedDecodingParams("choice", null, choices);
    }

    /**
     * Constrains output to conform to a context-free grammar (EBNF).
     *
     * @param grammar the EBNF grammar string
     * @return guided decoding params
     */
    public static GuidedDecodingParams grammar(String grammar) {
        return new GuidedDecodingParams("grammar", grammar, null);
    }

    /**
     * Constrains output to be any valid JSON object.
     *
     * @return guided decoding params
     */
    public static GuidedDecodingParams jsonObject() {
        return new GuidedDecodingParams("json_object", null, null);
    }

    /**
     * Builds the Python {@code GuidedDecodingParams} object.
     *
     * @param arena arena for native allocations
     * @return the Python object (new reference)
     */
    MemorySegment toPython(Arena arena) {
        MemorySegment kwargs = CPython.PyDict_New();

        switch (type) {
            case "json" -> {
                MemorySegment pySchema = PythonTypes.pyStr(arena, value);
                PythonTypes.putDictObj(arena, kwargs, "json", pySchema);
                PythonTypes.decref(pySchema);
            }
            case "regex" -> {
                MemorySegment pyRegex = PythonTypes.pyStr(arena, value);
                PythonTypes.putDictObj(arena, kwargs, "regex", pyRegex);
                PythonTypes.decref(pyRegex);
            }
            case "choice" -> {
                MemorySegment pyList = CPython.PyList_New(0);
                for (String c : choices) {
                    MemorySegment pyItem = PythonTypes.pyStr(arena, c);
                    CPython.PyList_Append(pyList, pyItem);
                    PythonTypes.decref(pyItem);
                }
                PythonTypes.putDictObj(arena, kwargs, "choice", pyList);
                PythonTypes.decref(pyList);
            }
            case "grammar" -> {
                MemorySegment pyGrammar = PythonTypes.pyStr(arena, value);
                PythonTypes.putDictObj(arena, kwargs, "grammar", pyGrammar);
                PythonTypes.decref(pyGrammar);
            }
            case "json_object" -> {
                PythonTypes.putDictObj(arena, kwargs, "json_object", PythonTypes.pyTrue());
            }
        }

        MemorySegment guidedClass = PythonCall.importClass(
                arena, "vllm.sampling_params", "GuidedDecodingParams");
        MemorySegment result = PythonCall.callWithKwargs(guidedClass, kwargs);
        PythonErrors.checkPythonError("GuidedDecodingParams construction");
        PythonTypes.decref(kwargs);
        PythonTypes.decref(guidedClass);

        return result;
    }
}
