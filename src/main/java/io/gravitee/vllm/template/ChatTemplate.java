package io.gravitee.vllm.template;

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.runtime.GIL;

import org.vllm.python.CPython;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Renders Jinja2 chat templates using the in-process CPython interpreter.
 *
 * <p>By default, the template is extracted automatically from the model's
 * tokenizer (via {@link VllmEngine#getChatTemplate()}). An explicit template
 * string can be provided to override the model's default.
 *
 * <h2>Usage — automatic template from model</h2>
 * <pre>{@code
 * var renderer = new ChatTemplate(engine);
 * String prompt = renderer.render(messages, true);
 * }</pre>
 *
 * <h2>Usage — with tools</h2>
 * <pre>{@code
 * var tools = List.of(Tool.function("get_weather", "Get weather", params));
 * String prompt = renderer.render(messages, tools, true);
 * }</pre>
 */
public final class ChatTemplate {

    private final MemorySegment jinja2Module;
    private final Arena arena;
    private final String templateString;

    /**
     * Creates a template renderer using the model's own chat template.
     *
     * <p>Extracts the Jinja2 template from the model's HuggingFace tokenizer
     * via {@link VllmEngine#getChatTemplate()}.
     *
     * @param engine an initialized {@link VllmEngine}
     * @throws IllegalStateException if the model has no chat template
     */
    public ChatTemplate(VllmEngine engine) {
        this.jinja2Module = engine.jinja2Module();
        this.arena = engine.arena();
        this.templateString = engine.getChatTemplate();
        if (this.templateString == null) {
            throw new IllegalStateException(
                    "Model has no chat template. Provide one explicitly via ChatTemplate(engine, templateString).");
        }
    }

    /**
     * Creates a template renderer with an explicit Jinja2 template string,
     * overriding the model's default.
     *
     * @param engine         an initialized {@link VllmEngine}
     * @param templateString the Jinja2 chat template to use
     */
    public ChatTemplate(VllmEngine engine, String templateString) {
        this.jinja2Module = engine.jinja2Module();
        this.arena = engine.arena();
        this.templateString = templateString;
    }

    /**
     * Returns the Jinja2 template string this renderer uses.
     */
    public String templateString() {
        return templateString;
    }

    /**
     * Renders the chat template with the given messages (no tools).
     *
     * @param messages            ordered list of chat messages
     * @param addGenerationPrompt whether to append the generation prompt token(s)
     * @return the rendered string
     */
    public String render(List<ChatMessage> messages, boolean addGenerationPrompt) {
        return renderTemplate(templateString, messages, null, addGenerationPrompt);
    }

    /**
     * Renders the chat template with the given messages and tools.
     *
     * <p>Tools are passed as a {@code tools} variable to the Jinja2 template.
     * The model's template handles formatting (Hermes-style, Llama-style, etc.).
     *
     * @param messages            ordered list of chat messages
     * @param tools               list of available tools (may be {@code null} or empty)
     * @param addGenerationPrompt whether to append the generation prompt token(s)
     * @return the rendered string
     */
    public String render(List<ChatMessage> messages, List<Tool> tools, boolean addGenerationPrompt) {
        return renderTemplate(templateString, messages, tools, addGenerationPrompt);
    }

    /**
     * Renders an explicit Jinja2 template string with the given messages.
     *
     * <p>This overload ignores the stored template and uses the provided one.
     * Useful for one-off rendering with a different template.
     *
     * @param templateString      the raw Jinja2 template string
     * @param messages            ordered list of chat messages
     * @param addGenerationPrompt whether to append the generation prompt token(s)
     * @return the rendered string
     */
    public String render(String templateString, List<ChatMessage> messages, boolean addGenerationPrompt) {
        return renderTemplate(templateString, messages, null, addGenerationPrompt);
    }

    // ── Internal ────────────────────────────────────────────────────────────

    private String renderTemplate(String template, List<ChatMessage> messages,
                                  List<Tool> tools, boolean addGenerationPrompt) {
        try (var gil = GIL.acquire()) {
            // env = jinja2.Environment()
            MemorySegment envClass = PythonTypes.getAttr(arena, jinja2Module, "Environment");
            MemorySegment emptyEnvKwargs = CPython.PyDict_New();
            MemorySegment env = PythonCall.callWithKwargs(envClass, emptyEnvKwargs);
            PythonErrors.checkPythonError("jinja2.Environment()");
            PythonTypes.decref(emptyEnvKwargs);
            PythonTypes.decref(envClass);

            // tmpl = env.from_string(templateString)
            MemorySegment pyTemplate = PythonTypes.pyStr(arena, template);
            MemorySegment fromStringName = PythonTypes.pyStr(arena, "from_string");
            MemorySegment tmpl = PythonCall.callMethodObjArgs(env, fromStringName, pyTemplate);
            PythonErrors.checkPythonError("jinja2 from_string()");
            PythonTypes.decref(fromStringName);
            PythonTypes.decref(pyTemplate);
            PythonTypes.decref(env);

            // Build messages list
            MemorySegment pyMessages = messagesToPyList(messages);

            // kwargs = {"messages": pyMessages, "add_generation_prompt": bool}
            MemorySegment kwargs = CPython.PyDict_New();
            PythonTypes.putDictObj(arena, kwargs, "messages", pyMessages);
            PythonTypes.decref(pyMessages);
            MemorySegment pyBool = CPython.PyBool_FromLong(addGenerationPrompt ? 1L : 0L);
            PythonTypes.putDictObj(arena, kwargs, "add_generation_prompt", pyBool);
            PythonTypes.decref(pyBool);

            // Add tools if provided
            if (tools != null && !tools.isEmpty()) {
                MemorySegment pyTools = toolsToPyList(tools);
                PythonTypes.putDictObj(arena, kwargs, "tools", pyTools);
                PythonTypes.decref(pyTools);
            }

            // rendered = tmpl.render(**kwargs)
            MemorySegment renderFn = PythonTypes.getAttr(arena, tmpl, "render");
            MemorySegment emptyTuple = PythonCall.makeTuple();
            MemorySegment rendered = PythonCall.pyObjectCall(renderFn, emptyTuple, kwargs);
            PythonErrors.checkPythonError("jinja2 template.render()");
            PythonTypes.decref(renderFn);
            PythonTypes.decref(emptyTuple);
            PythonTypes.decref(kwargs);
            PythonTypes.decref(tmpl);

            String result = PythonTypes.pyUnicodeToString(rendered);
            PythonTypes.decref(rendered);
            return result;
        }
    }

    // ── Message serialization ───────────────────────────────────────────────

    /**
     * Converts a list of {@link ChatMessage} to a Python list of dicts.
     *
     * <p>Each dict has at minimum {@code role} and {@code content}. Tool-use
     * messages add {@code tool_calls}, {@code tool_call_id}, and/or {@code name}.
     */
    private MemorySegment messagesToPyList(List<ChatMessage> messages) {
        MemorySegment pyList = CPython.PyList_New(0);

        for (ChatMessage msg : messages) {
            MemorySegment pyDict = CPython.PyDict_New();

            // role (always present)
            MemorySegment pyRole = PythonTypes.pyStr(arena, msg.role());
            CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("role"), pyRole);
            PythonTypes.decref(pyRole);

            // content — either a string or a list of content parts (multimodal)
            if (msg.hasContentParts()) {
                // Multimodal: content is a list of dicts (OpenAI format)
                // e.g. [{"type": "text", "text": "..."}, {"type": "image"}]
                MemorySegment pyContentList = CPython.PyList_New(0);
                for (Map<String, Object> part : msg.contentParts()) {
                    MemorySegment pyPart = mapToPyDict(part);
                    CPython.PyList_Append(pyContentList, pyPart);
                    PythonTypes.decref(pyPart);
                }
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("content"), pyContentList);
                PythonTypes.decref(pyContentList);
            } else if (msg.content() != null) {
                MemorySegment pyContent = PythonTypes.pyStr(arena, msg.content());
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("content"), pyContent);
                PythonTypes.decref(pyContent);
            } else {
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("content"), PythonTypes.pyNone());
            }

            // tool_calls (assistant messages with tool invocations)
            if (msg.hasToolCalls()) {
                MemorySegment pyToolCalls = toolCallsToPyList(msg.toolCalls());
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("tool_calls"), pyToolCalls);
                PythonTypes.decref(pyToolCalls);
            }

            // tool_call_id (tool result messages)
            if (msg.toolCallId() != null) {
                MemorySegment pyCallId = PythonTypes.pyStr(arena, msg.toolCallId());
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("tool_call_id"), pyCallId);
                PythonTypes.decref(pyCallId);
            }

            // name (tool result messages)
            if (msg.name() != null) {
                MemorySegment pyName = PythonTypes.pyStr(arena, msg.name());
                CPython.PyDict_SetItemString(pyDict, arena.allocateFrom("name"), pyName);
                PythonTypes.decref(pyName);
            }

            CPython.PyList_Append(pyList, pyDict);
            PythonTypes.decref(pyDict);
        }

        return pyList;
    }

    // ── Tool serialization ──────────────────────────────────────────────────

    /**
     * Converts a list of {@link Tool} to a Python list of dicts matching the
     * OpenAI tools format:
     * <pre>{@code
     * [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
     * }</pre>
     */
    private MemorySegment toolsToPyList(List<Tool> tools) {
        MemorySegment pyList = CPython.PyList_New(0);

        for (Tool tool : tools) {
            MemorySegment pyTool = CPython.PyDict_New();

            // type
            MemorySegment pyType = PythonTypes.pyStr(arena, tool.type());
            CPython.PyDict_SetItemString(pyTool, arena.allocateFrom("type"), pyType);
            PythonTypes.decref(pyType);

            // function dict
            MemorySegment pyFunc = CPython.PyDict_New();
            MemorySegment pyFuncName = PythonTypes.pyStr(arena, tool.function().name());
            CPython.PyDict_SetItemString(pyFunc, arena.allocateFrom("name"), pyFuncName);
            PythonTypes.decref(pyFuncName);

            MemorySegment pyFuncDesc = PythonTypes.pyStr(arena, tool.function().description());
            CPython.PyDict_SetItemString(pyFunc, arena.allocateFrom("description"), pyFuncDesc);
            PythonTypes.decref(pyFuncDesc);

            if (tool.function().parameters() != null) {
                MemorySegment pyParams = mapToPyDict(tool.function().parameters());
                CPython.PyDict_SetItemString(pyFunc, arena.allocateFrom("parameters"), pyParams);
                PythonTypes.decref(pyParams);
            }

            CPython.PyDict_SetItemString(pyTool, arena.allocateFrom("function"), pyFunc);
            PythonTypes.decref(pyFunc);

            CPython.PyList_Append(pyList, pyTool);
            PythonTypes.decref(pyTool);
        }

        return pyList;
    }

    /**
     * Converts a list of {@link ToolCall} to a Python list of dicts:
     * <pre>{@code
     * [{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}]
     * }</pre>
     */
    private MemorySegment toolCallsToPyList(List<ToolCall> toolCalls) {
        MemorySegment pyList = CPython.PyList_New(0);

        for (ToolCall tc : toolCalls) {
            MemorySegment pyCall = CPython.PyDict_New();

            MemorySegment pyId = PythonTypes.pyStr(arena, tc.id());
            CPython.PyDict_SetItemString(pyCall, arena.allocateFrom("id"), pyId);
            PythonTypes.decref(pyId);

            MemorySegment pyType = PythonTypes.pyStr(arena, tc.type());
            CPython.PyDict_SetItemString(pyCall, arena.allocateFrom("type"), pyType);
            PythonTypes.decref(pyType);

            // function: {"name": ..., "arguments": ...}
            MemorySegment pyFunc = CPython.PyDict_New();
            MemorySegment pyName = PythonTypes.pyStr(arena, tc.function().name());
            CPython.PyDict_SetItemString(pyFunc, arena.allocateFrom("name"), pyName);
            PythonTypes.decref(pyName);
            MemorySegment pyArgs = PythonTypes.pyStr(arena, tc.function().arguments());
            CPython.PyDict_SetItemString(pyFunc, arena.allocateFrom("arguments"), pyArgs);
            PythonTypes.decref(pyArgs);

            CPython.PyDict_SetItemString(pyCall, arena.allocateFrom("function"), pyFunc);
            PythonTypes.decref(pyFunc);

            CPython.PyList_Append(pyList, pyCall);
            PythonTypes.decref(pyCall);
        }

        return pyList;
    }

    // ── Recursive Java → Python conversion ──────────────────────────────────

    /**
     * Recursively converts a Java {@code Map<String, Object>} to a Python dict.
     *
     * <p>Supported value types:
     * <ul>
     *   <li>{@code String} → Python str</li>
     *   <li>{@code Integer}, {@code Long} → Python int</li>
     *   <li>{@code Double}, {@code Float} → Python float</li>
     *   <li>{@code Boolean} → Python bool</li>
     *   <li>{@code Map<String, Object>} → Python dict (recursive)</li>
     *   <li>{@code List<?>} / {@code Collection<?>} → Python list (recursive)</li>
     *   <li>{@code null} → Python None</li>
     * </ul>
     *
     * @param map the Java map to convert
     * @return new Python dict reference
     */
    @SuppressWarnings("unchecked")
    MemorySegment mapToPyDict(Map<String, Object> map) {
        MemorySegment pyDict = CPython.PyDict_New();
        for (var entry : map.entrySet()) {
            MemorySegment pyValue = javaToPython(entry.getValue());
            CPython.PyDict_SetItemString(pyDict, arena.allocateFrom(entry.getKey()), pyValue);
            PythonTypes.decref(pyValue);
        }
        return pyDict;
    }

    /**
     * Recursively converts a Java object to a Python object.
     */
    @SuppressWarnings("unchecked")
    private MemorySegment javaToPython(Object value) {
        if (value == null) {
            return PythonTypes.pyNone();
        }
        if (value instanceof String s) {
            return PythonTypes.pyStr(arena, s);
        }
        if (value instanceof Integer i) {
            return CPython.PyLong_FromLong(i);
        }
        if (value instanceof Long l) {
            return CPython.PyLong_FromLong(l);
        }
        if (value instanceof Double d) {
            return CPython.PyFloat_FromDouble(d);
        }
        if (value instanceof Float f) {
            return CPython.PyFloat_FromDouble(f.doubleValue());
        }
        if (value instanceof Boolean b) {
            return b ? PythonTypes.pyTrue() : PythonTypes.pyFalse();
        }
        if (value instanceof Map<?, ?> m) {
            return mapToPyDict((Map<String, Object>) m);
        }
        if (value instanceof Collection<?> c) {
            MemorySegment pyList = CPython.PyList_New(0);
            for (Object item : c) {
                MemorySegment pyItem = javaToPython(item);
                CPython.PyList_Append(pyList, pyItem);
                PythonTypes.decref(pyItem);
            }
            return pyList;
        }
        // Fallback: convert to string
        return PythonTypes.pyStr(arena, value.toString());
    }
}
