package io.gravitee.vllm.engine;

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.runtime.GIL;
import io.gravitee.vllm.runtime.PythonRuntime;

import org.vllm.python.CPython;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Wraps vLLM's synchronous {@code LLMEngine}.
 *
 * <p>Provides a clean Java API for request submission, stepping, and abort.
 * All public methods are thread-safe: they acquire the CPython GIL internally
 * via {@link GIL#acquire()}, so callers may invoke them from any thread.
 *
 * <h2>Lifecycle</h2>
 * <pre>{@code
 * try (var engine = VllmEngine.builder()
 *         .model("Qwen/Qwen3-0.6B")
 *         .dtype("auto")
 *         .build()) {
 *
 *     try (var sp = new SamplingParams().temperature(0.0).maxTokens(32)) {
 *         engine.addRequest(new VllmRequest("req-1", "Hello", sp));
 *     }
 *
 *     while (engine.hasUnfinishedRequests()) {
 *         List<RequestOutput> outputs = engine.step();
 *     }
 * }
 * }</pre>
 */
public final class VllmEngine implements AutoCloseable {

    /**
     * Returns a fluent builder for constructing a {@code VllmEngine}.
     */
    public static VllmEngineBuilder builder() {
        return new VllmEngineBuilder();
    }

    private final PythonRuntime runtime;
    private final MemorySegment engine;
    private final MemorySegment jinja2Module;
    private final Arena arena = Arena.ofAuto();

    /** Cached chat template string from the model's tokenizer. */
    private String chatTemplate;

    /** Cached Python function: vllm.lora.utils.get_adapter_absolute_path. */
    private MemorySegment loraPathResolver;

    /** Cached Python class: vllm.lora.request.LoRARequest. */
    private MemorySegment loraRequestClass;

    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Creates an {@code LLMEngine} instance using the given runtime.
     *
     * @param runtime an initialized {@link PythonRuntime}
     * @param model   HuggingFace model id, e.g. {@code "Qwen/Qwen3-0.6B"}
     * @param dtype   torch dtype string, e.g. {@code "auto"} or {@code "float32"}
     */
    public VllmEngine(PythonRuntime runtime, String model, String dtype) {
        this.runtime = runtime;

        try (var gil = GIL.acquire()) {
            MemorySegment engineArgsClass  = PythonCall.importClass(arena, "vllm.engine.arg_utils", "EngineArgs");
            MemorySegment engineArgsKwargs = buildEngineArgsKwargs(model, dtype);
            MemorySegment engineArgsInst   = PythonCall.callWithKwargs(engineArgsClass, engineArgsKwargs);
            PythonErrors.checkPythonError("EngineArgs instantiation");
            PythonTypes.decref(engineArgsKwargs);
            PythonTypes.decref(engineArgsClass);

            MemorySegment llmEngineClass = PythonCall.importClass(arena, "vllm.engine.llm_engine", "LLMEngine");
            MemorySegment fromEngineArgs = PythonTypes.getAttr(arena, llmEngineClass, "from_engine_args");
            MemorySegment eng = PythonCall.callOneArg(fromEngineArgs, engineArgsInst);
            PythonErrors.checkPythonError("LLMEngine.from_engine_args()");
            PythonTypes.decref(fromEngineArgs);
            PythonTypes.decref(llmEngineClass);
            PythonTypes.decref(engineArgsInst);

            MemorySegment jinja2;
            try {
                jinja2 = CPython.PyImport_ImportModule(arena.allocateFrom("jinja2"));
                PythonErrors.checkPythonError("import jinja2");
            } catch (Exception e) {
                PythonTypes.decref(eng);
                throw e;
            }

            // All Python objects created successfully — commit to fields
            this.engine = eng;
            this.jinja2Module = jinja2;
        }
    }

    /**
     * Creates an {@code LLMEngine} from a builder, applying all configured
     * engine parameters.
     *
     * @param runtime an initialized {@link PythonRuntime}
     * @param builder the builder containing engine configuration
     */
    VllmEngine(PythonRuntime runtime, VllmEngineBuilder builder) {
        this.runtime = runtime;

        try (var gil = GIL.acquire()) {
            MemorySegment engineArgsClass  = PythonCall.importClass(arena, "vllm.engine.arg_utils", "EngineArgs");
            MemorySegment engineArgsKwargs = buildEngineArgsKwargs(builder);
            MemorySegment engineArgsInst   = PythonCall.callWithKwargs(engineArgsClass, engineArgsKwargs);
            PythonErrors.checkPythonError("EngineArgs instantiation");
            PythonTypes.decref(engineArgsKwargs);
            PythonTypes.decref(engineArgsClass);

            MemorySegment llmEngineClass = PythonCall.importClass(arena, "vllm.engine.llm_engine", "LLMEngine");
            MemorySegment fromEngineArgs = PythonTypes.getAttr(arena, llmEngineClass, "from_engine_args");
            MemorySegment eng = PythonCall.callOneArg(fromEngineArgs, engineArgsInst);
            PythonErrors.checkPythonError("LLMEngine.from_engine_args()");
            PythonTypes.decref(fromEngineArgs);
            PythonTypes.decref(llmEngineClass);
            PythonTypes.decref(engineArgsInst);

            MemorySegment jinja2;
            try {
                jinja2 = CPython.PyImport_ImportModule(arena.allocateFrom("jinja2"));
                PythonErrors.checkPythonError("import jinja2");
            } catch (Exception e) {
                PythonTypes.decref(eng);
                throw e;
            }

            // All Python objects created successfully — commit to fields
            this.engine = eng;
            this.jinja2Module = jinja2;
        }
    }

    // ── LLMEngine operations ───────────────────────────────────────────────

    /**
     * Submits a {@link VllmRequest} to the engine.
     *
     * <p>Wraps {@code engine.add_request(request_id, prompt, sampling_params)}.
     * For multimodal requests, the prompt is wrapped in a
     * {@code TextPrompt} dict with {@code multi_modal_data}.
     *
     * <p>When the request carries a {@link LoraRequest}, the adapter path is
     * automatically resolved via {@code vllm.lora.utils.get_adapter_absolute_path()}
     * (which downloads from HuggingFace if needed), a Python
     * {@code LoRARequest} object is built, and it is passed as a
     * {@code lora_request} keyword argument.
     *
     * @param request the request to submit
     */
    public void addRequest(VllmRequest request) {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment pyRequestId = PythonTypes.pyStr(arena, request.requestId());

            MemorySegment pyPrompt;
            if (request.isMultiModal()) {
                // Build TextPrompt dict: {"prompt": text, "multi_modal_data": {...}}
                pyPrompt = CPython.PyDict_New();
                MemorySegment pyText = PythonTypes.pyStr(arena, request.prompt());
                PythonTypes.putDictObj(arena, pyPrompt, "prompt", pyText);
                PythonTypes.decref(pyText);
                MemorySegment pyMmData = request.multiModalData().toPythonDict(arena);
                if (pyMmData != null) {
                    PythonTypes.putDictObj(arena, pyPrompt, "multi_modal_data", pyMmData);
                    PythonTypes.decref(pyMmData);
                }
            } else {
                pyPrompt = PythonTypes.pyStr(arena, request.prompt());
            }

            if (request.hasLora()) {
                // ── LoRA path: use kwargs-based call ────────────────────────
                MemorySegment pyLoraRequest = buildPyLoraRequest(request.loraRequest());

                // Get bound method: engine.add_request
                MemorySegment addRequestFn = PythonTypes.getAttr(arena, engine, "add_request");
                PythonErrors.checkPythonError("getattr(engine, add_request)");

                // Positional args: (request_id, prompt, sampling_params)
                MemorySegment args = PythonCall.makeTuple(
                        pyRequestId, pyPrompt, request.samplingParams().get());

                // Keyword args: {"lora_request": lora_request}
                MemorySegment kwargs = CPython.PyDict_New();
                PythonTypes.putDictObj(arena, kwargs, "lora_request", pyLoraRequest);

                MemorySegment result = PythonCall.pyObjectCall(addRequestFn, args, kwargs);
                PythonErrors.checkPythonError("engine.add_request() requestId=" + request.requestId() + " [LoRA]");

                PythonTypes.decref(result);
                PythonTypes.decref(kwargs);
                PythonTypes.decref(args);
                PythonTypes.decref(addRequestFn);
                PythonTypes.decref(pyLoraRequest);
            } else {
                // ── Standard path: positional args only ─────────────────────
                MemorySegment pyName = PythonTypes.pyStr(arena, "add_request");
                MemorySegment result = PythonCall.callMethodObjArgs(engine, pyName,
                        pyRequestId, pyPrompt, request.samplingParams().get());
                PythonErrors.checkPythonError("engine.add_request() requestId=" + request.requestId());
                PythonTypes.decref(result);
                PythonTypes.decref(pyName);
            }

            PythonTypes.decref(pyRequestId);
            PythonTypes.decref(pyPrompt);
        }
    }

    // ── LoRA support ────────────────────────────────────────────────────────

    /**
     * Resolves a LoRA adapter path (local or HuggingFace repo ID) to an
     * absolute local filesystem path, downloading from HuggingFace if needed.
     *
     * <p>Calls {@code vllm.lora.utils.get_adapter_absolute_path(lora_path)}
     * through the CPython FFM bridge. The Python function handles:
     * <ul>
     *   <li>Absolute paths → returned as-is</li>
     *   <li>Relative paths → resolved against CWD</li>
     *   <li>HuggingFace repo IDs → downloaded via {@code huggingface_hub.snapshot_download()}</li>
     * </ul>
     *
     * @param loraPath local path or HuggingFace repo ID
     * @return the resolved absolute local path
     */
    public String resolveLoraPath(String loraPath) {
        try (var gil = GIL.acquire()) {
            if (loraPathResolver == null) {
                loraPathResolver = PythonCall.importClass(arena,
                        "vllm.lora.utils", "get_adapter_absolute_path");
            }

            MemorySegment pyPath = PythonTypes.pyStr(arena, loraPath);
            MemorySegment pyResult = PythonCall.callOneArg(loraPathResolver, pyPath);
            PythonErrors.checkPythonError("get_adapter_absolute_path(" + loraPath + ")");
            PythonTypes.decref(pyPath);

            String resolved = PythonTypes.pyUnicodeToString(pyResult);
            PythonTypes.decref(pyResult);
            return resolved;
        }
    }

    /**
     * Builds a Python {@code LoRARequest(lora_name, lora_int_id, lora_path)}
     * object from a Java {@link LoraRequest}.
     *
     * <p>The adapter path is resolved via {@link #resolveLoraPath(String)}
     * which auto-downloads from HuggingFace if the path is a repo ID.
     *
     * @param lora the Java LoRA request
     * @return new Python reference to a {@code LoRARequest} object
     */
    private MemorySegment buildPyLoraRequest(LoraRequest lora) {
        if (loraRequestClass == null) {
            loraRequestClass = PythonCall.importClass(arena,
                    "vllm.lora.request", "LoRARequest");
        }

        // Resolve the path (auto-download from HF if needed)
        String resolvedPath = resolveLoraPath(lora.loraPath());

        // Build positional args: LoRARequest(lora_name, lora_int_id, lora_path)
        MemorySegment pyName = PythonTypes.pyStr(arena, lora.loraName());
        MemorySegment pyIntId = CPython.PyLong_FromLong(lora.loraIntId());
        MemorySegment pyPath = PythonTypes.pyStr(arena, resolvedPath);

        MemorySegment args = PythonCall.makeTuple(pyName, pyIntId, pyPath);
        MemorySegment pyLoraRequest = PythonCall.pyObjectCall(loraRequestClass, args, MemorySegment.NULL);
        PythonErrors.checkPythonError("LoRARequest(" + lora.loraName() + ", " + lora.loraIntId() + ", " + resolvedPath + ")");

        PythonTypes.decref(args);
        PythonTypes.decref(pyPath);
        PythonTypes.decref(pyIntId);
        PythonTypes.decref(pyName);

        return pyLoraRequest;
    }

    /**
     * Wraps a single {@code engine.step()} call.
     *
     * @return list of request outputs (may be empty)
     */
    public List<RequestOutput> step() {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment pyName = PythonTypes.pyStr(arena, "step");
            MemorySegment pyOutputs = PythonCall.callMethodObjArgs(engine, pyName);
            PythonErrors.checkPythonError("engine.step()");
            PythonTypes.decref(pyName);

            List<RequestOutput> result = mapRequestOutputList(pyOutputs);
            PythonTypes.decref(pyOutputs);
            return result;
        }
    }

    /**
     * Wraps {@code engine.has_unfinished_requests()}.
     */
    public boolean hasUnfinishedRequests() {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment pyName   = PythonTypes.pyStr(arena, "has_unfinished_requests");
            MemorySegment result   = PythonCall.callMethodObjArgs(engine, pyName);
            PythonErrors.checkPythonError("engine.has_unfinished_requests()");
            boolean value = CPython.PyObject_IsTrue(result) != 0;
            PythonTypes.decref(pyName);
            PythonTypes.decref(result);
            return value;
        }
    }

    /**
     * Wraps {@code engine.abort_request([request_id])}.
     */
    public void abortRequest(String requestId) {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment pyName      = PythonTypes.pyStr(arena, "abort_request");
            MemorySegment pyRequestId = PythonTypes.pyStr(arena, requestId);
            MemorySegment pyList      = CPython.PyList_New(0);
            CPython.PyList_Append(pyList, pyRequestId);
            PythonTypes.decref(pyRequestId);
            MemorySegment result = PythonCall.callMethodObjArgs(engine, pyName, pyList);
            PythonErrors.checkPythonError("engine.abort_request() requestId=" + requestId);
            PythonTypes.decref(result);
            PythonTypes.decref(pyList);
            PythonTypes.decref(pyName);
        }
    }

    /**
     * Convenience method that submits a request, runs the step loop until
     * completion, and returns the final {@link RequestOutput}.
     *
     * <p>This is a blocking call — it does not return until the request
     * has finished generating. For streaming output, use
     * {@link io.gravitee.vllm.iterator.VllmIterator} instead.
     *
     * @param request the request to generate
     * @return the final request output with all completions
     */
    public RequestOutput generate(VllmRequest request) {
        checkNotClosed();
        addRequest(request);

        RequestOutput last = null;
        while (hasUnfinishedRequests()) {
            List<RequestOutput> outputs = step();
            for (RequestOutput out : outputs) {
                if (out.requestId().equals(request.requestId())) {
                    last = out;
                }
            }
        }

        if (last == null) {
            throw new VllmException("Request " + request.requestId() + " produced no output");
        }
        return last;
    }

    // ── Tokenizer / template access ────────────────────────────────────────

    /**
     * Returns the model's Jinja2 chat template string, extracted from the
     * HuggingFace tokenizer's {@code chat_template} attribute.
     *
     * <p>Calls {@code engine.tokenizer.get_chat_template()} which resolves
     * both simple string templates and named template dicts (e.g.
     * {@code {"default": "...", "tool_use": "..."}}).
     *
     * <p>The result is cached — the template is read once and reused.
     *
     * @return the Jinja2 template string, or {@code null} if the model has
     *         no chat template (e.g. a base model without chat fine-tuning)
     */
    public String getChatTemplate() {
        if (chatTemplate != null) return chatTemplate;

        try (var gil = GIL.acquire()) {
            // engine.tokenizer → the HuggingFace PreTrainedTokenizer
            MemorySegment tokenizer = PythonTypes.getAttr(arena, engine, "tokenizer");
            PythonErrors.checkPythonError("engine.tokenizer");

            // First check if chat_template is set at all (avoid exception from get_chat_template)
            MemorySegment rawTemplate = PythonTypes.getAttr(arena, tokenizer, "chat_template");
            if (PythonTypes.isNone(rawTemplate) || PythonTypes.isNull(rawTemplate)) {
                PythonTypes.decref(rawTemplate);
                PythonTypes.decref(tokenizer);
                CPython.PyErr_Clear();
                return null;
            }
            PythonTypes.decref(rawTemplate);

            // get_chat_template() handles str and dict cases, returns a single str
            MemorySegment pyMethodName = PythonTypes.pyStr(arena, "get_chat_template");
            MemorySegment resolved = PythonCall.callMethodObjArgs(tokenizer, pyMethodName);
            PythonErrors.checkPythonError("tokenizer.get_chat_template()");
            PythonTypes.decref(pyMethodName);

            chatTemplate = PythonTypes.pyUnicodeToString(resolved);
            PythonTypes.decref(resolved);
            PythonTypes.decref(tokenizer);

            return chatTemplate;
        }
    }

    /** Returns the cached jinja2 module reference. */
    public MemorySegment jinja2Module() {
        return jinja2Module;
    }

    /** Returns this engine's arena. */
    public Arena arena() {
        return arena;
    }

    // ── Tokenizer access ────────────────────────────────────────────────────

    /**
     * Encodes a text string into token IDs using the model's tokenizer.
     *
     * @param text the text to tokenize
     * @return the list of token IDs
     */
    public List<Integer> encode(String text) {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment tokenizer = PythonTypes.getAttr(arena, engine, "tokenizer");
            PythonErrors.checkPythonError("engine.tokenizer");

            MemorySegment pyText = PythonTypes.pyStr(arena, text);
            MemorySegment pyMethodName = PythonTypes.pyStr(arena, "encode");
            MemorySegment pyResult = PythonCall.callMethodObjArgs(tokenizer, pyMethodName, pyText);
            PythonErrors.checkPythonError("tokenizer.encode()");

            List<Integer> result = mapIntList(pyResult);
            PythonTypes.decref(pyMethodName);
            PythonTypes.decref(pyText);
            PythonTypes.decref(tokenizer);
            return result;
        }
    }

    /**
     * Decodes a list of token IDs back into text using the model's tokenizer.
     *
     * @param tokenIds the token IDs to decode
     * @return the decoded text
     */
    public String decode(List<Integer> tokenIds) {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment tokenizer = PythonTypes.getAttr(arena, engine, "tokenizer");
            PythonErrors.checkPythonError("engine.tokenizer");

            MemorySegment pyList = CPython.PyList_New(0);
            for (int id : tokenIds) {
                MemorySegment pyInt = CPython.PyLong_FromLong(id);
                CPython.PyList_Append(pyList, pyInt);
                PythonTypes.decref(pyInt);
            }

            MemorySegment pyMethodName = PythonTypes.pyStr(arena, "decode");
            MemorySegment pyResult = PythonCall.callMethodObjArgs(tokenizer, pyMethodName, pyList);
            PythonErrors.checkPythonError("tokenizer.decode()");

            String result = PythonTypes.pyUnicodeToString(pyResult);
            PythonTypes.decref(pyResult);
            PythonTypes.decref(pyMethodName);
            PythonTypes.decref(pyList);
            PythonTypes.decref(tokenizer);
            return result;
        }
    }

    /**
     * Returns the vocabulary size of the model's tokenizer.
     *
     * @return the number of tokens in the vocabulary
     */
    public int vocabSize() {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            MemorySegment tokenizer = PythonTypes.getAttr(arena, engine, "tokenizer");
            PythonErrors.checkPythonError("engine.tokenizer");

            long size = PythonTypes.getLongAttr(arena, tokenizer, "vocab_size");
            PythonTypes.decref(tokenizer);
            return (int) size;
        }
    }

    // ── Engine stats ──────────────────────────────────────────────────────

    /**
     * Returns a snapshot of the engine's current state.
     *
     * <p>Queries the Python engine for the number of unfinished requests
     * and model configuration details.
     *
     * @return engine stats snapshot
     */
    public EngineStats getStats() {
        checkNotClosed();
        try (var gil = GIL.acquire()) {
            // Number of unfinished requests
            int numUnfinished = 0;
            try {
                MemorySegment pyName = PythonTypes.pyStr(arena, "get_num_unfinished_requests");
                MemorySegment pyResult = PythonCall.callMethodObjArgs(engine, pyName);
                PythonErrors.checkPythonError("engine.get_num_unfinished_requests()");
                numUnfinished = (int) CPython.PyLong_AsLong(pyResult);
                PythonTypes.decref(pyResult);
                PythonTypes.decref(pyName);
            } catch (Exception e) {
                CPython.PyErr_Clear();
            }

            // Model config
            String model = "";
            String dtype = "";
            int maxModelLen = 0;
            try {
                MemorySegment pyModelConfig = PythonTypes.getAttr(arena, engine, "model_config");
                if (!PythonTypes.isNull(pyModelConfig) && !PythonTypes.isNone(pyModelConfig)) {
                    model = PythonTypes.getStrAttr(arena, pyModelConfig, "model");
                    CPython.PyErr_Clear(); // clear any error from model
                    try {
                        dtype = PythonTypes.getStrAttr(arena, pyModelConfig, "dtype");
                    } catch (Exception ignored) { /* dtype may be a torch.dtype, not str */ }
                    CPython.PyErr_Clear(); // clear any error from dtype
                    maxModelLen = (int) PythonTypes.getLongAttr(arena, pyModelConfig, "max_model_len");
                    CPython.PyErr_Clear();
                }
                PythonTypes.decref(pyModelConfig);
            } catch (Exception e) {
                // ignore
            }
            CPython.PyErr_Clear(); // ensure no stale Python error remains

            return new EngineStats(numUnfinished, model, dtype, maxModelLen);
        }
    }

    // ── AutoCloseable ──────────────────────────────────────────────────────

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;

        try (var gil = GIL.acquire()) {
            // Release the engine and jinja2 module references first.
            // When the engine's refcount drops to zero, CPython may trigger
            // LLMEngine.__del__() and free internal Python objects. However,
            // PyTorch's CUDA caching allocator retains GPU memory blocks even
            // after tensors are freed.
            PythonTypes.decref(engine);
            PythonTypes.decref(jinja2Module);

            // Force a Python garbage collection to break any circular references
            // that may be preventing the engine's internal objects from being freed.
            collectGarbage();
        }

        // Flush the GPU memory cache (CUDA or MPS) so VRAM is actually
        // returned to the driver. Without this, freed tensors remain in
        // PyTorch's block pool and the memory stays allocated to the process.
        GpuMemoryQuery.emptyCache();

        runtime.close();
    }

    /**
     * Calls {@code gc.collect()} to force Python garbage collection.
     *
     * <p>This breaks circular references in the engine's internal state
     * (e.g. scheduler ↔ model runner ↔ cache engine cycles) that prevent
     * automatic deallocation via reference counting alone.
     *
     * <p>Best-effort — silently ignores any errors.
     */
    private void collectGarbage() {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment gcModule = CPython.PyImport_ImportModule(tmp.allocateFrom("gc"));
            if (PythonTypes.isNull(gcModule)) { CPython.PyErr_Clear(); return; }
            MemorySegment collectName = PythonTypes.pyStr(tmp, "collect");
            MemorySegment result = PythonCall.callMethodObjArgs(gcModule, collectName);
            PythonTypes.decref(result);
            PythonTypes.decref(collectName);
            PythonTypes.decref(gcModule);
        } catch (Exception e) {
            CPython.PyErr_Clear();
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    private MemorySegment buildEngineArgsKwargs(String model, String dtype) {
        MemorySegment kwargs = CPython.PyDict_New();
        MemorySegment pyModel = PythonTypes.pyStr(arena, model);
        MemorySegment pyDtype = PythonTypes.pyStr(arena, dtype);
        PythonTypes.putDictObj(arena, kwargs, "model", pyModel);
        PythonTypes.putDictObj(arena, kwargs, "dtype", pyDtype);
        PythonTypes.decref(pyModel);
        PythonTypes.decref(pyDtype);
        return kwargs;
    }

    private MemorySegment buildEngineArgsKwargs(VllmEngineBuilder b) {
        MemorySegment kwargs = buildEngineArgsKwargs(b.model(), b.dtype());

        if (b.tokenizer() != null) {
            MemorySegment pyTokenizer = PythonTypes.pyStr(arena, b.tokenizer());
            PythonTypes.putDictObj(arena, kwargs, "tokenizer", pyTokenizer);
            PythonTypes.decref(pyTokenizer);
        }
        if (b.hfConfigPath() != null) {
            MemorySegment pyHfConfig = PythonTypes.pyStr(arena, b.hfConfigPath());
            PythonTypes.putDictObj(arena, kwargs, "hf_config_path", pyHfConfig);
            PythonTypes.decref(pyHfConfig);
        }
        if (b.gpuMemoryUtilization() != null)
            PythonTypes.putDictFloat(arena, kwargs, "gpu_memory_utilization", b.gpuMemoryUtilization());
        if (b.maxModelLen() != null)
            PythonTypes.putDictInt(arena, kwargs, "max_model_len", b.maxModelLen());
        if (b.maxNumSeqs() != null)
            PythonTypes.putDictInt(arena, kwargs, "max_num_seqs", b.maxNumSeqs());
        if (b.maxNumBatchedTokens() != null)
            PythonTypes.putDictInt(arena, kwargs, "max_num_batched_tokens", b.maxNumBatchedTokens());
        if (b.enforceEager() != null)
            PythonTypes.putDictObj(arena, kwargs, "enforce_eager",
                    b.enforceEager() ? PythonTypes.pyTrue() : PythonTypes.pyFalse());
        if (b.trustRemoteCode() != null)
            PythonTypes.putDictObj(arena, kwargs, "trust_remote_code",
                    b.trustRemoteCode() ? PythonTypes.pyTrue() : PythonTypes.pyFalse());
        if (b.quantization() != null) {
            MemorySegment pyQuant = PythonTypes.pyStr(arena, b.quantization());
            PythonTypes.putDictObj(arena, kwargs, "quantization", pyQuant);
            PythonTypes.decref(pyQuant);
        }
        if (b.swapSpace() != null)
            PythonTypes.putDictFloat(arena, kwargs, "swap_space", b.swapSpace());
        if (b.seed() != null)
            PythonTypes.putDictInt(arena, kwargs, "seed", b.seed());
        if (b.enablePrefixCaching() != null)
            PythonTypes.putDictObj(arena, kwargs, "enable_prefix_caching",
                    b.enablePrefixCaching() ? PythonTypes.pyTrue() : PythonTypes.pyFalse());
        if (b.enableLora() != null)
            PythonTypes.putDictObj(arena, kwargs, "enable_lora",
                    b.enableLora() ? PythonTypes.pyTrue() : PythonTypes.pyFalse());
        if (b.maxLoras() != null)
            PythonTypes.putDictInt(arena, kwargs, "max_loras", b.maxLoras());
        if (b.maxLoraRank() != null)
            PythonTypes.putDictInt(arena, kwargs, "max_lora_rank", b.maxLoraRank());
        if (b.enableChunkedPrefill() != null)
            PythonTypes.putDictObj(arena, kwargs, "enable_chunked_prefill",
                    b.enableChunkedPrefill() ? PythonTypes.pyTrue() : PythonTypes.pyFalse());
        if (b.kvCacheDtype() != null) {
            MemorySegment pyKvDtype = PythonTypes.pyStr(arena, b.kvCacheDtype());
            PythonTypes.putDictObj(arena, kwargs, "kv_cache_dtype", pyKvDtype);
            PythonTypes.decref(pyKvDtype);
        }

        return kwargs;
    }

    private List<RequestOutput> mapRequestOutputList(MemorySegment pyList) {
        long size = CPython.PyList_Size(pyList);
        List<RequestOutput> result = new ArrayList<>((int) size);
        for (long i = 0; i < size; i++) {
            MemorySegment pyReqOut = CPython.PyList_GetItem(pyList, i); // borrowed
            result.add(mapRequestOutput(pyReqOut));
        }
        return result;
    }

    private RequestOutput mapRequestOutput(MemorySegment pyReqOut) {
        String requestId = PythonTypes.getStrAttr(arena, pyReqOut, "request_id");
        boolean finished = PythonTypes.getBoolAttr(arena, pyReqOut, "finished");

        MemorySegment pyOutputs = PythonTypes.getAttr(arena, pyReqOut, "outputs");
        long size = CPython.PyList_Size(pyOutputs);
        List<CompletionOutput> completions = new ArrayList<>((int) size);
        for (long i = 0; i < size; i++) {
            MemorySegment pyComp = CPython.PyList_GetItem(pyOutputs, i); // borrowed
            completions.add(mapCompletionOutput(pyComp));
        }
        PythonTypes.decref(pyOutputs);

        // prompt_token_ids (may be None)
        List<Integer> promptTokenIds = mapIntList(PythonTypes.getAttr(arena, pyReqOut, "prompt_token_ids"));

        // num_cached_tokens (may be None → 0)
        int numCachedTokens = 0;
        MemorySegment pyCached = PythonTypes.getAttr(arena, pyReqOut, "num_cached_tokens");
        if (!PythonTypes.isNone(pyCached) && !PythonTypes.isNull(pyCached)) {
            numCachedTokens = (int) CPython.PyLong_AsLong(pyCached);
        }
        PythonTypes.decref(pyCached);
        CPython.PyErr_Clear(); // clear any AttributeError if field missing

        // metrics (may be None)
        RequestMetrics metrics = mapMetrics(PythonTypes.getAttr(arena, pyReqOut, "metrics"));

        // prompt_logprobs (may be None)
        List<Map<Integer, LogprobEntry>> promptLogprobs = null;
        try {
            MemorySegment pyPromptLogprobs = PythonTypes.getAttr(arena, pyReqOut, "prompt_logprobs");
            promptLogprobs = mapLogprobsList(pyPromptLogprobs);
        } catch (Exception e) {
            CPython.PyErr_Clear();
        }

        return new RequestOutput(requestId, completions, finished, promptTokenIds,
                numCachedTokens, metrics, promptLogprobs);
    }

    private CompletionOutput mapCompletionOutput(MemorySegment pyComp) {
        int    index = (int) PythonTypes.getLongAttr(arena, pyComp, "index");
        String text  = PythonTypes.getStrAttr(arena, pyComp, "text");

        // token_ids
        List<Integer> tokenIds = mapIntList(PythonTypes.getAttr(arena, pyComp, "token_ids"));

        // finish_reason (str or None)
        MemorySegment pyFinishReason = PythonTypes.getAttr(arena, pyComp, "finish_reason");
        FinishReason finishReason = null;
        if (!PythonTypes.isNone(pyFinishReason) && !PythonTypes.isNull(pyFinishReason)) {
            finishReason = FinishReason.fromVllmString(PythonTypes.pyUnicodeToString(pyFinishReason));
        }
        PythonTypes.decref(pyFinishReason);

        // logprobs (may be None)
        List<Map<Integer, LogprobEntry>> logprobs = null;
        try {
            MemorySegment pyLogprobs = PythonTypes.getAttr(arena, pyComp, "logprobs");
            logprobs = mapLogprobsList(pyLogprobs);
        } catch (Exception e) {
            CPython.PyErr_Clear();
        }

        return new CompletionOutput(index, text, tokenIds, finishReason, logprobs);
    }

    /**
     * Maps a Python list of logprob dicts to a Java list of maps.
     *
     * <p>vLLM logprobs structure: {@code List[Optional[Dict[int, Logprob]]]} where
     * {@code Logprob} has {@code logprob}, {@code rank}, {@code decoded_token}.
     *
     * @return list of maps, or {@code null} if None/null
     */
    private List<Map<Integer, LogprobEntry>> mapLogprobsList(MemorySegment pyLogprobs) {
        if (PythonTypes.isNone(pyLogprobs) || PythonTypes.isNull(pyLogprobs)) {
            PythonTypes.decref(pyLogprobs);
            CPython.PyErr_Clear();
            return null;
        }
        long size = CPython.PyList_Size(pyLogprobs);
        List<Map<Integer, LogprobEntry>> result = new ArrayList<>((int) size);
        for (long i = 0; i < size; i++) {
            MemorySegment pyDict = CPython.PyList_GetItem(pyLogprobs, i); // borrowed
            if (PythonTypes.isNone(pyDict) || PythonTypes.isNull(pyDict)) {
                result.add(null); // position has no logprobs (e.g. first prompt token)
                continue;
            }
            // Iterate the Python dict: {int token_id: Logprob}
            Map<Integer, LogprobEntry> posMap = new HashMap<>();
            MemorySegment pyMethodName = PythonTypes.pyStr(arena, "items");
            MemorySegment pyItemsView = PythonCall.callMethodObjArgs(pyDict, pyMethodName);
            PythonTypes.decref(pyMethodName);
            if (!PythonTypes.isNull(pyItemsView)) {
                MemorySegment pyIter = CPython.PyObject_GetIter(pyItemsView);
                PythonTypes.decref(pyItemsView);
                if (!PythonTypes.isNull(pyIter)) {
                    MemorySegment pyTuple;
                    while (!PythonTypes.isNull(pyTuple = CPython.PyIter_Next(pyIter))) {
                        int tokenId = (int) CPython.PyLong_AsLong(CPython.PyTuple_GetItem(pyTuple, 0));
                        MemorySegment pyLogprob = CPython.PyTuple_GetItem(pyTuple, 1); // borrowed

                        double logprob = PythonTypes.getDoubleAttr(arena, pyLogprob, "logprob");
                        int rank = (int) PythonTypes.getLongAttr(arena, pyLogprob, "rank");
                        String decodedToken = "";
                        try {
                            decodedToken = PythonTypes.getStrAttr(arena, pyLogprob, "decoded_token");
                        } catch (Exception e) {
                            CPython.PyErr_Clear();
                        }
                        posMap.put(tokenId, new LogprobEntry(tokenId, logprob, rank, decodedToken));
                        PythonTypes.decref(pyTuple); // PyIter_Next returns new reference
                    }
                    PythonTypes.decref(pyIter);
                }
            } else {
                CPython.PyErr_Clear();
            }
            result.add(posMap);
        }
        PythonTypes.decref(pyLogprobs);
        return result;
    }

    /** Maps a Python list of ints to a Java {@code List<Integer>}. Returns empty list if None/null. */
    private List<Integer> mapIntList(MemorySegment pyList) {
        if (PythonTypes.isNone(pyList) || PythonTypes.isNull(pyList)) {
            PythonTypes.decref(pyList);
            CPython.PyErr_Clear();
            return List.of();
        }
        long size = CPython.PyList_Size(pyList);
        List<Integer> result = new ArrayList<>((int) size);
        for (long i = 0; i < size; i++) {
            MemorySegment item = CPython.PyList_GetItem(pyList, i); // borrowed
            result.add((int) CPython.PyLong_AsLong(item));
        }
        PythonTypes.decref(pyList);
        return result;
    }

    /** Maps Python RequestMetrics to Java RequestMetrics. Returns null if None. */
    private RequestMetrics mapMetrics(MemorySegment pyMetrics) {
        if (PythonTypes.isNone(pyMetrics) || PythonTypes.isNull(pyMetrics)) {
            PythonTypes.decref(pyMetrics);
            CPython.PyErr_Clear();
            return null;
        }
        double arrivalTime = -1, lastTokenTime = -1, firstScheduledTime = -1;
        double firstTokenTime = -1, timeInQueue = -1, finishedTime = -1;
        try {
            arrivalTime = safeGetDouble(pyMetrics, "arrival_time");
            lastTokenTime = safeGetDouble(pyMetrics, "last_token_time");
            firstScheduledTime = safeGetDouble(pyMetrics, "first_scheduled_time");
            firstTokenTime = safeGetDouble(pyMetrics, "first_token_time");
            timeInQueue = safeGetDouble(pyMetrics, "time_in_queue");
            finishedTime = safeGetDouble(pyMetrics, "finished_time");
        } catch (Exception e) {
            CPython.PyErr_Clear();
        }
        PythonTypes.decref(pyMetrics);
        return new RequestMetrics(arrivalTime, lastTokenTime, firstScheduledTime,
                firstTokenTime, timeInQueue, finishedTime);
    }

    /** Safely reads a double attribute, returning -1 if None/missing. */
    private double safeGetDouble(MemorySegment obj, String name) {
        try {
            MemorySegment attr = PythonTypes.getAttr(arena, obj, name);
            if (PythonTypes.isNone(attr) || PythonTypes.isNull(attr)) {
                PythonTypes.decref(attr);
                CPython.PyErr_Clear();
                return -1;
            }
            double val = CPython.PyFloat_AsDouble(attr);
            PythonTypes.decref(attr);
            return val;
        } catch (Exception e) {
            CPython.PyErr_Clear();
            return -1;
        }
    }

    private void checkNotClosed() {
        if (closed.get()) throw new IllegalStateException("VllmEngine has been closed");
    }
}
