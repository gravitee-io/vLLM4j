/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.vllm.engine;

import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.platform.PlatformResolver;
import io.gravitee.vllm.platform.VllmBackend;
import io.gravitee.vllm.runtime.GIL;
import io.gravitee.vllm.runtime.PythonRuntime;
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

  private final MemorySegment engine;
  private final MemorySegment jinja2Module;
  private final Arena arena;

  /** Cached chat template string from the model's tokenizer. */
  private String chatTemplate;

  /** Cached Python function: vllm.lora.utils.get_adapter_absolute_path. */
  private MemorySegment loraPathResolver;

  /** Cached Python class: vllm.lora.request.LoRARequest. */
  private MemorySegment loraRequestClass;

  private final AtomicBoolean closed = new AtomicBoolean(false);
  private final boolean sleepModeEnabled;

  /**
   * Creates an {@code LLMEngine} instance using the given runtime.
   *
   * @param arena shared arena for native memory allocation (must outlive this engine)
   * @param model   HuggingFace model id, e.g. {@code "Qwen/Qwen3-0.6B"}
   * @param dtype   torch dtype string, e.g. {@code "auto"} or {@code "float32"}
   */
  public VllmEngine(Arena arena, String model, String dtype) {
    this.arena = arena;
    this.sleepModeEnabled = PlatformResolver.backend() == VllmBackend.CUDA;
    try (var gil = GIL.acquire()) {
      MemorySegment engineArgsClass = PythonCall.importClass(
        arena,
        "vllm.engine.arg_utils",
        "EngineArgs"
      );
      MemorySegment engineArgsKwargs = buildEngineArgsKwargs(model, dtype);
      MemorySegment engineArgsInst = PythonCall.callWithKwargs(
        engineArgsClass,
        engineArgsKwargs
      );
      PythonErrors.checkPythonError("EngineArgs instantiation");
      PythonTypes.decref(engineArgsKwargs);
      PythonTypes.decref(engineArgsClass);

      MemorySegment llmEngineClass = PythonCall.importClass(
        arena,
        "vllm.engine.llm_engine",
        "LLMEngine"
      );
      MemorySegment fromEngineArgs = PythonTypes.getAttr(
        arena,
        llmEngineClass,
        "from_engine_args"
      );
      MemorySegment eng = PythonCall.callOneArg(fromEngineArgs, engineArgsInst);
      PythonErrors.checkPythonError("LLMEngine.from_engine_args()");
      PythonTypes.decref(fromEngineArgs);
      PythonTypes.decref(llmEngineClass);
      PythonTypes.decref(engineArgsInst);

      MemorySegment jinja2;
      try {
        jinja2 = CPythonBinding.PyImport_ImportModule(
          arena.allocateFrom("jinja2")
        );
        PythonErrors.checkPythonError("import jinja2");
      } catch (Exception e) {
        PythonTypes.decref(eng);
        throw e;
      }

      // All Python objects created successfully — commit to fields
      this.engine = eng;
      this.jinja2Module = jinja2;
    }
    PythonRuntime.registerEngine();
  }

  /**
   * Creates an {@code LLMEngine} from a builder, applying all configured
   * engine parameters.
   *
   * @param arena   shared arena for native memory allocation (must outlive this engine)
   * @param builder the builder containing engine configuration
   */
  VllmEngine(Arena arena, VllmEngineBuilder builder) {
    this.arena = arena;
    this.sleepModeEnabled = builder.isSleepModeEnabled();
    try (var gil = GIL.acquire()) {
      MemorySegment engineArgsClass = PythonCall.importClass(
        arena,
        "vllm.engine.arg_utils",
        "EngineArgs"
      );
      MemorySegment engineArgsKwargs = buildEngineArgsKwargs(builder);
      MemorySegment engineArgsInst = PythonCall.callWithKwargs(
        engineArgsClass,
        engineArgsKwargs
      );
      PythonErrors.checkPythonError("EngineArgs instantiation");
      PythonTypes.decref(engineArgsKwargs);
      PythonTypes.decref(engineArgsClass);

      // Smart V0/V1 engine selection: distributed configs require V1's
      // SyncMPClient → EngineCore architecture for subprocess-based GPU
      // workers. Single-GPU mode stays on V0 (in-process, no subprocesses).
      MemorySegment llmEngineClass = PythonCall.importClass(
        arena,
        builder.isDistributed()
          ? "vllm.v1.engine.llm_engine"
          : "vllm.engine.llm_engine",
        "LLMEngine"
      );
      MemorySegment fromEngineArgs = PythonTypes.getAttr(
        arena,
        llmEngineClass,
        "from_engine_args"
      );
      MemorySegment eng = PythonCall.callOneArg(fromEngineArgs, engineArgsInst);
      PythonErrors.checkPythonError("LLMEngine.from_engine_args()");
      PythonTypes.decref(fromEngineArgs);
      PythonTypes.decref(llmEngineClass);
      PythonTypes.decref(engineArgsInst);

      MemorySegment jinja2;
      try {
        jinja2 = CPythonBinding.PyImport_ImportModule(
          arena.allocateFrom("jinja2")
        );
        PythonErrors.checkPythonError("import jinja2");
      } catch (Exception e) {
        PythonTypes.decref(eng);
        throw e;
      }

      // All Python objects created successfully — commit to fields
      this.engine = eng;
      this.jinja2Module = jinja2;
    }
    PythonRuntime.registerEngine();
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
        pyPrompt = CPythonBinding.PyDict_New();
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
        MemorySegment addRequestFn = PythonTypes.getAttr(
          arena,
          engine,
          "add_request"
        );
        PythonErrors.checkPythonError("getattr(engine, add_request)");

        // Positional args: (request_id, prompt, sampling_params)
        MemorySegment args = PythonCall.makeTuple(
          pyRequestId,
          pyPrompt,
          request.samplingParams().get()
        );

        // Keyword args: {"lora_request": lora_request}
        MemorySegment kwargs = CPythonBinding.PyDict_New();
        PythonTypes.putDictObj(arena, kwargs, "lora_request", pyLoraRequest);

        MemorySegment result = PythonCall.pyObjectCall(
          addRequestFn,
          args,
          kwargs
        );
        PythonErrors.checkPythonError(
          "engine.add_request() requestId=" + request.requestId() + " [LoRA]"
        );

        PythonTypes.decref(result);
        PythonTypes.decref(kwargs);
        PythonTypes.decref(args);
        PythonTypes.decref(addRequestFn);
        PythonTypes.decref(pyLoraRequest);
      } else {
        // ── Standard path: positional args only ─────────────────────
        MemorySegment pyName = PythonTypes.pyStr(arena, "add_request");
        MemorySegment result = PythonCall.callMethodObjArgs(
          engine,
          pyName,
          pyRequestId,
          pyPrompt,
          request.samplingParams().get()
        );
        PythonErrors.checkPythonError(
          "engine.add_request() requestId=" + request.requestId()
        );
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
        loraPathResolver = PythonCall.importClass(
          arena,
          "vllm.lora.utils",
          "get_adapter_absolute_path"
        );
      }

      MemorySegment pyPath = PythonTypes.pyStr(arena, loraPath);
      MemorySegment pyResult = PythonCall.callOneArg(loraPathResolver, pyPath);
      PythonErrors.checkPythonError(
        "get_adapter_absolute_path(" + loraPath + ")"
      );
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
      loraRequestClass = PythonCall.importClass(
        arena,
        "vllm.lora.request",
        "LoRARequest"
      );
    }

    // Resolve the path (auto-download from HF if needed)
    String resolvedPath = resolveLoraPath(lora.loraPath());

    // Build positional args: LoRARequest(lora_name, lora_int_id, lora_path)
    MemorySegment pyName = PythonTypes.pyStr(arena, lora.loraName());
    MemorySegment pyIntId = CPythonBinding.PyLong_FromLong(lora.loraIntId());
    MemorySegment pyPath = PythonTypes.pyStr(arena, resolvedPath);

    MemorySegment args = PythonCall.makeTuple(pyName, pyIntId, pyPath);
    MemorySegment pyLoraRequest = PythonCall.pyObjectCall(
      loraRequestClass,
      args,
      MemorySegment.NULL
    );
    PythonErrors.checkPythonError(
      "LoRARequest(" +
        lora.loraName() +
        ", " +
        lora.loraIntId() +
        ", " +
        resolvedPath +
        ")"
    );

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
      MemorySegment pyName = PythonTypes.pyStr(
        arena,
        "has_unfinished_requests"
      );
      MemorySegment result = PythonCall.callMethodObjArgs(engine, pyName);
      PythonErrors.checkPythonError("engine.has_unfinished_requests()");
      boolean value = CPythonBinding.PyObject_IsTrue(result) != 0;
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
      MemorySegment pyName = PythonTypes.pyStr(arena, "abort_request");
      MemorySegment pyRequestId = PythonTypes.pyStr(arena, requestId);
      MemorySegment pyList = CPythonBinding.PyList_New(0);
      CPythonBinding.PyList_Append(pyList, pyRequestId);
      PythonTypes.decref(pyRequestId);
      MemorySegment result = PythonCall.callMethodObjArgs(
        engine,
        pyName,
        pyList
      );
      PythonErrors.checkPythonError(
        "engine.abort_request() requestId=" + requestId
      );
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
      throw new VllmException(
        "Request " + request.requestId() + " produced no output"
      );
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
      MemorySegment rawTemplate = PythonTypes.getAttr(
        arena,
        tokenizer,
        "chat_template"
      );
      if (PythonTypes.isNone(rawTemplate) || PythonTypes.isNull(rawTemplate)) {
        PythonTypes.decref(rawTemplate);
        PythonTypes.decref(tokenizer);
        CPythonBinding.PyErr_Clear();
        return null;
      }
      PythonTypes.decref(rawTemplate);

      // get_chat_template() handles str and dict cases, returns a single str
      MemorySegment pyMethodName = PythonTypes.pyStr(
        arena,
        "get_chat_template"
      );
      MemorySegment resolved = PythonCall.callMethodObjArgs(
        tokenizer,
        pyMethodName
      );
      PythonErrors.checkPythonError("tokenizer.get_chat_template()");
      PythonTypes.decref(pyMethodName);

      chatTemplate = PythonTypes.pyUnicodeToString(resolved);
      PythonTypes.decref(resolved);
      PythonTypes.decref(tokenizer);

      return chatTemplate;
    }
  }

  /**
   * Returns the BOS (beginning-of-sentence) token string from the tokenizer.
   * Returns an empty string if the tokenizer has no BOS token.
   */
  public String getBosToken() {
    return getSpecialToken("bos_token");
  }

  /**
   * Returns the EOS (end-of-sentence) token string from the tokenizer.
   * Returns an empty string if the tokenizer has no EOS token.
   */
  public String getEosToken() {
    return getSpecialToken("eos_token");
  }

  private String getSpecialToken(String attrName) {
    try (var gil = GIL.acquire()) {
      MemorySegment tokenizer = PythonTypes.getAttr(arena, engine, "tokenizer");
      PythonErrors.checkPythonError("engine.tokenizer");

      MemorySegment pyToken = PythonTypes.getAttr(arena, tokenizer, attrName);
      if (PythonTypes.isNone(pyToken) || PythonTypes.isNull(pyToken)) {
        PythonTypes.decref(pyToken);
        PythonTypes.decref(tokenizer);
        CPythonBinding.PyErr_Clear();
        return "";
      }

      String token = PythonTypes.pyUnicodeToString(pyToken);
      PythonTypes.decref(pyToken);
      PythonTypes.decref(tokenizer);
      return token != null ? token : "";
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
      MemorySegment pyResult = PythonCall.callMethodObjArgs(
        tokenizer,
        pyMethodName,
        pyText
      );
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

      MemorySegment pyList = CPythonBinding.PyList_New(0);
      for (int id : tokenIds) {
        MemorySegment pyInt = CPythonBinding.PyLong_FromLong(id);
        CPythonBinding.PyList_Append(pyList, pyInt);
        PythonTypes.decref(pyInt);
      }

      MemorySegment pyMethodName = PythonTypes.pyStr(arena, "decode");
      MemorySegment pyResult = PythonCall.callMethodObjArgs(
        tokenizer,
        pyMethodName,
        pyList
      );
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
        MemorySegment pyName = PythonTypes.pyStr(
          arena,
          "get_num_unfinished_requests"
        );
        MemorySegment pyResult = PythonCall.callMethodObjArgs(engine, pyName);
        PythonErrors.checkPythonError("engine.get_num_unfinished_requests()");
        numUnfinished = (int) CPythonBinding.PyLong_AsLong(pyResult);
        PythonTypes.decref(pyResult);
        PythonTypes.decref(pyName);
      } catch (Exception e) {
        CPythonBinding.PyErr_Clear();
      }

      // Model config
      String model = "";
      String dtype = "";
      int maxModelLen = 0;
      try {
        MemorySegment pyModelConfig = PythonTypes.getAttr(
          arena,
          engine,
          "model_config"
        );
        if (
          !PythonTypes.isNull(pyModelConfig) &&
          !PythonTypes.isNone(pyModelConfig)
        ) {
          model = PythonTypes.getStrAttr(arena, pyModelConfig, "model");
          CPythonBinding.PyErr_Clear(); // clear any error from model
          try {
            dtype = PythonTypes.getStrAttr(arena, pyModelConfig, "dtype");
          } catch (Exception ignored) {
            /* dtype may be a torch.dtype, not str */
          }
          CPythonBinding.PyErr_Clear(); // clear any error from dtype
          maxModelLen = (int) PythonTypes.getLongAttr(
            arena,
            pyModelConfig,
            "max_model_len"
          );
          CPythonBinding.PyErr_Clear();
        }
        PythonTypes.decref(pyModelConfig);
      } catch (Exception e) {
        // ignore
      }
      CPythonBinding.PyErr_Clear(); // ensure no stale Python error remains

      return new EngineStats(numUnfinished, model, dtype, maxModelLen);
    }
  }

  // ── Memory management ─────────────────────────────────────────────────────

  /**
   * Releases GPU memory cached by PyTorch without destroying the engine.
   *
   * <p>Suitable for calling between inference requests to free temporary GPU
   * allocations. This does NOT restart the engine or release model weights —
   * only cached memory blocks that PyTorch's allocator is holding.
   *
   * <p>Performs:
   * <ol>
   *   <li>Synchronizes pending GPU operations (ensures all kernels complete)</li>
   *   <li>Flushes PyTorch's memory cache via {@code torch.cuda.empty_cache()}</li>
   * </ol>
   *
   * <p>Best-effort — silently ignores errors (e.g., if CUDA is unavailable).
   *
   * @see GpuMemoryQuery#synchronizeAndEmptyCache()
   */
  public void freeCache() {
    checkNotClosed();
    GpuMemoryQuery.synchronizeAndEmptyCache();
  }

  /**
   * Queries the current GPU memory usage.
   *
   * @return a {@link GpuMemoryQuery.GpuMemoryInfo} with free and total bytes,
   *         or {@code null} if GPU memory cannot be queried (e.g., CUDA unavailable)
   * @see GpuMemoryQuery#query()
   */
  public GpuMemoryQuery.GpuMemoryInfo queryMemory() {
    return GpuMemoryQuery.query();
  }

  /**
   * Performs aggressive memory cleanup without restarting the engine.
   *
   * <p>This is a heavier-weight cleanup than {@link #freeCache()}, suitable for
   * periodic maintenance (e.g., every N requests or every T seconds).
   *
   * <p>Performs:
   * <ol>
   *   <li>GPU synchronization and cache flushing (via {@link #freeCache()})</li>
   *   <li>Multiple passes of Python garbage collection to break circular references</li>
   * </ol>
   *
   * <p>Does NOT restart the engine or release model weights.
   *
   * <p>Best-effort — silently ignores errors.
   *
   * @see GpuMemoryQuery#aggressiveCacheCleanup()
   */
  public void reset() {
    checkNotClosed();
    GpuMemoryQuery.aggressiveCacheCleanup();
  }

  // ── AutoCloseable ──────────────────────────────────────────────────────

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) return;
    System.out.println("[vLLM4j] VllmEngine.close() — begin teardown");
    System.out.flush();

    // Stop the keepalive thread BEFORE tearing down Python objects.
    // vLLM's shutdown may invalidate Python thread states that the
    // keepalive thread touches via PyGILState_Ensure — if the keepalive
    // fires between shutdownEngineCore() and gc.collect(), it segfaults.
    PythonRuntime.unregisterEngine();
    System.out.println("[vLLM4j] close: keepalive stopped");
    System.out.flush();

    try (var gil = GIL.acquire()) {
      System.out.println("[vLLM4j] close: GIL acquired");
      System.out.flush();

      // Release cached LoRA helpers (if they were lazily initialized).
      if (loraPathResolver != null) {
        PythonTypes.decref(loraPathResolver);
        loraPathResolver = null;
      }
      if (loraRequestClass != null) {
        PythonTypes.decref(loraRequestClass);
        loraRequestClass = null;
      }

      if (PlatformResolver.backend() == VllmBackend.CUDA && sleepModeEnabled) {
        // ── CUDA teardown: sleep → reset allocator → shutdown ────────
        // Sleep mode releases GPU memory via CuMemAllocator (cuMemCreate/cuMemMap).
        System.out.println(
          "[vLLM4j] close: putting engine to sleep (release GPU memory via CuMemAllocator)"
        );
        System.out.flush();
        sleepEngine();
        System.out.println("[vLLM4j] close: sleep done");
        System.out.flush();

        // Clear CuMemAllocator.pointer_to_data so the next engine can
        // pass the get_current_usage() == 0 assertion. Then set
        // CuMemAllocator.instance = None so a fresh allocator is created.
        System.out.println("[vLLM4j] close: resetting CuMemAllocator");
        System.out.flush();
        resetCuMemAllocator();

        System.out.println("[vLLM4j] close: calling shutdownEngineCore()");
        System.out.flush();
        shutdownEngineCore();
        System.out.println("[vLLM4j] close: shutdownEngineCore() done");
        System.out.flush();

        // After sleep(), GPU memory is released. We intentionally DO NOT
        // call decref(engine) or gc.collect() because destroying the Python
        // objects triggers MemPool::~MemPool() which crashes with
        // "captures_underway.empty() INTERNAL ASSERT FAILED" — a known
        // PyTorch bug when pluggable allocator pools are destroyed after
        // sleep() has unmapped the underlying GPU memory.
        //
        // The engine's Python objects will leak in the CPython heap (they
        // hold no GPU memory after sleep). This is a deliberate tradeoff:
        // ~10-20 MB of CPU-side Python objects vs a guaranteed crash.
      } else {
        // ── Non-CUDA teardown (Metal, CPU): standard cleanup ─────────
        // No CuMemAllocator on these platforms — safe to decref + gc.
        System.out.println("[vLLM4j] close: calling shutdownEngineCore()");
        System.out.flush();
        shutdownEngineCore();
        System.out.println("[vLLM4j] close: shutdownEngineCore() done");
        System.out.flush();

        System.out.println("[vLLM4j] close: decref(engine)");
        System.out.flush();
        PythonTypes.decref(engine);

        System.out.println("[vLLM4j] close: gc.collect()");
        System.out.flush();
        collectGarbage();
      }

      System.out.println("[vLLM4j] close: decref(jinja2Module)");
      System.out.flush();
      PythonTypes.decref(jinja2Module);

      System.out.println("[vLLM4j] close: releasing GIL");
      System.out.flush();
    }

    System.out.println("[vLLM4j] close: teardown complete");
    System.out.flush();
  }

  /**
   * Resets the {@code CuMemAllocator} singleton so a new engine can be
   * created in the same process without hitting stale state.
   *
   * <h3>Problem</h3>
   * After {@code sleep()}, the allocator's {@code pointer_to_data} dict
   * still has entries (handles that were unmapped but not removed from the
   * dict). The next engine's {@code load_model()} asserts
   * {@code get_current_usage() == 0} and fails with "Sleep mode can only
   * be used for one instance per process."
   *
   * <h3>Strategy</h3>
   * <ol>
   *   <li>Clear {@code pointer_to_data} on the current instance — the
   *       entries are stale (GPU memory was already unmapped by
   *       {@code sleep()}). This makes {@code get_current_usage() == 0}.</li>
   *   <li>Set {@code CuMemAllocator.instance = None} — forces the next
   *       engine to create a fresh singleton with new
   *       {@code allocator_and_pools}.</li>
   * </ol>
   *
   * <h3>Why this is safe</h3>
   * <p>The old allocator instance (and its {@code allocator_and_pools}
   * containing the PyTorch {@code MemPool} objects) is <em>not</em>
   * destroyed. It stays alive because the old engine's Python objects
   * are intentionally leaked (we skip {@code decref(engine)} and
   * {@code gc.collect()} in {@link #close()}) and they hold a reference
   * chain back to the allocator via the model executor → workers →
   * memory pool context.
   *
   * <p>This avoids the {@code MemPool::~MemPool()} crash
   * ({@code captures_underway.empty() INTERNAL ASSERT FAILED}) that
   * happens whenever PyTorch pool objects are garbage-collected after
   * {@code sleep()} has unmapped the underlying GPU memory.
   *
   * <p>The CPU-side cost of the leaked objects is small (~10-20 MB).
   *
   * <p>Best-effort — silently ignores errors.
   */
  private void resetCuMemAllocator() {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment cumemModule = CPythonBinding.PyImport_ImportModule(
        tmp.allocateFrom("vllm.device_allocator.cumem")
      );
      if (PythonTypes.isNull(cumemModule)) {
        CPythonBinding.PyErr_Clear();
        return;
      }
      MemorySegment cls = PythonTypes.getAttr(
        tmp,
        cumemModule,
        "CuMemAllocator"
      );
      if (PythonTypes.isNull(cls) || PythonTypes.isNone(cls)) {
        PythonTypes.decref(cls);
        PythonTypes.decref(cumemModule);
        return;
      }
      MemorySegment instance = PythonTypes.getAttr(tmp, cls, "instance");
      if (!PythonTypes.isNull(instance) && !PythonTypes.isNone(instance)) {
        // Clear pointer_to_data dict — stale entries from sleep()
        MemorySegment pointerToData = PythonTypes.getAttr(
          tmp,
          instance,
          "pointer_to_data"
        );
        if (
          !PythonTypes.isNull(pointerToData) &&
          !PythonTypes.isNone(pointerToData)
        ) {
          MemorySegment clearName = PythonTypes.pyStr(tmp, "clear");
          MemorySegment clearResult = PythonCall.callMethodObjArgs(
            pointerToData,
            clearName
          );
          PythonTypes.decref(clearResult);
          PythonTypes.decref(clearName);
          PythonTypes.decref(pointerToData);
          System.out.println(
            "[vLLM4j] resetCuMemAllocator: cleared pointer_to_data"
          );
          System.out.flush();
        }
        PythonTypes.decref(instance);
      }

      // Set CuMemAllocator.instance = None so the next engine creates
      // a completely fresh allocator. The old instance stays alive
      // (prevented from GC by the leaked engine's reference chain).
      int rc = CPythonBinding.PyObject_SetAttrString(
        cls,
        tmp.allocateFrom("instance"),
        PythonTypes.pyNone()
      );
      if (rc != 0) {
        System.out.println(
          "[vLLM4j] resetCuMemAllocator: failed to set instance = None"
        );
        System.out.flush();
        CPythonBinding.PyErr_Clear();
      } else {
        System.out.println(
          "[vLLM4j] resetCuMemAllocator: set CuMemAllocator.instance = None"
        );
        System.out.flush();
      }

      PythonTypes.decref(cls);
      PythonTypes.decref(cumemModule);
    } catch (Exception e) {
      System.out.println(
        "[vLLM4j] resetCuMemAllocator: EXCEPTION — " + e.getMessage()
      );
      System.out.flush();
      CPythonBinding.PyErr_Clear();
    }
  }

  /**
   * Puts the engine to sleep, releasing GPU memory via vLLM's
   * {@code CuMemAllocator}.
   *
   * <p>vLLM V1 uses a custom CUDA allocator ({@code CuMemAllocator}) that
   * manages GPU memory via {@code cuMemCreate}/{@code cuMemMap} instead of
   * {@code cudaMalloc}. Standard {@code torch.cuda.empty_cache()} only
   * flushes PyTorch's default caching allocator and has <em>no effect</em>
   * on CuMemAllocator-managed memory.
   *
   * <p>The {@code sleep(level=1)} method unmaps physical GPU memory, offloads
   * model weights to CPU, and releases KV cache back to the driver.
   *
   * <p>Best-effort — silently ignores errors.
   */
  private void sleepEngine() {
    try (Arena tmp = Arena.ofConfined()) {
      // Navigate: engine → engine_core (InprocClient) → engine_core (EngineCore)
      MemorySegment client = PythonTypes.getAttr(tmp, engine, "engine_core");
      if (PythonTypes.isNull(client) || PythonTypes.isNone(client)) {
        System.out.println("[vLLM4j] sleepEngine: no engine_core, skipping");
        System.out.flush();
        PythonTypes.decref(client);
        return;
      }
      MemorySegment core = PythonTypes.getAttr(tmp, client, "engine_core");
      if (PythonTypes.isNull(core) || PythonTypes.isNone(core)) {
        System.out.println(
          "[vLLM4j] sleepEngine: no inner engine_core, skipping"
        );
        System.out.flush();
        PythonTypes.decref(core);
        PythonTypes.decref(client);
        return;
      }
      MemorySegment executor = PythonTypes.getAttr(tmp, core, "model_executor");
      if (PythonTypes.isNull(executor) || PythonTypes.isNone(executor)) {
        System.out.println("[vLLM4j] sleepEngine: no model_executor, skipping");
        System.out.flush();
        PythonTypes.decref(executor);
        PythonTypes.decref(core);
        PythonTypes.decref(client);
        return;
      }

      // Call: executor.collective_rpc("sleep", args=(1,))
      MemorySegment sleepStr = PythonTypes.pyStr(tmp, "sleep");
      MemorySegment levelOne = CPythonBinding.PyLong_FromLong(1L);
      MemorySegment innerArgs = PythonCall.makeTuple(levelOne);
      PythonTypes.decref(levelOne);

      // Build kwargs dict: {"args": (1,)}
      MemorySegment kwargs = CPythonBinding.PyDict_New();
      CPythonBinding.PyDict_SetItemString(
        kwargs,
        tmp.allocateFrom("args"),
        innerArgs
      );

      // Build positional args: ("sleep",)
      MemorySegment posArgs = PythonCall.makeTuple(sleepStr);

      // Call: executor.collective_rpc("sleep", args=(1,))
      // Using PyObject_Call on the bound method
      MemorySegment method = PythonTypes.getAttr(
        tmp,
        executor,
        "collective_rpc"
      );
      if (!PythonTypes.isNull(method)) {
        MemorySegment result = PythonCall.pyObjectCall(method, posArgs, kwargs);
        if (PythonTypes.isNull(result)) {
          System.out.println(
            "[vLLM4j] sleepEngine: collective_rpc('sleep') failed, clearing error"
          );
          System.out.flush();
          CPythonBinding.PyErr_Clear();
        } else {
          System.out.println(
            "[vLLM4j] sleepEngine: sleep(1) succeeded — GPU memory released"
          );
          System.out.flush();
          PythonTypes.decref(result);
        }
        PythonTypes.decref(method);
      }

      PythonTypes.decref(posArgs);
      PythonTypes.decref(kwargs);
      PythonTypes.decref(innerArgs);
      PythonTypes.decref(sleepStr);
      PythonTypes.decref(executor);
      PythonTypes.decref(core);
      PythonTypes.decref(client);
    } catch (Exception e) {
      System.out.println("[vLLM4j] sleepEngine: EXCEPTION — " + e.getMessage());
      System.out.flush();
      CPythonBinding.PyErr_Clear();
    }
  }

  /**
   * Calls {@code engine.engine_core.engine_core.model_executor.shutdown()}
   * to tear down the model executor and release GPU resources.
   *
   * <p>Best-effort — silently ignores errors.
   */
  private void shutdownEngineCore() {
    try {
      System.out.println(
        "[vLLM4j] shutdownEngineCore: getAttr(engine, 'engine_core')"
      );
      System.out.flush();
      MemorySegment engineCore = PythonTypes.getAttr(
        arena,
        engine,
        "engine_core"
      );
      if (!PythonTypes.isNull(engineCore) && !PythonTypes.isNone(engineCore)) {
        System.out.println(
          "[vLLM4j] shutdownEngineCore: getAttr(engineCore, 'engine_core') [inner]"
        );
        System.out.flush();
        // engine_core may be an InprocClient wrapping the real EngineCore
        MemorySegment innerCore = PythonTypes.getAttr(
          arena,
          engineCore,
          "engine_core"
        );
        MemorySegment target = (!PythonTypes.isNull(innerCore) &&
            !PythonTypes.isNone(innerCore))
          ? innerCore
          : engineCore;
        System.out.println(
          "[vLLM4j] shutdownEngineCore: calling shutdown() on " +
            (target == innerCore ? "inner engine_core" : "engine_core")
        );
        System.out.flush();
        MemorySegment shutdownName = PythonTypes.pyStr(arena, "shutdown");
        MemorySegment result = PythonCall.callMethodObjArgs(
          target,
          shutdownName
        );
        System.out.println("[vLLM4j] shutdownEngineCore: shutdown() returned");
        System.out.flush();
        PythonTypes.decref(result);
        PythonTypes.decref(shutdownName);
        if (target != engineCore) PythonTypes.decref(innerCore);
      } else {
        System.out.println(
          "[vLLM4j] shutdownEngineCore: engine_core is null/None, skipping"
        );
        System.out.flush();
      }
      PythonTypes.decref(engineCore);
    } catch (Exception e) {
      System.out.println(
        "[vLLM4j] shutdownEngineCore: EXCEPTION — " + e.getMessage()
      );
      System.out.flush();
      CPythonBinding.PyErr_Clear();
    }
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
      MemorySegment gcModule = CPythonBinding.PyImport_ImportModule(
        tmp.allocateFrom("gc")
      );
      if (PythonTypes.isNull(gcModule)) {
        CPythonBinding.PyErr_Clear();
        return;
      }
      MemorySegment collectName = PythonTypes.pyStr(tmp, "collect");
      MemorySegment result = PythonCall.callMethodObjArgs(
        gcModule,
        collectName
      );
      PythonTypes.decref(result);
      PythonTypes.decref(collectName);
      PythonTypes.decref(gcModule);
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  private MemorySegment buildEngineArgsKwargs(String model, String dtype) {
    MemorySegment kwargs = CPythonBinding.PyDict_New();
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
    if (b.gpuMemoryUtilization() != null) PythonTypes.putDictFloat(
      arena,
      kwargs,
      "gpu_memory_utilization",
      b.gpuMemoryUtilization()
    );
    if (b.maxModelLen() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "max_model_len",
      b.maxModelLen()
    );
    if (b.maxNumSeqs() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "max_num_seqs",
      b.maxNumSeqs()
    );
    if (b.maxNumBatchedTokens() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "max_num_batched_tokens",
      b.maxNumBatchedTokens()
    );
    if (b.enforceEager() != null) PythonTypes.putDictObj(
      arena,
      kwargs,
      "enforce_eager",
      b.enforceEager() ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
    );
    if (b.trustRemoteCode() != null) PythonTypes.putDictObj(
      arena,
      kwargs,
      "trust_remote_code",
      b.trustRemoteCode() ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
    );
    if (b.quantization() != null) {
      MemorySegment pyQuant = PythonTypes.pyStr(arena, b.quantization());
      PythonTypes.putDictObj(arena, kwargs, "quantization", pyQuant);
      PythonTypes.decref(pyQuant);
    }
    if (b.swapSpace() != null) PythonTypes.putDictFloat(
      arena,
      kwargs,
      "swap_space",
      b.swapSpace()
    );
    if (b.seed() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "seed",
      b.seed()
    );
    if (b.enablePrefixCaching() != null) PythonTypes.putDictObj(
      arena,
      kwargs,
      "enable_prefix_caching",
      b.enablePrefixCaching() ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
    );
    if (b.enableLora() != null) PythonTypes.putDictObj(
      arena,
      kwargs,
      "enable_lora",
      b.enableLora() ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
    );
    if (b.maxLoras() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "max_loras",
      b.maxLoras()
    );
    if (b.maxLoraRank() != null) PythonTypes.putDictInt(
      arena,
      kwargs,
      "max_lora_rank",
      b.maxLoraRank()
    );
    if (b.enableChunkedPrefill() != null) PythonTypes.putDictObj(
      arena,
      kwargs,
      "enable_chunked_prefill",
      b.enableChunkedPrefill() ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
    );
    if (b.kvCacheDtype() != null) {
      MemorySegment pyKvDtype = PythonTypes.pyStr(arena, b.kvCacheDtype());
      PythonTypes.putDictObj(arena, kwargs, "kv_cache_dtype", pyKvDtype);
      PythonTypes.decref(pyKvDtype);
    }

    // Enable sleep mode so that vLLM allocates GPU memory through
    // CuMemAllocator (cuMemCreate/cuMemMap) instead of cudaMalloc.
    // Required for sleepEngine() to release GPU memory on close().
    // Only supported on CUDA — vllm-metal does not have CuMemAllocator.
    if (b.isSleepModeEnabled()) {
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "enable_sleep_mode",
        PythonTypes.pyTrue()
      );
    }

    // Metal does not support chunked prefill. Disable it explicitly unless
    // the caller has already set a value, so that vllm-metal's
    // check_and_update_config() never reaches the branch that accesses
    // SchedulerConfig.max_num_scheduled_tokens — a field that was
    // introduced in vllm core after 0.16.0 and is absent in older releases.
    if (
      PlatformResolver.backend() == VllmBackend.METAL &&
      b.enableChunkedPrefill() == null
    ) {
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "enable_chunked_prefill",
        PythonTypes.pyFalse()
      );
    }

    // Distributed inference configuration
    if (b.tensorParallelSize() != null) {
      PythonTypes.putDictInt(
        arena,
        kwargs,
        "tensor_parallel_size",
        b.tensorParallelSize()
      );
    }
    if (b.pipelineParallelSize() != null) {
      PythonTypes.putDictInt(
        arena,
        kwargs,
        "pipeline_parallel_size",
        b.pipelineParallelSize()
      );
    }
    if (b.distributedExecutorBackend() != null) {
      MemorySegment pyBackend = PythonTypes.pyStr(
        arena,
        b.distributedExecutorBackend()
      );
      PythonTypes.putDictObj(
        arena,
        kwargs,
        "distributed_executor_backend",
        pyBackend
      );
      PythonTypes.decref(pyBackend);
    }

    return kwargs;
  }

  private List<RequestOutput> mapRequestOutputList(MemorySegment pyList) {
    long size = CPythonBinding.PyList_Size(pyList);
    List<RequestOutput> result = new ArrayList<>((int) size);
    for (long i = 0; i < size; i++) {
      MemorySegment pyReqOut = CPythonBinding.PyList_GetItem(pyList, i); // borrowed
      result.add(mapRequestOutput(pyReqOut));
    }
    return result;
  }

  private RequestOutput mapRequestOutput(MemorySegment pyReqOut) {
    String requestId = PythonTypes.getStrAttr(arena, pyReqOut, "request_id");
    boolean finished = PythonTypes.getBoolAttr(arena, pyReqOut, "finished");

    MemorySegment pyOutputs = PythonTypes.getAttr(arena, pyReqOut, "outputs");
    long size = CPythonBinding.PyList_Size(pyOutputs);
    List<CompletionOutput> completions = new ArrayList<>((int) size);
    for (long i = 0; i < size; i++) {
      MemorySegment pyComp = CPythonBinding.PyList_GetItem(pyOutputs, i); // borrowed
      completions.add(mapCompletionOutput(pyComp));
    }
    PythonTypes.decref(pyOutputs);

    // prompt_token_ids — just get the count via PyList_Size
    int numPromptTokens = 0;
    MemorySegment pyPromptIds = PythonTypes.getAttr(
      arena,
      pyReqOut,
      "prompt_token_ids"
    );
    if (!PythonTypes.isNone(pyPromptIds) && !PythonTypes.isNull(pyPromptIds)) {
      long len = CPythonBinding.PyList_Size(pyPromptIds);
      if (len >= 0) {
        numPromptTokens = (int) len;
      }
      CPythonBinding.PyErr_Clear();
    }
    PythonTypes.decref(pyPromptIds);

    // num_cached_tokens (may be None → 0)
    int numCachedTokens = 0;
    MemorySegment pyCached = PythonTypes.getAttr(
      arena,
      pyReqOut,
      "num_cached_tokens"
    );
    if (!PythonTypes.isNone(pyCached) && !PythonTypes.isNull(pyCached)) {
      numCachedTokens = (int) CPythonBinding.PyLong_AsLong(pyCached);
    }
    PythonTypes.decref(pyCached);
    CPythonBinding.PyErr_Clear(); // clear any AttributeError if field missing

    // metrics
    RequestMetrics metrics = mapMetrics(
      PythonTypes.getAttr(arena, pyReqOut, "metrics"),
      numPromptTokens
    );

    // prompt_logprobs (may be None)
    List<Map<Integer, LogprobEntry>> promptLogprobs = null;
    try {
      MemorySegment pyPromptLogprobs = PythonTypes.getAttr(
        arena,
        pyReqOut,
        "prompt_logprobs"
      );
      promptLogprobs = mapLogprobsList(pyPromptLogprobs);
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
    }

    return new RequestOutput(
      requestId,
      completions,
      finished,
      null,
      numCachedTokens,
      metrics,
      promptLogprobs
    );
  }

  private CompletionOutput mapCompletionOutput(MemorySegment pyComp) {
    int index = (int) PythonTypes.getLongAttr(arena, pyComp, "index");
    String text = PythonTypes.getStrAttr(arena, pyComp, "text");

    // token_ids
    List<Integer> tokenIds = mapIntList(
      PythonTypes.getAttr(arena, pyComp, "token_ids")
    );

    // finish_reason (str or None)
    MemorySegment pyFinishReason = PythonTypes.getAttr(
      arena,
      pyComp,
      "finish_reason"
    );
    FinishReason finishReason = null;
    if (
      !PythonTypes.isNone(pyFinishReason) && !PythonTypes.isNull(pyFinishReason)
    ) {
      finishReason = FinishReason.fromVllmString(
        PythonTypes.pyUnicodeToString(pyFinishReason)
      );
    }
    PythonTypes.decref(pyFinishReason);

    // logprobs (may be None)
    List<Map<Integer, LogprobEntry>> logprobs = null;
    try {
      MemorySegment pyLogprobs = PythonTypes.getAttr(arena, pyComp, "logprobs");
      logprobs = mapLogprobsList(pyLogprobs);
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
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
  private List<Map<Integer, LogprobEntry>> mapLogprobsList(
    MemorySegment pyLogprobs
  ) {
    if (PythonTypes.isNone(pyLogprobs) || PythonTypes.isNull(pyLogprobs)) {
      PythonTypes.decref(pyLogprobs);
      CPythonBinding.PyErr_Clear();
      return null;
    }
    long size = CPythonBinding.PyList_Size(pyLogprobs);
    List<Map<Integer, LogprobEntry>> result = new ArrayList<>((int) size);
    for (long i = 0; i < size; i++) {
      MemorySegment pyDict = CPythonBinding.PyList_GetItem(pyLogprobs, i); // borrowed
      if (PythonTypes.isNone(pyDict) || PythonTypes.isNull(pyDict)) {
        result.add(null); // position has no logprobs (e.g. first prompt token)
        continue;
      }
      // Iterate the Python dict: {int token_id: Logprob}
      Map<Integer, LogprobEntry> posMap = new HashMap<>();
      MemorySegment pyMethodName = PythonTypes.pyStr(arena, "items");
      MemorySegment pyItemsView = PythonCall.callMethodObjArgs(
        pyDict,
        pyMethodName
      );
      PythonTypes.decref(pyMethodName);
      if (!PythonTypes.isNull(pyItemsView)) {
        MemorySegment pyIter = CPythonBinding.PyObject_GetIter(pyItemsView);
        PythonTypes.decref(pyItemsView);
        if (!PythonTypes.isNull(pyIter)) {
          MemorySegment pyTuple;
          while (
            !PythonTypes.isNull(pyTuple = CPythonBinding.PyIter_Next(pyIter))
          ) {
            int tokenId = (int) CPythonBinding.PyLong_AsLong(
              CPythonBinding.PyTuple_GetItem(pyTuple, 0)
            );
            MemorySegment pyLogprob = CPythonBinding.PyTuple_GetItem(
              pyTuple,
              1
            ); // borrowed

            double logprob = PythonTypes.getDoubleAttr(
              arena,
              pyLogprob,
              "logprob"
            );
            int rank = (int) PythonTypes.getLongAttr(arena, pyLogprob, "rank");
            String decodedToken = "";
            try {
              decodedToken = PythonTypes.getStrAttr(
                arena,
                pyLogprob,
                "decoded_token"
              );
            } catch (Exception e) {
              CPythonBinding.PyErr_Clear();
            }
            posMap.put(
              tokenId,
              new LogprobEntry(tokenId, logprob, rank, decodedToken)
            );
            PythonTypes.decref(pyTuple); // PyIter_Next returns new reference
          }
          PythonTypes.decref(pyIter);
        }
      } else {
        CPythonBinding.PyErr_Clear();
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
      CPythonBinding.PyErr_Clear();
      return List.of();
    }
    long size = CPythonBinding.PyList_Size(pyList);
    if (size < 0) {
      // Not a list (V1 may return a tuple) — skip extraction, rely on metrics
      CPythonBinding.PyErr_Clear();
      PythonTypes.decref(pyList);
      return List.of();
    }
    List<Integer> result = new ArrayList<>((int) size);
    for (long i = 0; i < size; i++) {
      MemorySegment item = CPythonBinding.PyList_GetItem(pyList, i); // borrowed
      result.add((int) CPythonBinding.PyLong_AsLong(item));
    }
    PythonTypes.decref(pyList);
    return result;
  }

  /** Maps Python RequestMetrics to Java RequestMetrics. Returns null if None. */
  /**
   * Maps vLLM v1 {@code RequestStateStats} to Java {@link RequestMetrics}.
   *
   * <p>v1 field names differ from v0:
   * <ul>
   *   <li>{@code arrival_time} (wall-clock, same in both)
   *   <li>{@code last_token_ts} (v1) vs {@code last_token_time} (v0)
   *   <li>{@code scheduled_ts} (v1) vs {@code first_scheduled_time} (v0)
   *   <li>{@code first_token_ts} (v1) vs {@code first_token_time} (v0)
   *   <li>{@code queued_ts} (v1) vs {@code time_in_queue} (v0)
   *   <li>{@code first_token_latency} (v1 only)
   *   <li>{@code num_generation_tokens} (v1 only)
   *   <li>no {@code finished_time} in v1
   * </ul>
   */
  private RequestMetrics mapMetrics(
    MemorySegment pyMetrics,
    int numPromptTokens
  ) {
    if (PythonTypes.isNone(pyMetrics) || PythonTypes.isNull(pyMetrics)) {
      PythonTypes.decref(pyMetrics);
      CPythonBinding.PyErr_Clear();
      return null;
    }
    double arrivalTime = -1,
      lastTokenTime = -1,
      firstScheduledTime = -1;
    double firstTokenTime = -1,
      timeInQueue = -1,
      finishedTime = -1;
    double firstTokenLatency = -1;
    int numGenerationTokens = 0;
    try {
      arrivalTime = safeGetDouble(pyMetrics, "arrival_time");
      // v1 field names (fall back to v0 names if missing)
      lastTokenTime = safeGetDouble(pyMetrics, "last_token_ts");
      if (lastTokenTime < 0) lastTokenTime = safeGetDouble(
        pyMetrics,
        "last_token_time"
      );
      firstScheduledTime = safeGetDouble(pyMetrics, "scheduled_ts");
      if (firstScheduledTime < 0) firstScheduledTime = safeGetDouble(
        pyMetrics,
        "first_scheduled_time"
      );
      firstTokenTime = safeGetDouble(pyMetrics, "first_token_ts");
      if (firstTokenTime < 0) firstTokenTime = safeGetDouble(
        pyMetrics,
        "first_token_time"
      );
      timeInQueue = safeGetDouble(pyMetrics, "queued_ts");
      if (timeInQueue < 0) timeInQueue = safeGetDouble(
        pyMetrics,
        "time_in_queue"
      );
      finishedTime = safeGetDouble(pyMetrics, "finished_time");
      firstTokenLatency = safeGetDouble(pyMetrics, "first_token_latency");
      numGenerationTokens = safeGetInt(pyMetrics, "num_generation_tokens");
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
    }
    PythonTypes.decref(pyMetrics);
    return new RequestMetrics(
      arrivalTime,
      lastTokenTime,
      firstScheduledTime,
      firstTokenTime,
      timeInQueue,
      finishedTime,
      firstTokenLatency,
      numGenerationTokens,
      numPromptTokens
    );
  }

  /** Safely reads a double attribute, returning -1 if None/missing. */
  private double safeGetDouble(MemorySegment obj, String name) {
    try {
      MemorySegment attr = PythonTypes.getAttr(arena, obj, name);
      if (PythonTypes.isNone(attr) || PythonTypes.isNull(attr)) {
        PythonTypes.decref(attr);
        CPythonBinding.PyErr_Clear();
        return -1;
      }
      double val = CPythonBinding.PyFloat_AsDouble(attr);
      PythonTypes.decref(attr);
      return val;
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
      return -1;
    }
  }

  /** Safely reads an int attribute, returning 0 if None/missing. */
  private int safeGetInt(MemorySegment obj, String name) {
    try {
      MemorySegment attr = PythonTypes.getAttr(arena, obj, name);
      if (PythonTypes.isNone(attr) || PythonTypes.isNull(attr)) {
        PythonTypes.decref(attr);
        CPythonBinding.PyErr_Clear();
        return 0;
      }
      int val = (int) CPythonBinding.PyLong_AsLong(attr);
      PythonTypes.decref(attr);
      return val;
    } catch (Exception e) {
      CPythonBinding.PyErr_Clear();
      return 0;
    }
  }

  private void checkNotClosed() {
    if (closed.get()) throw new IllegalStateException(
      "VllmEngine has been closed"
    );
  }
}
