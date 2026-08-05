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
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.runtime.GIL;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Locale;

/**
 * Reads a model's shape from its HuggingFace {@code config.json} without
 * loading any weights.
 *
 * <p>Exists so a caller can decide whether a model will fit in VRAM
 * <em>before</em> paying to load it. Everything here is read-only metadata: no
 * engine is constructed and nothing is moved to the device.
 *
 * <p>Resolution goes through vLLM's own {@code get_config}, which means a
 * HuggingFace repo id, a local directory and a gated repo behind {@code HF_TOKEN}
 * all behave the same way, and {@code trust_remote_code} is honoured for
 * architectures that ship their own config class.
 *
 * <p>Every field degrades to 0 / false rather than throwing: an unknown shape
 * should downgrade the caller's estimate to "cannot tell", never break model
 * loading.
 *
 * <p>Requires the CPython runtime to be initialised
 * ({@link VllmEngineBuilder#initRuntime()}).
 */
public final class ModelIntrospection {

  private ModelIntrospection() {}

  /**
   * A model's memory-relevant shape.
   *
   * @param numHiddenLayers      transformer layer count, for KV-cache sizing
   * @param numKvHeads           key/value head count (falls back to the attention
   *                             head count for models without GQA)
   * @param headDim              per-head dimension (derived from
   *                             {@code hidden_size / num_attention_heads} when absent)
   * @param maxPositionEmbeddings longest sequence the positional encoding supports,
   *                             i.e. the context length vLLM defaults to
   * @param multimodal           whether the config declares a vision or audio tower
   * @param totalParams          parameter count, or 0 when it cannot be determined
   * @param bitsPerParam         storage width per parameter in bits, accounting
   *                             for quantization (so 4-bit AWQ is 4, not a
   *                             rounded-up byte), or 0 when unknown
   */
  public record ModelShape(
    int numHiddenLayers,
    int numKvHeads,
    int headDim,
    int maxPositionEmbeddings,
    boolean multimodal,
    long totalParams,
    int bitsPerParam
  ) {
    /** An entirely unknown shape — the caller should skip its estimate. */
    public static final ModelShape UNKNOWN = new ModelShape(
      0,
      0,
      0,
      0,
      false,
      0,
      0
    );

    /** True when there is enough here to size the weights and the KV cache. */
    public boolean isUsable() {
      // The weights need totalParams and bitsPerParam; the KV cache needs the
      // layer count and the per-layer geometry.
      return (
        totalParams > 0 &&
        bitsPerParam > 0 &&
        numHiddenLayers > 0 &&
        numKvHeads > 0 &&
        headDim > 0
      );
    }
  }

  /**
   * Reads {@code model}'s shape.
   *
   * @param arena           arena for native string allocation
   * @param model           HuggingFace repo id or local directory
   * @param trustRemoteCode whether to allow a custom config class from the repo
   * @return the shape, or {@link ModelShape#UNKNOWN} if it cannot be read
   */
  public static ModelShape read(
    Arena arena,
    String model,
    boolean trustRemoteCode
  ) {
    if (model == null || model.isBlank()) {
      return ModelShape.UNKNOWN;
    }
    try (var gil = GIL.acquire()) {
      MemorySegment config = loadConfig(arena, model, trustRemoteCode);
      if (config == null) {
        return ModelShape.UNKNOWN;
      }
      try {
        int attentionHeads = intAttr(arena, config, "num_attention_heads");
        int hiddenSize = intAttr(arena, config, "hidden_size");

        int kvHeads = intAttr(arena, config, "num_key_value_heads");
        if (kvHeads <= 0) {
          // No GQA/MQA: every attention head carries its own KV.
          kvHeads = attentionHeads;
        }

        int headDim = intAttr(arena, config, "head_dim");
        if (headDim <= 0 && attentionHeads > 0) {
          headDim = hiddenSize / attentionHeads;
        }

        return new ModelShape(
          intAttr(arena, config, "num_hidden_layers"),
          kvHeads,
          headDim,
          intAttr(arena, config, "max_position_embeddings"),
          hasAttr(arena, config, "vision_config") ||
            hasAttr(arena, config, "audio_config"),
          readTotalParams(arena, model),
          readBitsPerParam(arena, config)
        );
      } finally {
        PythonTypes.decref(config);
      }
    } catch (RuntimeException e) {
      CPythonBinding.PyErr_Clear();
      return ModelShape.UNKNOWN;
    }
  }

  /** Calls {@code vllm.transformers_utils.config.get_config(model, trust_remote_code)}. */
  private static MemorySegment loadConfig(
    Arena arena,
    String model,
    boolean trustRemoteCode
  ) {
    MemorySegment getConfig = null;
    MemorySegment args = null;
    MemorySegment pyModel = null;
    try {
      getConfig = PythonCall.importClass(
        arena,
        "vllm.transformers_utils.config",
        "get_config"
      );
      pyModel = PythonTypes.pyStr(arena, model);
      args = PythonCall.makeTuple(
        pyModel,
        trustRemoteCode ? PythonTypes.pyTrue() : PythonTypes.pyFalse()
      );
      MemorySegment config = PythonCall.pyObjectCall(
        getConfig,
        args,
        MemorySegment.NULL
      );
      if (PythonTypes.isNull(config) || PythonTypes.isNone(config)) {
        CPythonBinding.PyErr_Clear();
        return null;
      }
      return config;
    } catch (RuntimeException e) {
      // Offline, gated without a token, or an architecture needing
      // trust_remote_code — none of which should stop the model loading.
      CPythonBinding.PyErr_Clear();
      return null;
    } finally {
      if (args != null) PythonTypes.decref(args);
      if (pyModel != null) PythonTypes.decref(pyModel);
      if (getConfig != null) PythonTypes.decref(getConfig);
    }
  }

  /**
   * Reads the parameter count from the Hub's safetensors index
   * ({@code HfApi().model_info(repo).safetensors.total}).
   *
   * <p>Not in {@code config.json} — it is metadata the Hub computes. Returns 0
   * for a local directory or when the Hub is unreachable; the caller then falls
   * back to whatever it can measure itself.
   */
  private static long readTotalParams(Arena arena, String model) {
    if (java.nio.file.Files.isDirectory(java.nio.file.Path.of(model))) {
      // A local directory is not a Hub repo: model_info can only fail, after
      // a network timeout, on the one path most likely to be used offline.
      return 0;
    }
    MemorySegment hfApiClass = null;
    MemorySegment api = null;
    MemorySegment info = null;
    MemorySegment safetensors = null;
    MemorySegment noArgs = null;
    try {
      hfApiClass = PythonCall.importClass(arena, "huggingface_hub", "HfApi");
      noArgs = PythonCall.makeTuple();
      api = PythonCall.pyObjectCall(hfApiClass, noArgs, MemorySegment.NULL);
      if (PythonTypes.isNull(api)) {
        CPythonBinding.PyErr_Clear();
        return 0;
      }

      MemorySegment method = PythonTypes.pyStr(arena, "model_info");
      MemorySegment pyModel = PythonTypes.pyStr(arena, model);
      info = PythonCall.callMethodObjArgs(api, method, pyModel);
      PythonTypes.decref(method);
      PythonTypes.decref(pyModel);
      if (PythonTypes.isNull(info) || PythonTypes.isNone(info)) {
        CPythonBinding.PyErr_Clear();
        return 0;
      }

      safetensors = PythonTypes.getAttr(arena, info, "safetensors");
      if (PythonTypes.isNull(safetensors) || PythonTypes.isNone(safetensors)) {
        CPythonBinding.PyErr_Clear();
        return 0;
      }
      MemorySegment total = PythonTypes.getAttr(arena, safetensors, "total");
      long value = PythonTypes.isNone(total) || PythonTypes.isNull(total)
        ? 0
        : CPythonBinding.PyLong_AsLong(total);
      PythonTypes.decref(total);
      CPythonBinding.PyErr_Clear();
      return Math.max(value, 0);
    } catch (RuntimeException e) {
      CPythonBinding.PyErr_Clear();
      return 0;
    } finally {
      if (safetensors != null) PythonTypes.decref(safetensors);
      if (info != null) PythonTypes.decref(info);
      if (api != null) PythonTypes.decref(api);
      if (noArgs != null) PythonTypes.decref(noArgs);
      if (hfApiClass != null) PythonTypes.decref(hfApiClass);
    }
  }

  /**
   * Storage width of one parameter.
   *
   * <p>A {@code quantization_config} wins over the dtype: an AWQ or GPTQ
   * checkpoint declares {@code dtype: float16} for its activations while the
   * weights are 4-bit, and taking the dtype at face value would overstate the
   * weights by 4x.
   */
  private static int readBitsPerParam(Arena arena, MemorySegment config) {
    int quantBits = readQuantizationBits(arena, config);
    if (quantBits > 0) {
      return quantBits;
    }
    // "dtype" since transformers v5; "torch_dtype" before it.
    String dtype = strAttr(arena, config, "dtype");
    if (dtype.isEmpty()) {
      dtype = strAttr(arena, config, "torch_dtype");
    }
    return bitsForDtype(dtype);
  }

  /** Bits per weight from {@code quantization_config.bits}, or 0 when unquantized. */
  private static int readQuantizationBits(Arena arena, MemorySegment config) {
    MemorySegment quant = PythonTypes.getAttr(
      arena,
      config,
      "quantization_config"
    );
    if (PythonTypes.isNull(quant) || PythonTypes.isNone(quant)) {
      CPythonBinding.PyErr_Clear();
      return 0;
    }
    try {
      // Usually a dict, occasionally a config object — try the attribute, and
      // fall back to reading it as a mapping.
      MemorySegment bits = PythonTypes.getAttr(arena, quant, "bits");
      if (!PythonTypes.isNull(bits) && !PythonTypes.isNone(bits)) {
        long value = CPythonBinding.PyLong_AsLong(bits);
        PythonTypes.decref(bits);
        CPythonBinding.PyErr_Clear();
        return (int) Math.max(value, 0);
      }
      PythonTypes.decref(bits);
      CPythonBinding.PyErr_Clear();

      MemorySegment get = PythonTypes.pyStr(arena, "get");
      MemorySegment key = PythonTypes.pyStr(arena, "bits");
      MemorySegment value = PythonCall.callMethodObjArgs(quant, get, key);
      PythonTypes.decref(get);
      PythonTypes.decref(key);
      long bitsValue = PythonTypes.isNull(value) || PythonTypes.isNone(value)
        ? 0
        : CPythonBinding.PyLong_AsLong(value);
      if (value != null) PythonTypes.decref(value);
      CPythonBinding.PyErr_Clear();
      return (int) Math.max(bitsValue, 0);
    } finally {
      PythonTypes.decref(quant);
    }
  }

  /** Maps a torch dtype name to its width in bytes. 0 when unrecognised. */
  private static int bitsForDtype(String dtype) {
    String normalized = dtype.toLowerCase(Locale.ENGLISH);
    if (
      normalized.contains("float32") || normalized.contains("int32")
    ) return 32;
    if (
      normalized.contains("bfloat16") || normalized.contains("float16")
    ) return 16;
    if (normalized.contains("int16")) return 16;
    if (normalized.contains("float8") || normalized.contains("int8")) return 8;
    return 0;
  }

  private static boolean hasAttr(Arena arena, MemorySegment obj, String name) {
    MemorySegment value = PythonTypes.getAttr(arena, obj, name);
    boolean present = !PythonTypes.isNull(value) && !PythonTypes.isNone(value);
    PythonTypes.decref(value);
    CPythonBinding.PyErr_Clear();
    return present;
  }

  private static int intAttr(Arena arena, MemorySegment obj, String name) {
    MemorySegment value = PythonTypes.getAttr(arena, obj, name);
    int result = 0;
    if (!PythonTypes.isNull(value) && !PythonTypes.isNone(value)) {
      result = (int) CPythonBinding.PyLong_AsLong(value);
    }
    PythonTypes.decref(value);
    CPythonBinding.PyErr_Clear();
    return Math.max(result, 0);
  }

  private static String strAttr(Arena arena, MemorySegment obj, String name) {
    MemorySegment value = PythonTypes.getAttr(arena, obj, name);
    if (PythonTypes.isNull(value) || PythonTypes.isNone(value)) {
      PythonTypes.decref(value);
      CPythonBinding.PyErr_Clear();
      return "";
    }
    // dtype is a torch.dtype object, not a string — str() it.
    MemorySegment asStr = CPythonBinding.PyObject_Str(value);
    String result = PythonTypes.isNull(asStr)
      ? null
      : PythonTypes.pyUnicodeToString(asStr);
    PythonTypes.decref(asStr);
    PythonTypes.decref(value);
    CPythonBinding.PyErr_Clear();
    return result != null ? result : "";
  }
}
