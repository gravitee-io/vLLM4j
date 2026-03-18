package io.gravitee.vllm.engine;

import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.platform.PlatformResolver;
import io.gravitee.vllm.platform.VllmBackend;
import io.gravitee.vllm.runtime.PythonRuntime;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Fluent builder for {@link VllmEngine}.
 *
 * <p>Provides auto-detection of the {@code .venv} directory and compute
 * backend when not specified explicitly.
 *
 * <h2>Example</h2>
 * <pre>{@code
 * try (VllmEngine engine = VllmEngine.builder()
 *         .model("Qwen/Qwen3-0.6B")
 *         .dtype("auto")
 *         .build()) {
 *     // ...
 * }
 * }</pre>
 */
public final class VllmEngineBuilder {

    private Path venvPath;
    private String model = "Qwen/Qwen3-0.6B";
    private String dtype = "auto";
    private VllmBackend backend;

    private String tokenizer;
    private String hfConfigPath;

    // Engine configuration (nullable = use vLLM defaults)
    private Double gpuMemoryUtilization;
    private Integer maxModelLen;
    private Integer maxNumSeqs;
    private Integer maxNumBatchedTokens;
    private Boolean enforceEager;
    private Boolean trustRemoteCode;
    private String quantization;
    private Double swapSpace;
    private Integer seed;
    private Boolean enablePrefixCaching;
    private Boolean enableLora;
    private Integer maxLoras;
    private Integer maxLoraRank;
    private Boolean enableChunkedPrefill;
    private String kvCacheDtype;

    /** Runtime initialized by {@link #initRuntime()}, reused by {@link #build()}. */
    private PythonRuntime runtime;

    /**
     * Shared arena for all native memory allocations. Must outlive the engine.
     * If not set, the builder creates a {@link Arena#ofAuto()} arena.
     */
    private Arena arena;

    /** Package-private: obtain via {@link VllmEngine#builder()}. */
    VllmEngineBuilder() {}

    /**
     * Sets the absolute path to the uv-managed {@code .venv} directory.
     * When not set, {@link #build()} will auto-detect.
     */
    public VllmEngineBuilder venvPath(Path venvPath) {
        this.venvPath = venvPath;
        return this;
    }

    /** Sets the HuggingFace model id, e.g. {@code "Qwen/Qwen3-0.6B"}. */
    public VllmEngineBuilder model(String model) {
        this.model = model;
        return this;
    }

    /**
     * Sets an explicit tokenizer id, overriding the model's built-in tokenizer.
     * Recommended for GGUF models where the tokenizer conversion is slow and
     * unreliable — use the base model's tokenizer instead.
     *
     * <p>Example: {@code .tokenizer("Qwen/Qwen3.5-0.8B")}
     */
    public VllmEngineBuilder tokenizer(String tokenizer) {
        this.tokenizer = tokenizer;
        return this;
    }

    /**
     * Sets an explicit HuggingFace config path, overriding the config resolved
     * from the GGUF metadata. Useful when HuggingFace cannot auto-convert the
     * GGUF metadata to a model config.
     *
     * <p>Example: {@code .hfConfigPath("Qwen/Qwen3.5-0.8B")}
     */
    public VllmEngineBuilder hfConfigPath(String hfConfigPath) {
        this.hfConfigPath = hfConfigPath;
        return this;
    }

    /**
     * Sets the torch dtype string, e.g. {@code "auto"} or {@code "float32"}.
     * Defaults to {@code "auto"}.
     */
    public VllmEngineBuilder dtype(String dtype) {
        this.dtype = dtype;
        return this;
    }

    /**
     * Sets the compute backend explicitly. When not set, auto-detected via
     * {@link VllmBackend#detect()}.
     */
    public VllmEngineBuilder backend(VllmBackend backend) {
        this.backend = backend;
        return this;
    }

    /**
     * Sets the fraction of GPU memory to use for model weights and KV cache.
     * Default: 0.9 on CUDA, varies on other backends.
     */
    public VllmEngineBuilder gpuMemoryUtilization(double fraction) {
        this.gpuMemoryUtilization = fraction;
        return this;
    }

    /**
     * Overrides the model's maximum context length. Useful for limiting
     * memory usage or supporting longer contexts.
     */
    public VllmEngineBuilder maxModelLen(int maxModelLen) {
        this.maxModelLen = maxModelLen;
        return this;
    }

    /**
     * Sets the maximum number of concurrent sequences (batch size cap).
     */
    public VllmEngineBuilder maxNumSeqs(int maxNumSeqs) {
        this.maxNumSeqs = maxNumSeqs;
        return this;
    }

    /**
     * Sets the maximum number of tokens per batch.
     */
    public VllmEngineBuilder maxNumBatchedTokens(int maxNumBatchedTokens) {
        this.maxNumBatchedTokens = maxNumBatchedTokens;
        return this;
    }

    /**
     * When {@code true}, disables CUDA graphs and uses eager execution.
     * Useful for debugging or when CUDA graphs cause issues.
     */
    public VllmEngineBuilder enforceEager(boolean enforceEager) {
        this.enforceEager = enforceEager;
        return this;
    }

    /**
     * When {@code true}, allows loading models with custom code from
     * HuggingFace Hub. Required for many vision-language models.
     */
    public VllmEngineBuilder trustRemoteCode(boolean trustRemoteCode) {
        this.trustRemoteCode = trustRemoteCode;
        return this;
    }

    /**
     * Sets the quantization method, e.g. {@code "awq"}, {@code "gptq"},
     * {@code "squeezellm"}.
     */
    public VllmEngineBuilder quantization(String quantization) {
        this.quantization = quantization;
        return this;
    }

    /**
     * Sets the CPU swap space in GB for offloading KV cache.
     */
    public VllmEngineBuilder swapSpace(double swapSpaceGb) {
        this.swapSpace = swapSpaceGb;
        return this;
    }

    /**
     * Sets the global random seed for reproducibility.
     */
    public VllmEngineBuilder seed(int seed) {
        this.seed = seed;
        return this;
    }

    /**
     * Enables prefix caching (prompt caching) to reuse KV cache
     * for common prefixes across requests.
     */
    public VllmEngineBuilder enablePrefixCaching(boolean enable) {
        this.enablePrefixCaching = enable;
        return this;
    }

    /**
     * Enables LoRA adapter support.
     */
    public VllmEngineBuilder enableLora(boolean enable) {
        this.enableLora = enable;
        return this;
    }

    /**
     * Sets the maximum number of concurrent LoRA adapters.
     */
    public VllmEngineBuilder maxLoras(int maxLoras) {
        this.maxLoras = maxLoras;
        return this;
    }

    /**
     * Sets the maximum LoRA rank.
     */
    public VllmEngineBuilder maxLoraRank(int maxLoraRank) {
        this.maxLoraRank = maxLoraRank;
        return this;
    }

    /**
     * Enables chunked prefill for long prompts.
     */
    public VllmEngineBuilder enableChunkedPrefill(boolean enable) {
        this.enableChunkedPrefill = enable;
        return this;
    }

    /**
     * Sets the KV cache dtype, e.g. {@code "auto"}, {@code "fp8"}.
     */
    public VllmEngineBuilder kvCacheDtype(String kvCacheDtype) {
        this.kvCacheDtype = kvCacheDtype;
        return this;
    }

    /**
     * Sets the shared {@link Arena} for all native memory allocations.
     *
     * <p>The arena must outlive the engine. If not set, the builder creates
     * a {@link Arena#ofAuto()} arena that is garbage-collected when no longer
     * referenced.
     *
     * @param arena the arena to use for native allocations
     * @return this builder (for chaining)
     */
    public VllmEngineBuilder arena(Arena arena) {
        this.arena = arena;
        return this;
    }

    /**
     * Initializes the CPython runtime (and releases the GIL) without loading
     * the model.
     *
     * <p>Call this before any code that needs CPython — for example, a pre-flight
      * {@code GpuMemoryQuery} that calls {@code torch.cuda.mem_get_info()}.
     * A subsequent call to {@link #build()} will reuse the already-initialized
     * runtime instead of creating a new one.
     *
     * <p>Safe to call multiple times; the underlying {@code PythonRuntime} uses
     * an {@link java.util.concurrent.atomic.AtomicBoolean} guard to ensure
     * {@code Py_InitializeEx} is invoked at most once.
     *
     * @return this builder (for chaining)
     * @throws VllmException if no suitable venv can be found or initialization fails
     */
    public VllmEngineBuilder initRuntime() {
        if (runtime == null) {
            Path resolvedVenv = (venvPath != null) ? venvPath : resolveVenv();
            VllmBackend resolvedBackend = (backend != null) ? backend : PlatformResolver.backend();
            runtime = new PythonRuntime(resolvedVenv, resolvedBackend);
        }
        return this;
    }

    /**
     * Constructs a {@link VllmEngine}, initializing the CPython runtime (if not
     * already done via {@link #initRuntime()}) and loading the model.
     *
     * @throws VllmException if no suitable venv can be found or initialization fails
     */
    public VllmEngine build() {
        initRuntime();
        Arena resolvedArena = (arena != null) ? arena : Arena.ofAuto();
        return new VllmEngine(resolvedArena, this);
    }

    // ── Venv auto-detection ────────────────────────────────────────────────

    private static Path resolveVenv() {
        // 1. System property
        String prop = System.getProperty("vllm4j.venv");
        if (prop != null) {
            Path p = Path.of(prop);
            if (Files.isDirectory(p)) return p;
        }

        // 2. CWD/.venv
        Path cwd = Path.of(System.getProperty("user.dir"), ".venv");
        if (Files.isDirectory(cwd)) return cwd;

        // 3. HOME/.venv
        String home = System.getProperty("user.home");
        if (home != null) {
            Path homeDotVenv = Path.of(home, ".venv");
            if (Files.isDirectory(homeDotVenv)) return homeDotVenv;
        }

        throw new VllmException(
                "Cannot locate a .venv directory. " +
                "Set the system property 'vllm4j.venv' or use VllmEngine.builder().venvPath(...).");
    }

    // ── Package-private accessors for VllmEngine ────────────────────────────

    String model()                   { return model; }
    String dtype()                   { return dtype; }
    String tokenizer()               { return tokenizer; }
    String hfConfigPath()            { return hfConfigPath; }
    Double gpuMemoryUtilization()    { return gpuMemoryUtilization; }
    Integer maxModelLen()            { return maxModelLen; }
    Integer maxNumSeqs()             { return maxNumSeqs; }
    Integer maxNumBatchedTokens()    { return maxNumBatchedTokens; }
    Boolean enforceEager()           { return enforceEager; }
    Boolean trustRemoteCode()        { return trustRemoteCode; }
    String quantization()            { return quantization; }
    Double swapSpace()               { return swapSpace; }
    Integer seed()                   { return seed; }
    Boolean enablePrefixCaching()    { return enablePrefixCaching; }
    Boolean enableLora()             { return enableLora; }
    Integer maxLoras()               { return maxLoras; }
    Integer maxLoraRank()            { return maxLoraRank; }
    Boolean enableChunkedPrefill()   { return enableChunkedPrefill; }
    String kvCacheDtype()            { return kvCacheDtype; }
}
