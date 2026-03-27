# vLLM4j

JVM native binding for [vLLM](https://github.com/vllm-project/vllm) via Java FFM (Foreign Function & Memory API) and [jextract](https://jdk.java.net/jextract/).

Embeds CPython in-process, drives vLLM's synchronous `LLMEngine` directly, and exposes Jinja2 chat-template rendering -- all without HTTP, Ray, or `AsyncLLMEngine`.

## How it works

```
┌──────────────┐     FFM (Panama)     ┌──────────────┐     Python C API     ┌──────────┐
│   Java 25    │ ──────────────────── │  libpython   │ ──────────────────── │   vLLM   │
│  (your app)  │    jextract bindings │   3.12       │   LLMEngine.step()   │  engine  │
└──────────────┘                      └──────────────┘                      └──────────┘
```

- **No HTTP server** -- vLLM runs in the same process as the JVM
- **Continuous batching** -- multiple requests processed in parallel via `VllmIterator`
- **Token classification** -- Java-side FSM detects reasoning (`<think>`) and tool-call tags in generated text

> Targets **vLLM 0.16.0** across all backends. The version is pinned in `VLLM_VERSION` at the top of `scripts/setup-venv.sh`.

## Requirements

| Component | Version |
|-----------|---------|
| Java      | 25 (with `--enable-preview`) |
| Python    | 3.12 (miniforge recommended) |
| Maven     | 3.9+ |
| uv        | For venv management |

Supported backends:

| Backend | Platform | Notes |
|---------|----------|-------|
| **Metal** | macOS Apple Silicon | via [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| **CUDA**  | Linux x86_64 | NVIDIA GPU |
| **CPU**   | Any | Fallback |

## Quick start

```bash
# Build (macOS Apple Silicon + Metal)
JAVA_HOME=/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home \
  mvn clean package -P macosx-aarch64,metal -DskipTests

# Run interactive REPL
JAVA_HOME=/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home \
  $JAVA_HOME/bin/java --enable-preview \
       --enable-native-access=ALL-UNNAMED \
       -jar target/vLLM4j-1.0-SNAPSHOT.jar \
       --model Qwen/Qwen3-0.6B

# With reasoning token classification
java -jar target/vLLM4j-1.0-SNAPSHOT.jar \
     --model Qwen/Qwen3-0.6B \
     --reasoning_tags "<think>|</think>"
```

---

## Usage as a library

### Creating an engine

```java
// Minimal — uses defaults (Qwen/Qwen3-0.6B, dtype=auto)
try (var engine = VllmEngine.builder().build()) {
    // ...
}

// Full configuration
try (var engine = VllmEngine.builder()
        .model("Qwen/Qwen3-0.6B")
        .dtype("float16")
        .venvPath(Path.of("/path/to/.venv"))
        .gpuMemoryUtilization(0.85)
        .maxModelLen(4096)
        .maxNumSeqs(32)
        .maxNumBatchedTokens(8192)
        .enforceEager(true)
        .trustRemoteCode(false)
        .quantization("awq")
        .swapSpace(4.0)
        .seed(42)
        .kvCacheDtype("auto")
        .enablePrefixCaching(true)
        .enableChunkedPrefill(true)
        .build()) {
    // ...
}
```

### Chat templates

Use `ChatTemplate` to render conversations via the model's Jinja2 template:

```java
var template = new ChatTemplate(engine);

String prompt = template.render(List.of(
        ChatMessage.system("You are a helpful assistant."),
        ChatMessage.user("What is the capital of France?")
), true);  // true = add generation prompt
```

Override the model's template if needed:

```java
var template = new ChatTemplate(engine, "{% for m in messages %}..{% endfor %}");
```

### Blocking generation

The simplest way to get a complete response:

```java
try (var sp = new SamplingParams().temperature(0.0).maxTokens(128)) {
    var request = new VllmRequest("req-1", prompt, sp);
    RequestOutput result = engine.generate(request);

    String text = result.outputs().getFirst().text();
    FinishReason reason = result.outputs().getFirst().finishReason();
    int promptTokens = result.numPromptTokens();
    int generatedTokens = result.numGeneratedTokens();
}
```

### Streaming with VllmIterator

Stream tokens as they're generated:

```java
try (var sp = new SamplingParams().temperature(0.7).maxTokens(256)) {
    var request = new VllmRequest("req-1", prompt, sp);
    var iterator = new VllmIterator(engine);
    iterator.addRequest(request);

    iterator.stream().forEach(output -> {
        System.out.print(output.delta());   // incremental text
        // output.text()         — cumulative text so far
        // output.finished()     — true on last output
        // output.finishReason() — "stop", "length", etc.
        // output.tokenIds()     — all token IDs (only on last output)
        // output.logprobs()     — logprobs (only on last output, if requested)
    });
}
```

### Parallel requests (continuous batching)

Submit multiple requests — vLLM's scheduler interleaves them:

```java
var iterator = new VllmIterator(engine);

try (var sp1 = new SamplingParams().temperature(0.0).maxTokens(64);
     var sp2 = new SamplingParams().temperature(0.0).maxTokens(64);
     var sp3 = new SamplingParams().temperature(0.0).maxTokens(64)) {

    iterator.addRequest(new VllmRequest("france",  promptFrance,  sp1));
    iterator.addRequest(new VllmRequest("england", promptEngland, sp2));
    iterator.addRequest(new VllmRequest("poland",  promptPoland,  sp3));
}

// Stream interleaves outputs from all three requests
Map<String, StringBuilder> outputs = new HashMap<>();
iterator.stream().forEach(output ->
        outputs.computeIfAbsent(output.requestId(), k -> new StringBuilder())
               .append(output.delta()));
```

### SamplingParams — full API

```java
try (var sp = new SamplingParams()
        // Core sampling
        .temperature(0.8)
        .topP(0.95)
        .topK(50)
        .minP(0.05)
        .seed(42L)

        // Length control
        .maxTokens(512)
        .minTokens(10)

        // Repetition control
        .repetitionPenalty(1.1)
        .frequencyPenalty(0.5)
        .presencePenalty(0.3)

        // Stop conditions
        .stop("User:", "\n\n")
        .stopTokenIds(128001, 128009)
        .includeStopStrInOutput(false)

        // Multi-sequence
        .n(3)
        .bestOf(5)

        // Token handling
        .skipSpecialTokens(true)
        .spacesBetweenSpecialTokens(true)
        .truncatePromptTokens(4096)

        // Logprobs (see Logprobs section)
        .logprobs(5)
        .promptLogprobs(1)

        // Guided decoding (see Guided Decoding section)
        .guidedDecoding(GuidedDecodingParams.json(schema))) {

    // use sp...
}
```

### Token classification (reasoning / tool calls)

Classify generated tokens into reasoning, tool-call, and answer regions using a tag-based FSM:

```java
var state = new ConversationState()
        .reasoning("<think>", "</think>")
        .toolCall("<tool_call>", "</tool_call>");

var iterator = new VllmIterator(engine);
iterator.addRequest(request, state);

iterator.stream().forEach(output -> {
    GenerationState segment = output.state();  // REASONING, TOOLS, or ANSWER

    switch (segment) {
        case REASONING -> System.err.print(output.delta());   // internal reasoning
        case TOOLS     -> handleToolCall(output.delta());      // tool invocation JSON
        case ANSWER    -> System.out.print(output.delta());    // user-facing answer
    }
});

// After generation — token breakdown
ConversationState result = iterator.conversationState("req-1");
System.out.println("Reasoning tokens: " + result.reasoningTokens());
System.out.println("Answer tokens:    " + result.answerTokens());
System.out.println("Tool tokens:      " + result.toolsTokens());
System.out.println("Total output:     " + result.totalOutputTokens());
System.out.println("Total (w/ input): " + result.totalTokens());
System.out.println("Finish reason:    " + result.finishReason());
```

### Tool-use (multi-turn)

Define tools in OpenAI format, render them into the chat template, and handle multi-turn tool-call conversations:

```java
// 1. Define tools
var weatherTool = Tool.function(
        "get_weather",
        "Get the current weather for a location",
        Map.of(
                "type", "object",
                "properties", Map.of(
                        "location", Map.of("type", "string",
                                "description", "City name, e.g. 'Paris'"),
                        "unit", Map.of("type", "string",
                                "enum", List.of("celsius", "fahrenheit"))
                ),
                "required", List.of("location")
        ));

var tools = List.of(weatherTool);

// 2. Render prompt with tools
var template = new ChatTemplate(engine);
var messages = new ArrayList<>(List.of(
        ChatMessage.system("You are a helpful assistant with tool access."),
        ChatMessage.user("What's the weather in Paris?")
));

String prompt = template.render(messages, tools, true);

// 3. Generate — model emits tool call
try (var sp = new SamplingParams().temperature(0.0).maxTokens(256)) {
    RequestOutput result = engine.generate(new VllmRequest("tc-1", prompt, sp));
    String response = result.outputs().getFirst().text();

    // 4. Parse tool call from response and build tool-call message
    var toolCall = ToolCall.function("call_001", "get_weather", "{\"location\": \"Paris\"}");
    messages.add(ChatMessage.assistantWithToolCalls(null, List.of(toolCall)));

    // 5. Add tool result
    messages.add(ChatMessage.toolResult(
            "{\"temperature\": 18, \"unit\": \"celsius\", \"condition\": \"cloudy\"}",
            "call_001",   // tool_call_id
            "get_weather" // function name
    ));

    // 6. Re-render and generate final answer
    prompt = template.render(messages, tools, true);
    result = engine.generate(new VllmRequest("tc-2", prompt, sp));
    System.out.println(result.outputs().getFirst().text());
}
```

### Guided decoding

Constrain generation output to match a schema, regex, grammar, or choice list:

```java
// JSON schema
var jsonGuide = GuidedDecodingParams.json("""
        {
          "type": "object",
          "properties": {
            "name":  {"type": "string"},
            "age":   {"type": "integer"},
            "email": {"type": "string", "format": "email"}
          },
          "required": ["name", "age"]
        }
        """);

// Regex pattern
var regexGuide = GuidedDecodingParams.regex("[A-Z]{2}-\\d{4}");

// Choice from list
var choiceGuide = GuidedDecodingParams.choice(List.of("positive", "negative", "neutral"));

// EBNF grammar
var grammarGuide = GuidedDecodingParams.grammar("root ::= 'yes' | 'no'");

// Freeform JSON object (no specific schema)
var jsonObjectGuide = GuidedDecodingParams.jsonObject();

// Use with SamplingParams
try (var sp = new SamplingParams()
        .temperature(0.0)
        .maxTokens(256)
        .guidedDecoding(jsonGuide)) {

    RequestOutput result = engine.generate(new VllmRequest("guided-1", prompt, sp));
    String json = result.outputs().getFirst().text();  // guaranteed valid JSON
}
```

### Logprobs

Request log-probabilities for generated and/or prompt tokens:

```java
try (var sp = new SamplingParams()
        .temperature(0.0)
        .maxTokens(64)
        .logprobs(5)          // top-5 logprobs per generated token
        .promptLogprobs(1)) { // logprobs for prompt tokens

    RequestOutput result = engine.generate(new VllmRequest("lp-1", prompt, sp));

    // Generated token logprobs
    CompletionOutput completion = result.outputs().getFirst();
    for (Map<Integer, LogprobEntry> step : completion.logprobs()) {
        step.values().forEach(entry ->
                System.out.printf("  token=%s id=%d logprob=%.4f rank=%d%n",
                        entry.decodedToken(), entry.tokenId(),
                        entry.logprob(), entry.rank()));
    }

    // Prompt token logprobs
    for (Map<Integer, LogprobEntry> step : result.promptLogprobs()) {
        step.values().forEach(entry ->
                System.out.printf("  prompt token=%s logprob=%.4f%n",
                        entry.decodedToken(), entry.logprob()));
    }
}
```

When streaming via `VllmIterator`, `tokenIds` and `logprobs` are emitted only on the final output (`finished=true`) to avoid accumulating data in memory across every step:

```java
try (var sp = new SamplingParams().temperature(0.0).maxTokens(64).logprobs(5)) {
    var iterator = new VllmIterator(engine);
    iterator.addRequest(new VllmRequest("lp-stream", prompt, sp));

    iterator.stream().forEach(output -> {
        System.out.print(output.delta());           // available every step
        if (output.finished()) {
            // tokenIds and logprobs only available here
            List<Integer> tokenIds = output.tokenIds();
            List<Map<Integer, LogprobEntry>> logprobs = output.logprobs(); // null if not requested
        }
    });
}
```

> **Note:** `logprobs` is `null` unless `SamplingParams.logprobs(n)` is set with `n >= 1`.

### Tokenizer

Access the model's tokenizer directly:

```java
// Encode text to token IDs
List<Integer> tokens = engine.encode("Hello, world!");

// Decode token IDs back to text
String text = engine.decode(tokens);

// Get vocabulary size
int vocabSize = engine.vocabSize();
```

### Multimodal (image and audio)

Send images and audio alongside text prompts (requires a vision/audio model):

```java
// From byte arrays
var mmData = new MultiModalData()
        .addImage(imageBytes)
        .addAudio(audioBytes);

// From file paths
var mmData = new MultiModalData()
        .addImage(Path.of("photo.jpg"))
        .addAudio(Path.of("speech.wav"));

// Use in a request
try (var sp = new SamplingParams().temperature(0.0).maxTokens(256)) {
    var request = new VllmRequest("mm-1", prompt, sp, mmData);
    RequestOutput result = engine.generate(request);
    System.out.println(result.outputs().getFirst().text());
}
```

### Engine stats

Query live engine state:

```java
EngineStats stats = engine.getStats();
System.out.println("Model:        " + stats.model());
System.out.println("Dtype:        " + stats.dtype());
System.out.println("Max len:      " + stats.maxModelLen());
System.out.println("In-flight:    " + stats.numUnfinishedRequests());
```

### Request metrics

Timing information returned with each completed request:

```java
RequestOutput result = engine.generate(request);
RequestMetrics metrics = result.metrics();

System.out.printf("TTFT:       %.1f ms%n", metrics.ttftMs());
System.out.printf("Total time: %.1f ms%n", metrics.totalTimeMs());
System.out.printf("Queue time: %.3f s%n",  metrics.timeInQueue());
```

### Request priority

Prioritize requests in the scheduler:

```java
// Higher priority value = higher scheduling priority
var urgentRequest = new VllmRequest("urgent-1", prompt, sp, null, 10);
var normalRequest = new VllmRequest("normal-1", prompt, sp, null, 1);

var iterator = new VllmIterator(engine);
iterator.addRequest(normalRequest);
iterator.addRequest(urgentRequest);  // scheduled first despite being added second
```

---

## Project structure

```
vLLM4j/
├── pom.xml                           # Two-axis Maven profiles: OS/arch x backend
├── .mvn/jvm.config                   # --enable-preview
├── scripts/
│   ├── download-jextract.sh          # Downloads jextract EA build
│   ├── setup-venv.sh                 # Creates .venv, installs vLLM per backend
│   └── generate-sources.sh           # Resolves Python.h, runs jextract
└── src/
    ├── main/java/io/gravitee/vllm/
    │   ├── Main.java                 # Interactive REPL with ANSI-styled output
    │   ├── Freeable.java             # free() + isFree() interface
    │   ├── platform/                 # OS, arch, backend detection
    │   ├── runtime/                  # CPython lifecycle (PythonRuntime, PythonRef)
    │   ├── binding/                  # FFM helpers (PythonCall, PythonTypes, PythonErrors)
    │   ├── engine/                   # VllmEngine, SamplingParams, RequestOutput, ...
    │   ├── iterator/                 # VllmIterator (continuous batching), VllmOutput
    │   ├── state/                    # ConversationState, StateEvaluation FSM, TokenTracking
    │   └── template/                 # ChatTemplate (Jinja2), ChatMessage, Tool
    └── test/java/io/gravitee/vllm/
        ├── engine/                   # Unit tests for records, FinishReason, RequestMetrics
        ├── iterator/                 # Integration tests (real engine, parallel capitals)
        ├── platform/                 # Unit tests for platform detection
        ├── state/                    # Unit tests for FSM, token tracking, ConversationState
        └── template/                 # Unit tests for ChatMessage, Tool, ToolFunction
```

## Tests

```bash
# Unit tests only (fast, no CPython needed)
mvn test -P macosx-aarch64,metal
# 162 tests, ~4s

# Integration tests only (loads real model)
# Each test class runs in its own JVM fork (reuseForks=false)
# so GPU memory is fully reclaimed between classes.
mvn test -P integration,macosx-aarch64,metal

# Integration tests on Linux + CUDA
# Requires: LD_PRELOAD and .venv/bin on PATH (see Linux / CUDA notes)
mvn test -P integration,linux-x86_64,cuda
```

## Linux / CUDA notes

### `/dev/shm` permissions

vLLM 0.16+ creates a `multiprocessing.Lock()` during executor initialisation
(used by the shared-memory multimodal cache). On Linux this requires write
access to `/dev/shm`. If `/dev/shm` is owned by root with mode `755` (common
in restricted containers) the engine will fail with:

```
PermissionError: [Errno 13] Permission denied
_multiprocessing.SemLock(...)
```

**Fix on your dev machine (survives until next reboot):**

```bash
sudo chmod 1777 /dev/shm
```

**Fix permanently (add to `/etc/fstab`):**

```
tmpfs /dev/shm tmpfs defaults,size=8g 0 0
```

The `linux-x86_64` Maven profile runs `sudo chmod 1777 /dev/shm` automatically
during the `validate` phase, so `mvn` will prompt for your password once per
session if `/dev/shm` has wrong permissions.

### GPU memory and LoRA on small GPUs

CPython can only be initialized once per process, and GPU memory held by an
engine is not released until the process exits. To avoid memory contention,
the integration test profile uses **`reuseForks=false`** so that each test
class runs in its own JVM fork. The `_exit(0)` shutdown hook terminates each
fork cleanly and the OS reclaims all GPU memory before the next class starts.

This means each test class creates its own engine with the settings it needs:

- **`VllmIteratorTest`** — base model only (no LoRA), `gpuMemoryUtilization(0.85)`
- **`LoraTest`** — LoRA enabled, same GPU utilization (LoRA infrastructure
  takes its share, but the full GPU is available since the prior fork exited)

On cards with limited VRAM (8 GiB or less), both engines are configured with:

- **`enforceEager(true)`** — skips CUDA graph capture, which pre-allocates
  dummy LoRA adapters and OOMs on small cards
- **`maxModelLen(4096)`** (or `2048` for LoRA) — Qwen3-0.6B defaults to
  40960, but the KV cache for that length doesn't fit in limited VRAM.
  Tests only need short contexts.
- **`maxNumSeqs(4)`** — default 128 causes sampler warm-up OOM on small cards;
  4 is enough for the parallel/interleaving tests.

The `/no_think` suffix is appended to system prompts so Qwen3 skips its
internal reasoning chain, producing shorter and faster test outputs.

### `ninja` and venv PATH

vLLM's flashinfer backend JIT-compiles CUDA kernels at first use via
[ninja](https://ninja-build.org/). The `ninja` binary is installed inside
`.venv/bin/` by `setup-venv.sh`, but the forked JVM needs it on `PATH`.

Add the venv to your shell PATH (e.g. in `~/.zshrc` or `~/.bashrc`):

```bash
export PATH="/path/to/vLLM4j/.venv/bin:$PATH"
```

## Maven profiles

Two orthogonal axes:

**OS/arch** (pick one):
- `macosx-aarch64` -- macOS Apple Silicon
- `macosx-x86_64` -- macOS Intel
- `linux-x86_64` -- Linux

**Backend** (pick one):
- `metal` -- Apple Silicon Metal/MLX GPU
- `cuda` -- NVIDIA CUDA GPU
- `cpu` -- CPU-only fallback

**Test control:**
- `integration` -- runs `@Tag("integration")` tests (excluded by default). Uses `reuseForks=false` so each test class gets its own JVM fork with a clean GPU.

Example: `mvn clean package -P macosx-aarch64,metal`

## CLI options

```
Model:
  --model <id>                HuggingFace model id (default: Qwen/Qwen3-0.6B)
  --dtype <type>              Torch dtype: auto, float16, float32 (default: auto)
  --venv <path>               Path to Python venv (default: .venv)

Prompt:
  --system <msg>              System prompt
  --chat_template <jinja2>    Jinja2 chat template string (default: from model)

Sampling:
  --temperature <float>       Sampling temperature (default: 0.7)
  --max_tokens <int>          Max tokens to generate (default: 512)
  --top_p <float>             Top-P nucleus sampling (default: 0.9)
  --top_k <int>               Top-K sampling (default: -1)
  --repetition_penalty <f>    Repetition penalty (default: 1.0)
  --frequency_penalty <f>     Frequency penalty (default: 0.0)
  --presence_penalty <f>      Presence penalty (default: 0.0)

Token Classification:
  --reasoning_tags "<open>|<close>"   e.g. "<think>|</think>"
  --tool_tags "<open>|<close>"        e.g. "<tool_call>|</tool_call>"
```

## Design decisions

- **No `PyConfig`** -- uses `setenv(PYTHONHOME) + Py_InitializeEx(0)` (jextract can't generate PyConfig)
- **GIL held permanently** on the main thread -- no release between calls
- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** -- in-process mode required for embedded CPython
- **Delta extraction** -- vLLM's `CompletionOutput.text` is cumulative; the iterator tracks only the previous text length (an `int`) to extract per-step deltas without holding a full copy of the generated text
- **Lazy tokenIds/logprobs** -- `tokenIds` and `logprobs` are omitted from `VllmOutput` during streaming and emitted only on the final output (`finished=true`) to avoid O(n) per-step memory accumulation
- **State classification is Java-side** -- a tag-based FSM classifies generated text (following llamaj.cpp's pattern)
- **Tools are template-level** -- passed as a `tools` variable to the Jinja2 chat template
- **`/dev/shm` fix on Linux** -- vLLM 0.16+ uses `multiprocessing.Lock()` which requires `/dev/shm` write access. The `linux-x86_64` Maven profile runs `sudo chmod 1777 /dev/shm` during `validate` to fix misconfigured permissions

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
