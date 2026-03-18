package io.gravitee.vllm;

import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.iterator.VllmIterator;
import io.gravitee.vllm.iterator.VllmOutput;
import io.gravitee.vllm.platform.VllmBackend;
import io.gravitee.vllm.state.ConversationState;
import io.gravitee.vllm.state.GenerationState;
import io.gravitee.vllm.state.TagBounds;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;

import java.nio.file.Path;
import java.util.*;

/**
 * Interactive REPL for vLLM4j — chat with a model in the terminal.
 *
 * <p>Mirrors the llamaj.cpp Main pattern: CLI arg parsing, model loading,
 * interactive prompt loop with streaming token output, conversation history.
 *
 * <p>Run with:
 * <pre>{@code
 * java --enable-preview --enable-native-access=ALL-UNNAMED \
 *      -jar target/vLLM4j-1.0-SNAPSHOT.jar \
 *      --model Qwen/Qwen3-0.6B
 * }</pre>
 */
public class Main {

    // ANSI escape codes for styling output
    private static final String ANSI_DIM     = "\033[2m";
    private static final String ANSI_CYAN    = "\033[36m";
    private static final String ANSI_RESET   = "\033[0m";
    private static final String ANSI_YELLOW  = "\033[33m";

    public static void main(String[] args) {
        Map<String, String> params = parseArgs(args);

        String model    = params.getOrDefault("model", "Qwen/Qwen3-0.6B");
        String dtype    = params.getOrDefault("dtype", "auto");
        String venvPath = params.getOrDefault("venv", System.getProperty("user.dir") + "/.venv");

        // Engine parameters — CLI overridable, with small-GPU-friendly defaults
        boolean enforceEager         = Boolean.parseBoolean(params.getOrDefault("enforce_eager", "true"));
        int     maxModelLen          = Integer.parseInt(params.getOrDefault("max_model_len", "4096"));
        int     maxNumSeqs           = Integer.parseInt(params.getOrDefault("max_num_seqs", "16"));
        double  gpuMemoryUtilization = Double.parseDouble(params.getOrDefault("gpu_memory_utilization", "0.85"));

        String systemMessage = params.getOrDefault("system", "You are a helpful assistant.");

        double temperature    = Double.parseDouble(params.getOrDefault("temperature", "0.7"));
        int    maxTokens      = Integer.parseInt(params.getOrDefault("max_tokens", "512"));
        double topP           = Double.parseDouble(params.getOrDefault("top_p", "0.9"));
        int    topK           = Integer.parseInt(params.getOrDefault("top_k", "-1"));
        double repetitionPenalty = Double.parseDouble(params.getOrDefault("repetition_penalty", "1.0"));
        double frequencyPenalty  = Double.parseDouble(params.getOrDefault("frequency_penalty", "0.0"));
        double presencePenalty   = Double.parseDouble(params.getOrDefault("presence_penalty", "0.0"));

        // Tag-based classification config
        List<TagBounds> tagBoundsConfig = parseTagConfig(params);

        if (params.containsKey("help")) {
            printUsage();
            System.exit(0);
        }

        System.out.printf("vLLM4j%n  model   : %s%n  dtype   : %s%n  venv    : %s%n  backend : %s%n",
                model, dtype, venvPath, VllmBackend.detect());

        if (!tagBoundsConfig.isEmpty()) {
            System.out.printf("  classify:");
            for (TagBounds tb : tagBoundsConfig) {
                System.out.printf(" %s=%s|%s", tb.state().name().toLowerCase(), tb.openTag(), tb.closeTag());
            }
            System.out.println();
        }
        System.out.println();

        VllmEngine engine = VllmEngine.builder()
                .venvPath(Path.of(venvPath))
                .model(model)
                .dtype(dtype)
                .enforceEager(enforceEager)
                .maxModelLen(maxModelLen)
                .maxNumSeqs(maxNumSeqs)
                .gpuMemoryUtilization(gpuMemoryUtilization)
                .build();

        System.out.println("[OK] Engine initialized.");

        // Resolve chat template: CLI override > model's tokenizer > (fail)
        String userTemplate = params.get("chat_template");
        ChatTemplate renderer = userTemplate != null
                ? new ChatTemplate(engine, userTemplate)
                : new ChatTemplate(engine);

        System.out.println("[OK] Chat template: " +
                (userTemplate != null ? "user-provided" : "from model tokenizer"));

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(systemMessage));

        int requestCounter = 0;
        String input = "";

        while (!input.trim().equals("bye")) {
            Scanner scanIn = new Scanner(System.in);
            System.out.print("\n> ");
            input = scanIn.nextLine();

            if (input.isBlank() || input.trim().equals("bye")) {
                break;
            }

            messages.add(ChatMessage.user(input));

            String prompt = renderer.render(messages, true);

            try (var sp = samplingParams(temperature, maxTokens, topP, topK,
                    repetitionPenalty, frequencyPenalty, presencePenalty)) {

                String requestId = "req-" + (++requestCounter);
                var request = new VllmRequest(requestId, prompt, sp);

                // Create a fresh ConversationState per request (if tags configured)
                ConversationState conversationState = newConversationState(tagBoundsConfig);

                var iterator = new VllmIterator(engine);
                iterator.addRequest(request, conversationState);

                // Stream deltas — VllmOutput now carries delta() directly
                var fullText = new StringBuilder();
                iterator.stream().forEach(output -> {
                    String delta = output.delta();
                    if (!delta.isEmpty()) {
                        printStyled(delta, output.state());
                        fullText.append(delta);
                    }
                });

                // Reset ANSI at end of generation
                System.out.print(ANSI_RESET);

                messages.add(ChatMessage.assistant(fullText.toString()));

                // Print token breakdown if classification was active
                ConversationState state = iterator.conversationState(requestId);
                if (state != null) {
                    printTokenBreakdown(state);
                }
            }

            System.out.println();
        }

        engine.close();
    }

    // ── Styled output ───────────────────────────────────────────────────

    private static void printStyled(String delta, GenerationState state) {
        if (state == null) {
            System.out.print(delta);
            return;
        }

        switch (state) {
            case REASONING -> System.out.print(ANSI_DIM + delta);
            case TOOLS     -> System.out.print(ANSI_CYAN + delta);
            case ANSWER    -> System.out.print(ANSI_RESET + delta);
        }
    }

    // ── Token breakdown ─────────────────────────────────────────────────

    private static void printTokenBreakdown(ConversationState state) {
        var sb = new StringBuilder();
        sb.append(ANSI_YELLOW).append("\n  [tokens]");
        sb.append(" prompt=").append(state.inputTokens());
        sb.append(" answer=").append(state.answerTokens());
        if (state.reasoningTokens() > 0) {
            sb.append(" reasoning=").append(state.reasoningTokens());
        }
        if (state.toolsTokens() > 0) {
            sb.append(" tools=").append(state.toolsTokens());
        }
        sb.append(" total=").append(state.totalTokens());
        if (state.finishReason() != null) {
            sb.append(" finish=").append(state.finishReason().label());
        }
        sb.append(ANSI_RESET);
        System.out.println(sb);
    }

    // ── Tag config parsing ──────────────────────────────────────────────

    /**
     * Parses tag CLI flags into a reusable list of {@link TagBounds}.
     * This config is used as a template to create fresh ConversationState per request.
     */
    private static List<TagBounds> parseTagConfig(Map<String, String> params) {
        List<TagBounds> tags = new ArrayList<>();

        String reasoningTags = params.get("reasoning_tags");
        if (reasoningTags != null) {
            String[] parts = reasoningTags.split("\\|", 2);
            if (parts.length == 2 && !parts[0].isEmpty() && !parts[1].isEmpty()) {
                tags.add(new TagBounds(GenerationState.REASONING, parts[0], parts[1]));
            } else {
                System.err.println("[WARN] Invalid --reasoning_tags format. Expected \"<open>|<close>\"");
            }
        }

        String toolTags = params.get("tool_tags");
        if (toolTags != null) {
            String[] parts = toolTags.split("\\|", 2);
            if (parts.length == 2 && !parts[0].isEmpty() && !parts[1].isEmpty()) {
                tags.add(new TagBounds(GenerationState.TOOLS, parts[0], parts[1]));
            } else {
                System.err.println("[WARN] Invalid --tool_tags format. Expected \"<open>|<close>\"");
            }
        }

        return tags;
    }

    /**
     * Creates a fresh {@link ConversationState} from the tag config,
     * or returns {@code null} if no tags are configured.
     */
    private static ConversationState newConversationState(List<TagBounds> tagBoundsConfig) {
        if (tagBoundsConfig.isEmpty()) {
            return null;
        }

        var state = new ConversationState();
        for (TagBounds tb : tagBoundsConfig) {
            switch (tb.state()) {
                case REASONING -> state.reasoning(tb.openTag(), tb.closeTag());
                case TOOLS     -> state.toolCall(tb.openTag(), tb.closeTag());
                case ANSWER    -> {} // no tags for ANSWER
            }
        }
        return state;
    }

    // ── SamplingParams factory ──────────────────────────────────────────

    private static SamplingParams samplingParams(
            double temperature, int maxTokens, double topP, int topK,
            double repetitionPenalty, double frequencyPenalty, double presencePenalty) {
        var sp = new SamplingParams(java.lang.foreign.Arena.ofAuto())
                .temperature(temperature)
                .maxTokens(maxTokens)
                .topP(topP);

        if (topK > 0) sp.topK(topK);
        if (repetitionPenalty != 1.0) sp.repetitionPenalty(repetitionPenalty);
        if (frequencyPenalty != 0.0) sp.frequencyPenalty(frequencyPenalty);
        if (presencePenalty != 0.0) sp.presencePenalty(presencePenalty);

        return sp;
    }

    // ── CLI arg parsing ─────────────────────────────────────────────────

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> params = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                String key = arg.substring(2);
                if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                    params.put(key, args[++i]);
                } else {
                    params.put(key, "");
                }
            } else if (arg.startsWith("-")) {
                String key = arg.substring(1);
                if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                    params.put(key, args[++i]);
                } else {
                    params.put(key, "");
                }
            }
        }
        return params;
    }

    // ── Usage ───────────────────────────────────────────────────────────

    private static void printUsage() {
        System.err.println("""
            Usage: java --enable-preview --enable-native-access=ALL-UNNAMED \\
                        -jar vLLM4j.jar [options...]

            Model:
              --model <id>                HuggingFace model id (default: Qwen/Qwen3-0.6B)
              --dtype <type>              Torch dtype: auto, float16, float32 (default: auto)
              --venv <path>               Path to Python venv (default: .venv)

            Engine:
              --enforce_eager <bool>      Skip CUDA graph capture (default: true)
              --max_model_len <int>       Max sequence length (default: 4096)
              --max_num_seqs <int>        Max concurrent sequences (default: 16)
              --gpu_memory_utilization <f> GPU memory fraction (default: 0.85)

            Prompt:
              --system <msg>              System prompt (default: "You are a helpful assistant.")
              --chat_template <jinja2>    Jinja2 chat template string (default: from model)

            Sampling:
              --temperature <float>       Sampling temperature (default: 0.7)
              --max_tokens <int>          Max tokens to generate (default: 512)
              --top_p <float>             Top-P nucleus sampling (default: 0.9)
              --top_k <int>               Top-K sampling, -1 for all (default: -1)
              --repetition_penalty <f>    Repetition penalty, 1.0 = none (default: 1.0)
              --frequency_penalty <f>     Frequency penalty (default: 0.0)
              --presence_penalty <f>      Presence penalty (default: 0.0)

            Token Classification:
              --reasoning_tags "<open>|<close>"
                                          Tag pair for reasoning detection (e.g. "<think>|</think>")
              --tool_tags "<open>|<close>"
                                          Tag pair for tool-call detection (e.g. "<tool_call>|</tool_call>")

            Commands:
              Type 'bye' to exit the REPL.

            Examples:
              java -jar vLLM4j.jar --model Qwen/Qwen3-0.6B
              java -jar vLLM4j.jar --model Qwen/Qwen3-0.6B --reasoning_tags "<think>|</think>"
              java -jar vLLM4j.jar --model Qwen/Qwen3-0.6B --temperature 0.0 --max_tokens 128
              java -jar vLLM4j.jar --model mistralai/Mistral-7B-v0.1 --dtype float16
            """);
    }
}
