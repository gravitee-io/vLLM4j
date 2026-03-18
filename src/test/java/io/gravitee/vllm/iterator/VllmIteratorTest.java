package io.gravitee.vllm.iterator;

import io.gravitee.vllm.engine.EngineStats;
import io.gravitee.vllm.engine.LogprobEntry;
import io.gravitee.vllm.engine.RequestOutput;
import io.gravitee.vllm.engine.SamplingParams;
import io.gravitee.vllm.engine.VllmEngine;
import io.gravitee.vllm.engine.VllmRequest;
import io.gravitee.vllm.state.ConversationState;
import io.gravitee.vllm.state.GenerationState;
import io.gravitee.vllm.template.ChatMessage;
import io.gravitee.vllm.template.ChatTemplate;
import io.gravitee.vllm.template.Tool;
import io.gravitee.vllm.template.ToolCall;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Integration tests for {@link VllmIterator} against a real {@link VllmEngine}
 * backed by CPython + vLLM + Qwen/Qwen3-0.6B.
 *
 * <p>Mirrors the llamaj.cpp {@code BatchProcessorTest} pattern — parallel
 * conversations asking about country capitals, accumulating outputs per
 * request, and verifying token counts + finish reasons.
 *
 * <p>Tagged {@code "integration"} and excluded from the default
 * {@code mvn test} run. Execute with:
 * <pre>{@code
 * mvn test -P integration,macosx-aarch64,metal
 * }</pre>
 *
 * <p>The engine is created without LoRA (maximizing available GPU memory)
 * and closed in {@code @AfterAll}. With {@code reuseForks=false}, each
 * test class gets its own JVM fork — GPU memory is fully reclaimed between classes.
 */
@Tag("integration")
class VllmIteratorTest {

    static final String SYSTEM = """
            You are the best at guessing capitals. \
            Respond to the best of your ability. Just answer with the capital. /no_think
            What's the capital of France? Paris.
            What is the capital of England? London.
            What is the capital of Poland? Warsaw.""";

    private static VllmEngine engine;
    private static ChatTemplate chatTemplate;

    @BeforeAll
    static void initEngine() {
        engine = SharedEngine.baseEngine();
        chatTemplate = SharedEngine.chatTemplate();
    }

    @AfterAll
    static void closeEngine() {
        SharedEngine.close();
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════════════

    private static String renderPrompt(String systemMessage, String userMessage) {
        return chatTemplate.render(List.of(
                ChatMessage.system(systemMessage),
                ChatMessage.user(userMessage)
        ), true);
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Single sequence — capital of France
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void single_conversation_generates_tokens() {
        String prompt = renderPrompt(SYSTEM, "What is the capital of France?");

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(256)) {
            var request = new VllmRequest("req-1", prompt, sp);
            var state = new ConversationState()
                    .reasoning("<think>", "</think>");

            var iterator = new VllmIterator(engine);
            iterator.addRequest(request, state);

            var output = new StringBuilder();
            long startTime = System.nanoTime();

            iterator.stream().forEach(o -> output.append(o.delta()));

            long endTime = System.nanoTime();
            double durationMs = (endTime - startTime) / 1_000_000.0;

            System.out.println("\n=== Single Conversation Result ===");
            System.out.println("Prompt: What is the capital of France?");
            System.out.println("Output: " + output);
            System.out.println("  Input tokens:     " + state.inputTokens());
            System.out.println("  Answer tokens:    " + state.answerTokens());
            System.out.println("  Reasoning tokens: " + state.reasoningTokens());
            System.out.println("  Total output:     " + state.totalOutputTokens());
            System.out.println("  Finish reason:    " + state.finishReason());
            System.out.println("  Time: " + durationMs + " ms");
            System.out.println("===================================");

            assertThat(state.inputTokens()).isGreaterThan(0);
            assertThat(state.totalOutputTokens()).isGreaterThan(0);
            assertThat(state.finishReason()).isNotNull();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Parallel conversations — capitals of France, England, Poland
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void parallel_conversations_generate_tokens() {
        // Create 3 different conversations (mirrors llamaj.cpp BatchProcessorTest)
        String[] questions = {
                "What is the capital of France?",
                "What is the capital of England?",
                "What is the capital of Poland?"
        };
        String[] requestIds = { "req-france", "req-england", "req-poland" };

        // Create conversation states with reasoning classification
        ConversationState[] states = new ConversationState[3];
        for (int i = 0; i < 3; i++) {
            states[i] = new ConversationState()
                    .reasoning("<think>", "</think>");
        }

        var iterator = new VllmIterator(engine);

        for (int i = 0; i < 3; i++) {
            String prompt = renderPrompt(SYSTEM, questions[i]);
            try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(256)) {
                var request = new VllmRequest(requestIds[i], prompt, sp);
                iterator.addRequest(request, states[i]);
            }
        }

        // Map to accumulate outputs per request
        Map<String, StringBuilder> outputs = new HashMap<>();
        for (String id : requestIds) {
            outputs.put(id, new StringBuilder());
        }

        // Generate tokens in parallel using stream
        long startTime = System.nanoTime();

        iterator.stream().forEach(output ->
                outputs.get(output.requestId()).append(output.delta())
        );

        long endTime = System.nanoTime();
        double durationMs = (endTime - startTime) / 1_000_000.0;

        System.out.println("\n=== Parallel Generation Results ===");
        for (int i = 0; i < 3; i++) {
            System.out.println("Conversation " + (i + 1) + " (" + requestIds[i] + "): "
                    + outputs.get(requestIds[i]));
            System.out.println("  Answer tokens:    " + states[i].answerTokens());
            System.out.println("  Reasoning tokens: " + states[i].reasoningTokens());
            System.out.println("  Finish reason:    " + states[i].finishReason());
        }
        System.out.println("Total time: " + durationMs + " ms");
        System.out.println("===================================");

        // Verify all conversations generated tokens
        assertThat(states[0].totalOutputTokens()).isGreaterThan(0);
        assertThat(states[1].totalOutputTokens()).isGreaterThan(0);
        assertThat(states[2].totalOutputTokens()).isGreaterThan(0);

        // Verify all conversations have finish reasons
        assertThat(states[0].finishReason()).isNotNull();
        assertThat(states[1].finishReason()).isNotNull();
        assertThat(states[2].finishReason()).isNotNull();
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Delta correctness — deltas concatenate to full text
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void deltas_concatenate_to_full_text() {
        String prompt = renderPrompt(SYSTEM, "What is the capital of Germany?");

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(30)) {
            var request = new VllmRequest("req-deltas", prompt, sp);
            var iterator = new VllmIterator(engine);
            iterator.addRequest(request);

            List<VllmOutput> allOutputs = iterator.stream().toList();

            // Concatenation of all deltas should equal the final cumulative text
            String fromDeltas = allOutputs.stream()
                    .map(VllmOutput::delta)
                    .collect(Collectors.joining());
            String finalText = allOutputs.getLast().text();

            assertThat(fromDeltas).isEqualTo(finalText);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Parallel deltas — per-request delta correctness
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void parallel_deltas_are_per_request() {
        String prompt1 = renderPrompt(SYSTEM, "What is the capital of Italy?");
        String prompt2 = renderPrompt(SYSTEM, "What is the capital of Spain?");

        try (var sp1 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(20);
             var sp2 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(20)) {

            var iterator = new VllmIterator(engine);
            iterator.addRequest(new VllmRequest("req-italy", prompt1, sp1));
            iterator.addRequest(new VllmRequest("req-spain", prompt2, sp2));

            List<VllmOutput> allOutputs = iterator.stream().toList();
            var byRequest = allOutputs.stream()
                    .collect(Collectors.groupingBy(VllmOutput::requestId));

            // For each request, deltas should concatenate to the final text
            for (var entry : byRequest.entrySet()) {
                List<VllmOutput> reqOutputs = entry.getValue();
                String fromDeltas = reqOutputs.stream()
                        .map(VllmOutput::delta)
                        .collect(Collectors.joining());
                String finalText = reqOutputs.getLast().text();

                assertThat(fromDeltas)
                        .as("Deltas for %s", entry.getKey())
                        .isEqualTo(finalText);
            }

            // Both should have finished
            assertThat(byRequest.get("req-italy").getLast().finished()).isTrue();
            assertThat(byRequest.get("req-spain").getLast().finished()).isTrue();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Interleaving — outputs from parallel requests are interleaved
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void parallel_outputs_are_interleaved() {
        String prompt1 = renderPrompt(SYSTEM, "What is the capital of Japan?");
        String prompt2 = renderPrompt(SYSTEM, "What is the capital of Brazil?");

        try (var sp1 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(15);
             var sp2 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(15)) {

            var iterator = new VllmIterator(engine);
            iterator.addRequest(new VllmRequest("req-japan", prompt1, sp1));
            iterator.addRequest(new VllmRequest("req-brazil", prompt2, sp2));

            List<String> requestOrder = iterator.stream()
                    .map(VllmOutput::requestId)
                    .toList();

            // Both IDs must appear
            assertThat(requestOrder).contains("req-japan", "req-brazil");

            // Interleaving: first occurrence of req-brazil appears
            // before last occurrence of req-japan
            int firstBrazil = requestOrder.indexOf("req-brazil");
            int lastJapan = requestOrder.lastIndexOf("req-japan");
            assertThat(firstBrazil)
                    .as("Outputs should be interleaved")
                    .isLessThan(lastJapan);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Classification — independent state per request
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void classification_is_independent_per_request() {
        String prompt1 = renderPrompt(SYSTEM, "What is the capital of Canada?");
        String prompt2 = renderPrompt(SYSTEM, "What is the capital of Australia?");

        var state1 = new ConversationState()
                .reasoning("<think>", "</think>");
        // req-2 has no classification

        try (var sp1 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(30);
             var sp2 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(30)) {

            var iterator = new VllmIterator(engine);
            iterator.addRequest(new VllmRequest("req-canada", prompt1, sp1), state1);
            iterator.addRequest(new VllmRequest("req-australia", prompt2, sp2));

            List<VllmOutput> allOutputs = iterator.stream().toList();
            var byRequest = allOutputs.stream()
                    .collect(Collectors.groupingBy(VllmOutput::requestId));

            // req-canada outputs should have state classification (non-null)
            boolean canadaHasStates = byRequest.get("req-canada").stream()
                    .anyMatch(o -> o.state() != null);
            assertThat(canadaHasStates).isTrue();

            // req-australia outputs should have null state (no classification)
            boolean australiaAllNull = byRequest.get("req-australia").stream()
                    .allMatch(o -> o.state() == null);
            assertThat(australiaAllNull).isTrue();

            // req-canada conversation state should have counters
            assertThat(iterator.conversationState("req-canada")).isSameAs(state1);
            assertThat(state1.totalOutputTokens()).isGreaterThan(0);
            assertThat(state1.inputTokens()).isGreaterThan(0);

            // req-australia should have no conversation state
            assertThat(iterator.conversationState("req-australia")).isNull();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Stop — halt iteration mid-generation
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void stop_halts_iteration() {
        String prompt = renderPrompt(SYSTEM, "Count from 1 to 100");

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(100)) {
            var request = new VllmRequest("req-stop", prompt, sp);
            var iterator = new VllmIterator(engine);
            iterator.addRequest(request);

            // Generate a few tokens then stop
            int tokenCount = 0;
            while (iterator.hasNext() && tokenCount < 5) {
                iterator.next();
                tokenCount++;
            }

            // Explicitly stop the iterator
            iterator.stop();

            // Verify hasNext returns false after stop
            assertThat(iterator.hasNext()).isFalse();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Blocking generate — convenience method
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void blocking_generate_returns_final_output() {
        String prompt = renderPrompt(SYSTEM, "What is the capital of Sweden?");

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(30)) {
            var request = new VllmRequest("req-generate", prompt, sp);

            RequestOutput output = engine.generate(request);

            assertThat(output).isNotNull();
            assertThat(output.requestId()).isEqualTo("req-generate");
            assertThat(output.finished()).isTrue();
            assertThat(output.outputs()).isNotEmpty();
            assertThat(output.outputs().getFirst().text()).isNotEmpty();
            assertThat(output.numPromptTokens()).isGreaterThan(0);
            assertThat(output.numGeneratedTokens()).isGreaterThan(0);

            System.out.println("\n=== Blocking Generate Result ===");
            System.out.println("Output: " + output.outputs().getFirst().text());
            System.out.println("Prompt tokens: " + output.numPromptTokens());
            System.out.println("Generated tokens: " + output.numGeneratedTokens());
            System.out.println("================================");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Logprobs — verify logprob extraction
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void logprobs_should_be_extracted() {
        String prompt = renderPrompt(SYSTEM, "What is the capital of Mexico?");

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(20).logprobs(5)) {
            var request = new VllmRequest("req-logprobs", prompt, sp);

            RequestOutput output = engine.generate(request);

            assertThat(output).isNotNull();
            assertThat(output.finished()).isTrue();

            var completion = output.outputs().getFirst();
            assertThat(completion.text()).isNotEmpty();

            System.out.println("\n=== Logprobs Result ===");
            System.out.println("Output: " + completion.text());

            // Logprobs may be null/empty on backends that don't support them (e.g. vllm-metal)
            if (completion.logprobs() != null && !completion.logprobs().isEmpty()) {
                System.out.println("Logprob positions: " + completion.logprobs().size());

                // Each position should have a map with at most 5 entries
                for (var posMap : completion.logprobs()) {
                    assertThat(posMap).isNotNull();
                    assertThat(posMap).isNotEmpty();
                    assertThat(posMap.size()).isLessThanOrEqualTo(5);

                    // Verify LogprobEntry fields are populated
                    for (LogprobEntry entry : posMap.values()) {
                        assertThat(entry.rank()).isGreaterThanOrEqualTo(1);
                        assertThat(entry.logprob()).isLessThanOrEqualTo(0.0);
                        assertThat(entry.decodedToken()).isNotNull();
                    }
                }

                var first = completion.logprobs().getFirst().values().iterator().next();
                System.out.println("First position entries: " + completion.logprobs().getFirst().size());
                System.out.println("  tokenId=" + first.tokenId()
                        + " logprob=" + first.logprob()
                        + " rank=" + first.rank()
                        + " decoded='" + first.decodedToken() + "'");
            } else {
                System.out.println("Logprobs not available on this backend (expected on vllm-metal)");
            }

            System.out.println("=======================");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Tokenizer — encode/decode round-trip
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void tokenizer_encode_decode_round_trip() {
        String text = "Hello, world!";

        List<Integer> tokenIds = engine.encode(text);
        assertThat(tokenIds).isNotEmpty();

        String decoded = engine.decode(tokenIds);
        assertThat(decoded).isEqualTo(text);

        System.out.println("\n=== Tokenizer Round-Trip ===");
        System.out.println("Input:    '" + text + "'");
        System.out.println("Token IDs: " + tokenIds);
        System.out.println("Decoded:  '" + decoded + "'");
        System.out.println("============================");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Vocab size
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void vocab_size_should_be_positive() {
        int vocabSize = engine.vocabSize();

        // Qwen3-0.6B has a vocab of ~151,936 tokens
        assertThat(vocabSize).isGreaterThan(10000);

        System.out.println("\n=== Vocab Size ===");
        System.out.println("Vocab size: " + vocabSize);
        System.out.println("==================");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Tool-use template rendering — tools passed to Jinja2
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void render_with_tools_should_include_tool_definitions() {
        var tool = Tool.function("get_weather", "Get the current weather in a given location", Map.of(
                "type", "object",
                "properties", Map.of(
                        "location", Map.of("type", "string", "description", "The city name")
                ),
                "required", List.of("location")
        ));

        String prompt = chatTemplate.render(List.of(
                ChatMessage.system("You are a helpful assistant with access to tools. /no_think"),
                ChatMessage.user("What is the weather in Paris?")
        ), List.of(tool), true);

        assertThat(prompt).isNotEmpty();
        // The rendered template should contain the tool name somewhere
        assertThat(prompt).contains("get_weather");

        System.out.println("\n=== Tool-Use Template Rendering ===");
        System.out.println("Prompt length: " + prompt.length());
        System.out.println("Contains 'get_weather': " + prompt.contains("get_weather"));
        System.out.println("First 500 chars:\n" + prompt.substring(0, Math.min(500, prompt.length())));
        System.out.println("===================================");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Multi-turn tool-use — full conversation round-trip
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void render_multi_turn_tool_use_conversation() {
        var tool = Tool.function("get_weather", "Get weather", Map.of(
                "type", "object",
                "properties", Map.of(
                        "location", Map.of("type", "string")
                ),
                "required", List.of("location")
        ));

        // Full multi-turn conversation:
        // 1. User asks about weather
        // 2. Assistant decides to call get_weather
        // 3. Tool returns result
        // 4. Render with add_generation_prompt so assistant can respond
        var messages = List.of(
                ChatMessage.system("You are a helpful assistant. /no_think"),
                ChatMessage.user("What is the weather in Paris?"),
                ChatMessage.assistantWithToolCalls(null, List.of(
                        ToolCall.function("call_1", "get_weather", "{\"location\": \"Paris\"}")
                )),
                ChatMessage.toolResult("22°C, sunny", "call_1", "get_weather")
        );

        String prompt = chatTemplate.render(messages, List.of(tool), true);

        assertThat(prompt).isNotEmpty();
        // Should contain the tool result
        assertThat(prompt).contains("22°C, sunny");
        // Should contain the tool call function name
        assertThat(prompt).contains("get_weather");

        System.out.println("\n=== Multi-Turn Tool-Use Rendering ===");
        System.out.println("Prompt length: " + prompt.length());
        System.out.println("Contains 'get_weather': " + prompt.contains("get_weather"));
        System.out.println("Contains '22°C, sunny': " + prompt.contains("22°C, sunny"));
        System.out.println("Full prompt:\n" + prompt);
        System.out.println("=====================================");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Tool-use generation — model generates with tools available
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void generate_with_tools_should_produce_output() {
        var tool = Tool.function("get_weather", "Get the current weather", Map.of(
                "type", "object",
                "properties", Map.of(
                        "location", Map.of("type", "string", "description", "City name")
                ),
                "required", List.of("location")
        ));

        String prompt = chatTemplate.render(List.of(
                ChatMessage.system("You are a helpful assistant. Use tools when appropriate. /no_think"),
                ChatMessage.user("What is the weather in Tokyo?")
        ), List.of(tool), true);

        try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(300)) {
            var request = new VllmRequest("req-tools", prompt, sp);
            var output = engine.generate(request);

            assertThat(output).isNotNull();
            assertThat(output.finished()).isTrue();
            assertThat(output.outputs().getFirst().text()).isNotEmpty();

            System.out.println("\n=== Tool-Use Generation ===");
            System.out.println("Output: " + output.outputs().getFirst().text());
            System.out.println("===========================");
        }
    }

     // ═══════════════════════════════════════════════════════════════════
     //  Engine stats — snapshot of engine state
     // ═══════════════════════════════════════════════════════════════════

     @Test
     void engine_stats_should_return_model_info() {
         EngineStats stats = engine.getStats();

         assertThat(stats).isNotNull();
         assertThat(stats.model()).contains("Qwen3-0.6B");
         assertThat(stats.maxModelLen()).isGreaterThan(0);
         // No requests in progress → 0 unfinished
         assertThat(stats.numUnfinishedRequests()).isEqualTo(0);

         System.out.println("\n=== Engine Stats ===");
         System.out.println("Model: " + stats.model());
         System.out.println("Dtype: " + stats.dtype());
         System.out.println("Max model len: " + stats.maxModelLen());
         System.out.println("Unfinished: " + stats.numUnfinishedRequests());
         System.out.println("====================");
     }

     // ═══════════════════════════════════════════════════════════════════
     //  Tool calls — multi-turn streaming with tool invocation
     // ═══════════════════════════════════════════════════════════════════

     @Test
     void stream_tool_call_conversation_multi_turn() {
         // Define two tools
         var weatherTool = Tool.function("get_weather", "Get the weather for a location", Map.of(
                 "type", "object",
                 "properties", Map.of(
                         "location", Map.of("type", "string", "description", "City name"),
                         "unit", Map.of("type", "string", "enum", List.of("celsius", "fahrenheit"))
                 ),
                 "required", List.of("location")
         ));

         var calculatorTool = Tool.function("calculate", "Perform a calculation", Map.of(
                 "type", "object",
                 "properties", Map.of(
                         "expression", Map.of("type", "string", "description", "Math expression")
                 ),
                 "required", List.of("expression")
         ));

         var tools = List.of(weatherTool, calculatorTool);

         // First turn: User asks a question
         var messages = new ArrayList<>(List.of(
                 ChatMessage.system("You are a helpful assistant with access to tools. /no_think"),
                 ChatMessage.user("What is 2 plus 3?")
         ));

         // Generate first response (may use tools or answer directly)
         String prompt1 = chatTemplate.render(messages, tools, true);
         try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(128)) {
             var request1 = new VllmRequest("tool-conv-1", prompt1, sp);
             var iterator1 = new VllmIterator(engine);
             iterator1.addRequest(request1);

             StringBuilder response1 = new StringBuilder();
             iterator1.stream().forEach(output -> {
                 response1.append(output.delta());
                 assertThat(output.requestId()).isEqualTo("tool-conv-1");
             });

             String assistantResponse = response1.toString();
             System.out.println("\n=== Tool Call Stream Test (Turn 1) ===");
             System.out.println("User: What is 2 plus 3?");
             System.out.println("Assistant: " + assistantResponse);

             // For demonstration, assume the model might invoke a calculator tool
             // (In practice, whether it does depends on model capability)
             messages.add(ChatMessage.assistant(assistantResponse));

             // Simulate a tool call if the model didn't respond directly
             if (!assistantResponse.toLowerCase().contains("5")) {
                 // Add a simulated tool call
                 var toolCall = ToolCall.function("call_001", "calculate", "{\"expression\": \"2 + 3\"}");
                 messages.add(ChatMessage.assistantWithToolCalls(null, List.of(toolCall)));

                 // Add tool result
                 messages.add(ChatMessage.toolResult("5", "call_001", "calculate"));

                 // Second turn: Re-render with tool result and generate final answer
                 String prompt2 = chatTemplate.render(messages, tools, true);
                 try (var sp2 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(64)) {
                     var request2 = new VllmRequest("tool-conv-2", prompt2, sp2);
                     var iterator2 = new VllmIterator(engine);
                     iterator2.addRequest(request2);

                     StringBuilder response2 = new StringBuilder();
                     iterator2.stream().forEach(output -> {
                         response2.append(output.delta());
                         assertThat(output.requestId()).isEqualTo("tool-conv-2");
                     });

                     System.out.println("Assistant (after tool): " + response2.toString());
                     System.out.println("========================================");

                     assertThat(response2.toString()).isNotEmpty();
                 }
             }

             assertThat(response1.toString()).isNotEmpty();
         }
     }

     @Test
     void stream_tool_call_with_tool_tagging() {
         // Define a tool
         var tool = Tool.function("search", "Search for information", Map.of(
                 "type", "object",
                 "properties", Map.of(
                         "query", Map.of("type", "string", "description", "Search query")
                 ),
                 "required", List.of("query")
         ));

         String prompt = chatTemplate.render(List.of(
                 ChatMessage.system("You are a search assistant. Use tools to answer. /no_think"),
                 ChatMessage.user("Find information about the capital of France")
         ), List.of(tool), true);

         // Create a conversation state that tracks tool invocations
         var state = new ConversationState()
                 .toolCall("<tool_call>", "</tool_call>");

         try (var sp = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(256)) {
             var request = new VllmRequest("tool-stream-tag", prompt, sp);
             var iterator = new VllmIterator(engine);
             iterator.addRequest(request, state);

             StringBuilder toolSection = new StringBuilder();
             StringBuilder answerSection = new StringBuilder();

             // Stream and classify output regions
             iterator.stream().forEach(output -> {
                 GenerationState region = output.state();
                 String delta = output.delta();

                 if (region == GenerationState.TOOLS) {
                     toolSection.append(delta);
                 } else if (region == GenerationState.ANSWER) {
                     answerSection.append(delta);
                 }
             });

             System.out.println("\n=== Tool Call Tagging Test ===");
             System.out.println("Input tokens: " + state.inputTokens());
             System.out.println("Tool tokens:  " + state.toolsTokens());
             System.out.println("Answer tokens: " + state.answerTokens());
             System.out.println("Total output: " + state.totalOutputTokens());
             System.out.println("Finish reason: " + state.finishReason());
             if (toolSection.length() > 0) {
                 System.out.println("Tool section (first 200 chars):\n  "
                         + toolSection.toString().substring(0, Math.min(200, toolSection.length())));
             }
             if (answerSection.length() > 0) {
                 System.out.println("Answer section (first 200 chars):\n  "
                         + answerSection.toString().substring(0, Math.min(200, answerSection.length())));
             }
             System.out.println("==============================");

             // Verify state was updated
             assertThat(state.totalOutputTokens()).isGreaterThan(0);
             assertThat(state.finishReason()).isNotNull();
         }
     }

     @Test
     void parallel_tool_use_requests() {
         // Two parallel requests with tool context
         var tool = Tool.function("lookup", "Look up information", Map.of(
                 "type", "object",
                 "properties", Map.of(
                         "query", Map.of("type", "string")
                 ),
                 "required", List.of("query")
         ));

         String prompt1 = chatTemplate.render(List.of(
                 ChatMessage.system("Answer questions about world capitals. /no_think"),
                 ChatMessage.user("What is the capital of France?")
         ), List.of(tool), true);

         String prompt2 = chatTemplate.render(List.of(
                 ChatMessage.system("Answer questions about world capitals. /no_think"),
                 ChatMessage.user("What is the capital of Japan?")
         ), List.of(tool), true);

         var iterator = new VllmIterator(engine);

         try (var sp1 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(64);
              var sp2 = new SamplingParams(engine.arena()).temperature(0.0).maxTokens(64)) {

             iterator.addRequest(new VllmRequest("tool-france", prompt1, sp1));
             iterator.addRequest(new VllmRequest("tool-japan", prompt2, sp2));

             Map<String, StringBuilder> outputs = new HashMap<>();
             outputs.put("tool-france", new StringBuilder());
             outputs.put("tool-japan", new StringBuilder());

             // Stream both requests in parallel
             iterator.stream().forEach(output -> {
                 outputs.get(output.requestId()).append(output.delta());
             });

             System.out.println("\n=== Parallel Tool-Use Requests ===");
             System.out.println("France: " + outputs.get("tool-france").toString());
             System.out.println("Japan:  " + outputs.get("tool-japan").toString());
             System.out.println("==================================");

             assertThat(outputs.get("tool-france").toString()).isNotEmpty();
             assertThat(outputs.get("tool-japan").toString()).isNotEmpty();
         }
     }
}
