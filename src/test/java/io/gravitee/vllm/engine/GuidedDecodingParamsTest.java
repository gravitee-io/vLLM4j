package io.gravitee.vllm.engine;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for {@link GuidedDecodingParams} factory methods.
 *
 * <p>The {@code toPython()} method is package-private and requires CPython,
 * so it is tested at the integration level. These tests verify the Java-side
 * construction and factory method behavior.
 */
class GuidedDecodingParamsTest {

    @Test
    void json_shouldCreateInstance() {
        String schema = """
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}""";
        var params = GuidedDecodingParams.json(schema);
        assertThat(params).isNotNull();
    }

    @Test
    void regex_shouldCreateInstance() {
        var params = GuidedDecodingParams.regex("[A-Z][a-z]+");
        assertThat(params).isNotNull();
    }

    @Test
    void choice_shouldCreateInstance() {
        var params = GuidedDecodingParams.choice(List.of("yes", "no", "maybe"));
        assertThat(params).isNotNull();
    }

    @Test
    void grammar_shouldCreateInstance() {
        var params = GuidedDecodingParams.grammar("root ::= 'hello' | 'world'");
        assertThat(params).isNotNull();
    }

    @Test
    void jsonObject_shouldCreateInstance() {
        var params = GuidedDecodingParams.jsonObject();
        assertThat(params).isNotNull();
    }

    @Test
    void differentFactories_shouldCreateDistinctInstances() {
        var json = GuidedDecodingParams.json("{}");
        var regex = GuidedDecodingParams.regex(".*");
        var choice = GuidedDecodingParams.choice(List.of("a"));
        var grammar = GuidedDecodingParams.grammar("root ::= 'x'");
        var jsonObject = GuidedDecodingParams.jsonObject();

        // All are distinct instances
        assertThat(json).isNotSameAs(regex);
        assertThat(regex).isNotSameAs(choice);
        assertThat(choice).isNotSameAs(grammar);
        assertThat(grammar).isNotSameAs(jsonObject);
    }
}
