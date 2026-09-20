from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from subretrans.providers import (
    ModelConfig,
    ModelProtocol,
    build_chat_model,
    clean_response_text,
    invoke_text,
)
from subretrans.stats import UsageStats


def model_config(
    protocol: ModelProtocol,
    *,
    base_url: str | None = None,
    max_retries: int = 0,
) -> ModelConfig:
    return ModelConfig(
        protocol=protocol,
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        timeout=45.0,
        max_retries=max_retries,
    )


def test_protocol_values() -> None:
    assert [protocol.value for protocol in ModelProtocol] == [
        "openai-responses",
        "anthropic-messages",
        "google-gemini",
        "openai-chat-compatible",
    ]


def test_model_config_is_frozen_and_disables_retries_by_default() -> None:
    config = ModelConfig(
        protocol=ModelProtocol.OPENAI_RESPONSES,
        model="test-model",
        api_key="test-key",
        base_url=None,
        timeout=45.0,
    )

    assert config.max_retries == 0
    with pytest.raises(FrozenInstanceError):
        config.model = "replacement"  # type: ignore[misc]


@patch("subretrans.providers.ChatOpenAI")
def test_openai_responses_maps_generation_settings(chat_openai) -> None:
    config = ModelConfig(
        protocol=ModelProtocol.OPENAI_RESPONSES,
        model="test-model",
        api_key="test-key",
        base_url=None,
        timeout=45.0,
        max_output_tokens=2048,
        reasoning_effort="high",
        temperature=0.2,
    )

    build_chat_model(config)

    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=0,
        max_tokens=2048,
        reasoning_effort="high",
        temperature=0.2,
        use_responses_api=True,
        stream_usage=True,
    )


@patch("subretrans.providers.ChatOpenAI")
def test_openai_responses_routes_custom_base_url(chat_openai) -> None:
    config = model_config(
        ModelProtocol.OPENAI_RESPONSES,
        base_url="https://gemini-responses.example/v1",
        max_retries=2,
    )

    result = build_chat_model(config)

    assert result is chat_openai.return_value
    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=2,
        base_url="https://gemini-responses.example/v1",
        use_responses_api=True,
        stream_usage=True,
    )


@patch("subretrans.providers.ChatAnthropic")
def test_anthropic_messages_routes_custom_base_url(chat_anthropic) -> None:
    config = model_config(
        ModelProtocol.ANTHROPIC_MESSAGES,
        base_url="https://anthropic-gateway.example",
        max_retries=1,
    )

    result = build_chat_model(config)

    assert result is chat_anthropic.return_value
    chat_anthropic.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=1,
        base_url="https://anthropic-gateway.example",
    )


@patch("subretrans.providers.ChatGoogleGenerativeAI")
def test_google_gemini_routes_native_parameters(chat_google) -> None:
    result = build_chat_model(model_config(ModelProtocol.GOOGLE_GEMINI))

    assert result is chat_google.return_value
    chat_google.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=0,
    )


@patch("subretrans.providers.ChatGoogleGenerativeAI")
def test_google_gemini_maps_reasoning_effort_and_output_limit(chat_google) -> None:
    config = ModelConfig(
        protocol=ModelProtocol.GOOGLE_GEMINI,
        model="gemini-test",
        api_key="test-key",
        base_url="https://gemini-gateway.example",
        timeout=45.0,
        max_output_tokens=512,
        reasoning_effort="high",
    )

    build_chat_model(config)

    chat_google.assert_called_once_with(
        model="gemini-test",
        api_key="test-key",
        timeout=45.0,
        max_retries=0,
        base_url="https://gemini-gateway.example",
        max_output_tokens=512,
        reasoning_effort="high",
    )


@patch("subretrans.providers.ChatOpenAI")
def test_openai_chat_compatible_disables_responses_api(chat_openai) -> None:
    config = model_config(
        ModelProtocol.OPENAI_CHAT_COMPATIBLE,
        base_url="https://chat-compatible.example/v1",
    )

    result = build_chat_model(config)

    assert result is chat_openai.return_value
    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=0,
        base_url="https://chat-compatible.example/v1",
        use_responses_api=False,
        stream_usage=True,
    )


def test_invoke_text_returns_text_and_usage() -> None:
    response = AIMessage(
        content="result",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "output_token_details": {"reasoning": 3},
        },
    )
    seen = []
    model = SimpleNamespace(invoke=lambda messages: seen.append(messages) or response)

    text, usage = invoke_text(model, (("system", "rules"), ("human", "input")))

    assert text == "result"
    assert usage == UsageStats(11, 7, 18, 3)
    assert seen == [[("system", "rules"), ("human", "input")]]


def test_invoke_text_streams_deltas_and_merges_usage() -> None:
    chunks = [
        AIMessageChunk(content="<pa"),
        AIMessageChunk(content="ir>"),
        AIMessageChunk(
            content="",
            usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        ),
    ]
    model = SimpleNamespace(stream=lambda messages: iter(chunks))
    deltas: list[str] = []

    text, usage = invoke_text(model, [("human", "x")], stream=True, on_chunk=deltas.append)

    assert text == "<pair>"
    assert deltas == ["<pa", "ir>"]
    assert usage == UsageStats(5, 2, 7, 0)


def test_invoke_text_rejects_empty_text_with_stop_reason() -> None:
    response = AIMessage(
        content=[{"type": "thinking", "thinking": "still auditing"}],
        response_metadata={"stop_reason": "max_tokens"},
    )
    model = SimpleNamespace(invoke=lambda messages: response)

    with pytest.raises(ValueError, match="no text content.*max_tokens"):
        invoke_text(model, [("human", "x")])


def test_invoke_text_rejects_non_message_responses() -> None:
    with pytest.raises(TypeError, match="AIMessage"):
        invoke_text(SimpleNamespace(invoke=lambda messages: "text"), [("human", "x")])
    with pytest.raises(TypeError, match="AIMessageChunk"):
        invoke_text(
            SimpleNamespace(stream=lambda messages: iter(["text"])),
            [("human", "x")],
            stream=True,
        )
    with pytest.raises(ValueError, match="no streamed chunks"):
        invoke_text(SimpleNamespace(stream=lambda messages: iter(())), [("human", "x")], stream=True)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ("<think>\nplanning\n</think>\n[1, 2]", "[1, 2]"),
        ("Sure:\n```json\n{\"a\": 1}\n```\nDone.", '{"a": 1}'),
        ("```\n<pair>\nID=1\n</pair>\n```", "<pair>\nID=1\n</pair>"),
        ("<THINK>x</THINK>```xml-pair\n<pair>\n</pair>\n```", "<pair>\n</pair>"),
    ],
)
def test_clean_response_text(raw: str, expected: str) -> None:
    assert clean_response_text(raw) == expected
