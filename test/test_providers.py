from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from subretrans.providers import ModelConfig, ModelProtocol, build_chat_model


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


def test_protocol_values_and_legacy_marker() -> None:
    assert [protocol.value for protocol in ModelProtocol] == [
        "openai-responses",
        "anthropic-messages",
        "google-gemini",
        "openai-chat-compatible",
    ]
    assert ModelProtocol.OPENAI_CHAT_COMPATIBLE.is_legacy
    assert not ModelProtocol.OPENAI_RESPONSES.is_legacy
    assert not ModelProtocol.ANTHROPIC_MESSAGES.is_legacy
    assert not ModelProtocol.GOOGLE_GEMINI.is_legacy


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
def test_google_gemini_routes_custom_base_url(chat_google) -> None:
    config = model_config(
        ModelProtocol.GOOGLE_GEMINI,
        base_url="https://gemini-gateway.example",
    )

    result = build_chat_model(config)

    assert result is chat_google.return_value
    chat_google.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        timeout=45.0,
        max_retries=0,
        base_url="https://gemini-gateway.example",
    )


@patch("subretrans.providers.ChatOpenAI")
def test_legacy_openai_chat_compatible_disables_responses_api(chat_openai) -> None:
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
    )
