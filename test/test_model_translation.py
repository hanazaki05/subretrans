import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from subretrans.model_translation import build_model_translate_batch
from subretrans.providers import ModelConfig, ModelProtocol
from subretrans.translation import TranslationRequest, TranslationResult


def model_config() -> ModelConfig:
    return ModelConfig(
        protocol=ModelProtocol.OPENAI_RESPONSES,
        model="translation-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        timeout=30.0,
    )


@patch("subretrans.model_translation.build_chat_model")
def test_builds_model_and_parses_strict_memoryless_batch(build_chat_model) -> None:
    model = build_chat_model.return_value
    model.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "translations": [
                    {"id": 3, "translation": "再见"},
                    {"id": 1, "translation": "你好"},
                ]
            }
        )
    )
    config = model_config()

    translate = build_model_translate_batch(
        config,
        source_language="English",
        target_language="Simplified Chinese",
        user_instruction="Keep line breaks.",
    )
    result = translate(
        (
            TranslationRequest(1, "Hello"),
            TranslationRequest(3, "Goodbye"),
        )
    )

    build_chat_model.assert_called_once_with(config)
    assert result == (
        TranslationResult(3, "再见"),
        TranslationResult(1, "你好"),
    )
    messages = model.invoke.call_args.args[0]
    system_prompt = messages[0][1]
    assert "English" in system_prompt
    assert "Simplified Chinese" in system_prompt
    assert "Keep line breaks." in system_prompt
    assert "no glossary, story context, or cross-batch memory" in system_prompt
    request_payload = json.loads(messages[1][1])
    assert request_payload == [
        {"id": 1, "source": "Hello"},
        {"id": 3, "source": "Goodbye"},
    ]
    assert all(set(item) == {"id", "source"} for item in request_payload)


@pytest.mark.parametrize(
    "payload, match",
    [
        (
            {
                "translations": [{"id": 1, "translation": "你好"}],
                "extra": True,
            },
            "unknown fields",
        ),
        (
            {"translations": [{"id": 1, "translation": "你好", "extra": True}]},
            "unknown fields",
        ),
        ({"translations": []}, "wrong number"),
        ({"translations": [{"id": 2, "translation": "你好"}]}, "do not match"),
        ({"translations": [{"id": 1, "translation": "   "}]}, "non-empty"),
    ],
)
@patch("subretrans.model_translation.build_chat_model")
def test_rejects_invalid_model_output(build_chat_model, payload, match) -> None:
    build_chat_model.return_value.invoke.return_value = AIMessage(
        content=json.dumps(payload)
    )
    translate = build_model_translate_batch(
        model_config(), source_language="English", target_language="Chinese"
    )

    with pytest.raises(ValueError, match=match):
        translate((TranslationRequest(1, "Hello"),))


@patch("subretrans.model_translation.build_chat_model")
def test_rejects_non_ai_message_response(build_chat_model) -> None:
    build_chat_model.return_value.invoke.return_value = '{"translations": []}'
    translate = build_model_translate_batch(
        model_config(), source_language="English", target_language="Chinese"
    )

    with pytest.raises(TypeError, match="AIMessage"):
        translate(())
