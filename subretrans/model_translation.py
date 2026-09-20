"""LangChain-backed translation batches for the memoryless first pass."""

from __future__ import annotations

import json

from .fsutil import require_exact_fields
from .providers import ModelConfig, build_chat_model, clean_response_text, invoke_text
from .translation import TranslateBatch, TranslationBatch, TranslationResult


def build_model_translate_batch(
    config: ModelConfig,
    *,
    source_language: str,
    target_language: str,
    user_instruction: str | None = None,
) -> TranslateBatch:
    """Build a strict, memoryless model-backed ``TranslateBatch`` callable."""

    model = build_chat_model(config)
    instruction = user_instruction if user_instruction is not None else "none"
    system_prompt = (
        "Perform the initial subtitle translation from "
        f"{source_language} to {target_language}. "
        "This pass has no glossary, story context, or cross-batch memory. "
        "Use only the source text in the current batch. "
        f"Additional user instruction: {instruction}\n"
        "The user message is a JSON array whose elements contain exactly "
        'the keys "id" and "source". Return one strict JSON object with '
        'exactly the key "translations". Its value must be an array with '
        "one element per input element; every element must contain exactly "
        'the keys "id" and "translation". Copy every input id exactly and '
        "put a non-empty translated string in translation. Return JSON only."
    )

    def translate_batch(batch: TranslationBatch) -> tuple[TranslationResult, ...]:
        request_payload = [{"id": request.id, "source": request.source} for request in batch]
        response_text, _ = invoke_text(
            model,
            [
                ("system", system_prompt),
                ("human", json.dumps(request_payload, ensure_ascii=False)),
            ],
        )
        payload = require_exact_fields(
            json.loads(clean_response_text(response_text)),
            {"translations"},
            location="response",
        )
        raw_translations = payload["translations"]
        if type(raw_translations) is not list:
            raise ValueError("response.translations must be a JSON array")
        if len(raw_translations) != len(batch):
            raise ValueError("response has the wrong number of translations")

        translations: list[TranslationResult] = []
        for index, raw_translation in enumerate(raw_translations):
            item = require_exact_fields(
                raw_translation,
                {"id", "translation"},
                location=f"response.translations[{index}]",
            )
            item_id = item["id"]
            translation = item["translation"]
            if type(item_id) is not int:
                raise ValueError(f"response.translations[{index}].id must be an integer")
            if type(translation) is not str or not translation.strip():
                raise ValueError(
                    f"response.translations[{index}].translation must be a "
                    "non-empty string"
                )
            translations.append(TranslationResult(item_id, translation))

        expected_ids = sorted(request.id for request in batch)
        returned_ids = sorted(result.id for result in translations)
        if returned_ids != expected_ids:
            raise ValueError("response translation ids do not match the batch ids")
        return tuple(translations)

    return translate_batch
