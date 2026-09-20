"""LangChain chat-model construction and the single text transport for every role."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .stats import UsageStats


logger = logging.getLogger(__name__)

Message = tuple[str, str]
"""A ``(role, content)`` pair where role is ``"system"`` or ``"human"``."""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:[\w-]+)?[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL)


class ModelProtocol(StrEnum):
    """Wire protocol used to communicate with a chat model provider."""

    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"
    GOOGLE_GEMINI = "google-gemini"
    OPENAI_CHAT_COMPATIBLE = "openai-chat-compatible"


@dataclass(frozen=True)
class ModelConfig:
    """Provider-neutral settings needed to construct a chat model."""

    protocol: ModelProtocol
    model: str
    api_key: str
    base_url: str | None
    timeout: float | None
    max_retries: int = 0
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    """Construct the LangChain chat model selected by ``config.protocol``.

    Retries are delegated to the provider SDK through ``max_retries``; callers
    never implement their own retry loops.
    """

    common_args: dict[str, object] = {
        "model": config.model,
        "api_key": config.api_key,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    if config.base_url is not None:
        common_args["base_url"] = config.base_url
    if config.temperature is not None:
        common_args["temperature"] = config.temperature

    if config.protocol in (ModelProtocol.OPENAI_RESPONSES, ModelProtocol.OPENAI_CHAT_COMPATIBLE):
        if config.max_output_tokens is not None:
            common_args["max_tokens"] = config.max_output_tokens
        if config.reasoning_effort is not None:
            common_args["reasoning_effort"] = config.reasoning_effort
        return ChatOpenAI(
            **common_args,
            use_responses_api=config.protocol is ModelProtocol.OPENAI_RESPONSES,
            stream_usage=True,
        )

    if config.protocol is ModelProtocol.ANTHROPIC_MESSAGES:
        if config.max_output_tokens is not None:
            common_args["max_tokens"] = config.max_output_tokens
        return ChatAnthropic(**common_args)

    if config.protocol is ModelProtocol.GOOGLE_GEMINI:
        if config.max_output_tokens is not None:
            common_args["max_output_tokens"] = config.max_output_tokens
        if config.reasoning_effort is not None:
            common_args["reasoning_effort"] = config.reasoning_effort
        return ChatGoogleGenerativeAI(**common_args)

    raise ValueError(f"unsupported model protocol: {config.protocol}")


def usage_from_message(message: AIMessage) -> UsageStats:
    """Map LangChain usage metadata onto :class:`UsageStats`."""

    raw = message.usage_metadata or {}
    details = raw.get("output_token_details") or {}
    return UsageStats(
        prompt_tokens=raw.get("input_tokens", 0),
        completion_tokens=raw.get("output_tokens", 0),
        total_tokens=raw.get("total_tokens", 0),
        reasoning_tokens=details.get("reasoning", 0),
    )


def invoke_text(
    model: BaseChatModel,
    messages: Sequence[Message],
    *,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, UsageStats]:
    """Send ``messages`` and return the complete text plus usage.

    With ``stream=True`` every text delta is passed to ``on_chunk`` as it
    arrives; the merged message is returned either way.
    """

    if stream:
        merged: AIMessageChunk | None = None
        for chunk in model.stream(list(messages)):
            if not isinstance(chunk, AIMessageChunk):
                raise TypeError("streamed model response must be an AIMessageChunk")
            merged = chunk if merged is None else merged + chunk
            delta = chunk.text
            if delta and on_chunk is not None:
                on_chunk(delta)
        if merged is None:
            raise ValueError("model returned no streamed chunks")
        response: AIMessage = merged
    else:
        response = model.invoke(list(messages))
        if not isinstance(response, AIMessage):
            raise TypeError("model response must be an AIMessage")

    text = response.text
    if not isinstance(text, str):
        raise TypeError("AIMessage text must be a string")
    logger.debug(
        "Model raw response: content=%r metadata=%r",
        response.content,
        response.response_metadata,
    )
    if not text.strip():
        stop_reason = response.response_metadata.get("stop_reason", "unknown")
        raise ValueError(f"model returned no text content (stop_reason={stop_reason})")
    return text, usage_from_message(response)


def clean_response_text(text: str) -> str:
    """Strip ``<think>`` blocks and unwrap the first Markdown code fence, if any."""

    without_thinking = _THINK_BLOCK_RE.sub("", text).strip()
    fenced = _CODE_FENCE_RE.search(without_thinking)
    if fenced is not None:
        return fenced.group(1).strip()
    return without_thinking
