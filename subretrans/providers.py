"""Protocol-specific LangChain chat model construction."""

from dataclasses import dataclass
from enum import StrEnum

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class ModelProtocol(StrEnum):
    """Wire protocol used to communicate with a chat model provider."""

    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"
    GOOGLE_GEMINI = "google-gemini"
    OPENAI_CHAT_COMPATIBLE = "openai-chat-compatible"

    @property
    def is_legacy(self) -> bool:
        """Whether this protocol is the legacy OpenAI-compatible chat API."""

        return self is ModelProtocol.OPENAI_CHAT_COMPATIBLE


@dataclass(frozen=True)
class ModelConfig:
    """Provider-neutral settings needed to construct a chat model."""

    protocol: ModelProtocol
    model: str
    api_key: str
    base_url: str | None
    timeout: float | None
    max_retries: int = 0


def build_chat_model(
    config: ModelConfig,
) -> ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI:
    """Construct the LangChain chat model selected by ``config.protocol``."""

    common_args = {
        "model": config.model,
        "api_key": config.api_key,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }

    if config.protocol is ModelProtocol.OPENAI_RESPONSES:
        if config.base_url is not None:
            common_args["base_url"] = config.base_url
        return ChatOpenAI(**common_args, use_responses_api=True)

    if config.protocol is ModelProtocol.ANTHROPIC_MESSAGES:
        if config.base_url is not None:
            common_args["base_url"] = config.base_url
        return ChatAnthropic(**common_args)

    if config.protocol is ModelProtocol.GOOGLE_GEMINI:
        if config.base_url is not None:
            raise ValueError("google-gemini does not accept a custom base_url")
        return ChatGoogleGenerativeAI(**common_args)

    if config.protocol is ModelProtocol.OPENAI_CHAT_COMPATIBLE:
        if config.base_url is not None:
            common_args["base_url"] = config.base_url
        return ChatOpenAI(**common_args, use_responses_api=False)

    raise ValueError(f"unsupported model protocol: {config.protocol}")
