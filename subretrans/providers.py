"""LangChain model construction plus text and strict JSON-action transports."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError
from typing import Literal, Protocol, TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .stats import UsageStats


logger = logging.getLogger(__name__)

Message = tuple[str, str]
"""A ``(role, content)`` pair used by text and JSON-action transports."""

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


@dataclass(frozen=True)
class ToolDefinition:
    """Host-owned schema advertised to a model using the JSON-action protocol."""

    name: str
    description: str
    input_schema: Mapping[str, JSONValue]


@dataclass(frozen=True)
class ToolAction:
    """One strictly parsed model action."""

    kind: Literal["tool_call", "final"]
    name: str | None = None
    arguments: dict[str, JSONValue] | None = None
    result: JSONValue | None = None


@dataclass(frozen=True)
class ToolExchange:
    """One request/response/action unit suitable for run-scoped auditing."""

    step: int
    request: tuple[Message, ...]
    response: str
    action: ToolAction | None
    tool_result: JSONValue | None
    error: str | None


@dataclass(frozen=True)
class ToolLoopResult:
    """Final JSON result and all exchanges from one bounded tool loop."""

    final_result: JSONValue
    exchanges: tuple[ToolExchange, ...]
    usage: UsageStats
    steps: int


class ToolExecutor(Protocol):
    """Execute one validated tool call in the host-owned capability boundary."""

    def __call__(self, name: str, arguments: Mapping[str, JSONValue]) -> JSONValue:
        ...


class ToolLoopError(RuntimeError):
    """Base error for a malformed or exhausted JSON-action conversation."""

    def __init__(
        self,
        message: str,
        *,
        exchanges: tuple[ToolExchange, ...] = (),
        usage: UsageStats = UsageStats(),
    ) -> None:
        super().__init__(message)
        self.exchanges = exchanges
        self.usage = usage


class ToolStepLimitExceeded(ToolLoopError):
    """Raised when no final action arrives within ``max_tool_steps``."""

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


def _tool_protocol_message(tools: Sequence[ToolDefinition]) -> Message:
    """Serialize the advertised tool contract for a JSON-action request."""

    payload = {
        "protocol": "strict-json-actions-v1",
        "actions": ["tool_call", "final"],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in tools
        ],
        "constraints": [
            "Return exactly one JSON object per response.",
            "Return no Markdown, commentary, or multiple actions.",
            "Use action=tool_call until the task is complete, then action=final.",
        ],
    }
    try:
        encoded = _json_dumps(payload)
    except TypeError as exc:
        raise TypeError("tool definitions must contain JSON-serializable values") from exc
    return ("system", encoded)


def _json_dumps(value: object) -> str:
    """Encode a JSON value with stable keys for prompts and audit records."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_tool_definitions(tools: Sequence[ToolDefinition]) -> tuple[ToolDefinition, ...]:
    if not isinstance(tools, (tuple, list)) or not tools:
        raise ValueError("tools must be a non-empty tuple or list")
    normalized = tuple(tools)
    names: set[str] = set()
    for index, tool in enumerate(normalized):
        if not isinstance(tool, ToolDefinition):
            raise TypeError(f"tools[{index}] must be a ToolDefinition")
        if type(tool.name) is not str or not tool.name.strip():
            raise ValueError(f"tools[{index}].name must be a non-empty string")
        if tool.name in names:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if type(tool.description) is not str or not tool.description.strip():
            raise ValueError(f"tools[{index}].description must be a non-empty string")
        if not isinstance(tool.input_schema, Mapping):
            raise TypeError(f"tools[{index}].input_schema must be a mapping")
        names.add(tool.name)
    # Force serialization before the first model request so a malformed schema
    # cannot consume a tool step or leave a partial audit trail.
    _tool_protocol_message(normalized)
    return normalized


def _parse_tool_action(response_text: str, tool_names: set[str]) -> ToolAction:
    cleaned = clean_response_text(response_text)
    try:
        payload = json.loads(cleaned)
    except (JSONDecodeError, TypeError) as exc:
        raise ValueError("response must be exactly one JSON object") from exc
    if type(payload) is not dict:
        raise ValueError("response must be a JSON object")

    action = payload.get("action")
    if action == "tool_call":
        if set(payload) != {"action", "name", "arguments"}:
            raise ValueError("tool_call response must contain exactly action, name, arguments")
        name = payload["name"]
        arguments = payload["arguments"]
        if type(name) is not str or not name.strip():
            raise ValueError("tool_call.name must be a non-empty string")
        if name not in tool_names:
            raise ValueError(f"unknown tool: {name}")
        if type(arguments) is not dict:
            raise ValueError("tool_call.arguments must be a JSON object")
        return ToolAction("tool_call", name=name, arguments=arguments)

    if action == "final":
        if set(payload) != {"action", "result"}:
            raise ValueError("final response must contain exactly action and result")
        return ToolAction("final", result=payload["result"])

    raise ValueError("action must be either 'tool_call' or 'final'")


def _append_tool_result(
    messages: list[Message], response_text: str, name: str, result: JSONValue, *, ok: bool
) -> None:
    messages.append(("assistant", response_text))
    messages.append(
        (
            "human",
            _json_dumps(
                {
                    "tool_result": {
                        "name": name,
                        "ok": ok,
                        "result": result if ok else None,
                        "error": None if ok else result,
                    }
                }
            ),
        )
    )


def _append_protocol_error(messages: list[Message], response_text: str, error: str) -> None:
    messages.append(("assistant", response_text))
    messages.append(
        (
            "human",
            _json_dumps(
                {
                    "tool_protocol_error": {
                        "error": error,
                        "required": "exactly one JSON action object",
                    }
                }
            ),
        )
    )


def invoke_with_tools(
    model: BaseChatModel,
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition],
    execute: ToolExecutor,
    *,
    max_tool_steps: int,
    on_exchange: Callable[[ToolExchange], None] | None = None,
) -> ToolLoopResult:
    """Run a bounded, provider-neutral strict JSON action conversation.

    This is intentionally text-based. The host advertises typed tools in a
    system message, the model returns one JSON action, and the host appends a
    canonical JSON result as the next human message. No provider-native
    ``bind_tools`` or ``ToolMessage`` behavior is assumed, so custom gateways
    cannot silently change the protocol.
    """

    if type(max_tool_steps) is not int or max_tool_steps <= 0:
        raise ValueError("max_tool_steps must be a positive integer")
    if not callable(execute):
        raise TypeError("execute must be callable")
    normalized_tools = _validate_tool_definitions(tools)
    if not isinstance(messages, (tuple, list)):
        raise TypeError("messages must be a tuple or list")
    working_messages: list[Message] = [
        _tool_protocol_message(normalized_tools),
        *messages,
    ]
    tool_names = {tool.name for tool in normalized_tools}
    exchanges: list[ToolExchange] = []
    total_usage = UsageStats()

    for step in range(1, max_tool_steps + 1):
        request = tuple(working_messages)
        try:
            response_text, usage = invoke_text(model, request)
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            exchange = ToolExchange(step, request, "", None, None, error)
            exchanges.append(exchange)
            if on_exchange is not None:
                on_exchange(exchange)
            raise ToolLoopError(
                f"model invocation failed at tool step {step}: {error}",
                exchanges=tuple(exchanges),
                usage=total_usage,
            ) from exc
        total_usage += usage
        try:
            action = _parse_tool_action(response_text, tool_names)
        except ValueError as exc:
            exchange = ToolExchange(step, request, response_text, None, None, str(exc))
            exchanges.append(exchange)
            if on_exchange is not None:
                on_exchange(exchange)
            if step >= max_tool_steps:
                raise ToolStepLimitExceeded(
                    f"tool action protocol exceeded max_tool_steps={max_tool_steps}",
                    exchanges=tuple(exchanges),
                    usage=total_usage,
                ) from exc
            _append_protocol_error(working_messages, response_text, str(exc))
            continue

        if action.kind == "final":
            exchange = ToolExchange(step, request, response_text, action, None, None)
            exchanges.append(exchange)
            if on_exchange is not None:
                on_exchange(exchange)
            return ToolLoopResult(action.result, tuple(exchanges), total_usage, step)

        if action.name is None or action.arguments is None:
            raise ToolLoopError(
                "parsed tool_call is missing name or arguments",
                exchanges=tuple(exchanges),
                usage=total_usage,
            )
        try:
            tool_result = execute(action.name, action.arguments)
            _json_dumps(tool_result)
            exchange = ToolExchange(step, request, response_text, action, tool_result, None)
            exchanges.append(exchange)
            if on_exchange is not None:
                on_exchange(exchange)
            _append_tool_result(working_messages, response_text, action.name, tool_result, ok=True)
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            error_result: JSONValue = {"message": error}
            exchange = ToolExchange(step, request, response_text, action, error_result, error)
            exchanges.append(exchange)
            if on_exchange is not None:
                on_exchange(exchange)
            _append_tool_result(working_messages, response_text, action.name, error_result, ok=False)

    # The loop always returns or raises inside the range; keep a defensive
    # branch so future changes cannot accidentally create an unbounded caller.
    raise ToolStepLimitExceeded(
        f"tool action protocol exceeded max_tool_steps={max_tool_steps}",
        exchanges=tuple(exchanges),
        usage=total_usage,
    )


def clean_response_text(text: str) -> str:
    """Strip ``<think>`` blocks and unwrap the first Markdown code fence, if any."""

    without_thinking = _THINK_BLOCK_RE.sub("", text).strip()
    fenced = _CODE_FENCE_RE.search(without_thinking)
    if fenced is not None:
        return fenced.group(1).strip()
    return without_thinking
