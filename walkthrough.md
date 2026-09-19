# Multiple API Formats & Stream Flag Rename

## Summary

Added support for the OpenAI **Responses API** alongside the existing chat-completion API, renamed `--streaming` to `--stream`, and updated the `Mode:` display to clearly indicate which API format is in use.

## Changes Made

### Config

| File | Change |
|------|--------|
| [config.yaml](file:///Users/zerozaki07/Downloads/subretrans/config.yaml) | Added `api_mode: "chat-completion"`, renamed [use_streaming](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152) → [use_stream](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152) |
| [config.py](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py) | Added `api_mode` field + validation, [use_stream](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152) field, backward-compat [use_streaming](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152) property alias |

### CLI & Display

| File | Change |
|------|--------|
| [cli.py](file:///Users/zerozaki07/Downloads/subretrans/subretrans/cli.py) | New `--api-mode` flag, renamed `--stream`/`--no-stream` (old flags kept as hidden deprecated aliases), mode display now shows e.g. `openai chat-completion stream` |

### API Layer

| File | Change |
|------|--------|
| [llm.py](file:///Users/zerozaki07/Downloads/subretrans/subretrans/llm.py) | Added [call_openai_api_response()](file:///Users/zerozaki07/Downloads/subretrans/subretrans/llm.py#1088-1261) and [refine_chunk_sdk_response()](file:///Users/zerozaki07/Downloads/subretrans/subretrans/llm.py#1263-1431) for Responses API (always `stream=True`) |
| [__init__.py](file:///Users/zerozaki07/Downloads/subretrans/subretrans/__init__.py) | Exported new functions |
| [run.sh](file:///Users/zerozaki07/Downloads/subretrans/run.sh) | Updated example comments |

## Mode Display Examples

```
Mode:      openai chat-completion stream       # --stream (or --api-mode chat-completion --stream)
Mode:      openai chat-completion non-stream   # --no-stream (default chat-completion)
Mode:      openai response stream              # --api-mode response (always streams)
```

## Backward Compatibility

- `--streaming` / `--no-streaming` still work (hidden aliases)
- [use_streaming](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152) in YAML config is accepted as fallback for [use_stream](file:///Users/zerozaki07/Downloads/subretrans/subretrans/config.py#148-152)
- `config.use_streaming` property alias still works in code

## Verification

- ✅ All 4 Python files pass `ast.parse()` syntax check
