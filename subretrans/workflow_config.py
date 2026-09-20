"""Strict, independent configuration loaders for the agent workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import providers
from .subtitle_edit import SubtitleEditSettings


@dataclass(frozen=True)
class PipelineSettings:
    state_dir: Path
    checkpoint_db: Path
    batch_size: int
    max_workers: int
    source_language: str
    target_language: str
    user_instruction: str | None
    episode_replacements: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TranslationModelSettings:
    config: providers.ModelConfig


def _load_section(yaml_path: str | Path, section_name: str) -> tuple[dict[str, Any], Path]:
    path = Path(yaml_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")

    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be a mapping")
    return section, path.parent


def _validate_fields(
    section: dict[str, Any],
    section_name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"{section_name} has unknown fields: {', '.join(sorted(unknown))}"
        )
    missing = required - set(section)
    if missing:
        raise ValueError(
            f"{section_name} has missing fields: {', '.join(sorted(missing))}"
        )


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _resolve_path(value: Any, field_name: str, yaml_dir: Path) -> Path:
    raw_path = Path(_nonempty_string(value, field_name))
    if not raw_path.is_absolute():
        raw_path = yaml_dir / raw_path
    return raw_path.resolve()


def load_pipeline_settings(yaml_path: str | Path) -> PipelineSettings:
    """Load only the non-secret ``pipeline`` section of a workflow config."""

    section, yaml_dir = _load_section(yaml_path, "pipeline")
    _validate_fields(
        section,
        "pipeline",
        required={
            "state_dir",
            "checkpoint_db",
            "batch_size",
            "max_workers",
            "source_language",
            "target_language",
            "episode_replacements",
        },
        optional={"user_instruction"},
    )

    user_instruction = section.get("user_instruction")
    if user_instruction is not None and not isinstance(user_instruction, str):
        raise ValueError("pipeline.user_instruction must be a string or null")

    raw_replacements = section["episode_replacements"]
    if not isinstance(raw_replacements, list):
        raise ValueError("pipeline.episode_replacements must be a list")
    episode_replacements: list[tuple[str, str]] = []
    for index, replacement in enumerate(raw_replacements):
        if type(replacement) is not dict or set(replacement) != {"from", "to"}:
            raise ValueError(
                f"pipeline.episode_replacements[{index}] must contain exactly from and to"
            )
        source = _nonempty_string(
            replacement["from"],
            f"pipeline.episode_replacements[{index}].from",
        )
        target = replacement["to"]
        if not isinstance(target, str):
            raise ValueError(
                f"pipeline.episode_replacements[{index}].to must be a string"
            )
        episode_replacements.append((source, target))

    return PipelineSettings(
        state_dir=_resolve_path(section["state_dir"], "pipeline.state_dir", yaml_dir),
        checkpoint_db=_resolve_path(
            section["checkpoint_db"], "pipeline.checkpoint_db", yaml_dir
        ),
        batch_size=_positive_integer(section["batch_size"], "pipeline.batch_size"),
        max_workers=_positive_integer(
            section["max_workers"], "pipeline.max_workers"
        ),
        source_language=_nonempty_string(
            section["source_language"], "pipeline.source_language"
        ),
        target_language=_nonempty_string(
            section["target_language"], "pipeline.target_language"
        ),
        user_instruction=user_instruction,
        episode_replacements=tuple(episode_replacements),
    )


def load_translation_model_settings(
    yaml_path: str | Path,
) -> TranslationModelSettings:
    """Load the strict ``translation_model`` section and its API key file."""

    section, yaml_dir = _load_section(yaml_path, "translation_model")
    _validate_fields(
        section,
        "translation_model",
        required={
            "protocol",
            "name",
            "key_file",
            "base_url",
            "timeout",
            "max_retries",
        },
    )

    protocol_name = _nonempty_string(
        section["protocol"], "translation_model.protocol"
    )
    try:
        protocol = providers.ModelProtocol(protocol_name)
    except ValueError as exc:
        raise ValueError(
            f"translation_model.protocol is unsupported: {protocol_name}"
        ) from exc

    model_name = _nonempty_string(section["name"], "translation_model.name")
    key_path = _resolve_path(
        section["key_file"], "translation_model.key_file", yaml_dir
    )
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("translation_model key file must not be empty")

    base_url = section["base_url"]
    if base_url is not None:
        base_url = _nonempty_string(base_url, "translation_model.base_url")

    timeout = section["timeout"]
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("translation_model.timeout must be a positive number or null")
        if timeout <= 0:
            raise ValueError("translation_model.timeout must be a positive number or null")
        timeout = float(timeout)

    max_retries = section["max_retries"]
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("translation_model.max_retries must be a non-negative integer")

    return TranslationModelSettings(
        config=providers.ModelConfig(
            protocol=protocol,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    )


def load_subtitle_edit_settings(yaml_path: str | Path) -> SubtitleEditSettings:
    """Load the strict ``subtitle_edit`` section."""

    section, yaml_dir = _load_section(yaml_path, "subtitle_edit")
    _validate_fields(
        section,
        "subtitle_edit",
        required={
            "repository_url",
            "revision",
            "source_dir",
            "build_dir",
            "dotnet_executable",
            "operations",
        },
    )

    raw_operations = section["operations"]
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("subtitle_edit.operations must be a non-empty list")
    operations = tuple(
        _nonempty_string(operation, f"subtitle_edit.operations[{index}]")
        for index, operation in enumerate(raw_operations)
    )

    return SubtitleEditSettings(
        repository_url=_nonempty_string(
            section["repository_url"], "subtitle_edit.repository_url"
        ),
        revision=_nonempty_string(section["revision"], "subtitle_edit.revision"),
        source_dir=_resolve_path(
            section["source_dir"], "subtitle_edit.source_dir", yaml_dir
        ),
        build_dir=_resolve_path(
            section["build_dir"], "subtitle_edit.build_dir", yaml_dir
        ),
        dotnet_executable=_nonempty_string(
            section["dotnet_executable"], "subtitle_edit.dotnet_executable"
        ),
        operations=operations,
    )
