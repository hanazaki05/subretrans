"""Strict, independent configuration loaders for the agent workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import RoleModelSettings, load_api_roles
from .subtitle_edit import SubtitleEditSettings
from .subtitle_processing import POSTPROCESS_OPERATIONS


@dataclass(frozen=True)
class PipelineSettings:
    state_dir: Path
    checkpoint_db: Path
    primer_batch_size: int
    refine_batch_size: int | None
    primer_max_workers: int
    qa_batch_size: int
    qa_max_workers: int
    qa_window_offsets: tuple[int, ...]
    agent_max_repair_attempts: int
    source_language: str
    target_language: str
    user_instruction: str | None
    postprocess_operations: tuple[str, ...]
    episode_replacements: tuple[tuple[str, str], ...]


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


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _resolve_path(value: Any, field_name: str, yaml_dir: Path) -> Path:
    raw_path = Path(_nonempty_string(value, field_name))
    if not raw_path.is_absolute():
        raw_path = yaml_dir / raw_path
    return raw_path.resolve()


def load_pipeline_settings(yaml_path: str | Path) -> PipelineSettings:
    """Load the strict orchestration, primer, refine, and postprocess sections."""

    pipeline, yaml_dir = _load_section(yaml_path, "pipeline")
    _validate_fields(
        pipeline,
        "pipeline",
        required={"state_dir", "checkpoint_db", "agent_max_repair_attempts"},
    )
    primer, _ = _load_section(yaml_path, "primer")
    _validate_fields(
        primer,
        "primer",
        required={
            "batch_size",
            "max_workers",
            "source_language",
            "target_language",
            "user_instruction",
        },
    )
    refine, _ = _load_section(yaml_path, "refine")
    _validate_fields(
        refine,
        "refine",
        required={
            "batch_size",
            "chunk_token_soft_limit",
            "memory_token_limit",
            "intermediate_representation",
            "prompt_path",
        },
    )
    qa, _ = _load_section(yaml_path, "qa")
    _validate_fields(
        qa,
        "qa",
        required={"batch_size", "max_workers", "window_offsets"},
    )
    postprocess, _ = _load_section(yaml_path, "postprocess")
    _validate_fields(
        postprocess,
        "postprocess",
        required={"operations", "episode_replacements"},
    )

    user_instruction = primer["user_instruction"]
    if user_instruction is not None and not isinstance(user_instruction, str):
        raise ValueError("primer.user_instruction must be a string or null")

    refine_batch_size = refine["batch_size"]
    if refine_batch_size is not None:
        refine_batch_size = _positive_integer(
            refine_batch_size, "refine.batch_size"
        )

    raw_operations = postprocess["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError("postprocess.operations must be a list")
    postprocess_operations: list[str] = []
    for index, operation in enumerate(raw_operations):
        name = _nonempty_string(operation, f"postprocess.operations[{index}]")
        if name not in POSTPROCESS_OPERATIONS:
            raise ValueError(
                f"postprocess.operations[{index}] is unsupported: {name}"
            )
        if name in postprocess_operations:
            raise ValueError(f"postprocess.operations contains duplicate: {name}")
        postprocess_operations.append(name)

    raw_replacements = postprocess["episode_replacements"]
    if not isinstance(raw_replacements, list):
        raise ValueError("postprocess.episode_replacements must be a list")
    episode_replacements: list[tuple[str, str]] = []
    for index, replacement in enumerate(raw_replacements):
        prefix = f"postprocess.episode_replacements[{index}]"
        if type(replacement) is not dict or set(replacement) != {"from", "to"}:
            raise ValueError(f"{prefix} must contain exactly from and to")
        source = _nonempty_string(replacement["from"], f"{prefix}.from")
        target = replacement["to"]
        if not isinstance(target, str):
            raise ValueError(f"{prefix}.to must be a string")
        episode_replacements.append((source, target))

    qa_batch_size = _positive_integer(qa["batch_size"], "qa.batch_size")
    raw_offsets = qa["window_offsets"]
    if not isinstance(raw_offsets, list) or not raw_offsets:
        raise ValueError("qa.window_offsets must be a non-empty list")
    qa_window_offsets: list[int] = []
    for index, offset in enumerate(raw_offsets):
        if type(offset) is not int or not 0 <= offset < qa_batch_size:
            raise ValueError(
                f"qa.window_offsets[{index}] must be an integer from 0 to "
                f"{qa_batch_size - 1}"
            )
        if offset in qa_window_offsets:
            raise ValueError(f"qa.window_offsets contains duplicate: {offset}")
        qa_window_offsets.append(offset)
    if qa_window_offsets[0] != 0:
        raise ValueError("qa.window_offsets must start with 0")

    return PipelineSettings(
        state_dir=_resolve_path(pipeline["state_dir"], "pipeline.state_dir", yaml_dir),
        checkpoint_db=_resolve_path(
            pipeline["checkpoint_db"], "pipeline.checkpoint_db", yaml_dir
        ),
        primer_batch_size=_positive_integer(
            primer["batch_size"], "primer.batch_size"
        ),
        refine_batch_size=refine_batch_size,
        primer_max_workers=_positive_integer(
            primer["max_workers"], "primer.max_workers"
        ),
        qa_batch_size=qa_batch_size,
        qa_max_workers=_positive_integer(qa["max_workers"], "qa.max_workers"),
        qa_window_offsets=tuple(qa_window_offsets),
        agent_max_repair_attempts=_nonnegative_integer(
            pipeline["agent_max_repair_attempts"],
            "pipeline.agent_max_repair_attempts",
        ),
        source_language=_nonempty_string(
            primer["source_language"], "primer.source_language"
        ),
        target_language=_nonempty_string(
            primer["target_language"], "primer.target_language"
        ),
        user_instruction=user_instruction,
        postprocess_operations=tuple(postprocess_operations),
        episode_replacements=tuple(episode_replacements),
    )


def load_role_model_settings(
    yaml_path: str | Path, role: str
) -> RoleModelSettings:
    """Load one of the four strict model roles from ``api``."""

    try:
        return load_api_roles(yaml_path)[role]
    except KeyError as exc:
        raise ValueError(f"unsupported API role: {role}") from exc


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
            "settings_file",
            "multiple_replace_file",
            "first_pass_operations",
            "second_pass_operations",
        },
    )

    parsed_operations: dict[str, tuple[str, ...]] = {}
    for field_name in ("first_pass_operations", "second_pass_operations"):
        raw_operations = section[field_name]
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError(f"subtitle_edit.{field_name} must be a non-empty list")
        parsed_operations[field_name] = tuple(
            _nonempty_string(
                operation, f"subtitle_edit.{field_name}[{index}]"
            )
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
        settings_file=_resolve_path(
            section["settings_file"], "subtitle_edit.settings_file", yaml_dir
        ),
        multiple_replace_file=_resolve_path(
            section["multiple_replace_file"],
            "subtitle_edit.multiple_replace_file",
            yaml_dir,
        ),
        first_pass_operations=parsed_operations["first_pass_operations"],
        second_pass_operations=parsed_operations["second_pass_operations"],
    )
