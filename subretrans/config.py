"""Single strict loader for ``config.yaml`` shared by every entry point.

Every section is required and every field is validated once; entry points
receive one immutable :class:`AppConfig` and apply command-line overrides with
:func:`dataclasses.replace`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .providers import ModelConfig, ModelProtocol
from .subtitle_edit import SubtitleEditSettings
from .subtitle_processing import POSTPROCESS_OPERATIONS


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
API_ROLES = ("primer", "refine", "extraction", "agent", "repair")
INTERMEDIATE_REPRESENTATIONS = ("json", "xml-pair", "pseudo-toml")
CONFIG_SECTIONS = {
    "api",
    "pipeline",
    "prompts",
    "primer",
    "refine",
    "qa",
    "postprocess",
    "subtitle_edit",
    "glossary",
    "repair",
    "research",
    "reference_roots",
}
ROLE_FIELDS = {
    "protocol",
    "model",
    "key_file",
    "base_url",
    "timeout",
    "max_retries",
    "max_output_tokens",
    "reasoning_effort",
    "temperature",
}


def load_api_key_from_file(key_file_path: str | Path) -> str:
    """Read a non-empty API key from ``key_file_path``."""

    path = Path(key_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file must not be empty: {path}")
    return api_key


@dataclass(frozen=True)
class RoleModelSettings:
    """Complete provider and generation settings for one model role."""

    protocol: ModelProtocol
    model: str
    key_file: Path
    base_url: str | None
    timeout: float | None
    max_retries: int
    max_output_tokens: int
    reasoning_effort: str | None
    temperature: float | None

    @property
    def config(self) -> ModelConfig:
        """Build the provider-neutral model configuration, reading the key file."""

        return ModelConfig(
            protocol=self.protocol,
            model=self.model,
            api_key=load_api_key_from_file(self.key_file),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
        )


@dataclass(frozen=True)
class ApiRoles:
    """The five model roles used by the workflow."""

    primer: RoleModelSettings
    refine: RoleModelSettings
    extraction: RoleModelSettings
    agent: RoleModelSettings
    repair: RoleModelSettings

    def __getitem__(self, role: str) -> RoleModelSettings:
        if role not in API_ROLES:
            raise KeyError(f"unsupported API role: {role}")
        return getattr(self, role)

    def __iter__(self) -> Iterator[tuple[str, RoleModelSettings]]:
        return iter((role, getattr(self, role)) for role in API_ROLES)


@dataclass(frozen=True)
class PipelineSettings:
    state_dir: Path
    checkpoint_db: Path


@dataclass(frozen=True)
class PromptPaths:
    """Prompt components shared and composed by the pipeline stages."""

    shared: Path
    refine: Path
    qa: Path
    repair: Path


@dataclass(frozen=True)
class PrimerSettings:
    batch_size: int
    max_workers: int
    source_language: str
    target_language: str
    user_instruction: str | None


@dataclass(frozen=True)
class RefineSettings:
    batch_size: int | None
    chunk_token_soft_limit: int
    memory_token_limit: int
    intermediate_representation: str


@dataclass(frozen=True)
class QASettings:
    batch_size: int
    max_workers: int
    window_offsets: tuple[int, ...]


@dataclass(frozen=True)
class RepairSettings:
    """Finite budgets and bounds for the autonomous repair stage."""

    max_tool_steps: int
    max_full_sweeps: int
    max_repair_attempts: int
    context_radius: int
    max_group_span: int
    max_glossary_repair_attempts: int


@dataclass(frozen=True)
class ResearchSettings:
    """Bounded, read-only external research settings."""

    exa_key_file: Path | None
    timeout: float
    max_requests: int
    max_fetches_per_request: int
    max_response_bytes: int


@dataclass(frozen=True)
class PostprocessSettings:
    operations: tuple[str, ...]
    episode_replacements: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GlossarySettings:
    max_entries: int
    terminology_min_confidence: float


@dataclass(frozen=True)
class AppConfig:
    """Everything an entry point needs, loaded and validated exactly once."""

    path: Path
    api: ApiRoles
    pipeline: PipelineSettings
    prompts: PromptPaths
    primer: PrimerSettings
    refine: RefineSettings
    qa: QASettings
    repair: RepairSettings
    research: ResearchSettings
    postprocess: PostprocessSettings
    subtitle_edit: SubtitleEditSettings
    glossary: GlossarySettings
    reference_roots: tuple[Path, ...]

    @property
    def model_versions(self) -> dict[str, str]:
        """Configured model name per role, recorded in pipeline state."""

        return {role: settings.model for role, settings in self.api}


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_path(value: Any, field_name: str, yaml_dir: Path) -> Path:
    raw_path = Path(_nonempty_string(value, field_name))
    if not raw_path.is_absolute():
        raw_path = yaml_dir / raw_path
    return raw_path.resolve()


def _section(
    payload: dict[str, Any],
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    allowed = required | (optional or set())
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")
    missing = required - set(section)
    if missing:
        raise ValueError(f"{name} has missing fields: {', '.join(sorted(missing))}")
    return section


def _load_role(section: Any, role: str, yaml_dir: Path) -> RoleModelSettings:
    prefix = f"api.{role}"
    if not isinstance(section, dict):
        raise ValueError(f"{prefix} must be a mapping")
    unknown = set(section) - ROLE_FIELDS
    if unknown:
        raise ValueError(f"{prefix} has unknown fields: {', '.join(sorted(unknown))}")
    missing = ROLE_FIELDS - set(section)
    if missing:
        raise ValueError(f"{prefix} has missing fields: {', '.join(sorted(missing))}")

    protocol_name = _nonempty_string(section["protocol"], f"{prefix}.protocol")
    try:
        protocol = ModelProtocol(protocol_name)
    except ValueError as exc:
        raise ValueError(f"{prefix}.protocol is unsupported: {protocol_name}") from exc

    key_file = _resolve_path(section["key_file"], f"{prefix}.key_file", yaml_dir)
    load_api_key_from_file(key_file)

    base_url = section["base_url"]
    if base_url is not None:
        base_url = _nonempty_string(base_url, f"{prefix}.base_url")

    timeout = section["timeout"]
    if timeout is not None:
        if not _is_number(timeout) or timeout <= 0:
            raise ValueError(f"{prefix}.timeout must be a positive number of seconds or null")
        timeout = float(timeout)

    reasoning_effort = section["reasoning_effort"]
    if reasoning_effort is not None:
        reasoning_effort = _nonempty_string(reasoning_effort, f"{prefix}.reasoning_effort")

    temperature = section["temperature"]
    if temperature is not None:
        if not _is_number(temperature):
            raise ValueError(f"{prefix}.temperature must be a number or null")
        temperature = float(temperature)

    return RoleModelSettings(
        protocol=protocol,
        model=_nonempty_string(section["model"], f"{prefix}.model"),
        key_file=key_file,
        base_url=base_url,
        timeout=timeout,
        max_retries=_nonnegative_int(section["max_retries"], f"{prefix}.max_retries"),
        max_output_tokens=_positive_int(
            section["max_output_tokens"], f"{prefix}.max_output_tokens"
        ),
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )


def _load_api(payload: dict[str, Any], yaml_dir: Path) -> ApiRoles:
    api = payload.get("api")
    if not isinstance(api, dict):
        raise ValueError("api must be a mapping")
    unknown = set(api) - set(API_ROLES)
    if unknown:
        raise ValueError(f"api has unknown roles: {', '.join(sorted(unknown))}")
    missing = set(API_ROLES) - set(api)
    if missing:
        raise ValueError(f"api has missing roles: {', '.join(sorted(missing))}")
    return ApiRoles(**{role: _load_role(api[role], role, yaml_dir) for role in API_ROLES})


def _load_pipeline(payload: dict[str, Any], yaml_dir: Path) -> PipelineSettings:
    section = _section(payload, "pipeline", {"state_dir", "checkpoint_db"})
    return PipelineSettings(
        state_dir=_resolve_path(section["state_dir"], "pipeline.state_dir", yaml_dir),
        checkpoint_db=_resolve_path(
            section["checkpoint_db"], "pipeline.checkpoint_db", yaml_dir
        ),
    )


def _load_prompts(payload: dict[str, Any], yaml_dir: Path) -> PromptPaths:
    section = _section(
        payload, "prompts", {"shared_path", "refine_path", "qa_path", "repair_path"}
    )
    return PromptPaths(
        shared=_resolve_path(section["shared_path"], "prompts.shared_path", yaml_dir),
        refine=_resolve_path(section["refine_path"], "prompts.refine_path", yaml_dir),
        qa=_resolve_path(section["qa_path"], "prompts.qa_path", yaml_dir),
        repair=_resolve_path(section["repair_path"], "prompts.repair_path", yaml_dir),
    )


def _load_primer(payload: dict[str, Any]) -> PrimerSettings:
    section = _section(
        payload,
        "primer",
        {"batch_size", "max_workers", "source_language", "target_language", "user_instruction"},
    )
    user_instruction = section["user_instruction"]
    if user_instruction is not None and not isinstance(user_instruction, str):
        raise ValueError("primer.user_instruction must be a string or null")
    return PrimerSettings(
        batch_size=_positive_int(section["batch_size"], "primer.batch_size"),
        max_workers=_positive_int(section["max_workers"], "primer.max_workers"),
        source_language=_nonempty_string(section["source_language"], "primer.source_language"),
        target_language=_nonempty_string(section["target_language"], "primer.target_language"),
        user_instruction=user_instruction,
    )


def _load_refine(payload: dict[str, Any]) -> RefineSettings:
    section = _section(
        payload,
        "refine",
        {"batch_size", "chunk_token_soft_limit", "memory_token_limit", "intermediate_representation"},
    )
    batch_size = section["batch_size"]
    if batch_size is not None:
        batch_size = _positive_int(batch_size, "refine.batch_size")
    representation = _nonempty_string(
        section["intermediate_representation"], "refine.intermediate_representation"
    ).lower()
    if representation not in INTERMEDIATE_REPRESENTATIONS:
        raise ValueError(
            "refine.intermediate_representation must be one of: "
            + ", ".join(INTERMEDIATE_REPRESENTATIONS)
        )
    return RefineSettings(
        batch_size=batch_size,
        chunk_token_soft_limit=_positive_int(
            section["chunk_token_soft_limit"], "refine.chunk_token_soft_limit"
        ),
        memory_token_limit=_positive_int(
            section["memory_token_limit"], "refine.memory_token_limit"
        ),
        intermediate_representation=representation,
    )


def _load_qa(payload: dict[str, Any]) -> QASettings:
    section = _section(payload, "qa", {"batch_size", "max_workers", "window_offsets"})
    batch_size = _positive_int(section["batch_size"], "qa.batch_size")
    raw_offsets = section["window_offsets"]
    if not isinstance(raw_offsets, list) or not raw_offsets:
        raise ValueError("qa.window_offsets must be a non-empty list")
    offsets: list[int] = []
    for index, offset in enumerate(raw_offsets):
        if type(offset) is not int or not 0 <= offset < batch_size:
            raise ValueError(
                f"qa.window_offsets[{index}] must be an integer from 0 to {batch_size - 1}"
            )
        if offset in offsets:
            raise ValueError(f"qa.window_offsets contains duplicate: {offset}")
        offsets.append(offset)
    if offsets[0] != 0:
        raise ValueError("qa.window_offsets must start with 0")
    return QASettings(
        batch_size=batch_size,
        max_workers=_positive_int(section["max_workers"], "qa.max_workers"),
        window_offsets=tuple(offsets),
    )


def _load_repair(payload: dict[str, Any]) -> RepairSettings:
    section = _section(
        payload,
        "repair",
        {
            "max_tool_steps",
            "max_full_sweeps",
            "max_repair_attempts",
            "context_radius",
            "max_group_span",
            "max_glossary_repair_attempts",
        },
    )
    max_group_span = _positive_int(section["max_group_span"], "repair.max_group_span")
    if max_group_span > 3:
        raise ValueError("repair.max_group_span must not exceed 3")
    return RepairSettings(
        max_tool_steps=_positive_int(section["max_tool_steps"], "repair.max_tool_steps"),
        max_full_sweeps=_positive_int(
            section["max_full_sweeps"], "repair.max_full_sweeps"
        ),
        max_repair_attempts=_nonnegative_int(
            section["max_repair_attempts"], "repair.max_repair_attempts"
        ),
        context_radius=_nonnegative_int(
            section["context_radius"], "repair.context_radius"
        ),
        max_group_span=max_group_span,
        max_glossary_repair_attempts=_nonnegative_int(
            section["max_glossary_repair_attempts"],
            "repair.max_glossary_repair_attempts",
        ),
    )


def _load_reference_roots(payload: dict[str, Any], yaml_dir: Path) -> tuple[Path, ...]:
    values = payload["reference_roots"]
    if not isinstance(values, list):
        raise ValueError("reference_roots must be a list")
    roots = tuple(
        _resolve_path(value, f"reference_roots[{index}]", yaml_dir)
        for index, value in enumerate(values)
    )
    if len(roots) != 1:
        raise ValueError("reference_roots must contain exactly one root")
    return roots


def _load_research(payload: dict[str, Any], yaml_dir: Path) -> ResearchSettings:
    section = _section(
        payload,
        "research",
        {
            "exa_key_file",
            "timeout",
            "max_requests",
            "max_fetches_per_request",
            "max_response_bytes",
        },
    )
    exa_key_file = section["exa_key_file"]
    if exa_key_file is not None:
        exa_key_file = _resolve_path(exa_key_file, "research.exa_key_file", yaml_dir)
    timeout = section["timeout"]
    if not _is_number(timeout) or timeout <= 0:
        raise ValueError("research.timeout must be a positive number of seconds")
    return ResearchSettings(
        exa_key_file=exa_key_file,
        timeout=float(timeout),
        max_requests=_positive_int(section["max_requests"], "research.max_requests"),
        max_fetches_per_request=_positive_int(
            section["max_fetches_per_request"], "research.max_fetches_per_request"
        ),
        max_response_bytes=_positive_int(
            section["max_response_bytes"], "research.max_response_bytes"
        ),
    )


def _load_postprocess(payload: dict[str, Any]) -> PostprocessSettings:
    section = _section(payload, "postprocess", {"operations", "episode_replacements"})
    raw_operations = section["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError("postprocess.operations must be a list")
    operations: list[str] = []
    for index, operation in enumerate(raw_operations):
        name = _nonempty_string(operation, f"postprocess.operations[{index}]")
        if name not in POSTPROCESS_OPERATIONS:
            raise ValueError(f"postprocess.operations[{index}] is unsupported: {name}")
        if name in operations:
            raise ValueError(f"postprocess.operations contains duplicate: {name}")
        operations.append(name)

    raw_replacements = section["episode_replacements"]
    if not isinstance(raw_replacements, list):
        raise ValueError("postprocess.episode_replacements must be a list")
    replacements: list[tuple[str, str]] = []
    for index, replacement in enumerate(raw_replacements):
        prefix = f"postprocess.episode_replacements[{index}]"
        if type(replacement) is not dict or set(replacement) != {"from", "to"}:
            raise ValueError(f"{prefix} must contain exactly from and to")
        source = _nonempty_string(replacement["from"], f"{prefix}.from")
        target = replacement["to"]
        if not isinstance(target, str):
            raise ValueError(f"{prefix}.to must be a string")
        replacements.append((source, target))
    return PostprocessSettings(
        operations=tuple(operations), episode_replacements=tuple(replacements)
    )


def _load_subtitle_edit(payload: dict[str, Any], yaml_dir: Path) -> SubtitleEditSettings:
    section = _section(
        payload,
        "subtitle_edit",
        {
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
    passes: dict[str, tuple[str, ...]] = {}
    for field_name in ("first_pass_operations", "second_pass_operations"):
        raw_operations = section[field_name]
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError(f"subtitle_edit.{field_name} must be a non-empty list")
        passes[field_name] = tuple(
            _nonempty_string(operation, f"subtitle_edit.{field_name}[{index}]")
            for index, operation in enumerate(raw_operations)
        )
    return SubtitleEditSettings(
        repository_url=_nonempty_string(
            section["repository_url"], "subtitle_edit.repository_url"
        ),
        revision=_nonempty_string(section["revision"], "subtitle_edit.revision"),
        source_dir=_resolve_path(section["source_dir"], "subtitle_edit.source_dir", yaml_dir),
        build_dir=_resolve_path(section["build_dir"], "subtitle_edit.build_dir", yaml_dir),
        dotnet_executable=_nonempty_string(
            section["dotnet_executable"], "subtitle_edit.dotnet_executable"
        ),
        settings_file=_resolve_path(
            section["settings_file"], "subtitle_edit.settings_file", yaml_dir
        ),
        multiple_replace_file=_resolve_path(
            section["multiple_replace_file"], "subtitle_edit.multiple_replace_file", yaml_dir
        ),
        first_pass_operations=passes["first_pass_operations"],
        second_pass_operations=passes["second_pass_operations"],
    )


def _load_glossary(payload: dict[str, Any]) -> GlossarySettings:
    section = _section(payload, "glossary", {"max_entries", "terminology_min_confidence"})
    confidence = section["terminology_min_confidence"]
    if not _is_number(confidence) or not 0 <= confidence <= 1:
        raise ValueError("glossary.terminology_min_confidence must be a number from 0 to 1")
    return GlossarySettings(
        max_entries=_positive_int(section["max_entries"], "glossary.max_entries"),
        terminology_min_confidence=float(confidence),
    )


def load_config(yaml_path: str | Path | None = None) -> AppConfig:
    """Load and strictly validate the complete configuration file."""

    path = Path(yaml_path or DEFAULT_CONFIG_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    unknown_sections = set(payload) - CONFIG_SECTIONS
    if unknown_sections:
        raise ValueError(
            "configuration has unknown sections: " + ", ".join(sorted(unknown_sections))
        )
    missing_sections = CONFIG_SECTIONS - set(payload)
    if missing_sections:
        raise ValueError(
            "configuration has missing sections: " + ", ".join(sorted(missing_sections))
        )

    yaml_dir = path.parent
    return AppConfig(
        path=path,
        api=_load_api(payload, yaml_dir),
        pipeline=_load_pipeline(payload, yaml_dir),
        prompts=_load_prompts(payload, yaml_dir),
        primer=_load_primer(payload),
        refine=_load_refine(payload),
        qa=_load_qa(payload),
        repair=_load_repair(payload),
        research=_load_research(payload, yaml_dir),
        postprocess=_load_postprocess(payload),
        subtitle_edit=_load_subtitle_edit(payload, yaml_dir),
        glossary=_load_glossary(payload),
        reference_roots=_load_reference_roots(payload, yaml_dir),
    )
