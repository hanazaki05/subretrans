"""Configuration loading for subtitle model roles and runtime settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .providers import ModelConfig, ModelProtocol


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
API_ROLES = ("primer", "refine", "extraction", "agent")
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


def load_yaml_config(yaml_file_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration mapping."""

    path = Path(yaml_file_path or REPOSITORY_ROOT / "config.yaml").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass
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
        """Build the provider-neutral model configuration for this role."""

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


def _load_role_settings(
    section: Any, role: str, yaml_dir: Path
) -> RoleModelSettings:
    field_prefix = f"api.{role}"
    if not isinstance(section, dict):
        raise ValueError(f"{field_prefix} must be a mapping")
    unknown = set(section) - ROLE_FIELDS
    if unknown:
        raise ValueError(
            f"{field_prefix} has unknown fields: {', '.join(sorted(unknown))}"
        )
    missing = ROLE_FIELDS - set(section)
    if missing:
        raise ValueError(
            f"{field_prefix} has missing fields: {', '.join(sorted(missing))}"
        )

    protocol_name = _nonempty_string(section["protocol"], f"{field_prefix}.protocol")
    try:
        protocol = ModelProtocol(protocol_name)
    except ValueError as exc:
        raise ValueError(
            f"{field_prefix}.protocol is unsupported: {protocol_name}"
        ) from exc

    key_file = Path(_nonempty_string(section["key_file"], f"{field_prefix}.key_file"))
    if not key_file.is_absolute():
        key_file = yaml_dir / key_file
    key_file = key_file.resolve()

    base_url = section["base_url"]
    if base_url is not None:
        base_url = _nonempty_string(base_url, f"{field_prefix}.base_url")

    timeout = section["timeout"]
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"{field_prefix}.timeout must be a positive number or null")
        timeout = float(timeout)

    max_retries = section["max_retries"]
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError(f"{field_prefix}.max_retries must be a non-negative integer")

    max_output_tokens = section["max_output_tokens"]
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError(f"{field_prefix}.max_output_tokens must be a positive integer")

    reasoning_effort = section["reasoning_effort"]
    if reasoning_effort is not None:
        reasoning_effort = _nonempty_string(
            reasoning_effort, f"{field_prefix}.reasoning_effort"
        )

    temperature = section["temperature"]
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"{field_prefix}.temperature must be a number or null")
        temperature = float(temperature)

    settings = RoleModelSettings(
        protocol=protocol,
        model=_nonempty_string(section["model"], f"{field_prefix}.model"),
        key_file=key_file,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    load_api_key_from_file(key_file)
    return settings


def load_api_roles(
    yaml_file_path: str | Path | None = None,
) -> dict[str, RoleModelSettings]:
    """Load the exact four model roles under the top-level ``api`` mapping."""

    path = Path(yaml_file_path or REPOSITORY_ROOT / "config.yaml").resolve()
    payload = load_yaml_config(path)
    api = payload.get("api")
    if not isinstance(api, dict):
        raise ValueError("api must be a mapping")
    unknown = set(api) - set(API_ROLES)
    if unknown:
        raise ValueError(f"api has unknown roles: {', '.join(sorted(unknown))}")
    missing = set(API_ROLES) - set(api)
    if missing:
        raise ValueError(f"api has missing roles: {', '.join(sorted(missing))}")
    return {
        role: _load_role_settings(api[role], role, path.parent) for role in API_ROLES
    }


@dataclass
class ConfigSDK:
    """Runtime configuration with explicit model settings for every role."""

    primer: RoleModelSettings
    refine: RoleModelSettings
    extraction: RoleModelSettings
    agent: RoleModelSettings
    max_context_tokens: int = 128000
    memory_token_limit: int = 4000
    chunk_token_soft_limit: int = 60000
    pairs_per_chunk: Optional[int] = None
    use_stream: bool = True
    per_block_update: bool = True
    verbose: bool = False
    very_verbose: bool = False
    debug_prompts: bool = False
    stats_interval: float = 1.0
    dry_run: bool = False
    max_chunks: Optional[int] = None
    price_per_1k_prompt_tokens: float = 0.03
    price_per_1k_completion_tokens: float = 0.06
    glossary_max_entries: int = 100
    glossary_policy: str = "lock"
    user_prompt_path: str = "custom_main_prompt.md"
    terminology_min_confidence: float = 0.6
    intermediate_format: str = "json"

    def __post_init__(self) -> None:
        valid_formats = {"json", "xml-pair", "pseudo-toml"}
        if self.intermediate_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid intermediate format: {self.intermediate_format}. "
                f"Valid formats: {', '.join(sorted(valid_formats))}"
            )


def load_config_from_yaml(yaml_file_path: str | Path | None = None) -> ConfigSDK:
    """Load complete application configuration from YAML."""

    payload = load_yaml_config(yaml_file_path)
    deprecated_sections = {
        "main_model",
        "terminology_model",
        "translation_model",
    }.intersection(payload)
    if deprecated_sections:
        raise ValueError(
            "deprecated model sections are not supported: "
            + ", ".join(sorted(deprecated_sections))
        )
    roles = load_api_roles(yaml_file_path)
    token_settings = payload.get("tokens", {})
    chunking_settings = payload.get("chunking", {})
    pricing_settings = payload.get("pricing", {})
    glossary_settings = payload.get("glossary", {})
    user_settings = payload.get("user", {})
    runtime_settings = payload.get("runtime", {})
    if "api_mode" in runtime_settings:
        raise ValueError("runtime.api_mode is not supported; configure api.<role>.protocol")
    format_settings = payload.get("format", {})

    per_block_update = runtime_settings.get(
        "per_block_update", runtime_settings.get("incremental_output", True)
    )
    return ConfigSDK(
        primer=roles["primer"],
        refine=roles["refine"],
        extraction=roles["extraction"],
        agent=roles["agent"],
        max_context_tokens=token_settings.get("max_context_tokens", 128000),
        memory_token_limit=token_settings.get("memory_token_limit", 4000),
        chunk_token_soft_limit=token_settings.get("chunk_token_soft_limit", 60000),
        pairs_per_chunk=chunking_settings.get("pairs_per_chunk"),
        price_per_1k_prompt_tokens=pricing_settings.get("prompt_tokens", 0.03),
        price_per_1k_completion_tokens=pricing_settings.get("completion_tokens", 0.06),
        glossary_max_entries=glossary_settings.get("max_entries", 100),
        glossary_policy=glossary_settings.get("policy", "lock"),
        terminology_min_confidence=glossary_settings.get(
            "terminology_min_confidence", 0.6
        ),
        user_prompt_path=user_settings.get("prompt_path", "custom_main_prompt.md"),
        use_stream=runtime_settings.get(
            "use_stream", runtime_settings.get("use_streaming", True)
        ),
        per_block_update=per_block_update,
        verbose=runtime_settings.get("verbose", False),
        very_verbose=runtime_settings.get("very_verbose", False),
        debug_prompts=runtime_settings.get("debug_prompts", False),
        stats_interval=runtime_settings.get("stats_interval", 1.0),
        dry_run=runtime_settings.get("dry_run", False),
        max_chunks=runtime_settings.get("max_chunks"),
        intermediate_format=format_settings.get("intermediate_format", "json"),
    )


def load_config_sdk(
    yaml_file_path: str | Path | None = None,
    model_name: Optional[str] = None,
    use_stream: Optional[bool] = None,
    use_streaming: Optional[bool] = None,
    per_block_update: Optional[bool] = None,
    incremental_output: Optional[bool] = None,
    dry_run: bool = False,
    max_chunks: Optional[int] = None,
    memory_limit: Optional[int] = None,
    pairs_per_chunk: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    api_timeout: Optional[int] = None,
    verbose: bool = False,
    very_verbose: bool = False,
    debug_prompts: bool = False,
    stats_interval: Optional[float] = None,
    intermediate_format: Optional[str] = None,
) -> ConfigSDK:
    """Load YAML configuration and apply command-line runtime overrides."""

    config = load_config_from_yaml(yaml_file_path)
    if model_name:
        config.refine.model = model_name
    if use_stream is not None:
        config.use_stream = use_stream
    elif use_streaming is not None:
        config.use_stream = use_streaming
    if per_block_update is not None:
        config.per_block_update = per_block_update
    elif incremental_output is not None:
        config.per_block_update = incremental_output
    if dry_run:
        config.dry_run = True
    if max_chunks is not None:
        config.max_chunks = max_chunks
    if memory_limit is not None:
        config.memory_token_limit = memory_limit
    if pairs_per_chunk is not None:
        config.pairs_per_chunk = pairs_per_chunk
    if reasoning_effort is not None:
        config.refine.reasoning_effort = reasoning_effort
    if api_timeout is not None:
        config.refine.timeout = float(api_timeout)
    if verbose:
        config.verbose = True
    if very_verbose:
        config.very_verbose = True
        config.verbose = True
    if debug_prompts:
        config.debug_prompts = True
        config.very_verbose = True
        config.verbose = True
    if stats_interval is not None:
        config.stats_interval = stats_interval
    if intermediate_format is not None:
        config.intermediate_format = intermediate_format
    return config
