"""
Configuration module for OpenAI SDK-based subtitle refinement.

Loads configuration from YAML file and provides configuration compatible with
the main project's structure.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_api_key_from_file(key_file_path: str = None) -> str:
    """
    Load API key from key file.

    Args:
        key_file_path: Path to key file (defaults to ../key relative to this file)

    Returns:
        API key string

    Raises:
        FileNotFoundError: If key file doesn't exist
        ValueError: If key file is empty or invalid
    """
    if key_file_path is None:
        # Default to key file in parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(os.path.dirname(current_dir), "key")

    if not os.path.exists(key_file_path):
        raise FileNotFoundError(f"Key file not found: {key_file_path}")

    with open(key_file_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError("Key file is empty")

    return api_key


def load_yaml_config(yaml_file_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        yaml_file_path: Path to YAML config file (defaults to config.yaml in this directory)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML file is invalid
    """
    if yaml_file_path is None:
        # Default to config.yaml in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_path = os.path.join(current_dir, "config.yaml")

    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"Config file not found: {yaml_file_path}")

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data


@dataclass
class MainModelSettings:
    """Primary refinement model configuration."""

    name: str = "gpt-5-mini"
    max_output_tokens: int = 12000
    reasoning_effort: str = "medium"
    temperature: float = 1.0
    key_file: Optional[str] = None  # Optional per-model API key file override
    base_url: Optional[str] = None  # Optional per-model base URL override


@dataclass
class TerminologyModelSettings:
    """Dedicated terminology extractor model configuration."""

    name: str = "gpt-4o-mini"
    max_output_tokens: int = 1800
    temperature: float = 0.5
    key_file: Optional[str] = None  # Optional per-model API key file override
    base_url: Optional[str] = None  # Optional per-model base URL override


@dataclass
class ConfigSDK:
    """
    Configuration class for OpenAI SDK-based subtitle refinement.

    Compatible with the main project's Config structure but uses
    OpenAI SDK instead of direct HTTP requests.
    """

    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    max_context_tokens: int = 128000
    memory_token_limit: int = 4000
    chunk_token_soft_limit: int = 60000
    pairs_per_chunk: Optional[int] = None
    api_timeout: int = 280
    use_streaming: bool = True
    incremental_output: bool = True
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
    main_model: MainModelSettings = field(default_factory=MainModelSettings)
    terminology_model: TerminologyModelSettings = field(default_factory=TerminologyModelSettings)
    intermediate_format: str = "json"

    def __post_init__(self):
        """Load API key from key file if not set and validate format."""
        if not self.api_key:
            try:
                self.api_key = load_api_key_from_file()
                print(f"Loaded API key from key file: {self.api_key[:20]}...")
            except Exception as e:
                raise ValueError(f"Failed to load API key from key file: {str(e)}")

        valid_formats = ["json", "xml-pair", "pseudo-toml"]
        if self.intermediate_format.lower() not in valid_formats:
            raise ValueError(f"Invalid intermediate format: {self.intermediate_format}. "
                           f"Valid formats: {', '.join(valid_formats)}")

    @property
    def model_name(self) -> str:
        """Backward-compatible alias for the main model name."""
        return self.main_model.name

    @property
    def terminology_model_name(self) -> str:
        """Backward-compatible alias for the terminology model name."""
        return self.terminology_model.name


def load_config_from_yaml(yaml_file_path: str = None) -> ConfigSDK:
    """
    Load configuration from YAML file.

    Args:
        yaml_file_path: Path to YAML config file (defaults to config.yaml in this directory)

    Returns:
        ConfigSDK object with settings from YAML file
    """
    config_data = load_yaml_config(yaml_file_path)

    # Extract values from YAML structure
    api_settings = config_data.get("api", {})
    main_model_settings = config_data.get("main_model", {})
    terminology_model_settings = config_data.get("terminology_model", {})
    token_settings = config_data.get("tokens", {})
    chunking_settings = config_data.get("chunking", {})
    pricing_settings = config_data.get("pricing", {})
    glossary_settings = config_data.get("glossary", {})
    user_settings = config_data.get("user", {})
    runtime_settings = config_data.get("runtime", {})
    format_settings = config_data.get("format", {})

    # Resolve key file path relative to YAML config file
    key_file_path = api_settings.get("key_file")
    if key_file_path:
        # Determine YAML directory
        if yaml_file_path:
            yaml_dir = os.path.dirname(os.path.abspath(yaml_file_path))
        else:
            # Default location: experiment directory
            yaml_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve key file path relative to YAML directory
        key_file_path = os.path.join(yaml_dir, key_file_path)
        api_key = load_api_key_from_file(key_file_path)
    else:
        api_key = load_api_key_from_file()  # Use default location

    # Create model settings
    main_model = MainModelSettings(
        name=main_model_settings.get("name", "gpt-5-mini"),
        max_output_tokens=main_model_settings.get("max_output_tokens", 12000),
        reasoning_effort=main_model_settings.get("reasoning_effort", "medium"),
        temperature=main_model_settings.get("temperature", 1.0),
        key_file=main_model_settings.get("key_file"),
        base_url=main_model_settings.get("base_url"),
    )

    terminology_model = TerminologyModelSettings(
        name=terminology_model_settings.get("name", "gpt-4o-mini"),
        max_output_tokens=terminology_model_settings.get("max_output_tokens", 1800),
        temperature=terminology_model_settings.get("temperature", 0.5),
        key_file=terminology_model_settings.get("key_file"),
        base_url=terminology_model_settings.get("base_url"),
    )

    # Create config object
    config = ConfigSDK(
        api_key=api_key,
        api_base_url=api_settings.get("base_url", "https://api.openai.com/v1"),
        api_timeout=api_settings.get("timeout", 280),
        max_context_tokens=token_settings.get("max_context_tokens", 128000),
        memory_token_limit=token_settings.get("memory_token_limit", 4000),
        chunk_token_soft_limit=token_settings.get("chunk_token_soft_limit", 60000),
        pairs_per_chunk=chunking_settings.get("pairs_per_chunk"),
        price_per_1k_prompt_tokens=pricing_settings.get("prompt_tokens", 0.03),
        price_per_1k_completion_tokens=pricing_settings.get("completion_tokens", 0.06),
        glossary_max_entries=glossary_settings.get("max_entries", 100),
        glossary_policy=glossary_settings.get("policy", "lock"),
        terminology_min_confidence=glossary_settings.get("terminology_min_confidence", 0.6),
        user_prompt_path=user_settings.get("prompt_path", "custom_main_prompt.md"),
        use_streaming=runtime_settings.get("use_streaming", True),
        incremental_output=runtime_settings.get("incremental_output", True),
        verbose=runtime_settings.get("verbose", False),
        very_verbose=runtime_settings.get("very_verbose", False),
        debug_prompts=runtime_settings.get("debug_prompts", False),
        stats_interval=runtime_settings.get("stats_interval", 1.0),
        dry_run=runtime_settings.get("dry_run", False),
        max_chunks=runtime_settings.get("max_chunks"),
        main_model=main_model,
        terminology_model=terminology_model,
        intermediate_format=format_settings.get("intermediate_format", "json"),
    )

    # Skip __post_init__ API key loading since we already loaded it
    config.api_key = api_key

    return config


def load_config_sdk(
    yaml_file_path: str = None,
    model_name: Optional[str] = None,
    terminology_model: Optional[str] = None,
    use_streaming: Optional[bool] = None,
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
    """
    Load configuration from YAML file with optional overrides.

    Args:
        yaml_file_path: Path to YAML config file (defaults to config.yaml in this directory)
        model_name: Override for main refinement model name
        terminology_model: Override for terminology extraction model name
        use_streaming: Override for streaming API mode
        incremental_output: Override for incremental output mode (write file after each chunk)
        dry_run: Enable dry run mode
        max_chunks: Override for maximum chunks to process
        memory_limit: Override for memory token limit
        pairs_per_chunk: Override for pairs per chunk
        reasoning_effort: Override for reasoning effort hint
        api_timeout: Override for API request timeout
        verbose: Enable verbose mode
        very_verbose: Enable very verbose mode
        debug_prompts: Print system prompt for debugging
        stats_interval: Stats refresh interval in seconds
        intermediate_format: Override for intermediate representation format (json, xml-pair, pseudo-toml)

    Returns:
        ConfigSDK object with specified settings
    """
    # Load from YAML first
    config = load_config_from_yaml(yaml_file_path)

    # Apply CLI overrides
    if model_name:
        config.main_model.name = model_name
    if terminology_model:
        config.terminology_model.name = terminology_model
    if use_streaming is not None:
        config.use_streaming = use_streaming
    if incremental_output is not None:
        config.incremental_output = incremental_output
    if dry_run:
        config.dry_run = dry_run
    if max_chunks is not None:
        config.max_chunks = max_chunks
    if memory_limit is not None:
        config.memory_token_limit = memory_limit
    if pairs_per_chunk is not None:
        config.pairs_per_chunk = pairs_per_chunk
    if reasoning_effort is not None:
        config.main_model.reasoning_effort = reasoning_effort
    if api_timeout is not None:
        config.api_timeout = api_timeout
    if verbose:
        config.verbose = verbose
    if very_verbose:
        config.very_verbose = very_verbose
        config.verbose = True
    if debug_prompts:
        config.debug_prompts = debug_prompts
        config.very_verbose = True
        config.verbose = True
    if stats_interval is not None:
        config.stats_interval = stats_interval
    if intermediate_format is not None:
        config.intermediate_format = intermediate_format

    return config
