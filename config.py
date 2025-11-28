"""
Configuration module for subtitle refinement tool.

Defines the Config class with all necessary parameters for the tool,
including API settings, token limits, and model configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MainModelSettings:
    """Primary refinement model configuration."""

    name: str = "gpt-5.1"
    max_output_tokens: int = 12000
    reasoning_effort: str = "medium"
    temperature: float = 1.0  # GPT-5.* fixed temperature internally

@dataclass
class TerminologyModelSettings:
    """Dedicated terminology extractor model configuration."""

    name: str = "gpt-4o-mini"
    max_output_tokens: int = 1500
    temperature: float = 0.3


@dataclass
class Config:
    """
    Configuration class for subtitle refinement tool.

    Sections:
        main_model: Settings for the primary GPT-5.* refinement model
        terminology_model: Settings for the GPT-4o terminology extractor

    Shared attributes govern chunking, API access, and telemetry behaviour.
    """

    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    max_context_tokens: int = 128000
    memory_token_limit: int = 4000
    chunk_token_soft_limit: int = 60000  # ~0.8 * max_context_tokens
    pairs_per_chunk: Optional[int] = None  # If set, override token-based chunking
    api_timeout: int = 240  # API request timeout in seconds
    verbose: bool = False  # Enable verbose output with detailed progress
    very_verbose: bool = False  # Dump full API responses (requires verbose)
    debug_prompts: bool = False  # Print system prompt/memory (requires very verbose)
    stats_interval: float = 1.0  # Stats refresh interval in seconds (for verbose mode)
    dry_run: bool = False
    max_chunks: Optional[int] = None
    price_per_1k_prompt_tokens: float = 0.03
    price_per_1k_completion_tokens: float = 0.06
    glossary_max_entries: int = 100
    glossary_policy: str = "lock"  # lock: user glossary is authoritative; learned terms cannot override
    user_prompt_path: str = "custom_main_prompt.md"  # Relative or absolute path to extra user prompt
    terminology_min_confidence: float = 0.6  # Minimum confidence for extracted terminology entries
    main_model: MainModelSettings = field(default_factory=MainModelSettings)
    terminology_model: TerminologyModelSettings = field(default_factory=TerminologyModelSettings)

    def __post_init__(self):
        """Validate and set default values from environment variables if needed."""
        # Try to get API key from environment if not set
        if not self.api_key:
            self.api_key = os.getenv(
                "OPENAI_API_KEY",
                #"***REMOVED***"
                "***REMOVED***"
            )

        if not self.api_key:
            raise ValueError("API key must be provided either in config or OPENAI_API_KEY environment variable")

    @property
    def model_name(self) -> str:
        """Backward-compatible alias for the main model name."""
        return self.main_model.name

    @property
    def terminology_model_name(self) -> str:
        """Backward-compatible alias for the terminology model name."""
        return self.terminology_model.name


def load_config(
    model_name: Optional[str] = None,
    terminology_model: Optional[str] = None,
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
    glossary_limit: Optional[int] = None,
    glossary_policy: Optional[str] = None,
    user_prompt_path: Optional[str] = None,
    terminology_min_confidence: Optional[float] = None
) -> Config:
    """
    Load configuration with optional overrides.

    Args:
        model_name: Override for main refinement model name
        terminology_model: Override for terminology extraction model name
        dry_run: Enable dry run mode
        max_chunks: Override for maximum chunks to process
        memory_limit: Override for memory token limit
        pairs_per_chunk: Override for pairs per chunk
        reasoning_effort: Override for reasoning effort hint
        api_timeout: Override for API request timeout
        verbose: Enable verbose mode
        very_verbose: Enable very verbose mode (full API outputs)
        debug_prompts: Print system prompt/memory for debugging
        stats_interval: Stats refresh interval in seconds
        glossary_limit: Maximum number of terminology entries to keep in memory
        glossary_policy: Glossary merge policy ("lock" currently supported)
        user_prompt_path: Path to extra user prompt file (default: "custom_main_prompt.md")
        terminology_min_confidence: Minimum confidence for terminology entries (both prompt and filter)

    Returns:
        Config object with specified settings
    """
    config = Config()

    if model_name:
        config.main_model.name = model_name
    if terminology_model:
        config.terminology_model.name = terminology_model
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
        config.verbose = True  # Very verbose implies verbose
    if debug_prompts:
        config.debug_prompts = debug_prompts
        config.very_verbose = True
        config.verbose = True
    if stats_interval is not None:
        config.stats_interval = stats_interval
    if glossary_limit is not None:
        config.glossary_max_entries = glossary_limit
    if glossary_policy is not None:
        config.glossary_policy = glossary_policy
    if user_prompt_path is not None:
        config.user_prompt_path = user_prompt_path
    if terminology_min_confidence is not None:
        config.terminology_min_confidence = terminology_min_confidence

    return config
