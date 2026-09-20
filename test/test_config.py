from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from subretrans.config import (
    API_ROLES,
    PipelineSettings,
    PostprocessSettings,
    PrimerSettings,
    PromptPaths,
    QASettings,
    RefineSettings,
    RoleModelSettings,
    load_config,
)
from subretrans.providers import ModelProtocol
from subretrans.subtitle_edit import SubtitleEditSettings


REVISION = "7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9"


def role_yaml(**overrides: object) -> str:
    fields: dict[str, object] = {
        "protocol": "anthropic-messages",
        "model": "claude-test",
        "key_file": "secrets/provider.key",
        "base_url": "https://gateway.example",
        "timeout": 45,
        "max_retries": 2,
        "max_output_tokens": 2048,
        "reasoning_effort": "null",
        "temperature": 0.2,
    }
    fields.update(overrides)
    return "\n".join(f"    {name}: {value}" for name, value in fields.items())


def config_yaml(
    *,
    role_overrides: dict[str, object] | None = None,
    primer_batch_size: str = "8",
    refine_batch_size: str = "5",
    source_language: str = "English",
    representation: str = "xml-pair",
    confidence: str = "0.6",
    first_pass_operations: str = "[--fix-common-errors, --remove-text-for-hi]",
) -> str:
    roles = "\n".join(f"  {role}:\n{role_yaml(**(role_overrides or {}))}" for role in API_ROLES)
    return f"""api:
{roles}
pipeline:
  state_dir: runtime/state
  checkpoint_db: runtime/checkpoints.sqlite
  agent_max_repair_attempts: 2
prompts:
  shared_path: prompts/shared.md
  refine_path: prompts/refine.md
  qa_path: prompts/qa.md
primer:
  batch_size: {primer_batch_size}
  max_workers: 3
  source_language: {source_language}
  target_language: Simplified Chinese
  user_instruction: Preserve speaker tone.
refine:
  batch_size: {refine_batch_size}
  chunk_token_soft_limit: 80000
  memory_token_limit: 4000
  intermediate_representation: {representation}
qa:
  batch_size: 40
  max_workers: 3
  window_offsets: [0, 20]
postprocess:
  operations:
    - normalize_style_names
    - episode_replacements
  episode_replacements:
    - {{from: old, to: new}}
subtitle_edit:
  repository_url: https://github.com/SubtitleEdit/subtitleedit.git
  revision: {REVISION}
  source_dir: tools/subtitleedit
  build_dir: tools/seconv
  dotnet_executable: dotnet
  settings_file: subtitle-edit-settings.json
  multiple_replace_file: multiple-replace.template
  first_pass_operations: {first_pass_operations}
  second_pass_operations: [--fix-common-errors-rules:FixUnneededSpaces]
glossary:
  max_entries: 100
  terminology_min_confidence: {confidence}
"""


def write_config(tmp_path: Path, content: str, *, key: str = "  secret-value\n") -> Path:
    key_path = tmp_path / "secrets/provider.key"
    key_path.parent.mkdir(exist_ok=True)
    key_path.write_text(key, encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_validates_every_section_once(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, config_yaml()))

    assert config.path == (tmp_path / "config.yaml").resolve()
    assert config.api.primer == RoleModelSettings(
        protocol=ModelProtocol.ANTHROPIC_MESSAGES,
        model="claude-test",
        key_file=(tmp_path / "secrets/provider.key").resolve(),
        base_url="https://gateway.example",
        timeout=45.0,
        max_retries=2,
        max_output_tokens=2048,
        reasoning_effort=None,
        temperature=0.2,
    )
    assert config.api["agent"] == config.api.agent
    assert [role for role, _ in config.api] == list(API_ROLES)
    assert config.model_versions == {role: "claude-test" for role in API_ROLES}
    assert config.api.refine.config.api_key == "secret-value"
    assert config.pipeline == PipelineSettings(
        state_dir=(tmp_path / "runtime/state").resolve(),
        checkpoint_db=(tmp_path / "runtime/checkpoints.sqlite").resolve(),
        agent_max_repair_attempts=2,
    )
    assert config.prompts == PromptPaths(
        shared=(tmp_path / "prompts/shared.md").resolve(),
        refine=(tmp_path / "prompts/refine.md").resolve(),
        qa=(tmp_path / "prompts/qa.md").resolve(),
    )
    assert config.primer == PrimerSettings(8, 3, "English", "Simplified Chinese", "Preserve speaker tone.")
    assert config.refine == RefineSettings(5, 80000, 4000, "xml-pair")
    assert config.qa == QASettings(40, 3, (0, 20))
    assert config.postprocess == PostprocessSettings(
        ("normalize_style_names", "episode_replacements"), (("old", "new"),)
    )
    assert config.subtitle_edit == SubtitleEditSettings(
        repository_url="https://github.com/SubtitleEdit/subtitleedit.git",
        revision=REVISION,
        source_dir=(tmp_path / "tools/subtitleedit").resolve(),
        build_dir=(tmp_path / "tools/seconv").resolve(),
        dotnet_executable="dotnet",
        settings_file=(tmp_path / "subtitle-edit-settings.json").resolve(),
        multiple_replace_file=(tmp_path / "multiple-replace.template").resolve(),
        first_pass_operations=("--fix-common-errors", "--remove-text-for-hi"),
        second_pass_operations=("--fix-common-errors-rules:FixUnneededSpaces",),
    )
    assert config.glossary.max_entries == 100
    assert config.glossary.terminology_min_confidence == 0.6
    with pytest.raises(FrozenInstanceError):
        config.primer.batch_size = 4  # type: ignore[misc]
    with pytest.raises(KeyError, match="unsupported API role"):
        config.api["nope"]


def test_load_config_accepts_absolute_paths_and_nulls(tmp_path: Path) -> None:
    absolute_state = tmp_path / "absolute-state"
    content = (
        config_yaml(refine_batch_size="null")
        .replace("  state_dir: runtime/state", f"  state_dir: {absolute_state}")
        .replace("  user_instruction: Preserve speaker tone.", "  user_instruction: null")
        .replace(
            "  operations:\n    - normalize_style_names\n    - episode_replacements\n",
            "  operations: []\n",
        )
        .replace("  episode_replacements:\n    - {from: old, to: new}\n", "  episode_replacements: []\n")
    )

    config = load_config(write_config(tmp_path, content))

    assert config.pipeline.state_dir == absolute_state.resolve()
    assert config.refine.batch_size is None
    assert config.primer.user_instruction is None
    assert config.postprocess == PostprocessSettings((), ())


@pytest.mark.parametrize(
    "content, match",
    [
        (config_yaml(primer_batch_size="0"), "primer.batch_size must be a positive integer"),
        (config_yaml(refine_batch_size="true"), "refine.batch_size must be a positive integer"),
        (config_yaml(representation="csv"), "intermediate_representation must be one of"),
        (
            config_yaml().replace("qa:\n  batch_size: 40", "qa:\n  batch_size: 0"),
            "qa.batch_size must be a positive integer",
        ),
        (
            config_yaml().replace("  window_offsets: [0, 20]", "  window_offsets: [20]"),
            "qa.window_offsets must start with 0",
        ),
        (config_yaml(source_language="''"), "source_language must be a non-empty"),
        (
            config_yaml().replace("  agent_max_repair_attempts: 2", "  agent_max_repair_attempts: -1"),
            "agent_max_repair_attempts must be a non-negative integer",
        ),
        (
            config_yaml().replace("  max_workers: 3\n  source", "  max_workers: 3\n  extra: true\n  source"),
            "primer has unknown fields: extra",
        ),
        (
            config_yaml().replace("    - normalize_style_names\n", "    - unknown\n"),
            r"postprocess.operations\[0\] is unsupported: unknown",
        ),
        (config_yaml(role_overrides={"protocol": "unknown-wire-format"}), "protocol is unsupported"),
        (config_yaml(role_overrides={"max_retries": -1}), "max_retries must be a non-negative integer"),
        (config_yaml(role_overrides={"timeout": 0}), "timeout must be a positive number"),
        (config_yaml(role_overrides={"base_url": "''"}), "base_url must be a non-empty string"),
        (config_yaml(role_overrides={"extra": "unsupported"}), "api.primer has unknown fields: extra"),
        (config_yaml(confidence="1.5"), "terminology_min_confidence must be a number from 0 to 1"),
        (config_yaml(first_pass_operations="[]"), "first_pass_operations must be a non-empty list"),
        (config_yaml(first_pass_operations="['']"), r"first_pass_operations\[0\] must be a non-empty string"),
        (config_yaml() + "legacy_section: {}\n", "unknown sections: legacy_section"),
        (config_yaml().replace("glossary:\n  max_entries: 100\n  terminology_min_confidence: 0.6\n", ""), "missing sections: glossary"),
    ],
)
def test_load_config_rejects_invalid_config(tmp_path: Path, content: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(write_config(tmp_path, content))


def test_load_config_rejects_empty_key_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="key file must not be empty"):
        load_config(write_config(tmp_path, config_yaml(), key=" \n"))
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "absent.yaml")
