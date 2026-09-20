from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from subretrans.config import RoleModelSettings
from subretrans.providers import ModelProtocol
from subretrans.workflow_config import (
    PipelineSettings,
    load_pipeline_settings,
    load_role_model_settings,
    load_subtitle_edit_settings,
)


def write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def pipeline_yaml(*, batch_size: str = "8", source_language: str = "English") -> str:
    return f"""pipeline:
  state_dir: runtime/state
  checkpoint_db: runtime/checkpoints.sqlite
  batch_size: {batch_size}
  max_workers: 3
  agent_max_repair_attempts: 2
  source_language: {source_language}
  target_language: Simplified Chinese
  user_instruction: Preserve speaker tone.
  episode_replacements:
    - {{from: old, to: new}}
"""


def api_yaml(**overrides: object) -> str:
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
    rendered = "\n".join(f"      {name}: {value}" for name, value in fields.items())
    return "api:\n" + "\n".join(
        f"  {role}:\n{rendered}" for role in ("primer", "refine", "extraction", "agent")
    ) + "\n"


def test_pipeline_loader_resolves_paths_without_reading_translation_key(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "workflow.yaml", pipeline_yaml())

    settings = load_pipeline_settings(config_path)

    assert settings == PipelineSettings(
        state_dir=(tmp_path / "runtime/state").resolve(),
        checkpoint_db=(tmp_path / "runtime/checkpoints.sqlite").resolve(),
        batch_size=8,
        max_workers=3,
        agent_max_repair_attempts=2,
        source_language="English",
        target_language="Simplified Chinese",
        user_instruction="Preserve speaker tone.",
        episode_replacements=(("old", "new"),),
    )
    assert not (tmp_path / "missing-key").exists()
    with pytest.raises(FrozenInstanceError):
        settings.batch_size = 4  # type: ignore[misc]


def test_pipeline_loader_accepts_absolute_paths_and_optional_instruction(
    tmp_path: Path,
) -> None:
    absolute_state = tmp_path / "absolute-state"
    config_path = write_config(
        tmp_path / "workflow.yaml",
        f"""pipeline:
  state_dir: {absolute_state}
  checkpoint_db: checkpoint.sqlite
  batch_size: 1
  max_workers: 1
  agent_max_repair_attempts: 0
  source_language: English
  target_language: Chinese
  episode_replacements: []
""",
    )

    settings = load_pipeline_settings(config_path)

    assert settings.state_dir == absolute_state.resolve()
    assert settings.checkpoint_db == (tmp_path / "checkpoint.sqlite").resolve()
    assert settings.user_instruction is None
    assert settings.episode_replacements == ()


@pytest.mark.parametrize(
    "content, match",
    [
        (pipeline_yaml(batch_size="0"), "batch_size must be a positive integer"),
        (pipeline_yaml(batch_size="true"), "batch_size must be a positive integer"),
        (pipeline_yaml(source_language="''"), "source_language must be a non-empty"),
        (
            pipeline_yaml().replace("  agent_max_repair_attempts: 2", "  agent_max_repair_attempts: -1"),
            "agent_max_repair_attempts must be a non-negative integer",
        ),
        (
            pipeline_yaml() .replace("  max_workers: 3\n", "  max_workers: 3\n  extra: true\n"),
            "unknown fields: extra",
        ),
    ],
)
def test_pipeline_loader_rejects_invalid_config(
    tmp_path: Path, content: str, match: str
) -> None:
    config_path = write_config(tmp_path / "workflow.yaml", content)

    with pytest.raises(ValueError, match=match):
        load_pipeline_settings(config_path)


def test_role_model_loader_uses_protocol_and_resolves_relative_key(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "secrets/provider.key"
    key_path.parent.mkdir()
    key_path.write_text("  secret-value\n", encoding="utf-8")
    config_path = write_config(tmp_path / "workflow.yaml", api_yaml())

    settings = load_role_model_settings(config_path, "primer")

    assert settings == RoleModelSettings(
        protocol=ModelProtocol.ANTHROPIC_MESSAGES,
        model="claude-test",
        key_file=key_path.resolve(),
        base_url="https://gateway.example",
        timeout=45.0,
        max_retries=2,
        max_output_tokens=2048,
        reasoning_effort=None,
        temperature=0.2,
    )
    assert settings.config.api_key == "secret-value"


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"protocol": "unknown-wire-format"}, "protocol is unsupported"),
        ({"max_retries": -1}, "max_retries must be a non-negative integer"),
        ({"timeout": 0}, "timeout must be a positive number"),
        ({"base_url": "''"}, "base_url must be a non-empty string"),
    ],
)
def test_role_model_loader_rejects_invalid_config(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    key_path = tmp_path / "secrets/provider.key"
    key_path.parent.mkdir()
    key_path.write_text("secret-value", encoding="utf-8")
    config_path = write_config(
        tmp_path / "workflow.yaml", api_yaml(**overrides)
    )

    with pytest.raises(ValueError, match=match):
        load_role_model_settings(config_path, "primer")


def test_role_model_loader_rejects_empty_key_and_unknown_field(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "secrets/provider.key"
    key_path.parent.mkdir()
    key_path.write_text(" \n", encoding="utf-8")
    empty_key_config = write_config(tmp_path / "empty.yaml", api_yaml())

    with pytest.raises(ValueError, match="key file must not be empty"):
        load_role_model_settings(empty_key_config, "primer")

    unknown_config = write_config(
        tmp_path / "unknown.yaml",
        api_yaml(extra="unsupported"),
    )
    with pytest.raises(ValueError, match="unknown fields: extra"):
        load_role_model_settings(unknown_config, "primer")


@patch("subretrans.workflow_config.SubtitleEditSettings")
def test_subtitle_edit_loader_resolves_paths_and_builds_strict_settings(
    settings_type, tmp_path: Path
) -> None:
    config_path = write_config(
        tmp_path / "workflow.yaml",
        """subtitle_edit:
  repository_url: https://github.com/SubtitleEdit/subtitleedit.git
  revision: 7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9
  source_dir: tools/subtitleedit
  build_dir: tools/seconv
  dotnet_executable: dotnet
  settings_file: subtitle-edit-settings.json
  multiple_replace_file: multiple-replace.template
  operations:
    - --fix-common-errors
    - --remove-text-for-hi
""",
    )

    result = load_subtitle_edit_settings(config_path)

    assert result is settings_type.return_value
    settings_type.assert_called_once_with(
        repository_url="https://github.com/SubtitleEdit/subtitleedit.git",
        revision="7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9",
        source_dir=(tmp_path / "tools/subtitleedit").resolve(),
        build_dir=(tmp_path / "tools/seconv").resolve(),
        dotnet_executable="dotnet",
        settings_file=(tmp_path / "subtitle-edit-settings.json").resolve(),
        multiple_replace_file=(tmp_path / "multiple-replace.template").resolve(),
        operations=("--fix-common-errors", "--remove-text-for-hi"),
    )


@pytest.mark.parametrize(
    "operations, match",
    [
        ("[]", "operations must be a non-empty list"),
        ("--fix-common-errors", "operations must be a non-empty list"),
        ("['']", r"operations\[0\] must be a non-empty string"),
    ],
)
def test_subtitle_edit_loader_rejects_invalid_operations(
    tmp_path: Path, operations: str, match: str
) -> None:
    config_path = write_config(
        tmp_path / "workflow.yaml",
        f"""subtitle_edit:
  repository_url: https://github.com/SubtitleEdit/subtitleedit.git
  revision: 7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9
  source_dir: tools/subtitleedit
  build_dir: tools/seconv
  dotnet_executable: dotnet
  settings_file: subtitle-edit-settings.json
  multiple_replace_file: multiple-replace.template
  operations: {operations}
""",
    )

    with pytest.raises(ValueError, match=match):
        load_subtitle_edit_settings(config_path)


def test_subtitle_edit_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "workflow.yaml",
        """subtitle_edit:
  repository_url: https://github.com/SubtitleEdit/subtitleedit.git
  revision: 7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9
  source_dir: tools/subtitleedit
  build_dir: tools/seconv
  dotnet_executable: dotnet
  settings_file: subtitle-edit-settings.json
  multiple_replace_file: multiple-replace.template
  operations: [--fix-common-errors]
  extra: unsupported
""",
    )

    with pytest.raises(ValueError, match="unknown fields: extra"):
        load_subtitle_edit_settings(config_path)
