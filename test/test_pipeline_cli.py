import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from subretrans.config import PromptPaths
from subretrans.fsutil import sha256_file
from subretrans.model_agent import AgentQAResult
from subretrans.pipeline_cli import (
    _prompt_version,
    _refine_callable,
    _validate_thread_id,
    build_parser,
    resume_pipeline,
    review_pipeline,
    run_pipeline,
    status_pipeline,
)
from subretrans.refine import RefineProgress


VALID_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,你好
"""

MEMORY_YAML = "user_glossary: []\nglossary: []\nstory_description: Test episode context.\n"


def write_config(tmp_path: Path) -> Path:
    key = tmp_path / "key"
    key.write_text("test-key\n", encoding="utf-8")
    for name, body in (
        ("shared.md", "Shared subtitle rules.\n"),
        ("refine.md", "Refine task.\n"),
        ("qa.md", "QA task.\n"),
    ):
        (tmp_path / name).write_text(body, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""api:
  primer: &model
    protocol: openai-responses
    model: test-model
    key_file: {key}
    base_url: https://example.test/v1
    timeout: 30
    max_retries: 0
    max_output_tokens: 1000
    reasoning_effort: null
    temperature: null
  refine: *model
  extraction: *model
  agent: *model
pipeline:
  state_dir: {tmp_path / 'state'}
  checkpoint_db: {tmp_path / 'state/checkpoints.sqlite3'}
  agent_max_repair_attempts: 1
prompts:
  shared_path: {tmp_path / 'shared.md'}
  refine_path: {tmp_path / 'refine.md'}
  qa_path: {tmp_path / 'qa.md'}
primer:
  batch_size: 2
  max_workers: 2
  source_language: English
  target_language: Simplified Chinese
  user_instruction: null
refine:
  batch_size: 1
  chunk_token_soft_limit: 80000
  memory_token_limit: 4000
  intermediate_representation: xml-pair
qa:
  batch_size: 10
  max_workers: 2
  window_offsets: [0, 5]
postprocess:
  operations:
    - clean_chinese_dialogue
    - normalize_punctuation
    - normalize_style_names
    - normalize_event_fields
    - normalize_italics
    - episode_replacements
  episode_replacements: []
subtitle_edit:
  repository_url: https://github.com/SubtitleEdit/subtitleedit.git
  revision: 7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9
  source_dir: {tmp_path / 'tools/subtitleedit'}
  build_dir: {tmp_path / 'tools/seconv'}
  dotnet_executable: dotnet
  settings_file: {tmp_path / 'subtitle-edit-settings.json'}
  multiple_replace_file: {tmp_path / 'multiple-replace.template'}
  first_pass_operations: [--fix-common-errors]
  second_pass_operations: [--fix-common-errors]
glossary:
  max_entries: 100
  terminology_min_confidence: 0.6
""",
        encoding="utf-8",
    )
    return config


def write_progress(progress_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    progress_path.write_text(
        json.dumps(
            {
                "version": 1,
                "next_pair": 1,
                "artifact_path": str(output_path),
                "artifact_hash": sha256_file(output_path),
                "memory_checkpoint_path": str(checkpoint_path),
                "memory_hash": sha256_file(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )


def test_prompt_version_includes_all_prompt_components(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    shared = tmp_path / "shared.md"
    refine = tmp_path / "refine.md"
    qa = tmp_path / "qa.md"
    for path, body in ((config_path, "config"), (shared, "shared"), (refine, "refine"), (qa, "qa")):
        path.write_text(body, encoding="utf-8")
    config = SimpleNamespace(
        path=config_path, prompts=PromptPaths(shared=shared, refine=refine, qa=qa)
    )

    original = _prompt_version(config)
    qa.write_text("updated qa", encoding="utf-8")

    assert _prompt_version(config) != original


@pytest.mark.parametrize("thread_id", ["", ".hidden", "a/b", "..", "ep 1"])
def test_rejects_unsafe_thread_ids(thread_id: str) -> None:
    with pytest.raises(ValueError):
        _validate_thread_id(thread_id)


def test_parser_exposes_all_subcommands() -> None:
    parser = build_parser()

    run = parser.parse_args(["run", "in.ass", "out.ass", "--mode", "serial_memory", "--thread-id", "e1"])
    assert run.func is run_pipeline
    assert Path(run.config).name == "config.yaml"
    assert parser.parse_args(["status", "e1", "--debug"]).func is status_pipeline
    assert parser.parse_args(["review", "e1", "reject"]).func is review_pipeline
    assert parser.parse_args(["resume", "e1"]).func is resume_pipeline


def test_serial_pipeline_cli_resumes_failure_and_releases_review(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    release = tmp_path / "release.ass"
    config = write_config(tmp_path)
    refine_attempts = 0
    seen_configs = []

    def fake_refine_factory(app_config):
        seen_configs.append(app_config)

        def refine(input_path, output_path, checkpoint_path, progress_path):
            nonlocal refine_attempts
            refine_attempts += 1
            if refine_attempts == 1:
                raise RuntimeError("transient refine failure")
            output_path.write_bytes(input_path.read_bytes())
            checkpoint_path.write_text(MEMORY_YAML, encoding="utf-8")
            write_progress(progress_path, output_path, checkpoint_path)

        return refine

    monkeypatch.setattr("subretrans.pipeline_cli._refine_callable", fake_refine_factory)
    monkeypatch.setattr(
        "subretrans.pipeline_cli.build_agent_qa",
        lambda settings, system_prompt: (
            lambda pairs, structural_qa, repair_history, episode_memory: AgentQAResult(True, (), ())
        ),
    )
    run_args = argparse.Namespace(
        input=str(source),
        output=str(release),
        mode="serial_memory",
        thread_id="episode-01",
        config=str(config),
    )
    review_args = argparse.Namespace(thread_id="episode-01", decision="approve", config=str(config))
    resume_args = argparse.Namespace(thread_id="episode-01", config=str(config))
    status_args = argparse.Namespace(thread_id="episode-01", config=str(config))

    with pytest.raises(RuntimeError, match="transient refine failure"):
        run_pipeline(run_args)
    assert status_pipeline(status_args) == 0
    assert "stopped before refine_serial; run `pipeline resume" in capsys.readouterr().out
    assert resume_pipeline(resume_args) == 0
    assert refine_attempts == 2
    assert all(seen.path == config.resolve() for seen in seen_configs)
    assert "is awaiting human review" in capsys.readouterr().out
    assert status_pipeline(status_args) == 0
    assert "Status: awaiting human review" in capsys.readouterr().out
    review = tmp_path / "release.review.ass"
    assert review.exists()
    assert not release.exists()
    review.write_text(
        review.read_text(encoding="utf-8-sig").replace("你好", "人工修改"),
        encoding="utf-8-sig",
    )
    assert review_pipeline(review_args) == 0
    assert release.exists()
    assert "人工修改" in release.read_text(encoding="utf-8-sig")
    assert status_pipeline(status_args) == 0
    assert "Status: released" in capsys.readouterr().out
    metadata = json.loads((tmp_path / "state/episode-01/run.json").read_text(encoding="utf-8"))
    assert metadata == {
        "version": 2,
        "config_path": str(config.resolve()),
        "review_path": str(review),
        "release_path": str(release),
        "translation_mode": "serial_memory",
    }


def test_review_rejects_mismatched_config(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    run_dir = tmp_path / "state/episode-02"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "version": 2,
                "config_path": str(tmp_path / "other.yaml"),
                "review_path": str(tmp_path / "r.review.ass"),
                "release_path": str(tmp_path / "r.ass"),
                "translation_mode": "serial_memory",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="review config does not match the run config"):
        review_pipeline(argparse.Namespace(thread_id="episode-02", decision="reject", config=str(config)))


def test_refine_callable_resumes_from_committed_progress(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "original.ass"
    output = tmp_path / "refined.ass"
    memory = tmp_path / "memory.yaml"
    progress = tmp_path / "progress.json"
    output.write_text(VALID_ASS, encoding="utf-8")
    memory.write_text(MEMORY_YAML, encoding="utf-8")
    write_progress(progress, output, memory)
    config = object()
    calls: list[dict[str, object]] = []
    loaded: list[tuple[Path, Path, Path]] = []

    def fake_load_refine_progress(progress_path, output_path, checkpoint_path):
        loaded.append((progress_path, output_path, checkpoint_path))
        return RefineProgress(3, output_path, "hash", checkpoint_path, "hash")

    def fake_refine_serial(input_path, output_path, config_value, **kwargs):
        calls.append({"input": input_path, "output": output_path, "config": config_value, **kwargs})

    monkeypatch.setattr("subretrans.pipeline_cli.load_refine_progress", fake_load_refine_progress)
    monkeypatch.setattr("subretrans.pipeline_cli.refine_serial", fake_refine_serial)
    refine = _refine_callable(config)

    refine(original, output, memory, progress)
    progress.unlink()
    refine(original, output, memory, progress)

    assert loaded == [(progress, output, memory)]
    assert [call["input"] for call in calls] == [original, original]
    assert all(call["output"] == output and call["config"] is config for call in calls)
    assert all(
        call["checkpoint_path"] == memory and call["progress_path"] == progress for call in calls
    )
    assert calls[0]["options"].resume_index == 3
    assert calls[1]["options"].resume_index is None
