import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from subretrans.config import PromptPaths, RepairSettings
from subretrans.fsutil import sha256_file
from subretrans.pipeline_cli import (
    _existing_run,
    _prompt_version,
    _refine_callable,
    _repair_callable,
    _validate_thread_id,
    build_parser,
    run_pipeline,
)
from subretrans.refine import RefineProgress
from subretrans.run_manifest import RUN_MANIFEST_VERSION, create_manifest, load_manifest


VALID_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,你好
"""

MEMORY_YAML = "user_glossary: []\nglossary: []\nstory_description: Test episode context.\n"


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
    paths = {
        name: tmp_path / f"{name}.md" for name in ("shared", "refine", "qa", "repair")
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    config = SimpleNamespace(path=config_path, prompts=PromptPaths(**paths))

    original = _prompt_version(config)
    paths["repair"].write_text("updated repair", encoding="utf-8")

    assert _prompt_version(config) != original


@pytest.mark.parametrize("thread_id", ["", ".hidden", "a/b", "..", "ep 1"])
def test_rejects_unsafe_thread_ids(thread_id: str) -> None:
    with pytest.raises(ValueError):
        _validate_thread_id(thread_id)


def test_parser_exposes_all_subcommands() -> None:
    parser = build_parser()

    run = parser.parse_args(
        ["run", "in.ass", "out.ass", "--mode", "serial_memory", "--thread-id", "e1"]
    )
    assert run.func is run_pipeline
    assert Path(run.config).name == "config.yaml"
    assert parser.parse_args(["status", "e1", "--debug"]).command == "status"
    assert parser.parse_args(["review", "e1", "reject"]).command == "review"
    assert parser.parse_args(["resume", "e1"]).command == "resume"


def test_run_creates_v3_manifest_with_run_local_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test config", encoding="utf-8")
    state_dir = tmp_path / "state"
    release = tmp_path / "published" / "episode.ass"
    config = SimpleNamespace(
        path=config_path.resolve(),
        prompts=SimpleNamespace(
            shared=config_path,
            refine=config_path,
            qa=config_path,
            repair=config_path,
        ),
        pipeline=SimpleNamespace(
            state_dir=state_dir,
            checkpoint_db=state_dir / "checkpoints.sqlite3",
        ),
        repair=SimpleNamespace(
            max_repair_attempts=2,
            max_tool_steps=10,
            max_full_sweeps=1,
            max_glossary_repair_attempts=1,
        ),
        research=SimpleNamespace(max_requests=3),
    )

    class FakeGraph:
        def invoke(self, state, graph_config):
            assert state["next_stage"] == "preprocess"
            assert graph_config == {"configurable": {"thread_id": "episode-01"}}
            return {"__interrupt__": ("review",)}

    @contextmanager
    def fake_open_graph(context):
        yield FakeGraph()

    monkeypatch.setattr("subretrans.pipeline_cli.load_config", lambda path: config)
    monkeypatch.setattr("subretrans.pipeline_cli._open_graph", fake_open_graph)
    monkeypatch.setattr("subretrans.pipeline_cli._prompt_version", lambda config: "test-prompt")

    result = run_pipeline(
        argparse.Namespace(
            input=str(source),
            output=str(release),
            mode="serial_memory",
            thread_id="episode-01",
            config=str(config_path),
        )
    )

    manifest_path = state_dir / "episode-01" / "run.json"
    manifest = load_manifest(manifest_path, verify_artifacts=False)
    assert result == 0
    assert manifest["version"] == RUN_MANIFEST_VERSION
    assert Path(manifest["review"]["path"]).is_relative_to(manifest_path.parent)
    assert manifest["release"]["path"] == str(release.resolve())
    assert not release.parent.exists()


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
        call["checkpoint_path"] == memory and call["progress_path"] == progress
        for call in calls
    )
    assert calls[0]["options"].resume_index == 3
    assert calls[1]["options"].resume_index is None


def test_existing_run_rejects_changed_prompt_content(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    prompt_paths = {
        name: tmp_path / f"{name}.md" for name in ("shared", "refine", "qa", "repair")
    }
    for name, path in prompt_paths.items():
        path.write_text(name, encoding="utf-8")
    state_dir = tmp_path / "state"
    run_dir = state_dir / "episode"
    run_dir.mkdir(parents=True)
    create_manifest(
        run_dir / "run.json",
        run_id="episode",
        translation_mode="serial_memory",
        source_path=source,
        config_path=config_path,
        prompt_paths=prompt_paths,
        release_path=tmp_path / "release.ass",
        budget_limits={},
    )
    config = SimpleNamespace(
        path=config_path.resolve(),
        prompts=SimpleNamespace(**prompt_paths),
        pipeline=SimpleNamespace(state_dir=state_dir),
    )
    monkeypatch.setattr("subretrans.pipeline_cli.load_config", lambda path: config)
    prompt_paths["qa"].write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="qa prompt changed"):
        _existing_run(
            argparse.Namespace(thread_id="episode", config=str(config_path)), "resume"
        )


def test_repair_callable_resumes_one_run_level_session_across_qa_rounds(
    tmp_path: Path, monkeypatch
) -> None:
    from subretrans.repair import RepairOutcome

    shared = tmp_path / "shared.md"
    repair_prompt = tmp_path / "repair.md"
    shared.write_text("shared", encoding="utf-8")
    repair_prompt.write_text("repair", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    refined = run_dir / "refined.ass"
    current = run_dir / "current.ass"
    refined.write_text(VALID_ASS, encoding="utf-8")
    current.write_text(VALID_ASS, encoding="utf-8")
    (run_dir / "memory.yaml").write_text(MEMORY_YAML, encoding="utf-8")
    cue = run_dir / "cue.json"
    cue.write_text("{}", encoding="utf-8")
    glossary = run_dir / "glossary.json"
    glossary.write_text('{"authoritative": [], "learned": []}', encoding="utf-8")
    pool = run_dir / "pool.json"
    pool.write_text('{"suggestions": []}', encoding="utf-8")
    config = SimpleNamespace(
        prompts=SimpleNamespace(shared=shared, repair=repair_prompt),
        repair=RepairSettings(8, 1, 2, 1, 3, 0),
        postprocess=SimpleNamespace(operations=(), episode_replacements=()),
        api=SimpleNamespace(repair=object()),
    )
    seen_history_lengths: list[int] = []

    def fake_run_repair_agent(*, session, **kwargs):
        seen_history_lengths.append(len(session.history))
        session.history.append({"kind": "round", "index": len(seen_history_lengths)})
        session.tool_steps_used += 1
        session.repair_attempts_used += 1
        session._commit()
        return RepairOutcome(
            "finish",
            session.ordered_current,
            session.current_hash,
            session.state_dir,
            session.tool_steps_used,
            session.repair_attempts_used,
            None,
        )

    monkeypatch.setattr("subretrans.repair.run_repair_agent", fake_run_repair_agent)
    run_repair = _repair_callable(config)

    results = []
    for attempt in (1, 2):
        results.append(
            run_repair(
                run_dir=run_dir,
                refined_artifact_path=refined,
                current_artifact_path=current,
                cue_manifest_path=cue,
                effective_glossary_path=glossary,
                suggestion_pool_path=pool,
                decision_log_path=run_dir / "repair" / f"decisions-{attempt:03d}.json",
                coverage_path=run_dir / "repair" / f"coverage-{attempt:03d}.json",
                max_tool_steps=8,
                max_full_sweeps=1,
                max_group_span=3,
            )
        )

    assert seen_history_lengths == [0, 1]
    assert [result["tool_steps_used"] for result in results] == [1, 1]
    assert results[0]["history_path"] != results[1]["history_path"]
    assert Path(results[0]["history_path"]).is_file()
