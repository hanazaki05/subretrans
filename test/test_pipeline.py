from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from subretrans.pipeline import STAGES, build_pipeline
from subretrans.run_manifest import commit_manifest, create_manifest, load_manifest, mutable_manifest
from subretrans.state import PipelineState, Stage


def initial_state(tmp_path: Path, mode: str = "parallel_initial") -> PipelineState:
    source = tmp_path / ("source.mkv" if mode == "parallel_initial" else "source.ass")
    source.write_text("source", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    return create_manifest(
        tmp_path / "run.json",
        run_id=tmp_path.name,
        translation_mode=mode,  # type: ignore[arg-type]
        source_path=source,
        config_path=config,
        prompt_paths={name: config for name in ("shared", "refine", "qa", "repair")},
        release_path=tmp_path / "release.ass",
        budget_limits={
            "repair_attempts": 2,
            "tool_steps": 4,
            "full_sweeps": 1,
            "glossary_repairs": 0,
            "research_requests": 0,
        },
    )


def recording_handlers(calls: list[str], *, verify_repairs: int = 0):
    verify_calls = 0

    def handler(stage: Stage):
        def run(state: PipelineState) -> PipelineState:
            nonlocal verify_calls
            calls.append(stage)
            checked, manifest = mutable_manifest(state, expected_stage=stage)
            routes = {
                "translate_parallel": "merge_ass",
                "merge_ass": "freeze_manifest",
                "freeze_manifest": "refine_serial",
                "refine_serial": "postprocess",
                "postprocess": "glossary",
                "glossary": "qa",
                "qa": "repair",
                "repair": "qa_verify",
                "review_export": "human_review",
                "release": "end",
            }
            if stage == "preprocess":
                next_stage = (
                    "translate_parallel"
                    if manifest["translation_mode"] == "parallel_initial"
                    else "freeze_manifest"
                )
            elif stage == "qa_verify":
                verify_calls += 1
                next_stage = "repair" if verify_calls <= verify_repairs else "review_export"
            else:
                next_stage = routes[stage]
            return commit_manifest(
                checked,
                stage=stage,
                next_stage=next_stage,  # type: ignore[arg-type]
                route_reason=f"{stage}_complete",
                manifest=manifest,
            )

        return run

    handlers = {stage: handler(stage) for stage in STAGES if stage != "human_review"}

    def human_review(state: PipelineState, decision: str) -> PipelineState:
        calls.append("human_review")
        checked, manifest = mutable_manifest(state, expected_stage="human_review")
        return commit_manifest(
            checked,
            stage="human_review",
            next_stage="release" if decision == "approve" else "end",
            route_reason=f"human_{decision}",
            manifest=manifest,
        )

    handlers["human_review"] = human_review
    return handlers


def test_parallel_pipeline_runs_complete_second_batch_graph(tmp_path: Path) -> None:
    calls: list[str] = []
    graph = build_pipeline(recording_handlers(calls), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "parallel"}}

    result = graph.invoke(initial_state(tmp_path), config)
    assert result["__interrupt__"]
    graph.invoke(Command(resume="approve"), config)

    assert calls == list(STAGES)


def test_serial_mode_skips_primer_and_merge(tmp_path: Path) -> None:
    calls: list[str] = []
    graph = build_pipeline(recording_handlers(calls), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "serial"}}

    result = graph.invoke(initial_state(tmp_path, "serial_memory"), config)
    assert result["__interrupt__"]
    graph.invoke(Command(resume="reject"), config)

    assert "translate_parallel" not in calls
    assert "merge_ass" not in calls
    assert calls[:3] == ["preprocess", "freeze_manifest", "refine_serial"]
    assert "release" not in calls


def test_repair_runs_when_initial_qa_is_empty_and_verify_can_loop(tmp_path: Path) -> None:
    calls: list[str] = []
    graph = build_pipeline(
        recording_handlers(calls, verify_repairs=1), checkpointer=InMemorySaver()
    )
    result = graph.invoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "repair-loop"}}
    )

    assert result["__interrupt__"]
    assert calls.count("qa") == 1
    assert calls.count("repair") == 2
    assert calls.count("qa_verify") == 2


def test_missing_handler_fails_when_building_pipeline() -> None:
    handlers = {stage: (lambda state: state) for stage in STAGES}
    del handlers["qa"]
    with pytest.raises(ValueError, match="missing stage handlers: qa"):
        build_pipeline(handlers, checkpointer=InMemorySaver())


def test_manifest_stage_is_authoritative_after_completion(tmp_path: Path) -> None:
    calls: list[str] = []
    graph = build_pipeline(recording_handlers(calls), checkpointer=InMemorySaver())
    state = initial_state(tmp_path)
    graph.invoke(state, {"configurable": {"thread_id": "authority"}})

    manifest = load_manifest(state["manifest_path"], verify_artifacts=False)
    assert manifest["next_stage"] == "human_review"
    assert manifest["completed_stage"] == "review_export"
