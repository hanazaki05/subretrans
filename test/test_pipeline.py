from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from subretrans.pipeline import STAGES, StageHandler, build_pipeline
from subretrans.state import (
    MemorylessTranslationState,
    PipelineState,
    Stage,
    TranslationMode,
)


def initial_state(
    translation_mode: TranslationMode = "parallel_initial",
) -> PipelineState:
    return {
        "artifact_path": "/tmp/input.ass",
        "artifact_hash": "input-hash",
        "translation_manifest_path": (
            "/tmp/translations.json"
            if translation_mode == "parallel_initial"
            else None
        ),
        "translation_mode": translation_mode,
        "stage": "preprocess",
        "refine_chunk_cursor": 0,
        "memory_checkpoint_path": "/tmp/input.memory.yaml",
        "memory_hash": "memory-hash",
        "model_version": "test-model-v1",
        "prompt_version": "test-prompt-v1",
        "qa_conclusion": "pending",
    }


def recording_handlers(calls: list[str]) -> dict[Stage, StageHandler]:
    def make_handler(stage: Stage) -> Callable[[PipelineState], dict[str, Any]]:
        def handler(state: PipelineState) -> dict[str, Any]:
            calls.append(stage)
            assert state["stage"] == stage
            if stage == "qa":
                return {"qa_conclusion": "passed"}
            return {}

        return handler

    return {stage: make_handler(stage) for stage in STAGES}


def test_parallel_translation_cannot_see_proofreading_memory() -> None:
    seen_state: MemorylessTranslationState | None = None
    handlers = recording_handlers([])

    def translate(state: MemorylessTranslationState) -> dict[str, Any]:
        nonlocal seen_state
        seen_state = state
        return {}

    handlers["translate_parallel"] = translate  # type: ignore[assignment]
    pipeline = build_pipeline(handlers, checkpointer=InMemorySaver())
    pipeline.invoke(
        initial_state(), {"configurable": {"thread_id": "memoryless-translate"}}
    )

    assert seen_state is not None
    assert "memory_checkpoint_path" not in seen_state
    assert "memory_hash" not in seen_state
    assert "refine_chunk_cursor" not in seen_state


def run_until_review(calls: list[str], thread_id: str):
    pipeline = build_pipeline(
        recording_handlers(calls), checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke(initial_state(), config)
    assert result["__interrupt__"]
    return pipeline, config


def test_pipeline_runs_stages_in_fixed_order() -> None:
    calls: list[str] = []
    pipeline, config = run_until_review(calls, "fixed-order")

    pipeline.invoke(Command(resume="approve"), config)

    assert calls == list(STAGES)


def test_serial_memory_mode_skips_parallel_translation_and_merge() -> None:
    calls: list[str] = []
    pipeline = build_pipeline(
        recording_handlers(calls), checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "serial-memory"}}

    result = pipeline.invoke(initial_state("serial_memory"), config)
    assert result["__interrupt__"]
    pipeline.invoke(Command(resume="approve"), config)

    assert calls == [
        "preprocess",
        "refine_serial",
        "postprocess",
        "qa",
        "human_review",
        "release",
    ]


def test_release_runs_only_after_explicit_approval() -> None:
    calls: list[str] = []
    pipeline, config = run_until_review(calls, "approve")

    result = pipeline.invoke(Command(resume="approve"), config)

    assert calls[-2:] == ["human_review", "release"]
    assert result["stage"] == "release"


def test_rejection_terminates_without_release() -> None:
    calls: list[str] = []
    pipeline, config = run_until_review(calls, "reject")

    result = pipeline.invoke(Command(resume="reject"), config)

    assert calls[-1] == "human_review"
    assert "release" not in calls
    assert result["stage"] == "human_review"


def test_missing_handler_fails_when_building_pipeline() -> None:
    handlers = recording_handlers([])
    del handlers["qa"]

    with pytest.raises(ValueError, match="missing stage handlers: qa"):
        build_pipeline(handlers, checkpointer=InMemorySaver())
