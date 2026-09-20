"""LangGraph definition for the manifest-backed subtitle pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .run_manifest import load_manifest, reconcile_state
from .state import PipelineState, Stage, validate_pipeline_state


STAGES: tuple[Stage, ...] = (
    "preprocess",
    "translate_parallel",
    "merge_ass",
    "freeze_manifest",
    "refine_serial",
    "postprocess",
    "glossary",
    "qa",
    "repair",
    "qa_verify",
    "review_export",
    "human_review",
    "release",
)

StageHandler: TypeAlias = Callable[[PipelineState], PipelineState]
HumanReviewHandler: TypeAlias = Callable[[PipelineState, str], PipelineState]


def build_pipeline(
    handlers: Mapping[Stage, Callable[..., PipelineState]],
    *,
    checkpointer: Any,
) -> Any:
    """Compile the fixed graph; every node consumes and returns exact state."""

    missing = [stage for stage in STAGES if stage not in handlers]
    if missing:
        raise ValueError(f"missing stage handlers: {', '.join(missing)}")

    graph = StateGraph(PipelineState)
    for stage in STAGES:
        if stage == "human_review":
            graph.add_node(stage, _human_review_node(cast(HumanReviewHandler, handlers[stage])))
        else:
            graph.add_node(stage, _handler_node(stage, cast(StageHandler, handlers[stage])))

    graph.add_edge(START, "preprocess")
    graph.add_conditional_edges(
        "preprocess",
        _route_after_preprocess,
        {"translate_parallel": "translate_parallel", "freeze_manifest": "freeze_manifest"},
    )
    graph.add_edge("translate_parallel", "merge_ass")
    graph.add_edge("merge_ass", "freeze_manifest")
    graph.add_edge("freeze_manifest", "refine_serial")
    graph.add_edge("refine_serial", "postprocess")
    graph.add_edge("postprocess", "glossary")
    graph.add_edge("glossary", "qa")
    # Repair is mandatory even when QA returns an empty suggestion pool.
    graph.add_edge("qa", "repair")
    graph.add_edge("repair", "qa_verify")
    graph.add_conditional_edges(
        "qa_verify",
        _route_after_verify,
        {"repair": "repair", "review_export": "review_export"},
    )
    graph.add_edge("review_export", "human_review")
    graph.add_edge("release", END)
    return graph.compile(checkpointer=checkpointer)


def _authoritative(state: PipelineState) -> PipelineState:
    checked, _ = reconcile_state(validate_pipeline_state(state))
    return checked


def _route_after_preprocess(state: PipelineState) -> Stage:
    checked = _authoritative(state)
    if checked["next_stage"] in {"translate_parallel", "freeze_manifest"}:
        return cast(Stage, checked["next_stage"])
    raise ValueError(f"preprocess produced invalid route: {checked['next_stage']}")


def _route_after_verify(state: PipelineState) -> Stage:
    checked = _authoritative(state)
    if checked["next_stage"] in {"repair", "review_export"}:
        return cast(Stage, checked["next_stage"])
    raise ValueError(f"qa_verify produced invalid route: {checked['next_stage']}")


def _handler_node(stage: Stage, handler: StageHandler) -> StageHandler:
    def run(state: PipelineState) -> PipelineState:
        checked = validate_pipeline_state(state, location=f"{stage} input state")
        reconciled, advanced = reconcile_state(checked)
        if advanced:
            manifest = load_manifest(reconciled["manifest_path"])
            if manifest["completed_stage"] != stage:
                raise ValueError(
                    f"manifest advanced at {manifest['completed_stage']}, not replayed stage {stage}"
                )
            return validate_pipeline_state(reconciled, location=f"{stage} replay state")
        if reconciled["next_stage"] != stage:
            raise ValueError(f"pipeline expected {reconciled['next_stage']}, not {stage}")
        return validate_pipeline_state(handler(reconciled), location=f"{stage} output state")

    return run


def _human_review_node(handler: HumanReviewHandler) -> Callable[[PipelineState], Command]:
    def run(state: PipelineState) -> Command:
        checked = validate_pipeline_state(state, location="human_review input state")
        reconciled, advanced = reconcile_state(checked)
        if advanced:
            manifest = load_manifest(reconciled["manifest_path"], verify_artifacts=False)
            if manifest["completed_stage"] != "human_review":
                raise ValueError("manifest advanced outside human_review")
            goto = "release" if reconciled["next_stage"] == "release" else END
            return Command(update=reconciled, goto=goto)
        manifest = load_manifest(reconciled["manifest_path"], verify_artifacts=False)
        decision = interrupt(
            {
                "stage": "human_review",
                "review_path": manifest["review"]["path"],
                "review_hash": manifest["review"]["sha256"],
                "route_reason": manifest["route_reason"],
            }
        )
        if decision not in {"approve", "reject"}:
            raise ValueError("human review decision must be 'approve' or 'reject'")
        updated = validate_pipeline_state(
            handler(reconciled, decision), location="human_review output state"
        )
        return Command(update=updated, goto="release" if decision == "approve" else END)

    return run
