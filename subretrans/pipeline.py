"""LangGraph definition for the subtitle processing pipeline."""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .state import MemorylessTranslationState, PipelineState, Stage


STAGES: tuple[Stage, ...] = (
    "preprocess",
    "translate_parallel",
    "merge_ass",
    "refine_serial",
    "postprocess",
    "qa",
    "human_review",
    "release",
)

StateUpdate: TypeAlias = Mapping[str, Any]
StageHandler: TypeAlias = Callable[[PipelineState], StateUpdate]
TranslationHandler: TypeAlias = Callable[[MemorylessTranslationState], StateUpdate]


def build_pipeline(
    handlers: Mapping[Stage, StageHandler],
    *,
    checkpointer: Any,
) -> Any:
    """Compile the fixed pipeline with explicit handlers and persistence."""

    missing = [stage for stage in STAGES if stage not in handlers]
    if missing:
        raise ValueError(f"missing stage handlers: {', '.join(missing)}")

    graph = StateGraph(PipelineState)

    for stage in STAGES:
        if stage == "translate_parallel":
            graph.add_node(stage, _translation_node(cast(TranslationHandler, handlers[stage])))
        elif stage == "human_review":
            graph.add_node(stage, _human_review_node(handlers[stage]))
        else:
            graph.add_node(stage, _handler_node(stage, handlers[stage]))

    graph.add_edge(START, "preprocess")
    graph.add_conditional_edges(
        "preprocess",
        _route_translation_mode,
        {
            "translate_parallel": "translate_parallel",
            "refine_serial": "refine_serial",
        },
    )
    graph.add_edge("translate_parallel", "merge_ass")
    graph.add_edge("merge_ass", "refine_serial")
    graph.add_edge("refine_serial", "postprocess")
    graph.add_edge("postprocess", "qa")
    graph.add_edge("qa", "human_review")
    graph.add_edge("release", END)

    return graph.compile(checkpointer=checkpointer)


def _route_translation_mode(state: PipelineState) -> Stage:
    """Choose memoryless parallel translation or direct serial memory mode."""
    if state["translation_mode"] == "parallel_initial":
        return "translate_parallel"
    if state["translation_mode"] == "serial_memory":
        return "refine_serial"
    raise ValueError(f"unsupported translation mode: {state['translation_mode']}")


def _handler_node(stage: Stage, handler: StageHandler) -> StageHandler:
    def run(state: PipelineState) -> StateUpdate:
        current_state = cast(PipelineState, {**state, "stage": stage})
        return {**handler(current_state), "stage": stage}

    return run


def _translation_node(handler: TranslationHandler) -> StageHandler:
    """Run first-pass translation without exposing serial proofreading memory."""

    def run(state: PipelineState) -> StateUpdate:
        manifest_path = state["translation_manifest_path"]
        if manifest_path is None:
            raise ValueError("parallel translation requires translation_manifest_path")
        translation_state: MemorylessTranslationState = {
            "artifact_path": state["artifact_path"],
            "artifact_hash": state["artifact_hash"],
            "translation_manifest_path": manifest_path,
            "stage": "translate_parallel",
            "model_version": state["model_version"],
            "prompt_version": state["prompt_version"],
        }
        update = handler(translation_state)
        forbidden = {
            "refine_chunk_cursor",
            "memory_checkpoint_path",
            "memory_hash",
        }.intersection(update)
        if forbidden:
            raise ValueError(
                "parallel translation cannot update proofreading memory: "
                + ", ".join(sorted(forbidden))
            )
        return {**update, "stage": "translate_parallel"}

    return run


def _human_review_node(handler: StageHandler) -> Callable[[PipelineState], Command]:
    def run(state: PipelineState) -> Command:
        current_state = cast(PipelineState, {**state, "stage": "human_review"})
        decision = interrupt(
            {
                "stage": "human_review",
                "artifact_path": current_state["artifact_path"],
                "artifact_hash": current_state["artifact_hash"],
                "qa_conclusion": current_state["qa_conclusion"],
            }
        )
        update = {**handler(current_state), "stage": "human_review"}

        if decision == "approve":
            return Command(update=update, goto="release")
        if decision == "reject":
            return Command(update=update, goto=END)
        raise ValueError("human review decision must be 'approve' or 'reject'")

    return run
