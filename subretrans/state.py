"""Persistent state for the subtitle processing graph."""

from typing import Literal, TypedDict


TranslationMode = Literal["parallel_initial", "serial_memory"]


Stage = Literal[
    "preprocess",
    "translate_parallel",
    "merge_ass",
    "refine_serial",
    "postprocess",
    "qa",
    "human_review",
    "release",
]


class PipelineState(TypedDict):
    """Small, checkpoint-safe references to pipeline artifacts and progress."""

    artifact_path: str
    artifact_hash: str
    translation_manifest_path: str | None
    translation_mode: TranslationMode
    stage: Stage
    refine_chunk_cursor: int
    memory_checkpoint_path: str | None
    memory_hash: str
    model_versions: dict[str, str]
    prompt_version: str
    qa_conclusion: str
    qa_passed: bool
    qa_repair_applied: bool
    agent_repair_attempts: int


class MemorylessTranslationState(TypedDict):
    """Inputs visible to the parallel first-pass translation stage."""

    artifact_path: str
    artifact_hash: str
    translation_manifest_path: str
    stage: Literal["translate_parallel"]
    model_version: str
    prompt_version: str
