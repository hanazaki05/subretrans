"""Minimal checkpoint state for the manifest-backed pipeline."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

from .fsutil import require_exact_fields


TranslationMode = Literal["parallel_initial", "serial_memory"]

Stage = Literal[
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
]
NextStage = Stage | Literal["end"]

PIPELINE_STATE_FIELDS = {
    "manifest_path",
    "manifest_hash",
    "next_stage",
    "route_reason",
}


class PipelineState(TypedDict):
    """Only the authoritative manifest identity and routing cursor is checkpointed."""

    manifest_path: str
    manifest_hash: str
    next_stage: NextStage
    route_reason: str


# The primer sees the same deliberately small state. Its manifest revision is
# created before refine memory exists, so proofreading memory cannot leak in.
MemorylessTranslationState = PipelineState


def validate_pipeline_state(value: object, *, location: str = "pipeline state") -> PipelineState:
    """Validate the exact runtime state shape; ``TypedDict`` alone is not enough."""

    payload = require_exact_fields(value, PIPELINE_STATE_FIELDS, location=location)
    manifest_path = payload["manifest_path"]
    manifest_hash = payload["manifest_hash"]
    next_stage = payload["next_stage"]
    route_reason = payload["route_reason"]
    if type(manifest_path) is not str or not manifest_path:
        raise ValueError(f"{location}.manifest_path must be a non-empty string")
    if (
        type(manifest_hash) is not str
        or len(manifest_hash) != 64
        or any(character not in "0123456789abcdef" for character in manifest_hash)
    ):
        raise ValueError(f"{location}.manifest_hash must be a lowercase SHA-256 digest")
    allowed_stages = {
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
        "end",
    }
    if next_stage not in allowed_stages:
        raise ValueError(f"{location}.next_stage is unsupported: {next_stage!r}")
    if type(route_reason) is not str or not route_reason:
        raise ValueError(f"{location}.route_reason must be a non-empty string")
    return cast(PipelineState, dict(payload))
