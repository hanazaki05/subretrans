"""Concrete filesystem handlers for the subtitle processing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .subtitle_processing import SrtCue, audit_ass, merge_srt_to_ass, postprocess_ass, read_srt, write_srt
from .state import MemorylessTranslationState, PipelineState, Stage
from .translation import (
    MANIFEST_VERSION,
    TranslateBatch,
    TranslationManifest,
    TranslationUnit,
    load_manifest,
    save_manifest,
    translate_manifest,
)


StageUpdate = Mapping[str, Any]
PreprocessSubtitle = Callable[[Path, Path], Path]
Refine = Callable[[Path, Path, Path, Path], None]


@dataclass(frozen=True)
class WorkflowSettings:
    run_dir: Path
    release_path: Path
    batch_size: int
    max_workers: int
    episode_replacements: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_extension(path: Path, extension: str) -> None:
    if path.suffix.lower() != extension:
        raise ValueError(f"expected a {extension} artifact: {path}")


def _require_distinct(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"input and output paths must differ: {input_path}")


def _atomic_copy(source: Path, destination: Path) -> None:
    _require_distinct(source, destination)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            fd, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_refine_progress(
    progress_path: Path, output_path: Path, checkpoint_path: Path
) -> tuple[int, str, str]:
    with progress_path.open(encoding="utf-8") as handle:
        value = json.load(handle)

    expected_fields = {
        "version",
        "next_pair",
        "artifact_path",
        "artifact_hash",
        "memory_checkpoint_path",
        "memory_hash",
    }
    if type(value) is not dict:
        raise ValueError("refine progress must be a JSON object")
    fields = set(value)
    if fields != expected_fields:
        missing = sorted(expected_fields - fields)
        unknown = sorted(fields - expected_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"refine progress has invalid fields ({'; '.join(details)})")

    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("refine progress version must be 1")
    next_pair = value["next_pair"]
    if type(next_pair) is not int or next_pair < 0:
        raise ValueError("refine progress next_pair must be a non-negative integer")
    if value["artifact_path"] != str(output_path):
        raise ValueError("refine progress artifact_path does not match refined output")
    if value["memory_checkpoint_path"] != str(checkpoint_path):
        raise ValueError(
            "refine progress memory_checkpoint_path does not match memory output"
        )

    artifact_hash = _sha256(output_path)
    memory_hash = _sha256(checkpoint_path)
    if type(value["artifact_hash"]) is not str or value["artifact_hash"] != artifact_hash:
        raise ValueError("refine progress artifact_hash does not match refined output")
    if type(value["memory_hash"]) is not str or value["memory_hash"] != memory_hash:
        raise ValueError("refine progress memory_hash does not match memory output")
    return next_pair, artifact_hash, memory_hash


def _qa_conclusion(path: Path) -> str:
    audit = audit_ass(path)
    if audit.passed:
        return "passed"
    return (
        "failed: "
        f"english_events={audit.english_events}, "
        f"chinese_events={audit.chinese_events}, "
        f"empty_english_events={audit.empty_english_events}, "
        f"empty_chinese_events={audit.empty_chinese_events}, "
        f"unpaired_timestamps={audit.unpaired_timestamps}, "
        f"non_monotonic_events={audit.non_monotonic_events}, "
        f"parse_errors={audit.parse_errors}"
    )


def build_stage_handlers(
    settings: WorkflowSettings,
    *,
    preprocess_subtitle: PreprocessSubtitle,
    translate_batch: TranslateBatch,
    refine: Refine,
) -> dict[Stage, Callable[[Any], StageUpdate]]:
    """Build all concrete handlers for one run directory and release target."""

    run_dir = Path(settings.run_dir)
    release_path = Path(settings.release_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(state: PipelineState) -> StageUpdate:
        source_path = Path(state["artifact_path"])
        if state["translation_mode"] == "parallel_initial":
            preprocessed_path = run_dir / "preprocessed.en.srt"
            _require_distinct(source_path, preprocessed_path)
            produced_path = preprocess_subtitle(source_path, preprocessed_path)
            if produced_path != preprocessed_path:
                raise ValueError(
                    "subtitle preprocessor must return run_dir/preprocessed.en.srt"
                )
            cues = read_srt(preprocessed_path)
            translated_path = run_dir / "translated.zh.srt"
            _require_distinct(preprocessed_path, translated_path)
            manifest_path = run_dir / "translation.json"
            save_manifest(
                TranslationManifest(
                    version=MANIFEST_VERSION,
                    source_artifact_path=str(preprocessed_path),
                    translated_artifact_path=str(translated_path),
                    units=[
                        TranslationUnit(
                            id=cue.index, source=cue.text, translation=None
                        )
                        for cue in cues
                    ],
                ),
                manifest_path,
            )
            return {"translation_manifest_path": str(manifest_path)}

        if state["translation_mode"] == "serial_memory":
            _require_extension(source_path, ".ass")
            audit = audit_ass(source_path)
            if audit.parse_errors:
                raise ValueError(
                    f"ASS artifact is not parseable: parse_errors={audit.parse_errors}"
                )
            return {"translation_manifest_path": None}

        raise ValueError(f"unsupported translation mode: {state['translation_mode']}")

    def translate_parallel(state: MemorylessTranslationState) -> StageUpdate:
        manifest_path = Path(state["translation_manifest_path"])
        manifest_run_dir = manifest_path.parent
        translated_path = manifest_run_dir / "translated.zh.srt"
        manifest = translate_manifest(
            manifest_path,
            translate_batch,
            settings.batch_size,
            settings.max_workers,
        )
        if manifest.translated_artifact_path != str(translated_path):
            raise ValueError(
                "manifest translated_artifact_path must be translated.zh.srt beside the manifest"
            )

        source_path = Path(manifest.source_artifact_path)
        _require_distinct(source_path, translated_path)
        source_cues = read_srt(source_path)
        if len(source_cues) != len(manifest.units):
            raise ValueError("manifest units do not match the source SRT cues")

        translated_cues: list[SrtCue] = []
        for position, (cue, unit) in enumerate(
            zip(source_cues, manifest.units, strict=True), start=1
        ):
            if unit.id != cue.index or unit.source != cue.text:
                raise ValueError(
                    f"manifest unit {position} does not match the source SRT cue"
                )
            if unit.translation is None:
                raise ValueError(f"manifest unit {unit.id} has no translation")
            translated_cues.append(
                SrtCue(cue.index, cue.start, cue.end, unit.translation)
            )
        write_srt(translated_cues, translated_path)
        return {}

    def merge_ass(state: PipelineState) -> StageUpdate:
        manifest_value = state["translation_manifest_path"]
        if manifest_value is None:
            raise ValueError("merge_ass requires translation_manifest_path")
        manifest = load_manifest(manifest_value)
        output_path = run_dir / "merged.ass"
        source_path = Path(manifest.source_artifact_path)
        translated_path = Path(manifest.translated_artifact_path)
        _require_distinct(source_path, output_path)
        _require_distinct(translated_path, output_path)
        merge_srt_to_ass(source_path, translated_path, output_path)
        return {
            "artifact_path": str(output_path),
            "artifact_hash": _sha256(output_path),
        }

    def refine_serial(state: PipelineState) -> StageUpdate:
        input_path = Path(state["artifact_path"])
        output_path = run_dir / "refined.ass"
        checkpoint_path = run_dir / "memory.yaml"
        progress_path = run_dir / "refine-progress.json"
        _require_distinct(input_path, output_path)
        result = refine(input_path, output_path, checkpoint_path, progress_path)
        if result is not None:
            raise ValueError("refine must return None")
        next_pair, artifact_hash, memory_hash = _load_refine_progress(
            progress_path, output_path, checkpoint_path
        )
        return {
            "artifact_path": str(output_path),
            "artifact_hash": artifact_hash,
            "memory_checkpoint_path": str(checkpoint_path),
            "memory_hash": memory_hash,
            "refine_chunk_cursor": next_pair,
        }

    def postprocess(state: PipelineState) -> StageUpdate:
        input_path = Path(state["artifact_path"])
        output_path = run_dir / "postprocessed.ass"
        _require_distinct(input_path, output_path)
        postprocess_ass(input_path, output_path, settings.episode_replacements)
        return {
            "artifact_path": str(output_path),
            "artifact_hash": _sha256(output_path),
        }

    def qa(state: PipelineState) -> StageUpdate:
        return {"qa_conclusion": _qa_conclusion(Path(state["artifact_path"]))}

    def human_review(state: PipelineState) -> StageUpdate:
        return {}

    def release(state: PipelineState) -> StageUpdate:
        source_path = Path(state["artifact_path"])
        _atomic_copy(source_path, release_path)
        return {
            "artifact_path": str(release_path),
            "artifact_hash": _sha256(release_path),
        }

    return cast(
        dict[Stage, Callable[[Any], StageUpdate]],
        {
            "preprocess": preprocess,
            "translate_parallel": translate_parallel,
            "merge_ass": merge_ass,
            "refine_serial": refine_serial,
            "postprocess": postprocess,
            "qa": qa,
            "human_review": human_review,
            "release": release,
        },
    )
