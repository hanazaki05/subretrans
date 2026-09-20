"""Concrete filesystem handlers for the subtitle processing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from .ass_parser import (
    apply_pairs_to_ass_lines,
    build_pairs_from_ass_lines,
    parse_ass_file,
    render_ass_file,
    write_ass_file,
)
from .memory import validate_memory_structure
from .model_agent import (
    AgentQA,
    AgentQAGlossaryTerm,
    AgentQAMemory,
    AgentQATerm,
    AgentRepair,
    AgentRepairHistory,
)
from .subtitle_processing import (
    POSTPROCESS_OPERATIONS,
    SrtCue,
    audit_ass,
    merge_srt_to_ass,
    postprocess_ass,
    read_srt,
    write_srt,
)
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


logger = logging.getLogger(__name__)

StageUpdate = Mapping[str, Any]
PreprocessSubtitle = Callable[[Path, Path], Path]
Refine = Callable[[Path, Path, Path, Path], None]


@dataclass(frozen=True)
class WorkflowSettings:
    run_dir: Path
    review_path: Path
    release_path: Path
    primer_batch_size: int
    primer_max_workers: int
    agent_max_repair_attempts: int
    episode_replacements: tuple[tuple[str, str], ...]
    postprocess_operations: tuple[str, ...] = POSTPROCESS_OPERATIONS
    qa_batch_size: int = 100
    qa_max_workers: int = 1
    qa_window_offsets: tuple[int, ...] = (0,)


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_qa_progress(
    path: Path,
    artifact_path: Path,
    artifact_hash: str,
    batch_size: int,
    window_offsets: tuple[int, ...],
    history_hash: str,
    memory_hash: str,
) -> tuple[int, bool, list[str], list[dict[str, Any]]]:
    if not path.exists():
        return 0, True, [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "version",
        "artifact_path",
        "artifact_hash",
        "batch_size",
        "window_offsets",
        "history_hash",
        "memory_hash",
        "next_window",
        "agent_passed",
        "issues",
        "repairs",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("QA progress has invalid fields")
    if payload["version"] != 4:
        raise ValueError("QA progress version must be 4")
    if payload["artifact_path"] != str(artifact_path):
        raise ValueError("QA progress artifact_path does not match")
    if payload["artifact_hash"] != artifact_hash:
        raise ValueError("QA progress artifact_hash does not match")
    if payload["batch_size"] != batch_size:
        raise ValueError("QA progress batch_size does not match qa.batch_size")
    if payload["window_offsets"] != list(window_offsets):
        raise ValueError("QA progress window_offsets do not match qa.window_offsets")
    if payload["history_hash"] != history_hash:
        raise ValueError("QA progress history_hash does not match repair history")
    if payload["memory_hash"] != memory_hash:
        raise ValueError("QA progress memory_hash does not match episode memory")
    if type(payload["next_window"]) is not int or payload["next_window"] < 0:
        raise ValueError("QA progress next_window must be a non-negative integer")
    if type(payload["agent_passed"]) is not bool:
        raise ValueError("QA progress agent_passed must be a boolean")
    if type(payload["issues"]) is not list or not all(
        isinstance(issue, str) for issue in payload["issues"]
    ):
        raise ValueError("QA progress issues must be a string list")
    if type(payload["repairs"]) is not list or not all(
        type(repair) is dict and set(repair) == {"id", "translation"}
        for repair in payload["repairs"]
    ):
        raise ValueError("QA progress repairs must contain id and translation")
    return (
        payload["next_window"],
        payload["agent_passed"],
        payload["issues"],
        payload["repairs"],
    )


def _load_qa_memory(state: PipelineState) -> tuple[AgentQAMemory, str]:
    memory_path_value = state["memory_checkpoint_path"]
    if not memory_path_value:
        raise ValueError("QA requires the refine memory checkpoint")
    memory_path = Path(memory_path_value)
    if not memory_path.is_file():
        raise FileNotFoundError(f"QA memory checkpoint not found: {memory_path}")
    memory_hash = _sha256(memory_path)
    if memory_hash != state["memory_hash"]:
        raise ValueError("QA memory checkpoint hash does not match pipeline state")
    payload = yaml.safe_load(memory_path.read_text(encoding="utf-8"))
    if not validate_memory_structure(payload):
        raise ValueError(f"Invalid QA memory checkpoint: {memory_path}")

    def user_terms(values: list[dict[str, Any]]) -> tuple[AgentQATerm, ...]:
        parsed: list[AgentQATerm] = []
        for value in values:
            if not isinstance(value["eng"], str) or not isinstance(value["zh"], str):
                raise ValueError("QA memory glossary eng and zh must be strings")
            parsed.append(AgentQATerm(value["eng"], value["zh"]))
        return tuple(parsed)

    def learned_terms(
        values: list[dict[str, Any]],
    ) -> tuple[AgentQAGlossaryTerm, ...]:
        parsed: list[AgentQAGlossaryTerm] = []
        for value in values:
            eng = value["eng"]
            zh = value["zh"]
            term_type = value.get("type")
            confidence = value.get("confidence")
            evidence_ids = value.get("evidence_ids", [])
            if not isinstance(eng, str) or not isinstance(zh, str):
                raise ValueError("QA memory glossary eng and zh must be strings")
            if term_type is not None and not isinstance(term_type, str):
                raise ValueError("QA memory glossary type must be a string or null")
            if confidence is not None and (
                isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            ):
                raise ValueError("QA memory glossary confidence must be numeric or null")
            if not isinstance(evidence_ids, list) or not all(
                type(identifier) is int for identifier in evidence_ids
            ):
                raise ValueError("QA memory glossary evidence_ids must be integers")
            parsed.append(
                AgentQAGlossaryTerm(
                    eng,
                    zh,
                    term_type,
                    float(confidence) if confidence is not None else None,
                    tuple(evidence_ids),
                )
            )
        return tuple(parsed)

    return (
        AgentQAMemory(
            story_description=payload["story_description"],
            user_glossary=user_terms(payload["user_glossary"]),
            glossary=learned_terms(payload["glossary"]),
        ),
        memory_hash,
    )


def _read_repair_history(path: Path) -> list[AgentRepairHistory]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict or set(payload) != {"version", "entries"}:
        raise ValueError("repair history has invalid fields")
    if payload["version"] != 1 or type(payload["entries"]) is not list:
        raise ValueError("repair history must be version 1 with an entries list")
    entries: list[AgentRepairHistory] = []
    for index, value in enumerate(payload["entries"]):
        if type(value) is not dict or set(value) != {"attempt", "id", "before", "after"}:
            raise ValueError(f"repair history entry {index} has invalid fields")
        entries.append(AgentRepairHistory(**value))
    return entries


def _write_repair_history(path: Path, entries: list[AgentRepairHistory]) -> None:
    _atomic_json(
        path,
        {
            "version": 1,
            "entries": [
                {
                    "attempt": entry.attempt,
                    "id": entry.id,
                    "before": entry.before,
                    "after": entry.after,
                }
                for entry in entries
            ],
        },
    )


def _reconstruct_repair_history(run_dir: Path, attempt: int) -> list[AgentRepairHistory]:
    entries: list[AgentRepairHistory] = []
    for repair_attempt in range(1, attempt + 1):
        before_name = (
            "postprocessed.ass"
            if repair_attempt == 1
            else f"postprocessed-{repair_attempt - 1:03d}.ass"
        )
        before_path = run_dir / before_name
        after_path = run_dir / f"qa-repair-{repair_attempt:03d}.ass"
        if not before_path.exists() or not after_path.exists():
            raise ValueError(
                f"cannot reconstruct QA repair history for attempt {repair_attempt}"
            )
        _, before_lines = parse_ass_file(str(before_path))
        _, after_lines = parse_ass_file(str(after_path))
        before_pairs = build_pairs_from_ass_lines(before_lines)
        after_pairs = build_pairs_from_ass_lines(after_lines)
        if len(before_pairs) != len(after_pairs):
            raise ValueError("QA repair history artifacts have different pair counts")
        for before, after in zip(before_pairs, after_pairs, strict=True):
            if before.id != after.id:
                raise ValueError("QA repair history artifacts have different pair ids")
            if before.chinese != after.chinese:
                entries.append(
                    AgentRepairHistory(
                        repair_attempt,
                        before.id,
                        before.chinese,
                        after.chinese,
                    )
                )
    return entries


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
    agent_qa: AgentQA,
) -> dict[Stage, Callable[[Any], StageUpdate]]:
    """Build all concrete handlers for one run directory and release target."""

    run_dir = Path(settings.run_dir)
    review_path = Path(settings.review_path)
    release_path = Path(settings.release_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(state: PipelineState) -> StageUpdate:
        source_path = Path(state["artifact_path"])
        if state["translation_mode"] == "parallel_initial":
            logger.info("Stage preprocess: cleaning %s", source_path)
            preprocessed_path = run_dir / "preprocessed.en.srt"
            _require_distinct(source_path, preprocessed_path)
            produced_path = preprocess_subtitle(source_path, preprocessed_path)
            if produced_path != preprocessed_path:
                raise ValueError(
                    "subtitle preprocessor must return run_dir/preprocessed.en.srt"
                )
            cues = read_srt(preprocessed_path)
            logger.info(
                "Stage preprocess complete: %d cues -> %s",
                len(cues),
                preprocessed_path,
            )
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
            logger.info("Stage preprocess: validating bilingual ASS %s", source_path)
            _require_extension(source_path, ".ass")
            audit = audit_ass(source_path)
            if audit.parse_errors:
                raise ValueError(
                    f"ASS artifact is not parseable: parse_errors={audit.parse_errors}"
                )
            logger.info("Stage preprocess complete: bilingual ASS is parseable")
            return {"translation_manifest_path": None}

        raise ValueError(f"unsupported translation mode: {state['translation_mode']}")

    def translate_parallel(state: MemorylessTranslationState) -> StageUpdate:
        logger.info("Stage translate_parallel: starting memoryless initial translation")
        manifest_path = Path(state["translation_manifest_path"])
        manifest_run_dir = manifest_path.parent
        translated_path = manifest_run_dir / "translated.zh.srt"
        manifest = translate_manifest(
            manifest_path,
            translate_batch,
            settings.primer_batch_size,
            settings.primer_max_workers,
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
        logger.info("Stage translate_parallel complete: %s", translated_path)
        return {}

    def merge_ass(state: PipelineState) -> StageUpdate:
        logger.info("Stage merge_ass: merging bilingual ASS")
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
        logger.info("Stage merge_ass complete: %s", output_path)
        return {
            "artifact_path": str(output_path),
            "artifact_hash": _sha256(output_path),
        }

    def refine_serial(state: PipelineState) -> StageUpdate:
        input_path = Path(state["artifact_path"])
        output_path = run_dir / "refined.ass"
        checkpoint_path = run_dir / "memory.yaml"
        progress_path = run_dir / "refine-progress.json"
        logger.info("Stage refine_serial: refining %s", input_path)
        _require_distinct(input_path, output_path)
        result = refine(input_path, output_path, checkpoint_path, progress_path)
        if result is not None:
            raise ValueError("refine must return None")
        next_pair, artifact_hash, memory_hash = _load_refine_progress(
            progress_path, output_path, checkpoint_path
        )
        logger.info(
            "Stage refine_serial complete: %d pairs committed -> %s",
            next_pair,
            output_path,
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
        attempt = state["agent_repair_attempts"]
        output_name = "postprocessed.ass" if attempt == 0 else f"postprocessed-{attempt:03d}.ass"
        output_path = run_dir / output_name
        logger.info(
            "Stage postprocess: applying deterministic cleanup (repair attempt %d)",
            attempt,
        )
        _require_distinct(input_path, output_path)
        postprocess_ass(
            input_path,
            output_path,
            settings.postprocess_operations,
            settings.episode_replacements,
        )
        logger.info("Stage postprocess complete: %s", output_path)
        return {
            "artifact_path": str(output_path),
            "artifact_hash": _sha256(output_path),
        }

    def qa(state: PipelineState) -> StageUpdate:
        input_path = Path(state["artifact_path"])
        if _sha256(input_path) != state["artifact_hash"]:
            raise ValueError("QA artifact hash does not match pipeline state")

        structural_qa = _qa_conclusion(input_path)
        header, ass_lines = parse_ass_file(str(input_path))
        pairs = build_pairs_from_ass_lines(ass_lines)
        episode_memory, memory_hash = _load_qa_memory(state)
        attempt = state["agent_repair_attempts"]
        repair_history_path = run_dir / "qa-repair-history.json"
        repair_history = _read_repair_history(repair_history_path)
        if attempt and not repair_history:
            repair_history = _reconstruct_repair_history(run_dir, attempt)
            _write_repair_history(repair_history_path, repair_history)
        history_hash = _sha256(repair_history_path) if repair_history else ""
        windows = [
            (round_index, start, min(start + settings.qa_batch_size, len(pairs)))
            for round_index, offset in enumerate(settings.qa_window_offsets, start=1)
            for start in range(offset, len(pairs), settings.qa_batch_size)
        ]
        offset_key = "-".join(str(offset) for offset in settings.qa_window_offsets)
        qa_progress_path = run_dir / (
            f"qa-progress-{state['artifact_hash'][:12]}-"
            f"{settings.qa_batch_size}-{offset_key}-"
            f"{history_hash[:12] or 'nohistory'}-{memory_hash[:12]}.json"
        )
        next_window, agent_passed, issues, raw_repairs = _load_qa_progress(
            qa_progress_path,
            input_path,
            state["artifact_hash"],
            settings.qa_batch_size,
            settings.qa_window_offsets,
            history_hash,
            memory_hash,
        )
        if next_window > len(windows):
            raise ValueError("QA progress next_window exceeds the window count")
        repairs = [
            AgentRepair(repair["id"], repair["translation"])
            for repair in raw_repairs
        ]
        logger.info(
            "Stage qa: auditing %d pairs in %d windows across %d passes; "
            "workers=%d; structural QA=%s; memory=%s",
            len(pairs),
            len(windows),
            len(settings.qa_window_offsets),
            settings.qa_max_workers,
            structural_qa,
            memory_hash[:12],
        )
        if next_window:
            logger.info(
                "Stage qa: resuming from window %d/%d (%.1f%%)",
                next_window,
                len(windows),
                next_window / len(windows) * 100,
            )
        while next_window < len(windows):
            wave = windows[next_window : next_window + settings.qa_max_workers]
            logger.info(
                "Stage qa progress: wave started with windows %d-%d/%d",
                next_window + 1,
                next_window + len(wave),
                len(windows),
            )
            wave_started = time.monotonic()
            results = {}
            with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                futures = {
                    executor.submit(
                        agent_qa,
                        tuple(pairs[start:end]),
                        structural_qa,
                        tuple(
                            entry
                            for entry in repair_history
                            if start <= entry.id < end
                        ),
                        episode_memory,
                    ): (index, round_index, start, end)
                    for index, (round_index, start, end) in enumerate(
                        wave, start=next_window
                    )
                }
                for future in as_completed(futures):
                    index, round_index, start, end = futures[future]
                    result = future.result()
                    results[index] = result
                    logger.info(
                        "Stage qa progress: window %d/%d complete "
                        "(pass %d/%d, pairs %d-%d); passed=%s, issues=%d, repairs=%d",
                        index + 1,
                        len(windows),
                        round_index,
                        len(settings.qa_window_offsets),
                        start,
                        end - 1,
                        result.passed,
                        len(result.issues),
                        len(result.repairs),
                    )
            for index in sorted(results):
                result = results[index]
                agent_passed = agent_passed and result.passed
                issues.extend(result.issues)
                repairs.extend(result.repairs)
            next_window += len(wave)
            _atomic_json(
                qa_progress_path,
                {
                    "version": 4,
                    "artifact_path": str(input_path),
                    "artifact_hash": state["artifact_hash"],
                    "batch_size": settings.qa_batch_size,
                    "window_offsets": list(settings.qa_window_offsets),
                    "history_hash": history_hash,
                    "memory_hash": memory_hash,
                    "next_window": next_window,
                    "agent_passed": agent_passed,
                    "issues": issues,
                    "repairs": [
                        {"id": repair.id, "translation": repair.translation}
                        for repair in repairs
                    ],
                },
            )
            logger.info(
                "Stage qa progress: wave committed; %d/%d windows (%.1f%%); "
                "elapsed=%.1fs",
                next_window,
                len(windows),
                next_window / len(windows) * 100,
                time.monotonic() - wave_started,
            )

        passed = structural_qa == "passed" and agent_passed
        if passed:
            _atomic_copy(input_path, review_path)
            logger.info("Stage qa passed; review artifact: %s", review_path)
            return {
                "artifact_path": str(review_path),
                "artifact_hash": _sha256(review_path),
                "qa_conclusion": "passed",
                "qa_passed": True,
                "qa_repair_applied": False,
            }

        pair_by_id = {pair.id: pair for pair in pairs}
        repairs_by_id: dict[int, AgentRepair] = {}
        conflicting_repair_ids: set[int] = set()
        for repair in repairs:
            existing = repairs_by_id.get(repair.id)
            if existing is not None and existing.translation != repair.translation:
                conflicting_repair_ids.add(repair.id)
            else:
                repairs_by_id[repair.id] = repair
        for repair_id in sorted(conflicting_repair_ids):
            repairs_by_id.pop(repair_id, None)
            issues.append(
                f"Overlapping QA windows proposed conflicting repairs for ID {repair_id}."
            )
        issue_text = "; ".join(issues)
        conclusion = f"structural={structural_qa}; agent={issue_text}"
        applicable_repairs = tuple(
            repair
            for repair in repairs_by_id.values()
            if (pair_by_id[repair.id].meta or {}).get("chinese_line_id", -1) >= 0
        )
        if not applicable_repairs or attempt >= settings.agent_max_repair_attempts:
            _atomic_copy(input_path, review_path)
            logger.warning(
                "Stage qa requires human review: issues=%d, applicable_repairs=%d; "
                "review artifact: %s",
                len(issues),
                len(applicable_repairs),
                review_path,
            )
            return {
                "artifact_path": str(review_path),
                "artifact_hash": _sha256(review_path),
                "qa_conclusion": conclusion,
                "qa_passed": False,
                "qa_repair_applied": False,
            }

        for repair in applicable_repairs:
            repair_history.append(
                AgentRepairHistory(
                    attempt + 1,
                    repair.id,
                    pair_by_id[repair.id].chinese,
                    repair.translation,
                )
            )
            pair_by_id[repair.id].chinese = repair.translation
        _write_repair_history(repair_history_path, repair_history)
        repaired_path = run_dir / f"qa-repair-{attempt + 1:03d}.ass"
        updated_lines = apply_pairs_to_ass_lines(ass_lines, pairs)
        write_ass_file(str(repaired_path), render_ass_file(header, updated_lines))
        logger.warning(
            "Stage qa found %d issues; applied %d repairs (attempt %d/%d)",
            len(issues),
            len(applicable_repairs),
            attempt + 1,
            settings.agent_max_repair_attempts,
        )
        return {
            "artifact_path": str(repaired_path),
            "artifact_hash": _sha256(repaired_path),
            "qa_conclusion": conclusion,
            "qa_passed": False,
            "qa_repair_applied": True,
            "agent_repair_attempts": attempt + 1,
        }

    def human_review(state: PipelineState) -> StageUpdate:
        return {}

    def release(state: PipelineState) -> StageUpdate:
        source_path = Path(state["artifact_path"])
        logger.info("Stage release: publishing %s -> %s", source_path, release_path)
        _atomic_copy(source_path, release_path)
        logger.info("Stage release complete: %s", release_path)
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
