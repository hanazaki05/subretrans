"""Filesystem handlers for the strict manifest-backed pipeline."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .ass_parser import build_pairs_from_ass_lines, parse_ass_file
from .fsutil import atomic_copy, atomic_write_json, require_distinct_paths, sha256_file
from .memory import load_memory_checkpoint
from .model_agent import (
    AgentQADecisionHistory,
    AgentQAGlossaryTerm,
    AgentQAMemory,
    AgentQATerm,
)
from .refine import load_refine_progress
from .run_manifest import (
    artifact_path,
    commit_manifest,
    load_manifest,
    mutable_manifest,
    register_artifact,
    verify_manifest_artifacts,
)
from .state import PipelineState, Stage
from .subtitle_processing import (
    POSTPROCESS_OPERATIONS,
    SrtCue,
    audit_ass,
    merge_srt_to_ass,
    postprocess_ass,
    read_srt,
    write_srt,
)
from .translation import (
    MANIFEST_VERSION,
    TranslateBatch,
    TranslationManifest,
    TranslationUnit,
    save_manifest as save_translation_manifest,
    translate_manifest,
)


logger = logging.getLogger(__name__)
PreprocessSubtitle = Callable[[Path, Path], Path]
Refine = Callable[[Path, Path, Path, Path], None]
AgentQA = Callable[..., Any]


def _bounded_reason(value: object) -> str:
    return " ".join(value.split())[:240] if isinstance(value, str) else "unspecified"


class FreezeCueManifest(Protocol):
    def __call__(self, input_path: Path, output_path: Path) -> object: ...


class FreezeEffectiveGlossary(Protocol):
    def __call__(
        self,
        *,
        memory_path: Path,
        cue_manifest_path: Path,
        artifact_path: Path,
        output_path: Path,
        decisions_path: Path,
        research_path: Path,
        repair_path: Path,
        episode_replacements: Sequence[tuple[str, str]],
        research_runner: GlossaryResearch | None,
        glossary_repair_runner: GlossaryRepair | None,
        max_research_requests: int,
        max_glossary_repairs: int,
    ) -> object: ...


class GlossaryResearch(Protocol):
    def __call__(
        self,
        *,
        candidates: Sequence[object],
        authoritative_terms: Mapping[str, str],
        max_requests: int,
        state_path: Path,
    ) -> object: ...


class GlossaryRepair(Protocol):
    def __call__(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        cues: Sequence[Mapping[str, Any]],
        authoritative: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any],
        max_attempts: int,
    ) -> object: ...


class RunRepair(Protocol):
    def __call__(
        self,
        *,
        run_dir: Path,
        refined_artifact_path: Path,
        current_artifact_path: Path,
        cue_manifest_path: Path,
        effective_glossary_path: Path,
        suggestion_pool_path: Path,
        decision_log_path: Path,
        coverage_path: Path,
        max_tool_steps: int,
        max_full_sweeps: int,
        max_group_span: int,
    ) -> object: ...


@dataclass(frozen=True)
class WorkflowSettings:
    run_dir: Path
    primer_batch_size: int
    primer_max_workers: int
    episode_replacements: tuple[tuple[str, str], ...]
    postprocess_operations: tuple[str, ...] = POSTPROCESS_OPERATIONS
    qa_batch_size: int = 100
    qa_max_workers: int = 1
    qa_window_offsets: tuple[int, ...] = (0,)
    max_tool_steps: int = 0
    max_full_sweeps: int = 1
    max_repair_attempts: int = 1
    max_group_span: int = 3


def _default_freeze_cue_manifest(input_path: Path, output_path: Path) -> object:
    from .cue_manifest import build_cue_manifest, save_cue_manifest

    return save_cue_manifest(build_cue_manifest(input_path), output_path)


def _default_freeze_effective_glossary(**kwargs: object) -> object:
    from .cue_manifest import load_cue_manifest
    from .glossary_validation import (
        GlossaryDecision,
        build_effective_glossary,
        normalize_term_key,
        validate_learned_candidates,
    )
    from .research import (
        ResearchReport,
        match_rank_name_term,
        rank_candidates_from_terms,
        unavailable_research_report,
        write_research_report,
    )

    memory_path = Path(cast(Path, kwargs["memory_path"]))
    cue_path = Path(cast(Path, kwargs["cue_manifest_path"]))
    artifact = Path(cast(Path, kwargs["artifact_path"]))
    output = Path(cast(Path, kwargs["output_path"]))
    decisions = Path(cast(Path, kwargs["decisions_path"]))
    research_path = Path(cast(Path, kwargs["research_path"]))
    repair_path = Path(cast(Path, kwargs["repair_path"]))
    replacements = cast(Sequence[tuple[str, str]], kwargs["episode_replacements"])
    research_runner = cast(GlossaryResearch | None, kwargs.get("research_runner"))
    repair_runner = cast(GlossaryRepair | None, kwargs.get("glossary_repair_runner"))
    max_research_requests = int(kwargs.get("max_research_requests", 0))
    max_glossary_repairs = int(kwargs.get("max_glossary_repairs", 0))
    memory = load_memory_checkpoint(memory_path)
    if memory is None:
        raise FileNotFoundError(f"memory checkpoint not found: {memory_path}")
    cues = load_cue_manifest(cue_path)
    authority = build_effective_glossary(
        memory.user_glossary,
        (),
        replacements,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    validation = validate_learned_candidates(
        memory.glossary,
        cues.cues,
        authority,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    rank_terms = [
        term for term in authority.authoritative if term.source == "user_glossary"
    ]
    rank_name_keys = {
        normalize_term_key(entry.get("eng"))
        for entry in memory.glossary
        if isinstance(entry.get("eng"), str)
        and match_rank_name_term(entry["eng"], rank_terms) is not None
    }
    candidates = rank_candidates_from_terms(
        memory.glossary,
        cues.cues,
        rank_terms=rank_terms,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    authority_map = {term.eng: term.zh for term in authority.authoritative}
    if candidates and research_runner is not None and max_research_requests > 0:
        report = research_runner(
            candidates=candidates,
            authoritative_terms=authority_map,
            max_requests=max_research_requests,
            state_path=research_path.with_name("glossary-research-state.json"),
        )
        if not isinstance(report, ResearchReport):
            raise TypeError("glossary research runner must return ResearchReport")
    elif candidates:
        reason = "research_budget_exhausted" if max_research_requests <= 0 else "research_unavailable"
        report = unavailable_research_report(candidates, reason, max_research_requests)
    else:
        report = ResearchReport((), 0, max_research_requests)
    write_research_report(research_path, report)

    candidate_keys = {
        normalize_term_key(f"{candidate.source_rank} {candidate.full_name}")
        for candidate in candidates
    }
    deterministic_terms = [
        term for term in validation.accepted if term.key not in rank_name_keys
    ]
    validation_by_key = {
        normalize_term_key(entry.eng): entry
        for entry in validation.decisions
        if entry.decision != "accepted" and entry.eng
    }
    repair_candidates = [
        dict(entry)
        for entry in memory.glossary
        if normalize_term_key(entry.get("eng")) in validation_by_key
        and normalize_term_key(entry.get("eng")) not in rank_name_keys
    ]
    missing_rank_context_decisions = [
        GlossaryDecision(
            str(entry.get("eng", "")),
            str(entry.get("zh", "")),
            "unresolved",
            "research-context-unavailable",
            "learned",
            tuple(entry.get("evidence_ids", ())),
        )
        for entry in memory.glossary
        if normalize_term_key(entry.get("eng")) in rank_name_keys - candidate_keys
    ]
    repair_result: Mapping[str, Any]
    if repair_candidates and repair_runner is not None and max_glossary_repairs > 0:
        raw_repair_result = repair_runner(
            candidates=repair_candidates,
            cues=[cue.to_dict() for cue in cues.cues],
            authoritative=[term.to_dict() for term in authority.authoritative],
            validation={
                "decisions": [entry.to_dict() for entry in validation.decisions],
                "evidence_issues": [entry.to_dict() for entry in validation.evidence_issues],
            },
            max_attempts=max_glossary_repairs,
        )
        if not isinstance(raw_repair_result, Mapping):
            raise TypeError("glossary repair runner must return a mapping")
        repair_result = raw_repair_result
    else:
        repair_result = {
            "attempts_used": 0,
            "actions": [
                {"action": "escalate", "eng": entry.get("eng", ""), "reason": "glossary_repair_unavailable"}
                for entry in repair_candidates
            ],
        }
    attempts_used = repair_result.get("attempts_used")
    actions = repair_result.get("actions")
    if type(attempts_used) is not int or not 0 <= attempts_used <= max_glossary_repairs:
        raise ValueError("glossary repair returned an invalid attempt count")
    if type(actions) is not list:
        raise ValueError("glossary repair actions must be a list")
    repair_proposals: list[Mapping[str, Any]] = []
    repair_decisions: list[GlossaryDecision] = []
    originals = {normalize_term_key(entry.get("eng")): entry for entry in repair_candidates}
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("glossary repair action must be an object")
        name = action.get("action")
        eng = action.get("eng")
        reason = action.get("reason")
        if name not in {"accept", "correct", "drop", "escalate"} or not isinstance(eng, str):
            raise ValueError("glossary repair action is invalid")
        original = originals.get(normalize_term_key(eng))
        if original is None:
            raise ValueError("glossary repair action names an unknown candidate")
        if name == "accept":
            repair_proposals.append(original)
        elif name == "correct":
            corrected = action.get("candidate")
            if not isinstance(corrected, Mapping):
                raise ValueError("correct action requires a candidate object")
            repair_proposals.append(corrected)
        else:
            repair_decisions.append(
                GlossaryDecision(
                    eng,
                    str(original.get("zh", "")),
                    "rejected" if name == "drop" else "unresolved",
                    f"glossary-repair-{name}:{_bounded_reason(reason)}",
                    "learned",
                    tuple(original.get("evidence_ids", ())),
                )
            )
    repair_validation = validate_learned_candidates(
        repair_proposals,
        cues.cues,
        authority,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    repair_revalidation_failures = [
        GlossaryDecision(
            entry.eng,
            entry.zh,
            "unresolved",
            f"glossary-repair-revalidation:{entry.reason}",
            "learned",
            entry.evidence_ids,
        )
        for entry in repair_validation.decisions
        if entry.decision != "accepted"
    ]
    atomic_write_json(
        repair_path,
        {
            "version": 1,
            "attempts_used": attempts_used,
            "actions": [dict(action) for action in actions],
            "human_review_required": any(
                entry.decision == "unresolved" for entry in repair_decisions
            )
            or bool(repair_validation.unresolved)
            or bool(repair_revalidation_failures),
        },
    )
    research_validation = validate_learned_candidates(
        [proposal.to_dict() for proposal in report.proposals],
        cues.cues,
        authority,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    research_revalidation_failures = [
        GlossaryDecision(
            entry.eng,
            entry.zh,
            "unresolved",
            f"research-revalidation:{entry.reason}",
            "learned",
            entry.evidence_ids,
        )
        for entry in research_validation.decisions
        if entry.decision != "accepted"
    ]
    research_decisions = [
        GlossaryDecision(
            f"{entry.candidate.source_rank} {entry.candidate.full_name}",
            entry.proposal.zh if entry.proposal is not None else "",
            "accepted" if entry.status == "agree" else "unresolved",
            f"research-{entry.status}",
            "learned",
            entry.candidate.pair_ids,
        )
        for entry in report.decisions
        if entry.status != "skipped_authoritative"
    ]
    effective = build_effective_glossary(
        memory.user_glossary,
        (*deterministic_terms, *repair_validation.accepted, *research_validation.accepted),
        replacements,
        episode_id=cues.episode_id,
        manifest_hash=cues.manifest_hash,
        artifact_hash=sha256_file(artifact),
    )
    effective_payload = effective.to_dict()
    combined_decisions = [
        *validation.decisions,
        *repair_decisions,
        *repair_validation.decisions,
        *repair_revalidation_failures,
        *missing_rank_context_decisions,
        *research_decisions,
        *research_validation.decisions,
        *research_revalidation_failures,
    ]
    resolved_repair_keys = {
        term.key for term in repair_validation.accepted
    } | {
        normalize_term_key(entry.eng)
        for entry in repair_decisions
        if entry.decision == "rejected"
    }
    combined_unresolved = [
        *(
            entry
            for entry in validation.unresolved
            if normalize_term_key(entry.eng) not in resolved_repair_keys
        ),
        *(entry for entry in repair_decisions if entry.decision == "unresolved"),
        *repair_validation.unresolved,
        *repair_revalidation_failures,
        *missing_rank_context_decisions,
        *(entry for entry in research_decisions if entry.decision == "unresolved"),
        *research_validation.unresolved,
        *research_revalidation_failures,
    ]
    effective_payload["decisions"] = [entry.to_dict() for entry in combined_decisions]
    effective_payload["unresolved"] = [entry.to_dict() for entry in combined_unresolved]
    atomic_write_json(output, effective_payload)
    atomic_write_json(
        decisions,
        {
            "version": 1,
            "decisions": [entry.to_dict() for entry in combined_decisions],
            "unresolved": [entry.to_dict() for entry in combined_unresolved],
            "evidence_issues": [
                entry.to_dict()
                for entry in (
                    *validation.evidence_issues,
                    *repair_validation.evidence_issues,
                    *research_validation.evidence_issues,
                )
            ],
            "research_artifact": research_path.name,
            "human_review_required": bool(combined_unresolved),
        },
    )
    return {
        "effective": effective_payload,
        "research_requests_used": report.requests_used,
        "glossary_repairs_used": attempts_used,
    }


def _default_run_repair(**kwargs: object) -> object:
    raise RuntimeError("build_stage_handlers requires an injected repair_runner")


def _require_extension(path: Path, extension: str) -> None:
    if path.suffix.lower() != extension:
        raise ValueError(f"expected a {extension} artifact: {path}")


def _commit(
    state: PipelineState,
    manifest: dict[str, Any],
    stage: Stage,
    next_stage: Stage | str,
    reason: str,
    *,
    allow_modified_review: bool = False,
) -> PipelineState:
    return commit_manifest(
        state,
        stage=stage,
        next_stage=cast(Any, next_stage),
        route_reason=reason,
        manifest=manifest,
        allow_modified_review=allow_modified_review,
    )


def _current_or_source(state: PipelineState, manifest: Mapping[str, Any]) -> Path:
    if manifest["heads"]["current"] is not None:
        return artifact_path(state["manifest_path"], manifest, "current")
    source = Path(manifest["inputs"]["source"]["path"])
    if sha256_file(source) != manifest["inputs"]["source"]["sha256"]:
        raise ValueError("source artifact hash does not match run manifest")
    return source


def _qa_memory(memory_path: Path, effective_glossary_path: Path) -> AgentQAMemory:
    memory = load_memory_checkpoint(memory_path)
    if memory is None:
        raise FileNotFoundError(f"QA memory checkpoint not found: {memory_path}")
    effective = json.loads(effective_glossary_path.read_text(encoding="utf-8"))
    if type(effective) is not dict:
        raise ValueError("effective glossary must be a JSON object")
    authoritative = effective.get("authoritative")
    learned = effective.get("learned")
    if type(authoritative) is not list or type(learned) is not list:
        raise ValueError("effective glossary must contain authoritative and learned arrays")
    return AgentQAMemory(
        story_description=memory.story_description,
        user_glossary=tuple(
            AgentQATerm(entry["eng"], entry["zh"]) for entry in authoritative
        ),
        glossary=tuple(
            AgentQAGlossaryTerm(
                entry["eng"],
                entry["zh"],
                entry.get("type"),
                float(entry["confidence"]) if entry.get("confidence") is not None else None,
                tuple(entry.get("evidence_ids", ())),
            )
            for entry in learned
        ),
    )


def _decision_history(
    manifest_path: str, manifest: Mapping[str, Any]
) -> tuple[AgentQADecisionHistory, ...]:
    if manifest["heads"]["decision_log"] is None:
        return ()
    path = artifact_path(manifest_path, manifest, "decision_log")
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if type(decisions) is not list:
        raise ValueError("repair decision log must contain a decisions array")
    latest: dict[str, AgentQADecisionHistory] = {}
    for entry in decisions:
        if not isinstance(entry, dict) or not all(
            type(entry.get(name)) is str and entry[name]
            for name in ("issue_key", "status", "reason")
        ):
            continue
        latest[entry["issue_key"]] = AgentQADecisionHistory(
            entry["issue_key"], entry["status"], entry["reason"]
        )
    return tuple(latest.values())


def _audit_windows(
    *,
    input_path: Path,
    cue_manifest_hash: str,
    glossary_hash: str,
    pass_name: str,
    settings: WorkflowSettings,
    agent_qa: AgentQA,
    episode_memory: AgentQAMemory,
    decision_history: tuple[AgentQADecisionHistory, ...],
) -> dict[str, Any]:
    _, ass_lines = parse_ass_file(input_path)
    pairs = build_pairs_from_ass_lines(ass_lines)
    structural = audit_ass(input_path)
    structural_summary = "passed" if structural.passed else dataclasses.asdict(structural)
    windows = [
        (pass_index, start, min(start + settings.qa_batch_size, len(pairs)))
        for pass_index, offset in enumerate(settings.qa_window_offsets, start=1)
        for start in range(offset, len(pairs), settings.qa_batch_size)
    ]
    results: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=settings.qa_max_workers) as executor:
        futures = {
            executor.submit(
                agent_qa,
                tuple(pairs[start:end]),
                json.dumps(structural_summary, ensure_ascii=False),
                decision_history,
                episode_memory,
            ): (index, pass_index, start, end)
            for index, (pass_index, start, end) in enumerate(windows)
        }
        for future in as_completed(futures):
            index, pass_index, start, end = futures[future]
            results[index] = (pass_index, start, end, future.result())

    artifact_hash = sha256_file(input_path)
    window_records: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    from .repair import host_qa_suggestions

    for index in range(len(windows)):
        pass_index, start, end, result = results[index]
        raw_suggestions = tuple(result.suggestions)
        suggestion_ids: list[str] = []
        hosted = host_qa_suggestions(
            raw_suggestions,
            source_window=(start, end),
            source_pass=pass_index,
            artifact_hash=artifact_hash,
            effective_glossary_hash=glossary_hash,
            manifest_hash=cue_manifest_hash,
        )
        suppressed_keys = {
            entry.issue_key
            for entry in decision_history
            if entry.status in {"dismissed", "kept"}
        }
        for raw in hosted:
            if raw.key in suppressed_keys:
                continue
            suggestion = dataclasses.asdict(raw)
            suggestions.append(suggestion)
            suggestion_ids.append(raw.issue_id)
        window_records.append(
            {
                "window_id": index,
                "overlap_pass": pass_index,
                "start": start,
                "end": end,
                "passed": bool(result.passed),
                "suggestion_ids": suggestion_ids,
            }
        )
    return {
        "version": 1,
        "pass": pass_name,
        "artifact_path": str(input_path),
        "artifact_hash": artifact_hash,
        "cue_manifest_hash": cue_manifest_hash,
        "effective_glossary_hash": glossary_hash,
        "structural_qa": structural_summary,
        "windows": window_records,
        "suggestions": suggestions,
    }


def _result_value(result: object, name: str, default: object = None) -> object:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def build_stage_handlers(
    settings: WorkflowSettings,
    *,
    preprocess_subtitle: PreprocessSubtitle,
    translate_batch: TranslateBatch,
    refine: Refine,
    agent_qa: AgentQA,
    freeze_cues: FreezeCueManifest | None = None,
    freeze_glossary: FreezeEffectiveGlossary | None = None,
    glossary_research_runner: GlossaryResearch | None = None,
    glossary_repair_runner: GlossaryRepair | None = None,
    repair_runner: RunRepair | None = None,
) -> dict[Stage, Callable[..., PipelineState]]:
    """Build concrete handlers; every handler commits exactly one manifest revision."""

    run_dir = Path(settings.run_dir).resolve()
    freeze_cues = freeze_cues or _default_freeze_cue_manifest
    freeze_glossary = freeze_glossary or _default_freeze_effective_glossary
    repair_runner = repair_runner or _default_run_repair

    def preprocess(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="preprocess")
        source = Path(manifest["inputs"]["source"]["path"])
        if sha256_file(source) != manifest["inputs"]["source"]["sha256"]:
            raise ValueError("source artifact hash does not match run manifest")
        if manifest["translation_mode"] == "parallel_initial":
            output = run_dir / "preprocessed.en.srt"
            require_distinct_paths(source, output)
            if preprocess_subtitle(source, output) != output:
                raise ValueError("subtitle preprocessor returned an unexpected path")
            cues = read_srt(output)
            translated = run_dir / "translated.zh.srt"
            translation_path = run_dir / "translation-seed.json"
            save_translation_manifest(
                TranslationManifest(
                    version=MANIFEST_VERSION,
                    source_artifact_path=str(output),
                    translated_artifact_path=str(translated),
                    units=[
                        TranslationUnit(id=cue.index, source=cue.text, translation=None)
                        for cue in cues
                    ],
                ),
                translation_path,
            )
            register_artifact(
                manifest,
                checked["manifest_path"],
                name="preprocessed",
                path=output,
                kind="srt",
                head="preprocessed",
            )
            register_artifact(
                manifest,
                checked["manifest_path"],
                name="translation_seed",
                path=translation_path,
                kind="translation-seed",
                head="translation_manifest",
                depends_on=("preprocessed",),
            )
            return _commit(checked, manifest, "preprocess", "translate_parallel", "parallel_input_ready")

        if manifest["translation_mode"] != "serial_memory":
            raise ValueError("unsupported translation mode")
        _require_extension(source, ".ass")
        if audit_ass(source).parse_errors:
            raise ValueError("ASS artifact is not parseable")
        copied = run_dir / "source.ass"
        atomic_copy(source, copied)
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="source_ass",
            path=copied,
            kind="ass",
            head="current",
        )
        return _commit(checked, manifest, "preprocess", "freeze_manifest", "serial_input_ready")

    def translate_parallel(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="translate_parallel")
        translation_seed = artifact_path(
            checked["manifest_path"], manifest, "translation_manifest"
        )
        translation_path = run_dir / "translation-progress.json"
        if not translation_path.exists():
            atomic_copy(translation_seed, translation_path)
        translated_path = run_dir / "translated.zh.srt"
        translated_manifest = translate_manifest(
            translation_path,
            translate_batch,
            settings.primer_batch_size,
            settings.primer_max_workers,
        )
        source_cues = read_srt(Path(translated_manifest.source_artifact_path))
        translated_cues: list[SrtCue] = []
        for cue, unit in zip(source_cues, translated_manifest.units, strict=True):
            if unit.id != cue.index or unit.source != cue.text or unit.translation is None:
                raise ValueError("translation manifest does not match source cues")
            translated_cues.append(SrtCue(cue.index, cue.start, cue.end, unit.translation))
        write_srt(translated_cues, translated_path)
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="translation_manifest",
            path=translation_path,
            kind="translation-manifest",
            head="translation_manifest",
            depends_on=("translation_seed",),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="translated",
            path=translated_path,
            kind="srt",
            head="translated",
            depends_on=("translation_manifest",),
        )
        return _commit(checked, manifest, "translate_parallel", "merge_ass", "primer_complete")

    def merge_ass(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="merge_ass")
        source = artifact_path(checked["manifest_path"], manifest, "preprocessed")
        translated = artifact_path(checked["manifest_path"], manifest, "translated")
        output = run_dir / "merged.ass"
        merge_srt_to_ass(source, translated, output)
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="merged",
            path=output,
            kind="ass",
            head="current",
            depends_on=("preprocessed", "translated"),
        )
        return _commit(checked, manifest, "merge_ass", "freeze_manifest", "bilingual_ass_ready")

    def freeze_manifest(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="freeze_manifest")
        current = _current_or_source(checked, manifest)
        output = run_dir / "cue-manifest.json"
        freeze_cues(current, output)
        if not output.is_file():
            raise ValueError("freeze_cue_manifest did not write its output")
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="cue_manifest",
            path=output,
            kind="cue-manifest",
            head="cue_manifest",
            depends_on=(manifest["heads"]["current"],),
        )
        return _commit(checked, manifest, "freeze_manifest", "refine_serial", "cue_identity_frozen")

    def refine_serial(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="refine_serial")
        current = artifact_path(checked["manifest_path"], manifest, "current")
        output = run_dir / "refined.ass"
        memory = run_dir / "memory.yaml"
        progress = run_dir / "refine-progress.json"
        refine(current, output, memory, progress)
        committed = load_refine_progress(progress, output, memory)
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="refined",
            path=output,
            kind="ass",
            head="refined",
            depends_on=(manifest["heads"]["current"], "cue_manifest"),
        )
        manifest["heads"]["current"] = "refined"
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="memory",
            path=memory,
            kind="episode-memory",
            head="memory",
            depends_on=("refined",),
        )
        if committed.artifact_hash != sha256_file(output) or committed.memory_hash != sha256_file(memory):
            raise ValueError("refine progress does not match committed artifacts")
        return _commit(checked, manifest, "refine_serial", "postprocess", "refine_complete")

    def postprocess(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="postprocess")
        current = artifact_path(checked["manifest_path"], manifest, "current")
        output = run_dir / "postprocessed.ass"
        postprocess_ass(
            current,
            output,
            settings.postprocess_operations,
            settings.episode_replacements,
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="postprocessed",
            path=output,
            kind="ass",
            head="current",
            depends_on=(manifest["heads"]["current"],),
        )
        return _commit(checked, manifest, "postprocess", "glossary", "deterministic_cleanup_complete")

    def glossary(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="glossary")
        memory = artifact_path(checked["manifest_path"], manifest, "memory")
        cues = artifact_path(checked["manifest_path"], manifest, "cue_manifest")
        current = artifact_path(checked["manifest_path"], manifest, "current")
        output = run_dir / "effective-glossary.json"
        decisions = run_dir / "glossary-decisions.json"
        research_path = run_dir / "glossary-research.json"
        research_state_path = run_dir / "glossary-research-state.json"
        repair_path = run_dir / "glossary-repairs.json"
        research_budget = manifest["budgets"]["research_requests"]
        remaining_research = research_budget["max"] - research_budget["used"]
        glossary_repair_budget = manifest["budgets"]["glossary_repairs"]
        remaining_glossary_repairs = (
            glossary_repair_budget["max"] - glossary_repair_budget["used"]
        )
        result = freeze_glossary(
            memory_path=memory,
            cue_manifest_path=cues,
            artifact_path=current,
            output_path=output,
            decisions_path=decisions,
            research_path=research_path,
            repair_path=repair_path,
            episode_replacements=settings.episode_replacements,
            research_runner=glossary_research_runner,
            glossary_repair_runner=glossary_repair_runner,
            max_research_requests=remaining_research,
            max_glossary_repairs=remaining_glossary_repairs,
        )
        if not output.is_file() or not decisions.is_file():
            raise ValueError("freeze_effective_glossary did not write required artifacts")
        requests_used = _result_value(result, "research_requests_used", 0)
        if (
            type(requests_used) is not int
            or requests_used < 0
            or requests_used > remaining_research
        ):
            raise ValueError("glossary research returned an invalid request count")
        research_budget["used"] += requests_used
        glossary_repairs_used = _result_value(result, "glossary_repairs_used", 0)
        if (
            type(glossary_repairs_used) is not int
            or glossary_repairs_used < 0
            or glossary_repairs_used > remaining_glossary_repairs
        ):
            raise ValueError("glossary repair returned an invalid used count")
        glossary_repair_budget["used"] += glossary_repairs_used
        research_dependency: tuple[str, ...] = ()
        if research_state_path.is_file():
            register_artifact(
                manifest,
                checked["manifest_path"],
                name="glossary_research_state",
                path=research_state_path,
                kind="glossary-research-state",
                depends_on=("memory", "cue_manifest", manifest["heads"]["current"]),
            )
            research_dependency = ("glossary_research_state",)
        if research_path.is_file():
            register_artifact(
                manifest,
                checked["manifest_path"],
                name="glossary_research",
                path=research_path,
                kind="glossary-research",
                depends_on=("memory", "cue_manifest", manifest["heads"]["current"]),
            )
            research_dependency = (*research_dependency, "glossary_research")
        if repair_path.is_file():
            register_artifact(
                manifest,
                checked["manifest_path"],
                name="glossary_repairs",
                path=repair_path,
                kind="glossary-repairs",
                depends_on=("memory", "cue_manifest", manifest["heads"]["current"]),
            )
            research_dependency = (*research_dependency, "glossary_repairs")
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="effective_glossary",
            path=output,
            kind="effective-glossary",
            head="effective_glossary",
            depends_on=(
                "memory",
                "cue_manifest",
                manifest["heads"]["current"],
                *research_dependency,
            ),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name="glossary_decisions",
            path=decisions,
            kind="glossary-decisions",
            depends_on=("effective_glossary",),
        )
        return _commit(checked, manifest, "glossary", "qa", "effective_glossary_frozen")

    def run_qa(state: PipelineState, stage: Stage, pass_name: str) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage=stage)
        current = artifact_path(checked["manifest_path"], manifest, "current")
        cues = artifact_path(checked["manifest_path"], manifest, "cue_manifest")
        glossary_path = artifact_path(checked["manifest_path"], manifest, "effective_glossary")
        memory = artifact_path(checked["manifest_path"], manifest, "memory")
        pool = _audit_windows(
            input_path=current,
            cue_manifest_hash=sha256_file(cues),
            glossary_hash=sha256_file(glossary_path),
            pass_name=pass_name,
            settings=settings,
            agent_qa=agent_qa,
            episode_memory=_qa_memory(memory, glossary_path),
            decision_history=_decision_history(checked["manifest_path"], manifest),
        )
        output = run_dir / f"suggestion-pool-{pass_name}-{manifest['revision'] + 1:03d}.json"
        atomic_write_json(output, pool)
        name = f"suggestion_pool_{pass_name}_{manifest['revision'] + 1:03d}"
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=name,
            path=output,
            kind="qa-suggestion-pool",
            head="suggestion_pool",
            depends_on=(manifest["heads"]["current"], "cue_manifest", "effective_glossary"),
        )
        if stage == "qa":
            return _commit(checked, manifest, "qa", "repair", "initial_audit_committed")
        suggestions = pool["suggestions"]
        budget = manifest["budgets"]["repair_attempts"]
        if suggestions and budget["used"] < budget["max"]:
            return _commit(checked, manifest, "qa_verify", "repair", "actionable_suggestions_remain")
        reason = "qa_clean" if not suggestions else "repair_budget_exhausted"
        return _commit(checked, manifest, "qa_verify", "review_export", reason)

    def qa(state: PipelineState) -> PipelineState:
        return run_qa(state, "qa", "initial")

    def repair(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="repair")
        budget = manifest["budgets"]["repair_attempts"]
        if budget["used"] >= budget["max"]:
            return _commit(checked, manifest, "repair", "qa_verify", "repair_budget_exhausted")
        refined = artifact_path(checked["manifest_path"], manifest, "refined")
        current = artifact_path(checked["manifest_path"], manifest, "current")
        cues = artifact_path(checked["manifest_path"], manifest, "cue_manifest")
        glossary_path = artifact_path(checked["manifest_path"], manifest, "effective_glossary")
        pool = artifact_path(checked["manifest_path"], manifest, "suggestion_pool")
        attempt = budget["used"] + 1
        decision_log = run_dir / "repair" / f"decisions-{attempt:03d}.json"
        coverage = run_dir / "repair" / f"coverage-{attempt:03d}.json"
        decision_log.parent.mkdir(parents=True, exist_ok=True)
        result = repair_runner(
            run_dir=run_dir,
            refined_artifact_path=refined,
            current_artifact_path=current,
            cue_manifest_path=cues,
            effective_glossary_path=glossary_path,
            suggestion_pool_path=pool,
            decision_log_path=decision_log,
            coverage_path=coverage,
            max_tool_steps=manifest["budgets"]["tool_steps"]["max"]
            - manifest["budgets"]["tool_steps"]["used"],
            max_full_sweeps=manifest["budgets"]["full_sweeps"]["max"]
            - manifest["budgets"]["full_sweeps"]["used"],
            max_group_span=settings.max_group_span,
        )
        artifact_value = _result_value(result, "artifact_path", current)
        repaired = Path(cast(Path | str, artifact_value))
        repair_state = Path(cast(Path | str, _result_value(result, "repair_state_path")))
        history = Path(cast(Path | str, _result_value(result, "history_path")))
        staged = Path(cast(Path | str, _result_value(result, "staged_path")))
        exchanges = Path(cast(Path | str, _result_value(result, "exchanges_path")))
        if not repaired.is_file():
            raise ValueError("repair runner did not produce a current artifact")
        for required, label in (
            (decision_log, "decision log"),
            (coverage, "coverage ledger"),
            (repair_state, "state snapshot"),
            (history, "history ledger"),
            (staged, "staged-groups ledger"),
            (exchanges, "tool-exchange ledger"),
        ):
            if not required.is_file():
                raise ValueError(f"repair runner did not write its {label}")
        repaired_name = f"repaired_{attempt:03d}"
        if repaired.resolve() != current.resolve():
            register_artifact(
                manifest,
                checked["manifest_path"],
                name=repaired_name,
                path=repaired,
                kind="ass",
                head="current",
                depends_on=(manifest["heads"]["current"], manifest["heads"]["suggestion_pool"]),
            )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"repair_state_{attempt:03d}",
            path=repair_state,
            kind="repair-state",
            head="repair_state",
            depends_on=(manifest["heads"]["current"], manifest["heads"]["suggestion_pool"]),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"decision_log_{attempt:03d}",
            path=decision_log,
            kind="repair-decisions",
            head="decision_log",
            depends_on=(f"repair_state_{attempt:03d}",),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"repair_history_{attempt:03d}",
            path=history,
            kind="repair-history",
            head="repair_history",
            depends_on=(f"repair_state_{attempt:03d}",),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"repair_staged_{attempt:03d}",
            path=staged,
            kind="repair-staged-groups",
            head="repair_staged",
            depends_on=(f"repair_state_{attempt:03d}",),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"repair_exchanges_{attempt:03d}",
            path=exchanges,
            kind="repair-tool-exchanges",
            head="repair_exchanges",
            depends_on=(f"repair_state_{attempt:03d}",),
        )
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"coverage_{attempt:03d}",
            path=coverage,
            kind="repair-coverage",
            head="coverage",
            depends_on=(f"repair_state_{attempt:03d}", manifest["heads"]["current"]),
        )
        attempts_total = _result_value(result, "repair_attempts_total", attempt)
        if (
            type(attempts_total) is not int
            or attempts_total < budget["used"]
            or attempts_total > budget["max"]
        ):
            raise ValueError("repair runner repair_attempts_total is outside the run budget")
        budget["used"] = attempts_total
        for name, result_field, total_field in (
            ("tool_steps", "tool_steps_used", "tool_steps_total"),
            ("full_sweeps", "full_sweeps_used", "full_sweeps_total"),
        ):
            increment = _result_value(result, result_field, 0)
            if type(increment) is not int or increment < 0:
                raise ValueError(f"repair runner {result_field} must be a non-negative integer")
            counter = manifest["budgets"][name]
            total = _result_value(result, total_field, counter["used"] + increment)
            if type(total) is not int or total < counter["used"] or total > counter["max"]:
                raise ValueError(f"repair runner {total_field} is outside the run budget")
            counter["used"] = total
        reason = "repair_escalated" if _result_value(result, "escalated", False) else "repair_attempt_committed"
        return _commit(checked, manifest, "repair", "qa_verify", reason)

    def qa_verify(state: PipelineState) -> PipelineState:
        return run_qa(state, "qa_verify", "verify")

    def review_export(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="review_export")
        current = artifact_path(checked["manifest_path"], manifest, "current")
        review = Path(manifest["review"]["path"])
        review.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(current, review)
        name = f"review_{manifest['revision'] + 1:03d}"
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=name,
            path=review,
            kind="review-ass",
            head="review",
            depends_on=(manifest["heads"]["current"],),
        )
        manifest["review"].update(status="awaiting", sha256=sha256_file(review))
        return _commit(checked, manifest, "review_export", "human_review", "awaiting_human_review")

    def human_review(state: PipelineState, decision: str) -> PipelineState:
        checked, manifest = mutable_manifest(
            state, expected_stage="human_review", allow_modified_review=True
        )
        review = Path(manifest["review"]["path"])
        if not review.is_file():
            raise FileNotFoundError(f"review artifact not found: {review}")
        if decision == "reject":
            manifest["review"].update(status="rejected", sha256=sha256_file(review))
            review_name = manifest["heads"]["review"]
            manifest["artifacts"][review_name]["sha256"] = sha256_file(review)
            return _commit(
                checked,
                manifest,
                "human_review",
                "end",
                "human_rejected",
                allow_modified_review=True,
            )
        if decision != "approve":
            raise ValueError("human review decision must be approve or reject")
        audit = audit_ass(review)
        if not audit.passed:
            raise ValueError("edited review artifact failed structural validation")
        approved = run_dir / "approved" / f"approved-{manifest['revision'] + 1:03d}.ass"
        approved.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(review, approved)
        review_name = manifest["heads"]["review"]
        manifest["artifacts"][review_name]["sha256"] = sha256_file(review)
        register_artifact(
            manifest,
            checked["manifest_path"],
            name=f"approved_{manifest['revision'] + 1:03d}",
            path=approved,
            kind="approved-ass",
            head="approved",
            depends_on=(review_name,),
        )
        manifest["review"].update(status="approved", sha256=sha256_file(review))
        return _commit(
            checked,
            manifest,
            "human_review",
            "release",
            "human_approved",
            allow_modified_review=True,
        )

    def release(state: PipelineState) -> PipelineState:
        checked, manifest = mutable_manifest(state, expected_stage="release")
        approved = artifact_path(checked["manifest_path"], manifest, "approved")
        destination = Path(manifest["release"]["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(approved, destination)
        manifest["release"].update(status="released", sha256=sha256_file(destination))
        return _commit(checked, manifest, "release", "end", "release_complete")

    return cast(
        dict[Stage, Callable[..., PipelineState]],
        {
            "preprocess": preprocess,
            "translate_parallel": translate_parallel,
            "merge_ass": merge_ass,
            "freeze_manifest": freeze_manifest,
            "refine_serial": refine_serial,
            "postprocess": postprocess,
            "glossary": glossary,
            "qa": qa,
            "repair": repair,
            "qa_verify": qa_verify,
            "review_export": review_export,
            "human_review": human_review,
            "release": release,
        },
    )
