"""Command-line entry point for the strict manifest-backed pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .fsutil import atomic_copy, atomic_write_json, sha256_file
from .ass_parser import apply_pairs_to_ass_lines, build_pairs_from_ass_lines, parse_ass_file, render_ass_file, write_ass_file
from .memory import load_memory_checkpoint
from .model_agent import build_agent_qa
from .model_translation import build_model_translate_batch
from .pipeline import build_pipeline
from .prompts import compose_prompt, load_prompt_file, load_qa_prompt_template
from .refine import RefineOptions, load_refine_progress, refine_serial
from .run_manifest import create_manifest, load_manifest
from .stage_handlers import Refine, WorkflowSettings, build_stage_handlers
from .state import TranslationMode
from .subtitle_edit import preprocess_with_seconv
from .translation import TranslateBatch, TranslationBatch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    thread_id: str
    config: AppConfig
    run_dir: Path
    translation_mode: TranslationMode

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def graph_config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}


def _validate_thread_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("thread id must use letters, digits, dot, underscore, or dash")
    if value in {".", ".."}:
        raise ValueError("invalid thread id")
    return value


def _prompt_version(config: AppConfig) -> str:
    """Hash named prompt components with boundaries, including repair when configured."""

    digest = hashlib.sha256()
    paths = [config.path]
    for name in ("shared", "refine", "qa", "repair"):
        path = getattr(config.prompts, name, None)
        if path is not None:
            paths.append(Path(path))
    for path in paths:
        encoded_name = str(Path(path).resolve()).encode()
        content = Path(path).read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _lazy_translate_batch(config: AppConfig) -> TranslateBatch:
    lock = threading.Lock()
    translator: TranslateBatch | None = None

    def translate(batch: TranslationBatch):
        nonlocal translator
        if translator is None:
            with lock:
                if translator is None:
                    translator = build_model_translate_batch(
                        config.api.primer.config,
                        source_language=config.primer.source_language,
                        target_language=config.primer.target_language,
                        user_instruction=config.primer.user_instruction,
                    )
        return translator(batch)

    return translate


def _refine_callable(config: AppConfig) -> Refine:
    def refine(
        input_path: Path, output_path: Path, checkpoint_path: Path, progress_path: Path
    ) -> None:
        options = RefineOptions()
        if progress_path.exists():
            progress = load_refine_progress(progress_path, output_path, checkpoint_path)
            options = RefineOptions(resume_index=progress.next_pair)
        refine_serial(
            input_path,
            output_path,
            config,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            options=options,
        )

    return refine


def _subtitle_preprocessor(config: AppConfig):
    def preprocess(input_path: Path, output_path: Path) -> Path:
        return preprocess_with_seconv(config.subtitle_edit, input_path, output_path)

    return preprocess


def _repair_setting(config: AppConfig, name: str, default: int) -> int:
    repair = getattr(config, "repair", None)
    value = getattr(repair, name, default)
    if type(value) is not int or value < 0:
        raise ValueError(f"repair.{name} must be a non-negative integer")
    return value


def _repair_callable(config: AppConfig):
    """Adapt the host-owned repair engine to the pipeline artifact boundary."""

    from .repair import RepairSession, RepairSuggestion, run_repair_agent
    from .reference_reader import ReferenceReader
    from .webfetch import FetchBudget, fetch_url

    system_prompt = compose_prompt(
        load_prompt_file(config.prompts.shared), load_prompt_file(config.prompts.repair)
    )
    reference_roots = tuple(getattr(config, "reference_roots", ()))
    reference_reader = ReferenceReader(reference_roots[0]) if reference_roots else None

    def webfetch_executor(url: str) -> dict[str, object]:
        return fetch_url(
            url,
            FetchBudget(
                timeout=config.research.timeout,
                max_bytes=config.research.max_response_bytes,
            ),
        ).audit_dict()

    def run_repair(**kwargs: object) -> dict[str, object]:
        run_dir = Path(cast(Any, kwargs["run_dir"]))
        refined_path = Path(cast(Any, kwargs["refined_artifact_path"]))
        current_path = Path(cast(Any, kwargs["current_artifact_path"]))
        pool_path = Path(cast(Any, kwargs["suggestion_pool_path"]))
        cue_path = Path(cast(Any, kwargs["cue_manifest_path"]))
        glossary_path = Path(cast(Any, kwargs["effective_glossary_path"]))
        decision_log_path = Path(cast(Any, kwargs["decision_log_path"]))
        coverage_path = Path(cast(Any, kwargs["coverage_path"]))
        attempt = int(decision_log_path.stem.rsplit("-", 1)[-1])
        state_dir = run_dir / "repair" / "session"
        _, refined_lines = parse_ass_file(refined_path)
        current_header, current_lines = parse_ass_file(current_path)
        refined_pairs = tuple(build_pairs_from_ass_lines(refined_lines))
        current_pairs = tuple(build_pairs_from_ass_lines(current_lines))
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        suggestions = tuple(
            RepairSuggestion(
                issue_id=value["issue_id"],
                key=value["key"],
                affected_ids=tuple(value["affected_ids"]),
                kind=value["kind"],
                diagnosis=value["diagnosis"],
                evidence=tuple(value["evidence"]),
                suggested_translations=tuple(value["suggested_translations"]),
                source_window=tuple(value["source_window"]),
                source_pass=value["source_pass"],
                artifact_hash=value["artifact_hash"],
                effective_glossary_hash=value["effective_glossary_hash"],
                manifest_hash=value["manifest_hash"],
            )
            for value in pool["suggestions"]
        )
        cue_hash = sha256_file(cue_path)
        session_settings = dataclasses.replace(
            config.repair,
            max_group_span=int(cast(Any, kwargs["max_group_span"])),
        )
        effective_glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
        state_file = state_dir / "repair-state.json"
        if state_file.exists():
            session = RepairSession.resume(
                state_dir=state_dir,
                settings=session_settings,
                operations=config.postprocess.operations,
                episode_replacements=config.postprocess.episode_replacements,
                manifest_hash=cue_hash,
                reference_reader=reference_reader,
                webfetch_executor=webfetch_executor,
            )
            session.begin_round(current_pairs=current_pairs, suggestions=suggestions)
        else:
            session = RepairSession(
                state_dir=state_dir,
                settings=session_settings,
                refined_pairs=refined_pairs,
                current_pairs=current_pairs,
                suggestions=suggestions,
                operations=config.postprocess.operations,
                episode_replacements=config.postprocess.episode_replacements,
                manifest_hash=cue_hash,
                effective_glossary=effective_glossary,
                reference_reader=reference_reader,
                webfetch_executor=webfetch_executor,
            )
        tool_steps_before = session.tool_steps_used
        full_sweeps_before = session.full_sweeps_completed
        memory = load_memory_checkpoint(run_dir / "memory.yaml")
        if memory is None:
            raise FileNotFoundError("repair requires episode memory")
        outcome = run_repair_agent(
            model_settings=config.api.repair,
            system_prompt=system_prompt,
            session=session,
            effective_glossary=effective_glossary,
            episode_memory={"story_description": memory.story_description},
        )
        repaired_by_id = {pair.id: pair for pair in outcome.current_pairs}
        pairs = build_pairs_from_ass_lines(current_lines)
        for pair in pairs:
            pair.chinese = repaired_by_id[pair.id].chinese
        output = run_dir / "repair" / f"candidate-{attempt:03d}.ass"
        write_ass_file(output, render_ass_file(current_header, apply_pairs_to_ass_lines(current_lines, pairs)))
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        suggestion_payload = json.loads(
            session.latest_artifact_path("suggestions.json").read_text(encoding="utf-8")
        )
        issue_keys = {
            value["issue_id"]: value["key"] for value in suggestion_payload["suggestions"]
        }
        decision_payload = json.loads(
            session.latest_artifact_path("decisions.json").read_text(encoding="utf-8")
        )
        enriched_decisions = [
            {**entry, "issue_key": issue_keys.get(entry["issue_id"], "")}
            for entry in decision_payload["decisions"]
        ]
        atomic_write_json(
            decision_log_path,
            {"version": decision_payload["version"], "decisions": enriched_decisions},
        )
        atomic_copy(session.latest_artifact_path("coverage.json"), coverage_path)
        repair_state_path = state_dir / f"repair-state-{attempt:03d}.json"
        atomic_copy(state_file, repair_state_path)
        return {
            "artifact_path": output,
            "repair_state_path": repair_state_path,
            "history_path": session.latest_artifact_path("history.json"),
            "staged_path": session.latest_artifact_path("staged.json"),
            "exchanges_path": session.latest_artifact_path("exchanges.json"),
            "tool_steps_used": outcome.tool_steps_used - tool_steps_before,
            "full_sweeps_used": session.full_sweeps_completed - full_sweeps_before,
            "repair_attempts_total": session.repair_attempts_used,
            "tool_steps_total": session.tool_steps_used,
            "full_sweeps_total": session.full_sweeps_completed,
            "escalated": outcome.status == "escalate",
        }

    return run_repair


def _glossary_research_callable(config: AppConfig):
    """Build a lazy bounded research runner; credentials are read only on use."""

    def run_research(**kwargs: object):
        from .research import (
            ExaSearchClient,
            ResearchRun,
            build_model_finding_parser_factory,
            unavailable_research_report,
        )
        from .webfetch import FetchBudget, fetch_url

        candidates = tuple(cast(Any, kwargs["candidates"]))
        max_requests = int(cast(Any, kwargs["max_requests"]))
        state_path = Path(cast(Any, kwargs["state_path"]))
        candidate_keys = [candidate.key for candidate in candidates]
        initial_requests_used = 0
        if state_path.is_file():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                if (
                    type(saved) is dict
                    and saved.get("version") == 1
                    and saved.get("candidate_keys") == candidate_keys
                    and type(saved.get("requests_used")) is int
                ):
                    initial_requests_used = min(saved["requests_used"], max_requests)
            except (OSError, json.JSONDecodeError):
                initial_requests_used = 0
        if not candidates:
            return unavailable_research_report((), "no_candidates", max_requests)
        key_path = config.research.exa_key_file
        if key_path is None:
            return unavailable_research_report(candidates, "exa_key_not_configured", max_requests)
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except OSError:
            return unavailable_research_report(candidates, "exa_key_unavailable", max_requests)
        if not api_key:
            return unavailable_research_report(candidates, "exa_key_empty", max_requests)
        settings = dataclasses.replace(config.research, max_requests=max_requests)
        with ExaSearchClient(api_key, settings) as searcher:
            runner = ResearchRun(
                searcher,
                settings,
                finding_parser_factory=build_model_finding_parser_factory(config.api.repair),
                page_fetcher=lambda url: fetch_url(
                    url,
                    FetchBudget(timeout=settings.timeout, max_bytes=settings.max_response_bytes),
                ),
                initial_requests_used=initial_requests_used,
                on_request_count=lambda count: atomic_write_json(
                    state_path,
                    {
                        "version": 1,
                        "candidate_keys": candidate_keys,
                        "requests_used": count,
                    },
                ),
            )
            return runner.research_candidates(
                candidates,
                authoritative_terms=cast(Any, kwargs["authoritative_terms"]),
            )

    return run_research


def _glossary_repair_callable(config: AppConfig):
    """One fresh, bounded terminology-only action pass before glossary freeze."""

    def run_glossary_repair(**kwargs: object) -> dict[str, object]:
        from .providers import build_chat_model, invoke_text

        candidates = list(cast(Any, kwargs["candidates"]))
        max_attempts = int(cast(Any, kwargs["max_attempts"]))
        if not candidates or max_attempts <= 0:
            return {"attempts_used": 0, "actions": []}
        payload = {
            "task": "Decide each invalid glossary candidate without editing subtitles.",
            "candidates": candidates,
            "cues": cast(Any, kwargs["cues"]),
            "authoritative": cast(Any, kwargs["authoritative"]),
            "validation": cast(Any, kwargs["validation"]),
            "actions": ["accept", "correct", "drop", "escalate"],
            "rules": [
                "Return exactly one JSON object with an actions array.",
                "Return exactly one action per candidate eng.",
                "accept reuses the input candidate; correct supplies candidate with eng, zh, type, confidence, evidence_ids.",
                "Never override authoritative terminology; uncertain cases escalate.",
            ],
        }
        try:
            model = build_chat_model(config.api.repair.config)
            text, _ = invoke_text(
                model,
                (("human", json.dumps(payload, ensure_ascii=False, sort_keys=True)),),
            )
            response = json.loads(text)
            if type(response) is not dict or set(response) != {"actions"}:
                raise ValueError("glossary repair response must contain only actions")
            actions = response["actions"]
            if type(actions) is not list or len(actions) != len(candidates):
                raise ValueError("glossary repair must decide every candidate once")
            expected = {entry["eng"].casefold() for entry in candidates}
            seen: set[str] = set()
            normalized: list[dict[str, object]] = []
            for action in actions:
                if type(action) is not dict:
                    raise ValueError("glossary repair action must be an object")
                name = action.get("action")
                eng = action.get("eng")
                reason = action.get("reason")
                if name not in {"accept", "correct", "drop", "escalate"}:
                    raise ValueError("glossary repair action is unsupported")
                if type(eng) is not str or eng.casefold() not in expected or eng.casefold() in seen:
                    raise ValueError("glossary repair action candidate is invalid")
                if type(reason) is not str or not reason.strip():
                    raise ValueError("glossary repair action reason is required")
                item: dict[str, object] = {"action": name, "eng": eng, "reason": reason}
                if name == "correct":
                    corrected = action.get("candidate")
                    if type(corrected) is not dict:
                        raise ValueError("correct action requires candidate")
                    item["candidate"] = corrected
                seen.add(eng.casefold())
                normalized.append(item)
            return {"attempts_used": 1, "actions": normalized}
        except Exception:
            return {
                "attempts_used": 1,
                "actions": [
                    {
                        "action": "escalate",
                        "eng": entry["eng"],
                        "reason": "glossary_repair_model_failed",
                    }
                    for entry in candidates
                ],
            }

    return run_glossary_repair


def _budget_limits(config: AppConfig) -> dict[str, int]:
    return {
        "repair_attempts": _repair_setting(config, "max_repair_attempts", 1),
        "tool_steps": _repair_setting(config, "max_tool_steps", 0),
        "full_sweeps": _repair_setting(config, "max_full_sweeps", 1),
        "glossary_repairs": _repair_setting(config, "max_glossary_repair_attempts", 0),
        "research_requests": int(getattr(getattr(config, "research", None), "max_requests", 0)),
    }


def _handlers(context: RunContext):
    config = context.config
    limits = _budget_limits(config)
    return build_stage_handlers(
        WorkflowSettings(
            run_dir=context.run_dir,
            primer_batch_size=config.primer.batch_size,
            primer_max_workers=config.primer.max_workers,
            qa_batch_size=config.qa.batch_size,
            qa_max_workers=config.qa.max_workers,
            qa_window_offsets=config.qa.window_offsets,
            postprocess_operations=config.postprocess.operations,
            episode_replacements=config.postprocess.episode_replacements,
            max_tool_steps=limits["tool_steps"],
            max_full_sweeps=limits["full_sweeps"],
            max_repair_attempts=limits["repair_attempts"],
            max_group_span=_repair_setting(config, "max_group_span", 3),
        ),
        preprocess_subtitle=_subtitle_preprocessor(config),
        translate_batch=_lazy_translate_batch(config),
        refine=_refine_callable(config),
        agent_qa=build_agent_qa(config.api.agent, load_qa_prompt_template(config.prompts)),
        glossary_research_runner=_glossary_research_callable(config),
        glossary_repair_runner=_glossary_repair_callable(config),
        repair_runner=_repair_callable(config),
    )


@contextmanager
def _open_graph(context: RunContext) -> Iterator[Any]:
    with SqliteSaver.from_conn_string(str(context.config.pipeline.checkpoint_db)) as saver:
        yield build_pipeline(_handlers(context), checkpointer=saver)


def _existing_run(args: argparse.Namespace, command: str) -> RunContext:
    thread_id = _validate_thread_id(args.thread_id)
    config = load_config(args.config)
    run_dir = config.pipeline.state_dir / thread_id
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no pipeline run named '{thread_id}' under {config.pipeline.state_dir}"
        )
    manifest = load_manifest(manifest_path, verify_artifacts=False)
    if Path(manifest["configuration"]["path"]) != config.path:
        raise ValueError(f"{command} config does not match the run config")
    if manifest["configuration"]["sha256"] != sha256_file(config.path):
        raise ValueError(f"{command} config content changed since the run started")
    for name in ("shared", "refine", "qa", "repair"):
        prompt = manifest["configuration"]["prompts"][name]
        path = Path(getattr(config.prompts, name)).resolve()
        if Path(prompt["path"]) != path or prompt["sha256"] != sha256_file(path):
            raise ValueError(f"{command} {name} prompt changed since the run started")
    return RunContext(thread_id, config, run_dir, manifest["translation_mode"])


def _report_review_wait(context: RunContext, result: Mapping[str, Any]) -> int:
    if "__interrupt__" not in result:
        raise RuntimeError("pipeline did not stop for human review")
    manifest = load_manifest(context.manifest_path, verify_artifacts=False)
    print(f"Pipeline '{context.thread_id}' is awaiting human review.")
    print(f"Review artifact: {manifest['review']['path']}")
    print(f"Reason: {manifest['route_reason']}")
    print(
        "Next: ./run.sh pipeline review "
        f"{context.thread_id} approve --config {context.config.path}"
    )
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    thread_id = _validate_thread_id(args.thread_id)
    config = load_config(args.config)
    source_path = Path(args.input).resolve()
    release_path = Path(args.output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"input artifact not found: {source_path}")
    config.pipeline.state_dir.mkdir(parents=True, exist_ok=True)
    config.pipeline.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    run_dir = config.pipeline.state_dir / thread_id
    run_dir.mkdir(parents=False, exist_ok=False)
    context = RunContext(thread_id, config, run_dir, args.mode)
    initial_state = create_manifest(
        context.manifest_path,
        run_id=thread_id,
        translation_mode=args.mode,
        source_path=source_path,
        config_path=config.path,
        prompt_paths={
            name: Path(getattr(config.prompts, name))
            for name in ("shared", "refine", "qa", "repair")
        },
        release_path=release_path,
        budget_limits=_budget_limits(config),
    )
    logger.info("Pipeline %s starting; manifest=%s", thread_id, context.manifest_path)
    logger.info("Prompt fingerprint: %s", _prompt_version(config))
    with _open_graph(context) as graph:
        result = graph.invoke(initial_state, context.graph_config)
    return _report_review_wait(context, result)


def review_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "review")
    with _open_graph(context) as graph:
        result = graph.invoke(Command(resume=args.decision), context.graph_config)
    manifest = load_manifest(context.manifest_path, verify_artifacts=False)
    print(f"Pipeline '{context.thread_id}' review decision: {args.decision}")
    if args.decision == "approve":
        print(f"Released artifact: {manifest['release']['path']}")
    else:
        print(f"Not released: {manifest['release']['path']}")
    return 0


def resume_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "resume")
    with _open_graph(context) as graph:
        result = graph.invoke(None, context.graph_config)
    return _report_review_wait(context, result)


def status_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "status")
    manifest = load_manifest(context.manifest_path, verify_artifacts=False)
    print(f"Pipeline '{context.thread_id}' ({context.translation_mode})")
    print(f"Run state: {context.run_dir}")
    print(f"Manifest revision: {manifest['revision']}")
    print(f"Next stage: {manifest['next_stage']}")
    print(f"Reason: {manifest['route_reason']}")
    print(f"Review artifact: {manifest['review']['path']}")
    print(f"Release artifact: {manifest['release']['path']}")
    print(f"Status: {manifest['review']['status']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent subtitle agent pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
        subparser.add_argument("--debug", action="store_true")

    run = subparsers.add_parser("run", help="start a new run and stop at human review")
    run.add_argument("input")
    run.add_argument("output")
    run.add_argument("--mode", choices=("parallel_initial", "serial_memory"), required=True)
    run.add_argument("--thread-id", required=True)
    add_common(run)
    run.set_defaults(func=run_pipeline)

    review = subparsers.add_parser("review", help="approve or reject the review artifact")
    review.add_argument("thread_id")
    review.add_argument("decision", choices=("approve", "reject"))
    add_common(review)
    review.set_defaults(func=review_pipeline)

    resume = subparsers.add_parser("resume", help="continue a failed run from its checkpoint")
    resume.add_argument("thread_id")
    add_common(resume)
    resume.set_defaults(func=resume_pipeline)

    status = subparsers.add_parser("status", help="show authoritative run status")
    status.add_argument("thread_id")
    add_common(status)
    status.set_defaults(func=status_pipeline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.debug:
        for logger_name in ("httpx", "httpcore", "httpx2", "google_genai"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
