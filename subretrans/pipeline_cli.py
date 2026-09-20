"""Command-line entry point for the persistent subtitle agent pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .fsutil import atomic_write_json, sha256_file
from .model_agent import build_agent_qa
from .model_translation import build_model_translate_batch
from .pipeline import STAGES, build_pipeline
from .prompts import load_qa_prompt_template
from .refine import RefineOptions, load_refine_progress, refine_serial
from .stage_handlers import Refine, WorkflowSettings, build_stage_handlers
from .state import PipelineState, TranslationMode
from .subtitle_edit import preprocess_with_seconv
from .translation import TranslateBatch, TranslationBatch


logger = logging.getLogger(__name__)

RUN_METADATA_VERSION = 2
RUN_METADATA_FIELDS = {
    "version",
    "config_path",
    "review_path",
    "release_path",
    "translation_mode",
}


@dataclass(frozen=True)
class RunContext:
    """One persisted pipeline run: its thread, configuration, and artifact paths."""

    thread_id: str
    config: AppConfig
    run_dir: Path
    review_path: Path
    release_path: Path
    translation_mode: TranslationMode

    @property
    def graph_config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}


def _validate_thread_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("thread id must use letters, digits, dot, underscore, or dash")
    if value in {".", ".."}:
        raise ValueError("invalid thread id")
    return value


def _load_run_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if type(payload) is not dict or set(payload) != RUN_METADATA_FIELDS:
        raise ValueError(f"invalid run metadata: {path}")
    if payload["version"] != RUN_METADATA_VERSION:
        raise ValueError(f"run metadata version must be {RUN_METADATA_VERSION}")
    if payload["translation_mode"] not in {"parallel_initial", "serial_memory"}:
        raise ValueError("invalid run translation mode")
    for field in ("config_path", "review_path", "release_path"):
        if type(payload[field]) is not str or not payload[field]:
            raise ValueError(f"run metadata {field} must be a non-empty string")
    return payload


def _prompt_version(config: AppConfig) -> str:
    """Hash the configuration file and every composed prompt component."""

    digest = hashlib.sha256()
    for path in (config.path, config.prompts.shared, config.prompts.refine, config.prompts.qa):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _lazy_translate_batch(config: AppConfig) -> TranslateBatch:
    """Construct the primer model on first use so serial runs never touch it."""

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


def _handlers(context: RunContext):
    config = context.config
    return build_stage_handlers(
        WorkflowSettings(
            run_dir=context.run_dir,
            review_path=context.review_path,
            release_path=context.release_path,
            primer_batch_size=config.primer.batch_size,
            primer_max_workers=config.primer.max_workers,
            qa_batch_size=config.qa.batch_size,
            qa_max_workers=config.qa.max_workers,
            qa_window_offsets=config.qa.window_offsets,
            agent_max_repair_attempts=config.pipeline.agent_max_repair_attempts,
            postprocess_operations=config.postprocess.operations,
            episode_replacements=config.postprocess.episode_replacements,
        ),
        preprocess_subtitle=_subtitle_preprocessor(config),
        translate_batch=_lazy_translate_batch(config),
        refine=_refine_callable(config),
        agent_qa=build_agent_qa(config.api.agent, load_qa_prompt_template(config.prompts)),
    )


def _inspection_handlers():
    """No-op handlers: enough to compile the graph and read persisted state."""

    return {stage: (lambda state: {}) for stage in STAGES}


@contextmanager
def _open_graph(context: RunContext, handlers) -> Iterator[Any]:
    with SqliteSaver.from_conn_string(str(context.config.pipeline.checkpoint_db)) as saver:
        yield build_pipeline(handlers, checkpointer=saver)


def _existing_run(args: argparse.Namespace, command: str) -> RunContext:
    thread_id = _validate_thread_id(args.thread_id)
    config = load_config(args.config)
    run_dir = config.pipeline.state_dir / thread_id
    metadata_path = run_dir / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"no pipeline run named '{thread_id}' under {config.pipeline.state_dir}"
        )
    metadata = _load_run_metadata(metadata_path)
    if Path(metadata["config_path"]) != config.path:
        raise ValueError(f"{command} config does not match the run config")
    return RunContext(
        thread_id=thread_id,
        config=config,
        run_dir=run_dir,
        review_path=Path(metadata["review_path"]),
        release_path=Path(metadata["release_path"]),
        translation_mode=metadata["translation_mode"],
    )


def _report_review_wait(context: RunContext, result: Mapping[str, Any]) -> int:
    if "__interrupt__" not in result:
        raise RuntimeError("pipeline did not stop for human review")
    print(f"Pipeline '{context.thread_id}' is awaiting human review.")
    print(f"Review artifact: {result['artifact_path']}")
    print(f"QA: {result['qa_conclusion']}")
    print(f"Agent repair attempts: {result['agent_repair_attempts']}")
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
    review_path = source_path.parent / f"{release_path.stem}.review{release_path.suffix}"
    if not source_path.is_file():
        raise FileNotFoundError(f"input artifact not found: {source_path}")
    release_path.parent.mkdir(parents=True, exist_ok=True)

    config.pipeline.state_dir.mkdir(parents=True, exist_ok=True)
    config.pipeline.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    run_dir = config.pipeline.state_dir / thread_id
    run_dir.mkdir(parents=False, exist_ok=False)
    mode: TranslationMode = args.mode
    context = RunContext(thread_id, config, run_dir, review_path, release_path, mode)
    model_versions = config.model_versions
    logger.info("Pipeline %s starting in %s mode", thread_id, mode)
    logger.info("Input: %s", source_path)
    logger.info("Run state: %s", run_dir)
    logger.info("Review artifact: %s", review_path)
    logger.info("Release artifact: %s", release_path)
    logger.info(
        "Models: primer=%s, refine=%s, extraction=%s, agent=%s",
        model_versions["primer"],
        model_versions["refine"],
        model_versions["extraction"],
        model_versions["agent"],
    )
    logger.info(
        "Batching: primer=%d cues, primer_workers=%d, refine=%s",
        config.primer.batch_size,
        config.primer.max_workers,
        config.refine.batch_size or "token-based",
    )
    atomic_write_json(
        run_dir / "run.json",
        {
            "version": RUN_METADATA_VERSION,
            "config_path": str(config.path),
            "review_path": str(review_path),
            "release_path": str(release_path),
            "translation_mode": mode,
        },
    )

    initial_state: PipelineState = {
        "artifact_path": str(source_path),
        "artifact_hash": sha256_file(source_path),
        "translation_manifest_path": None,
        "translation_mode": mode,
        "stage": "preprocess",
        "refine_chunk_cursor": 0,
        "memory_checkpoint_path": None,
        "memory_hash": "",
        "model_versions": model_versions,
        "prompt_version": _prompt_version(config),
        "qa_conclusion": "pending",
        "qa_passed": False,
        "qa_repair_applied": False,
        "agent_repair_attempts": 0,
    }
    with _open_graph(context, _handlers(context)) as graph:
        result = graph.invoke(initial_state, context.graph_config)
    return _report_review_wait(context, result)


def review_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "review")
    logger.info(
        "Pipeline %s review decision=%s; review artifact=%s",
        context.thread_id,
        args.decision,
        context.review_path,
    )
    with _open_graph(context, _handlers(context)) as graph:
        result = graph.invoke(Command(resume=args.decision), context.graph_config)
    print(f"Pipeline '{context.thread_id}' review decision: {args.decision}")
    if args.decision == "approve":
        print(f"Released artifact: {result['artifact_path']}")
    else:
        print(f"Not released: {context.release_path}")
    return 0


def resume_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "resume")
    logger.info("Pipeline %s resuming from its latest checkpoint", context.thread_id)
    with _open_graph(context, _handlers(context)) as graph:
        result = graph.invoke(None, context.graph_config)
    return _report_review_wait(context, result)


def status_pipeline(args: argparse.Namespace) -> int:
    context = _existing_run(args, "status")
    with _open_graph(context, _inspection_handlers()) as graph:
        snapshot = graph.get_state(context.graph_config)
    values = snapshot.values
    print(f"Pipeline '{context.thread_id}' ({context.translation_mode})")
    print(f"Config: {context.config.path}")
    print(f"Run state: {context.run_dir}")
    print(f"Review artifact: {context.review_path}")
    print(f"Release artifact: {context.release_path}")
    if not values:
        print("Status: no checkpoint recorded")
        return 0
    print(f"Stage: {values['stage']}")
    print(f"Artifact: {values['artifact_path']}")
    print(f"QA: {values['qa_conclusion']} (passed={values['qa_passed']})")
    print(f"Agent repair attempts: {values['agent_repair_attempts']}")
    if snapshot.interrupts:
        status = "awaiting human review; run `pipeline review <thread_id> approve|reject`"
    elif snapshot.next:
        status = (
            f"stopped before {', '.join(snapshot.next)}; run `pipeline resume <thread_id>`"
        )
    elif values["stage"] == "release":
        status = "released"
    elif values["stage"] == "human_review":
        status = "rejected at human review"
    else:
        status = "finished"
    print(f"Status: {status}")
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

    status = subparsers.add_parser("status", help="show where a run currently stands")
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
