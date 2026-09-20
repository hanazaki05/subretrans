"""Command-line entry point for the persistent subtitle agent pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .ass_parser import build_pairs_from_ass_lines, parse_ass_file
from .cli import file_sha256, process_subtitles
from .config import load_config_sdk
from .model_translation import build_model_translate_batch
from .pipeline import build_pipeline
from .stage_handlers import WorkflowSettings, build_stage_handlers
from .state import PipelineState, TranslationMode
from .subtitle_edit import preprocess_with_seconv
from .translation import TranslateBatch, TranslationBatch
from .workflow_config import (
    PipelineSettings,
    load_pipeline_settings,
    load_subtitle_edit_settings,
    load_translation_model_settings,
)


RUN_METADATA_FIELDS = {
    "version",
    "config_path",
    "release_path",
    "translation_mode",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


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
    if payload["version"] != 1:
        raise ValueError("run metadata version must be 1")
    if payload["translation_mode"] not in {"parallel_initial", "serial_memory"}:
        raise ValueError("invalid run translation mode")
    for field in ("config_path", "release_path"):
        if type(payload[field]) is not str or not payload[field]:
            raise ValueError(f"run metadata {field} must be a non-empty string")
    return payload


def _configured_model_name(config_path: Path, mode: TranslationMode) -> str:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    section_name = "translation_model" if mode == "parallel_initial" else "main_model"
    section = payload.get(section_name)
    if not isinstance(section, dict) or not isinstance(section.get("name"), str):
        raise ValueError(f"{section_name}.name must be configured")
    return section["name"]


def _lazy_translate_batch(
    config_path: Path, settings: PipelineSettings
) -> TranslateBatch:
    lock = threading.Lock()
    translator: TranslateBatch | None = None

    def translate(batch: TranslationBatch):
        nonlocal translator
        if translator is None:
            with lock:
                if translator is None:
                    model = load_translation_model_settings(config_path)
                    translator = build_model_translate_batch(
                        model.config,
                        source_language=settings.source_language,
                        target_language=settings.target_language,
                        user_instruction=settings.user_instruction,
                    )
        return translator(batch)

    return translate


def _load_resume_index(progress_path: Path, output_path: Path, memory_path: Path) -> int:
    if not progress_path.exists():
        return 0
    with progress_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "version",
        "next_pair",
        "artifact_path",
        "artifact_hash",
        "memory_checkpoint_path",
        "memory_hash",
    }
    if type(payload) is not dict or set(payload) != expected or payload["version"] != 1:
        raise ValueError("invalid refine progress manifest")
    if payload["artifact_path"] != str(output_path):
        raise ValueError("refine progress artifact path mismatch")
    if payload["memory_checkpoint_path"] != str(memory_path):
        raise ValueError("refine progress memory path mismatch")
    if payload["artifact_hash"] != file_sha256(str(output_path)):
        raise ValueError("refine progress artifact hash mismatch")
    if payload["memory_hash"] != file_sha256(str(memory_path)):
        raise ValueError("refine progress memory hash mismatch")
    next_pair = payload["next_pair"]
    if type(next_pair) is not int or next_pair < 0:
        raise ValueError("refine progress next_pair must be a non-negative integer")
    return next_pair


def _refine_callable(config_path: Path):
    def refine(
        input_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        progress_path: Path,
    ) -> None:
        resume_index = _load_resume_index(progress_path, output_path, checkpoint_path)
        processing_input = input_path
        if resume_index:
            _, lines = parse_ass_file(str(output_path))
            if resume_index == len(build_pairs_from_ass_lines(lines)):
                return
            processing_input = output_path

        config = load_config_sdk(yaml_file_path=str(config_path))
        config.per_block_update = True
        success = process_subtitles(
            str(processing_input),
            str(output_path),
            config,
            api_mode=config.api_mode,
            use_stream=config.use_stream,
            resume_index=resume_index or None,
            enable_checkpoint=True,
            checkpoint_path_override=str(checkpoint_path),
            progress_manifest_path=str(progress_path),
        )
        if not success:
            raise RuntimeError("serial refinement failed")

    return refine


def _subtitle_preprocessor(config_path: Path):
    def preprocess(input_path: Path, output_path: Path) -> Path:
        settings = load_subtitle_edit_settings(config_path)
        return preprocess_with_seconv(settings, input_path, output_path)

    return preprocess


def _handlers(
    pipeline_settings: PipelineSettings,
    config_path: Path,
    run_dir: Path,
    release_path: Path,
):
    return build_stage_handlers(
        WorkflowSettings(
            run_dir=run_dir,
            release_path=release_path,
            batch_size=pipeline_settings.batch_size,
            max_workers=pipeline_settings.max_workers,
            episode_replacements=pipeline_settings.episode_replacements,
        ),
        preprocess_subtitle=_subtitle_preprocessor(config_path),
        translate_batch=_lazy_translate_batch(config_path, pipeline_settings),
        refine=_refine_callable(config_path),
    )


def run_pipeline(args: argparse.Namespace) -> int:
    thread_id = _validate_thread_id(args.thread_id)
    config_path = Path(args.config).resolve()
    source_path = Path(args.input).resolve()
    release_path = Path(args.output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"input artifact not found: {source_path}")
    release_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_settings = load_pipeline_settings(config_path)
    pipeline_settings.state_dir.mkdir(parents=True, exist_ok=True)
    pipeline_settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    run_dir = pipeline_settings.state_dir / thread_id
    run_dir.mkdir(parents=False, exist_ok=False)
    metadata_path = run_dir / "run.json"
    mode: TranslationMode = args.mode
    _atomic_json(
        metadata_path,
        {
            "version": 1,
            "config_path": str(config_path),
            "release_path": str(release_path),
            "translation_mode": mode,
        },
    )

    initial_state: PipelineState = {
        "artifact_path": str(source_path),
        "artifact_hash": file_sha256(str(source_path)),
        "translation_manifest_path": None,
        "translation_mode": mode,
        "stage": "preprocess",
        "refine_chunk_cursor": 0,
        "memory_checkpoint_path": None,
        "memory_hash": "",
        "model_version": _configured_model_name(config_path, mode),
        "prompt_version": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "qa_conclusion": "pending",
    }
    graph_config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(str(pipeline_settings.checkpoint_db)) as saver:
        graph = build_pipeline(
            _handlers(pipeline_settings, config_path, run_dir, release_path),
            checkpointer=saver,
        )
        result = graph.invoke(initial_state, graph_config)
    if "__interrupt__" not in result:
        raise RuntimeError("pipeline did not stop for human review")
    print(f"Pipeline '{thread_id}' is awaiting human review.")
    print(f"Review artifact: {result['artifact_path']}")
    print(f"QA: {result['qa_conclusion']}")
    return 0


def review_pipeline(args: argparse.Namespace) -> int:
    thread_id = _validate_thread_id(args.thread_id)
    config_path = Path(args.config).resolve()
    pipeline_settings = load_pipeline_settings(config_path)
    run_dir = pipeline_settings.state_dir / thread_id
    metadata = _load_run_metadata(run_dir / "run.json")
    if Path(metadata["config_path"]) != config_path:
        raise ValueError("review config does not match the run config")
    release_path = Path(metadata["release_path"])
    graph_config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(str(pipeline_settings.checkpoint_db)) as saver:
        graph = build_pipeline(
            _handlers(pipeline_settings, config_path, run_dir, release_path),
            checkpointer=saver,
        )
        result = graph.invoke(Command(resume=args.decision), graph_config)
    print(f"Pipeline '{thread_id}' review decision: {args.decision}")
    if args.decision == "approve":
        print(f"Released artifact: {result['artifact_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent subtitle agent pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("input")
    run.add_argument("output")
    run.add_argument(
        "--mode",
        choices=("parallel_initial", "serial_memory"),
        required=True,
    )
    run.add_argument("--thread-id", required=True)
    run.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    run.set_defaults(func=run_pipeline)

    review = subparsers.add_parser("review")
    review.add_argument("thread_id")
    review.add_argument("decision", choices=("approve", "reject"))
    review.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    review.set_defaults(func=review_pipeline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
