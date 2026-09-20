"""Command-line entry point for standalone serial subtitle refinement."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import INTERMEDIATE_REPRESENTATIONS, AppConfig, load_config
from .refine import RefineOptions, refine_serial, test_connection
from .stats import format_usage_report


logger = logging.getLogger(__name__)

DRY_RUN_PAIRS = 10
NOISY_LOGGERS = ("httpx", "httpcore", "google_genai", "urllib3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine bilingual (English-Chinese) ASS subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m subretrans.cli input.ass output.ass
  python -m subretrans.cli input.ass output.ass --stream -v
  python -m subretrans.cli input.ass output.ass --refine-batch-size 50 --max-chunks 2
  python -m subretrans.cli input.ass output.ass --checkpoint --resume 680

Model protocols and credentials come from the config.yaml api roles.
The output file is rewritten after every chunk.
""",
    )
    parser.add_argument("input", help="Input .ass subtitle file")
    parser.add_argument("output", help="Output .ass subtitle file")
    parser.add_argument("--config", help="YAML configuration path (default: repository config.yaml)")
    parser.add_argument("--stream", action="store_true", help="Stream model output to stdout")
    parser.add_argument("--model", help="Override the refine model name")
    parser.add_argument(
        "--dry-run", action="store_true", help=f"Process only the first {DRY_RUN_PAIRS} pairs"
    )
    parser.add_argument("--max-chunks", type=int, help="Process at most N chunks")
    parser.add_argument("--memory-limit", type=int, help="Override refine.memory_token_limit")
    parser.add_argument(
        "--refine-batch-size", type=int, help="Pairs per chunk (overrides token-based chunking)"
    )
    parser.add_argument(
        "--intermediate-representation",
        choices=INTERMEDIATE_REPRESENTATIONS,
        help="Override refine.intermediate_representation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v shows progress (default), -vv also logs prompts and raw responses",
    )
    parser.add_argument("--test-connection", action="store_true", help="Test the refine API and exit")
    parser.add_argument("--resume", type=int, metavar="INDEX", help="Resume from pair INDEX")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Persist episode memory beside the input as <input>.memory.yaml",
    )
    parser.add_argument("--checkpoint-path", help="Explicit memory checkpoint path")
    parser.add_argument(
        "--progress-manifest", help="Progress JSON path (requires a memory checkpoint)"
    )
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides to the immutable configuration."""

    refine = config.refine
    if args.memory_limit is not None:
        refine = replace(refine, memory_token_limit=args.memory_limit)
    if args.refine_batch_size is not None:
        refine = replace(refine, batch_size=args.refine_batch_size)
    if args.intermediate_representation is not None:
        refine = replace(refine, intermediate_representation=args.intermediate_representation)
    api = config.api
    if args.model:
        api = replace(api, refine=replace(api.refine, model=args.model))
    return replace(config, refine=refine, api=api)


def configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbosity >= 2 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbosity < 2:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def _write_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = apply_overrides(load_config(args.config), args)
    except (OSError, ValueError) as error:
        logger.error("Configuration error: %s", error)
        return 1

    if args.test_connection:
        if test_connection(config):
            print("API connection successful")
            return 0
        print("API connection failed")
        return 1

    if args.checkpoint_path:
        checkpoint_path: Path | None = Path(args.checkpoint_path)
    elif args.checkpoint:
        checkpoint_path = Path(f"{args.input}.memory.yaml")
    else:
        checkpoint_path = None
    progress_path = Path(args.progress_manifest) if args.progress_manifest else None
    options = RefineOptions(
        stream=args.stream,
        on_stream_chunk=_write_stdout if args.stream else None,
        resume_index=args.resume,
        dry_run_pairs=DRY_RUN_PAIRS if args.dry_run else None,
        max_chunks=args.max_chunks,
    )
    try:
        result = refine_serial(
            Path(args.input),
            Path(args.output),
            config,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            options=options,
        )
    except Exception:
        logger.exception("Refinement failed")
        return 1

    cost = result.cost.cost if result.cost is not None else None
    print()
    print(format_usage_report(result.usage, cost, title="REFINE TOKEN USAGE"))
    if result.extraction_usage.total_tokens:
        print(format_usage_report(result.extraction_usage, title="MEMORY UPDATE TOKEN USAGE"))
    if result.cost is not None:
        pricing = result.cost.pricing
        print(
            f"CCH pricing: {pricing.model_name} via {pricing.provider}; "
            f"table {pricing.version} ({pricing.refreshed_at})"
        )
    print(f"Refined {result.committed_pairs}/{result.total_pairs} pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
