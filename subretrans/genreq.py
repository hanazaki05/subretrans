"""Generate the exact refine prompts for an ASS file without calling any API."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .ass_parser import build_pairs_from_ass_lines, parse_ass_file
from .chunker import chunk_pairs
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .memory import GlobalMemory
from .prompts import (
    build_refine_system_prompt,
    load_refine_prompt_template,
    parse_authoritative_glossary,
)
from .serializers import serialize
from .utils import estimate_tokens


logger = logging.getLogger(__name__)


def generate_prompts(
    input_path: Path, config: AppConfig, max_chunks: int | None
) -> list[dict[str, Any]]:
    """Build the system and user prompt for every chunk plus token estimates."""

    _, ass_lines = parse_ass_file(str(input_path))
    pairs = build_pairs_from_ass_lines(ass_lines)
    if not pairs:
        raise ValueError(f"no subtitle pairs found in {input_path}")
    representation = config.refine.intermediate_representation
    model_name = config.api.refine.model
    memory = GlobalMemory(
        user_glossary=parse_authoritative_glossary(load_refine_prompt_template(config.prompts))
    )
    system_prompt = build_refine_system_prompt(memory, config.prompts, representation)
    system_tokens = estimate_tokens(system_prompt, model_name)
    chunks = chunk_pairs(
        pairs,
        batch_size=config.refine.batch_size,
        token_soft_limit=config.refine.chunk_token_soft_limit,
        base_prompt_tokens=system_tokens,
        model_name=model_name,
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    logger.info("Generating prompts for %d chunks of %d pairs", len(chunks), len(pairs))

    prompts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        user_prompt = serialize(chunk, representation)
        user_tokens = estimate_tokens(user_prompt, model_name)
        prompts.append(
            {
                "chunk_index": index,
                "chunk_size": len(chunk),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "total_tokens": system_tokens + user_tokens,
                "total_pairs": len(pairs),
            }
        )
    return prompts


def render_markdown(
    prompts: list[dict[str, Any]], config: AppConfig, input_name: str
) -> str:
    refine = config.api.refine
    total_pairs = prompts[0]["total_pairs"] if prompts else 0
    lines = [
        f"# Request Prompts for {input_name}",
        "",
        "## Configuration",
        "",
        f"- **Total pairs:** {total_pairs}",
        f"- **Pairs per chunk:** {config.refine.batch_size}",
        f"- **Total chunks:** {len(prompts)}",
        f"- **Intermediate representation:** {config.refine.intermediate_representation}",
        f"- **Model:** {refine.model}",
        f"- **Max output tokens:** {refine.max_output_tokens:,}",
        f"- **Temperature:** {refine.temperature}",
    ]
    if refine.reasoning_effort is not None:
        lines.append(f"- **Reasoning effort:** {refine.reasoning_effort}")
    lines += [
        "",
        "## Token Summary",
        "",
        "| Chunk | Pairs | System Tokens | User Tokens | Total Tokens |",
        "|-------|-------|---------------|-------------|-------------|",
    ]
    for prompt in prompts:
        lines.append(
            f"| {prompt['chunk_index'] + 1}/{len(prompts)} | {prompt['chunk_size']} | "
            f"{prompt['system_tokens']:,} | {prompt['user_tokens']:,} | {prompt['total_tokens']:,} |"
        )
    lines.append(
        f"| **Total** | {total_pairs} | {sum(p['system_tokens'] for p in prompts):,} | "
        f"{sum(p['user_tokens'] for p in prompts):,} | {sum(p['total_tokens'] for p in prompts):,} |"
    )
    lines += ["", "---", ""]
    for prompt in prompts:
        number = prompt["chunk_index"] + 1
        lines += [
            f"## Chunk {number}/{len(prompts)} ({prompt['chunk_size']} pairs)",
            "",
            "### System Prompt",
            "",
            "```",
            prompt["system_prompt"],
            "```",
            "",
            "### User Prompt",
            "",
            "```",
            prompt["user_prompt"],
            "```",
            "",
            "### Token Estimates",
            "",
            f"- **System prompt:** {prompt['system_tokens']:,} tokens",
            f"- **User content:** {prompt['user_tokens']:,} tokens",
            f"- **Total input:** {prompt['total_tokens']:,} tokens",
            f"- **Max output:** {refine.max_output_tokens:,} tokens",
            f"- **Estimated max total:** {prompt['total_tokens'] + refine.max_output_tokens:,} tokens",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate request prompts from an ASS file without calling any API"
    )
    parser.add_argument("input", help="Input .ass subtitle file")
    parser.add_argument(
        "--refine-batch-size", type=int, required=True, help="Pairs per chunk (required)"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Configuration YAML path")
    parser.add_argument("--output", help="Output markdown file (default: <input stem>_prompts.md)")
    parser.add_argument("--max-chunks", type=int, help="Generate prompts for at most N chunks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(f"{input_path.stem}_prompts.md")
    try:
        config = load_config(args.config)
        config = replace(config, refine=replace(config.refine, batch_size=args.refine_batch_size))
        prompts = generate_prompts(input_path, config, args.max_chunks)
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
    output_path.write_text(render_markdown(prompts, config, input_path.name), encoding="utf-8")
    print(f"Wrote {len(prompts)} chunk prompt(s) to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
