"""Serial, memory-aware subtitle refinement shared by the pipeline and the CLI.

One run parses a bilingual ASS file, refines it chunk by chunk with the
``refine`` model while the ``extraction`` model maintains episode memory, and
commits after every chunk in this order: refined artifact, memory checkpoint,
progress manifest. A run can therefore be resumed from its progress file.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .ass_parser import (
    apply_pairs_to_ass_lines,
    build_pairs_from_ass_lines,
    parse_ass_file,
    render_ass_file,
    write_ass_file,
)
from .chunker import chunk_pairs, format_chunk_statistics
from .config import AppConfig
from .fsutil import atomic_write_json, require_distinct_paths, require_exact_fields, sha256_file
from .memory import (
    GlobalMemory,
    compress_memory,
    estimate_memory_tokens,
    load_memory_checkpoint,
    save_memory_checkpoint,
    set_user_glossary,
    update_global_memory,
)
from .pairs import SubtitlePair
from .pricing import CostEstimate, estimate_cost
from .prompts import (
    build_refine_system_prompt,
    load_refine_prompt_template,
    parse_authoritative_glossary,
)
from .providers import build_chat_model, clean_response_text, invoke_text
from .serializers import (
    SerializationError,
    deserialize,
    deserialize_best_effort,
    extract_from_format_marker,
    serialize,
)
from .stats import UsageStats
from .utils import estimate_tokens, format_time


logger = logging.getLogger(__name__)

REFINE_PROGRESS_VERSION = 1
_PROGRESS_FIELDS = {
    "version",
    "next_pair",
    "artifact_path",
    "artifact_hash",
    "memory_checkpoint_path",
    "memory_hash",
}
_MISSING_ID_PREVIEW = 20


@dataclass(frozen=True)
class RefineOptions:
    """Run-mode switches that do not belong in the YAML configuration."""

    stream: bool = False
    on_stream_chunk: Callable[[str], None] | None = None
    resume_index: int | None = None
    dry_run_pairs: int | None = None
    max_chunks: int | None = None


@dataclass(frozen=True)
class RefineProgress:
    """The committed recovery point of a serial refinement run."""

    next_pair: int
    artifact_path: Path
    artifact_hash: str
    memory_checkpoint_path: Path
    memory_hash: str


@dataclass(frozen=True)
class RefineResult:
    total_pairs: int
    committed_pairs: int
    chunks: int
    usage: UsageStats
    extraction_usage: UsageStats
    cost: CostEstimate | None = None


@dataclass(frozen=True)
class ParsedChunk:
    """Corrected pairs recovered from one model response, with diagnostics."""

    pairs: list[SubtitlePair]
    warnings: tuple[str, ...]
    missing_ids: tuple[int, ...]


# Progress manifest ----------------------------------------------------------


def load_refine_progress(
    progress_path: Path, output_path: Path, checkpoint_path: Path
) -> RefineProgress:
    """Strictly load a progress manifest bound to the current output and memory files."""

    progress_path, output_path, checkpoint_path = (
        Path(progress_path),
        Path(output_path),
        Path(checkpoint_path),
    )
    with progress_path.open(encoding="utf-8") as handle:
        payload = require_exact_fields(
            json.load(handle), _PROGRESS_FIELDS, location="refine progress"
        )
    if type(payload["version"]) is not int or payload["version"] != REFINE_PROGRESS_VERSION:
        raise ValueError(f"refine progress version must be {REFINE_PROGRESS_VERSION}")
    next_pair = payload["next_pair"]
    if type(next_pair) is not int or next_pair < 0:
        raise ValueError("refine progress next_pair must be a non-negative integer")
    if payload["artifact_path"] != str(output_path):
        raise ValueError("refine progress artifact_path does not match refined output")
    if payload["memory_checkpoint_path"] != str(checkpoint_path):
        raise ValueError("refine progress memory_checkpoint_path does not match memory output")
    artifact_hash = sha256_file(output_path)
    memory_hash = sha256_file(checkpoint_path)
    if payload["artifact_hash"] != artifact_hash:
        raise ValueError("refine progress artifact_hash does not match refined output")
    if payload["memory_hash"] != memory_hash:
        raise ValueError("refine progress memory_hash does not match memory output")
    return RefineProgress(next_pair, output_path, artifact_hash, checkpoint_path, memory_hash)


def save_refine_progress(
    progress_path: Path, *, next_pair: int, output_path: Path, checkpoint_path: Path
) -> RefineProgress:
    """Atomically commit the recovery point after artifact and memory are on disk."""

    progress = RefineProgress(
        next_pair=next_pair,
        artifact_path=Path(output_path),
        artifact_hash=sha256_file(output_path),
        memory_checkpoint_path=Path(checkpoint_path),
        memory_hash=sha256_file(checkpoint_path),
    )
    atomic_write_json(
        progress_path,
        {
            "version": REFINE_PROGRESS_VERSION,
            "next_pair": progress.next_pair,
            "artifact_path": str(progress.artifact_path),
            "artifact_hash": progress.artifact_hash,
            "memory_checkpoint_path": str(progress.memory_checkpoint_path),
            "memory_hash": progress.memory_hash,
        },
    )
    return progress


# Response parsing -----------------------------------------------------------


def _deserialize_with_recovery(
    cleaned: str, representation: str, warnings: list[str]
) -> list[SubtitlePair]:
    try:
        return deserialize(cleaned, representation)
    except SerializationError as strict_error:
        warnings.append(f"strict {representation} parse failed: {strict_error}")
        extracted = extract_from_format_marker(cleaned, representation)
        if extracted is not None and extracted != cleaned:
            try:
                pairs = deserialize(extracted, representation)
            except SerializationError as retry_error:
                warnings.append(f"parse after marker extraction failed: {retry_error}")
            else:
                warnings.append("recovered by extracting from the first format marker")
                return pairs
        recovered, errors = deserialize_best_effort(
            extracted if extracted is not None else cleaned, representation
        )
        if recovered:
            warnings.append(
                f"best-effort recovery salvaged {len(recovered)} pair(s), "
                f"skipped {len(errors)} malformed"
            )
            return recovered
        raise ValueError(
            f"failed to parse {representation} response: {strict_error}; "
            f"excerpt: {cleaned[:300]!r}"
        ) from strict_error


def _deduplicate(pairs: list[SubtitlePair], warnings: list[str]) -> list[SubtitlePair]:
    last_by_id = {pair.id: pair for pair in pairs}
    if len(last_by_id) == len(pairs):
        return pairs
    duplicates = sorted(
        pair_id for pair_id in last_by_id if sum(1 for pair in pairs if pair.id == pair_id) > 1
    )
    warnings.append(f"duplicate pair ids {duplicates}; kept the last occurrence of each")
    seen: set[int] = set()
    ordered: list[SubtitlePair] = []
    for pair in pairs:
        if pair.id not in seen:
            seen.add(pair.id)
            ordered.append(last_by_id[pair.id])
    return ordered


def _align_ids(
    pairs: list[SubtitlePair], expected: Sequence[SubtitlePair], warnings: list[str]
) -> list[SubtitlePair]:
    """Map local ids back onto the chunk's ids, or drop ids that cannot belong to it."""

    expected_ids = [pair.id for pair in expected]
    expected_set = set(expected_ids)
    returned_set = {pair.id for pair in pairs}
    if not pairs or returned_set <= expected_set:
        return pairs

    count = len(expected_ids)
    offset: int | None = None
    if all(0 <= pair_id < count for pair_id in returned_set):
        offset = 0
    elif all(1 <= pair_id <= count for pair_id in returned_set):
        offset = 1
    if offset is not None:
        warnings.append(
            f"model returned {offset}-based local ids; remapped to {expected_ids[0]}-{expected_ids[-1]}"
        )
        return [
            SubtitlePair(id=expected_ids[pair.id - offset], eng=pair.eng, chinese=pair.chinese)
            for pair in pairs
        ]

    kept = [pair for pair in pairs if pair.id in expected_set]
    if not kept:
        raise ValueError(
            "corrected pair ids do not match the chunk "
            f"(expected {min(expected_set)}-{max(expected_set)}, "
            f"got {min(returned_set)}-{max(returned_set)})"
        )
    warnings.append(f"dropped {len(pairs) - len(kept)} pair(s) with ids outside the chunk")
    return kept


def parse_refined_pairs(
    response_text: str, representation: str, expected_pairs: Sequence[SubtitlePair]
) -> ParsedChunk:
    """Recover corrected pairs from a model response without touching any file.

    Cleans thinking blocks and code fences, parses strictly, then falls back to
    format-marker extraction and best-effort salvage; duplicate ids keep their
    last occurrence and per-chunk local ids are remapped onto the chunk.
    """

    warnings: list[str] = []
    cleaned = clean_response_text(response_text)
    pairs = _deserialize_with_recovery(cleaned, representation, warnings)
    pairs = _deduplicate(pairs, warnings)
    pairs = _align_ids(pairs, expected_pairs, warnings)
    returned = {pair.id for pair in pairs}
    missing = tuple(pair.id for pair in expected_pairs if pair.id not in returned)
    if missing:
        preview = ", ".join(str(pair_id) for pair_id in missing[:_MISSING_ID_PREVIEW])
        suffix = "..." if len(missing) > _MISSING_ID_PREVIEW else ""
        warnings.append(f"missing {len(missing)}/{len(expected_pairs)} pair(s): {preview}{suffix}")
    return ParsedChunk(pairs, tuple(warnings), missing)


# Run ------------------------------------------------------------------------


def _apply_corrections(pairs: list[SubtitlePair], corrected: Sequence[SubtitlePair]) -> None:
    by_id = {pair.id: pair for pair in pairs}
    for correction in corrected:
        target = by_id[correction.id]
        target.eng = correction.eng
        target.chinese = correction.chinese


def _preserve_committed_pairs(
    pairs: list[SubtitlePair], output_path: Path, resume_index: int
) -> None:
    """Copy already-refined text for ``pairs[:resume_index]`` from the existing output."""

    if not output_path.is_file():
        raise ValueError(f"resume requires the existing refined output: {output_path}")
    _, existing_lines = parse_ass_file(str(output_path))
    existing = build_pairs_from_ass_lines(existing_lines)
    if len(existing) != len(pairs):
        raise ValueError(
            f"existing output has {len(existing)} pairs but the input has {len(pairs)}"
        )
    for index in range(resume_index):
        if existing[index].id != pairs[index].id:
            raise ValueError(f"existing output pair {index} does not align with the input")
        pairs[index].eng = existing[index].eng
        pairs[index].chinese = existing[index].chinese


def _write_output(output_path: Path, header: str, ass_lines: list, pairs: list[SubtitlePair]) -> None:
    write_ass_file(str(output_path), render_ass_file(header, apply_pairs_to_ass_lines(ass_lines, pairs)))


def _maybe_compress(
    memory: GlobalMemory, model: BaseChatModel, config: AppConfig
) -> tuple[GlobalMemory, UsageStats]:
    model_name = config.api.refine.model
    limit = config.refine.memory_token_limit
    tokens = estimate_memory_tokens(memory, model_name)
    if tokens <= limit:
        return memory, UsageStats()
    logger.info("Memory size %d tokens exceeds limit %d; compressing", tokens, limit)
    try:
        compressed, usage = compress_memory(memory, model=model, target_tokens=limit)
    except Exception as error:
        logger.warning("Memory compression failed; continuing uncompressed: %s", error)
        return memory, UsageStats()
    logger.info("Memory compressed: %d -> %d tokens", tokens, estimate_memory_tokens(compressed, model_name))
    return compressed, usage


def refine_serial(
    input_path: Path,
    output_path: Path,
    config: AppConfig,
    *,
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    options: RefineOptions = RefineOptions(),
) -> RefineResult:
    """Refine ``input_path`` into ``output_path`` chunk by chunk with episode memory."""

    input_path, output_path = Path(input_path), Path(output_path)
    require_distinct_paths(input_path, output_path)
    if progress_path is not None and checkpoint_path is None:
        raise ValueError("progress tracking requires a memory checkpoint path")
    if options.dry_run_pairs is not None and options.dry_run_pairs <= 0:
        raise ValueError("dry_run_pairs must be a positive integer")
    if options.max_chunks is not None and options.max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")

    representation = config.refine.intermediate_representation
    model_name = config.api.refine.model
    header, ass_lines = parse_ass_file(str(input_path))
    pairs = build_pairs_from_ass_lines(ass_lines)
    if not pairs:
        raise ValueError(f"no subtitle pairs found in {input_path}")
    logger.info("Refine: %d pairs from %s (model=%s, representation=%s)", len(pairs), input_path, model_name, representation)

    resume_index = options.resume_index or 0
    if resume_index < 0 or resume_index > len(pairs):
        raise ValueError(f"resume_index must be between 0 and {len(pairs)}, got {resume_index}")
    if resume_index == len(pairs):
        logger.info("Refine: all %d pairs already committed; nothing to do", len(pairs))
        return RefineResult(len(pairs), len(pairs), 0, UsageStats(), UsageStats())
    if resume_index:
        _preserve_committed_pairs(pairs, output_path, resume_index)
        logger.info("Refine: resuming at pair %d, preserved %d refined pairs", resume_index, resume_index)
    pending = pairs[resume_index:]
    if options.dry_run_pairs is not None:
        pending = pending[: options.dry_run_pairs]
        logger.info("Refine: dry run limited to %d pairs", len(pending))

    user_glossary = parse_authoritative_glossary(load_refine_prompt_template(config.prompts))
    memory = GlobalMemory()
    if checkpoint_path is not None:
        loaded = load_memory_checkpoint(checkpoint_path)
        if loaded is not None:
            memory = loaded
            logger.info(
                "Refine: loaded memory checkpoint %s (%d learned terms, story=%s)",
                checkpoint_path,
                len(memory.glossary),
                "present" if memory.story_description else "empty",
            )
    pruned = set_user_glossary(memory, user_glossary)
    if pruned:
        logger.info("Refine: pruned %d learned terms covered by the user glossary", pruned)

    refine_model = build_chat_model(config.api.refine.config)
    extraction_model = build_chat_model(config.api.extraction.config)
    base_prompt_tokens = estimate_tokens(
        build_refine_system_prompt(memory, config.prompts, representation), model_name
    )
    chunks = chunk_pairs(
        pending,
        batch_size=config.refine.batch_size,
        token_soft_limit=config.refine.chunk_token_soft_limit,
        base_prompt_tokens=base_prompt_tokens,
        model_name=model_name,
    )
    if options.max_chunks is not None:
        chunks = chunks[: options.max_chunks]
    logger.info("Refine: base prompt %d tokens; %s", base_prompt_tokens, format_chunk_statistics(chunks, model_name))

    committed = resume_index
    usage = UsageStats()
    extraction_usage = UsageStats()
    for index, chunk in enumerate(chunks, start=1):
        started = time.monotonic()
        system_prompt = build_refine_system_prompt(memory, config.prompts, representation)
        if index == 1:
            logger.debug("Refine system prompt:\n%s", system_prompt)
        text, chunk_usage = invoke_text(
            refine_model,
            [("system", system_prompt), ("human", serialize(chunk, representation))],
            stream=options.stream,
            on_chunk=options.on_stream_chunk if options.stream else None,
        )
        usage += chunk_usage
        parsed = parse_refined_pairs(text, representation, chunk)
        for warning in parsed.warnings:
            logger.warning("Refine chunk %d/%d: %s", index, len(chunks), warning)

        _apply_corrections(pairs, parsed.pairs)
        _write_output(output_path, header, ass_lines, pairs)
        committed += len(chunk)

        memory, update_usage = update_global_memory(
            memory, parsed.pairs, model=extraction_model, settings=config.glossary
        )
        extraction_usage += update_usage
        memory, compression_usage = _maybe_compress(memory, refine_model, config)
        usage += compression_usage
        if checkpoint_path is not None:
            save_memory_checkpoint(memory, checkpoint_path)
        if progress_path is not None and checkpoint_path is not None:
            save_refine_progress(
                progress_path,
                next_pair=committed,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
            )
        logger.info(
            "Refine progress: chunk %d/%d committed; %d/%d pairs (%.1f%%); "
            "elapsed=%s; prompt_tokens=%d; completion_tokens=%d",
            index,
            len(chunks),
            committed,
            len(pairs),
            committed / len(pairs) * 100,
            format_time(time.monotonic() - started),
            chunk_usage.prompt_tokens,
            chunk_usage.completion_tokens,
        )

    logger.info("Refine complete: %d/%d pairs committed -> %s", committed, len(pairs), output_path)
    return RefineResult(
        total_pairs=len(pairs),
        committed_pairs=committed,
        chunks=len(chunks),
        usage=usage,
        extraction_usage=extraction_usage,
        cost=estimate_cost(model_name, usage),
    )


def test_connection(config: AppConfig) -> bool:
    """Send a trivial request to the refine role and report whether it answered."""

    try:
        text, _ = invoke_text(
            build_chat_model(config.api.refine.config), [("human", "Reply with just 'OK'")]
        )
    except Exception as error:
        logger.error("API connection test failed: %s", error)
        return False
    return "ok" in text.lower()
