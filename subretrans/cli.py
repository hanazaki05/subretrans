#!/usr/bin/env python3
"""Command-line entry point for provider-neutral subtitle refinement."""

import argparse
import hashlib
import json
import sys
import os
import tempfile
import time
import yaml
from typing import Optional

# Import SDK-specific modules
from .config import load_config_sdk
from .providers import ModelProtocol
from .llm import (
    refine_chunk_sdk,
    refine_chunk_sdk_streaming,
    refine_chunk_sdk_response,
    compress_memory_sdk,
    test_api_connection_sdk,
    LLMAPIError
)

# Import shared modules from main project
from .ass_parser import (
    parse_ass_file,
    build_pairs_from_ass_lines,
    apply_pairs_to_ass_lines,
    render_ass_file,
    write_ass_file
)
from .chunker import chunk_pairs, print_chunk_statistics
from .memory import (
    GlobalMemory,
    init_global_memory,
    update_global_memory,
    estimate_memory_tokens,
    prune_learned_glossary_against_user_glossary,
    validate_memory_structure,
)
from .stats import (
    init_usage_stats,
    accumulate_usage,
    print_usage_report,
    print_chunk_progress
)
from .prompts import build_system_prompt
from .utils import estimate_tokens, print_verbose_preview, format_time


def get_checkpoint_path(input_path: str) -> str:
    """
    Generate checkpoint file path from input file path.

    Args:
        input_path: Path to input subtitle file

    Returns:
        Path to checkpoint file (e.g., input.ass -> input.ass.memory.yaml)
    """
    return f"{input_path}.memory.yaml"


def save_memory_checkpoint(memory: GlobalMemory, checkpoint_path: str) -> None:
    """Persist the complete episode memory as YAML."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            memory.to_dict(),
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def load_memory_checkpoint(checkpoint_path: str) -> Optional[GlobalMemory]:
    """Load and validate complete episode memory from YAML."""
    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not validate_memory_structure(payload):
        raise ValueError(f"Invalid memory checkpoint: {checkpoint_path}")
    return GlobalMemory.from_dict(payload)


def file_sha256(path: str) -> str:
    """Return the SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_refine_progress(
    progress_path: str,
    *,
    next_pair: int,
    artifact_path: str,
    memory_checkpoint_path: Optional[str],
) -> None:
    """Atomically commit the serial refinement recovery point."""
    payload = {
        "version": 1,
        "next_pair": next_pair,
        "artifact_path": artifact_path,
        "artifact_hash": file_sha256(artifact_path),
        "memory_checkpoint_path": memory_checkpoint_path,
        "memory_hash": (
            file_sha256(memory_checkpoint_path)
            if memory_checkpoint_path is not None
            else None
        ),
    }
    destination = os.path.abspath(progress_path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def apply_corrections_to_global_pairs(
    pairs: list,
    corrected_pairs: list
) -> None:
    """
    Apply corrections from a chunk back to the global pairs list.

    Args:
        pairs: Global list of SubtitlePair objects (modified in-place)
        corrected_pairs: Corrected pairs from LLM
    """
    # Create a mapping from ID to corrected pair
    correction_map = {pair.id: pair for pair in corrected_pairs}

    # Apply corrections to matching IDs in global list
    for pair in pairs:
        if pair.id in correction_map:
            corrected = correction_map[pair.id]
            pair.eng = corrected.eng
            pair.chinese = corrected.chinese


def print_current_terminology(global_memory: GlobalMemory, show_user_defined: bool = True) -> None:
    """
    Print current terminology for debugging.

    Args:
        global_memory: Global memory object containing terminology
        show_user_defined: Whether to show user-defined glossary (default: True)
    """
    print("\n  Current Terminology:")
    print("  " + "=" * 58)

    # User-defined glossary (authoritative) - only if requested
    if show_user_defined:
        if global_memory.user_glossary:
            print("  📌 User-Defined Glossary (Authoritative):")
            for entry in global_memory.user_glossary:
                eng = entry.get("eng", "")
                zh = entry.get("zh", "")
                print(f"    • {eng} → {zh}")
        else:
            print("  📌 User-Defined Glossary: (none)")
        print()  # Add blank line before learned glossary

    # Learned glossary
    if global_memory.glossary:
        print(f"  🧠 Learned Glossary ({len(global_memory.glossary)} entries):")
        for entry in global_memory.glossary:
            eng = entry.get("eng", "")
            zh = entry.get("zh", "")
            entry_type = entry.get("type", "")
            confidence = entry.get("confidence", "")

            type_str = f" [{entry_type}]" if entry_type else ""
            conf_str = f" (conf: {confidence})" if confidence else ""
            print(f"    • {eng} → {zh}{type_str}{conf_str}")
    else:
        print("  🧠 Learned Glossary: (none yet)")

    print("  " + "=" * 58)


def estimate_base_prompt_tokens(config, global_memory: GlobalMemory) -> int:
    """
    Estimate tokens for base prompt (system prompt + memory).

    Args:
        config: Configuration object
        global_memory: Global memory object

    Returns:
        Estimated token count
    """
    # Build a sample system prompt with current memory (using new template-based approach)
    system_prompt = build_system_prompt(global_memory, config)

    return estimate_tokens(system_prompt, config.refine.model)


def process_subtitles(
    input_path: str,
    output_path: str,
    config,
    use_stream: bool = False,
    resume_index: Optional[int] = None,
    enable_checkpoint: bool = False,
    checkpoint_path_override: Optional[str] = None,
    progress_manifest_path: Optional[str] = None,
) -> bool:
    """
    Main processing function for subtitle refinement using SDK.

    Args:
        input_path: Path to input .ass file
        output_path: Path to output .ass file
        config: ConfigSDK object
        use_stream: Whether to use stream mode (chat-completion only; response always streams)
        resume_index: Optional pair index to resume from (skips pairs before this index)
        enable_checkpoint: Whether to persist episode memory after each chunk
        checkpoint_path_override: Run-scoped memory checkpoint path
        progress_manifest_path: Run-scoped serial refinement progress file

    Returns:
        True if successful, False otherwise
    """
    try:
        protocol = config.refine.protocol
        if protocol is ModelProtocol.OPENAI_RESPONSES:
            mode_str = "openai response stream"
        elif protocol is ModelProtocol.OPENAI_CHAT_COMPATIBLE:
            mode_str = f"openai chat-completion {'stream' if use_stream else 'non-stream'}"
        elif protocol in {
            ModelProtocol.ANTHROPIC_MESSAGES,
            ModelProtocol.GOOGLE_GEMINI,
        }:
            mode_str = f"{protocol.value} non-stream"
        else:
            raise ValueError(f"unsupported refinement protocol: {protocol}")

        print(f"\n{'='*60}")
        print("SUBTITLE REFINEMENT TOOL")
        print(f"{'='*60}")
        print(f"Input:     {input_path}")
        print(f"Output:    {output_path}")
        print(f"Model:     {config.refine.model}")
        print(f"Mode:      {mode_str}")
        print(f"Representation: {config.intermediate_representation.upper()}")
        print(f"{'='*60}\n")

        # Step 1: Parse ASS file
        print("Step 1: Parsing ASS file...")
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return False

        header, ass_lines = parse_ass_file(input_path)
        print(f"  Parsed {len(ass_lines)} dialogue lines")

        # Step 2: Build subtitle pairs
        print("\nStep 2: Building subtitle pairs...")
        pairs = build_pairs_from_ass_lines(ass_lines)
        print(f"  Created {len(pairs)} subtitle pairs")

        if not pairs:
            print("Error: No subtitle pairs found")
            return False

        # Apply resume logic if enabled
        if resume_index is not None:
            if resume_index < 0:
                print(f"Error: Resume index must be non-negative (got {resume_index})")
                return False
            if resume_index >= len(pairs):
                print(f"Error: Resume index {resume_index} exceeds total pairs {len(pairs)}")
                return False

            print(f"\n  [RESUME MODE] Starting from pair index {resume_index}")
            print(f"  Skipping first {resume_index} pairs, processing remaining {len(pairs) - resume_index} pairs")

            # Load existing output file if it exists to preserve earlier pairs
            preserved_pairs = None
            if os.path.exists(output_path):
                print(f"  Loading existing output file: {output_path}")
                try:
                    existing_header, existing_ass_lines = parse_ass_file(output_path)
                    existing_pairs = build_pairs_from_ass_lines(existing_ass_lines)

                    # Copy corrected pairs from existing file (before resume_index)
                    preserved_pairs = 0
                    for i in range(min(resume_index, len(existing_pairs))):
                        if i < len(pairs) and existing_pairs[i].id == pairs[i].id:
                            pairs[i].eng = existing_pairs[i].eng
                            pairs[i].chinese = existing_pairs[i].chinese
                            preserved_pairs += 1

                    if preserved_pairs == resume_index:
                        print(f"  Preserved {preserved_pairs} pairs from existing output")
                    else:
                        print(f"  Warning: Preserved {preserved_pairs}/{resume_index} pairs from existing output (pair alignment mismatch)")
                        print("  Warning: Continuing may overwrite earlier output content; consider verifying input/output match.")
                except Exception as e:
                    print(f"  Warning: Could not load existing output file: {e}")
                    print(f"  Continuing without preserving earlier pairs...")
            else:
                print(f"  Note: Output file does not exist yet, will create new file")

            # Filter to only process pairs from resume_index onwards
            pairs_to_process = pairs[resume_index:]
            print(f"  Processing pairs {resume_index} to {len(pairs)-1} ({len(pairs_to_process)} pairs)")
        else:
            pairs_to_process = pairs
            preserved_pairs = None

        # Apply dry-run limit if enabled
        if config.dry_run:
            original_count = len(pairs_to_process)
            pairs_to_process = pairs_to_process[:min(10, len(pairs_to_process))]  # Limit to first 10 pairs
            print(f"  [DRY RUN] Limited to {len(pairs_to_process)} pairs (from {original_count})")

        # Step 3: Initialize global memory
        # NOTE: The new template-based approach (plan3.md) loads the prompt template
        # directly in build_system_prompt() and injects terminology from GlobalMemory.
        # User glossary from template is parsed and merged at prompt build time.
        global_memory = init_global_memory()
        checkpoint_path = None

        def prune_learned_glossary(reason: str) -> int:
            removed_count, removed_entries = prune_learned_glossary_against_user_glossary(global_memory)
            if not removed_count:
                return 0

            if config.verbose or config.debug_prompts:
                examples = [e.get("eng", "") for e in removed_entries[:5] if isinstance(e, dict)]
                examples_str = ", ".join([x for x in examples if x]) or "(unavailable)"
                print(f"  [Glossary prune] Removed {removed_count} learned entr(y/ies) covered by user glossary ({reason}); e.g., {examples_str}")

            return removed_count

        # Load template glossary and populate user_glossary for lock mechanism
        from .prompts import load_main_prompt_template, _parse_template_glossary, _find_section_boundaries
        template_glossary = None
        try:
            template = load_main_prompt_template(config)
            TARGET_SECTION = "User Terminology (Authoritative Glossary)"
            section_start, section_end, _ = _find_section_boundaries(template, TARGET_SECTION)
            if section_start is not None:
                section_content = template[section_start:section_end]
                template_glossary = _parse_template_glossary(section_content)
                global_memory.user_glossary = template_glossary
                if config.verbose:
                    print(f"  Loaded {len(template_glossary)} user glossary entries from template")
        except Exception as e:
            if config.verbose:
                print(f"  Warning: Failed to load template glossary: {e}")

        # Load the complete episode-memory checkpoint if enabled. The current
        # prompt template remains authoritative for user-defined terminology.
        if enable_checkpoint:
            checkpoint_path = checkpoint_path_override or get_checkpoint_path(input_path)
            checkpoint_memory = load_memory_checkpoint(checkpoint_path)
            if checkpoint_memory is not None:
                global_memory = checkpoint_memory
                if template_glossary is not None:
                    global_memory.user_glossary = template_glossary
                story_status = "with story context" if global_memory.story_description else "without story context"
                print(
                    f"  [CHECKPOINT] Loaded {len(global_memory.glossary)} learned glossary entries "
                    f"({story_status}) from: {os.path.basename(checkpoint_path)}"
                )
                if prune_learned_glossary("after loading checkpoint"):
                    save_memory_checkpoint(global_memory, checkpoint_path)
            else:
                print(f"  [CHECKPOINT] No existing checkpoint found, will create: {os.path.basename(checkpoint_path)}")

        # Step 4: Chunk pairs
        print("\nStep 3: Splitting into chunks...")
        base_prompt_tokens = estimate_base_prompt_tokens(config, global_memory)
        print(f"  Base prompt tokens: {base_prompt_tokens:,}")

        if config.refine_batch_size:
            print(f"  Chunking strategy: Fixed {config.refine_batch_size} pairs per chunk")
        else:
            print(f"  Chunking strategy: Token-based (max ~{config.chunk_token_soft_limit:,} tokens)")

        chunks = chunk_pairs(pairs_to_process, config, base_prompt_tokens)
        print_chunk_statistics(chunks, config.refine.model)

        # Apply max_chunks limit if set
        if config.max_chunks is not None and config.max_chunks < len(chunks):
            print(f"  [LIMITED] Processing only first {config.max_chunks} chunks (from {len(chunks)})")
            chunks = chunks[:config.max_chunks]

        # Step 5: Initialize stats
        total_usage = init_usage_stats()

        # Step 6: Process each chunk
        print("\nStep 4: Processing chunks with LLM...")
        print("-" * 60)

        # Track cumulative pairs processed for per-block update status
        cumulative_pairs_processed = resume_index or 0
        if preserved_pairs is not None:
            cumulative_pairs_processed = preserved_pairs

        # Define streaming callback for progress indication
        def streaming_progress_callback(chunk_text: str):
            if config.debug_prompts:
                # In debug mode (-vvv), print actual LLM output in real-time
                print(chunk_text, end="", flush=True)
            elif config.verbose:
                # In verbose mode (-v), just print dots for progress
                print(".", end="", flush=True)

        for i, chunk in enumerate(chunks):
            try:
                print(f"\nProcessing chunk {i+1}/{len(chunks)} ({len(chunk)} pairs)...")

                # Determine if this is the first chunk
                is_first_chunk = (i == 0)

                # Always prune learned glossary entries covered by the user glossary
                # BEFORE building prompts / printing terminology.
                if prune_learned_glossary("before LLM request") and enable_checkpoint and checkpoint_path:
                    save_memory_checkpoint(global_memory, checkpoint_path)

                # In -vvv mode, show terminology before processing
                # First chunk: show user-defined + learned
                # Subsequent chunks: only show learned (user-defined doesn't change)
                if config.debug_prompts:
                    print_current_terminology(global_memory, show_user_defined=is_first_chunk)

                # Start timing
                start_time = time.time()

                # Choose the API implementation from the configured role protocol.
                if protocol is ModelProtocol.OPENAI_RESPONSES:
                    # Response API: always streams
                    if config.debug_prompts:
                        print("\n  LLM Output (real-time):")
                        print("  " + "-" * 58)
                        print("  ", end="", flush=True)
                    elif config.verbose:
                        print("  Stream: ", end="", flush=True)

                    corrected_pairs, usage, response_text = refine_chunk_sdk_response(
                        chunk,
                        global_memory,
                        config,
                        chunk_callback=streaming_progress_callback,
                        print_system_prompt=is_first_chunk
                    )

                    if config.debug_prompts or config.verbose:
                        print()  # New line after stream output
                        if config.debug_prompts:
                            print("  " + "-" * 58)
                elif protocol is ModelProtocol.OPENAI_CHAT_COMPATIBLE and use_stream:
                    if config.debug_prompts:
                        # In debug mode, show header for real-time LLM output
                        print("\n  LLM Output (real-time):")
                        print("  " + "-" * 58)
                        print("  ", end="", flush=True)
                    elif config.verbose:
                        # In verbose mode, just show "Stream: " prefix
                        print("  Stream: ", end="", flush=True)

                    corrected_pairs, usage, response_text = refine_chunk_sdk_streaming(
                        chunk,
                        global_memory,
                        config,
                        chunk_callback=streaming_progress_callback,
                        print_system_prompt=is_first_chunk  # Only print system prompt for first chunk
                    )

                    if config.debug_prompts or config.verbose:
                        print()  # New line after stream output
                        if config.debug_prompts:
                            print("  " + "-" * 58)
                else:
                    corrected_pairs, usage, response_text = refine_chunk_sdk(
                        chunk,
                        global_memory,
                        config,
                        print_system_prompt=is_first_chunk  # Only print system prompt for first chunk
                    )

                total_usage = accumulate_usage(total_usage, usage)

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Print progress
                print_chunk_progress(i, len(chunks), usage)

                # Print timing if verbose
                if config.verbose:
                    print(f"  Time: {format_time(elapsed_time)}")
                    print()  # Add blank line for spacing
                    print_verbose_preview(response_text, usage.reasoning_tokens)
                    # Only show full response in non-stream mode
                    # (in stream mode, content was already shown in real-time)
                    if config.very_verbose and not use_stream and protocol is not ModelProtocol.OPENAI_RESPONSES:
                        print("\n  Full API response:\n")
                        print(response_text.rstrip() if response_text else "[Empty response]")
                        print()

                # Report missing pairs (best-effort recovery may return partial results)
                expected_ids = [p.id for p in chunk]
                returned_id_set = {p.id for p in corrected_pairs}
                missing_ids = [pid for pid in expected_ids if pid not in returned_id_set]
                if missing_ids:
                    preview = ", ".join(str(i) for i in missing_ids[:20])
                    suffix = "..." if len(missing_ids) > 20 else ""
                    print(f"  [Partial]: Missing {len(missing_ids)}/{len(chunk)} pair(s): {preview}{suffix}")

                # Apply corrections back to global pairs list
                apply_corrections_to_global_pairs(pairs, corrected_pairs)

                # Get pair range for this chunk
                chunk_first_id = chunk[0].id if chunk else 0
                chunk_last_id = chunk[-1].id if chunk else 0

                # Update cumulative count
                cumulative_pairs_processed += len(chunk)

                # Update global memory
                global_memory = update_global_memory(global_memory, corrected_pairs, config)

                # Ensure we don't keep redundant learned entries.
                prune_learned_glossary("after memory update")

                # Write per-block updates if enabled
                if config.per_block_update:
                    updated_ass_lines = apply_pairs_to_ass_lines(ass_lines, pairs)
                    output_content = render_ass_file(header, updated_ass_lines)
                    write_ass_file(output_path, output_content)
                    missing_note = f", {len(missing_ids)} missing" if missing_ids else ""
                    print(f"  [Per-block] ✓ Updated pairs {chunk_first_id}-{chunk_last_id}{missing_note} ({cumulative_pairs_processed}/{len(pairs)} total) in {output_path}")

                # Commit memory only after the corresponding artifact write.
                if enable_checkpoint and checkpoint_path and config.per_block_update:
                    save_memory_checkpoint(global_memory, checkpoint_path)
                if progress_manifest_path and config.per_block_update:
                    save_refine_progress(
                        progress_manifest_path,
                        next_pair=cumulative_pairs_processed,
                        artifact_path=output_path,
                        memory_checkpoint_path=checkpoint_path,
                    )

                # Check if memory needs compression
                memory_tokens = estimate_memory_tokens(global_memory, config.refine.model)
                if memory_tokens > config.memory_token_limit:
                    print(f"  Memory size ({memory_tokens} tokens) exceeds limit. Compressing...")
                    try:
                        compressed_memory, compression_usage = compress_memory_sdk(
                            global_memory,
                            config
                        )
                        global_memory = compressed_memory
                        total_usage = accumulate_usage(total_usage, compression_usage)

                        new_size = estimate_memory_tokens(global_memory, config.refine.model)
                        print(f"  Memory compressed: {memory_tokens} → {new_size} tokens")

                        # Save compressed memory to checkpoint (if enabled)
                        if enable_checkpoint and checkpoint_path and config.per_block_update:
                            save_memory_checkpoint(global_memory, checkpoint_path)
                        if progress_manifest_path and config.per_block_update:
                            save_refine_progress(
                                progress_manifest_path,
                                next_pair=cumulative_pairs_processed,
                                artifact_path=output_path,
                                memory_checkpoint_path=checkpoint_path,
                            )
                    except LLMAPIError as e:
                        print(f"  Warning: Memory compression failed: {e}")
                        print(f"  Continuing with uncompressed memory...")

            except LLMAPIError as e:
                print(f"  Error processing chunk {i+1}: {e}")
                raise

        print("\n" + "-" * 60)

        # Step 7: Generate output file
        print("\nStep 5: Generating output file...")
        updated_ass_lines = apply_pairs_to_ass_lines(ass_lines, pairs)
        output_content = render_ass_file(header, updated_ass_lines)

        # Write output
        write_ass_file(output_path, output_content)
        print(f"  Output written to: {output_path}")
        if enable_checkpoint and checkpoint_path:
            save_memory_checkpoint(global_memory, checkpoint_path)
        if progress_manifest_path:
            save_refine_progress(
                progress_manifest_path,
                next_pair=cumulative_pairs_processed,
                artifact_path=output_path,
                memory_checkpoint_path=checkpoint_path,
            )

        # Step 8: Print statistics
        from .pricing import calculate_cost, load_model_pricing

        pricing = load_model_pricing(config.refine.model)
        if pricing is None:
            print_usage_report(total_usage)
            print(f"  CCH pricing: no exact match for {config.refine.model}")
        else:
            cost = calculate_cost(
                pricing,
                prompt_tokens=total_usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens,
            )
            print_usage_report(total_usage, cost)
            print(
                "  CCH pricing: "
                f"{pricing.model_name} via {pricing.provider}; "
                f"table {pricing.version} ({pricing.refreshed_at})"
            )

        print("\n✓ Subtitle refinement completed successfully!\n")
        return True

    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point for SDK version."""
    parser = argparse.ArgumentParser(
        description="Refine bilingual (English-Chinese) ASS subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (non-stream)
  python -m subretrans.cli input.ass output.ass

  # Use stream mode for real-time feedback
  python -m subretrans.cli input.ass output.ass --stream

  # Dry run with stream
  python -m subretrans.cli input.ass output.ass --stream --dry-run

  # Verbose stream mode
  python -m subretrans.cli input.ass output.ass --stream -v

  # Fixed pairs per chunk
  python -m subretrans.cli input.ass output.ass --refine-batch-size 50

  # Limit number of chunks
  python -m subretrans.cli input.ass output.ass --max-chunks 5

  # Resume from a specific pair index (e.g., after error)
  python -m subretrans.cli input.ass output.ass --resume 680 --refine-batch-size 75

  # Enable checkpointing for glossary and incremental episode story
  python -m subretrans.cli input.ass output.ass --checkpoint --stream

  # Resume with checkpoint (preserves complete episode memory)
  python -m subretrans.cli input.ass output.ass --resume 680 --checkpoint

  # Disable per-block update (write only at end)
  python -m subretrans.cli input.ass output.ass --no-per-block-update

  # Enable per-block update explicitly (default behavior)
  python -m subretrans.cli input.ass output.ass --per-block-update

Note: Model protocols and credentials are selected from config.yaml api roles
Note: Per-block update is enabled by default for data safety (write after each chunk)
        """
    )

    parser.add_argument(
        "input",
        help="Input .ass subtitle file"
    )
    parser.add_argument(
        "output",
        help="Output .ass subtitle file"
    )
    parser.add_argument(
        "--config",
        help="YAML configuration path (default: repository config.yaml)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=None,
        dest="stream",
        help="Use stream mode for OpenAI chat-compatible refinement"
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable stream mode"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 10 pairs for testing"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        help="Memory token limit"
    )
    parser.add_argument(
        "--refine-batch-size",
        type=int,
        default=None,
        help="Number of subtitle pairs per chunk (overrides token-based chunking)"
    )
    parser.add_argument(
        "--intermediate-representation",
        choices=("json", "xml-pair", "pseudo-toml"),
        default=None,
        help="Override refine intermediate representation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Enable verbose output (-v), very verbose (-vv) for full responses, or ultra verbose (-vvv) for system prompts"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connection and exit"
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        metavar="INDEX",
        help="Resume processing from a specific pair index (e.g., --resume 680 starts from pair 680)"
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Persist glossary and incremental story description in .memory.yaml"
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Explicit run-scoped memory checkpoint path",
    )
    parser.add_argument(
        "--progress-manifest",
        help="Explicit run-scoped serial refinement progress JSON path",
    )
    parser.add_argument(
        "--per-block-update",
        action="store_true",
        default=None,
        dest="per_block_update",
        help="Update output file after each chunk/block (enabled by default)"
    )
    parser.add_argument(
        "--no-per-block-update",
        action="store_false",
        default=None,
        dest="per_block_update",
        help="Write output file only once at the end (disables per-block updates)"
    )

    args = parser.parse_args()

    verbose_count = args.verbose or 0
    verbose_enabled = verbose_count >= 1
    very_verbose_enabled = verbose_count >= 2
    debug_prompts_enabled = verbose_count >= 3

    # Load configuration (SDK version)
    try:
        config = load_config_sdk(
            yaml_file_path=args.config,
            model_name=args.model,
            use_stream=args.stream,
            per_block_update=args.per_block_update,
            dry_run=args.dry_run,
            max_chunks=args.max_chunks,
            memory_limit=args.memory_limit,
            refine_batch_size=args.refine_batch_size,
            verbose=verbose_enabled,
            very_verbose=very_verbose_enabled,
            debug_prompts=debug_prompts_enabled,
            intermediate_representation=args.intermediate_representation,
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check that the 'key' file exists in the repository root")
        return 1

    # Test connection if requested
    if args.test_connection:
        print("Testing API connection...")
        if test_api_connection_sdk(config):
            print("✓ API connection successful!")
            return 0
        else:
            print("✗ API connection failed!")
            return 1

    # Process subtitles
    success = process_subtitles(
        args.input,
        args.output,
        config,
        use_stream=config.use_stream,
        resume_index=args.resume,
        enable_checkpoint=args.checkpoint or args.checkpoint_path is not None,
        checkpoint_path_override=args.checkpoint_path,
        progress_manifest_path=args.progress_manifest,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
