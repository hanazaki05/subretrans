#!/usr/bin/env python3
"""
Main entry point for subtitle refinement using OpenAI SDK.

This is the SDK version that supports both streaming and non-streaming modes.
"""

import argparse
import sys
import os
import time
import yaml
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SDK-specific modules
from config_sdk import load_config_sdk
from llm_client_sdk import (
    refine_chunk_sdk,
    refine_chunk_sdk_streaming,
    compress_memory_sdk,
    test_api_connection_sdk,
    LLMAPIError
)

# Import shared modules from main project
from ass_parser import (
    parse_ass_file,
    build_pairs_from_ass_lines,
    apply_pairs_to_ass_lines,
    render_ass_file,
    write_ass_file
)
from chunker import chunk_pairs, print_chunk_statistics
from memory import (
    GlobalMemory,
    init_global_memory,
    update_global_memory,
    estimate_memory_tokens
)
from stats import (
    init_usage_stats,
    accumulate_usage,
    estimate_cost,
    print_usage_report,
    print_chunk_progress
)
from prompts import build_system_prompt
from utils import estimate_tokens, print_verbose_preview, format_time


def get_checkpoint_path(input_path: str) -> str:
    """
    Generate checkpoint file path from input file path.

    Args:
        input_path: Path to input subtitle file

    Returns:
        Path to checkpoint file (e.g., input.ass -> input.ass.glossary.yaml)
    """
    return f"{input_path}.glossary.yaml"


def save_glossary_checkpoint(glossary: list, checkpoint_path: str) -> None:
    """
    Save learned glossary to checkpoint file in YAML format.

    Args:
        glossary: List of learned glossary entries
        checkpoint_path: Path to checkpoint file
    """
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            yaml.dump(glossary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"  Warning: Could not save glossary checkpoint: {e}")


def load_glossary_checkpoint(checkpoint_path: str) -> Optional[list]:
    """
    Load learned glossary from checkpoint file in YAML format.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        List of glossary entries if file exists and is valid, None otherwise
    """
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            glossary = yaml.safe_load(f)

        # Validate that it's a list
        if not isinstance(glossary, list):
            print(f"  Warning: Invalid checkpoint format in {checkpoint_path}")
            return None

        return glossary
    except Exception as e:
        print(f"  Warning: Could not load glossary checkpoint: {e}")
        return None


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

    return estimate_tokens(system_prompt, config.main_model.name)


def process_subtitles(
    input_path: str,
    output_path: str,
    config,
    use_streaming: bool = False,
    resume_index: Optional[int] = None,
    enable_checkpoint: bool = False
) -> bool:
    """
    Main processing function for subtitle refinement using SDK.

    Args:
        input_path: Path to input .ass file
        output_path: Path to output .ass file
        config: ConfigSDK object
        use_streaming: Whether to use streaming API
        resume_index: Optional pair index to resume from (skips pairs before this index)
        enable_checkpoint: Whether to enable glossary checkpoint system (default: False)

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"SUBTITLE REFINEMENT TOOL (OpenAI SDK)")
        print(f"{'='*60}")
        print(f"Input:     {input_path}")
        print(f"Output:    {output_path}")
        print(f"Model:     {config.main_model.name}")
        print(f"Mode:      {'Streaming' if use_streaming else 'Non-streaming'}")
        print(f"Format:    {config.intermediate_format.upper()}")
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
            if os.path.exists(output_path):
                print(f"  Loading existing output file: {output_path}")
                try:
                    existing_header, existing_ass_lines = parse_ass_file(output_path)
                    existing_pairs = build_pairs_from_ass_lines(existing_ass_lines)

                    # Copy corrected pairs from existing file (before resume_index)
                    for i in range(min(resume_index, len(existing_pairs))):
                        if i < len(pairs) and existing_pairs[i].id == pairs[i].id:
                            pairs[i].eng = existing_pairs[i].eng
                            pairs[i].chinese = existing_pairs[i].chinese

                    print(f"  Preserved {resume_index} pairs from existing output")
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

        # Load glossary checkpoint if enabled
        checkpoint_path = None
        if enable_checkpoint:
            checkpoint_path = get_checkpoint_path(input_path)
            checkpoint_glossary = load_glossary_checkpoint(checkpoint_path)
            if checkpoint_glossary:
                print(f"  [CHECKPOINT] Loaded {len(checkpoint_glossary)} glossary entries from: {os.path.basename(checkpoint_path)}")
                global_memory.glossary = checkpoint_glossary
            else:
                print(f"  [CHECKPOINT] No existing checkpoint found, will create: {os.path.basename(checkpoint_path)}")

        # Step 4: Chunk pairs
        print("\nStep 3: Splitting into chunks...")
        base_prompt_tokens = estimate_base_prompt_tokens(config, global_memory)
        print(f"  Base prompt tokens: {base_prompt_tokens:,}")

        if config.pairs_per_chunk:
            print(f"  Chunking strategy: Fixed {config.pairs_per_chunk} pairs per chunk")
        else:
            print(f"  Chunking strategy: Token-based (max ~{config.chunk_token_soft_limit:,} tokens)")

        chunks = chunk_pairs(pairs_to_process, config, base_prompt_tokens)
        print_chunk_statistics(chunks, config.main_model.name)

        # Apply max_chunks limit if set
        if config.max_chunks is not None and config.max_chunks < len(chunks):
            print(f"  [LIMITED] Processing only first {config.max_chunks} chunks (from {len(chunks)})")
            chunks = chunks[:config.max_chunks]

        # Step 5: Initialize stats
        total_usage = init_usage_stats()

        # Step 6: Process each chunk
        print("\nStep 4: Processing chunks with LLM...")
        print("-" * 60)

        # Track cumulative pairs processed for incremental output status
        cumulative_pairs_processed = 0

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

                # In -vvv mode, show terminology before processing
                # First chunk: show user-defined + learned
                # Subsequent chunks: only show learned (user-defined doesn't change)
                if config.debug_prompts:
                    print_current_terminology(global_memory, show_user_defined=is_first_chunk)

                # Start timing
                start_time = time.time()

                # Choose streaming or non-streaming based on flag
                if use_streaming:
                    if config.debug_prompts:
                        # In debug mode, show header for real-time LLM output
                        print("\n  LLM Output (real-time):")
                        print("  " + "-" * 58)
                        print("  ", end="", flush=True)
                    elif config.verbose:
                        # In verbose mode, just show "Streaming: " prefix
                        print("  Streaming: ", end="", flush=True)

                    corrected_pairs, usage, response_text = refine_chunk_sdk_streaming(
                        chunk,
                        global_memory,
                        config,
                        chunk_callback=streaming_progress_callback,
                        print_system_prompt=is_first_chunk  # Only print system prompt for first chunk
                    )

                    if config.debug_prompts or config.verbose:
                        print()  # New line after streaming output
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
                    # Only show full response in non-streaming mode
                    # (in streaming mode, content was already shown in real-time)
                    if config.very_verbose and not use_streaming:
                        print("\n  Full API response:\n")
                        print(response_text.rstrip() if response_text else "[Empty response]")
                        print()

                # Apply corrections back to global pairs list
                apply_corrections_to_global_pairs(pairs, corrected_pairs)

                # Get pair range for this chunk
                chunk_first_id = corrected_pairs[0].id if corrected_pairs else 0
                chunk_last_id = corrected_pairs[-1].id if corrected_pairs else 0

                # Update cumulative count
                cumulative_pairs_processed += len(corrected_pairs)

                # Update global memory
                global_memory = update_global_memory(global_memory, corrected_pairs, config)

                # Save updated glossary to checkpoint file (if enabled)
                if enable_checkpoint and checkpoint_path:
                    save_glossary_checkpoint(global_memory.glossary, checkpoint_path)

                # Write incremental output if enabled
                if config.incremental_output:
                    try:
                        updated_ass_lines = apply_pairs_to_ass_lines(ass_lines, pairs)
                        output_content = render_ass_file(header, updated_ass_lines)
                        write_ass_file(output_path, output_content)
                        # Always show incremental save status (not just in verbose mode)
                        print(f"  [Incremental] ✓ Saved pairs {chunk_first_id}-{chunk_last_id} ({cumulative_pairs_processed}/{len(pairs)} total) to {output_path}")
                    except Exception as e:
                        print(f"  [Incremental] ✗ Failed to save progress: {e}")

                # Check if memory needs compression
                memory_tokens = estimate_memory_tokens(global_memory, config.main_model.name)
                if memory_tokens > config.memory_token_limit:
                    print(f"  Memory size ({memory_tokens} tokens) exceeds limit. Compressing...")
                    try:
                        compressed_memory, compression_usage = compress_memory_sdk(
                            global_memory,
                            config
                        )
                        global_memory = compressed_memory
                        total_usage = accumulate_usage(total_usage, compression_usage)

                        new_size = estimate_memory_tokens(global_memory, config.main_model.name)
                        print(f"  Memory compressed: {memory_tokens} → {new_size} tokens")

                        # Save compressed glossary to checkpoint (if enabled)
                        if enable_checkpoint and checkpoint_path:
                            save_glossary_checkpoint(global_memory.glossary, checkpoint_path)
                    except LLMAPIError as e:
                        print(f"  Warning: Memory compression failed: {e}")
                        print(f"  Continuing with uncompressed memory...")

            except LLMAPIError as e:
                print(f"  Error processing chunk {i+1}: {e}")
                print(f"  Skipping this chunk and continuing...")
                continue

        print("\n" + "-" * 60)

        # Step 7: Generate output file
        print("\nStep 5: Generating output file...")
        updated_ass_lines = apply_pairs_to_ass_lines(ass_lines, pairs)
        output_content = render_ass_file(header, updated_ass_lines)

        # Write output
        write_ass_file(output_path, output_content)
        print(f"  Output written to: {output_path}")

        # Step 8: Print statistics
        cost = estimate_cost(
            total_usage,
            config.price_per_1k_prompt_tokens,
            config.price_per_1k_completion_tokens
        )
        print_usage_report(total_usage, cost)

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
        description="Refine bilingual (English-Chinese) ASS subtitles using OpenAI SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (non-streaming)
  python main_sdk.py input.ass output.ass

  # Use streaming API for real-time feedback
  python main_sdk.py input.ass output.ass --streaming

  # Dry run with streaming
  python main_sdk.py input.ass output.ass --streaming --dry-run

  # Verbose streaming mode
  python main_sdk.py input.ass output.ass --streaming -v

  # Fixed pairs per chunk
  python main_sdk.py input.ass output.ass --pairs-per-chunk 50

  # Limit number of chunks
  python main_sdk.py input.ass output.ass --max-chunks 5

  # Resume from a specific pair index (e.g., after error)
  python main_sdk.py input.ass output.ass --resume 680 --pairs-per-chunk 75

  # Enable checkpoint system to save/load learned terminology
  python main_sdk.py input.ass output.ass --checkpoint --streaming

  # Resume with checkpoint (preserves learned terms across runs)
  python main_sdk.py input.ass output.ass --resume 680 --checkpoint

  # Disable incremental output (write only at end)
  python main_sdk.py input.ass output.ass --no-incremental-output

  # Enable incremental output explicitly (default behavior)
  python main_sdk.py input.ass output.ass --incremental-output

Note: API key is automatically loaded from ../key file
Note: Incremental output is enabled by default for data safety (write after each chunk)
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
        "--streaming",
        action="store_true",
        default=None,
        help="Use streaming API for real-time token generation (default: from config.yaml)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming API"
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
        "--pairs-per-chunk",
        type=int,
        default=None,
        help="Number of subtitle pairs per chunk (overrides token-based chunking)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Enable verbose output (-v), very verbose (-vv) for full responses, or ultra verbose (-vvv) for system prompts"
    )
    parser.add_argument(
        "--stats",
        type=float,
        default=1.0,
        help="Stats refresh interval in seconds for verbose mode (default: 1.0)"
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
        help="Enable glossary checkpoint system (save/load learned terminology to/from .glossary.yaml file)"
    )
    parser.add_argument(
        "--incremental-output",
        action="store_true",
        default=None,
        dest="incremental_output",
        help="Write output file after each chunk (default: from config.yaml, typically enabled for safety)"
    )
    parser.add_argument(
        "--no-incremental-output",
        action="store_false",
        dest="incremental_output",
        help="Write output file only once at the end (disables incremental updates)"
    )

    args = parser.parse_args()

    verbose_count = args.verbose or 0
    verbose_enabled = verbose_count >= 1
    very_verbose_enabled = verbose_count >= 2
    debug_prompts_enabled = verbose_count >= 3

    # Load configuration (SDK version)
    try:
        config = load_config_sdk(
            model_name=args.model,
            use_streaming=args.streaming,
            incremental_output=args.incremental_output,
            dry_run=args.dry_run,
            max_chunks=args.max_chunks,
            memory_limit=args.memory_limit,
            pairs_per_chunk=args.pairs_per_chunk,
            verbose=verbose_enabled,
            very_verbose=very_verbose_enabled,
            debug_prompts=debug_prompts_enabled,
            stats_interval=args.stats
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check that the 'key' file exists in the parent directory")
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

    # Process subtitles (use_streaming from config, which may be overridden by CLI)
    success = process_subtitles(
        args.input,
        args.output,
        config,
        use_streaming=config.use_streaming,
        resume_index=args.resume,
        enable_checkpoint=args.checkpoint
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())