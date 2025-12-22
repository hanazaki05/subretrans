#!/usr/bin/env python3
"""
Generate request prompts from ASS file without calling API.

Useful for testing request timing with other tools or inspecting prompts.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from config_sdk import load_config_sdk
from ass_parser import parse_ass_file, build_pairs_from_ass_lines
from chunker import chunk_pairs
from memory import init_global_memory, estimate_memory_tokens
from prompts import build_system_prompt, build_user_prompt_for_chunk, split_user_prompt_and_glossary, set_user_instruction
from utils import estimate_tokens, estimate_pairs_tokens
from serializers import serialize


def generate_prompts(input_path, output_path, pairs_per_chunk, max_chunks, config):
    """
    Generate system and user prompts for each chunk without calling API.

    Args:
        input_path: Path to input .ass file
        output_path: Path to output markdown file
        pairs_per_chunk: Number of subtitle pairs per chunk
        max_chunks: Maximum number of chunks to process (None = all)
        config: Configuration object

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"PROMPT GENERATOR (No API Calls)")
        print(f"{'='*60}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Pairs per chunk: {pairs_per_chunk}")
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

        # Step 3: Initialize global memory and load custom prompt
        global_memory = init_global_memory()

        # Load custom main prompt (if present)
        prompt_path_cfg = getattr(config, "user_prompt_path", "custom_main_prompt.md")
        if os.path.isabs(prompt_path_cfg):
            custom_prompt_path = prompt_path_cfg
        else:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            custom_prompt_path = os.path.join(parent_dir, prompt_path_cfg)

        if os.path.exists(custom_prompt_path):
            try:
                with open(custom_prompt_path, "r", encoding="utf-8") as f:
                    custom_text = f.read()
                user_instructions, user_glossary = split_user_prompt_and_glossary(custom_text)
                if user_instructions:
                    set_user_instruction(user_instructions)
                if user_glossary:
                    global_memory.user_glossary = user_glossary
            except Exception as e:
                print(f"  Warning: Failed to load custom_main_prompt.md: {e}")

        # Step 4: Chunk pairs
        print("\nStep 3: Splitting into chunks...")

        # Temporarily set pairs_per_chunk in config
        config.pairs_per_chunk = pairs_per_chunk

        base_prompt_tokens = estimate_tokens(
            build_system_prompt(global_memory),
            config.main_model.name
        )
        print(f"  Base prompt tokens: {base_prompt_tokens:,}")
        print(f"  Chunking strategy: Fixed {pairs_per_chunk} pairs per chunk")

        chunks = chunk_pairs(pairs, config, base_prompt_tokens)
        print(f"  Created {len(chunks)} chunks")

        # Apply max_chunks limit if set
        if max_chunks is not None and max_chunks < len(chunks):
            print(f"  [LIMITED] Generating prompts for only first {max_chunks} chunks (from {len(chunks)})")
            chunks = chunks[:max_chunks]

        # Step 5: Generate prompts for each chunk
        print("\nStep 4: Generating prompts for each chunk...")
        print("-" * 60)

        all_prompts = []
        for i, chunk in enumerate(chunks):
            print(f"\nGenerating prompts for chunk {i+1}/{len(chunks)} ({len(chunk)} pairs)...")

            # Build system prompt (with format-aware example conversion)
            system_prompt = build_system_prompt(global_memory, config)

            # Build user prompt using configured intermediate format
            pairs_serialized = serialize(chunk, config.intermediate_format)
            user_prompt = build_user_prompt_for_chunk(pairs_serialized)

            # Estimate tokens
            system_tokens = estimate_tokens(system_prompt, config.main_model.name)
            user_tokens = estimate_tokens(user_prompt, config.main_model.name)
            total_tokens = system_tokens + user_tokens

            print(f"  System prompt: {system_tokens:,} tokens")
            print(f"  User prompt: {user_tokens:,} tokens")
            print(f"  Total: {total_tokens:,} tokens")

            all_prompts.append({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "total_tokens": total_tokens
            })

        print("\n" + "-" * 60)

        # Step 6: Write to markdown file
        print("\nStep 5: Writing prompts to markdown file...")
        write_markdown(
            chunks=chunks,
            prompts=all_prompts,
            output_path=output_path,
            config=config,
            input_filename=os.path.basename(input_path),
            total_pairs=len(pairs)
        )
        print(f"  Output written to: {output_path}")

        print("\n✓ Prompt generation completed successfully!\n")
        return True

    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def write_markdown(chunks, prompts, output_path, config, input_filename, total_pairs):
    """
    Write prompts to markdown file.

    Args:
        chunks: List of subtitle pair chunks
        prompts: List of prompt dictionaries
        output_path: Path to output markdown file
        config: Configuration object
        input_filename: Original input filename
        total_pairs: Total number of subtitle pairs
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Request Prompts for {input_filename}\n\n")

        # Write configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Total pairs:** {total_pairs}\n")
        f.write(f"- **Pairs per chunk:** {config.pairs_per_chunk}\n")
        f.write(f"- **Total chunks:** {len(chunks)}\n")
        f.write(f"- **Intermediate format:** {config.intermediate_format}\n")
        f.write(f"- **Model:** {config.main_model.name}\n")
        f.write(f"- **Max output tokens:** {config.main_model.max_output_tokens:,}\n")
        f.write(f"- **Temperature:** {config.main_model.temperature}\n")
        if hasattr(config.main_model, 'reasoning_effort'):
            f.write(f"- **Reasoning effort:** {config.main_model.reasoning_effort}\n")
        f.write("\n")

        # Write summary table
        f.write("## Token Summary\n\n")
        f.write("| Chunk | Pairs | System Tokens | User Tokens | Total Tokens |\n")
        f.write("|-------|-------|---------------|-------------|-------------|\n")

        total_system = 0
        total_user = 0
        total_all = 0

        for prompt in prompts:
            f.write(f"| {prompt['chunk_index']+1}/{len(prompts)} | "
                   f"{prompt['chunk_size']} | "
                   f"{prompt['system_tokens']:,} | "
                   f"{prompt['user_tokens']:,} | "
                   f"{prompt['total_tokens']:,} |\n")
            total_system += prompt['system_tokens']
            total_user += prompt['user_tokens']
            total_all += prompt['total_tokens']

        f.write(f"| **Total** | {total_pairs} | "
               f"{total_system:,} | {total_user:,} | {total_all:,} |\n")
        f.write("\n---\n\n")

        # Write each chunk's prompts
        for prompt in prompts:
            chunk_num = prompt['chunk_index'] + 1
            f.write(f"## Chunk {chunk_num}/{len(prompts)} ({prompt['chunk_size']} pairs)\n\n")

            # System prompt
            f.write("### System Prompt\n\n")
            f.write("```\n")
            f.write(prompt['system_prompt'])
            f.write("\n```\n\n")

            # User prompt
            f.write("### User Prompt\n\n")
            f.write("```\n")
            f.write(prompt['user_prompt'])
            f.write("\n```\n\n")

            # Token estimates
            f.write("### Token Estimates\n\n")
            f.write(f"- **System prompt:** {prompt['system_tokens']:,} tokens\n")
            f.write(f"- **User content:** {prompt['user_tokens']:,} tokens\n")
            f.write(f"- **Total input:** {prompt['total_tokens']:,} tokens\n")
            f.write(f"- **Max output:** {config.main_model.max_output_tokens:,} tokens\n")
            f.write(f"- **Estimated max total:** {prompt['total_tokens'] + config.main_model.max_output_tokens:,} tokens\n")
            f.write("\n---\n\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate request prompts from ASS file without calling API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate prompts with 120 pairs per chunk
  python genreq.py JAG.S04E09.zh-cn.ass --pairs-per-chunk 120

  # Limit to first 2 chunks
  python genreq.py JAG.S04E09.zh-cn.ass --pairs-per-chunk 120 --max-chunks 2

  # Custom output file
  python genreq.py input.ass --pairs-per-chunk 100 --output my_prompts.md

Note: This tool does NOT call the API, it only generates the prompts.
        """
    )

    parser.add_argument(
        "input",
        help="Input .ass subtitle file"
    )
    parser.add_argument(
        "--pairs-per-chunk",
        type=int,
        required=True,
        help="Number of subtitle pairs per chunk (required)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file (default: {input_basename}_prompts.md)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to generate (for testing)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_stem = Path(args.input).stem
        output_path = f"{input_stem}_prompts.md"

    # Load configuration (SDK version)
    try:
        config = load_config_sdk(
            pairs_per_chunk=args.pairs_per_chunk,
            verbose=False
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Generate prompts
    success = generate_prompts(
        args.input,
        output_path,
        args.pairs_per_chunk,
        args.max_chunks,
        config
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
