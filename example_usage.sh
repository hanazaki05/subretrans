#!/bin/bash
# Example invocations of the standalone refinement CLI against example_input.ass.
# Each run writes a complete ASS file after every chunk, so any of them can be interrupted safely.

unalias python 2>/dev/null
unalias python3 2>/dev/null

set -e
cd "$(dirname "$0")"

echo "Example 1: dry run (first 10 pairs)"
./run.sh example_input.ass example_output_dryrun.ass --dry-run -v

echo "Example 2: fixed batches, two chunks only"
./run.sh example_input.ass example_output_2chunks.ass --refine-batch-size 30 --max-chunks 2

echo "Example 3: full run with episode memory checkpoint and streaming output"
./run.sh example_input.ass example_output_full.ass --checkpoint --stream -v

echo "Example 4: resume the full run from pair 60 using the saved checkpoint"
./run.sh example_input.ass example_output_full.ass --checkpoint --resume 60

echo "Example 5: dump the prompts without calling any API"
./run.sh genreq example_input.ass --refine-batch-size 60 --max-chunks 1
