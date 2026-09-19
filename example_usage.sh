#!/bin/bash
# Example usage script for subtitle refinement tool

echo "=== Subtitle Refinement Tool - Example Usage ==="
echo ""

# Activate virtual environment
unalias python 2>/dev/null
unalias python3 2>/dev/null
source venv/bin/activate

# Example 1: Quick test with dry-run mode
echo "Example 1: Quick test (dry-run mode, first 10 pairs)"
./run.sh test_input.ass output_dryrun.ass --dry-run
echo ""

# Example 2: Process with fixed pairs per chunk
echo "Example 2: Process 50 pairs per chunk"
./run.sh test_input.ass output_50pairs.ass --pairs-per-chunk 50
echo ""

# Example 3: Process limited number of chunks
echo "Example 3: Process first 3 chunks only"
./run.sh test_input.ass output_3chunks.ass --max-chunks 3
echo ""

# Example 4: Combine chunk size with max chunks
echo "Example 4: 30 pairs per chunk, max 2 chunks"
./run.sh test_input.ass output_30pairs_2chunks.ass --pairs-per-chunk 30 --max-chunks 2
echo ""

# Example 5: Full processing (token-based chunking)
echo "Example 5: Full processing of test file (token-based)"
./run.sh test_input.ass output_full.ass
echo ""

# Example 6: Test API connection
echo "Example 6: Test API connection"
./run.sh test_input.ass output.ass --test-connection
echo ""

# Example 7: Verbose mode with timing and preview
echo "Example 7: Verbose mode (shows timing and response preview)"
./run.sh test_input.ass output_verbose.ass -v --pairs-per-chunk 30 --max-chunks 2
echo ""

# Example 8: Verbose mode with custom stats refresh interval
echo "Example 8: Verbose mode with 0.5s refresh interval"
./run.sh test_input.ass output_verbose_fast.ass -v --stats 0.5 --max-chunks 1
echo ""

# Example 9: Full verbose processing
echo "Example 9: Full processing with verbose output"
./run.sh test_input.ass output_full_verbose.ass -v
echo ""

echo "=== Examples completed ==="
