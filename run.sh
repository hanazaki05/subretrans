#!/bin/bash
#
# Wrapper script to run the refinement CLI, the prompt generator, or the agent pipeline from any directory.
#
# Usage:
#   ./run.sh input.ass output.ass [options]                       # subretrans.cli (standalone refinement)
#   ./run.sh genreq input.ass --refine-batch-size 120             # subretrans.genreq (prompt dump, no API calls)
#   ./run.sh pipeline run input.srt output.ass --mode ... --thread-id ...   # subretrans.pipeline_cli
#   ./run.sh pipeline status|resume|review <thread-id> [...]
#
# Examples:
#   ./run.sh ~/files/input.ass ~/files/output.ass --checkpoint --stream -v
#   ./run.sh input.ass output.ass --dry-run
#   ./run.sh pipeline run source.en.srt release.ass --mode parallel_initial --thread-id episode-s07e01
#
# Works with symlinks:
#   ln -s /path/to/subretrans/run.sh ~/bin/subretrans
#   subretrans input.ass output.ass --checkpoint -v
#

# Resolve the real path of this script, even if it's a symlink
if [ -L "$0" ]; then
    # This is a symlink, resolve it
    SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || greadlink -f "$0" 2>/dev/null || perl -MCwd -e 'print Cwd::abs_path shift' "$0")"
else
    # Not a symlink, get absolute path
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
fi

# Get the directory where the actual script is located (not the symlink)
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$SCRIPT_DIR"

# (IMPORTANT KEEP HERE) remove python alias 
unalias python &>/dev/null # IMPORTANT KEEP HERE
unalias python3 &>/dev/null # IMPORTANT KEEP HERE

# Explicit paths for clarity (updated automatically based on script location)
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
CLI_PATH="$PROJECT_DIR/subretrans/cli.py"
GENREQ_PATH="$PROJECT_DIR/subretrans/genreq.py"
PIPELINE_CLI_PATH="$PROJECT_DIR/subretrans/pipeline_cli.py"

# Debug info (uncomment to troubleshoot)
# echo "Script path: $SCRIPT_PATH"
# echo "Script dir: $SCRIPT_DIR"
# echo "Project dir: $PROJECT_DIR"
# echo "Venv: $VENV_PATH"
# echo "CLI: $CLI_PATH"
# echo "Genreq: $GENREQ_PATH"

# Activate virtual environment
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at: $VENV_PATH"
    echo "Please run from the correct location or check your installation."
    exit 1
fi

source "$VENV_PATH"

# Check if first argument is "genreq"
if [ "$1" = "pipeline" ]; then
    if [ ! -f "$PIPELINE_CLI_PATH" ]; then
        echo "Error: pipeline_cli.py not found at: $PIPELINE_CLI_PATH"
        exit 1
    fi

    shift
    cd "$PROJECT_DIR" || exit 1
    exec python -m subretrans.pipeline_cli "$@"
elif [ "$1" = "genreq" ]; then
    # Run the prompt generator with remaining arguments
    if [ ! -f "$GENREQ_PATH" ]; then
        echo "Error: genreq.py not found at: $GENREQ_PATH"
        echo "Please check your installation."
        exit 1
    fi

    shift  # Remove "genreq" from arguments
    cd "$PROJECT_DIR" || exit 1
    exec python -m subretrans.genreq "$@"
else
    # Run the main CLI with all arguments passed through
    if [ ! -f "$CLI_PATH" ]; then
        echo "Error: CLI module not found at: $CLI_PATH"
        echo "Please check your installation."
        exit 1
    fi

    cd "$PROJECT_DIR" || exit 1
    exec python -m subretrans.cli "$@"
fi
