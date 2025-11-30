#!/bin/bash
#
# Wrapper script to run main_sdk.py or genreq.py from any directory
#
# Usage:
#   ./run.sh input.ass output.ass [options]             # Run main_sdk.py
#   ./run.sh genreq input.ass --pairs-per-chunk 120     # Run genreq.py
#
# Examples:
#   ./run.sh ~/files/input.ass ~/files/output.ass --streaming -v
#   ./run.sh input.ass output.ass --dry-run
#   ./run.sh genreq JAG.S04E09.zh-cn.ass --pairs-per-chunk 120
#   cd /tmp && /path/to/experiment/run.sh input.ass output.ass
#
# Works with symlinks:
#   ln -s /path/to/experiment/run.sh ~/bin/subretrans
#   subretrans input.ass output.ass --streaming -v
#   subretrans genreq input.ass --pairs-per-chunk 120
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
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# (IMPORTANT KEEP HERE) remove python alias 
unalias python &>/dev/null # IMPORTANT KEEP HERE
unalias python3 &>/dev/null # IMPORTANT KEEP HERE

# Explicit paths for clarity (updated automatically based on script location)
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
MAIN_SDK_PATH="$SCRIPT_DIR/main_sdk.py"
GENREQ_PATH="$SCRIPT_DIR/genreq.py"

# Debug info (uncomment to troubleshoot)
# echo "Script path: $SCRIPT_PATH"
# echo "Script dir: $SCRIPT_DIR"
# echo "Project dir: $PROJECT_DIR"
# echo "Venv: $VENV_PATH"
# echo "Main SDK: $MAIN_SDK_PATH"
# echo "Genreq: $GENREQ_PATH"

# Activate virtual environment
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at: $VENV_PATH"
    echo "Please run from the correct location or check your installation."
    exit 1
fi

source "$VENV_PATH"

# Check if first argument is "genreq"
if [ "$1" = "genreq" ]; then
    # Run genreq.py with remaining arguments
    if [ ! -f "$GENREQ_PATH" ]; then
        echo "Error: genreq.py not found at: $GENREQ_PATH"
        echo "Please check your installation."
        exit 1
    fi

    shift  # Remove "genreq" from arguments
    exec python "$GENREQ_PATH" "$@"
else
    # Run main_sdk.py with all arguments passed through
    if [ ! -f "$MAIN_SDK_PATH" ]; then
        echo "Error: main_sdk.py not found at: $MAIN_SDK_PATH"
        echo "Please check your installation."
        exit 1
    fi

    exec python "$MAIN_SDK_PATH" "$@"
fi
