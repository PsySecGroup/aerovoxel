#!/bin/bash
# Resolve the project root (one level up from this scripts/ directory),
# regardless of where the caller invokes this from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -d "$PROJECT_ROOT/venv" ]; then
    python3 -m venv "$PROJECT_ROOT/venv"
fi

source "$PROJECT_ROOT/venv/bin/activate"