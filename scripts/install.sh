#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/activate.sh"

pip install --upgrade pip
pip install -r requirements.txt

source "$SCRIPT_DIR/recompile.sh"

python -c 'import process_image_cpp; print("SUCCESS: The C++ voxel tracker is working")'
mkdir -p fits