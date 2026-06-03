#!/bin/bash
# Shared run logic for all examples.
# Called by each example's run.sh with its own directory as $1.

EXAMPLE_DIR="$1"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── venv ────────────────────────────────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    source "$PROJECT_ROOT/scripts/activate.sh"
fi

# ── ray_voxel binary ────────────────────────────────────────────────────────
if [ ! -f "$PROJECT_ROOT/build/ray_voxel" ]; then
    echo "[run] ray_voxel not found, building..."
    bash "$PROJECT_ROOT/scripts/build.sh"
    if [ $? -ne 0 ]; then
        echo "[run] Build failed, aborting."
        exit 1
    fi
fi

# ── frames + metadata ───────────────────────────────────────────────────────
META="$EXAMPLE_DIR/metadata.json"
FRAME_COUNT=$(ls "$EXAMPLE_DIR"/frames/*.jpg "$EXAMPLE_DIR"/frames/*.png 2>/dev/null | wc -l)

if [ ! -f "$META" ] || [ "$FRAME_COUNT" -eq 0 ]; then
    echo "[run] Frames or metadata missing, running generate..."
    python "$EXAMPLE_DIR/generate.py"
    if [ $? -ne 0 ]; then
        echo "[run] generate.py failed, aborting."
        exit 1
    fi
fi

# ── ray_voxel ───────────────────────────────────────────────────────────────
# Re-run if either the voxel grid or its companion multicam JSON is absent.
# The JSON is produced in the same pass, so a missing JSON means the binary
# that wrote the existing .bin predates the multicam feature.
# Initialize FORCE to 0
FORCE=0
HEADLESS=0

# Loop through all arguments
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE=1
    fi

    if [ "$arg" = "--headless" ]; then
        HEADLESS=1
    fi
done

echo $FORCE

if [ "$FORCE" -eq 1 ] || [ ! -f "$EXAMPLE_DIR/voxel_grid.bin" ]; then
    echo "[run] Running ray_voxel..."
    "$PROJECT_ROOT/build/ray_voxel" "$META" "$EXAMPLE_DIR/frames" "$EXAMPLE_DIR/voxel_grid.bin"
    if [ $? -ne 0 ]; then
        echo "[run] ray_voxel failed, aborting."
        exit 1
    fi
fi

# ── viewer ───────────────────────────────────────────────────────────────────
if [ "$HEADLESS" -eq 0 ]; then
    python src/viewVoxelMotion.py "$EXAMPLE_DIR/voxel_grid.bin"
fi