#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
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
META="$HERE/metadata.json"
FRAME_COUNT=$(ls "$HERE"/frames/*.png 2>/dev/null | wc -l)

if [ ! -f "$META" ] || [ "$FRAME_COUNT" -eq 0 ]; then
    echo "[run] Frames or metadata missing, running generate..."
    python "$HERE/generate.py"
    if [ $? -ne 0 ]; then
        echo "[run] generate.py failed, aborting."
        exit 1
    fi
fi

# ── ray_voxel ───────────────────────────────────────────────────────────────
echo "[run] Running ray_voxel..."
"$PROJECT_ROOT/build/ray_voxel" "$META" "$HERE/frames" "$HERE/voxel_grid.bin"
python src/viewVoxelMotion.py "$HERE/voxel_grid.bin"