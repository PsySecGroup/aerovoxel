#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Build with g++
mkdir -p build
g++ -std=c++17 -O3 -fopenmp -I vendor -I src src/faster_ray_voxel.cpp -o build/ray_voxel
if [ $? -ne 0 ]; then
    echo "[Error] Compilation failed"
    exit 1
fi

echo "[Info] Compilation succeeded."

