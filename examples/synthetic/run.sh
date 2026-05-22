#!/bin/bash
source "scripts/activate.sh"

rm -f fits/* && python examples/synthetic/generate.py && python src/viewVoxelSpace.py