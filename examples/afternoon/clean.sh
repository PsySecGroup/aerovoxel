#!/bin/bash
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
rm $HERE/voxel_grid.bin
rm $HERE/voxel_grid_multicam.json
rm $HERE/metadata.json
rm $HERE/frames/*.png