#!/bin/bash
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
rm -f $HERE/voxel_grid.bin
rm -f $HERE/metadata.json
#rm -f $HERE/frames/*.png