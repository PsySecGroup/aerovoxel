# AeroVoxel

Projects motion of pixels to a voxel

## Execution

```bash
./scripts/build.sh`
# Usage: ray_voxel <metadata.json> <image_folder> <output_voxel_bin>
`./build/ray_voxel metadata.json images build/voxel_grid.bin
```

## Examples

```bash
./examples/synthetic/run.sh
```

## Development

```bash
./scripts/install.sh
. scripts/activate.sh
```

## Todo

Figure out PixelationDecensorer.py

## Clean Run
./script/build-faster
./examples/afternoon/clean.sh
./examples/afternoon/run.sh
./examples/afternoon/run.sh --force # Forces a rerun of the same binary
./examples/afternoon/run.sh --headless # Doesn't call voxel motion preview

python src/viewVoxelMotion.py examples/afternoon/voxel_grid.bin
python src/binToCsv.py examples/afternoon/voxel_grid.bin output.csv
