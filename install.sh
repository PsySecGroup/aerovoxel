#!/bin/bash
source activate.sh
pip install --upgrade pip
pip install numpy opencv-python matplotlib pybind11 setuptools astropy PyQt5
python setup.py build_ext --inplace
python setup.py install
python -c 'import process_image_cpp; print("SUCCESS: The C++ voxel tracker is working")'
mkdir -p fits
python generate_synthetic_frames.py
