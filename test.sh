#!/bin/bash
rm -f fits/* && python generate_synthetic_frames.py && python spacevoxelviewer.py