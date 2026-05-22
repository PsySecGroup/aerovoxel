#!/usr/bin/env python3
from pathlib import Path
import sys
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent))
from generator import generate
generate(HERE)