from pathlib import Path
import sys
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent))
from generate import run
run(HERE)