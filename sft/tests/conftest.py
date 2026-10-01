"""Make the SFT project root importable from nested test directories."""

import sys
from pathlib import Path

SFT = Path(__file__).parents[1]
sys.path[:0] = [str(SFT), str(SFT.parent)]
