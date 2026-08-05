from __future__ import annotations

import sys
from pathlib import Path

CONVERSION_ROOT = Path(__file__).resolve().parents[2] / "tools/conversion"
sys.path.insert(0, str(CONVERSION_ROOT))
