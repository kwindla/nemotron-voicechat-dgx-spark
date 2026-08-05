from __future__ import annotations

import sys
from pathlib import Path

RELEASE_TOOLS = Path(__file__).resolve().parents[2] / "tools/release"
sys.path.insert(0, str(RELEASE_TOOLS))
