#!/usr/bin/env python3
"""Run the retained serialized post-FC recovery prompt sequence."""

from __future__ import annotations

import sustained_strict_v3


sustained_strict_v3.DEFAULT_TYPED_PROMPTS = (
    "Call get_current_utc_time exactly twice using two separate function calls. "
    "After the second result, say only: done.",
    "Without calling any tool, say only: five.",
    "Call get_current_utc_time exactly twice again using two separate function calls. "
    "After the second result, say only: done.",
)


if __name__ == "__main__":
    sustained_strict_v3.main()
