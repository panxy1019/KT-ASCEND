#!/usr/bin/env python3
"""Run the verified full-logit sweep used as input to pairwise qualification.

The capture implementation remains in Round4A3 because it already enforces the
same-history invariant and verifies that the serving token matches the captured
serving logits. Round4A4 computes its top-32 union from the saved full logits,
not from the legacy top-16 manifest preview.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[2] / "round4a3" / "tools" / "run_teacher_forced_sweep.py"
    runpy.run_path(str(source), run_name="__main__")
