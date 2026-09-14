#!/usr/bin/env python3
"""Classify positions with a frozen pairwise contract.

This is a named entry point for the classification stage; the validation
implementation also checks exact stable ordering and ambiguity membership.
"""

from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("validate_h2_pairwise_contract.py")), run_name="__main__")
