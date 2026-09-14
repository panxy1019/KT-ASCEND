#!/usr/bin/env python3
"""Compute frozen-contract ambiguity sets and membership.

The shared validator emits the full per-position classification, set sizes,
probability mass, failures, and a canonical evidence hash.
"""

from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("validate_h2_pairwise_contract.py")), run_name="__main__")
