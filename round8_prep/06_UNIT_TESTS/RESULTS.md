# Local test evidence

2026-09-18, independent prep worktree, Python3.13 stdlib venv.

Commands:

```bash
python3 round8_prep/tools/prepare_static.py
round8_prep/07_BUILD/venv/bin/python -m unittest discover -s round8_prep/06_UNIT_TESTS -v
```

Result: metadata SHA verification/config audit PASS; 7/7 tests PASS (architecture, payload, mapping/ownership, nested masks, invalid budget rejection, missing-inventory rejection, CPU/NPU conflict rejection and valid synthetic admission).

No actual KTEP/model construction, CPU kernel, NPU operation or weight identity numerical test. Static expected mapping does not certify production remapping. Test inventory is synthetic, not live resource admission.
