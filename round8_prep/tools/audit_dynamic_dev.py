"""Read-only independent DEV artifact audit; does not promote candidates."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    report = {"qualification_entry": False, "candidates": {}}
    for threshold in (256, 512):
        folder = root / f"q32-dynamic-{threshold}-v1"
        events = [json.loads(x) for x in (folder / "generation-audit.jsonl").read_text().splitlines()]
        assert len(events) == 120
        per_layer = {i: [e for e in events if e["layer"] == i] for i in range(40)}
        for layer, rows in per_layer.items():
            assert [r["weights"] for r in rows] == [1, 2, 3], layer
            assert rows[0]["selected"] == rows[1]["selected"], layer
            for row in rows:
                assert row["weights"] == row["mapping"] == row["mask"] == row["wrapper_mask"]
                assert row["weight_identity_exact"] and row["mask_mapping_wrapper_exact"]
                assert len(set(row["selected"])) == 32
        report["candidates"][str(threshold)] = dict(
            update_events=len(events), layers=len(per_layer),
            actual_prefill_rows=sorted({e["prefill_rows"] for e in events}),
            changed_layers=sum(rows[1]["selected"] != rows[2]["selected"] for rows in per_layer.values()),
            recorded_invariants="PASS", warning_closed=False,
            evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(folder.iterdir()) if p.is_file()})
    folder = root / "q32-dynamic-1024-v1"
    state = json.loads((folder / "lane-state.json").read_text())
    assert state["status"] == "FAILED"
    report["candidates"]["1024"] = dict(status="FAILED", reason="TRIGGER_NOT_COVERED_WITH_CHUNK_512",
        state_sha256=hashlib.sha256((folder / "lane-state.json").read_bytes()).hexdigest())
    report["remaining"] = ["storage_offset attribution", "clean restart", "observer parity",
                           "DEV candidate selection and committed freeze", "stability final audit"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
