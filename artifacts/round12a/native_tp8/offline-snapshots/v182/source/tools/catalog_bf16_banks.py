"""Catalog already byte-verified banks and expose a flat immutable-path view.

This checks all48 per-layer receipts and live sizes; it does not repeat the
per-layer content verification or qualify a full-model run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    if args.catalog.exists():
        raise FileExistsError(args.catalog)
    rows = []
    identity = None
    for layer in range(48):
        folder = args.banks / f"layer{layer}-v1"
        receipt = folder / "manifest.json"
        item = json.loads(receipt.read_text())
        if (item["status"] != "PASS_SINGLE_LAYER_BF16_BANK_IDENTITY_ONLY"
                or item["layer"] != layer or item["storage"] != "BF16_ORIGINAL_BYTES"
                or item["exact_projection_instances"] != 1536
                or item["file"] != f"layer-{layer:02d}-bf16.gguf"):
            raise ValueError(f"unqualified bank: {layer}")
        if identity is None:
            identity = item["source_identity"]
            if identity["receipt_sha256"] != "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239":
                raise ValueError("unexpected original checkpoint identity")
        if item["source_identity"] != identity:
            raise ValueError("bank source identity mismatch")
        bank = folder / item["file"]
        if bank.stat().st_size != item["bytes"] or item["bytes"] != 5033165120:
            raise ValueError(f"bank size mismatch: {layer}")
        rows.append(dict(item, receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
    args.catalog.mkdir()
    for item in rows:
        target = args.banks / f"layer{item['layer']}-v1" / item["file"]
        (args.catalog / item["file"]).symlink_to(os.path.relpath(target.resolve(), args.catalog.resolve()))
    summary = dict(status="PASS_48_BANK_RECEIPTS_AND_LIVE_SIZES", source_identity=identity,
                   exact_projections_from_receipts=48*1536,
                   total_bank_bytes=sum(item["bytes"] for item in rows), banks=rows)
    (args.catalog / "catalog.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(summary["status"], summary["total_bank_bytes"])


if __name__ == "__main__":
    main()
