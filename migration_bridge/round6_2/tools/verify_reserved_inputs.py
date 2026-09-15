"""Read-only reserved FD/quality provenance checks; never execute the model."""
import argparse
import hashlib
import json
from pathlib import Path
from freeze_corpora import canonical_sha, normalize


def checked(path):
    value = json.loads(path.read_text())
    if canonical_sha({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
        raise ValueError(f"FROZEN_HASH_MISMATCH: {path}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--fd", type=Path, required=True)
    parser.add_argument("--qd2", type=Path, required=True)
    parser.add_argument("--hd2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse input verification overwrite")
    quality, fd, qd, hd = [checked(path) for path in (args.quality, args.fd, args.qd2, args.hd2)]
    if quality["sha256"] != "c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce":
        raise ValueError("QUALITY_REVISION_DRIFT")
    counts = {key: len(spec["samples"]) for key, spec in quality["benchmarks"].items()}
    if len(counts) != 2 or set(counts.values()) != {128}:
        raise ValueError("QUALITY_COUNT_DRIFT")
    reserved = {normalize(p["text"]) for p in fd["prompts"]}
    reserved.update(normalize(p["prompt"]) for spec in quality["benchmarks"].values() for p in spec["samples"])
    for corpus in (qd, hd):
        if any(normalize(prompt["text"]) in reserved for prompt in corpus["prompts"]):
            raise ValueError("RESERVED_CORPUS_OVERLAP")
    report = dict(status="INPUT_PROVENANCE_VERIFIED_ONLY", quality_counts=counts,
                  fd_count=len(fd["prompts"]), no_exact_reserved_prompt_overlap=True,
                  model_inference=False,
                  files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.quality, args.fd, args.qd2, args.hd2)},
                  canonical=dict(quality=quality["sha256"], fd=fd["sha256"], qd2=qd["sha256"], hd2=hd["sha256"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
