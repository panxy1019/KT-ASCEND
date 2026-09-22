"""Predeclare independent fixed-length timing shapes, not a quality corpus."""
import argparse
import hashlib
import json
from pathlib import Path

TOKENIZER = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
TEXT = ("A public workshop keeps records of materials, tools, and completed repairs. "
        "Volunteers sort incoming objects, describe their condition, and agree on a safe repair sequence. "
        "The record distinguishes an observation from a proposed explanation and from a completed action. ")
ENDING = "\nSummarize the notes and explain how clear records help the volunteers coordinate their work.\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() == TOKENIZER
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    body = tokenizer.encode(TEXT*30,add_special_tokens=False).ids
    ending = tokenizer.encode(ENDING,add_special_tokens=False).ids
    shapes = []
    for name,length in (("short",64),("medium",192),("long",512)):
        ids = body[:length-len(ending)]+ending
        assert len(ids) == length
        for budget in (64,256):
            shapes.append(dict(id=f"{name}-{length}-{budget}", input_ids=ids, input_tokens=length, output_tokens=budget))
    payload = dict(version="r64-performance-v1", status="FROZEN_BEFORE_FORMAL_TIMING", tokenizer_sha256=TOKENIZER,
        primary_shape="medium-192-256", primary_batch=1, shapes=shapes, paired_blocks=5,
        phase_order=["D1_NODEFER","D1","D1","D1_NODEFER"], warmup_shape="medium-192-256", warmup_requests_per_phase=2,
        ignore_eos=True, cache="OFF", teacher_force=False, observer=False, route_recorder=False,
        confidence_method="two_sided_95_student_t_on_five_independent_log_block_ratios",
        speedup_rule="primary_lower_confidence_bound_gt_1", other_shapes="descriptive_not_selection",
        limitations="Fixed-length streaming client timing; block independence/log-ratio approximation; shared-host confounding reviewed separately")
    payload["sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    with args.output.open("x") as out:
        json.dump(payload,out,indent=2)


if __name__ == "__main__":
    main()
