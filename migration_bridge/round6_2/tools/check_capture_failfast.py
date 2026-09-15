"""Exercise production dumper fail-fast checks with real CPU tensors on A3."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import torch
from sglang.srt.debug_utils.tensor_dump_forward_hook import TensorDumper
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse test overwrite")
    os.environ["KT_R62_COMPLETE_FORWARD_DUMP"] = "1"
    batch = ForwardBatch.__new__(ForwardBatch)
    batch.input_ids = torch.tensor([7], dtype=torch.int64)
    batch.positions = torch.tensor([0], dtype=torch.int64)
    batch.seq_lens = torch.tensor([1], dtype=torch.int32)
    batch.rids = ["failfast-only"]
    batch.forward_mode = "TEST_CPU_ONLY"
    output = LogitsProcessorOutput(next_token_logits=torch.ones((1, 102400)))
    results = {}
    with tempfile.TemporaryDirectory(prefix="r62-capture-failfast-") as directory:
        def expect(name, action, phrase):
            try:
                action()
            except RuntimeError as error:
                if phrase not in str(error):
                    raise
                results[name] = dict(status="PASS", observed_error=str(error))
            else:
                raise AssertionError(f"missing expected rejection: {name}")
        dumper = TensorDumper(directory, [999], 1, 0, 0)
        dumper._current_tensors["stale"] = torch.ones(1)
        expect("unconsumed_state", lambda: dumper.begin_complete_forward(None, (batch,), {}), "unflushed")
        dumper._current_tensors.clear()
        expect("missing_entry", lambda: dumper.end_complete_forward(None, (), {}, output), "without matching entry")
        dumper.begin_complete_forward(None, (batch,), {})
        dumper.end_complete_forward(None, (), {}, output)
        dumper._forward_pass_id = 0
        expect("duplicate_forward", lambda: dumper.begin_complete_forward(None, (batch,), {}), "duplicate forward")
        dumper._forward_pass_id = 1
        dumper.begin_complete_forward(None, (batch,), {})
        expect("missing_logits", lambda: dumper.end_complete_forward(None, (), {}, torch.ones(1)), "LogitsProcessorOutput")
    report = dict(status="PASS", execution="A3 container CPU-only real torch tensors", tests=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
