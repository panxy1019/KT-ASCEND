"""Default-OFF, batch1-only qualification adapter. Never modifies logits or KV.

The scheduler receives the returned forced IDs through its ordinary path. Each
process owns its own cache. The manifest is hashed before startup; arming occurs
only after the non-generating readiness and empty-capture checks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class HistoryLedger:
    """Pure history checker, also usable without torch for negative tests."""

    def __init__(self, manifest):
        if manifest.get("version") != "r63-forced-history-v1":
            raise ValueError("R63_MANIFEST_VERSION")
        self.session = manifest["session_id"]
        self.requests = {}
        self.steps = {}
        for request in manifest["requests"]:
            rid = request["rid"]
            if not isinstance(rid, str) or not rid or rid in self.requests:
                raise ValueError("R63_DUPLICATE_OR_INVALID_RID")
            for field in ("input_ids", "baseline_output_ids"):
                values = request[field]
                if not isinstance(values, list) or not values or any(
                    type(t) is not int or not 0 <= t < 102400 for t in values
                ):
                    raise ValueError("R63_INVALID_TOKEN_LIST")
            self.requests[rid] = request
        if not self.requests:
            raise ValueError("R63_EMPTY_MANIFEST")

    def observe(self, rid, input_ids, positions, seq_len, is_decode, natural_token):
        if rid not in self.requests:
            raise ValueError("R63_UNKNOWN_RID")
        request = self.requests[rid]
        prompt, tokens = request["input_ids"], request["baseline_output_ids"]
        step = self.steps.get(rid, 0)
        if step > len(tokens):
            raise ValueError("R63_EXTRA_FORWARD")
        expected_ids = prompt if step == 0 else [tokens[step - 1]]
        expected_positions = list(range(len(prompt))) if step == 0 else [len(prompt) + step - 1]
        if bool(is_decode) != (step > 0):
            raise ValueError("R63_FORWARD_MODE_OR_RESTART")
        if input_ids != expected_ids or positions != expected_positions or seq_len != len(prompt) + step:
            raise ValueError("R63_ACTUAL_CONSUMED_HISTORY_MISMATCH")
        if type(natural_token) is not int or not 0 <= natural_token < 102400:
            raise ValueError("R63_INVALID_NATURAL_TOKEN")
        # Production overlap can execute one extra decode. Account for it, but
        # never include its logits/exposure in qualification or force beyond N.
        lookahead = step == len(tokens)
        forced = natural_token if lookahead else tokens[step]
        self.steps[rid] = step + 1
        return dict(session_id=self.session, rid=rid, step=step,
                    classification="LOOKAHEAD" if lookahead else "TARGET",
                    input_ids=input_ids, positions=positions, seq_len=seq_len,
                    history_sha256=digest(prompt + tokens[:step]),
                    natural_token=natural_token, returned_token=forced,
                    forced=not lookahead)


_ledger = None
_trace = None
_manifest_sha = None
_sequence = 0


def apply_forced_history(next_token_ids, logits_output, forward_batch):
    """Called after ordinary sampling, before scheduler state advancement."""
    global _ledger, _trace, _manifest_sha, _sequence
    manifest_path = os.getenv("KT_R63_FORCE_MANIFEST")
    if not manifest_path:
        return next_token_ids
    arm_path = os.environ["KT_R63_FORCE_ARM_FILE"]
    if not Path(arm_path).exists():
        if _ledger is not None:
            raise RuntimeError("R63_DISARM_DURING_RUN")
        return next_token_ids
    if _ledger is None:
        raw = Path(manifest_path).read_bytes()
        _manifest_sha = hashlib.sha256(raw).hexdigest()
        if _manifest_sha != os.environ["KT_R63_FORCE_MANIFEST_SHA256"]:
            raise RuntimeError("R63_MANIFEST_HASH_MISMATCH")
        _ledger = HistoryLedger(json.loads(raw))
        if Path(arm_path).read_text().strip() != _manifest_sha:
            raise RuntimeError("R63_ARM_IDENTITY_MISMATCH")
        # Exclusive creation: a retry must have a fresh trace path/process.
        _trace = open(os.environ["KT_R63_FORCE_TRACE"], "x", buffering=1)
    import torch
    rids = list(forward_batch.rids or [])
    logits = logits_output.next_token_logits
    if len(rids) != 1 or tuple(logits.shape) != (1, 102400) or next_token_ids.numel() != 1:
        raise RuntimeError("R63_BATCH1_REQUIRED")
    if not forward_batch.sampling_info.is_all_greedy:
        raise RuntimeError("R63_GREEDY_REQUIRED")
    # Fail before injecting anything if ordinary sampling is inconsistent.
    natural = int(next_token_ids.item())
    if not bool(torch.isfinite(logits).all()) or logits[0, natural] != logits[0].max():
        raise RuntimeError("R63_NONFINITE_OR_NONMAX_NATURAL_SAMPLE")
    record = _ledger.observe(
        rids[0], forward_batch.input_ids.reshape(-1).tolist(),
        forward_batch.positions.reshape(-1).tolist(),
        int(forward_batch.seq_lens.item()), forward_batch.forward_mode.is_decode(), natural,
    )
    record.update(pid=os.getpid(), sequence=_sequence, manifest_sha256=_manifest_sha,
                  cache_lineage=f"{_ledger.session}:{os.getpid()}:{rids[0]}",
                  sampled_logits_sha256=hashlib.sha256(logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    _trace.write(json.dumps(record, sort_keys=True) + "\n")
    _sequence += 1
    return torch.full_like(next_token_ids, record["returned_token"])
