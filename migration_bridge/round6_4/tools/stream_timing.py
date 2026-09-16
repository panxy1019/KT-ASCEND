"""Native cumulative SSE timing; per-request output only, no forward disk trace."""
import json
import time
import urllib.request
from run_compact_controls import validate


def consume(lines, rid, budget, started, clock=time.monotonic):
    final, previous = None, []
    first = last = None
    events, cumulative_counts = 0, []
    done = False
    for raw in lines:
        line = raw.decode().strip() if isinstance(raw, bytes) else raw.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ValueError("UNEXPECTED_SSE_LINE")
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("meta_info",{}).get("id") != rid:
            raise ValueError("STREAM_RID")
        ids = event["output_ids"]
        if ids[:len(previous)] != previous or len(ids) > budget:
            raise ValueError("STREAM_TOKEN_HISTORY")
        stamp = clock()
        if len(ids) > len(previous):
            if first is None:
                first = stamp
            last = stamp
            cumulative_counts.append(len(ids))
        previous, final = ids, event
        events += 1
    ended = clock()
    if not done or final is None or first is None or last is None:
        raise ValueError("INCOMPLETE_STREAM")
    validate(final,rid,budget)
    if final["meta_info"]["finish_reason"]["type"] != "length":
        raise ValueError("FIXED_BUDGET_NOT_COMPLETED")
    return dict(response=final, seconds=ended-started, ttft_seconds=first-started,
                e2e_tokens_per_second=budget/(ended-started),
                decode_tokens_per_second=(budget-1)/(last-first) if budget>1 and last>first else None,
                timing_scope="client_observed_streaming", events=events,
                cumulative_token_counts=cumulative_counts)


def request(rid, input_ids, budget):
    payload = dict(rid=rid, input_ids=input_ids, sampling_params=dict(temperature=0,max_new_tokens=budget,ignore_eos=True),
                   stream=True, return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
    req = urllib.request.Request("http://127.0.0.1:31000/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req,timeout=900) as response:
        return consume(response,rid,budget,started)
