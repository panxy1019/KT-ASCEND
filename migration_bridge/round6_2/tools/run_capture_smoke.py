"""Twenty single-token requests on the five frozen non-qualification prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import torch
from tokenizers import Tokenizer
from assemble_capture import assemble, canonical_sha, byte_sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--mode",choices=["defer0","defer1"],required=True)
    p.add_argument("--capture",choices=["on","reference"],required=True)
    p.add_argument("--url",default="http://127.0.0.1:31000")
    args=p.parse_args()
    if (args.root/"smoke_result.json").exists():
        raise ValueError("refuse smoke overwrite")
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    prompts=json.loads(args.source.read_text())["prompts"]
    results=[]
    vectors=[]
    all_sequences=[]
    for index in range(20):
        text=prompts[index%len(prompts)]
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        before=set((args.root/"tensors").rglob("Pass*.pt"))
        request_id=f"r62-smoke-{args.mode}-{args.capture}-{index:03d}"
        body=dict(rid=request_id,input_ids=ids,
                  sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True),
                  return_logprob=True,logprob_start_len=-1,top_logprobs_num=16)
        request=urllib.request.Request(args.url+"/generate",data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(request,timeout=120) as response:
            answer=json.load(response)
        if answer["meta_info"]["id"] != request_id:
            raise ValueError("API_REQUEST_ID_CHANGED")
        deadline=time.monotonic()+30
        while True:
            files=sorted(set((args.root/"tensors").rglob("Pass*.pt"))-before)
            if files:
                break
            if time.monotonic()>deadline:
                raise TimeoutError("capture commit timeout")
            time.sleep(.1)
        matches=[]
        for path in files:
            data=torch.load(path,map_location="cpu",weights_only=False)
            tx=data["capture.transaction"]
            all_sequences.append((tx["pid"],tx["forward_sequence_id"]))
            if tx["batch"]["request_ids"] != [request_id]:
                continue
            if args.capture=="on":
                got=data["capture.forward_batch_info.input_ids"].reshape(-1).tolist()
                pos=data["capture.forward_batch_info.positions"].reshape(-1).tolist()
                if got != ids or pos != list(range(len(ids))):
                    continue
            matches.append((path,data))
        if not matches or (args.capture=="on" and len(matches)!=1):
            raise ValueError("MISSING_OR_DUPLICATE_REQUEST_FORWARD")
        path,data=matches[0]
        record=dict(request_id=request_id,prompt_id=f"smoke_{index%5}",
            prompt_hash=hashlib.sha256(text.encode()).hexdigest(),input_ids=ids,
            history_ids=[],baseline_history_sha256=canonical_sha([]),token_index=0,
            output_token=int(answer["output_ids"][0]))
        if args.capture=="on":
            row,logits=assemble(path,record,args.mode)
        else:
            logits=data["capture.next_token_logits"][-1].contiguous()
            if not torch.isfinite(logits).all() or float(logits.max()-logits[record["output_token"]])!=0:
                raise ValueError("REFERENCE_API_LOGIT_MISMATCH")
            row=dict(logits_sha256=byte_sha(logits),capture_file=str(path))
        row.update(index=index,output_token=record["output_token"],request=record)
        results.append(row)
        vectors.append(logits)
        print(json.dumps(dict(completed=index+1,mode=args.mode,capture=args.capture,
            token=record["output_token"],exposure=row.get("deferred_exposure"))),flush=True)
    if len(all_sequences)!=len(set(all_sequences)) or all_sequences!=sorted(all_sequences):
        raise ValueError("NON_MONOTONIC_FORWARD_SEQUENCE")
    torch.save(torch.stack(vectors),args.root/"smoke_logits.pt")
    result=dict(mode=args.mode,capture=args.capture,requests=len(results),
        complete_forwards=len(all_sequences),rows=results,
        deferred_route_count=sum(r.get("deferred_event_count_current_row",0)+r.get("deferred_event_count_prefix",0) for r in results))
    (args.root/"smoke_result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
