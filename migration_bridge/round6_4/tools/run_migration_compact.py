"""New-host bridge: both modes, four frozen prompts, repeats plus restart.

No equality requirement between hosts or between Deferred OFF/ON.
"""
import argparse
import json
from pathlib import Path
import time
from owned_server import OwnedServer
from run_compact_controls import checked, S6_SHA, PROMPTS, validate
from run_tokenizer_compat_probe import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-workers', type=int, choices=(16,32), default=16)
    args = parser.parse_args()
    corpus = checked(args.corpus)
    assert corpus['sha256'] == S6_SHA
    prompts = {p['id']:p for p in corpus['prompts']}
    args.output.mkdir(exist_ok=False)
    report = dict(status='RUNNING', started_unix=time.time(), rows=[],
                  scope='NEW_HOST_COMPACT_BRIDGE_NOT_FULL_QUALIFICATION')
    try:
        for profile in ('D1_NODEFER','D1'):
            references = {}
            for process in ('repeat','restart'):
                root = args.output/(profile+'-'+process)
                with OwnedServer(root, profile, tokenizer_compat=True, cpu_workers=args.cpu_workers):
                    with (root/'requests.jsonl').open('x') as raw:
                        for repeat in range(3 if process == 'repeat' else 1):
                            for pid in PROMPTS:
                                rid = f'r64-migrate-{profile}-{process}-{repeat}-{pid}'
                                payload = dict(rid=rid, input_ids=prompts[pid]['input_ids'],
                                    sampling_params=dict(temperature=0,max_new_tokens=64,ignore_eos=True),
                                    return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                                answer = generate(payload)
                                raw.write(json.dumps(dict(payload=payload,response=answer))+'\n')
                                raw.flush()
                                tokens = validate(answer,rid,64)
                                if process == 'repeat' and repeat == 0:
                                    references[pid] = tokens
                                else:
                                    assert tokens == references[pid], 'NEW_HOST_SAME_PATH_DRIFT'
                                row = dict(profile=profile,process=process,repeat=repeat,prompt=pid,exact=True)
                                report['rows'].append(row)
                                print(json.dumps(row),flush=True)
        report['status'] = 'MIGRATION_COMPACT_PASS'
    except BaseException as exc:
        report.update(status='FAILED',exception=repr(exc))
        raise
    finally:
        report['ended_unix'] = time.time()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
