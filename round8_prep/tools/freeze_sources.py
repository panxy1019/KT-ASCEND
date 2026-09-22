"""Small official-source snapshots; no submodule update or runtime import."""
import json
from freeze_metadata import fetch, ROOT

def main():
    out = ROOT/'02_OFFICIAL_AUDIT'
    # Frozen using git ls-remote and git ls-tree; GitHub REST rate-limited.
    pins = dict(kt='105359b4bf70073af9fab992ad29920666acccc9',
                sglang='9784d5f979c9eaa5a6d9b31e596a7a73cb76dab8',
                kt_sglang='3424f35d0e60f03b9e75d9b90941f8f4cb944524')
    out.mkdir(parents=True, exist_ok=True)
    pins['local_child'] = 'f02f304a25364d1e8fe9c52515a22f20f5089a52'
    pins['deployed_child'] = 'eaa58ccea4ee372b8b207dd16f398eafe3da37ff'
    (out/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    for file in ['doc/en/Qwen3.5.md','doc/en/kt-kernel/AVX2-Tutorial.md','kt-kernel/python/experts.py']:
        try: fetch(f'https://raw.githubusercontent.com/kvcache-ai/ktransformers/{pins["kt"]}/{file}',out/'kt'/file)
        except Exception as e: print(file,repr(e))
    for key,repo in [('kt_sglang','kvcache-ai/sglang'),('sglang','sgl-project/sglang')]:
        for file in ['python/sglang/srt/models/qwen3_5.py','python/sglang/srt/models/qwen2_moe.py',
                     'python/sglang/srt/models/qwen3_next.py','python/sglang/srt/configs/qwen3_5.py',
                     'python/sglang/srt/layers/moe/kt_ep_wrapper.py','python/sglang/srt/server_args.py',
                     'python/sglang/srt/hardware_backend/npu/quantization/loader.py','python/pyproject.toml']:
            try: fetch(f'https://raw.githubusercontent.com/{repo}/{pins[key]}/{file}',out/key/file)
            except Exception as e: print(key,file,repr(e))
    print(json.dumps(pins))

if __name__ == '__main__':main()
