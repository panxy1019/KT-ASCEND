"""Isolated A3 cache feasibility probe, never a model performance claim."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', layers=[], production_enabled=False,
                  scope='ISOLATED_STAGING_ONLY_NOT_MODEL_THROUGHPUT')
    try:
        import torch
        import torch_npu
        from kt_kernel.utils.loader import GGUFLoader
        from kt_kernel.utils.llamafile import LlamafileMoEWrapper
        from llamafile_bf16_staging import LlamafileBF16Stager
        from verified_bf16_cache import cached_stager_class
        from dynamic_audit import checkpoint_projection
        torch.npu.set_device(0)
        report['torch_threads'] = torch.get_num_threads()
        loader = GGUFLoader('/workspace/results/r65-assets-v1/all-moe-f32.gguf')
        cls = cached_stager_class(LlamafileBF16Stager)
        for layer in (1, 13, 26):
            wrapper = LlamafileMoEWrapper.__new__(LlamafileMoEWrapper)
            wrapper.num_experts = 64
            wrapper.hidden_size = 2048
            wrapper.moe_intermediate_size = 1408
            wrapper.layer_idx = layer
            wrapper.gguf_loader = loader
            buffers = dict(w13_weight=torch.empty((2,2816,2048), dtype=torch.bfloat16, pin_memory=True),
                           w2_weight=torch.empty((2,2048,1408), dtype=torch.bfloat16, pin_memory=True))
            experts = [e for e in range(64) if e % 8 != 0]

            def run(kind):
                # Reconstruct the stager just as each full-context prepare does.
                writer = kind(wrapper, buffers, 1)
                start = time.perf_counter()
                for i, expert in enumerate(experts):
                    writer.submit(expert, i % 2)
                return time.perf_counter()-start, writer

            baseline, plain = run(LlamafileBF16Stager)
            cold, cached = run(cls)
            warm = [run(cls)[0] for _ in range(3)]
            # Compare every CPU-owned expert to the original writer, with the
            # same pinned output slots; readbacks are outside timed sections.
            for expert in experts:
                plain.submit(expert, 0)
                expected = {k:v[0].clone() for k,v in buffers.items()}
                cached.submit(expert, 0)
                assert all(torch.equal(buffers[k][0],v) for k,v in expected.items()), 'CACHE_OUTPUT_CHANGED'
            for expert in (1, 63):
                cached.submit(expert, 0)
                for name, buf in buffers.items():
                    ref = checkpoint_projection(layer, expert, name)
                    assert torch.equal(buf[0],ref), 'CACHE_CHECKPOINT_IDENTITY'
                    out = torch.empty_like(buf[0],device='npu')
                    out.copy_(buf[0],non_blocking=True)
                    torch.npu.synchronize()
                    assert torch.equal(out.cpu(),ref), 'CACHE_H2D_IDENTITY'
            # Controlled tensor replacement must reject stale cache reuse.
            original = cached.projections['gate']
            cached.projections['gate'] = original.clone()
            try:
                cached.submit(1,0)
            except ValueError as error:
                assert str(error)=='BF16_CACHE_SOURCE_BINDING_CHANGED'
            else:
                raise AssertionError('STALE_CACHE_ACCEPTED')
            report['layers'].append(dict(layer=layer,experts=len(experts),
                baseline_seconds=baseline,cold_seconds=cold,warm_seconds=warm,
                warm_speedup=baseline/statistics.median(warm),
                cache_bytes=cached.cache.resident_bytes,output_exact=True,
                checkpoint_and_h2d_exact=True,stale_binding_rejected=True))
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            del cached, plain, wrapper, buffers, original, out, expected, ref
            gc.collect()
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
