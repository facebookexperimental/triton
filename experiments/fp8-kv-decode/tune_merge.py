"""Numerically gated merge-only warp sweep over real attention partials."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
from bench_decode import inputs, reference, timing, tlx_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for splits in (36, 128, 256, 384, 512, 1024):
        args = SimpleNamespace(page=64, length=2304 if splits == 36 else 262144, heads=16, kv_heads=2, dim=64, tokens=1,
                               fp8_q=False, scales='per-head', window=0, causal=True)
        x = inputs(args)
        expected = reference(x, args)
        mid = torch.empty((1, 16, splits, 64), device='cuda')
        lse = torch.empty((1, 16, splits), device='cuda')
        out = torch.empty_like(x.q, dtype=torch.bfloat16)
        source = tlx_source()
        source.fp8_kv_decode(x.q, x.kp, x.vp, x.contexts, x.table, x.ks, x.vs, max_context_len=args.length,
                             variant='matrix_tdm_flat', block_n=64, num_splits=splits, num_warps=1,
                             workspace=(mid, lse), out=out)
        if splits >= 256:
            mid = mid.view(torch.float16)
        merge = source._optimized_module()._merge
        for w in (1, 2, 4, 8, 16):
            if splits * 64 / (w * 32) > 256: continue
            row = dict(splits=splits, warps=w)
            try:
                fn = lambda: merge[
                    (16, )](mid, lse, out, 64, splits, triton.next_power_of_2(splits), x.vs, 16, 8, num_warps=w)
                fn()
                torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-2)
                row.update(status='pass', **timing(fn, 15, 20))
            except Exception as exc:
                row.update(status='failed', reason=str(exc))
            rows.append(row)
            print(json.dumps(row), flush=True)
            a.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__': main()
