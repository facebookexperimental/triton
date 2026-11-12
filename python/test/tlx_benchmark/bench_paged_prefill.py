"""Useful TFLOP/s for BF16 paged causal GQA, without a dense-input speedup claim.

Shuffled page tables and both global/SWA2048 use the public operator. The FP64
check samples query boundaries; the unit suite checks every row through 16K.
"""

import pathlib
import sys

import torch
import triton
from triton.tlx.ops import paged_prefill
from triton.tlx.ops.kernels.paged_prefill._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.paged_prefill._shapes import SYNTHETIC, flops

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import Case, Prepared, driver

OP = "paged_prefill"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
COLD_COMPILE = "first"
EXTRA_COLUMNS = (("latency us", "latency_us"), )


def shapes(synthetic=False, suites=None):
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic=False, suites=None):
    return [
        Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=shape,
             label=f"B={shape[0]} Q={shape[1]} KV={shape[2]} window={shape[3]}") for shape in shapes(synthetic, suites)
    ]


def prepare(case, space):
    batch, sq, sk, window = case.shape
    generator = torch.Generator(device="cuda").manual_seed(41)
    pages = triton.cdiv(sk, 64)
    q = torch.randn((batch, sq, 16, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch * pages + 3, 64, 2, 64), generator=generator, device=q.device, dtype=q.dtype)
    v = torch.randn(k.shape, generator=generator, device=q.device, dtype=q.dtype)
    table = torch.randperm(k.shape[0], generator=generator, device=q.device, dtype=torch.int32)
    table = table[:batch * pages].reshape(batch, pages)
    fn = lambda: paged_prefill(q, k, v, table, kv_length=sk, window=window, space=space)

    def check():
        out = fn()
        if not torch.isfinite(out).all():
            return False, "nonfinite output"
        rows = torch.tensor(sorted({x
                                    for x in (0, 7, 8, 15, 16, 31, 32, 63, 64, 2047, 2048, sq - 1)
                                    if x < sq}), device=q.device)
        keys = torch.arange(sk, device=q.device)
        end = rows[:, None] + sk - sq
        mask = keys[None, :] <= end
        if window:
            mask &= keys[None, :] > end - window
        error_sq = torch.zeros((), device=q.device, dtype=torch.float64)
        ref_sq = torch.zeros_like(error_sq)
        for b in range(batch):
            kk = k[table[b].long()].reshape(-1, 2, 64)[:sk].double()
            vv = v[table[b].long()].reshape(-1, 2, 64)[:sk].double()
            for h in range(16):
                scores = q[b, rows, h].double() @ kk[:, h // 8].T * 0.125
                ref = scores.masked_fill(~mask, -float("inf")).softmax(-1) @ vv[:, h // 8]
                actual = out[b, rows, h].double()
                if not torch.allclose(actual, ref, atol=0.025, rtol=0.025):
                    return False, "FP64 elementwise tolerance exceeded"
                error_sq += (actual - ref).square().sum()
                ref_sq += ref.square().sum()
        rrms = (error_sq / ref_sq).sqrt().item()
        return rrms < 0.005, f"FP64 relative RMS {rrms:.5f}"

    return Prepared(tlx_fn=fn, ref_fn=None, flop_count=flops(*case.shape), check=check)


def annotate(result):
    if result.tlx and result.flop_count:
        result.extra["latency_us"] = result.flop_count / (result.tlx.mean * 1e6)


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
