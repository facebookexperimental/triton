"""Perf guardrail for Hopper ``tlx.ops.addmm_mul_add``, the fused DCN cross layer ``x0 * (bias + a @ b) + res``.

The reference is the unfused sequence a torch caller runs: ``torch.addmm`` into a preallocated buffer, then one
``torch.addcmul`` for the mul/add. Run on H100 with:

    python python/test/tlx_benchmark/bench_addmm_mul_add.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.addmm_mul_add._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.addmm_mul_add._shapes import SYNTHETIC, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "addmm_mul_add"
REF_NAME = "torch.addmm + torch.addcmul"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = 0.02


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry.dtype]).removeprefix("torch."), shape=tuple(entry[:-1]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import addmm_mul_add

    m, n, k, has_res = case.shape
    dtype = getattr(torch, case.dtype)
    bias, a, b, x0, res = inputs((m, n, k, has_res, case.dtype), dtype)
    tlx_out = torch.empty_like(x0)
    lin = torch.empty_like(x0)
    ref_out = torch.empty_like(x0)

    def ref_fn():
        torch.addmm(bias, a, b, out=lin)
        return torch.addcmul(x0 if res is None else res, x0, lin, out=ref_out)

    tlx_fn = lambda: addmm_mul_add(bias, a, b, x0, res, out=tlx_out, space=space)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(m, n, k),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
