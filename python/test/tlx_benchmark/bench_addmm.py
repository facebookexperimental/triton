"""Perf guardrail for tuned gfx942 ``tlx.ops.addmm``, against ``torch.addmm``.

Both providers consume the same BF16 ``A[M, K]``, column-major ``B[K, N]`` and
vector bias tensors and write to preallocated outputs. Run on MI300X with:

    python python/test/tlx_benchmark/bench_addmm.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.addmm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.addmm._shapes import SYNTHETIC, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "addmm"
REF_NAME = "torch.addmm"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = 0.05


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry[5]]).removeprefix("torch."),
            shape=tuple(entry[:5]),
            label=label(*entry),
        ) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import addmm as tlx_addmm

    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    bias, a, b = inputs((M, N, K, a_strides, b_strides, case.dtype), dtype)
    tlx_out = torch.empty((M, N), device="cuda", dtype=dtype)
    ref_out = torch.empty_like(tlx_out)

    tlx_fn = lambda: tlx_addmm(bias, a, b, out=tlx_out, arch=driver.arch(), space=space)  # noqa: E731
    ref_fn = lambda: torch.addmm(bias, a, b, out=ref_out)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
