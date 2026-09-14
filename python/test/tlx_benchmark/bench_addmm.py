"""Perf guardrail for tuned gfx942 ``tlx.ops.addmm``, against ``torch.addmm``.

Both providers consume the same BF16 ``A[M, K]``, column-major ``B[K, N]`` and
vector bias tensors and write to preallocated outputs. Run on MI300X with:

    python python/test/tlx_benchmark/bench_addmm.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

from triton.tlx.ops.kernels.mm.gfx942 import TUNED_CONFIGS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "addmm"
REF_NAME = "torch.addmm"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPE = torch.bfloat16
REL_PRECISION = 0.05
ADDMM_SHAPES = tuple(shape for shape in TUNED_CONFIGS if shape != (2048, 25408, 10240))


def cases(synthetic: bool = False) -> list[Case]:
    if synthetic:
        return []
    return [
        Case(
            op=OP,
            arch="gfx942",
            dtype="bfloat16",
            shape=(M, N, K),
            label=f"{M}x{N}x{K} A:row-major B:column-major bias:N",
        ) for M, N, K in ADDMM_SHAPES
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import addmm as tlx_addmm

    M, N, K = case.shape
    a = torch.randn((M, K), device="cuda", dtype=DTYPE)
    weight = torch.randn((N, K), device="cuda", dtype=DTYPE)
    b = weight.T
    bias = torch.randn((N, ), device="cuda", dtype=DTYPE)
    tlx_out = torch.empty((M, N), device="cuda", dtype=DTYPE)
    ref_out = torch.empty_like(tlx_out)

    tlx_fn = lambda: tlx_addmm(bias, a, b, out=tlx_out, arch=driver.arch(), space=space)  # noqa: E731
    ref_fn = lambda: torch.addmm(bias, a, b, out=ref_out)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=2 * M * N * K,
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
