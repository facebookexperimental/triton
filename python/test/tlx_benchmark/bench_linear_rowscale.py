"""Perf and compile-time reporting for ``tlx.ops.linear_rowscale`` on Hopper.

The reference is ``torch.compile`` of the eager expression, which Inductor lowers to cuBLAS ``addmm`` followed by one
fused kernel that applies the row scale and rounds.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.linear_rowscale._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.linear_rowscale._shapes import SYNTHETIC, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "linear_rowscale"
REF_NAME = "torch.compile(eager): cuBLAS addmm + scale kernel"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry.dtype]).removeprefix("torch."), shape=tuple(entry),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def _eager(x, weight, bias, row_scale):
    return (torch.addmm(bias, x, weight.t()).float() * row_scale[:, None]).to(x.dtype)


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import linear_rowscale

    del space  # One fixed launch config.
    M, N, K, _ = case.shape
    x, weight, bias, row_scale = inputs(case.shape)
    compiled = torch.compile(_eager, dynamic=False)
    ref_fn = lambda: compiled(x, weight, bias, row_scale)  # noqa: E731
    ref_fn()  # compiled here, ahead of the harness's cold-compile pass, which times only the op
    tlx_fn = lambda: linear_rowscale(x, weight, bias, row_scale)  # noqa: E731

    def check():
        ref = (torch.addmm(bias.float(), x.float(), weight.float().t()) * row_scale[:, None]).to(x.dtype)
        return close_enough(tlx_fn(), ref, REL_PRECISION[case.dtype])

    return Prepared(tlx_fn=tlx_fn, ref_fn=ref_fn, flop_count=flops(M, N, K), check=check)


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
