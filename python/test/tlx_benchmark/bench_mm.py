"""Perf and compile-time guardrail for `tlx.ops.mm`, against `torch.matmul`."""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

from triton.tlx.ops.kernels.mm._shapes import flops, label, operand

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "mm"
REF_NAME = "torch.matmul"
#: mm has a shape heuristic, so its default is a single analytically chosen
#: config and a first call stays under a second.
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

#: Relative tolerance for the accuracy check, per dtype. Same values the L1
#: correctness suite uses, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.mm.{driver.arch()}").PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    # dtype is a Case field, so it is dropped from `shape` -- carrying it in
    # both duplicates it in the key and in the report.
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import mm as tlx_mm

    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    tlx_fn = lambda: tlx_mm(a, b, arch=driver.arch(), space=space)  # noqa: E731
    ref_fn = lambda: torch.matmul(a, b)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION[case.dtype]),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
