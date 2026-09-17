"""Perf guardrail for torchTLX `mm` -- the TLX template through torch.compile.

TLX is forced rather than merely allowed, so every case measures a TLX kernel;
the reference is the same `torch.compile` with TLX off, which is whatever stock
Inductor would have picked. `speedup > 1` is therefore exactly "TLX would have
won the autotune under tlx_mode=allow".
"""

from __future__ import annotations

import pathlib
import sys

import torch

try:
    from triton.tlx.ops.kernels.mm import sm100_torch
    from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC, flops, label, operand
except ImportError:  # not the fbtriton fork
    sm100_torch = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "mm_torchtlx"
REF_NAME = "torch.compile (TLX off)"
DEFAULT_SPACE = "heuristic"
#: Inductor's own caches survive `fresh_triton_cache`, so a cold pass here would
#: not be a cold pass. Compile time for this provider needs its own mechanism.
COLD_COMPILE = "none"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False) -> list[list]:
    if sm100_torch is None:
        return []
    return list(SYNTHETIC if synthetic else sm100_torch.PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic)
    ]


def prepare(case: Case, space: str) -> Prepared:
    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    tlx_fn = lambda: sm100_torch.mm(a, b, mode="force")  # noqa: E731
    ref_fn = lambda: sm100_torch.ref(a, b)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION[case.dtype]),
    )


_supported, default_json, run, main = driver.bind(sys.modules[__name__])


def supported() -> bool:
    return sm100_torch is not None and _supported()


if __name__ == "__main__":
    raise SystemExit(main())
