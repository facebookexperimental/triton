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
    from triton.language.extra.tlx.inductor import gfx950_torch, sm100_torch
    from triton.tlx.ops.kernels.mm._shapes import FOCUS as SHAPE_SUITES
    from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC, flops, label, operand
except ImportError:  # not the fbtriton fork
    gfx950_torch = None
    sm100_torch = None
    SHAPE_SUITES = None

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

# L2 intentionally benchmarks every registered shape; keep this placeholder for
# future benchmark-only exclusions.
FAILED_SHAPES = set()


def shapes(synthetic: bool = False, suites=None) -> list:
    if _provider() is None:
        return []
    entries = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites)
    return [entry for entry in entries if tuple(entry) not in FAILED_SHAPES]


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    provider = _provider()
    assert provider is not None
    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    tlx_fn = lambda: provider.mm(a, b, mode="force")  # noqa: E731
    ref_fn = lambda: provider.ref(a, b)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION[case.dtype]),
    )


_supported, default_json, run, main = driver.bind(sys.modules[__name__])


def _provider():
    return {"gfx950": gfx950_torch, "sm100": sm100_torch}.get(driver.arch())


def supported() -> bool:
    return _provider() is not None and _supported()


if __name__ == "__main__":
    raise SystemExit(main())
