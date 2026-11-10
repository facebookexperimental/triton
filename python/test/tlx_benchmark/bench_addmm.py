"""Perf guardrail for tuned gfx942 ``tlx.ops.addmm``, against ``torch.addmm``.

Both providers consume the same BF16 ``A[M, K]``, column-major ``B[K, N]`` and
recorded vector or matrix bias tensors and write to preallocated outputs. Run
on MI300X with:

    python python/test/tlx_benchmark/bench_addmm.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.addmm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.addmm._shapes import SYNTHETIC, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, driver  # noqa: E402

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
            dtype=str(DTYPES[entry.dtype]).removeprefix("torch."),
            shape=tuple(entry[:-1]),
            label=label(*entry),
        ) for entry in shapes(synthetic, suites)
    ]


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
