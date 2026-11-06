"""Perf and compile-time guardrail for `tlx.ops.flash_attn`, against SDPA.

Forward and backward are separate cases: the backward is 2.5x the FLOPs, is a
different kernel, and is where most of the recent work went, so folding it into
the forward's number would hide it.
"""

from __future__ import annotations

import pathlib
import sys

import torch

from triton.tlx.ops.kernels.flash_attn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn._shapes import FORWARD_ONLY, SYNTHETIC, label
from triton.tlx.ops.kernels.flash_attn._shapes import flops as flops

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, driver  # noqa: E402

OP = "flash_attn"
REF_NAME = "torch.nn.functional.scaled_dot_product_attention"
DEFAULT_SPACE = "heuristic"
#: Which SDPA kernel torch would dispatch. Earns a column because the headline
#: speedup is unreadable without it: 1.4x over `math` is a broken benchmark,
#: 1.4x over cuDNN is a result.
EXTRA_COLUMNS = (("ref bknd", "ref_backend"), )

#: One fresh-cache first call per direction, not per case: the cold pass
#: compiles this op's whole autotune space, and "does a first call take too
#: long" is not a per-shape question.
COLD_COMPILE = "first"

#: A full autotune on a cold cache is minutes, not seconds. The default 120s cap
#: is calibrated for mm's heuristic path and would fail every case here on
#: compile alone, which says nothing about the kernel.
COMPILE_CAP_S = 900.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")

#: atol tracks the magnitude of the result, not of the element compared. Same
#: values as `test_flash_attn.py`, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    arch = driver.arch()
    entries = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(arch, suites)
    # The shared synthetic list mirrors the original NVIDIA correctness cases,
    # which are FP16. gfx950 FlashAttention is deliberately BF16-only, so keep
    # the same small shapes while selecting the dtype accepted by its catalog
    # entry instead of turning every synthetic case into an InvalidInput row.
    if synthetic and arch == "gfx950":
        entries = (entry._replace(dtype="bf16") for entry in entries)
    return list(entries)


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             direction=direction, label=label(*entry, direction))
        for entry in shapes(synthetic, suites)
        for direction in (("fwd", ) if entry in FORWARD_ONLY else DIRECTIONS)
    ]


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
