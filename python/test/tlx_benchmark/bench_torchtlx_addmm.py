"""Perf guardrail for TorchTLX ``addmm`` on gfx950."""

from __future__ import annotations

import pathlib
import sys

import torch

try:
    from triton.tlx.ops.kernels.addmm import gfx950_torch
    from triton.tlx.ops.kernels.addmm._shapes import FOCUS as SHAPE_SUITES
    from triton.tlx.ops.kernels.addmm._shapes import SYNTHETIC, label
except ImportError:  # not the fbtriton fork
    gfx950_torch = None
    SHAPE_SUITES = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, driver  # noqa: E402

OP = "addmm_torchtlx"
REF_NAME = "torch.addmm"
DEFAULT_SPACE = "heuristic"
COLD_COMPILE = "none"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = {"float16": 2e-2, "bfloat16": 2e-2}


def shapes(synthetic: bool = False, suites=None) -> list:
    if gfx950_torch is None:
        return []
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes("gfx950", suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry.dtype]).removeprefix("torch."), shape=tuple(entry[:-1]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


_supported, default_json, run, main = driver.bind(sys.modules[__name__])


def supported() -> bool:
    return gfx950_torch is not None and _supported()


if __name__ == "__main__":
    raise SystemExit(main())
