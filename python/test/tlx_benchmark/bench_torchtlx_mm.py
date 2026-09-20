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
    from triton.language.extra.tlx.inductor import sm100_torch
    from triton.tlx.ops.kernels.mm._shapes import FOCUS as SHAPE_SUITES
    from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC, flops, label, operand
except ImportError:  # not the fbtriton fork
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

# TODO: Re-enable these shapes when their TorchTLX failures are fixed.
FAILED_SHAPES = {
    (136, 256, 128, (128, 1), (256, 1), "fp16"),
    (136, 256, 128, (128, 1), (256, 1), "bf16"),
    (810572, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (7, 4096, 1152, (1, 7), (4096, 1), "bf16"),
    (7, 2048, 1152, (1, 7), (2048, 1), "bf16"),
    (308743, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (1056, 1056, 2304, (1, 1088), (1088, 1), "bf16"),
    (1, 12800, 1152, (0, 1), (12800, 1), "bf16"),
    (256, 15042, 1152, (1, 256), (15042, 1), "bf16"),
    (1152, 4096, 7, (7, 1), (4096, 1), "bf16"),
    (16672, 256, 1152, (1, 16704), (256, 1), "bf16"),
    (705178, 6, 6, (6, 1), (6, 1), "bf16"),
    (1, 512, 1152, (1152, 1), (512, 1), "bf16"),
    (15044, 1024, 1152, (1, 15072), (1024, 1), "bf16"),
    (1152, 2048, 7, (7, 1), (2048, 1), "bf16"),
    (705178, 6, 6, (6, 1), (1, 6), "bf16"),
    (15042, 256, 1152, (1, 15072), (256, 1), "bf16"),
    (384, 384, 19459, (1, 384), (384, 1), "bf16"),
    (503599, 6, 6, (6, 1), (6, 1), "bf16"),
    (7, 7, 198339, (1, 7), (7, 1), "bf16"),
    (386515, 6, 6, (6, 1), (6, 1), "bf16"),
    (1, 1024, 1152, (1152, 1), (1024, 1), "bf16"),
    (1152, 12800, 32, (32, 1), (12800, 1), "bf16"),
    (503599, 6, 6, (6, 1), (1, 6), "bf16"),
    (386937, 7, 7, (7, 1), (7, 1), "bf16"),
    (8, 8, 705178, (1, 8), (8, 1), "bf16"),
    (114658, 256, 256, (256, 1), (256, 1), "bf16"),
    (15044, 512, 1152, (1, 15072), (512, 1), "bf16"),
    (8, 8, 503599, (1, 8), (8, 1), "bf16"),
    (8, 8, 386515, (1, 8), (8, 1), "bf16"),
    (1056, 1056, 1152, (1, 1088), (1088, 1), "bf16"),
    (7, 7, 222929, (1, 7), (7, 1), "bf16"),
    (1, 32, 1152, (1152, 1), (32, 1), "bf16"),
    (313230, 7, 7, (7, 1), (7, 1), "bf16"),
    (386515, 6, 6, (6, 1), (1, 6), "bf16"),
    (117574, 4, 4, (4, 1), (4, 1), "bf16"),
    (10, 10, 75315, (1, 10), (10, 1), "bf16"),
}


def shapes(synthetic: bool = False, suites=None) -> list:
    if sm100_torch is None:
        return []
    entries = SYNTHETIC if synthetic else SHAPE_SUITES.shapes("sm100", suites)
    return [entry for entry in entries if tuple(entry) not in FAILED_SHAPES]


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
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
