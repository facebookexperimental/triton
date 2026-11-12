"""Nightly performance coverage for ``tlx.ops.mm_mxfp8`` on gfx950.

The public PyTorch reference multiplies pre-dequantized BF16 operands through
hipBLASLt. Its throughput is reported for context, but does not gate the run
because it is not an MXFP8 storage- or instruction-equivalent comparator.
"""

from __future__ import annotations

import pathlib
import sys

import torch

from triton._internal_testing import swizzle_scale_to_5d
from triton.tlx.ops.kernels.mm_mxfp8._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.mm_mxfp8._shapes import (LOGICAL_K, MODEL_LINEAR_PRESHUFFLE_REPEATS, MODEL_LINEAR_SHAPES,
                                                     SYNTHETIC_SHAPES, flops, label)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "mm_mxfp8"
REF_NAME = "torch.mm on pre-dequantized BF16 inputs (diagnostic)"
REFERENCE_GATE = False
DEFAULT_SPACE = "heuristic"
COLD_COMPILE = "all"
EXTRA_COLUMNS = ()

SCALE_LAYOUTS = ("natural", "cublas_blocked")
MODEL_LINEAR_SHAPE_SET = frozenset(MODEL_LINEAR_SHAPES)


def _scale_layouts(shape: tuple[int, int, int]) -> tuple[str, ...]:
    if shape in MODEL_LINEAR_PRESHUFFLE_REPEATS:
        return ("mfma_preshuffled", )
    if shape in MODEL_LINEAR_SHAPE_SET:
        return ("cublas_blocked", )
    return SCALE_LAYOUTS


def shapes(synthetic: bool = False, suites=None) -> tuple[tuple[int, int, int], ...]:
    return SYNTHETIC_SHAPES if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites)


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype="float8_e4m3fn",
            shape=(m, n, k, sf_layout),
            label=label(m, n, k, sf_layout),
        ) for m, n, k in shapes(synthetic, suites) for sf_layout in _scale_layouts((m, n, k))
    ]


def _blocked_scale(scale_bytes: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    padded = torch.nn.functional.pad(scale_bytes, (0, 0, 0, -rows % 128), value=127)
    return swizzle_scale_to_5d(padded[None], (rows + 127) // 128, k // 128).view(torch.float8_e8m0fnu)


def _mfma_preshuffled_scale(scale_bytes: torch.Tensor, repeat: int) -> torch.Tensor:
    rows, groups = scale_bytes.shape
    logical_row_chunk = repeat * 32
    physical_row_chunk = ((repeat + 1) // 2) * 64
    padded_rows = (rows + logical_row_chunk - 1) // logical_row_chunk * physical_row_chunk
    padded = torch.full((padded_rows, groups), 127, dtype=torch.uint8, device=scale_bytes.device)
    logical_rows = torch.arange(rows, device=scale_bytes.device)
    physical_rows = logical_rows // logical_row_chunk * physical_row_chunk + logical_rows % logical_row_chunk
    padded[physical_rows] = scale_bytes
    return (padded.view(padded_rows // 64, 2, 32, groups // 4, 2, 2).permute(0, 3, 5, 2, 4,
                                                                             1).contiguous().view(torch.float8_e8m0fnu))


def _dequantize_bf16(data: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    rows, k = data.shape
    result = data.to(torch.bfloat16).view(rows, k // 32, 32)
    factors = torch.exp2(scale_bytes.to(torch.float32) - 127).to(torch.bfloat16)
    result.mul_(factors[:, :, None])
    return result.view(rows, k)


def _inputs(m: int, n: int, k: int, sf_layout: str):
    generator = torch.Generator(device="cuda").manual_seed(43)
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int8, generator=generator).to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (n, k), device="cuda", dtype=torch.int8, generator=generator).to(torch.float8_e4m3fn)
    a_scale_bytes = torch.randint(125, 128, (m, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
    b_scale_bytes = torch.randint(125, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)

    logical_k = LOGICAL_K.get(k, k)
    if logical_k != k:
        a[:, logical_k:] = 0
        b[:, logical_k:] = 0

    if sf_layout == "mfma_preshuffled":
        a_repeat, b_repeat = MODEL_LINEAR_PRESHUFFLE_REPEATS[(m, n, k)]
        a_scale = _mfma_preshuffled_scale(a_scale_bytes, a_repeat)
        b_scale = _mfma_preshuffled_scale(b_scale_bytes, b_repeat)
    elif sf_layout == "cublas_blocked":
        a_scale = _blocked_scale(a_scale_bytes, m, k)
        b_scale = _blocked_scale(b_scale_bytes, n, k)
    else:
        a_scale = a_scale_bytes.view(torch.float8_e8m0fnu)
        b_scale = b_scale_bytes.view(torch.float8_e8m0fnu)
    return (a, a_scale, b, b_scale), (a_scale_bytes, b_scale_bytes)


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import mm_mxfp8

    torch.backends.cuda.preferred_blas_library("hipblaslt")
    m, n, k, sf_layout = case.shape
    inputs, scale_bytes = _inputs(m, n, k, sf_layout)
    a, _, b, _ = inputs
    a_ref = _dequantize_bf16(a, scale_bytes[0])
    b_ref = _dequantize_bf16(b, scale_bytes[1])
    tlx_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_out = torch.empty_like(tlx_out)

    tlx_fn = lambda: mm_mxfp8(*inputs, out=tlx_out, sf_layout=sf_layout, space=space)  # noqa: E731
    ref_fn = lambda: torch.mm(a_ref, b_ref.T, out=ref_out)  # noqa: E731

    def check() -> tuple[bool, str]:
        actual, expected = tlx_fn(), ref_fn()
        rows, cols = min(m, 128), min(n, 128)
        try:
            torch.testing.assert_close(
                actual[:rows, :cols].float(),
                expected[:rows, :cols].float(),
                rtol=0.03,
                atol=0.5,
            )
        except AssertionError as mismatch:
            return False, f"output does not match PyTorch: {str(mismatch).splitlines()[0]}"
        return True, ""

    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(m, n, k),
        check=check,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
