"""Perf and compile-time guardrail for `tlx.ops.mm`, against `torch.matmul`."""

from __future__ import annotations

import pathlib
import sys

import torch

from triton.tlx.ops.kernels.mm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC, MMShape, flops, label, operand

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "mm"
REF_NAME = "torch.matmul (hipBLASLt)" if torch.version.hip is not None else "torch.matmul"
#: mm has a shape heuristic, so its default is a single analytically chosen
#: config and a first call stays under a second.
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

#: Relative tolerance for the accuracy check, per dtype. Same values the L1
#: correctness suite uses, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}

# L2 intentionally benchmarks every registered shape; keep this placeholder for
# future benchmark-only exclusions.
FAILED_SHAPES = set()

#: Focus shapes each arch's `tlx.ops.mm` is known to decline with
#: `InvalidInput`, so they have no tlx number to measure. On sm100 these are
#: operands whose TMA row stride is not 16-byte aligned, plus padded operands
#: (leading stride wider than the row), which the kernel feeds to TMA
#: transposed. Listed rather than derived from the catalog: if the op starts
#: declining a shape not listed here, that case still fails as an ERROR. Remove
#: an entry once the op accepts it, or that shape stays unmeasured.
DECLINED_SHAPES = {
    "sm100":
    frozenset({
        MMShape(256, 1056, 294912, (1, 256), (1088, 1), "bf16"),
        MMShape(1024, 65536, 2304, (1, 1024), (131072, 1), "bf16"),
        MMShape(1024, 65536, 1152, (1, 1024), (131072, 1), "bf16"),
        MMShape(2048, 12800, 1152, (1, 2048), (89600, 1), "bf16"),
        MMShape(7, 4096, 1152, (1, 7), (4096, 1), "bf16"),
        MMShape(7, 2048, 1152, (1, 7), (2048, 1), "bf16"),
        MMShape(2048, 15044, 1152, (1, 2048), (15072, 1), "bf16"),
        MMShape(1152, 15044, 2048, (2048, 1), (15072, 1), "bf16"),
        MMShape(1024, 65536, 1152, (1, 1024), (196608, 1), "bf16"),
        MMShape(1152, 16672, 3840, (3840, 1), (16704, 1), "bf16"),
        MMShape(1152, 256, 16672, (16704, 1), (256, 1), "bf16"),
        MMShape(1056, 1056, 2304, (1, 1088), (1088, 1), "bf16"),
        MMShape(3840, 16672, 1152, (1, 3840), (16704, 1), "bf16"),
        MMShape(1152, 1024, 15044, (15072, 1), (1024, 1), "bf16"),
        MMShape(1152, 16672, 256, (256, 1), (16704, 1), "bf16"),
        MMShape(256, 15042, 1152, (1, 256), (15042, 1), "bf16"),
        MMShape(2304, 1056, 1056, (1088, 1), (1, 1056), "bf16"),
        MMShape(2304, 1056, 1056, (1088, 1), (1056, 1), "bf16"),
        MMShape(1152, 4096, 7, (7, 1), (4096, 1), "bf16"),
        MMShape(705178, 6, 6, (6, 1), (6, 1), "bf16"),
        MMShape(1024, 15044, 1152, (1, 1024), (15072, 1), "bf16"),
        MMShape(1536, 16672, 1152, (1, 1536), (16704, 1), "bf16"),
        MMShape(1152, 2048, 7, (7, 1), (2048, 1), "bf16"),
        MMShape(705178, 6, 6, (6, 1), (1, 6), "bf16"),
        MMShape(1152, 512, 15044, (15072, 1), (512, 1), "bf16"),
        MMShape(1152, 16672, 1536, (1536, 1), (16704, 1), "bf16"),
        MMShape(503599, 6, 6, (6, 1), (6, 1), "bf16"),
        MMShape(7, 7, 198339, (1, 7), (7, 1), "bf16"),
        MMShape(386515, 6, 6, (6, 1), (6, 1), "bf16"),
        MMShape(503599, 6, 6, (6, 1), (1, 6), "bf16"),
        MMShape(386937, 7, 7, (7, 1), (7, 1), "bf16"),
        MMShape(1152, 256, 15044, (15072, 1), (256, 1), "bf16"),
        MMShape(256, 16672, 1152, (1, 256), (16704, 1), "bf16"),
        MMShape(1056, 1056, 1152, (1, 1088), (1088, 1), "bf16"),
        MMShape(7, 7, 222929, (1, 7), (7, 1), "bf16"),
        MMShape(115066, 4, 4, (4, 1), (4, 1), "bf16"),
        MMShape(1152, 1056, 1056, (1088, 1), (1, 1056), "bf16"),
        MMShape(313230, 7, 7, (7, 1), (7, 1), "bf16"),
        MMShape(386515, 6, 6, (6, 1), (1, 6), "bf16"),
        MMShape(117574, 4, 4, (4, 1), (4, 1), "bf16"),
        MMShape(512, 15044, 1152, (1, 512), (15072, 1), "bf16"),
        MMShape(10, 10, 75315, (1, 10), (10, 1), "bf16"),
    }),
}


def shapes(synthetic: bool = False, suites=None) -> list:
    arch = driver.arch()
    entries = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(arch, suites)
    skip = FAILED_SHAPES | DECLINED_SHAPES.get(arch, frozenset())
    return [entry for entry in entries if tuple(entry) not in skip]


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    # dtype is a Case field, so it is dropped from `shape` -- carrying it in
    # both duplicates it in the key and in the report.
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry[5]]).removeprefix("torch."),
            shape=tuple(entry[:5]),
            label=label(*entry),
        ) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import mm as tlx_mm

    if torch.version.hip is not None:
        # Pin the production AMD matmul backend instead of inheriting a process-
        # global preference that could silently switch the reference to rocBLAS.
        torch.backends.cuda.preferred_blas_library("hipblaslt")

    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)
    tlx_out = torch.empty((M, N), device=a.device, dtype=dtype)
    ref_out = torch.empty_like(tlx_out)

    tlx_fn = lambda: tlx_mm(a, b, out=tlx_out, space=space)  # noqa: E731
    ref_fn = lambda: torch.matmul(a, b, out=ref_out)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION[case.dtype]),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
