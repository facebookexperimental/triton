"""Perf benchmark for the TLX MI300X (gfx942 / CDNA3) GEMM tutorial.

Compares ``amd_gemm_gfx942`` against **aten** (``torch.matmul``, which dispatches
to hipBLASLt / rocBLAS on ROCm). There is one TLX provider, because it is a
single autotuned TLX kernel: its NUM_BUFFERS search space spans 1..3, so the
single-buffered ring that ``amd_gemm_pipelined`` is fixed at is one point inside
it rather than a separate provider -- and one that frequently wins.

The kernel autotunes, so the first call per shape pays for the search. Set
``TRITON_PRINT_AUTOTUNING=1`` to see the winning tile and ring depth.

Measurement, the ``iters``/``band`` columns and the ``--warmup-s`` / ``--rep-s``
knobs all live in ``gfx942_perf_harness.py``; see it for the method, and for why
short shapes measure launch overhead rather than kernel quality.

Recommended:
    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_amd_gemm_gfx942_perf.py --table

Facebook: If you are developing in fbsource, use tritonbench instead to collect
perf numbers.
"""

import torch

from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (
    DEVICE,
    OpSpec,
    main,
)

from triton.language.extra.tlx.tutorials.amd_gemm_gfx942 import (
    matmul as _amd_gemm_gfx942, )


def _make_inputs(shape, dtype):
    M, N, K = shape
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    return a, b


SPEC = OpSpec(
    name="amd_gemm_gfx942",
    axes=("M", "N", "K"),
    # Square shapes plus two skinny/fat cases, which pick different tiles.
    shapes=[
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 8192, 8192),
        (8192, 1024, 8192),
    ],
    make_inputs=_make_inputs,
    ref=lambda a, b: torch.matmul(a, b),
    providers={"tlx_gfx942": lambda a, b: _amd_gemm_gfx942(a, b)},
    flops=lambda shape: 2 * shape[0] * shape[1] * shape[2],
)

if __name__ == "__main__":
    main(SPEC)
