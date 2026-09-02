"""Perf benchmark for the TLX MI300X (gfx942 / CDNA3) fused addmm tutorial.

Compares ``amd_addmm_gfx942`` (``out = bias + a @ b``, one fused kernel) against
**aten**: ``torch.addmm``, which on ROCm is a hipBLASLt GEMM plus a fused or
separate bias epilogue.

The kernel autotunes, so the first call per shape pays for the search. Set
``TRITON_PRINT_AUTOTUNING=1`` to see the winning tile and ring depth.

Measurement, the ``iters``/``band`` columns and the ``--warmup-s`` / ``--rep-s``
knobs all live in ``gfx942_perf_harness.py``; see it for the method, and for why
short shapes measure launch overhead rather than kernel quality.

Recommended:
    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_amd_addmm_gfx942_perf.py --table

Facebook: If you are developing in fbsource, use tritonbench instead to collect
perf numbers.
"""

import torch

from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (
    DEVICE,
    OpSpec,
    main,
)

from triton.language.extra.tlx.tutorials.amd_addmm_gfx942 import (
    addmm as _amd_addmm_gfx942, )


def _make_inputs(shape, dtype):
    M, N, K = shape
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    # 1-D bias is the Linear case, and the one the kernel broadcasts for free.
    bias = torch.randn((N, ), device=DEVICE, dtype=dtype)
    return bias, a, b


SPEC = OpSpec(
    name="amd_addmm_gfx942",
    axes=("M", "N", "K"),
    shapes=[
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 8192, 8192),
        (8192, 1024, 8192),
    ],
    make_inputs=_make_inputs,
    ref=lambda bias, a, b: torch.addmm(bias, a, b),
    providers={"tlx_gfx942": lambda bias, a, b: _amd_addmm_gfx942(bias, a, b)},
    # The bias add is O(M*N) against O(M*N*K) of multiply-add, so it is left out
    # of the flop count -- this stays comparable with the plain GEMM table.
    flops=lambda shape: 2 * shape[0] * shape[1] * shape[2],
)

if __name__ == "__main__":
    main(SPEC)
