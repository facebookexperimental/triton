"""Perf benchmark for the TLX MI300X (gfx942 / CDNA3) batched GEMM tutorial.

Compares ``amd_bmm_gfx942`` against **aten** (``torch.bmm`` -> hipBLASLt).

Inputs are SHARED-A (one (M, K) broadcast over the batch, ``a.stride(0) == 0``),
matching ``amd_bmm.py``'s convention: hipBLASLt reads A once and keeps it
L2-resident, so benchmarking distinct-A would flatter TLX.

The kernel autotunes, so the first call per shape pays for the search. Set
``TRITON_PRINT_AUTOTUNING=1`` to see the winning tile and ring depth.

Measurement, the ``iters``/``band`` columns and the ``--warmup-s`` / ``--rep-s``
knobs all live in ``gfx942_perf_harness.py``; see it for the method, and for why
short shapes measure launch overhead rather than kernel quality -- which is most
of this table, since BMM shapes are small per matrix.

Recommended:
    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_amd_bmm_gfx942_perf.py --table

Facebook: If you are developing in fbsource, use tritonbench instead to collect
perf numbers.
"""

import torch

from triton.language.extra.tlx.tutorials.testing.gfx942_perf_harness import (
    DEVICE,
    OpSpec,
    main,
)

from triton.language.extra.tlx.tutorials.amd_bmm_gfx942 import (
    bmm as _amd_bmm_gfx942,
    make_bmm_inputs as _make_bmm_inputs,
)


def _make_inputs(shape, dtype):
    M, N, K, B = shape
    return _make_bmm_inputs(B, M, N, K, DEVICE, dtype=dtype)


SPEC = OpSpec(
    name="amd_bmm_gfx942",
    axes=("M", "N", "K", "B"),
    # Small per-matrix tiles with a large batch is the regime BMM actually shows
    # up in, so the batch carries the parallelism rather than M/N.
    shapes=[
        (256, 256, 256, 64),
        (512, 512, 512, 32),
        (1024, 1024, 1024, 16),
        (2048, 2048, 2048, 8),
        (128, 128, 4096, 64),
    ],
    make_inputs=_make_inputs,
    ref=lambda a, b: torch.bmm(a, b),
    providers={"tlx_gfx942": lambda a, b: _amd_bmm_gfx942(a, b)},
    flops=lambda shape: 2 * shape[3] * shape[0] * shape[1] * shape[2],
)

if __name__ == "__main__":
    main(SPEC)
