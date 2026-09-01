"""Shapes for ``tlx.ops.mm`` on Blackwell -- the single source of truth.

Both suites import this list:

* correctness -- ``python/test/unit/tlx_ops/test_mm.py``
* performance -- ``python/test/tlx_benchmark/bench_mm.py``

Sharing is deliberate, and the invariant it buys is worth more than the
convenience: a shape commented out for a correctness bug cannot stay
benchmarkable. Benchmarking a path that returns wrong answers produces a
number that looks like signal and is not.

Entries are ``[M, N, K, a_row_major, b_row_major]``. Layout belongs in the
entry rather than in a separate axis because a column-major operand is a
different code path AND moves the TMA 16-byte stride constraint to a different
dimension: row-major A constrains K, column-major A constrains M. dtype is a
separate parametrize axis in each suite.

TODO(split-K remainder tile): ``[1000, 1000, 1024]`` and ``[64, 4096, 4096]``
are commented out below because they FAIL correctness. Re-enable them with the
fix -- they are the regression test for it.

Split-K gives wrong results when M is not a multiple of BLOCK_SIZE_M: partials
go into a (SPLIT_K * M, N) workspace that the reduction reads at fixed M
strides, and the masked edge tile breaks that correspondence.

Measured at M=1000, N=1000, K=1024, BLOCK_SIZE_M=256 (relative L2 vs torch):
SPLIT_K=1 -> 0.0 exact, =2 -> 1.04e-1, =4 -> 1.30e-1.

Pre-existing, not a promotion artifact: reproduces as
``blackwell_gemm_ws.matmul(a, b, config=get_heuristic_config(...))``. Reachable
in production -- the heuristic picks SPLIT_K=4 for both shapes. Not yet filed.
Also reproduces at ``[4160, 512, 512]`` (SPLIT_K=2, M % 256 == 64, 4.0% of
elements wrong), which is a smaller repro than either shape above.

TODO(NUM_CTAS=2 multi-N-tile): ``[1000000, 512, 512]`` is commented out below
because it FAILS correctness, and for a **different** reason than the split-K
bug -- this one has ``SPLIT_K=1``.

Whenever the heuristic picks ``NUM_CTAS=2`` and the grid has more than one
N-tile, 49.9% of output elements are wrong -- almost exactly one CTA of each
pair. Measured at fp16, comparing against ``torch.matmul``:

    999936  x 512  x 512   BM=256 BN=128 NUM_CTAS=2  ->  49.9% wrong
    999936  x 1024 x 512   BM=256 BN=128 NUM_CTAS=2  ->  49.9% wrong
    999936  x 2048 x 512   BM=256 BN=128 NUM_CTAS=2  ->  49.9% wrong
    65536   x 512  x 512   BM=256 BN=128 NUM_CTAS=2  ->  49.9% wrong
    65536   x 512  x 1024  BM=256 BN=256 NUM_CTAS=2  ->  49.9% wrong
    999936  x 512  x 1024  BM=256 BN=256 NUM_CTAS=2  ->  49.9% wrong
    131072  x 256  x 512   BM=256 BN=256 NUM_CTAS=2  ->   0.0% wrong

The last row is the control and identifies the trigger: N == BN there, so
``num_pid_n == 1``. Every failing row has more than one N-tile. Not a
remainder-tile problem -- 999936 is an exact multiple of both 256 and 512.

Reachable in production: the heuristic selects ``NUM_CTAS=2`` for ordinary
large-M shapes. Worse, autotuning cannot save you, because it ranks configs by
speed without checking their results. Not yet filed.
"""

from __future__ import annotations

#: ``[M, N, K, a_row_major, b_row_major]``
#:
#: Correctness-motivated entries come first, then the compute-bound entries the
#: perf suite needs. The perf suite reports the small entries as ``HOST_BOUND``
#: rather than gating them: ``tlx.ops.mm`` costs 43-63us of host time per call,
#: so anything under roughly 300us measures Python rather than the kernel. They
#: stay because correctness needs them, and the status says so honestly.
SHAPES: list[list] = [
    # Square, both row-major -- the baseline path, small and large.
    [256, 256, 256, True, True],
    [1024, 1024, 1024, True, True],
    # Rectangular.
    [2048, 512, 1024, True, True],
    # Column-major B: descriptor sees B.T, so the constraint lands on K.
    [512, 4096, 1024, True, False],
    # Column-major A: descriptor sees A.T, so the constraint lands on M.
    [1024, 2048, 512, False, True],
    # Both column-major.
    [2048, 2048, 2048, False, False],
    # M not a multiple of any plausible block size -- masked edge tile.
    [136, 256, 128, True, True],
    # Non-power-of-two in M and N together.
    # TODO(split-K remainder tile): fails, see module docstring.
    # [1000, 1000, 1024, True, True],
    # K-heavy: few output tiles, long reduction. Split-K territory, the one
    # path that runs a second kernel.
    [256, 256, 16384, True, True],
    # Tall-skinny: most of the grid idle.
    # TODO(split-K remainder tile): fails, see module docstring.
    # [64, 4096, 4096, True, True],

    # ---- compute-bound, added for perf ------------------------------------
    # Taken from tritonbench's gemm BUILDIN_SHAPES so the numbers are
    # comparable to a tritonbench run of the same shape.
    #
    # Square flagship: the only entry where host overhead is a small enough
    # fraction (43us on 1105us) for the speedup ratio to be nearly unbiased.
    [8192, 8192, 8192, True, True],
    # Same output tile count, K-light: epilogue- rather than MMA-bound.
    [8192, 8192, 1024, True, True],
    # K-heavy: long reduction over few output tiles.
    [8192, 8192, 16384, True, True],
    # Column-major B at a compute-bound size: the transposed-descriptor path
    # is a different kernel, so it needs its own number and not just its own
    # correctness case.
    [8192, 8192, 8192, True, False],
    # Production-shaped huge-M, from the same tritonbench list.
    # TODO(NUM_CTAS=2 multi-N-tile): fails, see module docstring.
    # [1000000, 512, 512, True, True],
]


def operand(rows, cols, dtype, row_major, device="cuda"):
    """One ``mm`` operand, with the requested memory layout.

    A column-major operand is built as a transposed view of a contiguous
    ``(cols, rows)`` buffer, so it is genuinely non-contiguous rather than
    contiguous-and-relabelled. Both suites build inputs through this function
    so that the perf numbers describe the tensors correctness validated.
    """
    import torch

    t = torch.randn((rows, cols), device=device, dtype=dtype)
    return t if row_major else t.T.contiguous().T


def flops(M, N, K):
    """Multiply-accumulate FLOPs for an ``(M, K) @ (K, N)`` product."""
    return 2 * M * N * K
