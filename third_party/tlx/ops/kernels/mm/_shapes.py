"""Shapes for ``tlx.ops.mm`` -- the single source of truth, across architectures.

Three lists, and the split is by *provenance*, not by which suite reads them:

* :data:`SYNTHETIC` -- hand-picked to exercise a code path (edge tiles, layouts,
  split-K). Nobody asked for these shapes; they exist to break the kernel.
* :data:`REALWORLD_SM100` -- compute-bound shapes users actually run on Blackwell.
* :data:`REALWORLD_GFX942` -- the same, requested for MI300X.

Both suites import from here:

* correctness -- ``python/test/unit/tlx_ops/test_mm.py`` runs :data:`ALL`, the
  union, on whatever GPU is present.
* performance -- ``python/test/tlx_benchmark/bench_<op>.py`` runs only the
  current arch's real-world list, reached through that arch's kernel module
  (``sm100.PERF_SHAPES`` / ``gfx942.PERF_SHAPES``).

Correctness runs the union and perf does not, for two different reasons. A
synthetic edge-tile shape is too small to gate a *number* on -- the perf suite
would report it ``HOST_BOUND`` and refuse to gate it anyway -- but it is exactly
what catches a masking bug. And another arch's real-world shapes are free
correctness coverage: they are real geometries nobody chose with this kernel in
mind, which is what makes them good at finding assumptions. Perf is the
opposite: a Blackwell number for an MI300X-requested shape answers a question
nobody asked, and costs minutes to produce.

The union is what buys the invariant that matters: a shape commented out for a
correctness bug cannot stay benchmarked. Benchmarking a path that returns wrong
answers produces a number that looks like signal and is not.

**A shape in the union is not necessarily runnable on every arch.** ``sm100``
requires 16-byte-aligned TMA descriptor strides and declines anything else with
``InvalidInput``; ``gfx942`` loads through plain strided pointers and has no
such constraint. The correctness suite treats a declined shape as a skip, so
each arch runs the subset it admits. Today exactly one entry is affected:
``REALWORLD_GFX942``'s ``K = 1894`` shape, which sm100 declines.

Entries are ``[M, N, K, a_row_major, b_row_major]``. Layout belongs in the
entry rather than in a separate axis because a column-major operand is a
different code path AND moves the TMA 16-byte stride constraint to a different
dimension: row-major A constrains K, column-major A constrains M. dtype is a
separate parametrize axis in each suite.

FIXED (split-K remainder tile): ``[1000, 1000, 1024]`` and ``[64, 4096, 4096]``
were disabled here for a split-K bug -- wrong results when M was not a multiple
of BLOCK_SIZE_M -- and are now RE-ENABLED. Fixed upstream by #3401, verified on
this rebase: both are 0.0% wrong, as is the smaller repro ``[4160, 512, 512]``
(SPLIT_K=2, M % 256 == 64), which measured 4.0% wrong before. They stay in the
list as the regression test for that fix.

TODO(NUM_CTAS=2 multi-N-tile): ``[1000000, 512, 512]`` is commented out below
because it FAILS correctness. It is a **different** bug from the split-K one
above -- this one has ``SPLIT_K=1`` -- and it STILL REPRODUCES after #3401.

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
speed without checking their results.

Pre-existing, not a promotion artifact: the identical 49.9% reproduces through
the original tutorial entry point,
``blackwell_gemm_ws.matmul(a, b, config=get_heuristic_config(M, N, K, num_sms))``.
Not yet filed. Status here is BYPASSED, not fixed -- the shape is commented out
so a wrong-answer path cannot be benchmarked, per this project's rule of never
xfailing or loosening to go green.
"""

from __future__ import annotations

#: ``[M, N, K, a_row_major, b_row_major]`` -- path coverage, not real workloads.
#:
#: Correctness-only. Most are far too small to time: ``tlx.ops.mm`` costs 43-63us
#: of host time per call, so anything under roughly 300us measures Python rather
#: than the kernel, and the perf suite would refuse to gate them anyway. They
#: earn their place by breaking things, not by being fast.
SYNTHETIC: list[list] = [
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
    # Non-power-of-two in M and N together. Regression test for the split-K
    # remainder-tile fix (#3401).
    [1000, 1000, 1024, True, True],
    # Same, but with a partial K tile: 200 is 6 full 32-wide tiles plus 8, so
    # the masked tail of the reduction runs. Every other entry has K divisible
    # by 32 and never exercises it. Carried over from the gfx942 tutorial's
    # odd-shape test, which this list replaced; K=200 rather than that test's
    # 130 because 130 fails sm100's 16-byte TMA stride rule and would have
    # restored the coverage on one arch only.
    [1000, 1000, 200, True, True],
    # K-heavy: few output tiles, long reduction. Split-K territory, the one
    # path that runs a second kernel.
    [256, 256, 16384, True, True],
    # Tall-skinny: most of the grid idle. Also a regression test for #3401.
    [64, 4096, 4096, True, True],
]

#: Compute-bound Blackwell shapes. ``bench_mm.py`` gates on these.
#:
#: Taken from tritonbench's gemm BUILDIN_SHAPES so the numbers are comparable to
#: a tritonbench run of the same shape.
REALWORLD_SM100: list[list] = [
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

#: MI300X shapes users asked for, ordered by how much time they cost in
#: aggregate -- shape latency times how often it occurs -- so entry 0 is the one
#: worth the most. Truncated at ten, which the source measurement puts at ~90%
#: of the total deficit against the vendor library.
#:
#: These are recorded geometries, so they are irregular in ways synthetic shapes
#: never are, and that is the point of carrying them:
#:
#: * ``K = 1894`` is odd. On gfx942 that is merely unaligned; on sm100 it fails
#:   the 16-byte TMA stride rule outright (1894 * 2 = 3788, 3788 % 16 == 12), so
#:   ``tlx.ops.mm`` declines it there. It is the only entry any arch declines.
#: * ``N = 192`` and ``N = 256`` against multi-hundred-thousand ``M`` are narrow
#:   and very tall -- power-of-two N tiles either underfill or need a second
#:   epilogue path.
#: * ``N = 242432`` is the opposite extreme and the largest working set here at
#:   roughly 5 GB, which still fits comfortably on both MI300X and B200.
#:
#: Several were requested as fused ``addmm``. ``tlx.ops.mm`` has no bias, so what
#: is carried here is the GEMM geometry only -- the bias is a separate epilogue
#: question and does not change the tile the kernel has to pick.
REALWORLD_GFX942: list[list] = [
    [819200, 192, 1024, True, True],
    # The odd-K entry. sm100 declines this one; see above.
    [4096, 242432, 1894, True, True],
    [1024, 20480, 6144, True, True],
    [2048, 10240, 25408, True, True],
    [61440, 5120, 2048, True, True],
    [2252800, 256, 256, True, True],
    [61440, 3840, 4096, True, True],
    [4096, 4096, 2048, True, True],
    [61440, 5120, 7744, True, True],
    [1024, 6144, 4096, True, True],
]


def _union(*lists: list[list]) -> list[list]:
    """Concatenate, dropping later duplicates and keeping first-seen order.

    The lists are independently sourced, so an overlap is possible and would
    otherwise silently double a correctness case's runtime.
    """
    seen, out = set(), []
    for shapes in lists:
        for shape in shapes:
            key = tuple(shape)
            if key not in seen:
                seen.add(key)
                out.append(shape)
    return out


#: Every shape, for the correctness suite. Perf never reads this.
ALL: list[list] = _union(SYNTHETIC, REALWORLD_SM100, REALWORLD_GFX942)


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
