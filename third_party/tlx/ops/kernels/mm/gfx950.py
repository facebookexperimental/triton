"""gfx950 GEMM implementation for :func:`triton.tlx.ops.mm`.

The single public entry dispatches to reusable LocalSplitU, persistent,
register-resident, and direct-to-LDS execution paths in this module.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.driver import driver

from ..._catalog import InvalidInput

# Regular wave-grid implementation and dispatch policy.

# Macro-tile cost model.
_wg_NUM_CUS = 256
_wg_LDS_BYTES = 160 * 1024
_wg_MFMA_EXTENT = 16


@dataclass(frozen=True)
class _wg_WaveGridCandidate:
    """One executable regular-MFMA-grid candidate."""

    block_m: int
    block_n: int
    block_k: int
    mi_wave_tile_m: int
    mi_wave_tile_n: int
    warps_m: int
    warps_n: int


# Soft throughput prior learned from gfx950 hipBLASLt solutions.  This is not a
# support list: every architecture-legal geometry remains in the search space.
_wg_CALIBRATED_FAST_GEOMETRIES = frozenset((m, n) for m, n in (
    (16, 256),
    (48, 256),
    (64, 224),
    (96, 192),
    (112, 256),
    (128, 96),
    (128, 160),
    (128, 224),
    (128, 256),
    (144, 256),
    (160, 160),
    (160, 224),
    (176, 256),
    (192, 96),
    (192, 192),
    (192, 224),
    (208, 256),
    (224, 192),
    (224, 256),
    (240, 256),
    (256, 128),
    (256, 160),
    (256, 176),
    (256, 192),
    (256, 208),
    (256, 224),
    (256, 240),
    (256, 256),
    (384, 160),
))


def _wg_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _wg_wave_grid(block_m: int, block_n: int):
    """Choose a balanced legal four-wave decomposition for one macro tile."""
    choices = []
    for warps_m, warps_n in ((1, 4), (2, 2), (4, 1)):
        wave_m = _wg_MFMA_EXTENT * warps_m
        wave_n = _wg_MFMA_EXTENT * warps_n
        if block_m % wave_m or block_n % wave_n:
            continue
        mi_m = block_m // wave_m
        mi_n = block_n // wave_n
        # One MI16x16 accumulator fragment occupies four AGPRs.  Keep the
        # per-wave grid within the 256-AGPR architectural budget.
        if not (1 <= mi_m <= 16 and 1 <= mi_n <= 16):
            continue
        if mi_m * mi_n > 64:
            continue
        # Prefer a nearly square per-wave tile.  The second key makes the
        # choice deterministic when transposed decompositions tie.
        balance = abs(math.log2(mi_m / mi_n))
        choices.append((balance, abs(warps_m - warps_n), warps_m, warps_n, mi_m, mi_n))
    if not choices:
        return None
    _, _, warps_m, warps_n, mi_m, mi_n = min(choices)
    return mi_m, mi_n, warps_m, warps_n


def _wg_fits_lds(block_m: int, block_n: int, block_k: int) -> bool:
    # Two FP16 stages, including both A and B macro-tile operands.
    return 4 * (block_m + block_n) * block_k <= _wg_LDS_BYTES


@lru_cache(maxsize=1)
def _wg_candidate_space() -> tuple[_wg_WaveGridCandidate, ...]:
    dimensions = tuple(range(16, 257, 16)) + (320, 384, 448, 512)
    candidates = []
    for block_m in dimensions:
        for block_n in dimensions:
            area = block_m * block_n
            if not 4096 <= area <= 65536:
                continue
            if max(block_m, block_n) > 16 * min(block_m, block_n):
                continue
            wave_grid = _wg_wave_grid(block_m, block_n)
            if wave_grid is None:
                continue
            mi_m, mi_n, warps_m, warps_n = wave_grid
            # The common executable pipeline currently supports K32/K64.  K64
            # is the mature general path; short-K register kernels remain a
            # separate candidate family rather than being hidden here.
            block_k = 64
            if not _wg_fits_lds(block_m, block_n, block_k):
                continue
            candidates.append(_wg_WaveGridCandidate(
                block_m,
                block_n,
                block_k,
                mi_m,
                mi_n,
                warps_m,
                warps_n,
            ))
    return tuple(candidates)


def _wg_estimate_cost(m: int, n: int, k: int, candidate: _wg_WaveGridCandidate) -> float:
    """Estimate whole-grid latency in relative gfx950 units."""
    grid = _wg_ceil_div(m, candidate.block_m) * _wg_ceil_div(n, candidate.block_n)
    cu_rounds = _wg_ceil_div(grid, _wg_NUM_CUS)
    mfma_work = candidate.block_m * candidate.block_n * k / 8_388_608
    global_load_work = (candidate.block_m + candidate.block_n) * k / 4_608
    loop_turnover = 45 * k / (8 * candidate.block_k)
    output_work = candidate.block_m * candidate.block_n / 320
    workgroup_cost = (mfma_work + global_load_work + loop_turnover + output_work)
    if max(candidate.block_m, candidate.block_n) > 256:
        workgroup_cost *= 1.25
    if (candidate.block_m, candidate.block_n) not in _wg_CALIBRATED_FAST_GEOMETRIES:
        workgroup_cost *= 1.30
    return cu_rounds * workgroup_cost


@lru_cache(maxsize=None)
def _wg_ranked_wave_grid_candidates(
    m: int,
    n: int,
    k: int,
    limit: int = 3,
) -> tuple[_wg_WaveGridCandidate, ...]:
    """Return at most ``limit`` ranked, executable wave-grid candidates."""
    if min(m, n, k, limit) <= 0:
        raise ValueError("M, N, K, and limit must be positive")
    if k % 64:
        return ()
    candidates = sorted(
        _wg_candidate_space(),
        key=lambda candidate: (
            _wg_estimate_cost(m, n, k, candidate),
            -candidate.block_m * candidate.block_n,
            candidate.block_m,
            candidate.block_n,
        ),
    )
    return tuple(candidates[:limit])


# Wave-grid plan construction.
def _wg_regular_wave_grid_plan(
        mi_wave_tile_m,
        mi_wave_tile_n,
        warps_m,
        warps_n,
        block_k,
        tiles_per_program=1,
        *,
        persistent_program_count=0,
        local_stages=1,
        linear_tiles=False,
        pgr2_operands=False,
        direct_to_lds=False,
        refill_read_window=0,
        pgr2_late_read_count=0,
        read_cover_sixteenths=8,
        loop_unroll_factor=1,
        direct_chunk=128,
        pack_direct_chunks=False,
        row_major_b_lds=False,
        row_wise_epilogue=False,
        wide_epilogue=False,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=False,
        disable_unclustered_high_rp_reschedule=False,
        enable_sched_group_barrier_scheduler=False,
        sched_group_barrier_mfma_per_dwordx4=4,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
        llvm_sched_strategy="",
        split_k=1,
        pair_balanced_split_k=False,
        streamk_tail_split=0,
        reduce_tile=(16, 64),
        reduce_warps=4,
):
    """Build one compile-time regular-wave-grid specialization.

    Geometry and pipeline policy stay explicit so the selector can generate
    and benchmark a bounded set of plans without cloning the kernel source.
    """
    return {
        "kind": ("regular_mi32_wave_grid" if matrix_instr_nonkdim == 32 else "regular_mi16_wave_grid"),
        "mi_wave_tile_m": mi_wave_tile_m,
        "mi_wave_tile_n": mi_wave_tile_n,
        "warps_m": warps_m,
        "warps_n": warps_n,
        "block_k": block_k,
        "num_warps": warps_m * warps_n,
        "tiles_per_program": tiles_per_program,
        "persistent_program_count": persistent_program_count,
        "local_stages": local_stages,
        "linear_tiles": linear_tiles,
        "pgr2_operands": pgr2_operands,
        "direct_to_lds": direct_to_lds,
        "refill_read_window": refill_read_window,
        "pgr2_late_read_count": pgr2_late_read_count,
        "read_cover_sixteenths": read_cover_sixteenths,
        "loop_unroll_factor": loop_unroll_factor,
        "direct_chunk": direct_chunk,
        "pack_direct_chunks": pack_direct_chunks,
        "row_major_b_lds": row_major_b_lds,
        "row_wise_epilogue": row_wise_epilogue,
        "wide_epilogue": wide_epilogue,
        "num_xcds": num_xcds,
        "workgroup_mapping": workgroup_mapping,
        "reverse_local_assignment": reverse_local_assignment,
        "sink_insts_to_avoid_spills": sink_insts_to_avoid_spills,
        "disable_unclustered_high_rp_reschedule": (disable_unclustered_high_rp_reschedule),
        "enable_sched_group_barrier_scheduler": (enable_sched_group_barrier_scheduler),
        "sched_group_barrier_mfma_per_dwordx4": (sched_group_barrier_mfma_per_dwordx4),
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "waves_per_eu": waves_per_eu,
        "llvm_fn_attrs": ((("amdgpu-sched-strategy", llvm_sched_strategy), ) if llvm_sched_strategy else ""),
        "split_k": split_k,
        "pair_balanced_split_k": pair_balanced_split_k,
        "streamk_tail_split": streamk_tail_split,
        "reduce_tile": reduce_tile,
        "reduce_warps": reduce_warps,
    }


@lru_cache(maxsize=None)
def _wg_ranked_wave_grid_plans(m, n, k, limit=3):
    """Select a bounded pipeline family, then rank geometry within it."""
    candidates = _wg_ranked_wave_grid_candidates(m, n, k, limit)
    plans = []
    for candidate in candidates:
        # The direct-register path represents B as one tensor and therefore
        # requires a power-of-two N extent. Keep an irregular-N geometry in
        # the search, but route it through the fragmented LDS pipeline.
        if k <= 256 and candidate.block_n & (candidate.block_n - 1) == 0:
            plan = _wg_register_wave_grid_plan(
                candidate.mi_wave_tile_m,
                candidate.mi_wave_tile_n,
                candidate.warps_m,
                candidate.warps_n,
            )
        else:
            direct_to_lds = (k == 512 and candidate.mi_wave_tile_m == 8 and candidate.mi_wave_tile_n == 8
                             and candidate.warps_m == 2 and candidate.warps_n == 2)
            pgr2_operands = (k >= 1024 and candidate.warps_m == 1 and candidate.warps_n == 4
                             and candidate.mi_wave_tile_m in (11, 13, 15) and candidate.mi_wave_tile_n == 4)
            plan = _wg_regular_wave_grid_plan(
                candidate.mi_wave_tile_m,
                candidate.mi_wave_tile_n,
                candidate.warps_m,
                candidate.warps_n,
                candidate.block_k,
                local_stages=2 if direct_to_lds else 1,
                pgr2_operands=pgr2_operands,
                direct_to_lds=direct_to_lds,
                reverse_local_assignment=not pgr2_operands,
                sink_insts_to_avoid_spills=pgr2_operands,
                disable_unclustered_high_rp_reschedule=pgr2_operands,
            )
        plans.append(plan)
    return tuple(plans)


def _wg_register_wave_grid_plan(
    mi_wave_tile_m,
    mi_wave_tile_n,
    warps_m,
    warps_n,
    *,
    prefetch_next=True,
    reverse_local_assignment=True,
    disable_unclustered_high_rp_reschedule=False,
):
    """Build a short-K direct-register MI16 wave-grid specialization."""
    return {
        "kind": "register_mi16_wave_grid",
        "mi_wave_tile_m": mi_wave_tile_m,
        "mi_wave_tile_n": mi_wave_tile_n,
        "warps_m": warps_m,
        "warps_n": warps_n,
        "num_warps": warps_m * warps_n,
        "prefetch_next": prefetch_next,
        "reverse_local_assignment": reverse_local_assignment,
        "disable_unclustered_high_rp_reschedule": (disable_unclustered_high_rp_reschedule),
    }


def _wg_power_of_two_fragments(extent, *, maximum=128, minimum=16):
    """Decompose one MFMA-aligned extent into descending binary fragments.

    This is a capability helper, not a performance policy: callers still
    choose the macro tile and decide whether the resulting fragment count is
    suitable for a register or LDS pipeline.  Keeping those decisions separate
    lets the bounded tuner enumerate legal non-power-of-two tiles without
    silently enabling an unmeasured kernel path.
    """
    if extent <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("fragment extents and bounds must be positive")
    if minimum & (minimum - 1) or maximum & (maximum - 1):
        raise ValueError("fragment bounds must be powers of two")
    if maximum % minimum != 0 or extent % minimum != 0:
        raise ValueError("extent must be aligned to the minimum fragment")

    fragments = []
    remaining = extent
    while remaining:
        fragment = min(maximum, 1 << (remaining.bit_length() - 1))
        if fragment < minimum:
            raise ValueError("extent cannot be represented by legal fragments")
        fragments.append(fragment)
        remaining -= fragment
    return tuple(fragments)


def _wg_register_fragmented_m_plan(
    m_fragments,
    block_n,
    block_k,
    *,
    num_warps=4,
    num_stages=1,
):
    """Build a register pipeline over explicit power-of-two M fragments."""
    starts = []
    start = 0
    for extent in m_fragments:
        starts.append((start, extent))
        start += extent
    return {
        "kind": "register_fragmented_m",
        "m_fragments": tuple(starts),
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


# Validated non-power-of-two geometries.  Each entry selects parameters for
# the common kernel above; it does not introduce a shape-specific kernel.
_wg_REGULAR_WAVE_GRID_CANDIDATES = {
    # Geometry, persistence, and LDS depth are selected independently while
    # every entry reuses the same regular-wave-grid kernel.
    # The hipBLASLt winner is logically MT96x128 in its column-major view.
    # Transposed back to this row-major API that is MT128x96: four waves form
    # a 2x2 grid, and each wave owns a regular 4x3 MI16 accumulator grid.
    # PGR2 keeps one future K64 in VGPRs while one K64 is consumed from LDS;
    # B remains cacheable because its complete N tile is reused by every M CTA.
    (98304, 80, 512):
    _wg_regular_wave_grid_plan(
        4,
        3,
        2,
        2,
        64,
        local_stages=1,
        pgr2_operands=True,
        num_xcds=8,
        reverse_local_assignment=False,
    ),
    # MI32 expresses the cold-cache vendor winner as a regular 1x7 per-wave
    # grid: four waves cover the 128x224 macro tile without padding N to a
    # power of two.
    (112, 114688, 1024):
    _wg_regular_wave_grid_plan(
        1,
        7,
        4,
        1,
        64,
        num_xcds=8,
        matrix_instr_nonkdim=32,
    ),
    # Tensile's MT160x384 / MIWT5x3 is expressed in its column-major view.
    # In this TN API it becomes MT384x160 with a 4x1 wave grid.
    (81920, 144, 2048):
    _wg_regular_wave_grid_plan(
        3,
        5,
        4,
        1,
        64,
        num_xcds=8,
        matrix_instr_nonkdim=32,
        reverse_local_assignment=False,
    ),
    (144, 81920, 2048):
    _wg_regular_wave_grid_plan(9, 5, 1, 4, 64),
    # Small-M, very-wide K512 grids need more independent CTAs than the old
    # MT192x192 / three-tiles-per-program mapping exposed. Keep M at 192 so
    # one tile covers the short axis, but use an MT192x96 tile along N.  At
    # this shallow K, VGPR-staged PGR2 is faster for both the aligned M176 case
    # and its M173 tail neighbor: direct-to-LDS does not have enough K work to
    # amortize its extra synchronization.
    (173, 147440, 512):
    _wg_regular_wave_grid_plan(
        6,
        3,
        2,
        2,
        64,
        pgr2_operands=True,
        reverse_local_assignment=False,
    ),
    (176, 147456, 512):
    _wg_regular_wave_grid_plan(
        6,
        3,
        2,
        2,
        64,
        pgr2_operands=True,
        reverse_local_assignment=False,
    ),
    (122880, 208, 1024):
    _wg_regular_wave_grid_plan(
        4,
        13,
        4,
        1,
        64,
        2,
        local_stages=2,
        linear_tiles=True,
    ),
    (240, 196608, 2048):
    _wg_regular_wave_grid_plan(
        15,
        4,
        1,
        4,
        64,
        local_stages=1,
        pgr2_operands=True,
        num_xcds=2,
        workgroup_mapping=16,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # The bounded geometry search prefers MT192x128 here.  It matches the
    # exhaustive hipBLASLt winner while reusing the common four-wave kernel.
    (3568, 4816, 512):
    _wg_regular_wave_grid_plan(6, 4, 2, 2, 64),
    # The vendor winner for these K512 shapes is a four-wave
    # MT256x256x64 PGR2 kernel.  Reuse the same parameterized geometry and
    # direct-to-LDS pipeline; nearby shapes remain on their measured winners.
    (1776, 36912, 512):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        local_stages=2,
        direct_to_lds=True,
    ),
    (688, 76000, 512):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        local_stages=2,
        direct_to_lds=True,
    ),
    (240, 65536, 512):
    _wg_regular_wave_grid_plan(
        15,
        4,
        1,
        4,
        64,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
}

# Compute-heavy selector validation targets.  Keep these on the parameterized
# four-wave family instead of silently falling back to the established
# eight-wave inter-wave kernel: they are intended to expose the remaining
# single-CTA pipeline and register-allocation gaps.  The selector chooses the
# geometry; this tuple is only the bounded promotion/validation corpus.
_wg_COMPUTE_HEAVY_4WAVE_TARGETS = (
    (4320, 5168, 4096),
    (5936, 7216, 8192),
    (7664, 9328, 16384),
    (10128, 11824, 32768),
    (13552, 15664, 4096),
    (2288, 37856, 4096),
    (37472, 2416, 8192),
    (3152, 26704, 16384),
    (28496, 3392, 4096),
    (4592, 19472, 8192),
    (18608, 4720, 16384),
    (6256, 15344, 32768),
    (14992, 6384, 4096),
    (11024, 12880, 8192),
    (12752, 10864, 16384),
    (960, 120112, 8192),
    (129536, 1088, 8192),
    (1520, 30000, 32768),
    (80336, 1680, 8192),
    (2496, 14000, 65536),
    (5392, 6576, 4096),
    (8848, 10768, 8192),
    (11632, 13904, 16384),
    (7168, 17456, 32768),
    (19472, 5680, 8192),
    (4016, 24272, 16384),
    (26416, 3888, 32768),
    (1856, 47120, 8192),
    (58128, 2320, 16384),
    (3312, 21968, 65536),
)


def _wg_compute_heavy_wave_grid_plan(shape):
    plans = _wg_ranked_wave_grid_plans(*shape, limit=5)
    m, n, k = shape
    # A rectangular PGR2 tile is worthwhile when it removes substantial tail
    # work on an extreme aspect ratio, or when very deep K amortizes the extra
    # CTA count of a smaller accumulator grid. Otherwise retain MT256x256.
    use_ranked_rectangle = (max(m, n) >= 64 * min(m, n) or (k >= 65536 and min(m, n) < 4096))
    if k == 4096 and max(m, n) >= 8 * min(m, n):
        rectangular = next(candidate for candidate in plans
                           if (16 * candidate["mi_wave_tile_m"] * candidate["warps_m"] == 256 and 16 *
                               candidate["mi_wave_tile_n"] * candidate["warps_n"] == 224))
        plan = {
            **rectangular,
            "local_stages": 2,
            "pgr2_operands": True,
            "direct_to_lds": True,
            "direct_chunk": 32,
            "pack_direct_chunks": True,
            "num_xcds": 8,
            "row_wise_epilogue": True,
        }
    elif use_ranked_rectangle:
        plan = {
            **plans[0],
            "local_stages": 2,
            "pgr2_operands": True,
            "direct_to_lds": True,
            "num_xcds": 8,
            "row_wise_epilogue": True,
        }
    else:
        # Compute-heavy grids favor the compact PGR2 source order. The older
        # phased order costs 2--4% without increasing residency.
        plan = _wg_regular_wave_grid_plan(
            8,
            8,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            num_xcds=8,
            row_wise_epilogue=True,
        )
    plan_block_m = 16 * plan["mi_wave_tile_m"] * plan["warps_m"]
    plan_block_n = 16 * plan["mi_wave_tile_n"] * plan["warps_n"]
    grid_m = (m + plan_block_m - 1) // plan_block_m
    grid_n = (n + plan_block_n - 1) // plan_block_n
    # A positive blocked mapping improves L2 reuse only when both axes provide
    # a useful band. Narrow-N and single-row grids retain the original order.
    plan = {
        **plan,
        "workgroup_mapping": 8 if grid_m >= 5 and grid_n >= 16 else 1,
    }
    # Deep-K PGR2 plans split each 256-wide physical backing into independently
    # scheduled source windows. ATT shows that shorter windows reduce global-
    # load, LDS-read, and MFMA stalls without changing the logical MFMA grid.
    # K>=32768 amortizes one 2-KiB chunk per source instruction. K16384 uses
    # the same windows only when at least 1280 workgroups amortize the extra
    # scheduling boundaries; smaller grids retain 4-KiB chunks. Selected
    # K8192 geometries below also use packed 2-KiB windows when family-wide
    # A/B measurements show that their lower descriptor pressure pays off.
    if (plan.get("direct_to_lds", False) and plan.get("pgr2_operands", False) and plan.get("mi_wave_tile_m") == 8
            and plan.get("mi_wave_tile_n") == 8 and plan.get("warps_m") == 2 and plan.get("warps_n") == 2
            and k == 4096):
        # A 32-wide source window lowers to one LDSDMA instruction, so the
        # source scheduler can interleave each physical load independently.
        # Sharing one 256-wide allocation keeps the descriptor/LDS cost of the
        # coarser path while avoiding its four-instruction scheduling bundle.
        plan = {
            **plan,
            "direct_chunk": 32,
            "pack_direct_chunks": True,
        }
    elif (plan.get("direct_to_lds", False) and plan.get("pgr2_operands", False) and k >= 16384):
        plan_block_m = (16 * plan["mi_wave_tile_m"] * plan["warps_m"])
        plan_block_n = (16 * plan["mi_wave_tile_n"] * plan["warps_n"])
        plan_grid = ((m + plan_block_m - 1) // plan_block_m * ((n + plan_block_n - 1) // plan_block_n))
        direct_chunk = (32 if k >= 32768 or (k == 16384 and plan_grid >= 1280) else 64)
        plan = {
            **plan,
            "direct_chunk": direct_chunk,
            # Keep 32-wide scheduling windows but share one allocation on a
            # full 256-wide axis. This reduces memdesc/address state without
            # changing global traffic or the logical MFMA decomposition.
            "pack_direct_chunks": (direct_chunk == 32 and (plan_block_m == 256 or plan_block_n == 256)),
        }
    elif (plan.get("direct_to_lds", False) and plan.get("pgr2_operands", False) and k == 8192
          and (m >= n or n < 64 * m)):
        # Share one physical LDS allocation per operand while retaining
        # 32-wide source scheduling windows. Extreme wide-N rectangles keep
        # their larger windows because their B-side locality is already good.
        plan = {
            **plan,
            "direct_chunk": 32,
            "pack_direct_chunks": True,
        }
    if k == 16384 and n >= 6 * m:
        # A single XCD-linear tile order preserves A locality for moderately
        # wide, deep-K grids; striping these grids across all XCDs makes each
        # partition revisit the same M bands with less useful L2 reuse.
        plan = {**plan, "num_xcds": 1}
    square_pgr2: bool = (plan.get("direct_to_lds", False) and plan.get("pgr2_operands", False)
                         and plan.get("mi_wave_tile_m") == 8 and plan.get("mi_wave_tile_n") == 8
                         and plan.get("warps_m") == 2 and plan.get("warps_n") == 2)
    packed32: bool = (plan.get("direct_chunk", 128) == 32 and plan.get("pack_direct_chunks", False))
    refill_grid_m = (m + 255) // 256
    # Merge each group of eight K(t+2) refills with the matching K(t+1) LDS
    # reads when deep K or a sufficiently large M grid amortizes the extra
    # scheduling boundaries.
    refill_read_window = (8 if (square_pgr2 and packed32 and n >= m and k < 65536 and
                                (k >= 16384 or (k == 8192 and refill_grid_m >= 32) or
                                 (k == 4096 and refill_grid_m < 32))) else 0)
    plan = {**plan, "refill_read_window": refill_read_window}
    plan = {
        **plan,
        # Two pipeline pairs per loop branch reduce control overhead without
        # changing the steady-state schedule through K32768.  A factor of
        # four expands the scheduling region enough to raise register pressure
        # sharply, while K65536 no longer amortizes the larger loop body.
        "loop_unroll_factor":
        2 if square_pgr2 and k <= 32768 else 1,
    }
    if square_pgr2:
        grid_m = (m + 255) // 256
        grid_n = (n + 255) // 256
        # A smaller positive mapping keeps neighboring N tiles close enough
        # to reuse A without forcing every XCD through an overly long N band.
        # Deep K amortizes mapping overhead and favors four-tile bands; very
        # small M grids and long K8192 N grids favor two-tile bands instead.
        if k == 4096:
            # K4096 is short enough that XCD distribution and edge-band
            # balance remain visible. Extreme rectangles follow their long
            # axis; smaller balanced grids avoid an eight-tile band whose
            # period aliases the eight XCDs. Large balanced grids retain it.
            short_grid_axis = min(grid_m, grid_n)
            if m >= 4 * n:
                plan = {**plan, "workgroup_mapping": 7}
            elif n >= 4 * m or short_grid_axis < 32:
                plan = {
                    **plan,
                    "workgroup_mapping": (7 if n < 4 * m and short_grid_axis < 20 else 4),
                }
        elif n >= m:
            if k == 8192 and grid_n >= 48:
                plan = {**plan, "workgroup_mapping": 2}
            elif k == 16384:
                if 2 * n <= 3 * m:
                    plan = {**plan, "workgroup_mapping": 7}
                elif packed32 and grid_m >= 16:
                    plan = {**plan, "workgroup_mapping": 4}
            elif k >= 32768 and k < 65536 and packed32:
                plan = {
                    **plan,
                    "workgroup_mapping": 2 if grid_m < 16 else 4,
                }
        # The mapping-2 K8192 family visits two neighboring N tiles per M
        # band. Keep a large balanced M grid linear across the device so those
        # bands retain their A locality; smaller M grids need two XCD stripes
        # to expose enough independent bands. Mapping-8 and extreme rectangles
        # retain all eight XCDs because their longer N bands already provide
        # locality without sacrificing device-level parallelism.
        if k == 8192 and plan.get("workgroup_mapping", 1) == 2:
            if grid_m >= 32:
                plan = {**plan, "num_xcds": 1}
            elif grid_m >= 8:
                plan = {**plan, "num_xcds": 2}
    plan_block_m = 16 * plan["mi_wave_tile_m"] * plan["warps_m"]
    plan_block_n = 16 * plan["mi_wave_tile_n"] * plan["warps_n"]
    plan_grid_m = (m + plan_block_m - 1) // plan_block_m
    plan_grid_n = (n + plan_block_n - 1) // plan_block_n
    if (k >= 65536 and min(plan_grid_m, plan_grid_n) <= 16 and max(plan_grid_m, plan_grid_n) >= 48):
        # A very deep reduction amplifies L2 locality differences between
        # otherwise identical four-wave kernels.  Eight XCD stripes combined
        # with an eight-tile band make a narrow grid revisit the shared
        # operand from too many independent streams.  Two stripes and a
        # two-tile band retain enough device parallelism while substantially
        # shortening that reuse distance.
        plan = {**plan, "num_xcds": 2, "workgroup_mapping": 2}
    return plan


_wg_REGULAR_WAVE_GRID_CANDIDATES.update(
    {shape: _wg_compute_heavy_wave_grid_plan(shape)
     for shape in _wg_COMPUTE_HEAVY_4WAVE_TARGETS})

# Default dispatch is deliberately narrower than kernel capability.  A new
# geometry is promoted only after a paired measurement clearly beats the
# established inter-wave fallback; losing or noise-level candidates remain
# available to the bounded tuner.
_wg_REGULAR_WAVE_GRID_SPECIALIZATIONS = {
    shape: _wg_REGULAR_WAVE_GRID_CANDIDATES[shape]
    for shape in (
        (98304, 80, 512),
        (112, 114688, 1024),
        (81920, 144, 2048),
        (144, 81920, 2048),
        (173, 147440, 512),
        (176, 147456, 512),
        (122880, 208, 1024),
        (240, 196608, 2048),
        (3568, 4816, 512),
        (1776, 36912, 512),
        (688, 76000, 512),
        *_wg_COMPUTE_HEAVY_4WAVE_TARGETS,
    )
}

# A tall, narrow K6144 GEMM needs the vendor-shaped MT320x160 PGR2 pipeline.
# Two independent LDS banks let K(t+1) publication overlap K(t)'s MFMA work;
# operand-local padding keeps the two K32 reads out of the same bank phase.
_wg_REGULAR_WAVE_GRID_SPECIALIZATIONS[(101904, 304, 6144)] = (_wg_regular_wave_grid_plan(
    10,
    5,
    2,
    2,
    64,
    local_stages=2,
    pgr2_operands=True,
    num_xcds=8,
    workgroup_mapping=8,
    sink_insts_to_avoid_spills=True,
    llvm_sched_strategy="iterative-minreg",
))

# Bounded-search winners for the shape-diversity corpus.  These are compact
# plan records for the single parameterized wave-grid kernel, not independent
# kernels.  Keeping the measured choices separate from capability generation
# lets future selector work replace the table without touching kernel code.
_wg_SOTA77_WAVE_GRID_PROMOTIONS = {
    # The vendor-selected MT128x128x128/PGR2 geometry is also effective when
    # K contains an odd number of K32 blocks. Run 110 complete K64 blocks in
    # the two-stage pipeline, then append the final K32 in increasing K order.
    # A wide row-wise store plus two delayed PGR2 reads shorten the final
    # accumulator/store overlap; constraining occupancy keeps that schedule
    # stable.  A 301-round paired run measured this plan 1.016x faster than
    # the otherwise identical narrow epilogue.
    (2032, 2048, 7072):
    _wg_regular_wave_grid_plan(
        4,
        4,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        pgr2_late_read_count=2,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=1,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        waves_per_eu=1,
    ),
    # Match the vendor's K128 PGR2 depth while retaining an exactly balanced
    # four-way Split-K.  Each workgroup consumes 36 complete K128 macros plus
    # one K64 tail.  The K128 register bank is physically published through
    # two bank-safe K64 LDS stages, and streaming A keeps reusable B in cache.
    (2032, 512, 18688):
    _wg_regular_wave_grid_plan(
        4,
        4,
        2,
        2,
        128,
        local_stages=1,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        split_k=4,
        reduce_tile=(16, 64),
        reduce_warps=4,
    ),
    # Avoid padding the short M dimension from 80 to 96.  A one-by-four wave
    # grid keeps the logical MT80x192 decomposition regular and reduces LDS
    # and register use while retaining the common PGR2 source pipeline.
    (80, 65552, 6144):
    _wg_regular_wave_grid_plan(
        5,
        3,
        1,
        4,
        64,
        pgr2_operands=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (1072, 1584, 32768):
    _wg_regular_wave_grid_plan(
        5,
        7,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        # Keep the early B/A read set small, then pair the remaining reads
        # with next-stage publications.  This shortens operand lifetimes while
        # giving the long-K refill enough MFMA cover.  An eight-tile locality
        # band also keeps each complete N row adjacent under four-way Split-K.
        pgr2_late_read_count=8,
        row_wise_epilogue=False,
        num_xcds=1,
        workgroup_mapping=8,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        split_k=4,
        reduce_tile=(8, 64),
    ),
    # Avoid the two-launch N256 + N160 hybrid when one vendor-style
    # MT224x224 grid covers the shape with only a single masked N tail.
    (224, 106496, 6144):
    _wg_regular_wave_grid_plan(
        7,
        7,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # The same one-M-tile MT224x224 family scales to the wider 768-tile grid.
    # Its streaming-B cache policy remains valid because no workgroup can
    # reuse a B tile along M.
    (224, 172032, 6144):
    _wg_regular_wave_grid_plan(
        7,
        7,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # The register fallback keeps a much wider B operand live for this
    # small-M K1024 problem.  The reusable direct PGR2 pipeline is faster than
    # VGPR staging here; MT128x96 also gives the 80-row tail enough independent
    # MFMA work to cover each direct refill without widening the N tile.
    (80, 17456, 1024):
    _wg_regular_wave_grid_plan(
        4,
        3,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # Four K partitions turn the MT80x128 output grid into exactly 256
    # contributors.  K128 halves the direct-refill synchronization cadence;
    # the compact one-by-four wave layout also avoids register-path overhead.
    (279, 2048, 4096):
    _wg_regular_wave_grid_plan(
        5,
        2,
        1,
        4,
        128,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=4,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        split_k=4,
        reduce_tile=(16, 64),
        reduce_warps=4,
    ),
    # Two K partitions turn the MT96x128 output grid into exactly 256
    # contributors.  The K128 direct-refill loop removes half of the local
    # synchronization points while retaining a compact four-wave CTA.
    (677, 2048, 4096):
    _wg_regular_wave_grid_plan(
        3,
        4,
        2,
        2,
        128,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        split_k=2,
        reduce_tile=(16, 64),
        reduce_warps=4,
    ),
    # The wider MT96x128 tile reduces masked-M overhead while XCD4/WGM4 keeps
    # its 192-workgroup grid balanced.  Pair-unrolled K64 is faster than the
    # exact-fill MT80x128/K128 plan under the official cold-cache workload;
    # streaming B preserves the compact, heavily reused A working set.
    (279, 8192, 2048):
    _wg_regular_wave_grid_plan(
        3,
        4,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=4,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        loop_unroll_factor=2,
    ),
    # Use an exact one-CTA-per-CU grid instead of launching only 192
    # MT96x192 workgroups.  MT80x192 exposes 256 workgroups while preserving
    # the common two-stage PGR2 pipeline and avoiding Split-K reduction cost.
    (272, 12272, 6144):
    _wg_regular_wave_grid_plan(
        5,
        3,
        1,
        4,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=2,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # MT80x128 makes four M tiles, so a two-way K partition launches exactly
    # 256 contributors and saturates gfx950.  Four-XCD remapping keeps the
    # reduction tiles local while avoiding the underfilled MT96x128 grid.
    # Pair unrolling measured 1.008x faster in an official 301-round A/B.
    (279, 4096, 4352):
    _wg_regular_wave_grid_plan(
        5,
        2,
        1,
        4,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=4,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        loop_unroll_factor=2,
        split_k=2,
        reduce_tile=(16, 64),
        reduce_warps=4,
    ),
    # Match the exhaustive hipBLASLt winner's effective MT96x128 geometry.
    # K128 halves the direct-refill synchronization cadence, while four-XCD
    # workgroup bands preserve A locality across the wide N grid.
    (677, 4096, 2048):
    _wg_regular_wave_grid_plan(
        3,
        4,
        2,
        2,
        128,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=4,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (336, 12272, 256):
    _wg_regular_wave_grid_plan(3, 6, 2, 2, 64),
    (147472, 80, 256):
    _wg_regular_wave_grid_plan(4, 3, 2, 2, 64),
    # Match the vendor's rectangular MT224x160 instead of overcomputing an
    # MT224x224 tile.  The wide-N grid supplies ample CTA parallelism, so the
    # smaller N side wins by reducing both MFMA and operand traffic.
    (400, 73744, 4096):
    _wg_regular_wave_grid_plan(
        7,
        5,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (117136, 80, 512):
    _wg_regular_wave_grid_plan(8, 5, 4, 1, 64),
    (528, 73744, 256):
    _wg_regular_wave_grid_plan(
        6,
        4,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        row_wise_epilogue=True,
        num_xcds=8,
        # Eight adjacent N tiles reuse each A tile more effectively than the
        # vendor's six-tile band for this three-row, very-wide CTA grid.
        workgroup_mapping=8,
        reverse_local_assignment=False,
    ),
    (73744, 224, 4096):
    _wg_regular_wave_grid_plan(
        5,
        7,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (50160, 160, 256):
    _wg_regular_wave_grid_plan(4, 11, 4, 1, 64),
    (192, 147472, 8192):
    _wg_regular_wave_grid_plan(
        6,
        6,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        streamk_tail_split=32,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (28784, 80, 256):
    _wg_regular_wave_grid_plan(2, 5, 4, 1, 64),
    (15440, 19696, 256):
    _wg_regular_wave_grid_plan(
        7,
        4,
        2,
        2,
        64,
        wide_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
    ),
    # The K512 register fallback leaves the four-wave CTA load-bound.  Reuse
    # the vendor-shaped MT224x128 decomposition with a packed direct-to-LDS
    # PGR2 pipeline so the two resident K64 stages overlap with MFMA work.
    (4272, 3184, 512):
    _wg_regular_wave_grid_plan(
        7,
        4,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=32,
        pack_direct_chunks=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=8,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (1328, 12272, 8192):
    _wg_regular_wave_grid_plan(6, 6, 2, 2, 64),
    # Match the vendor's logical MT176x256 geometry without padding M to 192.
    # The two-stage VGPR-staged PGR2 pipeline is faster here than the packed
    # direct-to-LDS path while preserving the same reusable wave-grid kernel.
    (528, 147472, 8192):
    _wg_regular_wave_grid_plan(
        11,
        4,
        1,
        4,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
}
_wg_REGULAR_WAVE_GRID_SPECIALIZATIONS.update(_wg_SOTA77_WAVE_GRID_PROMOTIONS)

# BF16 uses the same 16-bit LDS representation as FP16.  Keep selection
# explicit: only measured shapes enter this path while the broader BF16
# corpus retains the established LDS fallback.
_wg_BF16_WAVE_GRID_SPECIALIZATIONS = {
    # MI32 halves the static MFMA count for this shallow-K production shape.
    # A 1x4 wave grid keeps the full N256 tile while halving the M tile, which
    # measured 1.079x faster than the persistent MI16 MT256x256 plan.
    (819200, 1024, 192):
    _wg_regular_wave_grid_plan(
        4,
        2,
        1,
        4,
        64,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
        matrix_instr_nonkdim=32,
    ),
}

_wg_SOTA77_TRANSPOSED_WAVE_GRID_PROMOTIONS = {
    # Transposition turns the extremely wide output into a regular traversal.
    # MT256x192 reduces the number of shallow-K workgroups relative to the
    # previous MT256x160 direct plan; VGPR staging is faster here and WGM1
    # keeps the transposed operand traversal contiguous.
    (1136, 117136, 1024):
    _wg_regular_wave_grid_plan(
        8,
        6,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # Match the pinned vendor winner's MT160x160 geometry in transposed
    # coordinates.  PGR2 shortens operand live ranges across the K768 loop;
    # XCD-aware WGM6 bands retain B.T locality over the very wide M grid.
    (144, 147472, 768):
    _wg_regular_wave_grid_plan(
        5,
        5,
        2,
        2,
        64,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # In transposed coordinates, two MT192x192 outputs per program reduce the
    # shallow-K grid from 461 one-tile programs to 193 persistent pairs.  This
    # retains enough device parallelism while overlapping the next tile's K0
    # loads with the current tile's epilogue.
    (73744, 192, 256):
    _wg_regular_wave_grid_plan(
        6,
        6,
        2,
        2,
        64,
        tiles_per_program=2,
    ),
    (240, 106512, 512):
    _wg_regular_wave_grid_plan(7, 8, 2, 2, 64),
    (17456, 224, 256):
    _wg_regular_wave_grid_plan(4, 5, 2, 2, 64),
    (176, 106512, 512):
    _wg_regular_wave_grid_plan(7, 6, 2, 2, 64),
    # For this K4096 family, the vendor's exact MT224x224 geometry is faster
    # than the generic direct MT256x192 fallback: the balanced tile removes
    # M/N padding without needing a second local stage.
    (896, 65552, 4096):
    _wg_regular_wave_grid_plan(
        7,
        7,
        2,
        2,
        64,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # Keep the vendor-selected MT256x256 geometry, but use packed 32-wide
    # direct publication for this long K loop.  Pair unrolling exposes two
    # refill windows at once and closes the residual global/LDS overlap gap.
    (1136, 90128, 4096):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=32,
        pack_direct_chunks=True,
        loop_unroll_factor=2,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # This transposed deep-K problem keeps the same MT192x192 geometry, but
    # needs the two-stage PGR2 pipeline to overlap its long K traversal.
    (12272, 1328, 8192):
    _wg_regular_wave_grid_plan(
        6,
        6,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # In transposed coordinates the vendor winner is MT256x160.  Matching
    # that rectangle and using packed 32-wide direct windows removes most of
    # the former tail work and register-lifetime gap.  Row-major B staging
    # plus the longer PGR2 read cover was 1.004x faster in a 301-round paired
    # run on GPU4 and 1.008x faster on GPU5.
    (2032, 18688, 512):
    _wg_regular_wave_grid_plan(
        8,
        5,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=32,
        pack_direct_chunks=True,
        pgr2_late_read_count=2,
        read_cover_sixteenths=12,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        num_xcds=8,
        workgroup_mapping=6,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    (208, 106496, 6144):
    _wg_regular_wave_grid_plan(7, 8, 2, 2, 64),
    (160, 106512, 512):
    _wg_regular_wave_grid_plan(5, 5, 2, 2, 64),
    (131088, 224, 256):
    _wg_regular_wave_grid_plan(7, 6, 2, 2, 64),
}


@lru_cache(maxsize=None)
def _wg_plan(m, n, k):
    """Return a validated intra-wave plan, or ``None`` to fall back."""
    # Exact bounded-search winners take precedence over reusable family
    # fallbacks.  Otherwise a broad rule can silently shadow a measured plan
    # for the same shape while making the specialization table look active.
    specialization = _wg_REGULAR_WAVE_GRID_SPECIALIZATIONS.get((m, n, k))
    if specialization is not None:
        return specialization
    # When the wide dimension is exactly divisible by 224, one MT224x224
    # grid avoids both the M padding of MT256 and the second launch used by
    # the N256 + N160 hybrid. This is one parameterized family; only the
    # launch extent changes between shapes.
    if 208 <= m <= 224 and k == 6144 and n >= 65536 and n % 224 == 0:
        return _wg_regular_wave_grid_plan(
            7,
            7,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            row_wise_epilogue=True,
            num_xcds=8,
            workgroup_mapping=1,
            reverse_local_assignment=True,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    # Very wide, deep-K grids with roughly two 256-row output bands need a
    # rectangular tile to avoid the half-empty final M tile.  MT192x256 also
    # matches the vendor wave geometry.  Keep 32-wide direct-load windows: the
    # packed N256 backing makes every window independently schedulable while
    # retaining one LDS descriptor per operand.
    if 480 <= m <= 576 and n >= 128 * m and k == 8192:
        return _wg_regular_wave_grid_plan(
            6,
            8,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            direct_chunk=32,
            pack_direct_chunks=True,
            row_wise_epilogue=True,
            num_xcds=8,
            workgroup_mapping=6,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    # Small-M deep reductions need enough N tiles to fill the device without
    # keeping the register kernel's complete B tile live across the K loop.
    # Both ranges use the same MT96x192 regular grid; PGR2 is profitable for
    # the single-M-tile case, whereas three M tiles expose enough independent
    # work for the lighter rectangular K64 pipeline.
    if k == 6144 and n >= 32 * m:
        if 64 <= m <= 96:
            return _wg_regular_wave_grid_plan(
                3,
                6,
                2,
                2,
                64,
                pgr2_operands=True,
                row_wise_epilogue=True,
                num_xcds=8,
                workgroup_mapping=6,
                reverse_local_assignment=False,
                sink_insts_to_avoid_spills=True,
                disable_unclustered_high_rp_reschedule=True,
            )
        if 240 <= m <= 288:
            return _wg_regular_wave_grid_plan(
                3,
                6,
                2,
                2,
                64,
                local_stages=2,
                pgr2_operands=True,
                row_wise_epilogue=True,
                reverse_local_assignment=False,
                sink_insts_to_avoid_spills=True,
                disable_unclustered_high_rp_reschedule=True,
            )
    # A shallow K1024 reduction with a very wide N grid benefits from
    # four-wave MT256x256 ownership and its PGR2 pipeline.  Keep this rule
    # bounded to the measured small-M family; the final artifact remains the
    # authority for residency rather than the source wave count.
    if 640 <= m <= 800 and n >= 128 * m and k == 1024:
        return _wg_regular_wave_grid_plan(
            8,
            8,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            num_xcds=8,
            workgroup_mapping=6,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
            row_wise_epilogue=True,
        )
    # A short-M, very-wide-N K4096 grid benefits from the vendor-style
    # MT224x192 mapping.  Four waves keep two compact PGR2 workgroups resident,
    # while the 7x6 MFMA grid avoids hard-coding one problem size.  The
    # positive mapping matches the operand-A reuse direction for this TN
    # family; outside this bounded geometry the mature inter-wave path wins.
    if 384 <= m <= 448 and n >= 65536 and k == 4096:
        return _wg_regular_wave_grid_plan(
            7,
            6,
            2,
            2,
            64,
            # Keep the two K64 banks inside one chunked allocation per
            # operand.  This removes the redundant stage-reuse barrier while
            # avoiding the base/m0 materialization of per-fragment banks.
            local_stages=2,
            pgr2_operands=True,
            # Keep the whole publication/read/MFMA window available to the
            # dependency-aware auto-interleaver.  Splitting out the last four
            # reads looked attractive in isolation, but over-constrained the
            # final schedule in repeated cold-L2 paired comparisons.
            pgr2_late_read_count=0,
            num_xcds=8,
            workgroup_mapping=6,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    # K768 is long enough to amortize the two-stage direct pipeline, but short
    # enough for output padding to dominate a tall, narrow GEMM.  In this
    # interval MT256x160 covers N with five nearly full tiles while MT256x256
    # needs four substantially padded tiles.  Bound the finer grid to four to
    # seven gfx950 device waves so its additional CTAs do not replace padding
    # waste with launch and fill/drain overhead.
    n160_tiles = (n + 159) // 160
    n256_tiles = (n + 255) // 256
    n160_grid = ((m + 255) // 256) * n160_tiles
    if (k == 768 and n160_tiles == 5 and n256_tiles == 4 and n160_tiles * 160 < n256_tiles * 256
            and 4 * 256 <= n160_grid <= 7 * 256):
        return _wg_regular_wave_grid_plan(
            8,
            5,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            direct_chunk=64,
            num_xcds=8,
        )
    # For tall, narrow K256 GEMMs, partition N into the smallest number of
    # equal <=128-wide, MFMA-aligned tiles.  For example, N=336 becomes
    # 3x112 instead of two padded 256-wide tiles.  Four waves span M=256;
    # fine 32-wide direct-load windows expose enough independent memory work
    # for the MFMA pipeline without changing the physical LDS allocation.
    if k == 256 and m >= 64 * n and n >= 64:
        n_tiles = (n + 127) // 128
        if n % n_tiles == 0:
            block_n = n // n_tiles
            if block_n % 16 == 0 and 32 <= block_n <= 128:
                return _wg_regular_wave_grid_plan(
                    4,
                    block_n // 16,
                    4,
                    1,
                    64,
                    local_stages=2,
                    pgr2_operands=True,
                    direct_to_lds=True,
                    direct_chunk=32,
                    pack_direct_chunks=True,
                    num_xcds=8,
                    reverse_local_assignment=False,
                )
    # Shallow K512 and a very wide output are sensitive to both the M padding
    # cliff and the number of independent CTAs.  Select one of four regular
    # geometries at MFMA-aligned M boundaries.  Up through M=192, one
    # MT192x192 tile amortizes the common B traversal; at M=160 the smaller
    # MT160x160 tile is measurably better.  Larger M values use one nearly-full
    # 224/256-row tile.  Exact promoted plans above retain precedence.
    if 160 <= m <= 240 and k == 512 and n >= 65536:
        if m <= 160:
            return _wg_regular_wave_grid_plan(5, 5, 2, 2, 64)
        if m <= 192:
            return _wg_regular_wave_grid_plan(6, 6, 2, 2, 64)
        if m <= 224:
            return _wg_regular_wave_grid_plan(7, 6, 2, 2, 64)
        return _wg_regular_wave_grid_plan(8, 6, 2, 2, 64)
    # A regular power-of-two tile pads M=48 to 64 and spends one quarter of
    # its MFMAs on discarded rows.  Express the physical tile as 32+16 while
    # retaining one shared B load.  This rule describes a geometry family;
    # the kernel itself is parameterized by both fragment extents.
    if m == 48 and k == 256 and n >= 32768:
        return _wg_register_fragmented_m_plan(_wg_power_of_two_fragments(m), 256, 64)
    if m == 112 and k == 1024 and n >= 32768:
        return _wg_register_fragmented_m_plan(_wg_power_of_two_fragments(m), 256, 64)
    return None


# Register-staged wave-grid kernels.
@triton.jit
def _wg_kernel_register_mi16_wave_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    MI_WAVE_TILE_M: tl.constexpr,
    MI_WAVE_TILE_N: tl.constexpr,
    WARPS_M: tl.constexpr,
    WARPS_N: tl.constexpr,
    PREFETCH_NEXT: tl.constexpr,
):
    """Short-K MI16 grid whose global loads directly produce dot operands."""
    block_k: tl.constexpr = 32
    block_m: tl.constexpr = 16 * MI_WAVE_TILE_M * WARPS_M
    block_n: tl.constexpr = 16 * MI_WAVE_TILE_N * WARPS_N
    tl.static_assert(WARPS_M * WARPS_N == 4)
    tl.static_assert(K % block_k == 0)

    program = tl.program_id(0).to(tl.int32)
    grid_n: tl.constexpr = tl.cdiv(N, block_n)
    pid_m = program // grid_n
    pid_n = program % grid_n

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)

    a_group_m: tl.constexpr = 16 * WARPS_M
    cols = pid_n * block_n + tl.arange(0, block_n)
    global_cols = tl.where(cols < N, cols, 0)
    rk = tl.arange(0, block_k)
    rows = tl.tuple([])
    a_offsets = tl.tuple([])
    for row in tl.static_range(MI_WAVE_TILE_M):
        row_indices = (pid_m * block_m + row * a_group_m + tl.arange(0, a_group_m))
        global_rows = tl.where(row_indices < M, row_indices, 0)
        rows += tl.tuple([row_indices])
        a_offsets += tl.tuple([tlx.require_layout(
            global_rows[:, None] * stride_am + rk[None, :] * stride_ak,
            dot_a,
        )])
    b_offsets = tlx.require_layout(
        rk[:, None] * stride_bk + global_cols[None, :] * stride_bn,
        dot_b,
    )

    accumulators = tl.tuple([tlx.zeros((a_group_m, block_n), tl.float32, layout=mma) for _ in range(MI_WAVE_TILE_M)])
    current_a = tl.tuple([tlx.buffer_load(a_ptr, a_offsets[row], contiguity=8) for row in range(MI_WAVE_TILE_M)])
    current_b = tlx.buffer_load(b_ptr, b_offsets, cache=".cg", contiguity=8)
    if PREFETCH_NEXT:
        for kb in tl.range(1, K // block_k, num_stages=1):
            next_a = tl.tuple([
                tlx.buffer_load(
                    a_ptr,
                    a_offsets[row] + kb * block_k * stride_ak,
                    contiguity=8,
                ) for row in range(MI_WAVE_TILE_M)
            ])
            next_b = tlx.buffer_load(
                b_ptr,
                b_offsets + kb * block_k * stride_bk,
                cache=".cg",
                contiguity=8,
            )
            accumulators = tl.tuple([
                tl.dot(
                    current_a[row],
                    current_b,
                    accumulators[row],
                    allow_tf32=False,
                    out_dtype=tl.float32,
                ) for row in range(MI_WAVE_TILE_M)
            ])
            current_a = next_a
            current_b = next_b
        accumulators = tl.tuple([
            tl.dot(
                current_a[row],
                current_b,
                accumulators[row],
                allow_tf32=False,
                out_dtype=tl.float32,
            ) for row in range(MI_WAVE_TILE_M)
        ])
    else:
        accumulators = tl.tuple([
            tl.dot(
                current_a[row],
                current_b,
                accumulators[row],
                allow_tf32=False,
                out_dtype=tl.float32,
            ) for row in range(MI_WAVE_TILE_M)
        ])
        for kb in tl.range(1, K // block_k, num_stages=1):
            a_values = tl.tuple([
                tlx.buffer_load(
                    a_ptr,
                    a_offsets[row] + kb * block_k * stride_ak,
                    contiguity=8,
                ) for row in range(MI_WAVE_TILE_M)
            ])
            b_value = tlx.buffer_load(
                b_ptr,
                b_offsets + kb * block_k * stride_bk,
                cache=".cg",
                contiguity=8,
            )
            accumulators = tl.tuple([
                tl.dot(
                    a_values[row],
                    b_value,
                    accumulators[row],
                    allow_tf32=False,
                    out_dtype=tl.float32,
                ) for row in range(MI_WAVE_TILE_M)
            ])

    for row in tl.static_range(MI_WAVE_TILE_M):
        output_ptrs = tlx.require_layout(
            c_ptr + rows[row][:, None] * stride_cm + cols[None, :] * stride_cn,
            mma,
            pin=False,
        )
        output_mask = tlx.require_layout(
            (rows[row][:, None] < M) & (cols[None, :] < N),
            mma,
            pin=False,
        )
        tl.store(
            output_ptrs,
            accumulators[row].to(c_ptr.dtype.element_ty),
            mask=output_mask,
        )


@triton.jit
def _wg_kernel_register_split_m2(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M_HEAD: tl.constexpr,
    M_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Register-staged GEMM with two reusable power-of-two M fragments."""
    block_m: tl.constexpr = M_HEAD + M_TAIL
    tl.static_assert(M_HEAD >= 16 and M_TAIL >= 16)
    tl.static_assert(K % BLOCK_K == 0)

    program = tl.program_id(0).to(tl.int32)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    pid_m = program // grid_n
    pid_n = program % grid_n

    head_rows = pid_m * block_m + tl.arange(0, M_HEAD)
    tail_rows = pid_m * block_m + M_HEAD + tl.arange(0, M_TAIL)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    input_head_rows = tl.where(head_rows < M, head_rows, 0)
    input_tail_rows = tl.where(tail_rows < M, tail_rows, 0)
    input_cols = tl.where(cols < N, cols, 0)
    rk = tl.arange(0, BLOCK_K)

    acc_head = tl.zeros((M_HEAD, BLOCK_N), tl.float32)
    acc_tail = tl.zeros((M_TAIL, BLOCK_N), tl.float32)
    for kb in tl.range(0, K // BLOCK_K, num_stages=1):
        reduction = kb * BLOCK_K + rk
        b_value = tl.load(
            b_ptr + reduction[:, None] * stride_bk + input_cols[None, :] * stride_bn,
            cache_modifier=".cg",
        )
        a_head = tl.load(a_ptr + input_head_rows[:, None] * stride_am + reduction[None, :] * stride_ak, )
        a_tail = tl.load(a_ptr + input_tail_rows[:, None] * stride_am + reduction[None, :] * stride_ak, )
        acc_head = tl.dot(
            a_head,
            b_value,
            acc_head,
            allow_tf32=False,
            out_dtype=tl.float32,
        )
        acc_tail = tl.dot(
            a_tail,
            b_value,
            acc_tail,
            allow_tf32=False,
            out_dtype=tl.float32,
        )

    head_ptrs = (c_ptr + head_rows[:, None] * stride_cm + cols[None, :] * stride_cn)
    tail_ptrs = (c_ptr + tail_rows[:, None] * stride_cm + cols[None, :] * stride_cn)
    tl.store(
        head_ptrs,
        acc_head.to(c_ptr.dtype.element_ty),
        mask=(head_rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        tail_ptrs,
        acc_tail.to(c_ptr.dtype.element_ty),
        mask=(tail_rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _wg_kernel_register_split_m3(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Three-fragment form of the shared-B register pipeline."""
    block_m: tl.constexpr = M0 + M1 + M2
    tl.static_assert(M0 >= 16 and M1 >= 16 and M2 >= 16)
    tl.static_assert(K % BLOCK_K == 0)

    program = tl.program_id(0).to(tl.int32)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    pid_m = program // grid_n
    pid_n = program % grid_n
    rows0 = pid_m * block_m + tl.arange(0, M0)
    rows1 = pid_m * block_m + M0 + tl.arange(0, M1)
    rows2 = pid_m * block_m + M0 + M1 + tl.arange(0, M2)
    input_rows0 = tl.where(rows0 < M, rows0, 0)
    input_rows1 = tl.where(rows1 < M, rows1, 0)
    input_rows2 = tl.where(rows2 < M, rows2, 0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    input_cols = tl.where(cols < N, cols, 0)
    rk = tl.arange(0, BLOCK_K)

    acc0 = tl.zeros((M0, BLOCK_N), tl.float32)
    acc1 = tl.zeros((M1, BLOCK_N), tl.float32)
    acc2 = tl.zeros((M2, BLOCK_N), tl.float32)
    for kb in tl.range(0, K // BLOCK_K, num_stages=1):
        reduction = kb * BLOCK_K + rk
        b_value = tl.load(
            b_ptr + reduction[:, None] * stride_bk + input_cols[None, :] * stride_bn,
            cache_modifier=".cg",
        )
        a0 = tl.load(a_ptr + input_rows0[:, None] * stride_am + reduction[None, :] * stride_ak)
        a1 = tl.load(a_ptr + input_rows1[:, None] * stride_am + reduction[None, :] * stride_ak)
        a2 = tl.load(a_ptr + input_rows2[:, None] * stride_am + reduction[None, :] * stride_ak)
        acc0 = tl.dot(a0, b_value, acc0, allow_tf32=False, out_dtype=tl.float32)
        acc1 = tl.dot(a1, b_value, acc1, allow_tf32=False, out_dtype=tl.float32)
        acc2 = tl.dot(a2, b_value, acc2, allow_tf32=False, out_dtype=tl.float32)

    tl.store(
        c_ptr + rows0[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc0.to(c_ptr.dtype.element_ty),
        mask=(rows0[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        c_ptr + rows1[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc1.to(c_ptr.dtype.element_ty),
        mask=(rows1[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        c_ptr + rows2[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc2.to(c_ptr.dtype.element_ty),
        mask=(rows2[:, None] < M) & (cols[None, :] < N),
    )


# Intra-wave LDS/MFMA kernels.
def _wg_swizzled_offset_bases(shape, contiguous_dim):
    """Build the padded-LDS bit order used by direct-to-LDS loads."""

    def basis(dim, bit):
        return [1 << bit, 0] if dim == 0 else [0, 1 << bit]

    free_dim = 1 - contiguous_dim
    contiguous_bits = int(shape[contiguous_dim]).bit_length() - 1
    free_bits = int(shape[free_dim]).bit_length() - 1
    return ([basis(contiguous_dim, bit)
             for bit in range(contiguous_bits)] + [basis(free_dim, bit) for bit in range(4, free_bits)] +
            [basis(free_dim, bit) for bit in range(min(4, free_bits))])


_wg_A_BASES_64X32 = tl.constexpr(_wg_swizzled_offset_bases((64, 32), 1))
_wg_A_BASES_64X64 = tl.constexpr(_wg_swizzled_offset_bases((64, 64), 1))
_wg_A_BASES_128X64 = tl.constexpr(_wg_swizzled_offset_bases((128, 64), 1))
_wg_N_BASES_64X128 = tl.constexpr(_wg_swizzled_offset_bases((64, 128), 1))
_wg_B_BASES_32X128 = tl.constexpr(_wg_swizzled_offset_bases((32, 128), 0))
_wg_B_BASES_32X16 = tl.constexpr(_wg_swizzled_offset_bases((32, 16), 0))
_wg_B_BASES_32X64 = tl.constexpr(_wg_swizzled_offset_bases((32, 64), 0))
_wg_B_BASES_32X32 = tl.constexpr(_wg_swizzled_offset_bases((32, 32), 0))
_wg_B_BASES_64X16 = tl.constexpr(_wg_swizzled_offset_bases((64, 16), 0))
_wg_B_BASES_64X32 = tl.constexpr(_wg_swizzled_offset_bases((64, 32), 0))
_wg_B_BASES_64X64 = tl.constexpr(_wg_swizzled_offset_bases((64, 64), 0))
_wg_B_BASES_64X128 = tl.constexpr(_wg_swizzled_offset_bases((64, 128), 0))
_wg_A_OFFSET_LAYOUT_64X32 = tlx.layout(
    shape=((4, 4, 16), (8, )),
    stride=((8, 512, 32), (1, )),
)
_wg_N_CONTIG_OFFSET_LAYOUT_64X64_4W = tlx.layout(
    shape=((8, 4, 8), (8, 2)),
    stride=((8, 1024, 64), (1, 512)),
)
_wg_N_CONTIG_OFFSET_LAYOUT_64X128_4W = tlx.layout(
    shape=((16, 4, 4), (8, 4)),
    stride=((8, 2048, 128), (1, 512)),
)
_wg_N_CONTIG_OFFSET_LAYOUT_64X128_8W = tlx.layout(
    shape=((16, 4, 8), (8, 2)),
    stride=((8, 2048, 128), (1, 1024)),
)
_wg_B_OFFSET_LAYOUT_32X128 = tlx.layout(
    shape=((4, 8, 8), (8, 2)),
    stride=((1024, 16, 1), (128, 8)),
)
_wg_A_OFFSET_LAYOUT_128X64_4W = tlx.layout(
    shape=((8, 8, 4), (8, 4)),
    stride=((8, 1024, 64), (1, 256)),
)
_wg_B_OFFSET_LAYOUT_64X128_4W = tlx.layout(
    shape=((8, 8, 4), (8, 2, 2)),
    stride=((1024, 16, 1), (128, 8, 4)),
)
_wg_A_OFFSET_LAYOUT_64X64_4W = tlx.layout(
    shape=((8, 4, 8), (8, 2)),
    stride=((8, 1024, 64), (1, 512)),
)
_wg_B_OFFSET_LAYOUT_64X64_4W = tlx.layout(
    shape=((8, 4, 8), (8, 2)),
    stride=((512, 16, 1), (64, 8)),
)
_wg_A_OFFSET_LAYOUT_32X64_4W = tlx.layout(
    shape=((8, 32), (8, )),
    stride=((8, 64), (1, )),
)
_wg_B_OFFSET_LAYOUT_64X32_4W = tlx.layout(
    shape=((8, 32), (8, )),
    stride=((256, 1), (32, )),
)
# Eight-wave direct-to-LDS variants.  Moving one value bit into the thread
# mode preserves the same logical coordinate map while distributing each
# backing extent across all 512 CTA threads.  The 128-wide forms match the
# mature inter-wave kernel; the 32/64-wide forms are their exact bit-factor
# counterparts for parameterized rectangular tiles.
_wg_A_OFFSET_LAYOUT_128X64_8W = tlx.layout(
    shape=((8, 8, 8), (8, 2)),
    stride=((8, 1024, 64), (1, 512)),
)
_wg_B_OFFSET_LAYOUT_64X128_8W = tlx.layout(
    shape=((8, 8, 8), (8, 2)),
    stride=((1024, 16, 1), (128, 8)),
)
_wg_A_OFFSET_LAYOUT_64X64_8W = tlx.layout(
    shape=((8, 4, 8, 2), (8, )),
    stride=((8, 1024, 64, 512), (1, )),
)
_wg_B_OFFSET_LAYOUT_64X64_8W = tlx.layout(
    shape=((8, 4, 8, 2), (8, )),
    stride=((512, 16, 1, 8), (64, )),
)
_wg_A_OFFSET_LAYOUT_32X64_8W = tlx.layout(
    shape=((8, 32, 2), (8, )),
    stride=((8, 64, 0), (1, )),
)
_wg_B_OFFSET_LAYOUT_64X32_8W = tlx.layout(
    shape=((8, 32, 2), (8, )),
    stride=((256, 1, 0), (32, )),
)


@triton.jit
def _wg_wave_grid_global_load_a(
    a_ptr,
    pid_m,
    kb,
    row: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    m: tl.constexpr,
    force_stream: tl.constexpr = False,
):
    """Load one MI-height A row for a regular MI16 wave grid."""
    local_rows = tl.arange(0, a_group_m)
    if m * stride_am >= 1 << 30:
        # tlx.buffer_load requires 32-bit vector offsets. Put the large tile
        # coordinate in the 64-bit scalar base and keep only one fragment's
        # small lane offsets in the buffer instruction.
        row_base = pid_m * block_m + row * a_group_m
        safe_row_base = tl.minimum(row_base, m - 1)
        a_ptr += (safe_row_base.to(tl.int64) * stride_am + kb * block_k * stride_ak)
        rows = local_rows
        if m % block_m != 0:
            absolute_rows = row_base + rows
            rows = tl.where(
                absolute_rows < m,
                absolute_rows - safe_row_base,
                0,
            )
    else:
        rows = pid_m * block_m + row * a_group_m + local_rows
        if m % block_m != 0:
            rows = tl.where(rows < m, rows, 0)
        # Keep the lane offsets loop-invariant; only the scalar K base advances.
        a_ptr += kb * block_k * stride_ak
    reduction = tl.arange(0, block_k)
    offsets = rows[:, None] * stride_am + reduction[None, :] * stride_ak
    preferred_contiguity: tl.constexpr = (8 if a_group_m >= 32 else a_group_m // 8)
    cta_waves: tl.constexpr = ((a_group_m // 16) * (b_group_n // 16))
    elements_per_thread: tl.constexpr = (a_group_m * block_k // (64 * cta_waves))
    a_contiguity: tl.constexpr = (elements_per_thread
                                  if elements_per_thread < preferred_contiguity else preferred_contiguity)
    if (stride_am % 8 != 0 and block_k == 64 and a_group_m == 32 and (cta_waves == 4 or cta_waves == 8)):
        # Strided rows otherwise make layout inference collapse the K register
        # dimension. Pin the full-cover distribution while retaining the
        # normal vector transaction width; buffer loads support unaligned rows.
        offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_32X64_8W if cta_waves == 8 else _wg_A_OFFSET_LAYOUT_32X64_4W)
        offsets = tlx.require_layout(offsets, offset_layout)
        return tlx.buffer_load(a_ptr, offsets, contiguity=a_contiguity)
    # For tall-M/small-N grids, A advances to a new macro tile while B is
    # reused by many neighboring M programs.  Bypass L1 for the streaming A
    # operand so it does not evict that reusable B working set.
    if force_stream or (
            m >= 32768 and
        ((block_m == 128 and b_group_n == 32) or block_m == 160 or (block_m >= 224 and block_m != 320)
         # This exact huge-M/K1024 production plan streams 1.6 GiB of A
         # past a compact, highly reused row-major B.  Keeping A out of L1
         # improved the official 512 MiB cold-cache ratio from 0.948x to
         # 1.010x and 1.012x in independent 301-round runs.
         or (m == 819200 and block_m == 320 and block_k == 64 and b_group_n == 32 and stride_am == 1024
             and stride_ak == 1))):
        return tlx.buffer_load(a_ptr, offsets, cache=".cg", contiguity=a_contiguity)
    return tlx.buffer_load(a_ptr, offsets, contiguity=a_contiguity)


@triton.jit
def _wg_wave_grid_global_load_a_pair_k32(
    a_ptr,
    pid_m,
    kb,
    pair: tl.constexpr,
    a_group_m: tl.constexpr,
    block_m: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    m: tl.constexpr,
):
    """Load two adjacent K32 A groups as one 16-byte-per-lane transfer."""
    tl.static_assert(a_group_m == 32)
    pair_m: tl.constexpr = 2 * a_group_m
    local_rows = tl.arange(0, pair_m)
    rows = pid_m * block_m + pair * pair_m + local_rows
    if m % block_m != 0:
        rows = tl.where(rows < m, rows, 0)
    reduction = tl.arange(0, 32)
    offsets = rows[:, None] * stride_am + reduction[None, :] * stride_ak
    return tlx.buffer_load(
        a_ptr + kb * 32 * stride_ak,
        offsets,
        contiguity=8,
    )


@triton.jit
def _wg_wave_grid_global_loads_pgr2_k32_packed_a(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Load K32 PGR2 with vendor-width B groups followed by A row pairs."""
    tl.static_assert(a_row_count % 2 == 0)
    b_values = tl.tuple([
        _wg_wave_grid_global_load_b(
            b_ptr,
            pid_n,
            kb,
            group,
            block_n,
            32,
            b_group_n,
            stride_bk,
            stride_bn,
            n,
        ) for group in range(b_group_count)
    ])
    a_values = tl.tuple([])
    a_pair_count: tl.constexpr = (a_row_count + 1) // 2
    for pair in tl.static_range(a_pair_count):
        a_values += tl.tuple(
            [_wg_wave_grid_global_load_a_pair_k32(
                a_ptr,
                pid_m,
                kb,
                pair,
                32,
                block_m,
                stride_am,
                stride_ak,
                m,
            )])
    return a_values, b_values


@triton.jit
def _wg_wave_grid_global_load_b(
    b_ptr,
    pid_n,
    kb,
    group: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    n: tl.constexpr,
    cache_cg: tl.constexpr = True,
):
    """Load one wave-group-wide B column fragment."""
    reduction = tl.arange(0, block_k)
    local_cols = tl.arange(0, b_group_n)
    if n * stride_bn >= 1 << 30:
        col_base = pid_n * block_n + group * b_group_n
        safe_col_base = tl.minimum(col_base, n - 1)
        b_ptr += (safe_col_base.to(tl.int64) * stride_bn + kb * block_k * stride_bk)
        cols = local_cols
        if n % block_n != 0:
            absolute_cols = col_base + cols
            cols = tl.where(
                absolute_cols < n,
                absolute_cols - safe_col_base,
                0,
            )
    else:
        # Keep the lane offsets loop-invariant; only the scalar K base advances.
        b_ptr += kb * block_k * stride_bk
        cols = pid_n * block_n + group * b_group_n + local_cols
        if n % block_n != 0:
            cols = tl.where(cols < n, cols, 0)
    offsets = reduction[:, None] * stride_bk + cols[None, :] * stride_bn
    # Wide MT160/MT192 plans load one small A tile but stream a much larger B
    # working set.  Bypassing L1 for B matches the selected hipBLASLt policy
    # and avoids evicting reusable A lines.  Tall plans retain the default
    # policy because neighboring M tiles can reuse B.
    stream_b = block_n == 192
    if cache_cg and stream_b:
        return tlx.buffer_load(
            b_ptr,
            offsets,
            cache=".cg",
            contiguity=min(b_group_n // 8, 8),
        )
    return tlx.buffer_load(b_ptr, offsets, contiguity=min(b_group_n // 8, 8))


@triton.jit
def _wg_wave_grid_global_load_pgr2_b_half(
    b_ptr,
    pid_n,
    kb,
    group: tl.constexpr,
    half: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Load one 32-column machine fragment of a PGR2 B group."""
    tl.static_assert(b_group_n == 32 or b_group_n == 64)
    half_n: tl.constexpr = 32
    tl.static_assert(half < b_group_n // half_n)
    reduction = tl.arange(0, block_k)
    local_cols = tl.arange(0, half_n)
    if n * stride_bn >= 1 << 30:
        col_base = (pid_n * block_n + group * b_group_n + half * half_n)
        safe_col_base = tl.minimum(col_base, n - 1)
        b_ptr += (safe_col_base.to(tl.int64) * stride_bn + kb * block_k * stride_bk)
        cols = local_cols
        if n % block_n != 0:
            absolute_cols = col_base + cols
            cols = tl.where(
                absolute_cols < n,
                absolute_cols - safe_col_base,
                0,
            )
    else:
        b_ptr += kb * block_k * stride_bk
        cols = (pid_n * block_n + group * b_group_n + half * half_n + local_cols)
        if n % block_n != 0:
            cols = tl.where(cols < n, cols, 0)
    offsets = reduction[:, None] * stride_bk + cols[None, :] * stride_bn
    # Match the operand-level cache policy used by the regular global-load
    # path: only bypass L1 when B itself is the large streaming operand.
    # Tall-M/small-N grids reuse B across many neighboring M workgroups.
    # A single-M-tile grid never reuses B across workgroups.  Bypass L1 for
    # its streaming MT224 operand so the many neighboring N workgroups can
    # retain their shared A rows.  Multi-row grids keep B cacheable because
    # the same B tile is consumed by each M workgroup.
    stream_b: tl.constexpr = (block_n == 192 or (block_n == 224 and m <= block_m))
    if stream_b:
        return tlx.buffer_load(b_ptr, offsets, cache=".cg", contiguity=half_n // 8)
    return tlx.buffer_load(b_ptr, offsets, contiguity=half_n // 8)


@triton.jit
def _wg_wave_grid_global_loads(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    b_cache_cg: tl.constexpr = True,
):
    """Prefetch one complete K block into VGPR fragments.

    B groups are issued first, matching the vendor pipeline.  The tuple sizes
    are derived from the MI wave geometry instead of an exact problem shape.
    """
    b_values = tl.tuple([])
    for group in tl.static_range(b_group_count):
        b_values += tl.tuple([
            _wg_wave_grid_global_load_b(
                b_ptr,
                pid_n,
                kb,
                group,
                block_n,
                block_k,
                b_group_n,
                stride_bk,
                stride_bn,
                n,
                b_cache_cg,
            )
        ])
    a_values = tl.tuple([])
    a_group_m: tl.constexpr = block_m // a_row_count
    for row in tl.static_range(a_row_count):
        a_values += tl.tuple([
            _wg_wave_grid_global_load_a(
                a_ptr,
                pid_m,
                kb,
                row,
                a_group_m,
                b_group_n,
                block_m,
                block_k,
                stride_am,
                stride_ak,
                m,
            )
        ])
    return a_values, b_values


@triton.jit
def _wg_wave_grid_global_loads_pgr2(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Prefetch PGR2 in the vendor's machine-width B-then-A order."""
    tl.static_assert(b_group_n == 32 or b_group_n == 64)
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    b_values = tl.tuple([])
    for group in tl.static_range(b_group_count):
        for half in tl.static_range(b_fragments_per_group):
            b_values += tl.tuple([
                _wg_wave_grid_global_load_pgr2_b_half(
                    b_ptr,
                    pid_n,
                    kb,
                    group,
                    half,
                    block_m,
                    block_n,
                    block_k,
                    b_group_n,
                    stride_bk,
                    stride_bn,
                    m,
                    n,
                )
            ])
    a_values = tl.tuple([])
    a_group_m: tl.constexpr = block_m // a_row_count
    for row in tl.static_range(a_row_count):
        a_values += tl.tuple([
            _wg_wave_grid_global_load_a(
                a_ptr,
                pid_m,
                kb,
                row,
                a_group_m,
                b_group_n,
                block_m,
                block_k,
                stride_am,
                stride_ak,
                m,
            )
        ])
    return a_values, b_values


@triton.jit
def _wg_wave_grid_global_load_pgr2_a_tail(
    a_ptr,
    pid_m,
    kb,
    row: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    block_m: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    m: tl.constexpr,
    tail_k: tl.constexpr,
):
    """Load a zero-padded K64 tail with the regular PGR2 A layout."""
    local_rows = tl.arange(0, a_group_m)
    if m * stride_am >= 1 << 30:
        row_base = pid_m * block_m + row * a_group_m
        safe_row_base = tl.minimum(row_base, m - 1)
        a_ptr += (safe_row_base.to(tl.int64) * stride_am + kb * 64 * stride_ak)
        rows = local_rows
        if m % block_m != 0:
            absolute_rows = row_base + rows
            rows = tl.where(
                absolute_rows < m,
                absolute_rows - safe_row_base,
                0,
            )
    else:
        rows = pid_m * block_m + row * a_group_m + local_rows
        if m % block_m != 0:
            rows = tl.where(rows < m, rows, 0)
        a_ptr += kb * 64 * stride_ak
    reduction = tl.arange(0, 64)
    # Keep every lane active so the load retains the exact distributed layout
    # used by the proven full-K64 PGR2 path.  Invalid tail lanes reread a valid
    # element here and are zeroed after the LDS round trip, before MFMA.
    offsets = rows[:, None] * stride_am + reduction[None, :] * stride_ak
    safe_offsets = tl.where(
        reduction[None, :] < tail_k,
        offsets,
        rows[:, None] * stride_am,
    )
    preferred_contiguity: tl.constexpr = (8 if a_group_m >= 32 else a_group_m // 8)
    cta_waves: tl.constexpr = ((a_group_m // 16) * (b_group_n // 16))
    elements_per_thread: tl.constexpr = (a_group_m * 64 // (64 * cta_waves))
    a_contiguity: tl.constexpr = (elements_per_thread
                                  if elements_per_thread < preferred_contiguity else preferred_contiguity)
    tl.static_assert(a_group_m == 32 and (cta_waves == 4 or cta_waves == 8))
    offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_32X64_8W if cta_waves == 8 else _wg_A_OFFSET_LAYOUT_32X64_4W)
    safe_offsets = tlx.require_layout(safe_offsets, offset_layout)
    if m >= 32768 and ((block_m == 128 and b_group_n == 32) or block_m == 160 or (block_m >= 224 and block_m != 320)):
        return tlx.buffer_load(
            a_ptr,
            safe_offsets,
            cache=".cg",
            contiguity=a_contiguity,
        )
    return tlx.buffer_load(a_ptr, safe_offsets, contiguity=a_contiguity)


@triton.jit
def _wg_wave_grid_global_load_pgr2_b_tail_half(
    b_ptr,
    pid_n,
    kb,
    group: tl.constexpr,
    half: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    tail_k: tl.constexpr,
):
    """Load a zero-padded K64 tail with the regular PGR2 B layout."""
    tl.static_assert(b_group_n == 32 or b_group_n == 64)
    half_n: tl.constexpr = 32
    tl.static_assert(half < b_group_n // half_n)
    reduction = tl.arange(0, 64)
    local_cols = tl.arange(0, half_n)
    if n * stride_bn >= 1 << 30:
        col_base = (pid_n * block_n + group * b_group_n + half * half_n)
        safe_col_base = tl.minimum(col_base, n - 1)
        b_ptr += (safe_col_base.to(tl.int64) * stride_bn + kb * 64 * stride_bk)
        cols = local_cols
        if n % block_n != 0:
            absolute_cols = col_base + cols
            cols = tl.where(
                absolute_cols < n,
                absolute_cols - safe_col_base,
                0,
            )
    else:
        b_ptr += kb * 64 * stride_bk
        cols = (pid_n * block_n + group * b_group_n + half * half_n + local_cols)
        if n % block_n != 0:
            cols = tl.where(cols < n, cols, 0)
    offsets = reduction[:, None] * stride_bk + cols[None, :] * stride_bn
    safe_offsets = tl.where(
        reduction[:, None] < tail_k,
        offsets,
        cols[None, :] * stride_bn,
    )
    if stride_bk == 1:
        # The masked select otherwise loses B's K-contiguous distribution.
        # Restore the same physical 64x32 mapping used by the unmasked TN
        # load; row-major B intentionally keeps its inferred N-contiguous
        # mapping instead.
        a_group_m: tl.constexpr = block_m // a_row_count
        cta_waves: tl.constexpr = ((a_group_m // 16) * (b_group_n // 16))
        tl.static_assert(b_group_n == 32 and (cta_waves == 4 or cta_waves == 8))
        offset_layout: tl.constexpr = (_wg_B_OFFSET_LAYOUT_64X32_8W if cta_waves == 8 else _wg_B_OFFSET_LAYOUT_64X32_4W)
        safe_offsets = tlx.require_layout(safe_offsets, offset_layout)
    stream_b: tl.constexpr = (block_n == 192 or (block_n == 224 and m <= block_m))
    if stream_b:
        return tlx.buffer_load(
            b_ptr,
            safe_offsets,
            cache=".cg",
            contiguity=4,
        )
    return tlx.buffer_load(b_ptr, safe_offsets, contiguity=half_n // 8)


@triton.jit
def _wg_wave_grid_global_loads_pgr2_tail(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    tail_k: tl.constexpr,
):
    """Load one masked K64 PGR2 tail in the native B-then-A layout."""
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    b_values = tl.tuple([])
    for group in tl.static_range(b_group_count):
        for half in tl.static_range(b_fragments_per_group):
            b_values += tl.tuple([
                _wg_wave_grid_global_load_pgr2_b_tail_half(
                    b_ptr,
                    pid_n,
                    kb,
                    group,
                    half,
                    block_m,
                    block_n,
                    a_row_count,
                    b_group_n,
                    stride_bk,
                    stride_bn,
                    m,
                    n,
                    tail_k,
                )
            ])
    a_group_m: tl.constexpr = block_m // a_row_count
    a_values = tl.tuple([])
    for row in tl.static_range(a_row_count):
        a_values += tl.tuple([
            _wg_wave_grid_global_load_pgr2_a_tail(
                a_ptr,
                pid_m,
                kb,
                row,
                a_group_m,
                b_group_n,
                block_m,
                stride_am,
                stride_ak,
                m,
                tail_k,
            )
        ])
    return a_values, b_values


@triton.jit
def _wg_wave_grid_local_store_b(
    b_local,
    stage: tl.constexpr,
    group: tl.constexpr,
    value,
    b_group_count: tl.constexpr,
    block_k: tl.constexpr,
    b_group_n: tl.constexpr,
):
    tl.static_assert(block_k == 32 or block_k == 64)
    tl.static_assert(b_group_n == 16 or b_group_n == 32 or b_group_n == 64 or b_group_n == 128)
    tlx.local_store(tlx.local_view(b_local[group], stage), value)


@triton.jit
def _wg_wave_grid_local_store_all(
    a_local,
    b_local,
    stage,
    a_values,
    b_values,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    block_k: tl.constexpr,
    b_group_n: tl.constexpr,
):
    for group in tl.static_range(b_group_count):
        _wg_wave_grid_local_store_b(
            b_local,
            stage,
            group,
            b_values[group],
            b_group_count,
            block_k,
            b_group_n,
        )
    for row in tl.static_range(a_row_count):
        tlx.local_store(tlx.local_view(a_local[row], stage), a_values[row])


@triton.jit
def _wg_wave_grid_local_store_fragment(
    a_local,
    b_local,
    stage,
    prefetched_a,
    prefetched_b,
    fragment: tl.constexpr,
    b_group_count: tl.constexpr,
    block_k: tl.constexpr,
    b_group_n: tl.constexpr,
):
    """Publish one fragment in vendor order: all B, followed by all A."""
    if fragment < b_group_count:
        _wg_wave_grid_local_store_b(
            b_local,
            stage,
            fragment,
            prefetched_b[fragment],
            b_group_count,
            block_k,
            b_group_n,
        )
    else:
        a_row: tl.constexpr = fragment - b_group_count
        tlx.local_store(tlx.local_view(a_local[a_row], stage), prefetched_a[a_row])


@triton.jit
def _wg_wave_grid_local_store_pgr2_fragment(
    a_local,
    b_local,
    prefetched_a,
    prefetched_b,
    fragment: tl.constexpr,
    b_group_count: tl.constexpr,
    a_row_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    block_k: tl.constexpr,
    stage=0,
    prefetched_a_offset=0,
    prefetched_b_offset=0,
):
    """Publish one machine-width fragment into consolidated PGR2 LDS."""
    tl.static_assert(b_group_n == 16 or b_group_n == 32 or b_group_n == 64)
    b_fragments_per_group: tl.constexpr = (1 if b_group_n == 16 else b_group_n // 32)
    b_fragment_n: tl.constexpr = 16 if b_group_n == 16 else 32
    b_fragment_count: tl.constexpr = (b_fragments_per_group * b_group_count)
    if fragment < b_fragment_count:
        b_group: tl.constexpr = fragment // b_fragments_per_group
        b_half: tl.constexpr = fragment % b_fragments_per_group
        separate_b_stages: tl.constexpr = (len(b_local) == 2 * b_group_count)
        exact_b_groups: tl.constexpr = (len(b_local) == b_group_count or separate_b_stages)
        b_groups_per_chunk: tl.constexpr = 256 // b_group_n
        b_chunk: tl.constexpr = (stage * b_group_count +
                                 b_group if separate_b_stages else b_group if exact_b_groups else b_group //
                                 b_groups_per_chunk)
        b_local_group: tl.constexpr = (0 if exact_b_groups else b_group % b_groups_per_chunk)
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(b_local[b_chunk], 0 if separate_b_stages else stage),
                [0, b_local_group * b_group_n + b_half * 32],
                [block_k, b_fragment_n],
            ),
            prefetched_b[prefetched_b_offset + fragment],
        )
    else:
        row: tl.constexpr = fragment - b_fragment_count
        separate_a_stages: tl.constexpr = (len(a_local) == 2 * a_row_count)
        exact_a_groups: tl.constexpr = (len(a_local) == a_row_count or separate_a_stages)
        a_groups_per_chunk: tl.constexpr = 256 // a_group_m
        a_chunk: tl.constexpr = (stage * a_row_count + row if separate_a_stages else row if exact_a_groups else row //
                                 a_groups_per_chunk)
        a_local_row: tl.constexpr = (0 if exact_a_groups else row % a_groups_per_chunk)
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(a_local[a_chunk], 0 if separate_a_stages else stage),
                [a_local_row * a_group_m, 0],
                [a_group_m, block_k],
            ),
            prefetched_a[prefetched_a_offset + row],
        )


@triton.jit
def _wg_wave_grid_local_store_pgr2_k32_packed_fragment(
    a_local,
    b_local,
    prefetched_a,
    prefetched_b,
    fragment: tl.constexpr,
    b_group_count: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_n: tl.constexpr,
):
    """Publish one K32 B group or one packed pair of A row groups."""
    tl.static_assert(a_row_count % 2 == 0)
    if fragment < b_group_count:
        _wg_wave_grid_local_store_pgr2_fragment(
            a_local,
            b_local,
            tl.tuple([]),
            prefetched_b,
            fragment,
            b_group_count,
            a_row_count,
            32,
            b_group_n,
            32,
            0,
        )
    else:
        pair: tl.constexpr = fragment - b_group_count
        a_pair_count: tl.constexpr = (a_row_count + 1) // 2
        tl.static_assert(pair < a_pair_count)
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(a_local[0], 0),
                [pair * 64, 0],
                [64, 32],
            ),
            prefetched_a[pair],
        )


@tl.core.builtin
def _wg_wave_grid_fragment_start(
    row: tl.constexpr,
    b_group_count: tl.constexpr,
    _semantic=None,
):
    row = tl.core._unwrap_if_constexpr(row)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    return (2 * row if row < b_group_count else row + b_group_count)


@triton.jit
def _wg_wave_grid_local_load_a(
    a_local,
    stage,
    row: tl.constexpr,
    a_row_count: tl.constexpr,
    dot_a: tl.constexpr,
):
    return tlx.local_load(
        tlx.local_view(a_local[row], stage),
        layout=dot_a,
        relaxed=True,
    )


@triton.jit
def _wg_wave_grid_local_load_b(
    b_local,
    stage,
    dot_b: tl.constexpr,
    b_group_count: tl.constexpr,
):
    return tl.tuple([
        tlx.local_load(
            tlx.local_view(b_local[group], stage),
            layout=dot_b,
            relaxed=True,
        ) for group in range(b_group_count)
    ])


@triton.jit
def _wg_wave_grid_local_load_a_half(
    a_local,
    stage,
    row: tl.constexpr,
    half: tl.constexpr,
    a_group_m: tl.constexpr,
    dot_a: tl.constexpr,
):
    view = tlx.local_slice(
        tlx.local_view(a_local[row], stage),
        [0, half * 32],
        [a_group_m, 32],
    )
    return tlx.local_load(view, layout=dot_a, relaxed=True)


@triton.jit
def _wg_wave_grid_local_load_b_half(
    b_local,
    stage,
    group: tl.constexpr,
    half: tl.constexpr,
    b_group_n: tl.constexpr,
    dot_b: tl.constexpr,
):
    view = tlx.local_slice(
        tlx.local_view(b_local[group], stage),
        [half * 32, 0],
        [32, b_group_n],
    )
    return tlx.local_load(view, layout=dot_b, relaxed=True)


@triton.jit
def _wg_wave_grid_local_load_pgr2_a_half(
    a_local,
    row: tl.constexpr,
    half: tl.constexpr,
    a_group_m: tl.constexpr,
    a_row_count: tl.constexpr,
    dot_a: tl.constexpr,
    stage=0,
):
    separate_stages: tl.constexpr = len(a_local) == 2 * a_row_count
    exact_groups: tl.constexpr = (len(a_local) == a_row_count or separate_stages)
    groups_per_chunk: tl.constexpr = 256 // a_group_m
    chunk: tl.constexpr = (stage * a_row_count + row if separate_stages else row if exact_groups else row //
                           groups_per_chunk)
    local_row: tl.constexpr = 0 if exact_groups else row % groups_per_chunk
    view = tlx.local_slice(
        tlx.local_view(a_local[chunk], 0 if separate_stages else stage),
        [local_row * a_group_m, half * 32],
        [a_group_m, 32],
    )
    return tlx.local_load(view, layout=dot_a, relaxed=True)


@triton.jit
def _wg_wave_grid_local_load_pgr2_b_half(
    b_local,
    group: tl.constexpr,
    half: tl.constexpr,
    b_group_n: tl.constexpr,
    b_group_count: tl.constexpr,
    dot_b: tl.constexpr,
    stage=0,
):
    separate_stages: tl.constexpr = len(b_local) == 2 * b_group_count
    exact_groups: tl.constexpr = (len(b_local) == b_group_count or separate_stages)
    groups_per_chunk: tl.constexpr = 256 // b_group_n
    chunk: tl.constexpr = (stage * b_group_count + group if separate_stages else group if exact_groups else group //
                           groups_per_chunk)
    local_group: tl.constexpr = (0 if exact_groups else group % groups_per_chunk)
    view = tlx.local_slice(
        tlx.local_view(b_local[chunk], 0 if separate_stages else stage),
        [half * 32, local_group * b_group_n],
        [32, b_group_n],
    )
    return tlx.local_load(view, layout=dot_b, relaxed=True)


@tl.core.builtin
def _wg_wave_grid_dot_row(
    a_operand,
    b_operands,
    accumulators,
    row: tl.constexpr,
    first_column: tl.constexpr,
    column_count: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Update one MI-height row across the complete N wave grid."""
    row = tl.core._unwrap_if_constexpr(row)
    first_column = tl.core._unwrap_if_constexpr(first_column)
    column_count = tl.core._unwrap_if_constexpr(column_count)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    values = list(accumulators)
    b_operands = list(b_operands)
    for column in range(first_column, first_column + column_count):
        index = row * b_group_count + column
        values[index] = tlx.amd_scheduled_mfma(
            tlx.require_layout(a_operand, dot_a, pin=False, _semantic=_semantic),
            tlx.require_layout(b_operands[column], dot_b, pin=False, _semantic=_semantic),
            tlx.require_layout(values[index], mma, pin=False, _semantic=_semantic),
            accumulator_role="persistent",
            resident_operand=None,
            initialize=initialize,
            _semantic=_semantic,
        )
    return tl.tuple(values)


@tl.core.builtin
def _wg_wave_grid_dot_column_major_at(
    a_operands,
    b_operands,
    accumulators,
    mfma_index: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Issue one MFMA from a column-major regular operand grid."""
    mfma_index = tl.core._unwrap_if_constexpr(mfma_index)
    a_row_count = tl.core._unwrap_if_constexpr(a_row_count)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    row = mfma_index // b_group_count
    column = mfma_index % b_group_count
    return _wg_wave_grid_dot_row(
        a_operands[row],
        b_operands,
        accumulators,
        row,
        column,
        1,
        b_group_count,
        mma,
        dot_a,
        dot_b,
        initialize,
        _semantic=_semantic,
    )


@tl.core.builtin
def _wg_wave_grid_dot_column_with_preloaded_a(
    a_operands,
    b_operand,
    accumulators,
    column: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Consume one B fragment immediately across a preloaded A column."""
    column = tl.core._unwrap_if_constexpr(column)
    a_row_count = tl.core._unwrap_if_constexpr(a_row_count)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    a_operands = list(a_operands)
    values = list(accumulators)
    for row in range(a_row_count):
        index = row * b_group_count + column
        values[index] = tlx.amd_scheduled_mfma(
            tlx.require_layout(a_operands[row], dot_a, pin=False, _semantic=_semantic),
            tlx.require_layout(b_operand, dot_b, pin=False, _semantic=_semantic),
            tlx.require_layout(values[index], mma, pin=False, _semantic=_semantic),
            accumulator_role="persistent",
            resident_operand=None,
            initialize=initialize,
            _semantic=_semantic,
        )
    return tl.tuple(values)


@tl.core.builtin
def _wg_wave_grid_dot_at(
    a_operand,
    b_operand,
    accumulators,
    row: tl.constexpr,
    column: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Issue one regular-grid MFMA from already selected operands."""
    row = tl.core._unwrap_if_constexpr(row)
    column = tl.core._unwrap_if_constexpr(column)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    values = list(accumulators)
    index = row * b_group_count + column
    values[index] = tlx.amd_scheduled_mfma(
        tlx.require_layout(a_operand, dot_a, pin=False, _semantic=_semantic),
        tlx.require_layout(b_operand, dot_b, pin=False, _semantic=_semantic),
        tlx.require_layout(values[index], mma, pin=False, _semantic=_semantic),
        accumulator_role="persistent",
        resident_operand=None,
        initialize=initialize,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@tl.core.builtin
def _wg_wave_grid_dot_column_major_range(
    a_operands,
    b_operands,
    accumulators,
    first_mfma: tl.constexpr,
    mfma_count: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Issue one compile-time consecutive range of a column-major grid."""
    first_mfma = tl.core._unwrap_if_constexpr(first_mfma)
    mfma_count = tl.core._unwrap_if_constexpr(mfma_count)
    values = accumulators
    for offset in range(mfma_count):
        values = _wg_wave_grid_dot_column_major_at(
            a_operands,
            b_operands,
            values,
            first_mfma + offset,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize,
            _semantic=_semantic,
        )
    return values


@tl.core.builtin
def _wg_wave_grid_dot_two_half_range(
    current_a,
    current_b,
    second_a,
    second_b,
    accumulators,
    first_mfma: tl.constexpr,
    mfma_count: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr = False,
    _semantic=None,
):
    """Issue a linear MFMA interval spanning two consecutive K32 halves."""
    first_mfma = tl.core._unwrap_if_constexpr(first_mfma)
    mfma_count = tl.core._unwrap_if_constexpr(mfma_count)
    a_row_count = tl.core._unwrap_if_constexpr(a_row_count)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    mfma_per_half = a_row_count * b_group_count
    values = accumulators
    for offset in range(mfma_count):
        linear_mfma = first_mfma + offset
        if linear_mfma < mfma_per_half:
            values = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                values,
                linear_mfma,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
                _semantic=_semantic,
            )
        else:
            values = _wg_wave_grid_dot_column_major_at(
                second_a,
                second_b,
                values,
                linear_mfma - mfma_per_half,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                _semantic=_semantic,
            )
    return values


@triton.jit
def _wg_wave_grid_pipeline_step_k32(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    current_stage: tl.constexpr,
    next_stage: tl.constexpr,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Advance one K32 stage with compiler-distributed memory cover.

    Source preserves one bounded scheduling window per MFMA row.  Within the
    window the memory stream publishes and refills the next LDS stage while
    the independent compute stream consumes the current row.  The scheduler
    derives the exact store/load-to-MFMA cover from the lowered instruction
    counts, so the same source handles two- and four-column wave grids.
    """
    tl.static_assert(b_group_count == 2 or b_group_count == 4)
    tl.static_assert(a_row_count >= 2)
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    for row in tl.static_range(a_row_count):
        a_operand = current_a[0]
        with tlx.warp_pipeline_stage("publish_prefetch_and_read", scope="intra_wave", pair=0):
            _wg_wave_grid_local_store_fragment(
                a_local,
                b_local,
                next_stage,
                prefetched_a,
                prefetched_b,
                _wg_wave_grid_fragment_start(row, b_group_count),
                b_group_count,
                block_k,
                b_group_n,
            )
            if _wg_wave_grid_fragment_start(row, b_group_count) < b_group_count:
                future_b += tl.tuple([
                    _wg_wave_grid_global_load_b(
                        b_ptr,
                        pid_n,
                        future_kb,
                        _wg_wave_grid_fragment_start(row, b_group_count),
                        block_n,
                        block_k,
                        b_group_n,
                        stride_bk,
                        stride_bn,
                        n,
                    )
                ])
            else:
                future_a += tl.tuple([
                    _wg_wave_grid_global_load_a(
                        a_ptr,
                        pid_m,
                        future_kb,
                        _wg_wave_grid_fragment_start(row, b_group_count) - b_group_count,
                        block_m // a_row_count,
                        b_group_n,
                        block_m,
                        block_k,
                        stride_am,
                        stride_ak,
                        m,
                    )
                ])

            if row < b_group_count:
                _wg_wave_grid_local_store_fragment(
                    a_local,
                    b_local,
                    next_stage,
                    prefetched_a,
                    prefetched_b,
                    _wg_wave_grid_fragment_start(row, b_group_count) + 1,
                    b_group_count,
                    block_k,
                    b_group_n,
                )
                if (_wg_wave_grid_fragment_start(row, b_group_count) + 1 < b_group_count):
                    future_b += tl.tuple([
                        _wg_wave_grid_global_load_b(
                            b_ptr,
                            pid_n,
                            future_kb,
                            _wg_wave_grid_fragment_start(row, b_group_count) + 1,
                            block_n,
                            block_k,
                            b_group_n,
                            stride_bk,
                            stride_bn,
                            n,
                        )
                    ])
                else:
                    future_a += tl.tuple([
                        _wg_wave_grid_global_load_a(
                            a_ptr,
                            pid_m,
                            future_kb,
                            _wg_wave_grid_fragment_start(row, b_group_count) + 1 - b_group_count,
                            block_m // a_row_count,
                            b_group_n,
                            block_m,
                            block_k,
                            stride_am,
                            stride_ak,
                            m,
                        )
                    ])

            # If M has fewer rows than B has groups, two publications per row
            # leave a short A tail. Keep it inside the final bounded window.
            if row == a_row_count - 1 and a_row_count < b_group_count:
                for tail in tl.static_range(
                        2 * a_row_count,
                        a_row_count + b_group_count,
                ):
                    _wg_wave_grid_local_store_fragment(
                        a_local,
                        b_local,
                        next_stage,
                        prefetched_a,
                        prefetched_b,
                        tail,
                        b_group_count,
                        block_k,
                        b_group_n,
                    )
                    future_a += tl.tuple([
                        _wg_wave_grid_global_load_a(
                            a_ptr,
                            pid_m,
                            future_kb,
                            tail - b_group_count,
                            block_m // a_row_count,
                            b_group_n,
                            block_m,
                            block_k,
                            stride_am,
                            stride_ak,
                            m,
                        )
                    ])

            if row + 2 < a_row_count:
                following_a = _wg_wave_grid_local_load_a(
                    a_local,
                    current_stage,
                    row + 2,
                    a_row_count,
                    dot_a,
                )
        with tlx.warp_pipeline_stage("consume_current_row", scope="intra_wave", pair=0):
            accumulators = _wg_wave_grid_dot_row(
                a_operand,
                current_b,
                accumulators,
                row,
                0,
                b_group_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
        if row + 2 < a_row_count:
            current_a = tl.tuple([current_a[1], following_a])
        elif row + 1 < a_row_count:
            current_a = tl.tuple([current_a[1]])

    tl.debug_barrier()
    current_b = _wg_wave_grid_local_load_b(b_local, next_stage, dot_b, b_group_count)
    current_a = tl.tuple([_wg_wave_grid_local_load_a(a_local, next_stage, row, a_row_count, dot_a) for row in range(2)])
    return accumulators, future_a, future_b, current_a, current_b


@triton.jit
def _wg_wave_grid_pipeline_pair(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    for half in tl.static_range(2):
        state = _wg_wave_grid_pipeline_step_k32(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2 + half,
            half,
            1 - half,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = state
    return accumulators, prefetched_a, prefetched_b, current_a, current_b


@triton.jit
def _wg_wave_grid_compute_stage(
    a_local,
    b_local,
    stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
):
    b_operands = _wg_wave_grid_local_load_b(b_local, stage, dot_b, b_group_count)
    for row in tl.static_range(a_row_count):
        a_operand = _wg_wave_grid_local_load_a(a_local, stage, row, a_row_count, dot_a)
        accumulators = _wg_wave_grid_dot_row(
            a_operand,
            b_operands,
            accumulators,
            row,
            0,
            b_group_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators


@triton.jit
def _wg_wave_grid_compute_stage_k64(
    a_local,
    b_local,
    stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume both K32 halves of one regular K64 operand tile."""
    for half in tl.static_range(2):
        b_operands = tl.tuple([
            _wg_wave_grid_local_load_b_half(b_local, stage, group, half, b_group_n, dot_b)
            for group in range(b_group_count)
        ])
        for row in tl.static_range(a_row_count):
            a_operand = _wg_wave_grid_local_load_a_half(a_local, stage, row, half, a_group_m, dot_a)
            accumulators = _wg_wave_grid_dot_row(
                a_operand,
                b_operands,
                accumulators,
                row,
                0,
                b_group_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
    return accumulators


@triton.jit
def _wg_grouped_square_k64_local_load_a_half(
    a_local,
    stage,
    row: tl.constexpr,
    half: tl.constexpr,
    fragment_count: tl.constexpr,
    dot_a: tl.constexpr,
):
    tl.static_assert(fragment_count >= 5 and fragment_count <= 8)
    if len(a_local) == fragment_count:
        buffer = a_local[row]
        local_row: tl.constexpr = 0
    elif row < 4:
        buffer = a_local[0]
        local_row: tl.constexpr = row * 32
    elif fragment_count == 7 and row >= 6:
        buffer = a_local[2]
        local_row: tl.constexpr = (row - 6) * 32
    else:
        buffer = a_local[1]
        local_row: tl.constexpr = (row - 4) * 32
    return tlx.local_load(
        tlx.local_slice(
            tlx.local_view(buffer, stage),
            [local_row, half * 32],
            [32, 32],
        ),
        layout=dot_a,
    )


@triton.jit
def _wg_grouped_square_k64_local_load_b_half(
    b_local,
    stage,
    group: tl.constexpr,
    half: tl.constexpr,
    fragment_count: tl.constexpr,
    dot_b: tl.constexpr,
):
    tl.static_assert(fragment_count >= 5 and fragment_count <= 8)
    if len(b_local) == fragment_count:
        buffer = b_local[group]
        local_column: tl.constexpr = 0
    elif group < 4:
        buffer = b_local[0]
        local_column: tl.constexpr = group * 32
    elif fragment_count == 7 and group >= 6:
        buffer = b_local[2]
        local_column: tl.constexpr = (group - 6) * 32
    else:
        buffer = b_local[1]
        local_column: tl.constexpr = (group - 4) * 32
    return tlx.local_load(
        tlx.local_slice(
            tlx.local_view(buffer, stage),
            [half * 32, local_column],
            [32, 32],
        ),
        layout=dot_b,
    )


@triton.jit
def _wg_grouped_square_k64_local_store_fragment(
    a_local,
    b_local,
    stage,
    prefetched_a,
    prefetched_b,
    fragment: tl.constexpr,
    fragment_count: tl.constexpr,
):
    tl.static_assert(fragment_count >= 5 and fragment_count <= 8)
    if fragment < fragment_count:
        if fragment < 4:
            buffer = b_local[0]
            local_column: tl.constexpr = fragment * 32
        elif fragment_count == 7 and fragment >= 6:
            buffer = b_local[2]
            local_column: tl.constexpr = (fragment - 6) * 32
        else:
            buffer = b_local[1]
            local_column: tl.constexpr = (fragment - 4) * 32
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(buffer, stage),
                [0, local_column],
                [64, 32],
            ),
            prefetched_b[fragment],
        )
    else:
        row: tl.constexpr = fragment - fragment_count
        if row < 4:
            buffer = a_local[0]
            local_row: tl.constexpr = row * 32
        elif fragment_count == 7 and row >= 6:
            buffer = a_local[2]
            local_row: tl.constexpr = (row - 6) * 32
        else:
            buffer = a_local[1]
            local_row: tl.constexpr = (row - 4) * 32
        tlx.local_store(
            tlx.local_slice(
                tlx.local_view(buffer, stage),
                [local_row, 0],
                [32, 64],
            ),
            prefetched_a[row],
        )


@triton.jit
def _wg_grouped_square_k64_local_store_all(
    a_local,
    b_local,
    stage: tl.constexpr,
    a_values,
    b_values,
    fragment_count: tl.constexpr,
):
    for fragment in tl.static_range(2 * fragment_count):
        _wg_grouped_square_k64_local_store_fragment(
            a_local,
            b_local,
            stage,
            a_values,
            b_values,
            fragment,
            fragment_count,
        )


@triton.jit
def _wg_grouped_square_k64_compute_stage(
    a_local,
    b_local,
    stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    fragment_count: tl.constexpr,
):
    for half in tl.static_range(2):
        b_operands = tl.tuple([
            _wg_grouped_square_k64_local_load_b_half(b_local, stage, group, half, fragment_count, dot_b)
            for group in range(fragment_count)
        ])
        for row in tl.static_range(fragment_count):
            a_operand = _wg_grouped_square_k64_local_load_a_half(a_local, stage, row, half, fragment_count, dot_a)
            accumulators = _wg_wave_grid_dot_row(
                a_operand,
                b_operands,
                accumulators,
                row,
                0,
                fragment_count,
                fragment_count,
                mma,
                dot_a,
                dot_b,
            )
    return accumulators


@triton.jit
def _wg_logical_rect_k64_direct_pgr2_step(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_stage,
    next_stage,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    direct_chunk: tl.constexpr,
    warps_per_cta: tl.constexpr,
    refill_read_window: tl.constexpr = 0,
    read_cover_sixteenths: tl.constexpr = 8,
    initialize: tl.constexpr = False,
):
    """Advance one K64 for an arbitrary regular MI16 grid using PGR2."""
    mfma_count: tl.constexpr = a_row_count * b_group_count
    operand_count: tl.constexpr = a_row_count + b_group_count
    tl.static_assert(read_cover_sixteenths >= 1 and read_cover_sixteenths <= 15)
    first_prefix: tl.constexpr = (mfma_count * read_cover_sixteenths // 16)
    second_suffix: tl.constexpr = mfma_count - first_prefix
    tl.static_assert(mfma_count % 2 == 0)

    # Spend half of K(t).kh0 on the K(t).kh1 LDS reads.  Once every read has
    # issued, all live data from current_stage is in operand registers and the
    # stage can safely be reused for K(t+2).
    a_is_smaller: tl.constexpr = a_row_count < b_group_count
    with tlx.warp_pipeline_stage("direct_pgr2_read_and_compute", scope="intra_wave"):
        second_a, second_b = _wg_logical_rect_k64_local_load_operand_half(
            a_local,
            b_local,
            current_stage,
            1,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            first_prefix,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize=initialize,
        )

    a_direct_load_count: tl.constexpr = ((block_m + direct_chunk - 1) //
                                         direct_chunk if len(a_local) == 1 else len(a_local))
    b_direct_load_count: tl.constexpr = ((block_n + direct_chunk - 1) //
                                         direct_chunk if len(b_local) == 1 else len(b_local))
    direct_load_count: tl.constexpr = (a_direct_load_count + b_direct_load_count)
    direct_group_count: tl.constexpr = (((a_direct_load_count + 3) // 4 +
                                         (b_direct_load_count + 3) // 4) if direct_chunk == 32 and len(a_local) == 1
                                        and len(b_local) == 1 else direct_load_count)
    # A positive refill/read window selects the merged steady-state schedule
    # below. Retire K(t+1) before issuing K(t+2), then use one
    # barrier both to publish K(t+1) and to prove that current_stage is no
    # longer being read.  This permits independent K(t+2) direct loads and
    # K(t+1) LDS reads to share one MFMA cover window.
    merged_refill_read: tl.constexpr = refill_read_window > 0
    if merged_refill_read:
        tlx.async_load_wait_group(0)

    # Ensure every wave has finished reading current_stage before direct loads
    # overwrite it.  Each refill is then covered by the remaining K(t).kh0
    # MFMAs plus the first half of K(t).kh1.
    tl.debug_barrier()

    if merged_refill_read:
        # Distribute arbitrary refill and read counts over the requested
        # source windows. The compiler then scales each anchor's target issue
        # cost across the complete MFMA budget.
        next_a = tl.tuple([])
        next_b = tl.tuple([])
        reads_per_group: tl.constexpr = refill_read_window
        groups: tl.constexpr = (operand_count + reads_per_group - 1) // reads_per_group
        remaining_mfmas: tl.constexpr = (2 * mfma_count - first_prefix)
        with tlx.warp_pipeline_stage("direct_pgr2_merged_memory", scope="intra_wave", pair=0):
            for group in tl.static_range(groups):
                for load in tl.static_range(
                        group * direct_load_count // groups,
                    (group + 1) * direct_load_count // groups,
                ):
                    _wg_logical_rect_k64_direct_load_chunk(
                        a_ptr,
                        b_ptr,
                        pid_m,
                        pid_n,
                        future_kb,
                        a_local,
                        b_local,
                        current_stage,
                        load,
                        direct_chunk,
                        block_m,
                        block_n,
                        stride_am,
                        stride_ak,
                        stride_bk,
                        stride_bn,
                        m,
                        n,
                        warps_per_cta,
                    )
                for read in tl.static_range(
                        group * operand_count // groups,
                    (group + 1) * operand_count // groups,
                ):
                    if a_is_smaller:
                        if read < a_row_count:
                            value = _wg_logical_rect_k64_local_load_a_half(
                                a_local,
                                next_stage,
                                read,
                                0,
                                a_group_m,
                                dot_a,
                            )
                            next_a += tl.tuple([value])
                        else:
                            value = _wg_logical_rect_k64_local_load_b_half(
                                b_local,
                                next_stage,
                                read - a_row_count,
                                0,
                                b_group_n,
                                dot_b,
                            )
                            next_b += tl.tuple([value])
                    else:
                        if read < b_group_count:
                            value = _wg_logical_rect_k64_local_load_b_half(
                                b_local,
                                next_stage,
                                read,
                                0,
                                b_group_n,
                                dot_b,
                            )
                            next_b += tl.tuple([value])
                        else:
                            value = _wg_logical_rect_k64_local_load_a_half(
                                a_local,
                                next_stage,
                                read - b_group_count,
                                0,
                                a_group_m,
                                dot_a,
                            )
                            next_a += tl.tuple([value])
        with tlx.warp_pipeline_stage("direct_pgr2_merged_compute", scope="intra_wave", pair=0):
            accumulators = _wg_wave_grid_dot_two_half_range(
                current_a,
                current_b,
                second_a,
                second_b,
                accumulators,
                first_prefix,
                remaining_mfmas,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize=initialize,
            )
        return accumulators, next_a, next_b

    with tlx.warp_pipeline_stage("direct_pgr2_refill", scope="intra_wave", pair=1):
        for load in tl.static_range(direct_load_count):
            _wg_logical_rect_k64_direct_load_chunk(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                future_kb,
                a_local,
                b_local,
                current_stage,
                load,
                direct_chunk,
                block_m,
                block_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                warps_per_cta,
            )
    with tlx.warp_pipeline_stage("direct_pgr2_compute_refill", scope="intra_wave", pair=1):
        accumulators = _wg_wave_grid_dot_two_half_range(
            current_a,
            current_b,
            second_a,
            second_b,
            accumulators,
            first_prefix,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize=initialize,
        )
    # K(t+1) was the older set of async groups. Retire those groups but
    # leave K(t+2) in flight, then read K(t+1).kh0 while the remaining
    # K(t).kh1 MFMAs execute.
    tlx.async_load_wait_group(direct_group_count)
    tl.debug_barrier()
    with tlx.warp_pipeline_stage("direct_pgr2_read_next", scope="intra_wave", pair=2):
        next_a, next_b = _wg_logical_rect_k64_local_load_operand_half(
            a_local,
            b_local,
            next_stage,
            0,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
    with tlx.warp_pipeline_stage("direct_pgr2_compute_next", scope="intra_wave", pair=2):
        accumulators = _wg_wave_grid_dot_column_major_range(
            second_a,
            second_b,
            accumulators,
            first_prefix,
            second_suffix,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators, next_a, next_b


@triton.jit
def _wg_logical_rect_k64_direct_pgr2_finish(
    a_local,
    b_local,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    current_stage: tl.constexpr,
    final_stage: tl.constexpr,
):
    """Consume the final two resident K64 stages without redundant reads."""
    mfma_count: tl.constexpr = a_row_count * b_group_count

    with tlx.warp_pipeline_stage("direct_pgr2_finish_current", scope="intra_wave"):
        second_a, second_b = _wg_logical_rect_k64_local_load_operand_half(
            a_local,
            b_local,
            current_stage,
            1,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    with tlx.warp_pipeline_stage("direct_pgr2_finish_second", scope="intra_wave"):
        final_a, final_b = _wg_logical_rect_k64_local_load_operand_half(
            a_local,
            b_local,
            final_stage,
            0,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            second_a,
            second_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    with tlx.warp_pipeline_stage("direct_pgr2_finish_final", scope="intra_wave"):
        final_second_a, final_second_b = (_wg_logical_rect_k64_local_load_operand_half(
            a_local,
            b_local,
            final_stage,
            1,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        ))
        accumulators = _wg_wave_grid_dot_column_major_range(
            final_a,
            final_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    return _wg_wave_grid_dot_column_major_range(
        final_second_a,
        final_second_b,
        accumulators,
        0,
        mfma_count,
        a_row_count,
        b_group_count,
        mma,
        dot_a,
        dot_b,
    )


@triton.jit
def _wg_logical_rect_k64_compute_tile_direct_pgr2(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    direct_chunk: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    warps_per_cta: tl.constexpr,
    refill_read_window: tl.constexpr = 0,
    read_cover_sixteenths: tl.constexpr = 8,
    loop_unroll_factor: tl.constexpr = 1,
):
    """Compute a regular MI16 grid with a two-stage direct PGR2 pipeline."""
    k_blocks: tl.constexpr = k // 64
    tl.static_assert(k_blocks == 3 or k_blocks % 2 == 0)
    _wg_logical_rect_k64_direct_load_stage(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        a_local,
        b_local,
        0,
        direct_chunk,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        warps_per_cta,
    )
    _wg_logical_rect_k64_direct_load_stage(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        a_local,
        b_local,
        1,
        direct_chunk,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        warps_per_cta,
    )
    a_direct_load_count: tl.constexpr = ((block_m + direct_chunk - 1) //
                                         direct_chunk if len(a_local) == 1 else len(a_local))
    b_direct_load_count: tl.constexpr = ((block_n + direct_chunk - 1) //
                                         direct_chunk if len(b_local) == 1 else len(b_local))
    direct_load_count: tl.constexpr = (a_direct_load_count + b_direct_load_count)
    direct_group_count: tl.constexpr = (((a_direct_load_count + 3) // 4 +
                                         (b_direct_load_count + 3) // 4) if direct_chunk == 32 and len(a_local) == 1
                                        and len(b_local) == 1 else direct_load_count)
    tlx.async_load_wait_group(direct_group_count)
    tl.debug_barrier()
    # Keep the longer-lived complete operand set on the smaller side of a
    # rectangular dot grid.  The other side is read later and consumed sooner.
    current_a, current_b = _wg_logical_rect_k64_local_load_operand_half(
        a_local,
        b_local,
        0,
        0,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        dot_a,
        dot_b,
    )
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])
    accumulators, current_a, current_b = (_wg_logical_rect_k64_direct_pgr2_step(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2,
        a_local,
        b_local,
        0,
        1,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        direct_chunk,
        warps_per_cta,
        refill_read_window,
        read_cover_sixteenths,
        True,
    ))
    if k_blocks == 3:
        return _wg_logical_rect_k64_direct_pgr2_finish(
            a_local,
            b_local,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            1,
            0,
        )
    accumulators, current_a, current_b = (_wg_logical_rect_k64_direct_pgr2_step(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        3,
        a_local,
        b_local,
        1,
        0,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        direct_chunk,
        warps_per_cta,
        refill_read_window,
        read_cover_sixteenths,
    ))
    for kb in tl.range(
            2,
            k_blocks - 2,
            2,
            num_stages=1,
            loop_unroll_factor=loop_unroll_factor,
    ):
        accumulators, current_a, current_b = (_wg_logical_rect_k64_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2,
            a_local,
            b_local,
            0,
            1,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            direct_chunk,
            warps_per_cta,
            refill_read_window,
            read_cover_sixteenths,
        ))
        accumulators, current_a, current_b = (_wg_logical_rect_k64_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 3,
            a_local,
            b_local,
            1,
            0,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            direct_chunk,
            warps_per_cta,
            refill_read_window,
            read_cover_sixteenths,
        ))

    return _wg_logical_rect_k64_direct_pgr2_finish(
        a_local,
        b_local,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        0,
        1,
    )


@triton.jit
def _wg_logical_rect_k64_local_load_a_half(
    a_local,
    stage: tl.constexpr,
    row: tl.constexpr,
    half: tl.constexpr,
    a_group_m: tl.constexpr,
    dot_a: tl.constexpr,
):
    """Load one logical A fragment from its LDS backing chunk."""
    chunk_rows: tl.constexpr = (256 if len(a_local) == 1 else (32 if len(a_local) == 8 else
                                                               (64 if len(a_local) == 4 else 128)))
    rows_per_chunk: tl.constexpr = chunk_rows // a_group_m
    buffer: tl.constexpr = row // rows_per_chunk
    local_row: tl.constexpr = (row % rows_per_chunk) * a_group_m
    return tlx.local_load(
        tlx.local_slice(
            tlx.local_view(a_local[buffer], stage),
            [local_row, half * 32],
            [a_group_m, 32],
        ),
        layout=dot_a,
        relaxed=True,
    )


@triton.jit
def _wg_logical_rect_k64_local_load_b_half(
    b_local,
    stage: tl.constexpr,
    group: tl.constexpr,
    half: tl.constexpr,
    b_group_n: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Load one logical B fragment from its LDS backing chunk."""
    chunk_columns: tl.constexpr = (256 if len(b_local) == 1 else (32 if len(b_local) == 8 else
                                                                  (64 if len(b_local) == 4 else 128)))
    groups_per_chunk: tl.constexpr = chunk_columns // b_group_n
    buffer: tl.constexpr = group // groups_per_chunk
    local_column: tl.constexpr = (group % groups_per_chunk) * b_group_n
    return tlx.local_load(
        tlx.local_slice(
            tlx.local_view(b_local[buffer], stage),
            [half * 32, local_column],
            [32, b_group_n],
        ),
        layout=dot_b,
        relaxed=True,
    )


@triton.jit
def _wg_logical_rect_k64_local_load_operand_half(
    a_local,
    b_local,
    stage: tl.constexpr,
    half: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Load one complete operand half, starting with the smaller side."""
    a_operands = tl.tuple([])
    b_operands = tl.tuple([])
    a_is_smaller: tl.constexpr = a_row_count < b_group_count
    for index in tl.static_range(a_row_count + b_group_count):
        if a_is_smaller:
            if index < a_row_count:
                value = _wg_logical_rect_k64_local_load_a_half(
                    a_local,
                    stage,
                    index,
                    half,
                    a_group_m,
                    dot_a,
                )
                a_operands += tl.tuple([value])
            else:
                value = _wg_logical_rect_k64_local_load_b_half(
                    b_local,
                    stage,
                    index - a_row_count,
                    half,
                    b_group_n,
                    dot_b,
                )
                b_operands += tl.tuple([value])
        else:
            if index < b_group_count:
                value = _wg_logical_rect_k64_local_load_b_half(
                    b_local,
                    stage,
                    index,
                    half,
                    b_group_n,
                    dot_b,
                )
                b_operands += tl.tuple([value])
            else:
                value = _wg_logical_rect_k64_local_load_a_half(
                    a_local,
                    stage,
                    index - b_group_count,
                    half,
                    a_group_m,
                    dot_a,
                )
                a_operands += tl.tuple([value])
    return a_operands, b_operands


@tl.core.builtin
def _wg_logical_rect_k64_allocate_direct_operand(
    logical_extent: tl.constexpr,
    local_stages: tl.constexpr,
    direct_chunk: tl.constexpr,
    pack_chunks: tl.constexpr,
    is_a: tl.constexpr,
    element_type: tl.constexpr,
    n_contiguous: tl.constexpr = False,
    _semantic=None,
):
    """Allocate one A or B axis using the common direct-LDS backing rules."""
    logical_extent = tl.core._unwrap_if_constexpr(logical_extent)
    direct_chunk = tl.core._unwrap_if_constexpr(direct_chunk)
    pack_chunks = tl.core._unwrap_if_constexpr(pack_chunks)
    is_a = tl.core._unwrap_if_constexpr(is_a)
    n_contiguous = tl.core._unwrap_if_constexpr(n_contiguous)
    assert isinstance(is_a, bool)
    assert isinstance(n_contiguous, bool)
    assert not n_contiguous or not is_a
    if is_a:
        shape128: tl.constexpr = [128, 64]
        shape256: tl.constexpr = [256, 64]
        shape64: tl.constexpr = [64, 64]
        shape32: tl.constexpr = [32, 64]
        order: tl.constexpr = [1, 0]
        bases128: tl.constexpr = _wg_A_BASES_128X64
        bases64: tl.constexpr = _wg_A_BASES_64X64
    else:
        shape128: tl.constexpr = [64, 128]
        shape256: tl.constexpr = [64, 256]
        shape64: tl.constexpr = [64, 64]
        shape32: tl.constexpr = [64, 32]
        order: tl.constexpr = [1, 0] if n_contiguous else [0, 1]
        bases128: tl.constexpr = _wg_B_BASES_64X128
        bases64: tl.constexpr = _wg_B_BASES_64X64

    if logical_extent <= 128:
        if pack_chunks:
            small_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)], shape128,
                                                                                              order=order))
        else:
            small_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], bases128, shape128))
        return tl.tuple(
            [tlx.local_alloc(
                shape128,
                element_type,
                local_stages,
                layout=small_layout,
                _semantic=_semantic,
            )])

    if pack_chunks:
        packed_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)], shape256,
                                                                                           order=order))
        return tl.tuple(
            [tlx.local_alloc(
                shape256,
                element_type,
                local_stages,
                layout=packed_layout,
                _semantic=_semantic,
            )])

    if logical_extent == 256 and direct_chunk == 32:
        chunk32_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)], shape32,
                                                                                            order=order))
        return tl.tuple([
            tlx.local_alloc(
                shape32,
                element_type,
                local_stages,
                layout=chunk32_layout,
                _semantic=_semantic,
            ) for _ in range(8)
        ])

    layout64: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _wg_A_BASES_64X64, shape64) if
                              n_contiguous else tlx.padded_shared_layout_encoding.with_bases([(512,
                                                                                               16)], bases64, shape64))
    if logical_extent == 256 and direct_chunk == 64:
        return tl.tuple([
            tlx.local_alloc(
                shape64,
                element_type,
                local_stages,
                layout=layout64,
                _semantic=_semantic,
            ) for _ in range(4)
        ])

    layout128: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _wg_N_BASES_64X128, shape128)
                               if n_contiguous else tlx.padded_shared_layout_encoding.with_bases([(512, 16)], bases128,
                                                                                                 shape128))
    if logical_extent == 192:
        return tl.tuple([
            tlx.local_alloc(
                shape128,
                element_type,
                local_stages,
                layout=layout128,
                _semantic=_semantic,
            ),
            tlx.local_alloc(
                shape64,
                element_type,
                local_stages,
                layout=layout64,
                _semantic=_semantic,
            ),
        ])
    return tl.tuple([
        tlx.local_alloc(
            shape128,
            element_type,
            local_stages,
            layout=layout128,
            _semantic=_semantic,
        ) for _ in range(2)
    ])


@tl.core.builtin
def _wg_logical_rect_k128_allocate_direct_operand(
    logical_extent: tl.constexpr,
    is_a: tl.constexpr,
    element_type: tl.constexpr,
    _semantic=None,
):
    """Allocate two K128 banks from independently addressable K64 chunks."""
    logical_extent = tl.core._unwrap_if_constexpr(logical_extent)
    is_a = tl.core._unwrap_if_constexpr(is_a)
    chunks_per_half = 1 if logical_extent <= 128 else 2
    buffers = []
    # A K128 stage contains two K64 halves.  Keeping each half in its own
    # allocation lets the mature bank-safe K64 direct-load layouts be reused,
    # while a pair of stages halves the barrier rate versus the K64 pipeline.
    for _ in range(2 * 2):
        chunks = _wg_logical_rect_k64_allocate_direct_operand(
            logical_extent,
            tl.constexpr(1),
            tl.constexpr(128),
            tl.constexpr(False),
            is_a,
            element_type,
            _semantic=_semantic,
        )
        for chunk in range(chunks_per_half):
            buffers.append(chunks[chunk])
    return tl.tuple(buffers)


@tl.core.builtin
def _wg_wave_grid_allocate_pgr2_operand(
    logical_extent: tl.constexpr,
    group_extent: tl.constexpr,
    block_k: tl.constexpr,
    allocation_stages: tl.constexpr,
    is_a: tl.constexpr,
    element_type: tl.constexpr,
    n_contiguous: tl.constexpr = False,
    _semantic=None,
):
    """Allocate one consolidated PGR2 operand with its bank-safe layout."""
    logical_extent = tl.core._unwrap_if_constexpr(logical_extent)
    group_extent = tl.core._unwrap_if_constexpr(group_extent)
    block_k = tl.core._unwrap_if_constexpr(block_k)
    allocation_stage_count = tl.core._unwrap_if_constexpr(allocation_stages)
    is_a = tl.core._unwrap_if_constexpr(is_a)
    n_contiguous = tl.core._unwrap_if_constexpr(n_contiguous)
    assert isinstance(is_a, bool)
    assert isinstance(n_contiguous, bool)
    assert not n_contiguous or not is_a
    assert group_extent in (16, 32, 64)

    padded_extent = 128 if logical_extent <= 128 else 256
    if n_contiguous:
        identity = [(512, 16)]
    elif allocation_stage_count == 2:
        # Make the swizzle period span both physical K64 banks.  Reusing the
        # one-bank identity independently in each stage can place
        # corresponding K32 accesses in the same swizzle phase; the two-bank
        # identity preserves the intended phase separation across stages.
        identity = [(2 * block_k if is_a else 2 * group_extent, 16)]
    else:
        identity = ([(64, 16)] if group_extent == 16 else ([(512, 16)] if group_extent == 32 else [(256, 16)]))
    shape = ([padded_extent, block_k] if is_a else [block_k, padded_extent])
    order = [1, 0] if is_a or n_contiguous else [0, 1]
    layout = tlx.padded_shared_layout_encoding.with_identity_for(identity, shape, order=order)
    return tlx.local_alloc(
        shape,
        element_type,
        allocation_stages,
        layout=layout,
        _semantic=_semantic,
    )


@tl.core.builtin
def _wg_wave_grid_allocate_pgr2_operand_chunks(
    logical_extent: tl.constexpr,
    group_extent: tl.constexpr,
    block_k: tl.constexpr,
    allocation_stages: tl.constexpr,
    is_a: tl.constexpr,
    element_type: tl.constexpr,
    n_contiguous: tl.constexpr = False,
    _semantic=None,
):
    """Allocate a PGR2 operand in addressable, at-most-256-wide chunks."""
    logical_extent = tl.core._unwrap_if_constexpr(logical_extent)
    group_extent = tl.core._unwrap_if_constexpr(group_extent)
    block_k = tl.core._unwrap_if_constexpr(block_k)
    allocation_stages = tl.core._unwrap_if_constexpr(allocation_stages)
    is_a = tl.core._unwrap_if_constexpr(is_a)
    n_contiguous = tl.core._unwrap_if_constexpr(n_contiguous)
    chunk_count = (logical_extent + 255) // 256
    chunks = []
    for chunk in range(chunk_count):
        remaining = logical_extent - 256 * chunk
        chunk_extent = min(remaining, 256)
        chunks.append(
            _wg_wave_grid_allocate_pgr2_operand(
                chunk_extent,
                group_extent,
                block_k,
                tl.constexpr(allocation_stages),
                is_a,
                element_type,
                n_contiguous,
                _semantic=_semantic,
            ))
    return tl.tuple(chunks)


@tl.core.builtin
def _wg_wave_grid_allocate_pgr2_double_buffer_operand(
    logical_extent: tl.constexpr,
    group_extent: tl.constexpr,
    block_k: tl.constexpr,
    is_a: tl.constexpr,
    element_type: tl.constexpr,
    n_contiguous: tl.constexpr = False,
    _semantic=None,
):
    """Allocate two exact, independently addressable PGR2 operand banks."""
    logical_extent = tl.core._unwrap_if_constexpr(logical_extent)
    group_extent = tl.core._unwrap_if_constexpr(group_extent)
    block_k = tl.core._unwrap_if_constexpr(block_k)
    is_a = tl.core._unwrap_if_constexpr(is_a)
    n_contiguous = tl.core._unwrap_if_constexpr(n_contiguous)
    assert not n_contiguous or not is_a
    group_count = logical_extent // group_extent
    # A PGR2 group is read twice, one K32 half at a time.  Padding at two
    # contiguous rows keeps those half-tile reads on distinct LDS banks while
    # preserving a regular layout for every supported wave geometry.
    identity = ([(512, 16)] if n_contiguous else [(
        2 * block_k if is_a else 2 * group_extent,
        16,
    )])
    shape = ([group_extent, block_k] if is_a else [block_k, group_extent])
    order = [1, 0] if is_a or n_contiguous else [0, 1]
    layout = tlx.padded_shared_layout_encoding.with_identity_for(identity, shape, order=order)
    return tl.tuple([
        tlx.local_alloc(
            shape,
            element_type,
            tl.constexpr(1),
            layout=layout,
            _semantic=_semantic,
        ) for _ in range(2 * group_count)
    ])


@triton.jit
def _wg_logical_rect_k64_direct_load_extent(
    ptr,
    pid,
    kb,
    destination,
    outer_start: tl.constexpr,
    outer_extent: tl.constexpr,
    block_extent: tl.constexpr,
    stride_outer: tl.constexpr,
    stride_k: tl.constexpr,
    logical_extent: tl.constexpr,
    offset_layout: tl.constexpr,
    is_a: tl.constexpr,
    scalar_k_base: tl.constexpr = False,
):
    """Load one A-row or B-column backing extent directly into LDS."""
    source = ptr
    if scalar_k_base:
        source += kb * 64 * stride_k
    reduction = tl.arange(0, 64)
    local_outer = outer_start + tl.arange(0, outer_extent)
    if logical_extent * stride_outer >= 1 << 30:
        outer_base = pid * block_extent + outer_start
        safe_outer_base = tl.minimum(outer_base, logical_extent - 1)
        source += safe_outer_base.to(tl.int64) * stride_outer
        if not scalar_k_base:
            source += kb * 64 * stride_k
        absolute_outer = pid * block_extent + local_outer
        outer = tl.where(
            (local_outer < block_extent)
            & (absolute_outer < logical_extent),
            absolute_outer - safe_outer_base,
            0,
        )
    else:
        outer = pid * block_extent + local_outer
        outer = tl.where(
            (local_outer < block_extent) & (outer < logical_extent),
            outer,
            0,
        )

    if is_a:
        if scalar_k_base:
            offsets = (outer[:, None] * stride_outer + reduction[None, :] * stride_k)
        else:
            offsets = (outer[:, None] * stride_outer + (kb * 64 + reduction[None, :]) * stride_k)
        offsets = tl.max_contiguous(tl.multiple_of(offsets, (1, 8)), (1, 8))
    else:
        if scalar_k_base:
            offsets = (reduction[:, None] * stride_k + outer[None, :] * stride_outer)
        else:
            offsets = ((kb * 64 + reduction[:, None]) * stride_k + outer[None, :] * stride_outer)
        if stride_outer == 1:
            offsets = tl.max_contiguous(tl.multiple_of(offsets, (1, 8)), (1, 8))
        else:
            offsets = tl.max_contiguous(tl.multiple_of(offsets, (8, 1)), (8, 1))
    if offset_layout is not None:
        offsets = tlx.require_layout(offsets, offset_layout)
    # The M279/N8192 production plan reuses its compact A working set across
    # many N tiles while B is a one-pass 32 MiB stream.  Keep A in L1 and
    # bypass L1 for B so cold-cache runs do not evict the reusable operand.
    cache_modifier: tl.constexpr = (".cg" if not is_a and logical_extent == 8192 and block_extent == 128 else "")
    tlx.buffer_load_to_local(
        destination,
        source,
        offsets,
        cache_modifier=cache_modifier,
        contiguity=8,
    )


@triton.jit
def _wg_logical_rect_k64_direct_load_packed_chunk(
    ptr,
    pid,
    kb,
    local,
    stage,
    chunk: tl.constexpr,
    chunk_extent: tl.constexpr,
    block_extent: tl.constexpr,
    stride_outer: tl.constexpr,
    stride_k: tl.constexpr,
    logical_extent: tl.constexpr,
    warps_per_cta: tl.constexpr,
    is_a: tl.constexpr,
):
    """Load one A-row or B-column window into a shared packed backing."""
    source = ptr + kb * 64 * stride_k
    reduction = tl.arange(0, 64)
    local_outer = chunk * chunk_extent + tl.arange(0, chunk_extent)
    if logical_extent % block_extent == 0:
        # Keep the program and chunk displacement in the scalar resource
        # base so every scheduled window reuses one compact vector offset.
        source += (pid * block_extent + chunk * chunk_extent) * stride_outer
        outer = tl.arange(0, chunk_extent)
    elif logical_extent * stride_outer >= 1 << 30:
        outer_start: tl.constexpr = chunk * chunk_extent
        outer_base = pid * block_extent + outer_start
        safe_outer_base = tl.minimum(outer_base, logical_extent - 1)
        source += safe_outer_base.to(tl.int64) * stride_outer
        absolute_outer = pid * block_extent + local_outer
        outer = tl.where(
            (local_outer < block_extent)
            & (absolute_outer < logical_extent),
            absolute_outer - safe_outer_base,
            0,
        )
    else:
        outer = pid * block_extent + local_outer
        outer = tl.where(
            (local_outer < block_extent) & (outer < logical_extent),
            outer,
            0,
        )

    if is_a:
        offsets = (outer[:, None] * stride_outer + reduction[None, :] * stride_k)
        offsets = tl.max_contiguous(tl.multiple_of(offsets, (1, 8)), (1, 8))
        if chunk_extent == 32:
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_32X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_32X64_4W)
        elif chunk_extent == 64:
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_64X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_64X64_4W)
        else:
            tl.static_assert(chunk_extent == 128)
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_128X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_128X64_4W)
        destination = tlx.local_slice(
            tlx.local_view(local, stage),
            [chunk * chunk_extent, 0],
            [chunk_extent, 64],
        )
    else:
        offsets = (reduction[:, None] * stride_k + outer[None, :] * stride_outer)
        if stride_outer == 1:
            offsets = tl.max_contiguous(tl.multiple_of(offsets, (1, 8)), (1, 8))
        else:
            offsets = tl.max_contiguous(tl.multiple_of(offsets, (8, 1)), (8, 1))
        if chunk_extent == 32:
            if stride_outer == 1:
                tl.static_assert(warps_per_cta == 4)
                offset_layout: tl.constexpr = _wg_A_OFFSET_LAYOUT_64X32
            else:
                offset_layout: tl.constexpr = (_wg_B_OFFSET_LAYOUT_64X32_8W
                                               if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X32_4W)
        elif chunk_extent == 64:
            offset_layout: tl.constexpr = (_wg_B_OFFSET_LAYOUT_64X64_8W
                                           if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X64_4W)
        else:
            tl.static_assert(chunk_extent == 128)
            offset_layout: tl.constexpr = (_wg_B_OFFSET_LAYOUT_64X128_8W
                                           if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X128_4W)
        destination = tlx.local_slice(
            tlx.local_view(local, stage),
            [0, chunk * chunk_extent],
            [64, chunk_extent],
        )
    offsets = tlx.require_layout(offsets, offset_layout)
    tlx.buffer_load_to_local(destination, source, offsets, contiguity=8)


@triton.jit
def _wg_logical_rect_k64_direct_load_chunk(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    a_local,
    b_local,
    stage,
    chunk: tl.constexpr,
    direct_chunk: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    warps_per_cta: tl.constexpr = 4,
):
    """Load one physical backing chunk directly into LDS.

    This does not pre-pad or copy either input. Logical macro tiles may end
    before the 256-wide LDS backing tile. The default representation gives a
    192-element axis an exact 128+64 backing. Deep-K plans may instead share
    one 256-wide allocation while retaining independent 32-wide load windows;
    a non-power-of-two axis issues only enough windows for its logical extent.
    Unused lanes in the final window are redirected to a safe in-bounds
    element that no MFMA consumes. This keeps direct loads unmasked while
    supporting NPOT MI wave grids in one kernel launch.
    """
    # A single backing allocation may also represent a logical tile narrower
    # than 128.  In that case issue only the required 32/64-wide windows; the
    # remaining LDS capacity is address padding, not extra global traffic.
    packed_a: tl.constexpr = len(a_local) == 1 and direct_chunk < 128
    packed_b: tl.constexpr = len(b_local) == 1 and direct_chunk < 128
    b_chunk_count: tl.constexpr = ((block_n + direct_chunk - 1) // direct_chunk if packed_b else len(b_local))
    a_chunk_count: tl.constexpr = ((block_m + direct_chunk - 1) // direct_chunk if packed_a else len(a_local))
    if chunk < b_chunk_count:
        commit_index: tl.constexpr = chunk
        commit_count: tl.constexpr = b_chunk_count
        if packed_b:
            _wg_logical_rect_k64_direct_load_packed_chunk(
                b_ptr,
                pid_n,
                kb,
                b_local[0],
                stage,
                chunk,
                direct_chunk,
                block_n,
                stride_bn,
                stride_bk,
                n,
                warps_per_cta,
                False,
            )
        elif b_chunk_count == 8:
            # Fine 32-wide backing is selected only for deep K. Moving the
            # common K coordinate into the scalar resource base removes one
            # repeated vector add per lane and per backing chunk.
            offset_layout: tl.constexpr = ((
                _wg_A_OFFSET_LAYOUT_32X64_8W if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_64X32)
                                           if stride_bn == 1 else _wg_B_OFFSET_LAYOUT_64X32_4W)
            _wg_logical_rect_k64_direct_load_extent(
                b_ptr,
                pid_n,
                kb,
                b_local[chunk][stage],
                chunk * 32,
                32,
                block_n,
                stride_bn,
                stride_bk,
                n,
                offset_layout,
                False,
                True,
            )
        elif b_chunk_count == 4:
            offset_layout: tl.constexpr = (_wg_N_CONTIG_OFFSET_LAYOUT_64X64_4W if stride_bn == 1 else (
                _wg_B_OFFSET_LAYOUT_64X64_8W if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X64_4W))
            _wg_logical_rect_k64_direct_load_extent(
                b_ptr,
                pid_n,
                kb,
                b_local[chunk][stage],
                chunk * 64,
                64,
                block_n,
                stride_bn,
                stride_bk,
                n,
                offset_layout,
                False,
            )
        elif block_n == 192 and chunk == 1:
            offset_layout: tl.constexpr = (_wg_B_OFFSET_LAYOUT_64X64_8W
                                           if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X64_4W)
            _wg_logical_rect_k64_direct_load_extent(
                b_ptr,
                pid_n,
                kb,
                b_local[1][stage],
                128,
                64,
                block_n,
                stride_bn,
                stride_bk,
                n,
                offset_layout,
                False,
            )
        else:
            offset_layout: tl.constexpr = (
                (_wg_N_CONTIG_OFFSET_LAYOUT_64X128_8W if warps_per_cta == 8 else _wg_N_CONTIG_OFFSET_LAYOUT_64X128_4W)
                if stride_bn == 1 else
                (_wg_B_OFFSET_LAYOUT_64X128_8W if warps_per_cta == 8 else _wg_B_OFFSET_LAYOUT_64X128_4W))
            _wg_logical_rect_k64_direct_load_extent(
                b_ptr,
                pid_n,
                kb,
                b_local[chunk][stage],
                chunk * 128,
                128,
                block_n,
                stride_bn,
                stride_bk,
                n,
                offset_layout,
                False,
            )
    else:
        a_chunk: tl.constexpr = chunk - b_chunk_count
        commit_index: tl.constexpr = a_chunk
        commit_count: tl.constexpr = a_chunk_count
        if packed_a:
            _wg_logical_rect_k64_direct_load_packed_chunk(
                a_ptr,
                pid_m,
                kb,
                a_local[0],
                stage,
                a_chunk,
                direct_chunk,
                block_m,
                stride_am,
                stride_ak,
                m,
                warps_per_cta,
                True,
            )
        elif a_chunk_count == 8:
            _wg_logical_rect_k64_direct_load_extent(
                a_ptr,
                pid_m,
                kb,
                a_local[a_chunk][stage],
                a_chunk * 32,
                32,
                block_m,
                stride_am,
                stride_ak,
                m,
                _wg_A_OFFSET_LAYOUT_32X64_4W,
                True,
                True,
            )
        elif a_chunk_count == 4:
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_64X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_64X64_4W)
            _wg_logical_rect_k64_direct_load_extent(
                a_ptr,
                pid_m,
                kb,
                a_local[a_chunk][stage],
                a_chunk * 64,
                64,
                block_m,
                stride_am,
                stride_ak,
                m,
                offset_layout,
                True,
            )
        elif block_m == 192 and a_chunk == 1:
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_64X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_64X64_4W)
            _wg_logical_rect_k64_direct_load_extent(
                a_ptr,
                pid_m,
                kb,
                a_local[1][stage],
                128,
                64,
                block_m,
                stride_am,
                stride_ak,
                m,
                offset_layout,
                True,
            )
        else:
            offset_layout: tl.constexpr = (_wg_A_OFFSET_LAYOUT_128X64_8W
                                           if warps_per_cta == 8 else _wg_A_OFFSET_LAYOUT_128X64_4W)
            _wg_logical_rect_k64_direct_load_extent(
                a_ptr,
                pid_m,
                kb,
                a_local[a_chunk][stage],
                a_chunk * 128,
                128,
                block_m,
                stride_am,
                stride_ak,
                m,
                offset_layout,
                True,
            )
    # A 32-wide packed window lowers to one LDSDMA instruction. Keep those
    # windows individually schedulable, but commit four together so the async
    # pipeline retains the same 128-wide group granularity as the mature path.
    packed_fine_loads: tl.constexpr = (direct_chunk == 32 and packed_a and packed_b)
    if (not packed_fine_loads or (commit_index + 1) % 4 == 0 or commit_index + 1 == commit_count):
        tlx.async_load_commit_group()


@triton.jit
def _wg_logical_rect_k64_direct_load_stage(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    a_local,
    b_local,
    stage: tl.constexpr,
    direct_chunk: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    warps_per_cta: tl.constexpr = 4,
):
    direct_load_count: tl.constexpr = (
        ((block_m + direct_chunk - 1) // direct_chunk if len(a_local) == 1 else len(a_local)) +
        ((block_n + direct_chunk - 1) // direct_chunk if len(b_local) == 1 else len(b_local)))
    for chunk in tl.static_range(direct_load_count):
        _wg_logical_rect_k64_direct_load_chunk(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb,
            a_local,
            b_local,
            stage,
            chunk,
            direct_chunk,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            warps_per_cta,
        )


@triton.jit
def _wg_logical_rect_k128_direct_load_stage(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    macro_k,
    a_local,
    b_local,
    stage: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Publish one K128 stage as independently scheduled K64 chunks."""
    a_chunks: tl.constexpr = 1 if block_m <= 128 else 2
    b_chunks: tl.constexpr = 1 if block_n <= 128 else 2
    for half in tl.static_range(2):
        for chunk in tl.static_range(b_chunks):
            _wg_logical_rect_k64_direct_load_extent(
                b_ptr,
                pid_n,
                macro_k * 2 + half,
                b_local[(stage * 2 + half) * b_chunks + chunk][0],
                chunk * 128,
                64 if block_n == 192 and chunk == 1 else 128,
                block_n,
                stride_bn,
                stride_bk,
                n,
                (_wg_B_OFFSET_LAYOUT_64X64_4W if block_n == 192 and chunk == 1 else _wg_B_OFFSET_LAYOUT_64X128_4W),
                False,
            )
            tlx.async_load_commit_group()
        for chunk in tl.static_range(a_chunks):
            _wg_logical_rect_k64_direct_load_extent(
                a_ptr,
                pid_m,
                macro_k * 2 + half,
                a_local[(stage * 2 + half) * a_chunks + chunk][0],
                chunk * 128,
                64 if block_m == 192 and chunk == 1 else 128,
                block_m,
                stride_am,
                stride_ak,
                m,
                (_wg_A_OFFSET_LAYOUT_64X64_4W if block_m == 192 and chunk == 1 else _wg_A_OFFSET_LAYOUT_128X64_4W),
                True,
            )
            tlx.async_load_commit_group()


@triton.jit
def _wg_logical_rect_k128_local_load_quarter(
    a_local,
    b_local,
    stage: tl.constexpr,
    quarter: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Read one K32 quarter from a K128 direct-to-LDS stage."""
    a_chunks: tl.constexpr = len(a_local) // 4
    b_chunks: tl.constexpr = len(b_local) // 4
    bank: tl.constexpr = stage * 2 + quarter // 2
    half: tl.constexpr = quarter % 2
    a_rows_per_chunk: tl.constexpr = 128 // a_group_m
    b_groups_per_chunk: tl.constexpr = 128 // b_group_n
    a_operands = tl.tuple([])
    b_operands = tl.tuple([])
    a_is_smaller: tl.constexpr = a_row_count < b_group_count
    if a_is_smaller:
        for row in tl.static_range(a_row_count):
            a_operands += tl.tuple([
                tlx.local_load(
                    tlx.local_slice(
                        tlx.local_view(
                            a_local[bank * a_chunks + row // a_rows_per_chunk],
                            0,
                        ),
                        [
                            (row % a_rows_per_chunk) * a_group_m,
                            half * 32,
                        ],
                        [a_group_m, 32],
                    ),
                    layout=dot_a,
                    relaxed=True,
                )
            ])
        for group in tl.static_range(b_group_count):
            b_operands += tl.tuple([
                tlx.local_load(
                    tlx.local_slice(
                        tlx.local_view(
                            b_local[bank * b_chunks + group // b_groups_per_chunk],
                            0,
                        ),
                        [
                            half * 32,
                            (group % b_groups_per_chunk) * b_group_n,
                        ],
                        [32, b_group_n],
                    ),
                    layout=dot_b,
                    relaxed=True,
                )
            ])
    else:
        for group in tl.static_range(b_group_count):
            b_operands += tl.tuple([
                tlx.local_load(
                    tlx.local_slice(
                        tlx.local_view(
                            b_local[bank * b_chunks + group // b_groups_per_chunk],
                            0,
                        ),
                        [
                            half * 32,
                            (group % b_groups_per_chunk) * b_group_n,
                        ],
                        [32, b_group_n],
                    ),
                    layout=dot_b,
                    relaxed=True,
                )
            ])
        for row in tl.static_range(a_row_count):
            a_operands += tl.tuple([
                tlx.local_load(
                    tlx.local_slice(
                        tlx.local_view(
                            a_local[bank * a_chunks + row // a_rows_per_chunk],
                            0,
                        ),
                        [
                            (row % a_rows_per_chunk) * a_group_m,
                            half * 32,
                        ],
                        [a_group_m, 32],
                    ),
                    layout=dot_a,
                    relaxed=True,
                )
            ])
    return a_operands, b_operands


@triton.jit
def _wg_logical_rect_k128_direct_pgr2_step(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_macro,
    a_local,
    b_local,
    current_stage: tl.constexpr,
    next_stage: tl.constexpr,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance one K128 stage while publishing the stage after next."""
    mfma_count: tl.constexpr = a_row_count * b_group_count
    with tlx.warp_pipeline_stage("direct_k128_read", scope="intra_wave"):
        q1_a, q1_b = _wg_logical_rect_k128_local_load_quarter(
            a_local,
            b_local,
            current_stage,
            1,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        q2_a, q2_b = _wg_logical_rect_k128_local_load_quarter(
            a_local,
            b_local,
            current_stage,
            2,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        q3_a, q3_b = _wg_logical_rect_k128_local_load_quarter(
            a_local,
            b_local,
            current_stage,
            3,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize=initialize,
        )

    tl.debug_barrier()
    with tlx.warp_pipeline_stage("direct_k128_refill", scope="intra_wave", pair=0):
        _wg_logical_rect_k128_direct_load_stage(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            future_macro,
            a_local,
            b_local,
            current_stage,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
    with tlx.warp_pipeline_stage("direct_k128_compute_refill", scope="intra_wave", pair=0):
        accumulators = _wg_wave_grid_dot_column_major_range(
            q1_a,
            q1_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            q2_a,
            q2_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    groups_per_stage: tl.constexpr = 2 * (len(a_local) // 4 + len(b_local) // 4)
    tlx.async_load_wait_group(groups_per_stage)
    tl.debug_barrier()
    with tlx.warp_pipeline_stage("direct_k128_next", scope="intra_wave"):
        next_a, next_b = _wg_logical_rect_k128_local_load_quarter(
            a_local,
            b_local,
            next_stage,
            0,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            q3_a,
            q3_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators, next_a, next_b


@triton.jit
def _wg_logical_rect_k128_direct_pgr2_finish_stage(
    a_local,
    b_local,
    stage: tl.constexpr,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
):
    """Consume all four K32 quarters of one already-resident K128 stage."""
    mfma_count: tl.constexpr = a_row_count * b_group_count
    accumulators = _wg_wave_grid_dot_column_major_range(
        current_a,
        current_b,
        accumulators,
        0,
        mfma_count,
        a_row_count,
        b_group_count,
        mma,
        dot_a,
        dot_b,
    )
    for quarter in tl.static_range(1, 4):
        quarter_a, quarter_b = _wg_logical_rect_k128_local_load_quarter(
            a_local,
            b_local,
            stage,
            quarter,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            dot_a,
            dot_b,
        )
        accumulators = _wg_wave_grid_dot_column_major_range(
            quarter_a,
            quarter_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators


@triton.jit
def _wg_logical_rect_k128_compute_tile_direct_pgr2(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    """Compute a regular four-wave grid with a K128 direct PGR2 pipeline."""
    macro_blocks: tl.constexpr = k // 128
    tl.static_assert(k % 128 == 0 and macro_blocks >= 4)
    _wg_logical_rect_k128_direct_load_stage(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        a_local,
        b_local,
        0,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    _wg_logical_rect_k128_direct_load_stage(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        a_local,
        b_local,
        1,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    groups_per_stage: tl.constexpr = 2 * (len(a_local) // 4 + len(b_local) // 4)
    tlx.async_load_wait_group(groups_per_stage)
    tl.debug_barrier()
    current_a, current_b = _wg_logical_rect_k128_local_load_quarter(
        a_local,
        b_local,
        0,
        0,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        dot_a,
        dot_b,
    )
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])
    accumulators, current_a, current_b = (_wg_logical_rect_k128_direct_pgr2_step(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2,
        a_local,
        b_local,
        0,
        1,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        initialize=True,
    ))
    accumulators, current_a, current_b = (_wg_logical_rect_k128_direct_pgr2_step(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        3,
        a_local,
        b_local,
        1,
        0,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        block_m,
        block_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    ))
    for macro in tl.range(2, macro_blocks - 2, 2, num_stages=1):
        accumulators, current_a, current_b = (_wg_logical_rect_k128_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            macro + 2,
            a_local,
            b_local,
            0,
            1,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        ))
        accumulators, current_a, current_b = (_wg_logical_rect_k128_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            macro + 3,
            a_local,
            b_local,
            1,
            0,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        ))
    accumulators = _wg_logical_rect_k128_direct_pgr2_finish_stage(
        a_local,
        b_local,
        0,
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    final_a, final_b = _wg_logical_rect_k128_local_load_quarter(
        a_local,
        b_local,
        1,
        0,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        dot_a,
        dot_b,
    )
    return _wg_logical_rect_k128_direct_pgr2_finish_stage(
        a_local,
        b_local,
        1,
        final_a,
        final_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )


@triton.jit
def _wg_logical_rect_k64_compute_stage(
    a_local,
    b_local,
    stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
):
    for half in tl.static_range(2):
        a_operands = tl.tuple([
            _wg_logical_rect_k64_local_load_a_half(a_local, stage, row, half, a_group_m, dot_a)
            for row in range(a_row_count)
        ])
        b_operands = tl.tuple([
            _wg_logical_rect_k64_local_load_b_half(b_local, stage, group, half, b_group_n, dot_b)
            for group in range(b_group_count)
        ])
        for mfma in tl.static_range(a_row_count * b_group_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                a_operands,
                b_operands,
                accumulators,
                mfma,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
    return accumulators


@triton.jit
def _wg_logical_rect_k64_compute_and_prefetch_direct(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    next_kb,
    a_local,
    b_local,
    current_stage: tl.constexpr,
    next_stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    direct_chunk: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    warps_per_cta: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume one logical grid while refilling its physical LDS backing."""
    operand_count: tl.constexpr = a_row_count + b_group_count
    mfma_count: tl.constexpr = a_row_count * b_group_count
    tl.static_assert(mfma_count >= operand_count)

    current_a = tl.tuple([
        _wg_logical_rect_k64_local_load_a_half(a_local, current_stage, row, 0, a_group_m, dot_a)
        for row in range(a_row_count)
    ])
    current_b = tl.tuple([
        _wg_logical_rect_k64_local_load_b_half(b_local, current_stage, group, 0, b_group_n, dot_b)
        for group in range(b_group_count)
    ])

    second_values = tl.tuple([])
    with tlx.warp_pipeline_stage("logical_rect_read_and_consume", scope="intra_wave"):
        for read in tl.static_range(operand_count):
            if read < b_group_count:
                value = _wg_logical_rect_k64_local_load_b_half(b_local, current_stage, read, 1, b_group_n, dot_b)
            else:
                value = _wg_logical_rect_k64_local_load_a_half(
                    a_local,
                    current_stage,
                    read - b_group_count,
                    1,
                    a_group_m,
                    dot_a,
                )
            second_values += tl.tuple([value])
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize=initialize,
        )
    second_b = tl.tuple([second_values[index] for index in range(b_group_count)])
    second_a = tl.tuple([second_values[b_group_count + index] for index in range(a_row_count)])

    direct_load_count: tl.constexpr = (
        ((block_m + direct_chunk - 1) // direct_chunk if len(a_local) == 1 else len(a_local)) +
        ((block_n + direct_chunk - 1) // direct_chunk if len(b_local) == 1 else len(b_local)))
    with tlx.warp_pipeline_stage("logical_rect_prefetch_chunk", scope="intra_wave", pair=0):
        for chunk in tl.static_range(direct_load_count):
            _wg_logical_rect_k64_direct_load_chunk(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                next_kb,
                a_local,
                b_local,
                next_stage,
                chunk,
                direct_chunk,
                block_m,
                block_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                warps_per_cta,
            )
    with tlx.warp_pipeline_stage("logical_rect_mfma_second_half", scope="intra_wave", pair=0):
        accumulators = _wg_wave_grid_dot_column_major_range(
            second_a,
            second_b,
            accumulators,
            0,
            mfma_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators


@triton.jit
def _wg_logical_rect_k64_compute_tile_direct(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    direct_chunk: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    warps_per_cta: tl.constexpr,
    preloaded: tl.constexpr = False,
):
    """Two-stage direct-to-LDS pipeline for one logical regular MI16 grid."""
    k_blocks: tl.constexpr = k // 64
    tl.static_assert(block_m <= 256 and block_n <= 256)
    tl.static_assert(block_m >= 32 and block_n >= 32)
    tl.static_assert(k_blocks >= 4 and k_blocks % 2 == 0)
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])
    if not preloaded:
        _wg_logical_rect_k64_direct_load_stage(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            0,
            a_local,
            b_local,
            0,
            direct_chunk,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            warps_per_cta,
        )
        _wg_logical_rect_k64_direct_load_stage(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            1,
            a_local,
            b_local,
            1,
            direct_chunk,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            warps_per_cta,
        )
    a_direct_load_count: tl.constexpr = ((block_m + direct_chunk - 1) //
                                         direct_chunk if len(a_local) == 1 else len(a_local))
    b_direct_load_count: tl.constexpr = ((block_n + direct_chunk - 1) //
                                         direct_chunk if len(b_local) == 1 else len(b_local))
    direct_group_count: tl.constexpr = (((a_direct_load_count + 3) // 4 +
                                         (b_direct_load_count + 3) // 4) if direct_chunk == 32 and len(a_local) == 1
                                        and len(b_local) == 1 else a_direct_load_count + b_direct_load_count)
    tlx.async_load_wait_group(direct_group_count)
    tl.debug_barrier()

    accumulators = _wg_logical_rect_k64_compute_and_prefetch_direct(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2,
        a_local,
        b_local,
        0,
        0,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        direct_chunk,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        warps_per_cta,
        True,
    )
    tlx.async_load_wait_group(direct_group_count)
    tl.debug_barrier()
    accumulators = _wg_logical_rect_k64_compute_and_prefetch_direct(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        3,
        a_local,
        b_local,
        1,
        1,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        direct_chunk,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        warps_per_cta,
    )
    tlx.async_load_wait_group(direct_group_count)
    tl.debug_barrier()

    for kb in tl.range(2, k_blocks - 2, 2, num_stages=1):
        accumulators = _wg_logical_rect_k64_compute_and_prefetch_direct(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2,
            a_local,
            b_local,
            0,
            0,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            direct_chunk,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            warps_per_cta,
        )
        tlx.async_load_wait_group(direct_group_count)
        tl.debug_barrier()
        accumulators = _wg_logical_rect_k64_compute_and_prefetch_direct(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 3,
            a_local,
            b_local,
            1,
            1,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            direct_chunk,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            warps_per_cta,
        )
        tlx.async_load_wait_group(direct_group_count)
        tl.debug_barrier()

    accumulators = _wg_logical_rect_k64_compute_stage(
        a_local,
        b_local,
        0,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )
    tlx.async_load_wait_group(0)
    tl.debug_barrier()
    return _wg_logical_rect_k64_compute_stage(
        a_local,
        b_local,
        1,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )


@triton.jit
def _wg_wave_grid_pipeline_step_rect_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    current_stage,
    next_stage,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Advance any regular rectangular MI grid through one K64 stage.

    Every LDS read is paired with one MFMA.  The MFMA interval between the
    current-stage reads and next-stage reads is divided evenly over the B-then-A
    publish sequence.  This derives bounded source scheduling windows from the
    grid geometry instead of spelling out a schedule for each macro tile.
    """
    tl.static_assert(block_k == 64)
    a_group_m: tl.constexpr = block_m // a_row_count
    operand_count: tl.constexpr = a_row_count + b_group_count
    mfma_per_half: tl.constexpr = a_row_count * b_group_count
    pre_publish_count: tl.constexpr = (mfma_per_half - operand_count) // 4
    publish_cover_count: tl.constexpr = (2 * (mfma_per_half - operand_count) - pre_publish_count)
    tl.static_assert(mfma_per_half >= operand_count)

    second_values = tl.tuple([])
    with tlx.warp_pipeline_stage("read_and_consume_current", scope="intra_wave"):
        for read in tl.static_range(operand_count):
            if read < b_group_count:
                value = _wg_wave_grid_local_load_b_half(
                    b_local,
                    current_stage,
                    read,
                    1,
                    b_group_n,
                    dot_b,
                )
            else:
                value = _wg_wave_grid_local_load_a_half(
                    a_local,
                    current_stage,
                    read - b_group_count,
                    1,
                    a_group_m,
                    dot_a,
                )
            second_values += tl.tuple([value])
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            operand_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    second_b = tl.tuple([second_values[index] for index in range(b_group_count)])
    second_a = tl.tuple([second_values[b_group_count + index] for index in range(a_row_count)])

    accumulators = _wg_wave_grid_dot_two_half_range(
        current_a,
        current_b,
        second_a,
        second_b,
        accumulators,
        operand_count,
        pre_publish_count,
        a_row_count,
        b_group_count,
        mma,
        dot_a,
        dot_b,
    )
    tl.debug_barrier()

    future_a = tl.tuple([])
    future_b = tl.tuple([])
    with tlx.warp_pipeline_stage("publish_and_prefetch", scope="intra_wave", pair=0):
        for fragment in tl.static_range(operand_count):
            _wg_wave_grid_local_store_fragment(
                a_local,
                b_local,
                next_stage,
                prefetched_a,
                prefetched_b,
                fragment,
                b_group_count,
                block_k,
                b_group_n,
            )
            if fragment < b_group_count:
                future_b += tl.tuple([
                    _wg_wave_grid_global_load_b(
                        b_ptr,
                        pid_n,
                        future_kb,
                        fragment,
                        block_n,
                        block_k,
                        b_group_n,
                        stride_bk,
                        stride_bn,
                        n,
                    )
                ])
            else:
                future_a += tl.tuple([
                    _wg_wave_grid_global_load_a(
                        a_ptr,
                        pid_m,
                        future_kb,
                        fragment - b_group_count,
                        a_group_m,
                        b_group_n,
                        block_m,
                        block_k,
                        stride_am,
                        stride_ak,
                        m,
                    )
                ])
    with tlx.warp_pipeline_stage("cover_publish", scope="intra_wave", pair=0):
        accumulators = _wg_wave_grid_dot_two_half_range(
            current_a,
            current_b,
            second_a,
            second_b,
            accumulators,
            operand_count + pre_publish_count,
            publish_cover_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    tl.debug_barrier()
    next_values = tl.tuple([])
    second_mfma_start: tl.constexpr = (mfma_per_half - operand_count)
    with tlx.warp_pipeline_stage("read_and_cover_next_kh0", scope="intra_wave"):
        for read in tl.static_range(operand_count):
            if read < b_group_count:
                value = _wg_wave_grid_local_load_b_half(
                    b_local,
                    next_stage,
                    read,
                    0,
                    b_group_n,
                    dot_b,
                )
            else:
                value = _wg_wave_grid_local_load_a_half(
                    a_local,
                    next_stage,
                    read - b_group_count,
                    0,
                    a_group_m,
                    dot_a,
                )
            next_values += tl.tuple([value])
        accumulators = _wg_wave_grid_dot_column_major_range(
            second_a,
            second_b,
            accumulators,
            second_mfma_start,
            operand_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    next_b = tl.tuple([next_values[index] for index in range(b_group_count)])
    next_a = tl.tuple([next_values[b_group_count + index] for index in range(a_row_count)])

    return accumulators, future_a, future_b, next_a, next_b


@triton.jit
def _wg_wave_grid_pipeline_pair_rect_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    kb,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Advance two rectangular K64 blocks through static ping-pong stages."""
    for half in tl.static_range(2):
        state = _wg_wave_grid_pipeline_step_rect_k64(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2 + half,
            half,
            1 - half,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = state
    return accumulators, prefetched_a, prefetched_b, current_a, current_b


@triton.jit
def _wg_wave_grid_pipeline_step_vendor_square_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    current_stage,
    next_stage,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Vendor-shaped square K64 pipeline with bounded MFMA windows."""
    a_group_m: tl.constexpr = block_m // a_row_count
    b_group_n: tl.constexpr = block_n // b_group_count
    operand_count: tl.constexpr = a_row_count + b_group_count
    mfma_per_half: tl.constexpr = a_row_count * b_group_count
    final_publish: tl.constexpr = operand_count - 2
    second_read_start: tl.constexpr = (mfma_per_half - 3 * a_row_count + 1)
    tl.static_assert(block_k == 64)
    tl.static_assert(a_row_count == b_group_count)
    tl.static_assert(a_row_count >= 5 and a_row_count <= 8)
    tl.static_assert(a_group_m == 32)
    tl.static_assert(b_group_n == 32)

    # The first operand_count MFMAs cover all second-K32 LDS reads. Source
    # provides one independent window; the compiler derives its memory and
    # compute streams and reconstructs one bounded pair per operand.
    second_a = tl.tuple([])
    second_b = tl.tuple([])
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    with tlx.warp_pipeline_stage("read_and_consume_current", scope="intra_wave"):
        for mfma_index in tl.static_range(operand_count):
            if mfma_index == 0:
                second_value = _wg_grouped_square_k64_local_load_a_half(a_local, current_stage, 0, 1, a_row_count,
                                                                        dot_a)
            elif mfma_index == 1:
                second_value = _wg_grouped_square_k64_local_load_b_half(b_local, current_stage, 0, 1, b_group_count,
                                                                        dot_b)
            elif mfma_index <= a_row_count:
                second_value = _wg_grouped_square_k64_local_load_a_half(
                    a_local,
                    current_stage,
                    mfma_index - 1,
                    1,
                    a_row_count,
                    dot_a,
                )
            else:
                second_value = _wg_grouped_square_k64_local_load_b_half(
                    b_local,
                    current_stage,
                    mfma_index - a_row_count,
                    1,
                    b_group_count,
                    dot_b,
                )
            if mfma_index == 0 or (mfma_index >= 2 and mfma_index <= a_row_count):
                second_a += tl.tuple([second_value])
            else:
                second_b += tl.tuple([second_value])
        for mfma_index in tl.static_range(operand_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )

    # Publish every next-stage fragment and prefetch the following K64. The
    # compiler divides the independent MFMA range across the real DS-write and
    # VMEM anchors, replacing the old source-level fixed cover per fragment.
    with tlx.warp_pipeline_stage("publish_and_prefetch", scope="intra_wave", pair=0):
        for fragment in tl.static_range(operand_count - 1):
            _wg_grouped_square_k64_local_store_fragment(
                a_local,
                b_local,
                next_stage,
                prefetched_a,
                prefetched_b,
                fragment,
                b_group_count,
            )
            if fragment < b_group_count:
                future_value = _wg_wave_grid_global_load_b(
                    b_ptr,
                    pid_n,
                    future_kb,
                    fragment,
                    block_n,
                    block_k,
                    b_group_n,
                    stride_bk,
                    stride_bn,
                    n,
                )
            else:
                future_value = _wg_wave_grid_global_load_a(
                    a_ptr,
                    pid_m,
                    future_kb,
                    fragment - b_group_count,
                    a_group_m,
                    b_group_n,
                    block_m,
                    block_k,
                    stride_am,
                    stride_ak,
                    m,
                )
            if fragment == final_publish:
                _wg_grouped_square_k64_local_store_fragment(
                    a_local,
                    b_local,
                    next_stage,
                    prefetched_a,
                    prefetched_b,
                    operand_count - 1,
                    b_group_count,
                )
                last_future_value = _wg_wave_grid_global_load_a(
                    a_ptr,
                    pid_m,
                    future_kb,
                    a_row_count - 1,
                    a_group_m,
                    b_group_n,
                    block_m,
                    block_k,
                    stride_am,
                    stride_ak,
                    m,
                )
            if fragment < b_group_count:
                future_b += tl.tuple([future_value])
            else:
                future_a += tl.tuple([future_value])
            if fragment == final_publish:
                future_a += tl.tuple([last_future_value])
    with tlx.warp_pipeline_stage("cover_publish", scope="intra_wave", pair=0):
        accumulators = _wg_wave_grid_dot_two_half_range(
            current_a,
            current_b,
            second_a,
            second_b,
            accumulators,
            operand_count,
            2 * mfma_per_half - 5 * a_row_count + 1,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )

    # All next-stage stores are now visible.  Seventeen remaining MFMAs cover
    # the operand_count half-zero reads needed at the following iteration
    # entry. The compiler reconstructs one read/MFMA window per operand.
    tl.debug_barrier()
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    # Keep the operand order visible in source, but derive the repeated
    # one-read/one-MFMA windows in the compiler for this square path.
    with tlx.warp_pipeline_stage("read_and_cover_next_kh0", scope="intra_wave"):
        for read in tl.static_range(operand_count):
            if read == 0:
                next_value = _wg_grouped_square_k64_local_load_a_half(a_local, next_stage, 0, 0, a_row_count, dot_a)
            elif read == 1:
                next_value = _wg_grouped_square_k64_local_load_b_half(b_local, next_stage, 0, 0, b_group_count, dot_b)
            elif read <= a_row_count:
                next_value = _wg_grouped_square_k64_local_load_a_half(
                    a_local,
                    next_stage,
                    read - 1,
                    0,
                    a_row_count,
                    dot_a,
                )
            else:
                next_value = _wg_grouped_square_k64_local_load_b_half(
                    b_local,
                    next_stage,
                    read - a_row_count,
                    0,
                    b_group_count,
                    dot_b,
                )
            if read == 0 or (read >= 2 and read <= a_row_count):
                next_a += tl.tuple([next_value])
            else:
                next_b += tl.tuple([next_value])
        for read in tl.static_range(operand_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                second_a,
                second_b,
                accumulators,
                second_read_start + read,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )

    for tail_index in tl.static_range(second_read_start + operand_count, mfma_per_half):
        accumulators = _wg_wave_grid_dot_column_major_at(
            second_a,
            second_b,
            accumulators,
            tail_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators, future_a, future_b, next_a, next_b


@triton.jit
def _wg_wave_grid_compute_tile_rect_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    local_stages: tl.constexpr,
):
    """Compute one tile using the geometry-derived rectangular K64 pipeline."""
    tl.static_assert(block_k == 64)
    a_group_m: tl.constexpr = block_m // a_row_count
    k_blocks: tl.constexpr = k // block_k
    tl.static_assert(k_blocks % 2 == 0)

    _wg_wave_grid_local_store_all(
        a_local,
        b_local,
        0,
        first_a,
        first_b,
        a_row_count,
        b_group_count,
        block_k,
        b_group_n,
    )
    tl.debug_barrier()
    prefetched_a, prefetched_b = _wg_wave_grid_global_loads(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    current_a = tl.tuple(
        [_wg_wave_grid_local_load_a_half(a_local, 0, row, 0, a_group_m, dot_a) for row in range(a_row_count)])
    current_b = tl.tuple(
        [_wg_wave_grid_local_load_b_half(b_local, 0, group, 0, b_group_n, dot_b) for group in range(b_group_count)])
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])

    if local_stages == 2:
        for kb in tl.range(0, k_blocks - 2, 2, num_stages=1):
            state = _wg_wave_grid_pipeline_pair_rect_k64(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb,
                a_local,
                b_local,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
            )
            (
                accumulators,
                prefetched_a,
                prefetched_b,
                current_a,
                current_b,
            ) = state
    else:
        for kb in tl.range(0, k_blocks - 2, num_stages=1):
            state = _wg_wave_grid_pipeline_step_rect_k64(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2,
                0,
                0,
                a_local,
                b_local,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
            )
            (
                accumulators,
                prefetched_a,
                prefetched_b,
                current_a,
                current_b,
            ) = state

    mfma_per_half: tl.constexpr = a_row_count * b_group_count
    for mfma_index in tl.static_range(mfma_per_half):
        accumulators = _wg_wave_grid_dot_column_major_at(
            current_a,
            current_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    second_a = tl.tuple([
        _wg_wave_grid_local_load_a_half(a_local, (k_blocks - 2) % local_stages, row, 1, a_group_m, dot_a)
        for row in range(a_row_count)
    ])
    second_b = tl.tuple([
        _wg_wave_grid_local_load_b_half(b_local, (k_blocks - 2) % local_stages, group, 1, b_group_n, dot_b)
        for group in range(b_group_count)
    ])
    for mfma_index in tl.static_range(mfma_per_half):
        accumulators = _wg_wave_grid_dot_column_major_at(
            second_a,
            second_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    _wg_wave_grid_local_store_all(
        a_local,
        b_local,
        (k_blocks - 1) % local_stages,
        prefetched_a,
        prefetched_b,
        a_row_count,
        b_group_count,
        block_k,
        b_group_n,
    )
    tl.debug_barrier()
    return _wg_wave_grid_compute_stage_k64(
        a_local,
        b_local,
        (k_blocks - 1) % local_stages,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )


@triton.jit
def _wg_wave_grid_pgr2_stage_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    prefetch_future: tl.constexpr,
    double_buffered: tl.constexpr,
    late_read_count: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance a regular PGR2 grid through compiler-derived cover windows.

    One MFMA covers each second-half operand read.  After the old LDS tile is
    drained, all remaining independent work except one MFMA per next-stage
    operand is distributed uniformly over the local-store/global-load pairs.
    The final operand-count MFMAs then cover the next stage's first-half LDS
    reads.  This preserves the source-visible scheduling windows without
    baking one macro-tile's MFMA counts into the pipeline.
    """
    tl.static_assert(block_k == 64)
    a_group_m: tl.constexpr = block_m // a_row_count
    b_group_n: tl.constexpr = block_n // b_group_count
    operand_count: tl.constexpr = a_row_count + b_group_count
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    publish_count: tl.constexpr = (a_row_count + b_fragments_per_group * b_group_count)
    mfma_count: tl.constexpr = a_row_count * b_group_count
    cover_count: tl.constexpr = 2 * (mfma_count - operand_count)
    cover_windows: tl.constexpr = 2 * publish_count
    compact_publish: tl.constexpr = cover_count < cover_windows
    overlap_late_reads: tl.constexpr = (double_buffered and not compact_publish and late_read_count > 0)
    tl.static_assert(b_group_n == 32 or b_group_n == 64)
    tl.static_assert(mfma_count >= operand_count)
    tl.static_assert(not compact_publish or cover_count >= publish_count)
    tl.static_assert(late_read_count >= 0)
    tl.static_assert(late_read_count <= operand_count)
    tl.static_assert(late_read_count == 0 or overlap_late_reads)
    tl.static_assert(late_read_count <= publish_count)

    # Read the B groups first, then A, and immediately cover every LDS
    # read with one first-half MFMA.
    second_a = tl.tuple([])
    second_b = tl.tuple([])
    if compact_publish:
        accumulators = _wg_wave_grid_dot_column_major_at(
            current_a,
            current_b,
            accumulators,
            0,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize,
        )
        with tlx.warp_pipeline_stage("pgr2_compact_read_and_compute", scope="intra_wave"):
            for read in tl.static_range(operand_count):
                if read < b_group_count:
                    second_b += tl.tuple([
                        _wg_wave_grid_local_load_pgr2_b_half(
                            b_local,
                            read,
                            1,
                            b_group_n,
                            b_group_count,
                            dot_b,
                            current_stage,
                        )
                    ])
                else:
                    second_a += tl.tuple([
                        _wg_wave_grid_local_load_pgr2_a_half(
                            a_local,
                            read - b_group_count,
                            1,
                            a_group_m,
                            a_row_count,
                            dot_a,
                            current_stage,
                        )
                    ])
            for read in tl.static_range(operand_count):
                accumulators = _wg_wave_grid_dot_column_major_at(
                    current_a,
                    current_b,
                    accumulators,
                    read + 1,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
        for mfma_index in tl.static_range(operand_count + 1, mfma_count - 2):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
    else:
        early_read_count: tl.constexpr = (operand_count - late_read_count if overlap_late_reads else operand_count)
        with tlx.warp_pipeline_stage("pgr2_regular_read_and_compute", scope="intra_wave"):
            for read in tl.static_range(early_read_count):
                if read < b_group_count:
                    second_value = _wg_wave_grid_local_load_pgr2_b_half(
                        b_local,
                        read,
                        1,
                        b_group_n,
                        b_group_count,
                        dot_b,
                        current_stage,
                    )
                else:
                    second_value = _wg_wave_grid_local_load_pgr2_a_half(
                        a_local,
                        read - b_group_count,
                        1,
                        a_group_m,
                        a_row_count,
                        dot_a,
                        current_stage,
                    )
                if read < b_group_count:
                    second_b += tl.tuple([second_value])
                else:
                    second_a += tl.tuple([second_value])
            for read in tl.static_range(early_read_count):
                accumulators = _wg_wave_grid_dot_column_major_at(
                    current_a,
                    current_b,
                    accumulators,
                    read,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
    # Every old LDS value is now resident in registers.  Reuse the one-stage
    # allocation fragment by fragment and refill the global prefetch bank.
    if not double_buffered:
        tl.debug_barrier()
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    if compact_publish:
        # Keep every publication fragment in one memory window.  This lets
        # register allocation reuse the four VGPRs immediately after each LDS
        # write for the corresponding future global load, matching PGR2's
        # short prefetched-operand lifetime.
        with tlx.warp_pipeline_stage("pgr2_compact_publish", scope="intra_wave", pair=0):
            for publish in tl.static_range(publish_count):
                _wg_wave_grid_local_store_pgr2_fragment(
                    a_local,
                    b_local,
                    prefetched_a,
                    prefetched_b,
                    publish,
                    b_group_count,
                    a_row_count,
                    a_group_m,
                    b_group_n,
                    block_k,
                    next_stage,
                )
                if prefetch_future:
                    if publish < b_fragments_per_group * b_group_count:
                        future_b += tl.tuple([
                            _wg_wave_grid_global_load_pgr2_b_half(
                                b_ptr,
                                pid_n,
                                future_kb,
                                publish // b_fragments_per_group,
                                publish % b_fragments_per_group,
                                block_m,
                                block_n,
                                block_k,
                                b_group_n,
                                stride_bk,
                                stride_bn,
                                m,
                                n,
                            )
                        ])
                    else:
                        future_a += tl.tuple([
                            _wg_wave_grid_global_load_a(
                                a_ptr,
                                pid_m,
                                future_kb,
                                publish - (b_fragments_per_group * b_group_count),
                                a_group_m,
                                b_group_n,
                                block_m,
                                block_k,
                                stride_am,
                                stride_ak,
                                m,
                            )
                        ])
        with tlx.warp_pipeline_stage("pgr2_compact_cover_publish", scope="intra_wave", pair=0):
            for window in tl.static_range(4):
                accumulators = _wg_wave_grid_dot_two_half_range(
                    current_a,
                    current_b,
                    second_a,
                    second_b,
                    accumulators,
                    mfma_count - 2 + window,
                    1,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
    else:
        # Source preserves the stage-reuse boundary and the independent work
        # streams. The scheduler derives per-store/load cover from the actual
        # lowering instead of requiring one source pair per fragment.
        publish_start: tl.constexpr = (late_read_count if overlap_late_reads else 0)
        publish_cover_first: tl.constexpr = (mfma_count if overlap_late_reads else operand_count)
        publish_cover_count: tl.constexpr = (mfma_count - operand_count if overlap_late_reads else cover_count)
        if overlap_late_reads:
            late_cover_count: tl.constexpr = mfma_count - early_read_count
            for late in tl.static_range(late_read_count):
                with tlx.warp_pipeline_stage(
                        "pgr2_late_read_publish",
                        scope="intra_wave",
                        pair=1,
                ):
                    if early_read_count + late < b_group_count:
                        second_value = _wg_wave_grid_local_load_pgr2_b_half(
                            b_local,
                            early_read_count + late,
                            1,
                            b_group_n,
                            b_group_count,
                            dot_b,
                            current_stage,
                        )
                    else:
                        second_value = _wg_wave_grid_local_load_pgr2_a_half(
                            a_local,
                            early_read_count + late - b_group_count,
                            1,
                            a_group_m,
                            a_row_count,
                            dot_a,
                            current_stage,
                        )
                    if early_read_count + late < b_group_count:
                        second_b += tl.tuple([second_value])
                    else:
                        second_a += tl.tuple([second_value])
                    _wg_wave_grid_local_store_pgr2_fragment(
                        a_local,
                        b_local,
                        prefetched_a,
                        prefetched_b,
                        late,
                        b_group_count,
                        a_row_count,
                        a_group_m,
                        b_group_n,
                        block_k,
                        next_stage,
                    )
                    if prefetch_future:
                        if late < b_fragments_per_group * b_group_count:
                            future_b += tl.tuple([
                                _wg_wave_grid_global_load_pgr2_b_half(
                                    b_ptr,
                                    pid_n,
                                    future_kb,
                                    late // b_fragments_per_group,
                                    late % b_fragments_per_group,
                                    block_m,
                                    block_n,
                                    block_k,
                                    b_group_n,
                                    stride_bk,
                                    stride_bn,
                                    m,
                                    n,
                                )
                            ])
                        else:
                            future_a += tl.tuple([
                                _wg_wave_grid_global_load_a(
                                    a_ptr,
                                    pid_m,
                                    future_kb,
                                    late - (b_fragments_per_group * b_group_count),
                                    a_group_m,
                                    b_group_n,
                                    block_m,
                                    block_k,
                                    stride_am,
                                    stride_ak,
                                    m,
                                )
                            ])
                with tlx.warp_pipeline_stage(
                        "pgr2_late_read_cover",
                        scope="intra_wave",
                        pair=1,
                ):
                    accumulators = _wg_wave_grid_dot_two_half_range(
                        current_a,
                        current_b,
                        second_a,
                        second_b,
                        accumulators,
                        early_read_count + late * late_cover_count // late_read_count,
                        (late + 1) * late_cover_count // late_read_count - late * late_cover_count // late_read_count,
                        a_row_count,
                        b_group_count,
                        mma,
                        dot_a,
                        dot_b,
                        initialize,
                    )
        with tlx.warp_pipeline_stage(
                "pgr2_regular_publish_and_prefetch",
                scope="intra_wave",
                pair=2,
        ):
            for publish in tl.static_range(publish_start, publish_count):
                _wg_wave_grid_local_store_pgr2_fragment(
                    a_local,
                    b_local,
                    prefetched_a,
                    prefetched_b,
                    publish,
                    b_group_count,
                    a_row_count,
                    a_group_m,
                    b_group_n,
                    block_k,
                    next_stage,
                )
                if prefetch_future:
                    if publish < b_fragments_per_group * b_group_count:
                        future_b += tl.tuple([
                            _wg_wave_grid_global_load_pgr2_b_half(
                                b_ptr,
                                pid_n,
                                future_kb,
                                publish // b_fragments_per_group,
                                publish % b_fragments_per_group,
                                block_m,
                                block_n,
                                block_k,
                                b_group_n,
                                stride_bk,
                                stride_bn,
                                m,
                                n,
                            )
                        ])
                    else:
                        future_a += tl.tuple([
                            _wg_wave_grid_global_load_a(
                                a_ptr,
                                pid_m,
                                future_kb,
                                publish - b_fragments_per_group * b_group_count,
                                a_group_m,
                                b_group_n,
                                block_m,
                                block_k,
                                stride_am,
                                stride_ak,
                                m,
                            )
                        ])
        with tlx.warp_pipeline_stage(
                "pgr2_regular_cover_publish",
                scope="intra_wave",
                pair=2,
        ):
            accumulators = _wg_wave_grid_dot_two_half_range(
                current_a,
                current_b,
                second_a,
                second_b,
                accumulators,
                publish_cover_first,
                publish_cover_count,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )

    # K(t+1) is visible in LDS.  Its first-half reads are covered by the final
    # operand-count updates from K(t)'s second half.
    tl.debug_barrier()
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    final_first: tl.constexpr = (2 if compact_publish else 2 * mfma_count - operand_count)
    if compact_publish:
        accumulators = _wg_wave_grid_dot_column_major_at(
            second_a,
            second_b,
            accumulators,
            final_first,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    if compact_publish:
        with tlx.warp_pipeline_stage("pgr2_compact_read_and_compute_next", scope="intra_wave"):
            for read in tl.static_range(operand_count):
                if read < b_group_count:
                    next_b += tl.tuple([
                        _wg_wave_grid_local_load_pgr2_b_half(
                            b_local,
                            read,
                            0,
                            b_group_n,
                            b_group_count,
                            dot_b,
                            next_stage,
                        )
                    ])
                else:
                    next_a += tl.tuple([
                        _wg_wave_grid_local_load_pgr2_a_half(
                            a_local,
                            read - b_group_count,
                            0,
                            a_group_m,
                            a_row_count,
                            dot_a,
                            next_stage,
                        )
                    ])
            for read in tl.static_range(operand_count):
                accumulators = _wg_wave_grid_dot_column_major_at(
                    second_a,
                    second_b,
                    accumulators,
                    final_first + read + 1,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                )
    else:
        with tlx.warp_pipeline_stage("pgr2_regular_read_and_compute_next", scope="intra_wave"):
            for read in tl.static_range(operand_count):
                if read < b_group_count:
                    next_value = _wg_wave_grid_local_load_pgr2_b_half(
                        b_local,
                        read,
                        0,
                        b_group_n,
                        b_group_count,
                        dot_b,
                        next_stage,
                    )
                else:
                    next_value = _wg_wave_grid_local_load_pgr2_a_half(
                        a_local,
                        read - b_group_count,
                        0,
                        a_group_m,
                        a_row_count,
                        dot_a,
                        next_stage,
                    )
                if read < b_group_count:
                    next_b += tl.tuple([next_value])
                else:
                    next_a += tl.tuple([next_value])
            for read in tl.static_range(operand_count):
                accumulators = _wg_wave_grid_dot_two_half_range(
                    current_a,
                    current_b,
                    second_a,
                    second_b,
                    accumulators,
                    final_first + read,
                    1,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                )
    if compact_publish:
        for mfma_index in tl.static_range(final_first + operand_count + 1, mfma_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                second_a,
                second_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
    return accumulators, future_a, future_b, next_a, next_b


@triton.jit
def _wg_wave_grid_pgr2_stage_pair_k64(
    kb,
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    late_read_count: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume two K64 blocks using distinct, statically selected banks."""
    state = _wg_wave_grid_pgr2_stage_k64(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        kb + 2,
        a_local,
        b_local,
        0,
        1,
        current_a,
        current_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        True,
        True,
        late_read_count,
        initialize,
    )
    (
        accumulators,
        prefetched_a,
        prefetched_b,
        current_a,
        current_b,
    ) = state
    return _wg_wave_grid_pgr2_stage_k64(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        kb + 3,
        a_local,
        b_local,
        1,
        0,
        current_a,
        current_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        True,
        True,
        late_read_count,
        False,
    )


@triton.jit
def _wg_wave_grid_consume_final_pgr2_k64(
    a_local,
    b_local,
    stage,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
):
    """Consume the final K64 after its first operand bank was preloaded."""
    second_a = tl.tuple([
        _wg_wave_grid_local_load_pgr2_a_half(a_local, row, 1, a_group_m, a_row_count, dot_a, stage)
        for row in range(a_row_count)
    ])
    second_b = tl.tuple([
        _wg_wave_grid_local_load_pgr2_b_half(b_local, group, 1, b_group_n, b_group_count, dot_b, stage)
        for group in range(b_group_count)
    ])
    mfma_count: tl.constexpr = a_row_count * b_group_count
    for mfma_index in tl.static_range(mfma_count):
        accumulators = _wg_wave_grid_dot_column_major_at(
            current_a,
            current_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    for mfma_index in tl.static_range(mfma_count):
        accumulators = _wg_wave_grid_dot_column_major_at(
            second_a,
            second_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators


@triton.jit
def _wg_wave_grid_compute_tile_pgr2_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    local_stages: tl.constexpr,
    late_read_count: tl.constexpr,
):
    """Run the parameterized PGR2 two-operand-bank pipeline."""
    a_group_m: tl.constexpr = block_m // a_row_count
    b_group_n: tl.constexpr = block_n // b_group_count
    k_blocks: tl.constexpr = k // block_k
    tl.static_assert(k_blocks >= 3)
    tl.static_assert(local_stages == 1 or local_stages == 2)
    double_buffered: tl.constexpr = local_stages == 2
    # The one-bank pipeline naturally handles K0/K1/K2.  The paired
    # double-buffered prologue consumes two stages at once and therefore needs
    # one additional K64 block.
    tl.static_assert(not double_buffered or k_blocks >= 4)
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    for fragment in tl.static_range(b_fragments_per_group * b_group_count + a_row_count):
        _wg_wave_grid_local_store_pgr2_fragment(
            a_local,
            b_local,
            first_a,
            first_b,
            fragment,
            b_group_count,
            a_row_count,
            a_group_m,
            b_group_n,
            block_k,
        )
    tl.debug_barrier()
    current_a = tl.tuple([
        _wg_wave_grid_local_load_pgr2_a_half(a_local, row, 0, a_group_m, a_row_count, dot_a)
        for row in range(a_row_count)
    ])
    current_b = tl.tuple([
        _wg_wave_grid_local_load_pgr2_b_half(b_local, group, 0, b_group_n, b_group_count, dot_b)
        for group in range(b_group_count)
    ])
    prefetched_a, prefetched_b = _wg_wave_grid_global_loads_pgr2(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])

    # The first K64 initializes every persistent accumulator directly in its
    # MFMA destination.  Peeling this stage avoids materializing a zero tensor
    # through one ACCVGPR write per lane value before the chain begins.
    if double_buffered:
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = _wg_wave_grid_pgr2_stage_pair_k64(
            0,
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            late_read_count,
            True,
        )
        # Pair only while both future blocks are in bounds.  For an odd
        # number of K64 blocks, the old upper bound admitted a final pair
        # whose second stage prefetched K[k_blocks], then consumed it as the
        # tail.  Leave three blocks for the odd tail and advance one stage
        # below before entering the common two-block epilogue.
        paired_loop_end: tl.constexpr = (k_blocks - 2 - (k_blocks % 2))
        for kb in tl.range(2, paired_loop_end, 2, num_stages=1):
            (
                accumulators,
                prefetched_a,
                prefetched_b,
                current_a,
                current_b,
            ) = _wg_wave_grid_pgr2_stage_pair_k64(
                kb,
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                a_local,
                b_local,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                late_read_count,
                False,
            )
        if k_blocks % 2 != 0:
            (
                accumulators,
                prefetched_a,
                prefetched_b,
                current_a,
                current_b,
            ) = _wg_wave_grid_pgr2_stage_k64(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                k_blocks - 1,
                a_local,
                b_local,
                (k_blocks - 3) % 2,
                (k_blocks - 2) % 2,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                True,
                True,
                late_read_count,
                False,
            )
    else:
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = _wg_wave_grid_pgr2_stage_k64(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            2,
            a_local,
            b_local,
            0,
            0,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            True,
            False,
            late_read_count,
            True,
        )
        for kb in tl.range(1, k_blocks - 2, num_stages=1):
            (
                accumulators,
                prefetched_a,
                prefetched_b,
                current_a,
                current_b,
            ) = _wg_wave_grid_pgr2_stage_k64(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2,
                a_local,
                b_local,
                0,
                0,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                True,
                False,
                late_read_count,
                False,
            )

    (
        accumulators,
        _,
        _,
        current_a,
        current_b,
    ) = _wg_wave_grid_pgr2_stage_k64(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        a_local,
        b_local,
        ((k_blocks - 2) % 2 if double_buffered else 0),
        ((k_blocks - 1) % 2 if double_buffered else 0),
        current_a,
        current_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        False,
        double_buffered,
        late_read_count,
        False,
    )
    return _wg_wave_grid_consume_final_pgr2_k64(
        a_local,
        b_local,
        ((k_blocks - 1) % 2 if double_buffered else 0),
        current_a,
        current_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )


@triton.jit
def _wg_wave_grid_global_loads_pgr2_k128(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    macro,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
):
    """Load one logical K128 macro as two bank-safe K64 fragments."""
    first_a, first_b = _wg_wave_grid_global_loads_pgr2(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2 * macro,
        block_m,
        block_n,
        64,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    second_a, second_b = _wg_wave_grid_global_loads_pgr2(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2 * macro + 1,
        block_m,
        block_n,
        64,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    return first_a + second_a, first_b + second_b


@triton.jit
def _wg_wave_grid_local_store_pgr2_k128_fragment(
    a_local,
    b_local,
    prefetched_a,
    prefetched_b,
    fragment: tl.constexpr,
    b_group_count: tl.constexpr,
    a_row_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
):
    """Publish one K64-width fragment of a logical K128 macro tile."""
    b_fragment_count: tl.constexpr = b_group_count * (b_group_n // 32)
    a_fragment_count: tl.constexpr = a_row_count
    if fragment < 2 * b_fragment_count:
        k_half: tl.constexpr = fragment // b_fragment_count
        local_fragment: tl.constexpr = fragment % b_fragment_count
        _wg_wave_grid_local_store_pgr2_fragment(
            a_local,
            b_local,
            prefetched_a,
            prefetched_b,
            local_fragment,
            b_group_count,
            a_row_count,
            a_group_m,
            b_group_n,
            64,
            k_half,
            0,
            k_half * b_fragment_count,
        )
    else:
        a_fragment: tl.constexpr = fragment - 2 * b_fragment_count
        k_half: tl.constexpr = a_fragment // a_fragment_count
        local_fragment: tl.constexpr = (b_fragment_count + a_fragment % a_fragment_count)
        _wg_wave_grid_local_store_pgr2_fragment(
            a_local,
            b_local,
            prefetched_a,
            prefetched_b,
            local_fragment,
            b_group_count,
            a_row_count,
            a_group_m,
            b_group_n,
            64,
            k_half,
            k_half * a_fragment_count,
            0,
        )


@triton.jit
def _wg_wave_grid_local_load_pgr2_quarter_k128(
    a_local,
    b_local,
    quarter: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Read one K32 quarter from a one-stage K128 PGR2 allocation."""
    return (
        tl.tuple([
            _wg_wave_grid_local_load_pgr2_a_half(
                a_local,
                row,
                quarter % 2,
                a_group_m,
                a_row_count,
                dot_a,
                quarter // 2,
            ) for row in range(a_row_count)
        ]),
        tl.tuple([
            _wg_wave_grid_local_load_pgr2_b_half(
                b_local,
                group,
                quarter % 2,
                b_group_n,
                b_group_count,
                dot_b,
                quarter // 2,
            ) for group in range(b_group_count)
        ]),
    )


@triton.jit
def _wg_wave_grid_read_quarter_and_compute_k128(
    a_local,
    b_local,
    quarter: tl.constexpr,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume one K32 quarter while reading the following LDS quarter."""
    operand_count: tl.constexpr = a_row_count + b_group_count
    mfma_count: tl.constexpr = a_row_count * b_group_count
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    with tlx.warp_pipeline_stage(
            "pgr2_k128_read_and_compute",
            scope="intra_wave",
    ):
        for read in tl.static_range(operand_count):
            if read < b_group_count:
                next_b += tl.tuple([
                    _wg_wave_grid_local_load_pgr2_b_half(
                        b_local,
                        read,
                        quarter % 2,
                        b_group_n,
                        b_group_count,
                        dot_b,
                        quarter // 2,
                    )
                ])
            else:
                next_a += tl.tuple([
                    _wg_wave_grid_local_load_pgr2_a_half(
                        a_local,
                        read - b_group_count,
                        quarter % 2,
                        a_group_m,
                        a_row_count,
                        dot_a,
                        quarter // 2,
                    )
                ])
        for mfma_index in tl.static_range(mfma_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
    return accumulators, next_a, next_b


@triton.jit
def _wg_wave_grid_read_two_quarters_and_compute_k128(
    a_local,
    b_local,
    first_quarter: tl.constexpr,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume one K32 quarter while reading two successor quarters."""
    operand_count: tl.constexpr = a_row_count + b_group_count
    mfma_count: tl.constexpr = a_row_count * b_group_count
    next_q0_a = tl.tuple([])
    next_q0_b = tl.tuple([])
    next_q1_a = tl.tuple([])
    next_q1_b = tl.tuple([])
    with tlx.warp_pipeline_stage(
            "pgr2_k128_read_two_and_compute",
            scope="intra_wave",
    ):
        for read in tl.static_range(2 * operand_count):
            if read % operand_count < b_group_count:
                next_value = _wg_wave_grid_local_load_pgr2_b_half(
                    b_local,
                    read % operand_count,
                    (first_quarter + read // operand_count) % 2,
                    b_group_n,
                    b_group_count,
                    dot_b,
                    (first_quarter + read // operand_count) // 2,
                )
                if read < operand_count:
                    next_q0_b += tl.tuple([next_value])
                else:
                    next_q1_b += tl.tuple([next_value])
            else:
                next_value = _wg_wave_grid_local_load_pgr2_a_half(
                    a_local,
                    read % operand_count - b_group_count,
                    (first_quarter + read // operand_count) % 2,
                    a_group_m,
                    a_row_count,
                    dot_a,
                    (first_quarter + read // operand_count) // 2,
                )
                if read < operand_count:
                    next_q0_a += tl.tuple([next_value])
                else:
                    next_q1_a += tl.tuple([next_value])
        for mfma_index in tl.static_range(mfma_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
    return accumulators, next_q0_a, next_q0_b, next_q1_a, next_q1_b


@triton.jit
def _wg_wave_grid_pgr2_stage_k128(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_macro,
    a_local,
    b_local,
    current_q0_a,
    current_q0_b,
    current_q1_a,
    current_q1_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    prefetch_future: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance one register-prefetched K128 stage through one LDS bank."""
    a_group_m: tl.constexpr = block_m // a_row_count
    b_group_n: tl.constexpr = block_n // b_group_count
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    b_fragment_count: tl.constexpr = (b_fragments_per_group * b_group_count)
    publish_count: tl.constexpr = 2 * (b_fragment_count + a_row_count)

    accumulators, q2_a, q2_b = (_wg_wave_grid_read_quarter_and_compute_k128(
        a_local,
        b_local,
        2,
        current_q0_a,
        current_q0_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        initialize,
    ))
    accumulators, q3_a, q3_b = (_wg_wave_grid_read_quarter_and_compute_k128(
        a_local,
        b_local,
        3,
        current_q1_a,
        current_q1_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    ))

    # The last two K32 quarters of the old macro tile are now resident in
    # VGPRs. Refill both physical K64 stages while maintaining a rolling window
    # of 16 machine-width global loads, matching the selected hipBLASLt path.
    tl.debug_barrier()
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    with tlx.warp_pipeline_stage(
            "pgr2_k128_publish_and_prefetch",
            scope="intra_wave",
            pair=0,
    ):
        for publish in tl.static_range(publish_count):
            _wg_wave_grid_local_store_pgr2_k128_fragment(
                a_local,
                b_local,
                prefetched_a,
                prefetched_b,
                publish,
                b_group_count,
                a_row_count,
                a_group_m,
                b_group_n,
            )
            if prefetch_future:
                if publish < 2 * b_fragment_count:
                    future_b += tl.tuple([
                        _wg_wave_grid_global_load_pgr2_b_half(
                            b_ptr,
                            pid_n,
                            2 * future_macro + publish // b_fragment_count,
                            (publish % b_fragment_count) // b_fragments_per_group,
                            (publish % b_fragment_count) % b_fragments_per_group,
                            block_m,
                            block_n,
                            64,
                            b_group_n,
                            stride_bk,
                            stride_bn,
                            m,
                            n,
                        )
                    ])
                else:
                    future_a += tl.tuple([
                        _wg_wave_grid_global_load_a(
                            a_ptr,
                            pid_m,
                            2 * future_macro + (publish - 2 * b_fragment_count) // a_row_count,
                            (publish - 2 * b_fragment_count) % a_row_count,
                            a_group_m,
                            b_group_n,
                            block_m,
                            64,
                            stride_am,
                            stride_ak,
                            m,
                            True,
                        )
                    ])
    with tlx.warp_pipeline_stage(
            "pgr2_k128_cover_publish",
            scope="intra_wave",
            pair=0,
    ):
        for mfma_index in tl.static_range(a_row_count * b_group_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                q2_a,
                q2_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
    tl.debug_barrier()
    (
        accumulators,
        next_q0_a,
        next_q0_b,
        next_q1_a,
        next_q1_b,
    ) = _wg_wave_grid_read_two_quarters_and_compute_k128(
        a_local,
        b_local,
        0,
        q3_a,
        q3_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    )
    return (
        accumulators,
        future_a,
        future_b,
        next_q0_a,
        next_q0_b,
        next_q1_a,
        next_q1_b,
    )


@triton.jit
def _wg_wave_grid_compute_tile_pgr2_k128(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    """Run a one-LDS-stage K128 pipeline with register PGR2 prefetching."""
    block_k: tl.constexpr = 128
    a_group_m: tl.constexpr = block_m // a_row_count
    b_group_n: tl.constexpr = block_n // b_group_count
    macro_blocks: tl.constexpr = k // block_k
    tl.static_assert(k % block_k == 0 and macro_blocks >= 3)
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    for fragment in tl.static_range(2 * (b_fragments_per_group * b_group_count + a_row_count)):
        _wg_wave_grid_local_store_pgr2_k128_fragment(
            a_local,
            b_local,
            first_a,
            first_b,
            fragment,
            b_group_count,
            a_row_count,
            a_group_m,
            b_group_n,
        )
    tl.debug_barrier()
    current_q0_a, current_q0_b = (_wg_wave_grid_local_load_pgr2_quarter_k128(
        a_local,
        b_local,
        0,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        dot_a,
        dot_b,
    ))
    current_q1_a, current_q1_b = _wg_wave_grid_local_load_pgr2_quarter_k128(
        a_local,
        b_local,
        1,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        dot_a,
        dot_b,
    )
    prefetched_a, prefetched_b = _wg_wave_grid_global_loads_pgr2_k128(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])

    (
        accumulators,
        prefetched_a,
        prefetched_b,
        current_q0_a,
        current_q0_b,
        current_q1_a,
        current_q1_b,
    ) = _wg_wave_grid_pgr2_stage_k128(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        2,
        a_local,
        b_local,
        current_q0_a,
        current_q0_b,
        current_q1_a,
        current_q1_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        True,
        True,
    )
    for macro in tl.range(1, macro_blocks - 2, num_stages=1):
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_q0_a,
            current_q0_b,
            current_q1_a,
            current_q1_b,
        ) = _wg_wave_grid_pgr2_stage_k128(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            macro + 2,
            a_local,
            b_local,
            current_q0_a,
            current_q0_b,
            current_q1_a,
            current_q1_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            a_row_count,
            b_group_count,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            True,
            False,
        )
    (
        accumulators,
        _,
        _,
        current_q0_a,
        current_q0_b,
        current_q1_a,
        current_q1_b,
    ) = _wg_wave_grid_pgr2_stage_k128(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        a_local,
        b_local,
        current_q0_a,
        current_q0_b,
        current_q1_a,
        current_q1_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        False,
        False,
    )
    accumulators, q2_a, q2_b = (_wg_wave_grid_read_quarter_and_compute_k128(
        a_local,
        b_local,
        2,
        current_q0_a,
        current_q0_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    ))
    accumulators, q3_a, q3_b = (_wg_wave_grid_read_quarter_and_compute_k128(
        a_local,
        b_local,
        3,
        current_q1_a,
        current_q1_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
    ))
    mfma_count: tl.constexpr = a_row_count * b_group_count
    accumulators = _wg_wave_grid_dot_column_major_range(
        q2_a,
        q2_b,
        accumulators,
        0,
        mfma_count,
        a_row_count,
        b_group_count,
        mma,
        dot_a,
        dot_b,
    )
    accumulators = _wg_wave_grid_dot_column_major_range(
        q3_a,
        q3_b,
        accumulators,
        0,
        mfma_count,
        a_row_count,
        b_group_count,
        mma,
        dot_a,
        dot_b,
    )
    return accumulators


@triton.jit
def _wg_wave_grid_pgr2_stage_k32(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    prefetch_future: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance one K32 while keeping only the smaller A set resident.

    B(K(t)) stays in LDS and is read one column at a time.  Each B read is
    consumed immediately by ``a_row_count`` MFMAs, avoiding a live range for
    all B columns.  The same windows publish K(t+1) and prefetch K(t+2).
    The final B column is reserved to cover the next stage's A reads.
    """
    tl.static_assert(block_k == 32)
    tl.static_assert(a_row_count <= b_group_count)
    future_a = tl.tuple([])
    future_b = tl.tuple([])

    # Read B0 before replacing its LDS slot.  Subsequent B columns are read
    # one window ahead, before their own slots are overwritten.
    current_b_column = _wg_wave_grid_local_load_pgr2_b_half(b_local, 0, 0, b_group_n, b_group_count, dot_b,
                                                            current_stage)
    for column in tl.static_range(b_group_count):
        if column + 1 < b_group_count:
            with tlx.warp_pipeline_stage("pgr2_k32_cover_publish", scope="intra_wave", pair=0):
                accumulators = _wg_wave_grid_dot_column_with_preloaded_a(
                    current_a,
                    current_b_column,
                    accumulators,
                    column,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
        else:
            # Keep the final publish window paired, but reserve the other
            # final-column MFMAs to cover the next A operand reads below.
            with tlx.warp_pipeline_stage("pgr2_k32_cover_final_publish", scope="intra_wave", pair=0):
                accumulators = _wg_wave_grid_dot_at(
                    current_a[0],
                    current_b_column,
                    accumulators,
                    0,
                    column,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
        with tlx.warp_pipeline_stage("pgr2_k32_publish_and_read", scope="intra_wave", pair=0):
            _wg_wave_grid_local_store_pgr2_fragment(
                a_local,
                b_local,
                prefetched_a,
                prefetched_b,
                column,
                b_group_count,
                a_row_count,
                block_m // a_row_count,
                b_group_n,
                block_k,
                next_stage,
            )
            if column < a_row_count:
                _wg_wave_grid_local_store_pgr2_fragment(
                    a_local,
                    b_local,
                    prefetched_a,
                    prefetched_b,
                    b_group_count + column,
                    b_group_count,
                    a_row_count,
                    block_m // a_row_count,
                    b_group_n,
                    block_k,
                    next_stage,
                )
            if prefetch_future:
                future_b_value = _wg_wave_grid_global_load_b(
                    b_ptr,
                    pid_n,
                    future_kb,
                    column,
                    block_n,
                    block_k,
                    b_group_n,
                    stride_bk,
                    stride_bn,
                    n,
                )
                if column < a_row_count:
                    future_a_value = _wg_wave_grid_global_load_a(
                        a_ptr,
                        pid_m,
                        future_kb,
                        column,
                        block_m // a_row_count,
                        b_group_n,
                        block_m,
                        block_k,
                        stride_am,
                        stride_ak,
                        m,
                    )
            if column + 1 < b_group_count:
                next_b_column = _wg_wave_grid_local_load_pgr2_b_half(
                    b_local,
                    column + 1,
                    0,
                    b_group_n,
                    b_group_count,
                    dot_b,
                    current_stage,
                )
        if prefetch_future:
            future_b += tl.tuple([future_b_value])
            if column < a_row_count:
                future_a += tl.tuple([future_a_value])
        if column + 1 < b_group_count:
            current_b_column = next_b_column

    tl.debug_barrier()
    # Row zero's MFMA was used to cover the final publication window.
    next_a = tl.tuple(
        [_wg_wave_grid_local_load_pgr2_a_half(
            a_local,
            0,
            0,
            block_m // a_row_count,
            a_row_count,
            dot_a,
            next_stage,
        )])
    for row in tl.static_range(1, a_row_count):
        with tlx.warp_pipeline_stage("pgr2_k32_read_next_a", scope="intra_wave", pair=1):
            next_a_value = _wg_wave_grid_local_load_pgr2_a_half(
                a_local,
                row,
                0,
                block_m // a_row_count,
                a_row_count,
                dot_a,
                next_stage,
            )
        with tlx.warp_pipeline_stage("pgr2_k32_cover_a_read", scope="intra_wave", pair=1):
            accumulators = _wg_wave_grid_dot_at(
                current_a[row],
                current_b_column,
                accumulators,
                row,
                b_group_count - 1,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
        next_a += tl.tuple([next_a_value])
    return accumulators, future_a, future_b, next_a, tl.tuple([])


@triton.jit
def _wg_wave_grid_pgr2_stage_k32_b_resident(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    prefetch_future: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance K32 while keeping the smaller B operand set resident."""
    tl.static_assert(block_k == 32)
    tl.static_assert(b_group_count < a_row_count)
    a_group_m: tl.constexpr = block_m // a_row_count
    future_a = tl.tuple([])
    future_b = tl.tuple([])

    # Stream one A row from LDS and immediately consume it across the resident
    # B columns.  This is the transpose of the A-resident path above and keeps
    # the smaller operand live for the whole K32 update.
    current_a_row = _wg_wave_grid_local_load_pgr2_a_half(a_local, 0, 0, a_group_m, a_row_count, dot_a, current_stage)
    for row in tl.static_range(a_row_count):
        if row + 1 < a_row_count:
            with tlx.warp_pipeline_stage(
                    "pgr2_k32_b_resident_cover_publish",
                    scope="intra_wave",
                    pair=0,
            ):
                accumulators = _wg_wave_grid_dot_row(
                    current_a_row,
                    current_b,
                    accumulators,
                    row,
                    0,
                    b_group_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
        else:
            # Reserve the remaining final-row MFMAs to cover next-stage B
            # reads after the stage handoff.
            with tlx.warp_pipeline_stage(
                    "pgr2_k32_b_resident_cover_final_publish",
                    scope="intra_wave",
                    pair=0,
            ):
                accumulators = _wg_wave_grid_dot_at(
                    current_a_row,
                    current_b[0],
                    accumulators,
                    row,
                    0,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )
        with tlx.warp_pipeline_stage(
                "pgr2_k32_b_resident_publish_and_read",
                scope="intra_wave",
                pair=0,
        ):
            if row < b_group_count:
                _wg_wave_grid_local_store_pgr2_fragment(
                    a_local,
                    b_local,
                    prefetched_a,
                    prefetched_b,
                    row,
                    b_group_count,
                    a_row_count,
                    a_group_m,
                    b_group_n,
                    block_k,
                    next_stage,
                )
            _wg_wave_grid_local_store_pgr2_fragment(
                a_local,
                b_local,
                prefetched_a,
                prefetched_b,
                b_group_count + row,
                b_group_count,
                a_row_count,
                a_group_m,
                b_group_n,
                block_k,
                next_stage,
            )
            if prefetch_future:
                if row < b_group_count:
                    future_b_value = _wg_wave_grid_global_load_b(
                        b_ptr,
                        pid_n,
                        future_kb,
                        row,
                        block_n,
                        block_k,
                        b_group_n,
                        stride_bk,
                        stride_bn,
                        n,
                    )
                future_a_value = _wg_wave_grid_global_load_a(
                    a_ptr,
                    pid_m,
                    future_kb,
                    row,
                    a_group_m,
                    b_group_n,
                    block_m,
                    block_k,
                    stride_am,
                    stride_ak,
                    m,
                )
            if row + 1 < a_row_count:
                next_a_row = _wg_wave_grid_local_load_pgr2_a_half(
                    a_local,
                    row + 1,
                    0,
                    a_group_m,
                    a_row_count,
                    dot_a,
                    current_stage,
                )
        if prefetch_future:
            if row < b_group_count:
                future_b += tl.tuple([future_b_value])
            future_a += tl.tuple([future_a_value])
        if row + 1 < a_row_count:
            current_a_row = next_a_row

    tl.debug_barrier()
    next_b = tl.tuple(
        [_wg_wave_grid_local_load_pgr2_b_half(b_local, 0, 0, b_group_n, b_group_count, dot_b, next_stage)])
    for column in tl.static_range(1, b_group_count):
        with tlx.warp_pipeline_stage(
                "pgr2_k32_b_resident_read_next_b",
                scope="intra_wave",
                pair=1,
        ):
            next_b_value = _wg_wave_grid_local_load_pgr2_b_half(
                b_local,
                column,
                0,
                b_group_n,
                b_group_count,
                dot_b,
                next_stage,
            )
        with tlx.warp_pipeline_stage(
                "pgr2_k32_b_resident_cover_b_read",
                scope="intra_wave",
                pair=1,
        ):
            accumulators = _wg_wave_grid_dot_at(
                current_a_row,
                current_b[column],
                accumulators,
                a_row_count - 1,
                column,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
        next_b += tl.tuple([next_b_value])
    return accumulators, future_a, future_b, tl.tuple([]), next_b


@triton.jit
def _wg_wave_grid_pgr2_stage_k32_all_resident(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    prefetch_future: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Advance K32 after retiring every read from one reusable LDS bank."""
    tl.static_assert(block_k == 32)
    a_group_m: tl.constexpr = block_m // a_row_count
    a_pair_count: tl.constexpr = (a_row_count + 1) // 2
    publish_count: tl.constexpr = a_pair_count + b_group_count
    read_count: tl.constexpr = a_row_count + b_group_count
    mfma_count: tl.constexpr = a_row_count * b_group_count
    publish_cover: tl.constexpr = mfma_count - read_count
    tl.static_assert(publish_cover >= 0)
    cover_per_publish: tl.constexpr = publish_cover // publish_count
    cover_remainder: tl.constexpr = publish_cover % publish_count

    # All current operands are resident, so the old LDS tile is dead.  One
    # barrier establishes that boundary before the same slots are overwritten.
    tl.debug_barrier()
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    for fragment in tl.static_range(publish_count):
        with tlx.warp_pipeline_stage("pgr2_k32_publish", scope="intra_wave", pair=0):
            _wg_wave_grid_local_store_pgr2_k32_packed_fragment(
                a_local,
                b_local,
                prefetched_a,
                prefetched_b,
                fragment,
                b_group_count,
                a_row_count,
                b_group_n,
            )
            if prefetch_future:
                if fragment < b_group_count:
                    future_b += tl.tuple([
                        _wg_wave_grid_global_load_b(
                            b_ptr,
                            pid_n,
                            future_kb,
                            fragment,
                            block_n,
                            block_k,
                            b_group_n,
                            stride_bk,
                            stride_bn,
                            n,
                        )
                    ])
                else:
                    future_a += tl.tuple([
                        _wg_wave_grid_global_load_a_pair_k32(
                            a_ptr,
                            pid_m,
                            future_kb,
                            fragment - b_group_count,
                            32,
                            block_m,
                            stride_am,
                            stride_ak,
                            m,
                        )
                    ])
        with tlx.warp_pipeline_stage("pgr2_k32_cover_publish", scope="intra_wave", pair=0):
            # Spread the independent MFMA prefix as evenly as possible over
            # every publication.  The first ``remainder`` fragments receive
            # one extra MFMA, so this works for any legal operand grid rather
            # than encoding the motivating 6x5 geometry's 3/3/3/2/... split.
            for offset in tl.static_range(cover_per_publish + (1 if fragment < cover_remainder else 0)):
                accumulators = _wg_wave_grid_dot_column_major_at(
                    current_a,
                    current_b,
                    accumulators,
                    fragment * cover_per_publish + (fragment if fragment < cover_remainder else cover_remainder) +
                    offset,
                    a_row_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                    initialize,
                )

    # Make the complete next tile visible, then reload its operands while the
    # remaining independent MFMAs finish the current K32 update.
    tl.debug_barrier()
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    for operand in tl.static_range(read_count):
        with tlx.warp_pipeline_stage("pgr2_k32_read", scope="intra_wave", pair=1):
            if operand < b_group_count:
                next_b += tl.tuple(
                    [_wg_wave_grid_local_load_pgr2_b_half(
                        b_local,
                        operand,
                        0,
                        b_group_n,
                        b_group_count,
                        dot_b,
                        0,
                    )])
            else:
                next_a += tl.tuple([
                    _wg_wave_grid_local_load_pgr2_a_half(
                        a_local,
                        operand - b_group_count,
                        0,
                        a_group_m,
                        a_row_count,
                        dot_a,
                        0,
                    )
                ])
        with tlx.warp_pipeline_stage("pgr2_k32_cover_read", scope="intra_wave", pair=1):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                publish_cover + operand,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize,
            )
    return accumulators, future_a, future_b, next_a, next_b


@triton.jit
def _wg_wave_grid_pgr2_stage_pair_k32_all_resident(
    kb,
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    initialize: tl.constexpr = False,
):
    for half in tl.static_range(2):
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = _wg_wave_grid_pgr2_stage_k32_all_resident(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2 + half,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            True,
            initialize and half == 0,
        )
    return accumulators, prefetched_a, prefetched_b, current_a, current_b


@triton.jit
def _wg_wave_grid_pgr2_stage_pair_k32(
    kb,
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    a_local,
    b_local,
    current_a,
    current_b,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Consume two K32 blocks with statically selected ping-pong stages."""
    for half in tl.static_range(2):
        if a_row_count <= b_group_count:
            state = _wg_wave_grid_pgr2_stage_k32(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2 + half,
                a_local,
                b_local,
                half,
                1 - half,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                True,
                initialize and half == 0,
            )
        else:
            state = _wg_wave_grid_pgr2_stage_k32_b_resident(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2 + half,
                a_local,
                b_local,
                half,
                1 - half,
                current_a,
                current_b,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                block_k,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                m,
                n,
                True,
                initialize and half == 0,
            )
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = state
    return accumulators, prefetched_a, prefetched_b, current_a, current_b


@triton.jit
def _wg_wave_grid_compute_tile_pgr2_k32(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    """Run the geometry-independent two-stage K32 operand pipeline."""
    tl.static_assert(block_k == 32)
    k_blocks: tl.constexpr = k // block_k
    tl.static_assert(k_blocks >= 4 and k_blocks % 2 == 0)
    a_group_m: tl.constexpr = block_m // a_row_count
    tl.static_assert(b_group_n == 16 or b_group_n == 32)
    tl.static_assert(a_group_m == 32 and a_row_count % 2 == 0)
    a_pair_count: tl.constexpr = a_row_count // 2
    for fragment in tl.static_range(a_pair_count + b_group_count):
        _wg_wave_grid_local_store_pgr2_k32_packed_fragment(
            a_local,
            b_local,
            first_a,
            first_b,
            fragment,
            b_group_count,
            a_row_count,
            b_group_n,
        )
    tl.debug_barrier()
    current_a = tl.tuple([
        _wg_wave_grid_local_load_pgr2_a_half(a_local, row, 0, a_group_m, a_row_count, dot_a)
        for row in range(a_row_count)
    ])
    current_b = tl.tuple([
        _wg_wave_grid_local_load_pgr2_b_half(b_local, group, 0, b_group_n, b_group_count, dot_b)
        for group in range(b_group_count)
    ])
    prefetched_a, prefetched_b = (_wg_wave_grid_global_loads_pgr2_k32_packed_a(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    ))
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])

    (
        accumulators,
        prefetched_a,
        prefetched_b,
        current_a,
        current_b,
    ) = _wg_wave_grid_pgr2_stage_pair_k32_all_resident(
        0,
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        a_local,
        b_local,
        current_a,
        current_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        True,
    )
    for kb in tl.range(2, k_blocks - 2, 2, num_stages=1):
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = _wg_wave_grid_pgr2_stage_pair_k32_all_resident(
            kb,
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            False,
        )
    (
        accumulators,
        _,
        _,
        current_a,
        current_b,
    ) = _wg_wave_grid_pgr2_stage_k32_all_resident(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        a_local,
        b_local,
        current_a,
        current_b,
        prefetched_a,
        prefetched_b,
        accumulators,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
        False,
        False,
    )
    for mfma_index in tl.static_range(a_row_count * b_group_count):
        accumulators = _wg_wave_grid_dot_column_major_at(
            current_a,
            current_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    return accumulators


@triton.jit
def _wg_wave_grid_consume_register_k_tail(
    a_ptr,
    b_ptr,
    a_local,
    b_local,
    pid_m,
    pid_n,
    kb,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    tail_k: tl.constexpr,
):
    """Append one 1..64-element tail as ordered K32 MFMAs."""
    tl.static_assert(tail_k > 0 and tail_k <= 64)
    a_group_m: tl.constexpr = block_m // a_row_count
    accumulators = tlx.amd_mfma_commit(
        tl.tuple([tlx.require_layout(accumulator, mma, pin=False) for accumulator in accumulators]))
    b_fragments_per_group: tl.constexpr = b_group_n // 32
    # Preserve the exact distributed layouts produced by the proven PGR2
    # loader while suppressing accesses beyond the logical K extent.
    # Keep the tail tile coordinates opaque so LLVM rematerializes its small
    # address set here instead of spilling the prologue's offsets across the
    # entire K loop for a one-time reuse.
    tail_pid_m = tl.inline_asm_elementwise(
        "s_mov_b32 $0, $1;",
        "=s,s,~{memory}",
        [pid_m],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tail_pid_n = tl.inline_asm_elementwise(
        "s_mov_b32 $0, $1;",
        "=s,s,~{memory}",
        [pid_n],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    if tail_k == 64:
        tail_a, tail_b = _wg_wave_grid_global_loads_pgr2(
            a_ptr,
            b_ptr,
            tail_pid_m,
            tail_pid_n,
            kb // 2,
            block_m,
            block_n,
            64,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
    else:
        tail_a, tail_b = _wg_wave_grid_global_loads_pgr2_tail(
            a_ptr,
            b_ptr,
            tail_pid_m,
            tail_pid_n,
            kb // 2,
            block_m,
            block_n,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
            tail_k,
        )

    # The main PGR2 epilogue has finished reading LDS. Reuse its first bank for
    # the zero-padded final K64, retaining the proven LDS -> MFMA layouts.
    tl.debug_barrier()
    for fragment in tl.static_range(b_fragments_per_group * b_group_count + a_row_count):
        _wg_wave_grid_local_store_pgr2_fragment(
            a_local,
            b_local,
            tail_a,
            tail_b,
            fragment,
            b_group_count,
            a_row_count,
            a_group_m,
            b_group_n,
            64,
            0,
        )
    tl.debug_barrier()
    for half in tl.static_range(2):
        current_a = tl.tuple([
            _wg_wave_grid_local_load_pgr2_a_half(a_local, row, half, a_group_m, a_row_count, dot_a, 0)
            for row in range(a_row_count)
        ])
        current_b = tl.tuple([
            _wg_wave_grid_local_load_pgr2_b_half(b_local, group, half, b_group_n, b_group_count, dot_b, 0)
            for group in range(b_group_count)
        ])
        if tail_k < (half + 1) * 32:
            a_tail_mask = tlx.require_layout(
                tl.broadcast_to(
                    tl.arange(0, 32)[None, :] < tail_k - half * 32,
                    (a_group_m, 32),
                ),
                dot_a,
                pin=False,
            )
            b_tail_mask = tlx.require_layout(
                tl.broadcast_to(
                    tl.arange(0, 32)[:, None] < tail_k - half * 32,
                    (32, b_group_n),
                ),
                dot_b,
                pin=False,
            )
            current_a = tl.tuple([tl.where(a_tail_mask, value, 0.0) for value in current_a])
            current_b = tl.tuple([tl.where(b_tail_mask, value, 0.0) for value in current_b])
        for mfma_index in tl.static_range(a_row_count * b_group_count):
            accumulators = _wg_wave_grid_dot_column_major_at(
                current_a,
                current_b,
                accumulators,
                mfma_index,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize=False,
            )
    return tlx.amd_mfma_commit(accumulators)


@triton.jit
def _wg_wave_grid_compute_tile_regular_k64(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    """Run one regular 5x5..8x8 square MI grid through the K64 pipeline."""
    tl.static_assert(block_k == 64)
    tl.static_assert(a_row_count == b_group_count and a_row_count >= 5 and a_row_count <= 8)
    a_group_m: tl.constexpr = block_m // a_row_count
    k_blocks: tl.constexpr = k // block_k

    _wg_grouped_square_k64_local_store_all(a_local, b_local, 0, first_a, first_b, a_row_count)
    tl.debug_barrier()
    prefetched_a, prefetched_b = _wg_wave_grid_global_loads(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])

    current_a = tl.tuple([
        _wg_grouped_square_k64_local_load_a_half(a_local, 0, row, 0, a_row_count, dot_a) for row in range(a_row_count)
    ])
    current_b = tl.tuple([
        _wg_grouped_square_k64_local_load_b_half(b_local, 0, group, 0, b_group_count, dot_b)
        for group in range(b_group_count)
    ])
    for kb in tl.range(0, k_blocks - 2, num_stages=1):
        # One geometry-derived traversal now covers every supported 5x5..8x8
        # square grid. The compiler distributes each coarse read/publish
        # window using the lowered instruction counts.
        state = _wg_wave_grid_pipeline_step_vendor_square_k64(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb + 2,
            kb % 2,
            (kb + 1) % 2,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = state

    # Drain K(k_blocks-2), then publish and consume final K64.
    mfma_per_half: tl.constexpr = a_row_count * b_group_count
    for mfma_index in tl.static_range(mfma_per_half):
        accumulators = _wg_wave_grid_dot_column_major_at(
            current_a,
            current_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    second_a = tl.tuple([
        _wg_grouped_square_k64_local_load_a_half(a_local, 0, row, 1, a_row_count, dot_a) for row in range(a_row_count)
    ])
    second_b = tl.tuple([
        _wg_grouped_square_k64_local_load_b_half(b_local, 0, group, 1, b_group_count, dot_b)
        for group in range(b_group_count)
    ])
    for mfma_index in tl.static_range(mfma_per_half):
        accumulators = _wg_wave_grid_dot_column_major_at(
            second_a,
            second_b,
            accumulators,
            mfma_index,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    _wg_grouped_square_k64_local_store_all(
        a_local,
        b_local,
        1,
        prefetched_a,
        prefetched_b,
        a_row_count,
    )
    tl.debug_barrier()
    return _wg_grouped_square_k64_compute_stage(
        a_local,
        b_local,
        1,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_row_count,
    )


@triton.jit
def _wg_wave_grid_compute_tile(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    first_a,
    first_b,
    a_local,
    b_local,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    """Compute one output tile with the regular fragmented K32 pipeline."""
    k_blocks: tl.constexpr = k // block_k
    a_group_m: tl.constexpr = block_m // a_row_count
    _wg_wave_grid_local_store_all(
        a_local,
        b_local,
        0,
        first_a,
        first_b,
        a_row_count,
        b_group_count,
        block_k,
        b_group_n,
    )
    tl.debug_barrier()
    prefetched_a, prefetched_b = _wg_wave_grid_global_loads(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        1,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        m,
        n,
    )
    current_b = _wg_wave_grid_local_load_b(b_local, 0, dot_b, b_group_count)
    current_a = tl.tuple([_wg_wave_grid_local_load_a(a_local, 0, row, a_row_count, dot_a) for row in range(2)])

    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])
    for kb in tl.range(0, k_blocks - 2, 2, num_stages=1):
        state = _wg_wave_grid_pipeline_pair(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            kb,
            a_local,
            b_local,
            current_a,
            current_b,
            prefetched_a,
            prefetched_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            m,
            n,
        )
        (
            accumulators,
            prefetched_a,
            prefetched_b,
            current_a,
            current_b,
        ) = state

    # Drain K(k_blocks-2) and the prefetched final K block.
    for row in tl.static_range(a_row_count):
        if row < 2:
            a_operand = current_a[row]
        else:
            a_operand = _wg_wave_grid_local_load_a(a_local, 0, row, a_row_count, dot_a)
        accumulators = _wg_wave_grid_dot_row(
            a_operand,
            current_b,
            accumulators,
            row,
            0,
            b_group_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    _wg_wave_grid_local_store_all(
        a_local,
        b_local,
        1,
        prefetched_a,
        prefetched_b,
        a_row_count,
        b_group_count,
        block_k,
        b_group_n,
    )
    tl.debug_barrier()
    accumulators = _wg_wave_grid_compute_stage(
        a_local,
        b_local,
        1,
        accumulators,
        mma,
        dot_a,
        dot_b,
        a_row_count,
        b_group_count,
    )
    return accumulators


@triton.jit
def _wg_wave_grid_compute_stage_mi32_k64(
    a_local,
    b_local,
    stage: tl.constexpr,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    block_k: tl.constexpr = 64,
    initialize: tl.constexpr = False,
):
    """Consume one LDS stage as native MI32 K16 updates."""
    tl.static_assert(block_k == 32 or block_k == 64)
    for kh in tl.static_range(block_k // 16):
        a_operands = tl.tuple([
            tlx.local_load(
                tlx.local_slice(
                    tlx.local_view(a_local[row], stage),
                    [0, kh * 16],
                    [a_group_m, 16],
                ),
                layout=dot_a,
                relaxed=True,
            ) for row in range(a_row_count)
        ])
        b_operands = tl.tuple([
            tlx.local_load(
                tlx.local_slice(
                    tlx.local_view(b_local[group], stage),
                    [kh * 16, 0],
                    [16, b_group_n],
                ),
                layout=dot_b,
                relaxed=True,
            ) for group in range(b_group_count)
        ])
        accumulators = _wg_wave_grid_dot_column_major_range(
            a_operands,
            b_operands,
            accumulators,
            0,
            a_row_count * b_group_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
            initialize=initialize and kh == 0,
        )
    return accumulators


@tl.core.builtin
def _wg_wave_grid_dot_mi32_deferred_at(
    all_a,
    all_b,
    accumulators,
    deferred_index: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    prefix_columns: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    _semantic=None,
):
    """Issue one deferred MI32 MFMA from one retained K16 window."""
    deferred_index = tl.core._unwrap_if_constexpr(deferred_index)
    a_row_count = tl.core._unwrap_if_constexpr(a_row_count)
    b_group_count = tl.core._unwrap_if_constexpr(b_group_count)
    prefix_columns = tl.core._unwrap_if_constexpr(prefix_columns)
    columns = b_group_count - prefix_columns
    row = deferred_index // columns
    column = prefix_columns + deferred_index % columns
    index = row * b_group_count + column
    values = list(accumulators)
    all_a = list(all_a)
    all_b = list(all_b)
    values[index] = tlx.amd_scheduled_mfma(
        tlx.require_layout(
            all_a[row],
            dot_a,
            pin=False,
            _semantic=_semantic,
        ),
        tlx.require_layout(
            all_b[column],
            dot_b,
            pin=False,
            _semantic=_semantic,
        ),
        tlx.require_layout(values[index], mma, pin=False, _semantic=_semantic),
        accumulator_role="persistent",
        resident_operand=None,
        initialize=False,
        _semantic=_semantic,
    )
    return tl.tuple(values)


@tl.core.builtin
def _wg_wave_grid_dot_mi32_deferred_range(
    all_a,
    all_b,
    accumulators,
    first_mfma: tl.constexpr,
    mfma_count: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    prefix_columns: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    _semantic=None,
):
    """Issue a compile-time interval of deferred MI32 MFMAs."""
    first_mfma = tl.core._unwrap_if_constexpr(first_mfma)
    mfma_count = tl.core._unwrap_if_constexpr(mfma_count)
    values = accumulators
    for offset in range(mfma_count):
        values = _wg_wave_grid_dot_mi32_deferred_at(
            all_a,
            all_b,
            values,
            first_mfma + offset,
            a_row_count,
            b_group_count,
            prefix_columns,
            mma,
            dot_a,
            dot_b,
            _semantic=_semantic,
        )
    return values


@triton.jit
def _wg_wave_grid_pipeline_step_mi32_k64_ring(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    prefetched_a,
    prefetched_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    initialize: tl.constexpr = False,
):
    """Run the K64 single-stage VGPR ring for a regular MI32 grid."""
    all_a = tl.tuple([])
    all_b = tl.tuple([])
    for kh in tl.static_range(4):
        b_operands = tl.tuple([
            tlx.local_load(
                tlx.local_slice(
                    tlx.local_view(b_local[group], 0),
                    [kh * 16, 0],
                    [16, b_group_n],
                ),
                layout=dot_b,
                relaxed=True,
            ) for group in range(b_group_count)
        ])
        a_operands = tl.tuple([])
        for row in tl.static_range(a_row_count):
            a_value = tlx.local_load(
                tlx.local_slice(
                    tlx.local_view(a_local[row], 0),
                    [0, kh * 16],
                    [a_group_m, 16],
                ),
                layout=dot_a,
                relaxed=True,
            )
            a_operands += tl.tuple([a_value])
            if kh < 3:
                accumulators = _wg_wave_grid_dot_row(
                    a_value,
                    b_operands,
                    accumulators,
                    row,
                    0,
                    b_group_count,
                    b_group_count,
                    mma,
                    dot_a,
                    dot_b,
                )
        if kh == 3:
            all_a += a_operands
            all_b += b_operands

    tl.debug_barrier()
    future_a = tl.tuple([])
    future_b = tl.tuple([])
    operand_count: tl.constexpr = a_row_count + b_group_count
    deferred_count: tl.constexpr = a_row_count * b_group_count
    # Expose the independent memory and compute streams once. The scheduler
    # preserves source-anchor spacing when cover is scarce, then uses the
    # target cost model only for surplus cover.
    with tlx.warp_pipeline_stage("mi32_publish_and_prefetch", scope="intra_wave", pair=0):
        for fragment in tl.static_range(operand_count):
            _wg_wave_grid_local_store_fragment(
                a_local,
                b_local,
                0,
                prefetched_a,
                prefetched_b,
                fragment,
                b_group_count,
                64,
                b_group_n,
            )
            if fragment < b_group_count:
                future_b += tl.tuple([
                    _wg_wave_grid_global_load_b(
                        b_ptr,
                        pid_n,
                        future_kb,
                        fragment,
                        block_n,
                        64,
                        b_group_n,
                        stride_bk,
                        stride_bn,
                        n,
                    )
                ])
            else:
                future_a += tl.tuple([
                    _wg_wave_grid_global_load_a(
                        a_ptr,
                        pid_m,
                        future_kb,
                        fragment - b_group_count,
                        a_group_m,
                        b_group_n,
                        block_m,
                        64,
                        stride_am,
                        stride_ak,
                        m,
                    )
                ])
    with tlx.warp_pipeline_stage("mi32_cover_publish", scope="intra_wave", pair=0):
        accumulators = _wg_wave_grid_dot_mi32_deferred_range(
            all_a,
            all_b,
            accumulators,
            0,
            deferred_count,
            a_row_count,
            b_group_count,
            0,
            mma,
            dot_a,
            dot_b,
        )
    tl.debug_barrier()
    return accumulators, future_a, future_b


@triton.jit
def _wg_mi32_direct_local_load_a(
    a_local,
    stage: tl.constexpr,
    quarter: tl.constexpr,
    row: tl.constexpr,
    a_group_m: tl.constexpr,
    dot_a: tl.constexpr,
):
    if len(a_local) == 1:
        source = tlx.local_slice(
            tlx.local_view(a_local[0], stage),
            [row * a_group_m, quarter * 16],
            [a_group_m, 16],
        )
    else:
        source = tlx.local_slice(
            tlx.local_view(a_local[row], stage),
            [0, quarter * 16],
            [a_group_m, 16],
        )
    return tlx.local_load(
        source,
        layout=dot_a,
        relaxed=True,
    )


@triton.jit
def _wg_mi32_direct_local_load_b(
    b_local,
    stage: tl.constexpr,
    quarter: tl.constexpr,
    group: tl.constexpr,
    b_group_n: tl.constexpr,
    dot_b: tl.constexpr,
):
    if len(b_local) == 1:
        source = tlx.local_slice(
            tlx.local_view(b_local[0], stage),
            [quarter * 16, group * b_group_n],
            [16, b_group_n],
        )
    else:
        source = tlx.local_slice(
            tlx.local_view(b_local[group], stage),
            [quarter * 16, 0],
            [16, b_group_n],
        )
    return tlx.local_load(
        source,
        layout=dot_b,
        relaxed=True,
    )


@triton.jit
def _wg_mi32_direct_load_quarter_operands(
    a_local,
    b_local,
    stage: tl.constexpr,
    quarter: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Read one K16 quarter instead of retaining the complete K64 tile."""
    a_operands = tl.tuple(
        [_wg_mi32_direct_local_load_a(a_local, stage, quarter, row, a_group_m, dot_a) for row in range(a_row_count)])
    b_operands = tl.tuple([
        _wg_mi32_direct_local_load_b(b_local, stage, quarter, group, b_group_n, dot_b) for group in range(b_group_count)
    ])
    return a_operands, b_operands


@triton.jit
def _wg_mi32_direct_pgr2_step(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    future_kb,
    a_local,
    b_local,
    current_stage: tl.constexpr,
    next_stage: tl.constexpr,
    current_a,
    current_b,
    accumulators,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    direct_chunk: tl.constexpr,
    load_future: tl.constexpr = True,
    initialize: tl.constexpr = False,
):
    """Stream K16 operand banks while advancing one direct-to-LDS K64."""
    direct_load_count: tl.constexpr = ((block_m + direct_chunk - 1) // direct_chunk +
                                       (block_n + direct_chunk - 1) // direct_chunk)
    mfmas_per_quarter: tl.constexpr = a_row_count * b_group_count
    operand_count: tl.constexpr = a_row_count + b_group_count
    tl.static_assert(mfmas_per_quarter % direct_load_count == 0)
    # K16 quarter q's MFMAs cover the LDS reads for q+1.  Only two operand
    # banks are live, instead of all four K16 quarters of the K64 stage.
    for quarter in tl.static_range(3):
        next_a = tl.tuple([])
        next_b = tl.tuple([])
        with tlx.warp_pipeline_stage(
                "mi32_read_and_compute_quarter",
                scope="intra_wave",
        ):
            for read in tl.static_range(operand_count):
                if read < b_group_count:
                    value = _wg_mi32_direct_local_load_b(
                        b_local,
                        current_stage,
                        quarter + 1,
                        read,
                        b_group_n,
                        dot_b,
                    )
                    next_b += tl.tuple([value])
                else:
                    value = _wg_mi32_direct_local_load_a(
                        a_local,
                        current_stage,
                        quarter + 1,
                        read - b_group_count,
                        a_group_m,
                        dot_a,
                    )
                    next_a += tl.tuple([value])
            accumulators = _wg_wave_grid_dot_column_major_range(
                current_a,
                current_b,
                accumulators,
                0,
                mfmas_per_quarter,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
                initialize=initialize and quarter == 0,
            )
        current_a = next_a
        current_b = next_b

    # Every current-stage LDS value is now in registers.  Refill that stage
    # one 32-wide machine load at a time, covered by the final K16 quarter.
    tl.debug_barrier()
    if load_future:
        with tlx.warp_pipeline_stage("mi32_refill_quarter", scope="intra_wave", pair=0):
            for load in tl.static_range(direct_load_count):
                _wg_logical_rect_k64_direct_load_chunk(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    future_kb,
                    a_local,
                    b_local,
                    current_stage,
                    load,
                    direct_chunk,
                    block_m,
                    block_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    m,
                    n,
                )
        with tlx.warp_pipeline_stage("mi32_compute_refill_quarter", scope="intra_wave", pair=0):
            accumulators = _wg_wave_grid_dot_column_major_range(
                current_a,
                current_b,
                accumulators,
                0,
                mfmas_per_quarter,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
        # Retire the previous K64 while leaving K(t+2)'s groups in flight.
        direct_group_count: tl.constexpr = (((block_m + 3 * direct_chunk) // direct_chunk // 4) +
                                            ((block_n + 3 * direct_chunk) // direct_chunk // 4)
                                            if direct_chunk == 32 else direct_load_count)
        tlx.async_load_wait_group(direct_group_count)
    else:
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            mfmas_per_quarter,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
        tlx.async_load_wait_group(0)
    tl.debug_barrier()
    next_a, next_b = _wg_mi32_direct_load_quarter_operands(
        a_local,
        b_local,
        next_stage,
        0,
        a_row_count,
        b_group_count,
        a_group_m,
        b_group_n,
        dot_a,
        dot_b,
    )
    return accumulators, next_a, next_b


@triton.jit
def _wg_kernel_regular_mi32_wave_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    MI_WAVE_TILE_M: tl.constexpr,
    MI_WAVE_TILE_N: tl.constexpr,
    WARPS_M: tl.constexpr,
    WARPS_N: tl.constexpr,
    DIRECT_TO_LDS: tl.constexpr,
    DIRECT_CHUNK: tl.constexpr,
):
    """Regular MI32 wave grid with a parameterized one-stage K64 pipeline."""
    BLOCK_K: tl.constexpr = 64
    block_m: tl.constexpr = 32 * MI_WAVE_TILE_M * WARPS_M
    block_n: tl.constexpr = 32 * MI_WAVE_TILE_N * WARPS_N
    a_row_count: tl.constexpr = MI_WAVE_TILE_M
    b_group_count: tl.constexpr = MI_WAVE_TILE_N
    a_group_m: tl.constexpr = 32 * WARPS_M
    b_group_n: tl.constexpr = 32 * WARPS_N
    k_blocks: tl.constexpr = K // BLOCK_K
    tl.static_assert(K % BLOCK_K == 0)
    tl.static_assert(WARPS_M * WARPS_N == 4 or WARPS_M * WARPS_N == 8)

    program = tl.program_id(0).to(tl.int32)
    grid_n: tl.constexpr = tl.cdiv(N, block_n)
    pid_m = program // grid_n
    pid_n = program % grid_n

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_dtype: tl.constexpr = tlx.dtype_of(a_ptr)
    b_dtype: tl.constexpr = tlx.dtype_of(b_ptr)

    if DIRECT_TO_LDS:
        a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)], [block_m, BLOCK_K],
                                                                                      order=[1, 0]))
        b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)], [BLOCK_K, block_n],
                                                                                      order=[0, 1]))
        a_local = tl.tuple([tlx.local_alloc((block_m, BLOCK_K), a_dtype, 2, layout=a_layout)])
        b_local = tl.tuple([tlx.local_alloc((BLOCK_K, block_n), b_dtype, 2, layout=b_layout)])
    else:
        a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)], [a_group_m, BLOCK_K],
                                                                                      order=[1, 0]))
        b_bases: tl.constexpr = (_wg_B_BASES_64X32 if b_group_n == 32 else
                                 (_wg_B_BASES_64X64 if b_group_n == 64 else _wg_B_BASES_64X128))
        b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases,
                                                                               [BLOCK_K, b_group_n]))
        a_local = tl.tuple(
            [tlx.local_alloc((a_group_m, BLOCK_K), a_dtype, 1, layout=a_layout) for _ in range(a_row_count)])
        b_local = tl.tuple(
            [tlx.local_alloc((BLOCK_K, b_group_n), b_dtype, 1, layout=b_layout) for _ in range(b_group_count)])

    accumulators = tl.tuple(
        [tlx.zeros((a_group_m, b_group_n), tl.float32, layout=mma) for _ in range(a_row_count * b_group_count)])
    if DIRECT_TO_LDS:
        tl.static_assert(block_m == 256 and block_n == 256)
        _wg_logical_rect_k64_direct_load_stage(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            0,
            a_local,
            b_local,
            0,
            DIRECT_CHUNK,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
        )
        _wg_logical_rect_k64_direct_load_stage(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            1,
            a_local,
            b_local,
            1,
            DIRECT_CHUNK,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
        )
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        current_a, current_b = _wg_mi32_direct_load_quarter_operands(
            a_local,
            b_local,
            0,
            0,
            a_row_count,
            b_group_count,
            a_group_m,
            b_group_n,
            dot_a,
            dot_b,
        )
        accumulators, current_a, current_b = _wg_mi32_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            2,
            a_local,
            b_local,
            0,
            1,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_row_count,
            b_group_count,
            a_group_m,
            b_group_n,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
            DIRECT_CHUNK,
            True,
            True,
        )
        for kb in tl.range(1, k_blocks - 2, num_stages=1):
            current_stage = kb % 2
            next_stage = (kb + 1) % 2
            accumulators, current_a, current_b = _wg_mi32_direct_pgr2_step(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2,
                a_local,
                b_local,
                current_stage,
                next_stage,
                current_a,
                current_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                a_row_count,
                b_group_count,
                a_group_m,
                b_group_n,
                block_m,
                block_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
                DIRECT_CHUNK,
            )
        final_current_stage = (k_blocks - 2) % 2
        final_next_stage = (k_blocks - 1) % 2
        accumulators, current_a, current_b = _wg_mi32_direct_pgr2_step(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            0,
            a_local,
            b_local,
            final_current_stage,
            final_next_stage,
            current_a,
            current_b,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_row_count,
            b_group_count,
            a_group_m,
            b_group_n,
            block_m,
            block_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
            DIRECT_CHUNK,
            False,
            False,
        )
        for quarter in tl.static_range(3):
            next_a, next_b = _wg_mi32_direct_load_quarter_operands(
                a_local,
                b_local,
                final_next_stage,
                quarter + 1,
                a_row_count,
                b_group_count,
                a_group_m,
                b_group_n,
                dot_a,
                dot_b,
            )
            accumulators = _wg_wave_grid_dot_column_major_range(
                current_a,
                current_b,
                accumulators,
                0,
                a_row_count * b_group_count,
                a_row_count,
                b_group_count,
                mma,
                dot_a,
                dot_b,
            )
            current_a = next_a
            current_b = next_b
        accumulators = _wg_wave_grid_dot_column_major_range(
            current_a,
            current_b,
            accumulators,
            0,
            a_row_count * b_group_count,
            a_row_count,
            b_group_count,
            mma,
            dot_a,
            dot_b,
        )
    else:
        current_a, current_b = _wg_wave_grid_global_loads(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            0,
            block_m,
            block_n,
            BLOCK_K,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
        )
        _wg_wave_grid_local_store_all(
            a_local,
            b_local,
            0,
            current_a,
            current_b,
            a_row_count,
            b_group_count,
            BLOCK_K,
            b_group_n,
        )
        tl.debug_barrier()
        prefetched_a, prefetched_b = _wg_wave_grid_global_loads(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            1,
            block_m,
            block_n,
            BLOCK_K,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
        )
        for kb in tl.range(0, k_blocks - 2, num_stages=1):
            accumulators, prefetched_a, prefetched_b = (_wg_wave_grid_pipeline_step_mi32_k64_ring(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                kb + 2,
                a_local,
                b_local,
                prefetched_a,
                prefetched_b,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                a_row_count,
                b_group_count,
                a_group_m,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
            ))
        accumulators = _wg_wave_grid_compute_stage_mi32_k64(
            a_local,
            b_local,
            0,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_row_count,
            b_group_count,
            a_group_m,
            b_group_n,
            BLOCK_K,
        )
        tl.debug_barrier()
        _wg_wave_grid_local_store_all(
            a_local,
            b_local,
            0,
            prefetched_a,
            prefetched_b,
            a_row_count,
            b_group_count,
            BLOCK_K,
            b_group_n,
        )
        tl.debug_barrier()
        accumulators = _wg_wave_grid_compute_stage_mi32_k64(
            a_local,
            b_local,
            0,
            accumulators,
            mma,
            dot_a,
            dot_b,
            a_row_count,
            b_group_count,
            a_group_m,
            b_group_n,
            BLOCK_K,
        )
    accumulators = tlx.amd_mfma_commit(accumulators)
    use_output_mask: tl.constexpr = M % block_m != 0 or N % block_n != 0

    for row in tl.static_range(a_row_count):
        rows = (pid_m * block_m + row * a_group_m + tl.arange(0, a_group_m))
        for group in tl.static_range(b_group_count):
            cols = (pid_n * block_n + group * b_group_n + tl.arange(0, b_group_n))
            value = tlx.release_layout(tlx.require_layout(
                accumulators[row * b_group_count + group],
                mma,
                pin=False,
            )).to(c_ptr.dtype.element_ty)
            ptrs = (c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn)
            if use_output_mask:
                tl.store(
                    ptrs,
                    value,
                    mask=(rows[:, None] < M) & (cols[None, :] < N),
                )
            else:
                tl.store(ptrs, value)


@triton.jit
def _wg_store_mi16_wave_grid_output(
    c_ptr,
    accumulators,
    pid_m,
    pid_n,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    a_group_m: tl.constexpr,
    b_group_n: tl.constexpr,
    a_row_count: tl.constexpr,
    b_group_count: tl.constexpr,
    row_wise: tl.constexpr,
    wide_epilogue: tl.constexpr,
    use_mask: tl.constexpr,
    mma: tl.constexpr,
):
    """Commit and store one regular MI16 accumulator grid."""
    if not row_wise:
        accumulators = tlx.amd_mfma_commit(
            tl.tuple([
                tlx.require_layout(accumulators[index], mma, pin=False) for index in range(a_row_count * b_group_count)
            ]))
    for row in tl.static_range(a_row_count):
        if row_wise:
            row_values = tlx.amd_mfma_commit(
                tl.tuple([
                    tlx.require_layout(
                        accumulators[row * b_group_count + group],
                        mma,
                        pin=False,
                    ) for group in range(b_group_count)
                ]))
        else:
            row_values = tl.tuple([accumulators[row * b_group_count + group] for group in range(b_group_count)])
        rows = (pid_m * block_m + row * a_group_m + tl.arange(0, a_group_m))
        if wide_epilogue:
            tl.static_assert(a_group_m == TILE)
            tl.static_assert(b_group_n == TILE)
            for wide_group in tl.static_range(b_group_count // N_GROUP_FRAGMENTS):
                cols = (pid_n * block_n + wide_group * N_GROUP_FRAGMENTS * b_group_n +
                        tl.arange(0, N_GROUP_FRAGMENTS * TILE))
                ptrs = (c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn)
                lo = tl.cat(
                    tlx.require_layout(row_values[wide_group * N_GROUP_FRAGMENTS], mma, pin=False),
                    tlx.require_layout(row_values[wide_group * N_GROUP_FRAGMENTS + 1], mma, pin=False),
                    dim=1,
                )
                hi = tl.cat(
                    tlx.require_layout(row_values[wide_group * N_GROUP_FRAGMENTS + 2], mma, pin=False),
                    tlx.require_layout(row_values[wide_group * N_GROUP_FRAGMENTS + 3], mma, pin=False),
                    dim=1,
                )
                value = tl.cat(lo, hi, dim=1)
                value = tlx.require_layout(
                    value.to(c_ptr.dtype.element_ty),
                    _C_STORE_32X128_LAYOUT,
                )
                if use_mask:
                    mask = (rows[:, None] < m) & (cols[None, :] < n)
                    mask = tlx.require_layout(mask, _C_STORE_32X128_LAYOUT, pin=False)
                    tl.store(ptrs, value, mask=mask)
                else:
                    tl.store(ptrs, value)
        for group in tl.static_range(
            (b_group_count // N_GROUP_FRAGMENTS * N_GROUP_FRAGMENTS if wide_epilogue else 0),
                b_group_count,
        ):
            cols = (pid_n * block_n + group * b_group_n + tl.arange(0, b_group_n))
            ptrs = (c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn)
            value = row_values[group].to(c_ptr.dtype.element_ty)
            if use_mask:
                mask = (rows[:, None] < m) & (cols[None, :] < n)
                tl.store(
                    ptrs,
                    value,
                    mask=mask,
                )
            else:
                tl.store(ptrs, value)


@triton.jit
def _wg_kernel_persistent_mi16_wave_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    MI_WAVE_TILE_M: tl.constexpr,
    MI_WAVE_TILE_N: tl.constexpr,
    WARPS_M: tl.constexpr,
    WARPS_N: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    ROW_WISE_EPILOGUE: tl.constexpr,
    WIDE_EPILOGUE: tl.constexpr,
    PGR2_LATE_READ_COUNT: tl.constexpr,
):
    """Run a compact persistent PGR2 loop over strided output tiles."""
    block_k: tl.constexpr = 64
    block_m: tl.constexpr = 16 * MI_WAVE_TILE_M * WARPS_M
    block_n: tl.constexpr = 16 * MI_WAVE_TILE_N * WARPS_N
    a_row_count: tl.constexpr = MI_WAVE_TILE_M
    b_group_count: tl.constexpr = MI_WAVE_TILE_N
    a_group_m: tl.constexpr = 16 * WARPS_M
    b_group_n: tl.constexpr = 16 * WARPS_N
    grid_m: tl.constexpr = tl.cdiv(M, block_m)
    grid_n: tl.constexpr = tl.cdiv(N, block_n)
    tile_count: tl.constexpr = grid_m * grid_n
    tiles_per_program: tl.constexpr = tile_count // PROGRAM_COUNT
    tl.static_assert(WARPS_M * WARPS_N == 4)
    tl.static_assert(K % block_k == 0 and K // block_k >= 3)
    tl.static_assert(PROGRAM_COUNT <= tile_count)
    tl.static_assert(tile_count % PROGRAM_COUNT == 0)
    tl.static_assert(PROGRAM_COUNT % grid_n == 0)

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
        block_m,
        a_group_m,
        block_k,
        1,
        True,
        tlx.dtype_of(a_ptr),
    )
    b_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
        block_n,
        b_group_n,
        block_k,
        1,
        False,
        tlx.dtype_of(b_ptr),
    )

    # Striding by the physical program count keeps pid_n invariant, preserving
    # reuse of the small B tile while each CTA walks the tall M dimension.
    program = tl.program_id(0).to(tl.int32)
    tile_id = program
    pid_m = tile_id // grid_n
    pid_n = tile_id % grid_n
    first_a, first_b = _wg_wave_grid_global_loads_pgr2(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        0,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        b_group_n,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        M,
        N,
    )

    # K0 for the next tile is issued before the current epilogue.
    for tile_offset in tl.range(
            0,
            tiles_per_program - 1,
            num_stages=1,
            loop_unroll_factor=1,
    ):
        tile_id = program + tile_offset * PROGRAM_COUNT
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
        next_tile_id = tile_id + PROGRAM_COUNT
        next_pid_m = next_tile_id // grid_n
        next_pid_n = next_tile_id % grid_n
        accumulators = _wg_wave_grid_compute_tile_pgr2_k64(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            first_a,
            first_b,
            a_local,
            b_local,
            mma,
            dot_a,
            dot_b,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
            K,
            1,
            PGR2_LATE_READ_COUNT,
        )
        first_a, first_b = _wg_wave_grid_global_loads_pgr2(
            a_ptr,
            b_ptr,
            next_pid_m,
            next_pid_n,
            0,
            block_m,
            block_n,
            block_k,
            a_row_count,
            b_group_count,
            b_group_n,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            M,
            N,
        )
        _wg_store_mi16_wave_grid_output(
            c_ptr,
            accumulators,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            block_m,
            block_n,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            ROW_WISE_EPILOGUE,
            WIDE_EPILOGUE,
            M % block_m != 0 or N % block_n != 0,
            mma,
        )

    tile_id = program + (tiles_per_program - 1) * PROGRAM_COUNT
    pid_m = tile_id // grid_n
    pid_n = tile_id % grid_n
    accumulators = _wg_wave_grid_compute_tile_pgr2_k64(
        a_ptr,
        b_ptr,
        pid_m,
        pid_n,
        first_a,
        first_b,
        a_local,
        b_local,
        mma,
        dot_a,
        dot_b,
        block_m,
        block_n,
        block_k,
        a_row_count,
        b_group_count,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        M,
        N,
        K,
        1,
        PGR2_LATE_READ_COUNT,
    )
    _wg_store_mi16_wave_grid_output(
        c_ptr,
        accumulators,
        pid_m,
        pid_n,
        stride_cm,
        stride_cn,
        M,
        N,
        block_m,
        block_n,
        a_group_m,
        b_group_n,
        a_row_count,
        b_group_count,
        ROW_WISE_EPILOGUE,
        WIDE_EPILOGUE,
        M % block_m != 0 or N % block_n != 0,
        mma,
    )


@triton.jit
def _wg_kernel_regular_mi16_wave_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    MI_WAVE_TILE_M: tl.constexpr,
    MI_WAVE_TILE_N: tl.constexpr,
    WARPS_M: tl.constexpr,
    WARPS_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOCAL_STAGES: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    LINEAR_TILES: tl.constexpr,
    PGR2_OPERANDS: tl.constexpr,
    DIRECT_TO_LDS: tl.constexpr,
    REFILL_READ_WINDOW: tl.constexpr,
    PGR2_LATE_READ_COUNT: tl.constexpr,
    READ_COVER_SIXTEENTHS: tl.constexpr,
    DIRECT_CHUNK: tl.constexpr,
    PACK_DIRECT_CHUNKS: tl.constexpr,
    ROW_MAJOR_B_LDS: tl.constexpr,
    ROW_WISE_EPILOGUE: tl.constexpr,
    WIDE_EPILOGUE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    WORKGROUP_MAPPING: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PAIR_BALANCED_SPLIT_K: tl.constexpr,
):
    """Regular MI16 wave grid with a VGPR -> LDS K32 pipeline."""
    tl.static_assert(BLOCK_K == 32 or BLOCK_K == 64 or (BLOCK_K == 128 and PGR2_OPERANDS))
    tl.static_assert(WARPS_M * WARPS_N == 4 or WARPS_M * WARPS_N == 8)
    tl.static_assert(WARPS_M == 1 or WARPS_M == 2 or WARPS_M == 4)
    tl.static_assert(WARPS_N == 1 or WARPS_N == 2 or WARPS_N == 4)
    tl.static_assert(WORKGROUP_MAPPING >= 1)
    # Keep the supported MIWaveTile range explicit but geometry-independent.
    # In particular, vendor-selected non-power-of-two tiles such as
    # MT256x240 use MIWaveTile=4x15 and should not require a new kernel.
    tl.static_assert(MI_WAVE_TILE_N >= 2 and MI_WAVE_TILE_N <= 16)
    block_m: tl.constexpr = 16 * MI_WAVE_TILE_M * WARPS_M
    block_n: tl.constexpr = 16 * MI_WAVE_TILE_N * WARPS_N
    a_row_count: tl.constexpr = MI_WAVE_TILE_M
    a_group_m: tl.constexpr = 16 * WARPS_M
    b_group_count: tl.constexpr = MI_WAVE_TILE_N
    b_group_n: tl.constexpr = 16 * WARPS_N
    tl.static_assert(SPLIT_K >= 1)
    tl.static_assert(K % SPLIT_K == 0)
    split_k: tl.constexpr = K // SPLIT_K
    tail_k: tl.constexpr = (split_k % BLOCK_K if PGR2_OPERANDS and (BLOCK_K == 64 or BLOCK_K == 128) else 0)
    has_k_tail: tl.constexpr = tail_k != 0
    # One or two masked K32 MFMAs can extend the K64 PGR2 pipeline without
    # changing the unmasked main loop. Keep this fail-closed for outer Split-K
    # and non-PGR2 paths, whose partition boundaries require separate proof.
    tl.static_assert(split_k % BLOCK_K == 0 or (has_k_tail and PGR2_OPERANDS and ((BLOCK_K == 64 and SPLIT_K == 1) or
                                                                                  (BLOCK_K == 128 and tail_k == 64))))
    main_k: tl.constexpr = split_k - tail_k
    k_blocks: tl.constexpr = main_k // BLOCK_K
    tl.static_assert(k_blocks >= 3)
    if PAIR_BALANCED_SPLIT_K:
        tl.static_assert(SPLIT_K > 1)
        tl.static_assert(BLOCK_K == 64)
        tl.static_assert(LOCAL_STAGES == 2)
        tl.static_assert(PGR2_OPERANDS)
        tl.static_assert(not DIRECT_TO_LDS)
        tl.static_assert(TILES_PER_PROGRAM == 1)
        tl.static_assert(K % (2 * BLOCK_K) == 0)
        split_k_pairs: tl.constexpr = K // (2 * BLOCK_K)
        base_pairs: tl.constexpr = split_k_pairs // SPLIT_K
        extra_pairs: tl.constexpr = split_k_pairs % SPLIT_K
        long_split_k: tl.constexpr = (base_pairs + 1) * 2 * BLOCK_K
        short_split_k: tl.constexpr = base_pairs * 2 * BLOCK_K

    grid_n: tl.constexpr = tl.cdiv(N, block_n)
    if LINEAR_TILES:
        program_count: tl.constexpr = tl.cdiv(tl.cdiv(M, block_m) * grid_n, TILES_PER_PROGRAM)
    else:
        program_grid_n: tl.constexpr = tl.cdiv(grid_n, TILES_PER_PROGRAM)
        program_count: tl.constexpr = tl.cdiv(M, block_m) * program_grid_n
    launch_program = tl.program_id(0).to(tl.int32)
    if SPLIT_K > 1:
        split_id = launch_program // program_count
        program = launch_program % program_count
        if PAIR_BALANCED_SPLIT_K:
            split_start_pair = split_id * base_pairs + min(split_id, extra_pairs)
            split_start = split_start_pair * 2 * BLOCK_K
        else:
            split_start = split_id * split_k
        a_ptr += split_start.to(tl.int64) * stride_ak
        b_ptr += split_start.to(tl.int64) * stride_bk
        c_ptr += split_id.to(tl.int64) * M * stride_cm
    else:
        # Preserve the original non-Split-K program-id path exactly.  Keeping
        # div/mod outside this constexpr branch changes the generated ISA and
        # costs materially on short-K and very-wide grids.
        program = launch_program
    if NUM_XCDS != 1:
        programs_per_xcd: tl.constexpr = tl.cdiv(program_count, NUM_XCDS)
        remainder_xcds: tl.constexpr = program_count % NUM_XCDS
        tall_xcds: tl.constexpr = (NUM_XCDS if remainder_xcds == 0 else remainder_xcds)
        xcd = program % NUM_XCDS
        local_program = program // NUM_XCDS
        if xcd < tall_xcds:
            program = xcd * programs_per_xcd + local_program
        else:
            program = (tall_xcds * programs_per_xcd + (xcd - tall_xcds) * (programs_per_xcd - 1) + local_program)
    grid_m: tl.constexpr = tl.cdiv(M, block_m)
    if LINEAR_TILES:
        first_tile = program * TILES_PER_PROGRAM
        pid_m = first_tile // grid_n
        first_pid_n = first_tile % grid_n
    elif WORKGROUP_MAPPING > 1:
        tl.static_assert(TILES_PER_PROGRAM == 1)
        # Tensile-style positive blocked workgroup mapping. Start from an
        # M-major hardware grid, then visit a narrow band of N tiles for each
        # M tile to improve operand-A locality in L2.
        source_m = program % grid_m
        source_n = program // grid_m
        n_block = source_n // WORKGROUP_MAPPING
        block_start_n = n_block * WORKGROUP_MAPPING
        block_width = tl.minimum(WORKGROUP_MAPPING, grid_n - block_start_n)
        serial = source_m + (source_n % WORKGROUP_MAPPING) * grid_m
        pid_m = serial // block_width
        first_pid_n = serial % block_width + block_start_n
    else:
        pid_m = program // program_grid_n
        first_pid_n = (program % program_grid_n) * TILES_PER_PROGRAM

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)

    a_element_type: tl.constexpr = tlx.dtype_of(a_ptr)
    b_element_type: tl.constexpr = tlx.dtype_of(b_ptr)

    if DIRECT_TO_LDS and BLOCK_K == 128:
        tl.static_assert(LOCAL_STAGES == 2)
        tl.static_assert(WARPS_M * WARPS_N == 4)
        tl.static_assert(block_m <= 256 and block_n <= 256)
        a_local = _wg_logical_rect_k128_allocate_direct_operand(
            block_m,
            True,
            a_element_type,
        )
        b_local = _wg_logical_rect_k128_allocate_direct_operand(
            block_n,
            False,
            b_element_type,
        )
    elif BLOCK_K == 64:
        if DIRECT_TO_LDS:
            a_local = _wg_logical_rect_k64_allocate_direct_operand(
                block_m,
                LOCAL_STAGES,
                DIRECT_CHUNK,
                PACK_DIRECT_CHUNKS,
                True,
                a_element_type,
            )
            b_local = _wg_logical_rect_k64_allocate_direct_operand(
                block_n,
                LOCAL_STAGES,
                DIRECT_CHUNK,
                PACK_DIRECT_CHUNKS,
                False,
                b_element_type,
                stride_bn == 1,
            )
        elif PGR2_OPERANDS:
            tl.static_assert(LOCAL_STAGES == 1 or LOCAL_STAGES == 2)
            # Keep each operand in one allocation so every fragment is a
            # constant LDS offset from one base, as in the vendor PGR2 loop.
            # Pad a non-power-of-two wave tile to a 32-element allocation
            # boundary while retaining the exact logical fragments.
            # The allocator chooses the smallest power-of-two backing extent;
            # compact grids avoid reserving the full 256-element axis.
            if LOCAL_STAGES == 2 and block_m <= 256 and block_n <= 256:
                a_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_m,
                    a_group_m,
                    BLOCK_K,
                    2,
                    True,
                    a_element_type,
                )
                b_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_n,
                    b_group_n,
                    BLOCK_K,
                    2,
                    False,
                    b_element_type,
                    ROW_MAJOR_B_LDS,
                )
            elif LOCAL_STAGES == 2:
                # Larger rectangular tiles need exact fragment allocations:
                # padding every two-stage chunk to 256 would exceed gfx950's
                # 160 KiB LDS limit (for example MT320x160).
                a_local = _wg_wave_grid_allocate_pgr2_double_buffer_operand(
                    block_m,
                    a_group_m,
                    BLOCK_K,
                    True,
                    a_element_type,
                )
                b_local = _wg_wave_grid_allocate_pgr2_double_buffer_operand(
                    block_n,
                    b_group_n,
                    BLOCK_K,
                    False,
                    b_element_type,
                    ROW_MAJOR_B_LDS,
                )
            else:
                a_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_m,
                    a_group_m,
                    BLOCK_K,
                    1,
                    True,
                    a_element_type,
                )
                b_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_n,
                    b_group_n,
                    BLOCK_K,
                    1,
                    False,
                    b_element_type,
                )
        elif (a_row_count == b_group_count and a_row_count >= 5 and a_row_count <= 8):
            # Pack four regular 32-wide fragments in one allocation and keep
            # the remaining 1..4 fragments in a second allocation.  Thus one
            # source representation covers 5x5 through 8x8 MI16 grids.
            tail_extent: tl.constexpr = (64 if a_row_count == 7 else (a_row_count - 4) * 32)
            a_main_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                               [128, BLOCK_K],
                                                                                               order=[1, 0]))
            a_tail_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                               [tail_extent, BLOCK_K],
                                                                                               order=[1, 0]))
            b_main_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)],
                                                                                               [BLOCK_K, 128],
                                                                                               order=[0, 1]))
            b_tail_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)],
                                                                                               [BLOCK_K, tail_extent],
                                                                                               order=[0, 1]))
            if a_row_count == 7:
                a_last_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                                   [32, BLOCK_K],
                                                                                                   order=[1, 0]))
                b_last_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)],
                                                                                                   [BLOCK_K, 32],
                                                                                                   order=[0, 1]))
                a_local = tl.tuple([
                    tlx.local_alloc(
                        (128, BLOCK_K),
                        a_element_type,
                        2,
                        layout=a_main_layout,
                    ),
                    tlx.local_alloc(
                        (64, BLOCK_K),
                        a_element_type,
                        2,
                        layout=a_tail_layout,
                    ),
                    tlx.local_alloc(
                        (32, BLOCK_K),
                        a_element_type,
                        2,
                        layout=a_last_layout,
                    ),
                ])
                b_local = tl.tuple([
                    tlx.local_alloc(
                        (BLOCK_K, 128),
                        b_element_type,
                        2,
                        layout=b_main_layout,
                    ),
                    tlx.local_alloc(
                        (BLOCK_K, 64),
                        b_element_type,
                        2,
                        layout=b_tail_layout,
                    ),
                    tlx.local_alloc(
                        (BLOCK_K, 32),
                        b_element_type,
                        2,
                        layout=b_last_layout,
                    ),
                ])
            else:
                a_local = tl.tuple([
                    tlx.local_alloc(
                        (128, BLOCK_K),
                        a_element_type,
                        2,
                        layout=a_main_layout,
                    ),
                    tlx.local_alloc(
                        (tail_extent, BLOCK_K),
                        a_element_type,
                        2,
                        layout=a_tail_layout,
                    ),
                ])
                b_local = tl.tuple([
                    tlx.local_alloc(
                        (BLOCK_K, 128),
                        b_element_type,
                        2,
                        layout=b_main_layout,
                    ),
                    tlx.local_alloc(
                        (BLOCK_K, tail_extent),
                        b_element_type,
                        2,
                        layout=b_tail_layout,
                    ),
                ])
        elif a_row_count == 11 and b_group_count == 4:
            # MT176x256 consumes nearly the full LDS budget with two raw K64
            # stages.  Keep one power-of-two allocation per MFMA row/column;
            # padding is evaluated separately because it would exceed 120 KiB.
            a_fragment_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 8)],
                                                                                                   [a_group_m, BLOCK_K],
                                                                                                   order=[1, 0]))
            b_fragment_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                [(512, 16)],
                _wg_B_BASES_64X64,
                [BLOCK_K, b_group_n],
            ))
            a_local = tl.tuple([
                tlx.local_alloc(
                    (a_group_m, BLOCK_K),
                    a_element_type,
                    2,
                    layout=a_fragment_layout,
                ) for _ in range(a_row_count)
            ])
            b_local = tl.tuple([
                tlx.local_alloc(
                    (BLOCK_K, b_group_n),
                    b_element_type,
                    2,
                    layout=b_fragment_layout,
                ) for _ in range(b_group_count)
            ])
        else:
            a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                          [a_group_m, BLOCK_K],
                                                                                          order=[1, 0]))
            if b_group_n == 16:
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    _wg_B_BASES_64X16,
                    [BLOCK_K, b_group_n],
                ))
            elif b_group_n == 32:
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    _wg_B_BASES_64X32,
                    [BLOCK_K, b_group_n],
                ))
            else:
                tl.static_assert(b_group_n == 64)
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    _wg_B_BASES_64X64,
                    [BLOCK_K, b_group_n],
                ))
            a_local = tl.tuple([
                tlx.local_alloc(
                    (a_group_m, BLOCK_K),
                    a_element_type,
                    LOCAL_STAGES,
                    layout=a_layout,
                ) for _ in range(a_row_count)
            ])
            b_local = tl.tuple([
                tlx.local_alloc(
                    (BLOCK_K, b_group_n),
                    b_element_type,
                    LOCAL_STAGES,
                    layout=b_layout,
                ) for _ in range(b_group_count)
            ])
    else:
        if PGR2_OPERANDS:
            tl.static_assert(LOCAL_STAGES == 1)
            if BLOCK_K == 128:
                a_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_m,
                    a_group_m,
                    64,
                    2,
                    True,
                    a_element_type,
                )
                b_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_n,
                    b_group_n,
                    64,
                    2,
                    False,
                    b_element_type,
                )
            else:
                a_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_m,
                    a_group_m,
                    BLOCK_K,
                    1,
                    True,
                    a_element_type,
                )
                b_local = _wg_wave_grid_allocate_pgr2_operand_chunks(
                    block_n,
                    b_group_n,
                    BLOCK_K,
                    1,
                    False,
                    b_element_type,
                )
        else:
            a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                          [a_group_m, BLOCK_K],
                                                                                          order=[1, 0]))
            a_local = tl.tuple([
                tlx.local_alloc(
                    (a_group_m, BLOCK_K),
                    a_element_type,
                    2,
                    layout=a_layout,
                ) for _ in range(a_row_count)
            ])
            if b_group_n == 16:
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    _wg_B_BASES_32X16,
                    [BLOCK_K, b_group_n],
                ))
            elif b_group_n == 32:
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    _wg_B_BASES_32X32,
                    [BLOCK_K, b_group_n],
                ))
            else:
                b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 64)],
                    _wg_B_BASES_32X64,
                    [BLOCK_K, b_group_n],
                ))
            b_local = tl.tuple([
                tlx.local_alloc(
                    (BLOCK_K, b_group_n),
                    b_element_type,
                    2,
                    layout=b_layout,
                ) for _ in range(b_group_count)
            ])

    if DIRECT_TO_LDS:
        tl.static_assert(BLOCK_K == 64 or BLOCK_K == 128)
        tl.static_assert(block_m <= 256 and block_n <= 256)
        tl.static_assert(block_m >= 32 and block_n >= 32)
        tl.static_assert(DIRECT_CHUNK == 32 or DIRECT_CHUNK == 64 or DIRECT_CHUNK == 128)
        if PACK_DIRECT_CHUNKS:
            tl.static_assert(DIRECT_CHUNK == 32)
        if WARPS_M * WARPS_N == 8:
            # A 32x64 FP16 window has only four values per thread at eight
            # waves, so it cannot satisfy the direct-load contiguity of eight.
            # Use at least a 64-wide window for the eight-wave mapping.
            tl.static_assert(DIRECT_CHUNK >= 64)
        tl.static_assert(TILES_PER_PROGRAM == 1 or TILES_PER_PROGRAM == 2)
    else:
        if PGR2_OPERANDS and BLOCK_K == 128:
            first_a, first_b = _wg_wave_grid_global_loads_pgr2_k128(
                a_ptr,
                b_ptr,
                pid_m,
                first_pid_n,
                0,
                block_m,
                block_n,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
            )
        elif PGR2_OPERANDS and BLOCK_K == 64:
            first_a, first_b = _wg_wave_grid_global_loads_pgr2(
                a_ptr,
                b_ptr,
                pid_m,
                first_pid_n,
                0,
                block_m,
                block_n,
                BLOCK_K,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
            )
        elif PGR2_OPERANDS:
            first_a, first_b = _wg_wave_grid_global_loads_pgr2_k32_packed_a(
                a_ptr,
                b_ptr,
                pid_m,
                first_pid_n,
                0,
                block_m,
                block_n,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
            )
        else:
            first_a, first_b = _wg_wave_grid_global_loads(
                a_ptr,
                b_ptr,
                pid_m,
                first_pid_n,
                0,
                block_m,
                block_n,
                BLOCK_K,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
            )
    for tile in tl.static_range(TILES_PER_PROGRAM):
        if LINEAR_TILES:
            tile_id = program * TILES_PER_PROGRAM + tile
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
        else:
            pid_n = first_pid_n + tile
        if DIRECT_TO_LDS:
            if PGR2_OPERANDS and BLOCK_K == 128:
                accumulators = _wg_logical_rect_k128_compute_tile_direct_pgr2(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    a_group_m,
                    b_group_n,
                    a_row_count,
                    b_group_count,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                )
            elif PGR2_OPERANDS:
                tl.static_assert(LOCAL_STAGES == 2)
                accumulators = _wg_logical_rect_k64_compute_tile_direct_pgr2(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    a_group_m,
                    b_group_n,
                    a_row_count,
                    b_group_count,
                    DIRECT_CHUNK,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                    WARPS_M * WARPS_N,
                    REFILL_READ_WINDOW,
                    READ_COVER_SIXTEENTHS,
                    LOOP_UNROLL_FACTOR,
                )
            else:
                accumulators = _wg_logical_rect_k64_compute_tile_direct(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    a_group_m,
                    b_group_n,
                    a_row_count,
                    b_group_count,
                    DIRECT_CHUNK,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                    WARPS_M * WARPS_N,
                    tile > 0,
                )
        elif BLOCK_K == 128:
            tl.static_assert(PGR2_OPERANDS)
            tl.static_assert(LOCAL_STAGES == 1)
            accumulators = _wg_wave_grid_compute_tile_pgr2_k128(
                a_ptr,
                b_ptr,
                pid_m,
                pid_n,
                first_a,
                first_b,
                a_local,
                b_local,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                a_row_count,
                b_group_count,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
                main_k,
            )
        elif BLOCK_K == 64:
            if PGR2_OPERANDS:
                if PAIR_BALANCED_SPLIT_K:
                    if split_id < extra_pairs:
                        accumulators = _wg_wave_grid_compute_tile_pgr2_k64(
                            a_ptr,
                            b_ptr,
                            pid_m,
                            pid_n,
                            first_a,
                            first_b,
                            a_local,
                            b_local,
                            mma,
                            dot_a,
                            dot_b,
                            block_m,
                            block_n,
                            BLOCK_K,
                            a_row_count,
                            b_group_count,
                            stride_am,
                            stride_ak,
                            stride_bk,
                            stride_bn,
                            M,
                            N,
                            long_split_k,
                            LOCAL_STAGES,
                            PGR2_LATE_READ_COUNT,
                        )
                    else:
                        accumulators = _wg_wave_grid_compute_tile_pgr2_k64(
                            a_ptr,
                            b_ptr,
                            pid_m,
                            pid_n,
                            first_a,
                            first_b,
                            a_local,
                            b_local,
                            mma,
                            dot_a,
                            dot_b,
                            block_m,
                            block_n,
                            BLOCK_K,
                            a_row_count,
                            b_group_count,
                            stride_am,
                            stride_ak,
                            stride_bk,
                            stride_bn,
                            M,
                            N,
                            short_split_k,
                            LOCAL_STAGES,
                            PGR2_LATE_READ_COUNT,
                        )
                else:
                    accumulators = _wg_wave_grid_compute_tile_pgr2_k64(
                        a_ptr,
                        b_ptr,
                        pid_m,
                        pid_n,
                        first_a,
                        first_b,
                        a_local,
                        b_local,
                        mma,
                        dot_a,
                        dot_b,
                        block_m,
                        block_n,
                        BLOCK_K,
                        a_row_count,
                        b_group_count,
                        stride_am,
                        stride_ak,
                        stride_bk,
                        stride_bn,
                        M,
                        N,
                        main_k,
                        LOCAL_STAGES,
                        PGR2_LATE_READ_COUNT,
                    )
            elif (a_row_count == b_group_count and a_row_count >= 5 and a_row_count <= 8):
                accumulators = _wg_wave_grid_compute_tile_regular_k64(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    first_a,
                    first_b,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                )
            else:
                accumulators = _wg_wave_grid_compute_tile_rect_k64(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    first_a,
                    first_b,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                    LOCAL_STAGES,
                )
        else:
            if PGR2_OPERANDS:
                accumulators = _wg_wave_grid_compute_tile_pgr2_k32(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    first_a,
                    first_b,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    split_k,
                )
            else:
                accumulators = _wg_wave_grid_compute_tile(
                    a_ptr,
                    b_ptr,
                    pid_m,
                    pid_n,
                    first_a,
                    first_b,
                    a_local,
                    b_local,
                    mma,
                    dot_a,
                    dot_b,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    main_k,
                )
        if has_k_tail:
            accumulators = _wg_wave_grid_consume_register_k_tail(
                a_ptr,
                b_ptr,
                a_local,
                b_local,
                pid_m,
                pid_n,
                main_k // 32,
                accumulators,
                mma,
                dot_a,
                dot_b,
                block_m,
                block_n,
                a_row_count,
                b_group_count,
                b_group_n,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                N,
                tail_k,
            )
        # Issue the following tile's K0 global loads before draining and
        # storing this tile's persistent AGPR accumulators.  This is the
        # multi-tile prologue/epilogue overlap used by the persistent kernel.
        if tile + 1 < TILES_PER_PROGRAM:
            if LINEAR_TILES:
                next_tile_id = program * TILES_PER_PROGRAM + tile + 1
                next_pid_m = next_tile_id // grid_n
                next_pid_n = next_tile_id % grid_n
            else:
                next_pid_m = pid_m
                next_pid_n = pid_n + 1
            if DIRECT_TO_LDS:
                _wg_logical_rect_k64_direct_load_stage(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    0,
                    a_local,
                    b_local,
                    0,
                    DIRECT_CHUNK,
                    block_m,
                    block_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    WARPS_M * WARPS_N,
                )
                _wg_logical_rect_k64_direct_load_stage(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    1,
                    a_local,
                    b_local,
                    1,
                    DIRECT_CHUNK,
                    block_m,
                    block_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                    WARPS_M * WARPS_N,
                )
            elif PGR2_OPERANDS and BLOCK_K == 128:
                first_a, first_b = _wg_wave_grid_global_loads_pgr2_k128(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    0,
                    block_m,
                    block_n,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                )
            elif PGR2_OPERANDS and BLOCK_K == 64:
                first_a, first_b = _wg_wave_grid_global_loads_pgr2(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    0,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                )
            elif PGR2_OPERANDS:
                first_a, first_b = (_wg_wave_grid_global_loads_pgr2_k32_packed_a(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    0,
                    block_m,
                    block_n,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                ))
            else:
                first_a, first_b = _wg_wave_grid_global_loads(
                    a_ptr,
                    b_ptr,
                    next_pid_m,
                    next_pid_n,
                    0,
                    block_m,
                    block_n,
                    BLOCK_K,
                    a_row_count,
                    b_group_count,
                    b_group_n,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    M,
                    N,
                )
        _wg_store_mi16_wave_grid_output(
            c_ptr,
            accumulators,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            block_m,
            block_n,
            a_group_m,
            b_group_n,
            a_row_count,
            b_group_count,
            ROW_WISE_EPILOGUE,
            WIDE_EPILOGUE,
            M % block_m != 0 or N % block_n != 0,
            mma,
        )


@lru_cache(maxsize=None)
def _wg_device_arch(device):
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


@lru_cache(maxsize=None)
def _wg_plan_for_shape(m, n, k, dtype):
    """Return the cached wave-grid plan for one logical problem."""
    if dtype == torch.float16:
        return _wg_plan(m, n, k)
    if dtype == torch.bfloat16:
        return _wg_BF16_WAVE_GRID_SPECIALIZATIONS.get((m, n, k))
    return None


def _wg_plan_for(a, b):
    """Return the selected dense-TN 16-bit wave-grid plan, if supported."""
    if (a.ndim != 2 or b.ndim != 2 or a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype
            or not a.is_cuda or a.device != b.device or a.shape[1] != b.shape[0] or a.stride(1) != 1 or b.stride(0) != 1
            or _wg_device_arch(a.device) != "gfx950"):
        return None
    return _wg_plan_for_shape(a.shape[0], b.shape[1], a.shape[1], a.dtype)


def _wg_supports(a, b):
    """Return whether the parameterized wave-grid family supports ``a @ b``."""
    return _wg_plan_for(a, b) is not None


def _wg_streamk_tail_schedule(m, n, k, plan):
    """Validate and describe the bounded single-M-tile Stream-K schedule.

    Complete device waves run as one persistent multi-tile launch.  The
    remaining contiguous N suffix uses the existing deterministic Split-K
    path, whose reduction consumes contiguous K partitions in increasing
    split order.  Keep this capability fail-closed until arbitrary 2D tile
    subsets have an equally compact workspace representation.
    """
    tail_split = plan.get("streamk_tail_split", 0)
    if not tail_split:
        return None
    if (plan.get("kind") != "regular_mi16_wave_grid" or not plan.get("pgr2_operands", False)
            or plan.get("direct_to_lds", False) or plan.get("split_k", 1) != 1):
        raise ValueError("wave-grid Stream-K tail requires a non-direct regular MI16 "
                         "PGR2 plan without outer Split-K")
    block_m = 16 * plan["mi_wave_tile_m"] * plan["warps_m"]
    block_n = 16 * plan["mi_wave_tile_n"] * plan["warps_n"]
    grid_m = triton.cdiv(m, block_m)
    grid_n = triton.cdiv(n, block_n)
    if grid_m != 1:
        raise ValueError("wave-grid Stream-K tail currently requires one logical M tile")
    full_rounds, tail_tiles = divmod(grid_n, _wg_NUM_CUS)
    if full_rounds < 1 or tail_tiles < 1:
        raise ValueError("wave-grid Stream-K tail requires complete and partial device waves")
    if k % tail_split:
        raise ValueError(f"wave-grid Stream-K tail split {tail_split} must divide K={k}")
    if k % (tail_split * plan["block_k"]):
        raise ValueError("wave-grid Stream-K tail must contain whole K blocks")
    tail_k_blocks = k // tail_split // plan["block_k"]
    minimum_k_blocks = 4 if plan.get("local_stages", 1) == 2 else 3
    if tail_k_blocks < minimum_k_blocks or (plan.get("local_stages", 1) == 2 and tail_k_blocks % 2):
        raise ValueError("wave-grid Stream-K tail does not satisfy the selected PGR2 "
                         "pipeline depth")
    full_n = full_rounds * _wg_NUM_CUS * block_n
    return full_rounds, full_n, tail_split


def _wg_launch_streamk_tail(a, b, out, plan, schedule):
    """Compose persistent full waves with a deterministic Split-K N tail."""
    full_rounds, full_n, tail_split = schedule
    full_plan = dict(plan)
    full_plan.update(
        streamk_tail_split=0,
        split_k=1,
        pair_balanced_split_k=False,
        tiles_per_program=full_rounds,
        linear_tiles=True,
    )
    tail_plan = dict(plan)
    tail_plan.update(
        streamk_tail_split=0,
        split_k=1,
        pair_balanced_split_k=False,
        tiles_per_program=1,
        linear_tiles=False,
        num_xcds=1,
    )
    _wg_matmul(
        a,
        b[:, :full_n],
        out=out[:, :full_n],
        _candidate_plan=full_plan,
    )
    _wg_matmul(
        a,
        b[:, full_n:],
        out=out[:, full_n:],
        _candidate_plan=tail_plan,
        _split_k=tail_split,
        _reduce_tile=plan.get("reduce_tile", (16, 64)),
        _reduce_warps=plan.get("reduce_warps", 4),
    )
    return out


def _wg_matmul(
    a,
    b,
    *,
    out=None,
    use_intra_wave_pipeline=True,
    _candidate_plan=None,
    _split_k=None,
    _reduce_tile=None,
    _reduce_warps=None,
):
    """Run the selected intra-wave GEMM plan."""
    selected_plan = _wg_plan_for(a, b) if _candidate_plan is None else _candidate_plan
    if selected_plan is None:
        raise ValueError("no gfx950 wave-grid GEMM plan for the supplied operands")
    row_major_b_candidate = (_candidate_plan is not None and b.ndim == 2 and b.stride(1) == 1
                             and selected_plan.get("kind") == "regular_mi16_wave_grid" and
                             (selected_plan.get("pgr2_operands", False) or selected_plan.get("direct_to_lds", False))
                             and (not selected_plan.get("direct_to_lds", False)
                                  or selected_plan.get("row_major_b_lds", False)))
    if _candidate_plan is not None and (a.ndim != 2 or b.ndim != 2 or a.dtype not in (torch.float16, torch.bfloat16)
                                        or b.dtype != a.dtype or not a.is_cuda or a.device != b.device
                                        or a.shape[1] != b.shape[0] or a.stride(1) != 1 or
                                        (b.stride(0) != 1 and not row_major_b_candidate)
                                        or _wg_device_arch(a.device) != "gfx950"):
        raise ValueError("invalid operands for a gfx950 intra-wave candidate")
    m, k = a.shape
    _, n = b.shape
    plan = selected_plan
    if plan is None:
        raise ValueError(f"no gfx950 intra-wave GEMM plan for {(m, n, k)}")
    if a.dtype == torch.bfloat16 and plan["kind"] not in (
            "regular_mi16_wave_grid",
            "regular_mi32_wave_grid",
    ):
        raise ValueError("gfx950 BF16 candidates require a regular wave-grid path")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    elif (out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device):
        raise ValueError("output must match the GEMM shape, dtype, and device; "
                         f"got shape={tuple(out.shape)}, dtype={out.dtype}, "
                         f"device={out.device}")
    if _split_k is None:
        streamk_schedule = _wg_streamk_tail_schedule(m, n, k, plan)
        if streamk_schedule is not None:
            return _wg_launch_streamk_tail(a, b, out, plan, streamk_schedule)
    if _split_k is None:
        _split_k = plan.get("split_k", 1)
    if _reduce_tile is None:
        _reduce_tile = plan.get("reduce_tile", (16, 64))
    if _reduce_warps is None:
        _reduce_warps = plan.get("reduce_warps", 4)
    if _split_k < 1 or k % _split_k != 0:
        raise ValueError(f"invalid wave-grid Split-K={_split_k} for K={k}")
    pair_balanced_split_k = plan.get("pair_balanced_split_k", False)
    if pair_balanced_split_k and (_split_k <= 1 or k % (2 * plan["block_k"]) != 0
                                  or not plan.get("pgr2_operands", False) or plan.get("local_stages", 1) != 2
                                  or plan.get("direct_to_lds", False) or plan.get("tiles_per_program", 1) != 1):
        raise ValueError("pair-balanced wave-grid Split-K requires a one-tile, two-stage "
                         "PGR2 plan and a whole number of K-block pairs")
    if (_split_k > 1 and plan.get("pgr2_operands", False) and plan.get("local_stages", 1) == 2
            and plan.get("direct_to_lds", False) and not pair_balanced_split_k
            and (k // _split_k // plan["block_k"]) % 2 != 0):
        raise ValueError("direct two-stage wave-grid Split-K requires an even number of "
                         f"K blocks per split; got K={k}, Split-K={_split_k}, "
                         f"BLOCK_K={plan['block_k']}")
    operands = (
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    if plan["kind"] == "register_fragmented_m":
        block_m = sum(extent for _, extent in plan["m_fragments"])
        block_n = plan["block_n"]
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )
        common = dict(
            BLOCK_N=block_n,
            BLOCK_K=plan["block_k"],
            num_warps=plan["num_warps"],
            num_stages=plan["num_stages"],
            matrix_instr_nonkdim=16,
        )
        if len(plan["m_fragments"]) == 2:
            _wg_kernel_register_split_m2[grid](
                *operands,
                M_HEAD=plan["m_fragments"][0][1],
                M_TAIL=plan["m_fragments"][1][1],
                **common,
            )
        elif len(plan["m_fragments"]) == 3:
            _wg_kernel_register_split_m3[grid](
                *operands,
                M0=plan["m_fragments"][0][1],
                M1=plan["m_fragments"][1][1],
                M2=plan["m_fragments"][2][1],
                **common,
            )
        else:
            raise ValueError("register fragmented-M supports two or three fragments")
        return out
    if plan["kind"] == "register_mi16_wave_grid":
        block_m = 16 * plan["mi_wave_tile_m"] * plan["warps_m"]
        block_n = 16 * plan["mi_wave_tile_n"] * plan["warps_n"]
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )
        _wg_kernel_register_mi16_wave_grid[grid](
            *operands,
            MI_WAVE_TILE_M=plan["mi_wave_tile_m"],
            MI_WAVE_TILE_N=plan["mi_wave_tile_n"],
            WARPS_M=plan["warps_m"],
            WARPS_N=plan["warps_n"],
            PREFETCH_NEXT=plan.get("prefetch_next", True),
            num_warps=plan["num_warps"],
            num_stages=1,
            matrix_instr_nonkdim=16,
            waves_per_eu=plan.get("waves_per_eu", 0),
            regclass_priority_trumps_globalness=True,
            reverse_local_assignment=plan.get("reverse_local_assignment", True),
            disable_unclustered_high_rp_reschedule=plan.get("disable_unclustered_high_rp_reschedule", False),
        )
        return out
    if plan["kind"] in (
            "regular_mi16_wave_grid",
            "regular_mi32_wave_grid",
    ):
        matrix_instr_nonkdim = plan.get("matrix_instr_nonkdim", 16)
        block_m = (matrix_instr_nonkdim * plan["mi_wave_tile_m"] * plan["warps_m"])
        block_n = (matrix_instr_nonkdim * plan["mi_wave_tile_n"] * plan["warps_n"])
        persistent_program_count = plan.get("persistent_program_count", 0)
        if persistent_program_count:
            tile_count = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
            grid_n = triton.cdiv(n, block_n)
            if (plan["kind"] != "regular_mi16_wave_grid" or not plan.get("pgr2_operands", False)
                    or plan.get("direct_to_lds", False) or plan.get("local_stages", 1) != 1 or plan["block_k"] != 64
                    or _split_k != 1 or persistent_program_count > tile_count or tile_count % persistent_program_count
                    or persistent_program_count % grid_n):
                raise ValueError("persistent MI16 wave-grid requires an evenly divided "
                                 "one-stage K64 PGR2 grid")
            _wg_kernel_persistent_mi16_wave_grid[(persistent_program_count, )](
                *operands,
                MI_WAVE_TILE_M=plan["mi_wave_tile_m"],
                MI_WAVE_TILE_N=plan["mi_wave_tile_n"],
                WARPS_M=plan["warps_m"],
                WARPS_N=plan["warps_n"],
                PROGRAM_COUNT=persistent_program_count,
                ROW_WISE_EPILOGUE=plan.get("row_wise_epilogue", False),
                WIDE_EPILOGUE=plan.get("wide_epilogue", False),
                PGR2_LATE_READ_COUNT=plan.get("pgr2_late_read_count", 0),
                num_warps=plan["num_warps"],
                num_stages=1,
                matrix_instr_nonkdim=16,
                waves_per_eu=plan.get("waves_per_eu", 0),
                regclass_priority_trumps_globalness=True,
                reverse_local_assignment=plan.get("reverse_local_assignment", True),
                sink_insts_to_avoid_spills=plan.get("sink_insts_to_avoid_spills", False),
                disable_unclustered_high_rp_reschedule=plan.get("disable_unclustered_high_rp_reschedule", False),
                enable_sched_group_barrier_scheduler=plan.get("enable_sched_group_barrier_scheduler", False),
                sched_group_barrier_mfma_per_dwordx4=plan.get("sched_group_barrier_mfma_per_dwordx4", 4),
                llvm_fn_attrs=plan.get("llvm_fn_attrs", ""),
            )
            return out
        tiles_per_program = plan["tiles_per_program"]
        if _split_k > 1 and (plan["kind"] != "regular_mi16_wave_grid" or tiles_per_program != 1):
            raise ValueError("wave-grid Split-K requires one-tile regular MI16 plans")
        if plan.get("linear_tiles", False):
            grid = (triton.cdiv(
                triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
                tiles_per_program,
            ), )
        else:
            grid = (triton.cdiv(m, block_m) * triton.cdiv(triton.cdiv(n, block_n), tiles_per_program), )
        if _split_k > 1:
            workspace = torch.empty((_split_k * m, n), device=a.device, dtype=torch.float32)
            kernel_operands = (
                a,
                b,
                workspace,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                workspace.stride(0),
                workspace.stride(1),
            )
            grid = (grid[0] * _split_k, )
        else:
            workspace = None
            kernel_operands = operands
        if plan["kind"] == "regular_mi32_wave_grid":
            _wg_kernel_regular_mi32_wave_grid[grid](
                *kernel_operands,
                MI_WAVE_TILE_M=plan["mi_wave_tile_m"],
                MI_WAVE_TILE_N=plan["mi_wave_tile_n"],
                WARPS_M=plan["warps_m"],
                WARPS_N=plan["warps_n"],
                DIRECT_TO_LDS=plan.get("direct_to_lds", False),
                DIRECT_CHUNK=plan.get("direct_chunk", 32),
                num_warps=plan["num_warps"],
                num_stages=1,
                matrix_instr_nonkdim=32,
                regclass_priority_trumps_globalness=True,
                reverse_local_assignment=plan.get("reverse_local_assignment", True),
                sink_insts_to_avoid_spills=plan.get("sink_insts_to_avoid_spills", False),
                disable_unclustered_high_rp_reschedule=plan.get("disable_unclustered_high_rp_reschedule", False),
            )
            return out
        _wg_kernel_regular_mi16_wave_grid[grid](
            *kernel_operands,
            MI_WAVE_TILE_M=plan["mi_wave_tile_m"],
            MI_WAVE_TILE_N=plan["mi_wave_tile_n"],
            WARPS_M=plan["warps_m"],
            WARPS_N=plan["warps_n"],
            BLOCK_K=plan["block_k"],
            LOCAL_STAGES=plan.get("local_stages", 1),
            TILES_PER_PROGRAM=tiles_per_program,
            LINEAR_TILES=plan.get("linear_tiles", False),
            PGR2_OPERANDS=plan.get("pgr2_operands", False),
            DIRECT_TO_LDS=plan.get("direct_to_lds", False),
            REFILL_READ_WINDOW=plan.get("refill_read_window", 0),
            PGR2_LATE_READ_COUNT=plan.get("pgr2_late_read_count", 0),
            READ_COVER_SIXTEENTHS=plan.get("read_cover_sixteenths", 8),
            DIRECT_CHUNK=plan.get("direct_chunk", 128),
            PACK_DIRECT_CHUNKS=plan.get("pack_direct_chunks", False),
            ROW_MAJOR_B_LDS=plan.get("row_major_b_lds", False),
            ROW_WISE_EPILOGUE=plan.get("row_wise_epilogue", False),
            WIDE_EPILOGUE=plan.get("wide_epilogue", False),
            NUM_XCDS=plan.get("num_xcds", 1),
            WORKGROUP_MAPPING=plan.get("workgroup_mapping", 1),
            LOOP_UNROLL_FACTOR=plan.get("loop_unroll_factor", 1),
            SPLIT_K=_split_k,
            PAIR_BALANCED_SPLIT_K=pair_balanced_split_k,
            num_warps=plan["num_warps"],
            num_stages=1,
            matrix_instr_nonkdim=16,
            waves_per_eu=plan.get("waves_per_eu", 0),
            regclass_priority_trumps_globalness=True,
            reverse_local_assignment=plan.get("reverse_local_assignment", True),
            sink_insts_to_avoid_spills=plan.get("sink_insts_to_avoid_spills", False),
            disable_unclustered_high_rp_reschedule=plan.get("disable_unclustered_high_rp_reschedule", False),
            enable_sched_group_barrier_scheduler=plan.get("enable_sched_group_barrier_scheduler", False),
            sched_group_barrier_mfma_per_dwordx4=plan.get("sched_group_barrier_mfma_per_dwordx4", 4),
            llvm_fn_attrs=plan.get("llvm_fn_attrs", ""),
        )
        if _split_k > 1:
            rbm, rbn = _reduce_tile
            _reduce_k_kernel[(triton.cdiv(m, rbm), triton.cdiv(n, rbn))](
                workspace,
                out,
                out,
                m,
                n,
                0,
                0,
                out.stride(0),
                out.stride(1),
                SPLIT_K=_split_k,
                BLOCK_SIZE_M=rbm,
                BLOCK_SIZE_N=rbn,
                OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
                ADD_BIAS=False,
                num_warps=_reduce_warps,
            )
        return out

    raise AssertionError(f"unknown intra-wave GEMM plan kind: {plan['kind']}")


# Family-level dispatch policy.
_wg_NUM_CU = 256


def _wg_supports_hybrid_n160(m, n, k):
    """Return whether one N160 tail can balance the final device wave."""
    tail_n = 160 * _wg_NUM_CU
    main_n = n - tail_n
    full_n_wave = 256 * _wg_NUM_CU
    return (208 <= m <= 224 and k == 6144 and (m, n, k) not in _wg_SOTA77_WAVE_GRID_PROMOTIONS and n % 224 != 0
            and full_n_wave <= main_n <= 2 * full_n_wave and main_n % full_n_wave == 0)


def _wg_transposed_wave_grid_plan(m, n, k):
    """Return a plan for computing ``(A @ B).T = B.T @ A.T``."""
    specialization = _wg_SOTA77_TRANSPOSED_WAVE_GRID_PROMOTIONS.get((m, n, k))
    if specialization is not None:
        return specialization
    # For shallow reductions with a very wide N dimension, transposition turns
    # thousands of short M-tail tiles into a regular MT160x160 grid while
    # preserving the same mathematical GEMM and output layout.
    if 128 <= m <= 160 and n >= 512 * m and k == 768:
        return _wg_regular_wave_grid_plan(5, 5, 2, 2, 64)
    if (208 <= n <= 224 and 4096 <= k <= 8192 and k % 128 == 0 and m % 256 == 0):
        full_waves, tail_tiles = divmod(m // 256, _wg_NUM_CU)
        if (full_waves == 1 and 3 * _wg_NUM_CU // 4 <= tail_tiles <= 13 * _wg_NUM_CU // 16):
            return _wg_regular_wave_grid_plan(
                7,
                8,
                2,
                2,
                64,
                local_stages=2,
                pgr2_operands=True,
                direct_to_lds=True,
                loop_unroll_factor=2,
                direct_chunk=32,
                pack_direct_chunks=True,
                num_xcds=1,
                reverse_local_assignment=False,
                waves_per_eu=1,
            )
    if 1024 <= m <= 1536 and n >= 48 * m and k == 1024:
        return _wg_regular_wave_grid_plan(
            8,
            5,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            num_xcds=8,
            workgroup_mapping=16,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    if 896 <= m <= 1152 and n >= 64 * m and k == 4096:
        return _wg_regular_wave_grid_plan(
            8,
            6,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            num_xcds=8,
            workgroup_mapping=8,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    square_tiles = ((m + 255) // 256) * ((n + 255) // 256)
    rectangular_tiles = ((m + 255) // 256) * ((n + 191) // 192)
    if (k == 4096 and m <= n < 2 * m and 256 < square_tiles <= 3 * _wg_NUM_CU // 2
            and 7 * _wg_NUM_CU // 4 <= rectangular_tiles <= 2 * _wg_NUM_CU):
        return _wg_regular_wave_grid_plan(
            6,
            8,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            direct_chunk=32,
            pack_direct_chunks=True,
            num_xcds=8,
            workgroup_mapping=8,
            row_wise_epilogue=True,
            reverse_local_assignment=False,
            sink_insts_to_avoid_spills=True,
            disable_unclustered_high_rp_reschedule=True,
        )
    return None


def _wg_prefer_tuned_wave_grid(m, n, k, plan):
    """Prefer measured non-power-of-two wave-grid decompositions."""
    return plan is not None and (
        (m, n, k) in _wg_SOTA77_WAVE_GRID_PROMOTIONS or (k == 8192 and 480 <= m <= 576 and n >= 128 * m) or
        (k == 6144 and 64 <= m <= 96 and n >= 32 * m) or (k == 6144 and 240 <= m <= 288 and n >= 32 * m) or
        (k == 6144 and 208 <= m <= 224 and n >= 65536 and n % 224 == 0) or
        (k == 512 and 160 <= m <= 240 and n >= 65536) or
        (k == 768 and plan.get("direct_to_lds", False) and plan.get("mi_wave_tile_m") == 8
         and plan.get("mi_wave_tile_n") == 5 and plan.get("warps_m") == 2 and plan.get("warps_n") == 2) or
        (k == 1024 and 640 <= m <= 800 and n >= 128 * m) or
        (k == 4096 and 384 <= m <= 448 and n >= 65536) or plan["kind"] == "regular_mi32_wave_grid" or
        (k == 256 and m >= 64 * n and plan.get("direct_to_lds", False) and plan.get("pack_direct_chunks", False)))


def _wg_prefer_lds_over_wave_grid(m, n, k, plan):
    """Keep measured compute families on the mature eight-wave pipeline."""
    if plan is None:
        return False
    if ((m, n, k) in _wg_SOTA77_WAVE_GRID_PROMOTIONS or (k == 8192 and 480 <= m <= 576 and n >= 128 * m)):
        return False
    # At K256, a very tall output with a moderately wide N dimension has
    # enough CTA parallelism that the mature eight-wave pipeline beats the
    # finer four-wave N128 decomposition despite its extra N padding.
    tall_short_k = k == 256 and m >= 64 * n and n >= 512
    wide_short_m = k >= 2048 and 224 <= m <= 256 and n >= 256 * m
    wide_shallow_k = k == 512 and 512 <= m <= 768 and n >= 64 * m
    narrow_deep_k = k == 1024 and 96 <= m <= 128 and n >= 512 * m
    return tall_short_k or wide_short_m or wide_shallow_k or narrow_deep_k or (plan.get("direct_to_lds", False)
                                                                               and k >= 4096 and m < 64 * n)


def _wg_prefer_m192n256(m, n, k):
    """Use MT192x256 when it improves device-wave fill without excess work."""
    wide_shallow = k == 512 and 768 < m <= 960 and n >= 96 * m
    if wide_shallow:
        return True
    tall_short = k == 768 and m >= 65536 and 768 < n <= 1664
    if tall_short:
        padded_m = ((m + 191) // 192) * 192
        grid_mn = ((m + 191) // 192) * ((n + 255) // 256)
        if padded_m * 1000 <= m * 1003 and grid_mn >= 4 * _wg_NUM_CU:
            return True
    square_tiles = ((m + 255) // 256) * ((n + 255) // 256)
    rectangular_tiles = ((m + 191) // 192) * ((n + 255) // 256)
    square_waves = (square_tiles + _wg_NUM_CU - 1) // _wg_NUM_CU
    rectangular_waves = (rectangular_tiles + _wg_NUM_CU - 1) // _wg_NUM_CU
    same_m_tile_count = (m + 191) // 192 == (m + 255) // 256
    if (k == 1024 and n >= 8 * m and square_waves == 4 and rectangular_waves == 5
            and 20 * rectangular_tiles * 3 <= 21 * square_tiles * 4):
        return True
    return (
        k >= 512 and k % 128 == 0 and rectangular_waves == square_waves
        and ((m > 192 and same_m_tile_count and square_waves >= 3) or (k >= 4096 and (square_waves <= 3 or k >= 16384)))
        # Equal device-wave count is not sufficient: MT192 launches 4/3 as
        # many M tiles as MT256.  Require its total padded output work to be
        # lower as well.  This keeps deep-K shapes such as M=2032 on MT256,
        # where 11 MT192 rows do more work than 8 MT256 rows.
        and rectangular_tiles * 3 < square_tiles * 4)


def _wg_prefer_streamk(m, n, k):
    """Use Stream-K only for a sparse final device wave and deep reduction."""
    grid_m = (m + 255) // 256
    grid_n = (n + 255) // 256
    tile_count = grid_m * grid_n
    tail_tiles = tile_count % _wg_NUM_CU
    k_pipe_pairs = (k // 64) // 2
    arithmetic_intensity = m * n * k / (m * k + k * n + m * n)
    has_streamk_work = (k % 128 == 0 and tile_count >= _wg_NUM_CU and tail_tiles > 0
                        and tail_tiles * k_pipe_pairs >= _wg_NUM_CU)
    return (k >= 8192 and arithmetic_intensity >= 800 and 2 * max(m * k, k * n) < 2**31
            and tail_tiles <= 5 * _wg_NUM_CU // 8 and has_streamk_work)


def _wg_prefer_ragged_n(m, n, k, plan):
    """Return whether a separate narrow N-tail launch saves enough work."""
    n_tail = n % 256
    padded_n = ((n + 255) // 256) * 256
    return (plan is not None and plan.get("direct_to_lds", False) and k == 4096 and m >= 8 * n and 0 < n_tail <= 64
            and 8 * (256 - n_tail) >= padded_n)


_wave_grid_plan_for_shape = _wg_plan_for_shape
_launch_wave_grid = _wg_matmul
prefer_lds_over_wave_grid = _wg_prefer_lds_over_wave_grid
prefer_m192n256 = _wg_prefer_m192n256
prefer_ragged_n = _wg_prefer_ragged_n
prefer_streamk = _wg_prefer_streamk
prefer_tuned_wave_grid = _wg_prefer_tuned_wave_grid
supports_hybrid_n160 = _wg_supports_hybrid_n160
transposed_wave_grid_plan = _wg_transposed_wave_grid_plan

# Dense row-major-B direct-to-LDS path.


def _row_major_direct_plan(block_m, block_n, block_k, num_buffers, num_warps, matrix_instr_nonkdim):
    """Build one measured plan for the shared row-major direct pipeline."""
    return MappingProxyType({
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "NUM_BUFFERS": num_buffers,
        "num_warps": num_warps,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "A_BASES": tuple(tuple(base) for base in _wg_swizzled_offset_bases((block_m, block_k), 1)),
        "B_BASES": tuple(tuple(base) for base in _wg_swizzled_offset_bases((block_k, block_n), 1)),
    })


_ROW_MAJOR_DIRECT_PLANS = {
    # Three LDS buffers are important for this short, wide-N family: two
    # buffers leave an exposed LDSDMA bubble after every dot iteration.
    (1024, 6144, 4096):
    _row_major_direct_plan(128, 256, 64, 3, 8, 32),
    (1024, 20480, 6144):
    _row_major_direct_plan(128, 256, 64, 3, 8, 32),
    # K256 needs smaller K32 transfers; the exact N256 tile avoids all output
    # padding while preserving the same three-buffer pipeline.
    (2252800, 256, 256):
    _row_major_direct_plan(128, 256, 32, 3, 8, 32),
}

# Dense row-major-B problems that are faster with the regular MI16 wave-grid
# pipeline than with the larger eight-wave direct kernel.  The common
# wave-grid kernel already carries arbitrary compile-time B strides; keep the
# row-major selection explicit so unmeasured layouts do not enter this path.
_ROW_MAJOR_WAVE_GRID_PLANS = {
    # This 397-block reduction reuses the parity-safe odd PGR2 tail.  The
    # MT192x256 grid keeps enough CTAs for M2048 while preserving B's native
    # N-contiguous vectors; it is consistently faster than both the generic
    # register fallback and the vendor's wider MT192x288 macro tile.
    (2048, 10240, 25408):
    _wg_regular_wave_grid_plan(
        6,
        8,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # The vendor uses a four-wave MT256x256 PGR2 kernel for this wide output.
    # Run the complete K64 prefix through the two-stage pipeline, then append
    # the 38-element tail directly into the resident accumulator set. Keeping
    # the tail coordinates opaque rematerializes its addresses after the loop
    # and avoids spilling prologue offsets across all 29 K64 blocks.
    (4096, 242432, 1894):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=8,
        workgroup_mapping=8,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # Preserve B's native N-contiguous vectors through LDS.  With PGR2 this
    # removes the long MI32 dependency chains of the register fallback; the
    # row-wise wide epilogue then amortizes the four-wave MT256x256 stores.
    (4096, 4096, 2048):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=1,
        workgroup_mapping=4,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # This large output grid uses the same N-contiguous LDS pipeline.  Linear
    # single-XCD mapping keeps each long N row together and beats the previous
    # four-wave register fallback without requiring a shape-specific kernel.
    (61440, 5120, 2048):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # K7744 has 121 K64 blocks.  The parity-safe PGR2 tail keeps the same
    # four-wave pipeline effective for this deep reduction, and linear
    # single-XCD mapping is consistently faster than the register fallback.
    (61440, 5120, 7744):
    _wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
    # The exact N192 tile preserves B reuse, while MT320 amortizes its loads
    # and epilogue across 20% fewer workgroups than MT256 on this huge-M case.
    (819200, 192, 1024):
    _wg_regular_wave_grid_plan(
        10,
        6,
        2,
        2,
        64,
        pgr2_operands=True,
        local_stages=2,
        # B is physically row-major here.  Preserve its N-contiguous load
        # vectors through LDS instead of scalarizing them into K-major banks.
        row_major_b_lds=True,
        wide_epilogue=True,
        num_xcds=8,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    ),
}


@triton.jit
def _row_major_direct_remap(program, program_count, num_xcds: tl.constexpr, tiles_per_problem: tl.constexpr):
    """Preserve the proven shared-A BMM program mapping for batch one."""
    aligned = (program_count // (num_xcds * tiles_per_problem)) * (num_xcds * tiles_per_problem)
    if program >= aligned:
        return program
    xcd = program % num_xcds
    local_program = program // num_xcds
    return (local_program // tiles_per_problem) * num_xcds * tiles_per_problem + xcd * tiles_per_problem + (
        local_program % tiles_per_problem)


@triton.jit
def _row_major_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    TILES_PER_PROBLEM: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    A_BASES: tl.constexpr,
    B_BASES: tl.constexpr,
):
    """Triple-buffered direct-to-LDS GEMM for dense row-major B."""
    k_tiles = tl.cdiv(K, BLOCK_K)

    grid_n = tl.cdiv(N, BLOCK_N)
    remapped_program = _row_major_direct_remap(tl.program_id(0), PROGRAM_COUNT, NUM_XCDS, TILES_PER_PROBLEM)
    batch_id = remapped_program // TILES_PER_PROBLEM
    tile_id = remapped_program % TILES_PER_PROBLEM
    pid_m = tile_id // grid_n
    pid_n = tile_id % grid_n

    a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], A_BASES, [BLOCK_M, BLOCK_K]))
    b_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases([(512, 16)], B_BASES, [BLOCK_K, BLOCK_N]))
    a_local = tlx.local_alloc(
        (BLOCK_M, BLOCK_K),
        tlx.dtype_of(a_ptr),
        NUM_BUFFERS,
        layout=a_layout,
    )
    b_local = tlx.local_alloc(
        (BLOCK_K, BLOCK_N),
        tlx.dtype_of(b_ptr),
        NUM_BUFFERS,
        layout=b_layout,
    )

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    safe_rows = rows % M
    safe_cols = cols % N
    rk = tl.arange(0, BLOCK_K)
    a_ptr += batch_id.to(tl.int64) * stride_ab
    b_ptr += batch_id.to(tl.int64) * stride_bb
    a_offsets = safe_rows[:, None] * stride_am
    b_offsets = safe_cols[None, :] * stride_bn

    for stage in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_offset = stage * BLOCK_K
        tlx.buffer_load_to_local(
            tlx.local_view(a_local, stage),
            a_ptr,
            a_offsets + (k_offset + rk[None, :]) * stride_ak,
        )
        tlx.buffer_load_to_local(
            tlx.local_view(b_local, stage),
            b_ptr,
            (k_offset + rk[:, None]) * stride_bk + b_offsets,
        )
        tlx.async_load_commit_group()

    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    a = tlx.local_load(tlx.local_view(a_local, 0))
    b = tlx.local_load(tlx.local_view(b_local, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in tl.range(0, k_tiles - NUM_BUFFERS):
        next_stage = (k_tile + 1) % NUM_BUFFERS
        refill_stage = k_tile % NUM_BUFFERS
        refill_k = (k_tile + NUM_BUFFERS) * BLOCK_K
        acc = tl.dot(a, b, acc)
        tlx.buffer_load_to_local(
            tlx.local_view(a_local, refill_stage),
            a_ptr,
            a_offsets + (refill_k + rk[None, :]) * stride_ak,
        )
        tlx.buffer_load_to_local(
            tlx.local_view(b_local, refill_stage),
            b_ptr,
            (refill_k + rk[:, None]) * stride_bk + b_offsets,
        )
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(NUM_BUFFERS - 2)
        a = tlx.local_load(tlx.local_view(a_local, next_stage))
        b = tlx.local_load(tlx.local_view(b_local, next_stage))

    acc = tl.dot(a, b, acc)
    tlx.async_load_wait_group(0)
    for tail in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        stage = (k_tiles - (NUM_BUFFERS - 1) + tail) % NUM_BUFFERS
        acc = tl.dot(
            tlx.local_load(tlx.local_view(a_local, stage)),
            tlx.local_load(tlx.local_view(b_local, stage)),
            acc,
        )

    c_ptr += batch_id.to(tl.int64) * stride_cb
    offsets = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(
        offsets,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _launch_row_major_direct(a, b, out, plan):
    """Launch one validated dense row-major-B direct-to-LDS plan."""
    m, k = a.shape
    _, n = b.shape
    block_m = plan["BLOCK_M"]
    block_n = plan["BLOCK_N"]
    block_k = plan["BLOCK_K"]
    if k % block_k != 0 or k // block_k < plan["NUM_BUFFERS"]:
        raise ValueError("row-major direct-to-LDS plan requires a full, sufficiently deep "
                         f"K loop; got K={k}, BLOCK_K={block_k}, "
                         f"NUM_BUFFERS={plan['NUM_BUFFERS']}")
    tiles_per_problem = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    program_count = tiles_per_problem
    grid = (program_count, )
    _row_major_direct_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        0,
        a.stride(0),
        a.stride(1),
        0,
        b.stride(0),
        b.stride(1),
        0,
        out.stride(0),
        out.stride(1),
        NUM_XCDS=1,
        TILES_PER_PROBLEM=tiles_per_problem,
        PROGRAM_COUNT=program_count,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        num_stages=1,
        **plan,
    )
    return out


# Register-resident path.

_BLOCK_M = 256
_BLOCK_K = 64
_NUM_CU = 256
_MIN_KTILES_PER_SPLIT = 16


def _fixed_register_plan(block_m, block_n, block_k, group_m, num_xcds, num_warps, num_stages, matrix_instr_nonkdim=16,
                         waves_per_eu=0):
    return MappingProxyType({
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": num_xcds,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "waves_per_eu": waves_per_eu,
        "kpack": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })


_SMALL_SQUARE_REGISTER_CONFIG = _fixed_register_plan(32, 16, 256, 4, 1, 2, 2)
_MT64X64_BK256_REGISTER_CONFIG = _fixed_register_plan(64, 64, 256, 4, 8, 8, 2)
_TUNED_SHAPE_CONFIGS = {
    (2048, 256, 1024): _MT64X64_BK256_REGISTER_CONFIG,
    (2041, 2041, 2048): _fixed_register_plan(128, 128, 128, 16, 8, 8, 2),
}

_FP16_TUNED_SHAPE_CONFIGS = {
    (256, 257, 4096):
    _SMALL_SQUARE_REGISTER_CONFIG,
    (257, 257, 4096):
    _SMALL_SQUARE_REGISTER_CONFIG,
    (272, 3072, 4608):
    _MT64X64_BK256_REGISTER_CONFIG,
    (279, 2048, 4096):
    _fixed_register_plan(64, 64, 128, 16, 8, 8, 3),
    # Transposing the rectangular macro tile raises XCD-level parallelism for
    # this moderately tall problem while MI32 keeps its MFMA count compact.
    (677, 2048, 4096):
    _fixed_register_plan(64, 128, 128, 2, 8, 8, 3, matrix_instr_nonkdim=32),
    # MI32 halves the static MFMA count for this balanced row-major-B shape.
    # A two-tile M grouping preserves locality without the cross-XCD remap
    # that was slower for the compact 16x16 output grid. Three paired runs
    # measured it 0.3--1.4% faster than the four-tile grouping.
    (4096, 4096, 2048):
    _fixed_register_plan(256, 256, 64, 2, 1, 8, 2, matrix_instr_nonkdim=32),
    # The deeper K tile amortizes loop/LDS overhead, while the 128x64 output
    # tile keeps enough CTAs for this low-M, wide-N problem.
    (279, 4096, 4352):
    _fixed_register_plan(128, 64, 128, 8, 8, 8, 3),
    # A four-wave K32 tile exposes twice as many independent CTAs as the
    # generic eight-wave square plan.  That occupancy gain outweighs the
    # additional K-loop iterations for this large-M rectangular problem.
    (61440, 3840, 4096):
    _fixed_register_plan(128, 256, 32, 16, 8, 4, 2),
    (61440, 5120, 2048):
    _fixed_register_plan(128, 256, 32, 8, 8, 4, 2),
    (61440, 5120, 7744):
    _fixed_register_plan(128, 256, 32, 16, 8, 4, 2),
    # The MI32 instruction halves the static MFMA count for this moderate-K,
    # very-wide-N problem while retaining the same macro tile and K-tail path.
    (4096, 242432, 1894):
    _fixed_register_plan(256, 256, 64, 8, 8, 8, 2, matrix_instr_nonkdim=32),
}


@triton.jit
def _short_k_register_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
):
    """Direct-register GEMM for one masked K64 tile."""
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 64
    block_k: tl.constexpr = 64
    grid_n: tl.constexpr = tl.cdiv(N, block_n)
    pid = tl.program_id(0).to(tl.int32)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rows = pid_m * block_m + tl.arange(0, block_m).to(tl.int32)
    cols = pid_n * block_n + tl.arange(0, block_n).to(tl.int32)
    input_rows = rows if M % block_m == 0 else tl.where(rows < M, rows, 0)
    input_cols = cols if N % block_n == 0 else tl.where(cols < N, cols, 0)
    reduction = tl.arange(0, block_k).to(tl.int32)
    reg_m = tl.max_contiguous(tl.multiple_of(input_rows, block_m), block_m)
    reg_n = tl.max_contiguous(tl.multiple_of(input_cols, block_n), block_n)
    reduction_mask = reduction < K
    a = tl.load(
        a_ptr + reg_m[:, None] * stride_am + reduction[None, :] * stride_ak,
        mask=reduction_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr + reduction[:, None] * stride_bk + reg_n[None, :] * stride_bn,
        mask=reduction_mask[:, None],
        other=0.0,
    )
    acc = tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    output_ptrs = (c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn)
    if M % block_m == 0 and N % block_n == 0:
        tl.store(output_ptrs, acc)
    else:
        tl.store(
            output_ptrs,
            acc,
            mask=(rows[:, None] < M) & (cols[None, :] < N),
        )


def _launch_short_k_register(a, b, *, out=None):
    """Launch the bounded one-K64 direct-register family."""
    m, k = a.shape
    b_k, n = b.shape
    if k != b_k or not 0 < k <= 64:
        raise ValueError("short-K register GEMM requires matching 0 < K <= 64")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64), )
    _short_k_register_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        num_warps=1,
        num_stages=1,
        matrix_instr_nonkdim=32,
        waves_per_eu=0,
    )
    return out


@triton.jit
# Triton TR001: callers select either a measured fixed plan or an autotuned one.
def _register_kernel_impl(  # noqa: TR001
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    WRITE_STATS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    CACHE_A_CG: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_mn = grid_m * grid_n

    xcd_chunk: tl.constexpr = 4
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * xcd_chunk)) * (NUM_XCDS * xcd_chunk)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // xcd_chunk) * NUM_XCDS * xcd_chunk + xcd * xcd_chunk + local_pid % xcd_chunk)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    input_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    input_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    offs_m = (input_rows if M % BLOCK_M == 0 else tl.where(input_rows < M, input_rows, 0))
    offs_n = (input_cols if N % BLOCK_N == 0 else tl.where(input_cols < N, input_cols, 0))
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    full_k_tiles: tl.constexpr = K // BLOCK_K
    for k_idx in range(0, full_k_tiles):
        k = k_idx * BLOCK_K
        a_ptrs = (a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = (b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn)
        if CACHE_A_CG:
            a = tl.load(a_ptrs, cache_modifier=".cg")
        else:
            a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    if K % BLOCK_K != 0:
        k = full_k_tiles * BLOCK_K
        k_mask = offs_k < K - k
        a_ptrs = (a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = (b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rows[:, None]
    idx_n = cols[None, :]
    mask = (idx_m < M) & (idx_n < N)
    if ADD_BIAS:
        bias_offsets = idx_m * stride_bias_m + idx_n * stride_bias_n
        bias = tl.load(
            bias_ptr + bias_offsets,
            mask=mask,
            eviction_policy="evict_last",
        )
        acc += bias.to(tl.float32)
    output_offsets = idx_m * stride_cm + idx_n * stride_cn
    if WRITE_STATS:
        value = acc.to(c_ptr.dtype.element_ty)
        value_fp32 = tl.where(mask, value.to(tl.float32), 0.0)
        stats_offsets = rows * grid_n + pid_n
        if not IS_RMS_NORM:
            tl.store(
                row_sum_ptr + stats_offsets,
                tl.sum(value_fp32, axis=1),
                mask=rows < M,
            )
        tl.store(
            row_sum_sq_ptr + stats_offsets,
            tl.sum(value_fp32 * value_fp32, axis=1),
            mask=rows < M,
        )
        tl.store(c_ptr + output_offsets, value, mask=mask)
    else:
        # Preserve the original no-stats epilogue. Let store lowering perform
        # the output conversion directly from the accumulator; materializing a
        # separate truncation costs several percent on short-K register plans.
        tl.store(c_ptr + output_offsets, acc, mask=mask)


def _launch_register_plan(a, b, *, config, bias=None, out=None, _validated=False):
    """Launch one validated register-resident plan."""
    m, k = a.shape
    b_k, n = b.shape
    if not _validated and k != b_k:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and "
                         f"{tuple(b.shape)}")
    if bias is not None:
        if bias.shape != (m, n):
            raise ValueError(f"Bias must expand to ({m}, {n}), got {tuple(bias.shape)}")
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("Bias and matrix operands must have matching device and dtype")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    disable_agpr = (k == 256 and n > 256) or (k > 512 and (k % _BLOCK_K != 0 or m * n <= 2 * 1024 * 1024))
    launch_options = ({"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), )} if disable_agpr else {})
    if config["BLOCK_K"] == 128 and config["num_stages"] == 3:
        launch_options["reverse_local_assignment"] = True
        if config["BLOCK_M"] == 64 and config["BLOCK_N"] == 64:
            launch_options["sink_insts_to_avoid_spills"] = True
            launch_options["disable_unclustered_high_rp_reschedule"] = True
    bias_ptr = bias if bias is not None else out
    grid = (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]), )
    _register_kernel_impl[grid](
        a,
        b,
        bias_ptr,
        out,
        out,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
        ADD_BIAS=bias is not None,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        **config,
        **launch_options,
    )
    return out


_RANGE_REGISTER_COMPILED_CACHE = None
_RANGE_REGISTER_COMPILED_CACHE_LIMIT = 64


def _range_register_compiled_cache():
    global _RANGE_REGISTER_COMPILED_CACHE
    if _RANGE_REGISTER_COMPILED_CACHE is None:
        _RANGE_REGISTER_COMPILED_CACHE = {}
    return _RANGE_REGISTER_COMPILED_CACHE


def _can_use_range_register_compiled_cache():
    """Keep instrumentation and forced compilation on the normal JIT path."""
    return not (triton.knobs.compilation.always_compile or _register_kernel_impl.pre_run_hooks
                or _register_kernel_impl.used_global_vals or triton.knobs.runtime.add_stages_inspection_hook is not None
                or triton.knobs.runtime.launch_enter_hook or triton.knobs.runtime.launch_exit_hook
                or _register_kernel_impl.launch_metadata or os.environ.get("TRITON_DUMP_TLX_BENCHMARK")
                or os.environ.get("TRITON_COMPILE_IQ_COLLECT"))


def _range_register_cache_key(a, b, out, config, launch_options):
    m, k = a.shape
    n = b.shape[1]
    return (
        a.device,
        a.dtype,
        m,
        n,
        k,
        a.stride(),
        b.stride(),
        out.stride(),
        a.data_ptr() % 16,
        b.data_ptr() % 16,
        out.data_ptr() % 16,
        tuple(sorted(config.items())),
        tuple(sorted(launch_options.items())),
    )


def _run_range_register_compiled(compiled, grid, compiled_args):
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    compiled.run(
        grid[0],
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *compiled_args,
    )


def _remember_range_register_compiled(cache_key, compiled):
    cache = _range_register_compiled_cache()
    if len(cache) >= _RANGE_REGISTER_COMPILED_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[cache_key] = compiled


def _launch_range_register_plan(a, b, *, config, out=None, _use_compiled_cache=None):
    """Launch one unsplit register configuration from the supported range."""
    m, k = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]), )
    runtime_args = (
        a,
        b,
        out,
        out,
        out,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0,
        0,
        out.stride(0),
        out.stride(1),
    )
    launch_options = _range_register_launch_options(a, b, config)
    compiled_args = runtime_args + (
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_M"],
        config["NUM_XCDS"],
        False,
        False,
        False,
        launch_options.get("CACHE_A_CG", False),
    )
    use_compiled_cache = (_can_use_range_register_compiled_cache()
                          if _use_compiled_cache is None else _use_compiled_cache)
    cache_key = None
    if use_compiled_cache:
        cache_key = _range_register_cache_key(a, b, out, config, launch_options)
        compiled = _range_register_compiled_cache().get(cache_key)
        if compiled is not None:
            _run_range_register_compiled(compiled, grid, compiled_args)
            return out
    compiled = _register_kernel_impl[grid](
        *runtime_args,
        ADD_BIAS=False,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        **config,
        **launch_options,
    )
    if use_compiled_cache:
        _remember_range_register_compiled(cache_key, compiled)
    return out


@lru_cache(maxsize=None)
def _range_family_candidates(m, n, k, dtype, element_size, a_strides, b_strides):
    """Return bounded unsplit candidates for contiguous-A GEMM families."""
    if (dtype not in (torch.float16, torch.bfloat16) or min(m, n, k) <= 0 or a_strides[1] != 1
            or min(*a_strides, *b_strides) <= 0):
        return ()
    if _supports_short_k_register(m, n, k, dtype):
        return (("short_k_register", None), )
    max_offset = max((m - 1) * a_strides[0] + k - 1, (k - 1) * b_strides[0] + (n - 1) * b_strides[1], m * n)
    if not 128 <= k <= 4096 or n < 128 or max_offset * element_size >= 2**31:
        return ()

    def register(bm, bn, bk, group, xcds, warps, stages, waves=0):
        return ("range_register", _fixed_register_plan(bm, bn, bk, group, xcds, warps, stages, waves_per_eu=waves))

    if dtype == torch.bfloat16 and m <= 32:
        return (register(16, 16, 256, 1, 4, 2, 3), register(32, 32, 256, 4, 8, 4, 3, 2))
    if dtype == torch.bfloat16 and m <= 128:
        return (register(32, 32, 256, 4, 8, 4, 3, 2), register(64, 32, 256, 4, 8, 4, 3, 2))
    if dtype == torch.bfloat16 and k % 64:
        return (register(64, 128, 64, 4, 1, 8, 3), register(128, 64, 64, 4, 1, 4, 3))
    if b_strides[1] != 1:
        if dtype == torch.bfloat16 and b_strides[0] != 1:
            return (register(128, 64, 64, 4, 1, 4, 3), register(64, 128, 64, 4, 1, 8, 3))
        return ()
    if dtype == torch.float16:
        if not (256 <= m < 8192 and 256 <= n <= 8192 and 512 <= k and k % 64 == 0):
            return ()
        return (register(128, 128, 64, 16, 8, 4, 2), register(256, 256, 64, 8, 8, 8, 2))

    square = register(256, 256, 64, min(triton.cdiv(m, 256), 256) if n % 256 == 0 else 8, 1, 8, 2)
    compact = register(128, 128, 64, 16, 8, 4, 3)
    direct_legal = (k % 128 == 0 and k >= 512 and n % 8 == 0 and a_strides[0] % 8 == 0 and b_strides[0] % 8 == 0)
    if not direct_legal:
        return (compact, square)
    if n <= 256 and m >= 16 * n:
        mi_n = min(8, triton.next_power_of_2(triton.cdiv(n, 32)))
        if n % (32 * mi_n):
            return (compact, square)
        narrow = _wg_regular_wave_grid_plan(5, mi_n, 2, 2, 64, local_stages=1, pgr2_operands=True,
                                            row_wise_epilogue=True, num_xcds=1, workgroup_mapping=1,
                                            reverse_local_assignment=True, sink_insts_to_avoid_spills=True,
                                            disable_unclustered_high_rp_reschedule=True, waves_per_eu=1)
        return (("wave_grid", narrow), compact, square)
    wide = _wg_regular_wave_grid_plan(8, 8, 2, 2, 64, local_stages=2, pgr2_operands=True, direct_to_lds=True,
                                      direct_chunk=128, row_major_b_lds=True, row_wise_epilogue=True,
                                      wide_epilogue=True, num_xcds=8, workgroup_mapping=6,
                                      reverse_local_assignment=False, sink_insts_to_avoid_spills=True,
                                      disable_unclustered_high_rp_reschedule=True)
    if m >= 8 * n and n % 256 == 0:
        return (square, ("wave_grid", wide), compact)
    return (("wave_grid", wide), square, compact)


def _range_prefers_family(m, n, k, dtype, b_strides):
    if dtype == torch.float16:
        return m >= 1024
    if m <= 128:
        return n <= 8192
    if k < 512:
        return False
    if k % 64:
        return m < 4096
    if b_strides[1] != 1:
        return True
    if m >= 8 * n:
        return k >= 1024
    return m >= 1024 and n >= 8 * k


@lru_cache(maxsize=64)
def _range_dispatch_candidates(m, n, k, dtype, element_size, a_strides, b_strides, *, include_incumbent=True):
    candidates = _range_family_candidates(m, n, k, dtype, element_size, a_strides, b_strides)
    if not candidates or candidates[0][0] == "short_k_register":
        return candidates
    prefer_family = _range_prefers_family(m, n, k, dtype, b_strides)
    if prefer_family and not include_incumbent:
        return candidates
    genuinely_strided = a_strides[1] != 1 or (b_strides[0] != 1 and b_strides[1] != 1)
    if genuinely_strided or (b_strides[0] != 1 and m < 1024):
        config = _unsplit_register_config_for_shape(m, n, k, None if genuinely_strided else dtype)
        incumbent = "register", config or _intermediate_register_config(m, n, k)
    else:
        incumbent = _incumbent_heuristic_config(m, n, k, dtype, element_size, a_strides, b_strides)
    if incumbent is None or incumbent[0] != "register":
        return candidates
    if prefer_family:
        return candidates + (incumbent, )
    return (incumbent, ) + candidates


def _range_dispatch_for(a, b):
    if (a.ndim != 2 or b.ndim != 2 or a.dtype != b.dtype or not a.is_cuda or a.device != b.device
            or a.shape[1] != b.shape[0] or _device_arch(a.device) != "gfx950"):
        return None
    candidates = _range_dispatch_candidates(a.shape[0], b.shape[1], a.shape[1], a.dtype, a.element_size(), a.stride(),
                                            b.stride(), include_incumbent=False)
    for path, plan in candidates:
        if path == "wave_grid" and (a.data_ptr() % 16 or b.data_ptr() % 16):
            continue
        return path, plan
    return None


def _range_register_launch_options(a, b, config):
    options = {}
    if a.dtype == torch.bfloat16 and config["BLOCK_K"] == 64:
        options["llvm_fn_attrs"] = (("amdgpu-agpr-alloc", "0,0"), )
        if (config["BLOCK_M"] >= 128 and b.shape[1] % config["BLOCK_N"] != 0 and b.stride(1) == 1
                and a.shape[0] >= 8 * b.shape[1]):
            options["CACHE_A_CG"] = True
        if config["BLOCK_M"] == config["BLOCK_N"] == 256:
            grid_m = triton.cdiv(a.shape[0], 256)
            grid_n = triton.cdiv(b.shape[1], 256)
            if grid_m >= 2 * grid_n:
                if b.shape[1] % 256:
                    options["reverse_local_assignment"] = True
                else:
                    options["disable_unclustered_high_rp_reschedule"] = True
    return options


_RANGE_TUNED_PLAN_CACHE = {}
_RANGE_TUNED_PLAN_CACHE_LIMIT = 64


def _range_tuned_plan_cache_key(a, b, out):
    return (a.device, a.dtype, tuple(a.shape), tuple(b.shape), a.stride(), b.stride(), out.stride(), a.data_ptr() % 16,
            b.data_ptr() % 16, out.data_ptr() % 16)


def _launch_range_autotuned(a, b, out, candidates):
    """Benchmark legal range candidates and cache the selected dispatch."""
    from triton import testing

    use_cache = _can_use_range_register_compiled_cache()
    key = _range_tuned_plan_cache_key(a, b, out)
    selected = _RANGE_TUNED_PLAN_CACHE.get(key) if use_cache else None
    if selected is None:
        executable = []
        for dispatch in candidates:
            if dispatch[0] == "wave_grid" and (a.data_ptr() % 16 or b.data_ptr() % 16):
                continue
            try:
                _launch_dispatch(a, b, out, dispatch)
            except triton.OutOfResources:
                continue
            executable.append(dispatch)
        if not executable:
            raise InvalidInput("no executable gfx950 GEMM candidate in the selected range")
        timings = [[] for _ in executable]
        for round_index in range(3 if use_cache else 1):
            offset = round_index % len(executable)
            for index in list(range(offset, len(executable))) + list(range(offset)):

                def launch(dispatch=executable[index]):
                    return _launch_dispatch(a, b, out, dispatch)

                if use_cache:
                    elapsed = testing.do_bench_cudagraph(launch, rep=10, return_mode="median")
                else:
                    elapsed = testing.do_bench(launch, warmup=5, rep=10, return_mode="median")
                timings[index].append(elapsed)
        winner = min(range(len(executable)), key=lambda index: sorted(timings[index])[len(timings[index]) // 2])
        if winner and not all(candidate < default for candidate, default in zip(timings[winner], timings[0])):
            winner = 0
        selected = executable[winner]
        if use_cache:
            if len(_RANGE_TUNED_PLAN_CACHE) >= _RANGE_TUNED_PLAN_CACHE_LIMIT:
                _RANGE_TUNED_PLAN_CACHE.pop(next(iter(_RANGE_TUNED_PLAN_CACHE)))
            _RANGE_TUNED_PLAN_CACHE[key] = selected
    return _launch_dispatch(a, b, out, selected)


def _register_split_k_for(grid_mn, k):
    min_ks = _MIN_KTILES_PER_SPLIT * _BLOCK_K
    best = 1
    for split_k in range(2, _NUM_CU // grid_mn + 1):
        split_size = k // split_k
        if (k % split_k == 0 and split_size >= min_ks and split_size % _BLOCK_K == 0):
            best = split_k
    return best


def _default_lds_block_m(m, n, k):
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    large_fill = large_grid * _register_split_k_for(large_grid, k)
    if large_fill >= _NUM_CU // 2:
        return 256
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    small_fill = small_grid * _register_split_k_for(small_grid, k)
    return 128 if small_fill > large_fill else 256


def _register_config_for(m, n, k):
    small_grid = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    large_grid = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    if not (k > 512 and k % _BLOCK_K == _BLOCK_K // 2 and large_grid < _NUM_CU <= small_grid):
        return None
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "NUM_XCDS": 8,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 4,
        "num_stages": 4,
    }


def _balanced_short_k_register_config(m, n, k):
    """Select the register pipeline for balanced K256/K512 GEMMs."""
    if (k not in (256, 512) or min(m, n) < 2048 or max(m, n) > 2 * min(m, n)):
        return None
    if k == 512:
        grid_mn_128 = triton.cdiv(m, 128) * triton.cdiv(n, 128)
        if grid_mn_128 > 5 * _NUM_CU:
            return None
        if (grid_mn_128 >= 2 * _NUM_CU and 2 * max(m, n) < 3 * min(m, n)):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 16,
                "NUM_XCDS": 8,
                "matrix_instr_nonkdim": 16,
                "waves_per_eu": 0,
                "kpack": 1,
                "num_warps": 8,
                "num_stages": 3,
            }
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "NUM_XCDS": 8,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": 0,
            "kpack": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    grid_mn = triton.cdiv(m, 256) * triton.cdiv(n, 256)
    if _NUM_CU < grid_mn < 4 * _NUM_CU:
        return None
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 8 if grid_mn >= 4 * _NUM_CU else 16,
        "NUM_XCDS": 8,
        "matrix_instr_nonkdim": 32 if grid_mn <= _NUM_CU else 16,
        "waves_per_eu": 1 if grid_mn <= _NUM_CU else 0,
        "kpack": 1,
        "num_warps": 8,
        "num_stages": 2,
    }


def _tall_skinny_short_k_register_config(m, n, k):
    """Select a compact register pipeline for tall, narrow K512 GEMMs."""
    if k != 512 or not 64 <= n <= 128 or m < 512 * n:
        return None
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "NUM_XCDS": 8,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 4,
        "num_stages": 2,
    }


def _intermediate_register_config(m, n, k):
    if n >= 2 * k:
        block_m, block_n, block_k = 128, 128, 128
        group_m, num_warps, num_stages = 16, 8, 2
    elif k >= 2 * n and 4 * m < 3 * triton.cdiv(m, 128) * 128:
        block_m, block_n, block_k = 64, 32, 128
        group_m, num_warps, num_stages = 8, 4, 2
    elif k >= 2 * n:
        block_m, block_n, block_k = 128, 64, 128
        group_m, num_warps, num_stages = 4, 8, 3
    else:
        block_m, block_n, block_k = 128, 64, 64
        group_m, num_warps, num_stages = 4, 4, 3
    grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": 8 if grid_mn >= _NUM_CU else 1,
        "matrix_instr_nonkdim": 16 if block_m == 64 else 32,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _unsplit_register_config_for_shape(m, n, k, dtype=None):
    """Select a register configuration without estimating Split-K coverage."""
    shape = (m, n, k)
    tuned = _TUNED_SHAPE_CONFIGS.get(shape)
    if dtype == torch.float16:
        tuned = _FP16_TUNED_SHAPE_CONFIGS.get(shape, tuned)
    if tuned is not None:
        return tuned

    config = _register_config_for(m, n, k)
    if config is not None:
        return MappingProxyType(config)
    config = _balanced_short_k_register_config(m, n, k)
    if config is not None:
        return MappingProxyType(config)
    config = _tall_skinny_short_k_register_config(m, n, k)
    if config is not None:
        return MappingProxyType(config)
    return None


def _register_plan_for_shape(m, n, k, dtype=None):
    """Return the bounded register plan selected by the gfx950 geometry."""
    config = _unsplit_register_config_for_shape(m, n, k, dtype)
    if config is not None:
        return config

    block_m = _default_lds_block_m(m, n, k)
    padded_m = triton.cdiv(m, block_m) * block_m
    is_intermediate_m = _BLOCK_M // 4 < m < 4 * _BLOCK_M
    has_high_m_padding = 4 * m < 3 * padded_m
    if not is_intermediate_m or (block_m == _BLOCK_M and not has_high_m_padding):
        return None
    return MappingProxyType(_intermediate_register_config(m, n, k))


# Direct-to-LDS and Stream-K paths.

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

MIN_K = 2 * BLOCK_K  # pipeline prefetches 2 whole K-tiles; the rest goes to the masked tail
KERNEL_NAME = "a16w16_8wave"
_LLVM_ATTRS = (("amdgpu-agpr-alloc", "0,0"), )
_READY_VALUE = 3


def _prune_register_configs(configs, named_args, **_):
    k = named_args["K"]
    if 128 <= k < 256 and named_args["M"] >= 16384 and named_args["N"] <= 128:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 1 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 64 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 4
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 1536 and named_args["M"] == 3072 and named_args["N"] == 3072:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 8 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 16
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 256 and named_args["M"] <= 1024 and named_args["N"] >= 16384:
        preferred = [
            config for config in configs
            if config.kwargs["BLOCK_M"] == 256 and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 32
            and config.kwargs["GROUP_M"] == 4 and config.kwargs["waves_per_eu"] == 2 and config.num_warps == 4
        ]
        if preferred:
            return preferred
    return configs


# Triton TR001: autotune the register path across stock ROCm tiles and the
# deeper software-pipelined tiles used by the TorchTLX register path.
_REGISTER_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=2,
    ) for block_m, block_n, block_k, group_m, num_warps, waves_per_eu in (
        (16, 16, 256, 4, 4, 2),
        (32, 16, 256, 4, 4, 0),
        (32, 32, 16, 8, 4, 2),
        (32, 32, 128, 8, 4, 0),
        (32, 64, 64, 8, 4, 0),
        (64, 16, 128, 8, 4, 2),
        (64, 32, 32, 8, 4, 0),
        (64, 32, 64, 8, 4, 0),
        (64, 32, 64, 8, 8, 0),
        (64, 32, 128, 8, 4, 0),
        (64, 64, 16, 8, 4, 0),
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 16, 8, 0),
        (64, 64, 256, 4, 8, 0),
        (64, 128, 32, 4, 4, 2),
        (64, 128, 32, 8, 8, 0),
        (64, 128, 64, 4, 8, 0),
        (64, 128, 128, 4, 8, 0),
        (128, 32, 32, 8, 4, 0),
        (128, 32, 64, 8, 4, 0),
        (128, 64, 32, 8, 4, 2),
        (128, 64, 64, 16, 4, 0),
        (128, 64, 128, 4, 8, 0),
        (128, 128, 32, 16, 4, 2),
        (128, 128, 32, 16, 8, 0),
        (128, 128, 32, 16, 8, 2),
        (128, 128, 64, 16, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (128, 128, 128, 16, 8, 0),
        (128, 256, 32, 16, 4, 2),
        (128, 256, 64, 4, 8, 0),
        (256, 64, 64, 4, 8, 0),
        (256, 128, 32, 4, 4, 2),
        (256, 128, 32, 16, 8, 0),
        (256, 128, 64, 4, 8, 0),
        (256, 256, 64, 4, 8, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 64, 64, 16, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (128, 256, 64, 8, 8, 3, 0),
        (128, 256, 64, 8, 8, 3, 1),
        (256, 128, 32, 4, 4, 3, 2),
        (256, 128, 32, 4, 4, 4, 2),
        (256, 256, 64, 4, 8, 3, 0),
        (256, 256, 64, 4, 8, 4, 0),
        (128, 256, 32, 8, 8, 3, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 8,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 2, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (256, 128, 32, 4, 4, 3, 2),
    )
]

# Coalesced SIMD register layout for the [HALF_M, HALF_N] = [128, 128] fp16 quadrant
# store (num_warps=8, warp_size=64): each thread holds 8 contiguous N elements ->
# 128-bit buffer_store_dwordx4. Applied to the epilogue store via tlx.require_layout
# so tritongpu-coalesce sets the store to this #linear layout and AMD
# OptimizeEpilogue leaves it alone (it only rewrites #blocked stores) -- keeping the
# wide coalesced store instead of the narrow MMA-accumulator (dwordx2) fallback.
_C_STORE_SIMD_LAYOUT = tlx.layout(shape=((16, 4, 8), (8, 4)), stride=((8, 128, 512), (1, 4096)))


def _swz_offset_bases(shape, contig_dim):
    """Padded-shared swizzle offset bases for a 2D fp16 half-tile, derived from the
    tile shape so both tile sizes share one path (no per-size branch).

    `contig_dim` is the K-contiguous axis (0 or 1); its bits come first (fastest),
    then the free axis contributes its high bits (>= bit 4) before its low bits --
    the row/col permutation that makes the direct-to-LDS ds_reads bank-conflict-free
    on the 128x64 / 64x128 halves. A 128-wide free axis simply carries the extra top
    bit ([64,0] resp. [0,64]) that a 64-wide one omits. Used for both operands: the
    a half-tile [HALF_M, BLOCK_K] has K on dim 1, the b half-tile [BLOCK_K, HALF_N]
    has K on dim 0."""

    def basis(dim, i):
        return [1 << i, 0] if dim == 0 else [0, 1 << i]

    free_dim = 1 - contig_dim
    # log2 of each extent: int(n).bit_length() - 1 == floor(log2(n)), exact for the
    # power-of-two tile extents here (integer math, no float log2).
    cb = int(shape[contig_dim]).bit_length() - 1
    fb = int(shape[free_dim]).bit_length() - 1
    contig = [basis(contig_dim, i) for i in range(cb)]
    free = ([basis(free_dim, i) for i in range(4, fb)] + [basis(free_dim, i) for i in range(min(4, fb))])
    return contig + free


# Swizzle offset bases per (square) tile size, computed once from the tile shape by
# _swz_offset_bases. The @jit body can't call the generator (only constexpr module
# values are referenceable inside @jit), so precompute the base lists here and build
# the layout in-body, selecting by the constexpr tile size.
# The bases are built for a half-tile (2x2 quadrant tiling): HALF = tile // 2.
_HALF_256 = 256 // 2  # half of the 256x256 tile
_HALF_128 = 128 // 2  # half of the 128x128 tile
_A_BASES_256 = tl.constexpr(_swz_offset_bases([_HALF_256, BLOCK_K], 1))
_A_BASES_128 = tl.constexpr(_swz_offset_bases([_HALF_128, BLOCK_K], 1))
_B_BASES_256 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_256], 0))
_B_BASES_128 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_128], 0))
# Direct-to-LDS offset layouts inferred by the aligned 256x256 path. Pinning
# these keeps a merely 16-byte-aligned leading stride from falling back to a
# blocked layout that the AMD buffer-load lowering cannot consume.
_A_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((8, 1024, 64), (1, 512)))
_B_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((1024, 16, 1), (128, 8)))

_register_kernel = triton.autotune(
    configs=_REGISTER_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_register_configs},
)(_register_kernel_impl)


def _launch_register(a, b, bias=None, config=None, out=None):
    """Launch the register-resident gfx950 GEMM path.

    ``config=None`` autotunes over `_REGISTER_CONFIGS`. Passing an explicit
    config dict bypasses the autotuner and launches that config directly --
    the same convention the Blackwell/Hopper tutorials use so correctness
    tests can pin a config instead of paying for a sweep.
    """
    if config is not None:
        return _launch_register_plan(
            a,
            b,
            bias=bias,
            config=config,
            out=out,
        )

    M, K = a.shape
    b_k, N = b.shape
    if K != b_k:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if bias is not None:
        if bias.shape != (M, N):
            raise ValueError(f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}")
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("Bias and matrix operands must have matching device and dtype")
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    disable_agpr = (K == 256 and N > 256) or (K > 512 and (K % BLOCK_K != 0 or M * N <= 2 * 1024 * 1024))
    launch_options = {"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), )} if disable_agpr else {}
    bias_ptr = bias if bias is not None else out
    args = (
        a,
        b,
        bias_ptr,
        out,
        out,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
    )
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    _register_kernel[grid](
        *args,
        ADD_BIAS=bias is not None,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        **launch_options,
    )
    return out


@triton.jit
def matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, a_top_off, a_bot_off, b_left_off,
                b_right_off, ka, kb, n_steps, stride_ak, stride_bk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """Compute one output tile over an even contiguous range of K64 steps.

    ``ka`` and ``kb`` are the initial element offsets along K. ``n_steps`` must
    be even and at least two. Both the original data-centric kernel and the new
    Stream-K kernel use this same LDS/MFMA pipeline.
    """
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Keep the direct-to-LDS producer contract local to this extracted helper.
    # K is contiguous in A's second tensor dimension and B's first tensor
    # dimension. The helper boundary otherwise hides those width/alignment
    # facts from AxisInfo and buffer-load lowering falls back to an illegal
    # scalar copy.
    a_top_off = tl.max_contiguous(tl.multiple_of(a_top_off, (1, 8)), (1, 8))
    a_bot_off = tl.max_contiguous(tl.multiple_of(a_bot_off, (1, 8)), (1, 8))
    b_left_off = tl.max_contiguous(tl.multiple_of(b_left_off, (8, 1)), (8, 1))
    b_right_off = tl.max_contiguous(tl.multiple_of(b_right_off, (8, 1)), (8, 1))

    k_step_a = BLOCK_K * stride_ak
    k_step_b = BLOCK_K * stride_bk
    a_top_off_n = a_top_off + k_step_a
    a_bot_off_n = a_bot_off + k_step_a
    b_left_off_n = b_left_off + k_step_b
    b_right_off_n = b_right_off + k_step_b
    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    # ── Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ──
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
    tlx.async_load_commit_group()

    ka += BLOCK_K * stride_ak * 2
    kb += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    # ── Main loop (2x unrolled): 8 (mfma + local_load + async refill) regions ──
    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
        # --- sub-iter 0 (buffer 0) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, _next offsets) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
            tlx.async_load_commit_group()
            ka += BLOCK_K * stride_ak * 2
            kb += BLOCK_K * stride_bk * 2

    # ── Epilogue: last 2 pipelined K-steps, drain LDS loads ──
    # iter n_steps-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (n_steps - 2) % 2, always 0 since n_steps is even
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1  # 1 - l_idx
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    # iter n_steps-1: finish ALL four mfmas before returning so the dot operands
    # die and the caller holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


def _streamk_schedule(M, N, K, block_m=BLOCK_M, block_n=BLOCK_N):
    """Build a persistent or variable-work Stream-K schedule."""
    num_pid_m = M // block_m
    num_pid_n = N // block_n
    total_tiles = num_pid_m * num_pid_n
    n_full = K // BLOCK_K
    k_pipe_steps = (n_full // 2) * 2
    k_pipe_pairs = k_pipe_steps // 2
    streamk_tiles = total_tiles % NUM_CU
    total_streamk_units = streamk_tiles * k_pipe_pairs
    # The optimized fixup assumes one resident full-tile wave followed by the
    # distributed tail. Multi-wave full-tile loops currently put the helper's
    # async waits inside an outer warp-pipeline region on the AMD pipeline pass.
    use_streamk = (K == k_pipe_steps * BLOCK_K and total_tiles - streamk_tiles == NUM_CU and streamk_tiles > 0
                   and total_streamk_units >= NUM_CU)
    units_per_program = total_streamk_units // NUM_CU if use_streamk else 0
    remainder_units = total_streamk_units % NUM_CU if use_streamk else 0

    return {
        "HAS_STREAMK": use_streamk,
        "HAS_K_TAIL": K != k_pipe_steps * BLOCK_K,
        "NUM_PROGRAMS": NUM_CU if use_streamk else min(NUM_CU, total_tiles),
        "NUM_FULL_TILES": total_tiles - streamk_tiles if use_streamk else total_tiles,
        "NUM_PID_M": num_pid_m,
        "NUM_PID_N": num_pid_n,
        "K_PIPE_STEPS": k_pipe_steps,
        "K_PIPE_PAIRS": k_pipe_pairs,
        "UNITS_PER_PROGRAM": units_per_program,
        "REMAINDER_UNITS": remainder_units,
    }


@triton.jit
def a16w16_8wave(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    workspace_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    M,
    N,
    K,
    KS,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_m,
    stride_bias_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    HAS_REGISTER_TAIL: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr,
    USE_I64_B_OFFSETS: tl.constexpr,
    USE_I64_C_OFFSETS: tl.constexpr,
    UNEVEN_SPLIT_K: tl.constexpr,
    HAS_M_TAIL: tl.constexpr,
    HAS_N_TAIL: tl.constexpr,
    PIN_OFFSET_LAYOUT: tl.constexpr,
    DEFER_EPILOGUE: tl.constexpr,
    WRITE_STATS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # ── Split-K: grid is grid_mn*SPLIT_K. Peel off split_id, keep the MN pid for
    # the XCD/group remap below. Exact partitions use KS; uneven partitions
    # derive a contiguous whole-K64 range from split_id.
    # We do NOT shift a_ptr/b_ptr (AMD buffer_load builds its resource descriptor
    # from the raw kernel-arg pointer, so an arith'd base fails to lower); instead
    # the split's K byte-offset is folded into the running ka/kb offset (used by
    # every buffer_load) and into the masked-tail addresses. Partials go to a
    # (SPLIT_K*M, N) workspace (row_base=split_id*M); a reduce kernel sums (fp32).
    #
    # On the exact path, KS (per-split K length) is passed as a runtime ARG,
    # not computed as
    # K // SPLIT_K here: the in-kernel divide only proves divisibility 2 for large
    # SPLIT_K (K is known div-16, //8 -> div-2), which collapses the buffer_load
    # offset from the coalesced #linear layout to #blocked and fails to lower. As
    # an arg, KS gets Triton's div-by-16 specialization, so split_id*KS*stride
    # keeps enough divisibility for #linear.
    linear_pid = tl.program_id(0)
    grid_mn = tl.num_programs(0) // SPLIT_K
    split_id = linear_pid // grid_mn
    pid = linear_pid % grid_mn
    if UNEVEN_SPLIT_K:
        full_k_tiles = K // BLOCK_K
        base_k_tiles = full_k_tiles // SPLIT_K
        extra_k_tiles = full_k_tiles % SPLIT_K
        split_start_tile = split_id * base_k_tiles + min(split_id, extra_k_tiles)
        split_k_tiles = base_k_tiles + (split_id < extra_k_tiles)
        split_ks = split_k_tiles * BLOCK_K
        split_start = split_start_tile * BLOCK_K
    else:
        split_ks = KS
        split_start = split_id * KS
    if USE_I64_A_OFFSETS:
        ak_split = split_start.to(tl.int64) * stride_ak
    else:
        ak_split = split_start * stride_ak
    if USE_I64_B_OFFSETS:
        bk_split = split_start.to(tl.int64) * stride_bk
    else:
        bk_split = split_start * stride_bk
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ── Grid-level scheduling: XCD PID remap + GROUP_SIZE_M swizzle (v9-style) ──
    if NUM_XCDS != 1:
        pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = grid_mn % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    if PIN_OFFSET_LAYOUT:
        stride_am = tl.multiple_of(stride_am, 8)
        stride_bn = tl.multiple_of(stride_bn, 8)

    TOP_M: tl.constexpr = 128 if BLOCK_M == 192 else BLOCK_M // 2
    BOTTOM_M: tl.constexpr = BLOCK_M - TOP_M
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Four separate double-buffered LDS allocations — one per operand half-tile.
    # Pin the *swizzled* padded_shared layout (row/col-permuted offset bases) so
    # the ds_reads feeding the MFMAs are bank-conflict-free. The default inferred
    # padded layout ({order, shape})
    # conflicts on CDNA4 (measured 50M SQ_LDS_BANK_CONFLICT vs 0 for this one).
    # Swizzle bases are derived from the half-tile shape (_swz_offset_bases), so the
    # 256x256 (128x64 / 64x128 halves) and thin-N 128x128 (64x64 halves) tiles share
    # one path -- the 64-wide free axis just drops the top bit the 128-wide one adds.
    # TODO(perf): the 64x64 swizzle still shows ~1.5M SQ_LDS_BANK_CONFLICT (10%
    # LDS stall) vs 0 for 128x64. It can't be made conflict-free as a padded layout
    # (direct-to-LDS needs pad interval >=512, but 64x64 lacks a high offset bit for
    # the 4th MFMA row-bit); a swizzled_shared layout is conflict-free but slower
    # (gfx950 has no direct-to-LDS scattering -> extra write swizzle). Net: this
    # padded layout is the fastest option and still beats vendor -- the stall is the
    # price of the cheap direct-to-LDS write on a small square tile.
    a_top_bases: tl.constexpr = (_A_BASES_256 if TOP_M == 128 else _A_BASES_128)
    a_bot_bases: tl.constexpr = (_A_BASES_256 if BOTTOM_M == 128 else _A_BASES_128)
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_top_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_top_bases,
                                                                              [TOP_M, BLOCK_K])
    a_bot_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bot_bases,
                                                                              [BOTTOM_M, BLOCK_K])
    b_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    smem_a_top = tlx.local_alloc((TOP_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_top_shared)
    smem_a_bot = tlx.local_alloc((BOTTOM_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_bot_shared)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)

    # The direct-to-LDS buffer_load write is coalesced only when each offset
    # tensor's #linear layout matches the swizzled LDS layout above. We pin only
    # the shared layouts; the matching offset layouts are inferred from them by
    # tlx-insert-require-layout (no explicit offset_layout= needed).
    offs_am = pid_m * BLOCK_M + tl.arange(0, TOP_M)
    if BLOCK_M == 192 or HAS_M_TAIL:
        offs_am_bot = pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Direct-to-LDS vectorizes its address construction before masked zero-fill
    # lowering. Redirect padded edge rows/columns to valid elements so every
    # source address is legal; the corresponding accumulator lanes are later
    # discarded by the output masks. Keep this coordinate work entirely inside
    # constexpr tail branches so complete tiles retain the original address IR.
    if HAS_M_TAIL:
        global_am = tl.where(offs_am < M, offs_am, 0)
        global_am_bot = tl.where(offs_am_bot < M, offs_am_bot, 0)
    if HAS_N_TAIL:
        offs_bn_right = offs_bn + HALF_N
        global_bn = tl.where(offs_bn < N, offs_bn, 0)
        global_bn_right = tl.where(offs_bn_right < N, offs_bn_right, 0)

    # Widen coordinates before multiplying by strides so large tensors cannot
    # overflow while constructing the pointer offset.
    if USE_I64_A_OFFSETS:
        if HAS_M_TAIL:
            a_row_off = global_am.to(tl.int64)[:, None] * stride_am
            a_bot_row_off = global_am_bot.to(tl.int64)[:, None] * stride_am
        else:
            a_row_off = offs_am.to(tl.int64)[:, None] * stride_am
            if BLOCK_M == 192:
                a_bot_row_off = offs_am_bot.to(tl.int64)[:, None] * stride_am
        a_k_off = offs_k.to(tl.int64)[None, :] * stride_ak
    else:
        if HAS_M_TAIL:
            a_row_off = global_am[:, None] * stride_am
            a_bot_row_off = global_am_bot[:, None] * stride_am
        else:
            a_row_off = offs_am[:, None] * stride_am
            if BLOCK_M == 192:
                a_bot_row_off = offs_am_bot[:, None] * stride_am
        a_k_off = offs_k[None, :] * stride_ak
    if USE_I64_B_OFFSETS:
        if HAS_N_TAIL:
            b_col_off = global_bn.to(tl.int64)[None, :] * stride_bn
            b_right_col_off = global_bn_right.to(tl.int64)[None, :] * stride_bn
        else:
            b_col_off = offs_bn.to(tl.int64)[None, :] * stride_bn
        b_k_off = offs_k.to(tl.int64)[:, None] * stride_bk
    else:
        if HAS_N_TAIL:
            b_col_off = global_bn[None, :] * stride_bn
            b_right_col_off = global_bn_right[None, :] * stride_bn
        else:
            b_col_off = offs_bn[None, :] * stride_bn
        b_k_off = offs_k[:, None] * stride_bk
    if PIN_OFFSET_LAYOUT:
        a_row_off = tl.multiple_of(a_row_off, (8, 8))
        b_col_off = tl.multiple_of(b_col_off, (8, 8))
        if BLOCK_M == 192 or HAS_M_TAIL:
            a_bot_row_off = tl.multiple_of(a_bot_row_off, (8, 8))
        if HAS_N_TAIL:
            b_right_col_off = tl.multiple_of(b_right_col_off, (8, 8))
    a_top_off = a_row_off + a_k_off
    if BLOCK_M == 192 or HAS_M_TAIL:
        a_bot_off = a_bot_row_off + a_k_off
    else:
        a_bot_off = a_top_off + TOP_M * stride_am
    b_left_off = b_k_off + b_col_off
    if HAS_N_TAIL:
        b_right_off = b_k_off + b_right_col_off
    else:
        b_right_off = b_left_off + HALF_N * stride_bn
    if PIN_OFFSET_LAYOUT:
        a_top_off = tlx.require_layout(a_top_off, _A_OFFSET_LAYOUT_256)
        a_bot_off = tlx.require_layout(a_bot_off, _A_OFFSET_LAYOUT_256)
        b_left_off = tlx.require_layout(b_left_off, _B_OFFSET_LAYOUT_256)
        b_right_off = tlx.require_layout(b_right_off, _B_OFFSET_LAYOUT_256)
    a_k_mask = offs_k[None, :] < BLOCK_K
    a_top_mask = (offs_am[:, None] < M) & a_k_mask
    if BLOCK_M == 192 or HAS_M_TAIL:
        a_bot_mask = (offs_am_bot[:, None] < M) & a_k_mask
    else:
        a_bot_mask = ((offs_am[:, None] + TOP_M) < M) & a_k_mask
    b_left_mask = tl.broadcast_to(offs_bn[None, :] < N, b_left_off.shape)
    if HAS_N_TAIL:
        b_right_mask = tl.broadcast_to(offs_bn_right[None, :] < N, b_right_off.shape)
    else:
        b_right_mask = tl.broadcast_to((offs_bn[None, :] + HALF_N) < N, b_right_off.shape)
    # Keep this pipeline inline: its K-contiguous B producer layout is inferred
    # together with the bank-conflict-free LDS layout. Moving it through a JIT
    # helper boundary loses that relationship on current layout propagation.
    a_top_off_n = a_top_off + BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + BLOCK_K * stride_ak
    b_left_off_n = b_left_off + BLOCK_K * stride_bk
    b_right_off_n = b_right_off + BLOCK_K * stride_bk

    ka = ak_split
    kb = bk_split

    acc_tl = tl.zeros((TOP_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((BOTTOM_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((TOP_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((BOTTOM_M, HALF_N), dtype=tl.float32)

    # The pipeline consumes K in pairs of BLOCK_K tiles (prologue prefetches 2,
    # the loop 2/iter, the epilogue drains 2), so it covers only an EVEN number of
    # whole K-tiles: n_pipe. Any leftover -- an odd whole tile and/or a partial
    # final tile (K not a multiple of BLOCK_K) -- is handled by the masked scalar
    # tail after the epilogue.
    n_full = split_ks // BLOCK_K
    n_pipe = (n_full // 2) * 2

    tlx.async_load(
        b_ptr + b_left_off + kb,
        smem_b_left[0],
        mask=b_left_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        a_ptr + a_top_off + ka,
        smem_a_top[0],
        mask=a_top_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        a_ptr + a_bot_off + ka,
        smem_a_bot[0],
        mask=a_bot_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        b_ptr + b_right_off + kb,
        smem_b_right[0],
        mask=b_right_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()

    tlx.async_load(
        b_ptr + b_left_off_n + kb,
        smem_b_left[1],
        mask=b_left_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        a_ptr + a_top_off_n + ka,
        smem_a_top[1],
        mask=a_top_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        a_ptr + a_bot_off_n + ka,
        smem_a_bot[1],
        mask=a_bot_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()
    tlx.async_load(
        b_ptr + b_right_off_n + kb,
        smem_b_right[1],
        mask=b_right_mask,
        other=0.0,
    )
    tlx.async_load_commit_group()

    ka += BLOCK_K * stride_ak * 2
    kb += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.async_load(
                b_ptr + b_left_off + kb,
                smem_b_left[0],
                mask=b_left_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.async_load(
                a_ptr + a_top_off + ka,
                smem_a_top[0],
                mask=a_top_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.async_load(
                a_ptr + a_bot_off + ka,
                smem_a_bot[0],
                mask=a_bot_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.async_load(
                b_ptr + b_right_off + kb,
                smem_b_right[0],
                mask=b_right_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.async_load(
                b_ptr + b_left_off_n + kb,
                smem_b_left[1],
                mask=b_left_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.async_load(
                a_ptr + a_top_off_n + ka,
                smem_a_top[1],
                mask=a_top_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.async_load(
                a_ptr + a_bot_off_n + ka,
                smem_a_bot[1],
                mask=a_bot_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.async_load(
                b_ptr + b_right_off_n + kb,
                smem_b_right[1],
                mask=b_right_mask,
                other=0.0,
            )
            tlx.async_load_commit_group()
            ka += BLOCK_K * stride_ak * 2
            kb += BLOCK_K * stride_bk * 2

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)

    # ── Masked scalar tail: K columns past the pipelined region (an odd leftover
    # tile and/or a partial final tile). Plain masked tl.load + tl.dot -- no LDS,
    # no pipeline. The K-mask zeros the missing contraction elements (they add 0
    # to C = sum_k A*B), so this is correct for arbitrary K. Runs 0-2 iterations;
    # the whole-tile even hot path (n_pipe*BLOCK_K == K) skips it entirely.
    if HAS_REGISTER_TAIL:
        offs_am_bot = (pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M))
        offs_bn_right = offs_bn + HALF_N
        for kk in tl.range(
                n_pipe * BLOCK_K,
                split_ks,
                BLOCK_K,
                num_stages=1,
        ):
            offs_kt = kk + offs_k
            k_mask = offs_kt < split_ks
            a_top_t = tl.load(a_ptr + ak_split + offs_am[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                              mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
            a_bot_t = tl.load(a_ptr + ak_split + offs_am_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                              mask=(offs_am_bot[:, None] < M) & k_mask[None, :], other=0.0)
            b_left_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                               mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
            b_right_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn_right[None, :] * stride_bn,
                                mask=k_mask[:, None] & (offs_bn_right[None, :] < N), other=0.0)
            acc_tl = tl.dot(a_top_t, b_left_t, acc_tl)
            acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl)
            acc_tr = tl.dot(a_top_t, b_right_t, acc_tr)
            acc_br = tl.dot(a_bot_t, b_right_t, acc_br)

    offs_cm_top = pid_m * BLOCK_M + tl.arange(0, TOP_M)
    offs_cm_bot = (pid_m * BLOCK_M + TOP_M + tl.arange(0, BOTTOM_M))
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N
    m_top = offs_cm_top[:, None] < M
    m_bot = offs_cm_bot[:, None] < M
    n_left = offs_cn_left[None, :] < N
    n_right = offs_cn_right[None, :] < N
    if USE_I64_C_OFFSETS:
        c_row_top = offs_cm_top.to(tl.int64)[:, None] * stride_cm
        c_row_bot = offs_cm_bot.to(tl.int64)[:, None] * stride_cm
        c_col_left = offs_cn_left.to(tl.int64)[None, :] * stride_cn
        c_col_right = offs_cn_right.to(tl.int64)[None, :] * stride_cn
    else:
        c_row_top = offs_cm_top[:, None] * stride_cm
        c_row_bot = offs_cm_bot[:, None] * stride_cm
        c_col_left = offs_cn_left[None, :] * stride_cn
        c_col_right = offs_cn_right[None, :] * stride_cn
    c_top_left = c_row_top + c_col_left
    c_bot_left = c_row_bot + c_col_left
    c_top_right = c_row_top + c_col_right
    c_bot_right = c_row_bot + c_col_right

    if SPLIT_K == 1 and not DEFER_EPILOGUE:
        if ADD_BIAS:
            acc_tl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_top & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_bl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_bot & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_tr += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_top & n_right,
                other=0.0,
            ).to(tl.float32)
            acc_br += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_bot & n_right,
                other=0.0,
            ).to(tl.float32)

        et = c_ptr.dtype.element_ty
        c_tl = acc_tl.to(et)
        c_bl = acc_bl.to(et)
        c_tr = acc_tr.to(et)
        c_br = acc_br.to(et)
        if WRITE_STATS:
            top_left = tl.where(m_top & n_left, c_tl.to(tl.float32), 0.0)
            bot_left = tl.where(m_bot & n_left, c_bl.to(tl.float32), 0.0)
            top_right = tl.where(m_top & n_right, c_tr.to(tl.float32), 0.0)
            bot_right = tl.where(m_bot & n_right, c_br.to(tl.float32), 0.0)
            top_sum_sq = tl.sum(top_left * top_left, axis=1) + tl.sum(
                top_right * top_right,
                axis=1,
            )
            bot_sum_sq = tl.sum(bot_left * bot_left, axis=1) + tl.sum(
                bot_right * bot_right,
                axis=1,
            )
            top_stats_offsets = offs_cm_top * num_pid_n + pid_n
            bot_stats_offsets = offs_cm_bot * num_pid_n + pid_n
            if not IS_RMS_NORM:
                tl.store(
                    row_sum_ptr + top_stats_offsets,
                    tl.sum(top_left, axis=1) + tl.sum(top_right, axis=1),
                    mask=offs_cm_top < M,
                )
                tl.store(
                    row_sum_ptr + bot_stats_offsets,
                    tl.sum(bot_left, axis=1) + tl.sum(bot_right, axis=1),
                    mask=offs_cm_bot < M,
                )
            tl.store(
                row_sum_sq_ptr + top_stats_offsets,
                top_sum_sq,
                mask=offs_cm_top < M,
            )
            tl.store(
                row_sum_sq_ptr + bot_stats_offsets,
                bot_sum_sq,
                mask=offs_cm_bot < M,
            )

        # Direct store to C.
        if TOP_M == 128 and BOTTOM_M == 128 and HALF_N == 128:
            # Stop the wide epilogue-store layout from propagating backward
            # through the extracted tile function. Split-K and the 128 tile keep
            # the original inferred accumulator layout.
            acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                           warps_per_cta=[2, 4])
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
            # Pin each 128x128 quadrant to the coalesced SIMD #linear layout (no LDS
            # staging) so OptimizeEpilogue keeps the wide dwordx4 store.
            # _C_STORE_SIMD_LAYOUT is derived for the 128x128 quadrant, so it only
            # applies to the 256x256 tile; a smaller tile (64x64 quadrant) uses a
            # plain store.
            L: tl.constexpr = _C_STORE_SIMD_LAYOUT
            c_tl = tlx.require_layout(c_tl, L)
            # Static guard (no device code): the pin must survive coalesce /
            # remove-layout-conversions / AMD optimize-epilogue so the store stays
            # a wide dwordx4. Fails compilation if a future change drops the pin.
            tlx.assert_same_layout(c_tl, L)
            tl.store(c_ptr + c_top_left, c_tl, mask=m_top & n_left)
            tl.store(c_ptr + c_bot_left, tlx.require_layout(c_bl, L), mask=m_bot & n_left)
            tl.store(c_ptr + c_top_right, tlx.require_layout(c_tr, L), mask=m_top & n_right)
            tl.store(c_ptr + c_bot_right, tlx.require_layout(c_br, L), mask=m_bot & n_right)
        else:
            tl.store(c_ptr + c_top_left, c_tl, mask=m_top & n_left)
            tl.store(c_ptr + c_bot_left, c_bl, mask=m_bot & n_left)
            tl.store(c_ptr + c_top_right, c_tr, mask=m_top & n_right)
            tl.store(c_ptr + c_bot_right, c_br, mask=m_bot & n_right)
    else:
        # Split-K: every split writes its fp32 partial into its workspace slice
        # (rows [split_id*M, split_id*M+M)). Mask stays in relative-M coords; the
        # row offset is added only to the store index.
        rb = split_id * M
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_left[None, :], acc_tl,
                 mask=m_top & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_left[None, :], acc_bl,
                 mask=m_bot & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_right[None, :], acc_tr,
                 mask=m_top & n_right)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_right[None, :], acc_br,
                 mask=m_bot & n_right)


@triton.jit
def _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                      HAS_K_TAIL: tl.constexpr, c_layout: tl.constexpr):
    """Compute and store one complete output tile."""
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m_top = pid_m * BLOCK_M + offs_m
    offs_m_bot = offs_m_top + HALF_M
    offs_n_left = pid_n * BLOCK_N + offs_n
    offs_n_right = offs_n_left + HALF_N
    a_top_off = offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_left_off = offs_k[:, None] * stride_bk + offs_n_left[None, :] * stride_bn
    b_right_off = offs_k[:, None] * stride_bk + offs_n_right[None, :] * stride_bn
    acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                 a_top_off, a_bot_off, b_left_off, b_right_off, 0, 0, K_PIPE_STEPS,
                                                 stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)
    if HAS_K_TAIL:
        # Mask odd full and/or partial K64 steps left after the even pipelined prefix.
        for kk in tl.range(K_PIPE_STEPS * BLOCK_K, K, BLOCK_K, num_stages=1):
            offs_kt = kk + offs_k
            k_mask = offs_kt < K
            a_top = tl.load(a_ptr + offs_m_top[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            a_bot = tl.load(a_ptr + offs_m_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            b_left = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_left[None, :] * stride_bn,
                             mask=k_mask[:, None], other=0.0)
            b_right = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_right[None, :] * stride_bn,
                              mask=k_mask[:, None], other=0.0)
            acc_tl = tl.dot(a_top, b_left, acc_tl)
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
            acc_tr = tl.dot(a_top, b_right, acc_tr)
            acc_br = tl.dot(a_bot, b_right, acc_br)
    et: tl.constexpr = c_ptr.dtype.element_ty
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_tl.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_bl.to(et), c_layout))
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_tr.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_br.to(et), c_layout))


@triton.jit
def _grouped_tile_coords(tile_id, NUM_PID_M: tl.constexpr, NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Map a linear tile ID to the grouped M-major output grid."""
    tiles_per_group: tl.constexpr = GROUP_SIZE_M * NUM_PID_N
    group_id = tile_id // tiles_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_SIZE_M)
    tile_in_group = tile_id % tiles_per_group
    pid_m = first_pid_m + tile_in_group % group_size_m
    pid_n = tile_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _wait_for_streamk_partial(locks_ptr, slot, ready_value):
    """Wait until one producer has published its partial tile."""
    while tl.load(locks_ptr + slot, cache_modifier=".cv", volatile=True) != ready_value:
        pass


@triton.jit
def _reduce_and_store_streamk_quadrant(
    partials_ptr,
    c_ptrs,
    first_contributor,
    num_contributors: tl.constexpr,
    partial_off,
    tile_elems: tl.constexpr,
    acc_layout: tl.constexpr,
    c_layout: tl.constexpr,
):
    """Reduce one quadrant across a tile's contributors and store it."""
    acc = tlx.require_layout(tl.load(partials_ptr + first_contributor * tile_elems + partial_off, cache_modifier=".cv"),
                             acc_layout, pin=False)
    for peer in range(1, num_contributors):
        acc += tlx.require_layout(
            tl.load(partials_ptr + (first_contributor + peer) * tile_elems + partial_off, cache_modifier=".cv"),
            acc_layout, pin=False)
    tl.store(c_ptrs, tlx.require_layout(acc.to(c_ptrs.dtype.element_ty), c_layout))


@triton.jit
def streamk_kernel(a_ptr, b_ptr, c_ptr, partials_ptr, locks_ptr, ready_value, K, stride_am, stride_ak, stride_bk,
                   stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   NUM_XCDS: tl.constexpr, NUM_CU: tl.constexpr, NUM_PROGRAMS: tl.constexpr, HAS_STREAMK: tl.constexpr,
                   NUM_FULL_TILES: tl.constexpr, HAS_K_TAIL: tl.constexpr, NUM_PID_M: tl.constexpr,
                   NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                   K_PIPE_PAIRS: tl.constexpr, UNITS_PER_PROGRAM: tl.constexpr, REMAINDER_UNITS: tl.constexpr):
    """Persistent full tiles plus owner or distributed Stream-K fixup."""
    pid = tl.program_id(0)
    contributors_per_tile: tl.constexpr = K_PIPE_PAIRS // max(UNITS_PER_PROGRAM, 1)
    # When every flattened interval is one equal segment of one tile, all
    # contributors can participate in fixup instead of serializing it in the
    # tile owner. This is a reduction optimization, not a separate schedule.
    DISTRIBUTED_FIXUP: tl.constexpr = (HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS and REMAINDER_UNITS == 0
                                       and K_PIPE_PAIRS % max(UNITS_PER_PROGRAM, 1) == 0
                                       and (contributors_per_tile == 2 or contributors_per_tile == 4))
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                   warps_per_cta=[2, 4])
    a_bases: tl.constexpr = _A_BASES_256 if BLOCK_M == 256 else _A_BASES_128
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [HALF_M, BLOCK_K])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    et: tl.constexpr = a_ptr.dtype.element_ty
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    C: tl.constexpr = _C_STORE_SIMD_LAYOUT if BLOCK_M == 256 and BLOCK_N == 256 else acc_layout
    tile_elems: tl.constexpr = BLOCK_M * BLOCK_N
    partial_tl_off = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    partial_bl_off = partial_tl_off + HALF_M * BLOCK_N
    partial_tr_off = partial_tl_off + HALF_N
    partial_br_off = partial_bl_off + HALF_N
    stream_pid = pid
    if HAS_STREAMK:
        stream_pid = (pid % NUM_XCDS) * (NUM_CU // NUM_XCDS) + pid // NUM_XCDS
        # Fuse TritonBLAS-style lock initialization into the resident kernel;
        # each program clears the slot it may later publish.
        tl.store(locks_ptr + stream_pid, 0, cache_modifier=".wt")
        tl.debug_barrier()

    if HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS:
        # Fast path: each Stream-K program owns one full tile, so no loop is needed.
        head_pid_m = stream_pid % NUM_PID_M
        head_pid_n = stream_pid // NUM_PID_M
        _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, head_pid_m,
                          head_pid_n, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                          BLOCK_N, BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    else:
        # General path for both persistent and generic Stream-K.
        pids_per_xcd: tl.constexpr = (NUM_FULL_TILES + NUM_XCDS - 1) // NUM_XCDS
        remainder_xcds: tl.constexpr = NUM_FULL_TILES % NUM_XCDS
        tall_xcds: tl.constexpr = NUM_XCDS if remainder_xcds == 0 else remainder_xcds
        for virtual_pid in range(pid, NUM_FULL_TILES, NUM_PROGRAMS):
            xcd = virtual_pid % NUM_XCDS
            local_pid = virtual_pid // NUM_XCDS
            if xcd < tall_xcds:
                tile_id = xcd * pids_per_xcd + local_pid
            else:
                tile_id = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
                              BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    if not HAS_STREAMK:
        return

    if DISTRIBUTED_FIXUP:
        # Unlike owner-based (standard Stream-K) fixup, every contributor publishes
        # its partial tile, synchronizes, and reduces a disjoint output region.
        # This lets all contributing CTAs actively reduce instead of relying on a
        # single owner CTA, as in standard Stream-K.
        contributor_id = stream_pid % contributors_per_tile
        tail_tile = NUM_FULL_TILES + stream_pid // contributors_per_tile

        tail_pid_m, tail_pid_n = _grouped_tile_coords(tail_tile, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
        tail_offs_m_top = tail_pid_m * BLOCK_M + offs_m
        tail_offs_m_bot = tail_offs_m_top + HALF_M
        tail_offs_n_left = tail_pid_n * BLOCK_N + offs_n
        tail_offs_n_right = tail_offs_n_left + HALF_N
        tail_a_top_off = tail_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_a_bot_off = tail_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_b_left_off = offs_k[:, None] * stride_bk + tail_offs_n_left[None, :] * stride_bn
        tail_b_right_off = offs_k[:, None] * stride_bk + tail_offs_n_right[None, :] * stride_bn
        segment_k_steps: tl.constexpr = UNITS_PER_PROGRAM * 2
        segment_k_offset = contributor_id * segment_k_steps * BLOCK_K
        acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                     tail_a_top_off, tail_a_bot_off, tail_b_left_off, tail_b_right_off,
                                                     segment_k_offset * stride_ak, segment_k_offset * stride_bk,
                                                     segment_k_steps, stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)

        # Pin and publish all four partial quadrants before any contributor waits.
        # This avoids cyclic dependencies and lets the MFMA accumulators die before
        # fixup. Publishing a runtime-selected quadrant instead costs more VGPR and
        # select work than it saves in workspace traffic on gfx950.
        acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
        acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
        acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
        acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
        partial_base = partials_ptr + stream_pid * tile_elems
        partial_tl_ptr = tlx.require_layout(partial_base + partial_tl_off, acc_layout, pin=False)
        partial_bl_ptr = tlx.require_layout(partial_base + partial_bl_off, acc_layout, pin=False)
        partial_tr_ptr = tlx.require_layout(partial_base + partial_tr_off, acc_layout, pin=False)
        partial_br_ptr = tlx.require_layout(partial_base + partial_br_off, acc_layout, pin=False)
        tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
        tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
        tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
        tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks_ptr + stream_pid, ready_value, cache_modifier=".wt")

        # Cooperatively reduce the tile, assigning consecutive output quadrants to
        # each contributor so all programs remain useful during fixup.
        first_contributor = stream_pid - contributor_id
        for peer in range(contributors_per_tile):
            _wait_for_streamk_partial(locks_ptr, first_contributor + peer, ready_value)
        # Two contributors own the left and right halves; four contributors
        # own one quadrant each.
        bottom_left_owner: tl.constexpr = 0 if contributors_per_tile == 2 else 1
        top_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 2
        bottom_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 3
        if contributor_id == 0:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tl_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_left_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_bl_off, tile_elems, acc_layout, C)
        if contributor_id == top_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tr_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_br_off, tile_elems, acc_layout, C)
    else:
        # Standard Stream-K fixup path: Divide remaining flattened pairs of K
        # pipeline steps almost evenly across CUs.
        logical_pid = stream_pid
        streamk_base = NUM_FULL_TILES * K_PIPE_PAIRS
        start_unit = (streamk_base + logical_pid * UNITS_PER_PROGRAM + min(logical_pid, REMAINDER_UNITS))
        last_unit = (streamk_base + (logical_pid + 1) * UNITS_PER_PROGRAM + min(logical_pid + 1, REMAINDER_UNITS))

        while start_unit < last_unit:
            tile_id = start_unit // K_PIPE_PAIRS
            tile_start = tile_id * K_PIPE_PAIRS
            tile_end = tile_start + K_PIPE_PAIRS
            segment_end = min(last_unit, tile_end)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            tile_offs_m_top = pid_m * BLOCK_M + offs_m
            tile_offs_m_bot = tile_offs_m_top + HALF_M
            tile_offs_n_left = pid_n * BLOCK_N + offs_n
            tile_offs_n_right = tile_offs_n_left + HALF_N
            tile_a_top_off = tile_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_a_bot_off = tile_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_b_left_off = offs_k[:, None] * stride_bk + tile_offs_n_left[None, :] * stride_bn
            tile_b_right_off = offs_k[:, None] * stride_bk + tile_offs_n_right[None, :] * stride_bn
            k_step = (start_unit - tile_start) * 2 * BLOCK_K
            acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left,
                                                         smem_b_right, tile_a_top_off, tile_a_bot_off, tile_b_left_off,
                                                         tile_b_right_off, k_step * stride_ak, k_step * stride_bk,
                                                         (segment_end - start_unit) * 2, stride_ak, stride_bk, BLOCK_M,
                                                         BLOCK_N, BLOCK_K)
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)

            if start_unit != tile_start:
                # A contributor publishes one partial. Its range can then start
                # the next tile, so continue instead of returning immediately.
                base = logical_pid * tile_elems
                partial_tl_ptr = tlx.require_layout(partials_ptr + base + partial_tl_off, acc_layout, pin=False)
                partial_bl_ptr = tlx.require_layout(partials_ptr + base + partial_bl_off, acc_layout, pin=False)
                partial_tr_ptr = tlx.require_layout(partials_ptr + base + partial_tr_off, acc_layout, pin=False)
                partial_br_ptr = tlx.require_layout(partials_ptr + base + partial_br_off, acc_layout, pin=False)
                tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
                tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
                tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
                tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
                tl.debug_barrier()
                tl.store(locks_ptr + logical_pid, ready_value, cache_modifier=".wt")
            else:
                # The program owning the first work unit of a tile collects the
                # following contributors, exactly as the TritonBLAS fixup does.
                covered_end = segment_end
                next_pid = logical_pid + 1
                while covered_end < tile_end:
                    _wait_for_streamk_partial(locks_ptr, next_pid, ready_value)
                    peer_base = next_pid * tile_elems
                    acc_tl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_bl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_bl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_tr += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tr_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_br += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_br_off, cache_modifier=".cv"), acc_layout, pin=False)
                    covered_end += UNITS_PER_PROGRAM + (next_pid < REMAINDER_UNITS)
                    next_pid += 1
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_tl.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_bl.to(et), C))
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_tr.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_br.to(et), C))
            start_unit = segment_end


_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _reduce_k_kernel(workspace_ptr, bias_ptr, c_ptr, M, N, stride_bias_m, stride_bias_n, stride_cm, stride_cn,
                     SPLIT_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     OUTPUT_DTYPE: tl.constexpr, ADD_BIAS: tl.constexpr):
    # Sum the SPLIT_K partials (each a contiguous (M, N) slab in workspace) into
    # C with fp32 accumulation. Small tiles (32x32) so small outputs still spawn
    # many CTAs -- else the reduce is CTA-starved and dominates (D97513062).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    if ADD_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
            mask=mask,
            other=0.0,
        )
        acc += bias.to(tl.float32)
    output_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + output_offsets, acc.to(OUTPUT_DTYPE), mask=mask)


NUM_CU = 256  # gfx950 (CDNA4) compute units
# Minimum K-tiles per split. Two forces set this floor: (1) below it the per-split
# prologue/epilogue overhead dominates the shrinking K work; (2) more splits means
# a proportionally larger fp32 workspace for the reduce to stream back (reduce cost
# ~ SPLIT_K*M*N), so an over-split that only marginally improves the GEMM loses the
# gain to the reduce. Every measured production optimum uses >= 16 tiles/split
# (e.g. K=12288 wants SPLIT_K=12 (16 tiles) not 16 (12 tiles); the latter fills the
# CUs but its extra reduce traffic makes it net slower).
MIN_KTILES_PER_SPLIT = 16

# Tile candidates, largest first. The big tile is the tuned default; the smaller
# one is used only when the big tile can't fill the CUs (see choose_tile).
# choose_tile scans the fallbacks generically, so adding another tile here (e.g.
# (64, 64)) needs no logic change; today only the 128x128 fallback is used.
TILE_CANDIDATES = ((256, 256), (128, 128))


def _lds_split_k_for(grid_mn, K):
    """Largest SPLIT_K keeping grid_mn*SK within one CU wave and each split a whole,
    BLOCK_K-aligned chunk of >= MIN_KTILES_PER_SPLIT tiles.

    All divisors of K are considered, not just powers of two: for K with odd factors
    (e.g. 22272 = 64*348) a non-pow2 SPLIT_K divides K and fills the CUs far more
    precisely than the nearest pow2 (SPLIT_K=12 -> 192 CUs vs SPLIT_K=4 -> 64 CUs).
    The scan is <= NUM_CU/grid_mn (~20) iterations, negligible at compile time."""
    min_ks = MIN_KTILES_PER_SPLIT * BLOCK_K
    best = 1
    for sk in range(2, NUM_CU // grid_mn + 1):  # grid_mn*sk <= NUM_CU
        ks = K // sk
        if K % sk == 0 and ks >= min_ks and ks % BLOCK_K == 0:
            best = sk  # fill = grid_mn*sk grows with sk, so the last valid sk wins
    return best


def _fill_with_uneven_split_k(grid_mn, K, split_k):
    """Increase split_k with whole-K64, unevenly sized partitions."""
    exact_fill = grid_mn * split_k
    if (K % BLOCK_K != 0 or exact_fill >= NUM_CU or exact_fill * 4 >= NUM_CU * 3):
        return split_k
    k_tiles = K // BLOCK_K
    max_split = min(
        NUM_CU // grid_mn,
        k_tiles // MIN_KTILES_PER_SPLIT,
    )
    return max(split_k, max_split)


@lru_cache(maxsize=None)
def _lds_plan_for_shape(M, N, K):
    """Pick (BLOCK_M, BLOCK_N, SPLIT_K) by CU fill -- no shape hardcoding.

    Prefer the tuned 256x256 tile; it is more MFMA-efficient per work-group than the
    128x128 tile. Fall back to the smaller tile only when the 256 grid leaves most of
    the machine idle even after split-K (fill < NUM_CU/2) -- the genuinely thin-N /
    small-tile-count shapes (e.g. N=256, gmn=8): the 4x-denser MN grid then reaches
    occupancy the big tile can't. When the big tile fills at least half the CUs, its
    efficiency beats a full grid of small tiles, so it is kept (comparing raw
    work-group counts across tile sizes is apples-to-oranges -- a 128 tile does 1/4
    the work -- so a bigger small-tile count does not mean it is faster)."""
    bm, bn = TILE_CANDIDATES[0]
    gmn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _lds_split_k_for(gmn, K)
    best_fill = gmn * sk
    if best_fill < NUM_CU // 2:  # big tile leaves most CUs idle even with split-K
        for cbm, cbn in TILE_CANDIDATES[1:]:
            g = triton.cdiv(M, cbm) * triton.cdiv(N, cbn)
            s = _lds_split_k_for(g, K)
            if g * s > best_fill:  # smaller tile fills the machine better
                bm, bn, sk, best_fill = cbm, cbn, s, g * s
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _fill_with_uneven_split_k(grid_mn, K, sk)
    return bm, bn, sk


def choose_split_k(M, N, K):
    """Back-compat: SPLIT_K for the auto-chosen tile."""
    return _lds_plan_for_shape(M, N, K)[2]


def _needs_i64_offsets(tensor):
    """Return whether this view can address beyond signed i32 byte offsets."""
    if any(stride < 0 for stride in tensor.stride()):
        return True
    max_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    max_byte_offset = max_element_offset * tensor.element_size()
    return max_byte_offset > (1 << 31) - 1


_STRONG_LDS_PLANS = {
    (677, 4096, 8192): (192, 256, 4),
    # Preserve the generic LDS path used by the output-stride contract test.
    # The broader inter-wave fallback is not validated for arbitrary output
    # views on this shape.
    (2048, 512, 2048): (128, 128, 2),
    # Preserve the measured D120380534 winners.  The newer rectangular and
    # general inter-wave policies do not beat this LDS implementation for the
    # production M=2032 family.
    (2032, 2560, 18688): (256, 256, 3),
    (2032, 512, 18688): (128, 128, 4),
    (2032, 18688, 512): (256, 256, 1),
}

# The sched-group-barrier scheduler lengthens the critical path for this exact
# deep-K BF16 production kernel. Two independent paired runs measured the plain
# backend schedule 1.010x and 1.015x faster, while other LDS shapes keep the
# scheduler that the shared pipeline was tuned with.
_LDS_DISABLE_SCHEDULER_SHAPES = {
    (1024, 6144, 20480, torch.bfloat16),
}

# The deep-K BF16 production shape benefits from distributing each mapping
# group across four XCDs.  A 101-round paired run measured this 1.004x faster
# than the shared LDS mapping without changing the kernel geometry.
_LDS_GRID_MAPPINGS = {
    (4096, 1894, 242432, torch.bfloat16): (8, 4),
}


def _lds_grid_mapping(M, N, K, dtype):
    mapping = _LDS_GRID_MAPPINGS.get((M, N, K, dtype))
    if mapping is not None:
        return mapping
    if M == N and K >= 8192:
        return 4, 1
    if M <= 1024 and N >= 16384:
        return 2, NUM_XCDS
    return GROUP_SIZE_M, NUM_XCDS


def _strong_lds_plan(M, N, K):
    return _STRONG_LDS_PLANS.get((M, N, K))


@lru_cache(maxsize=None)
def _matmul_plan(M, N, K, dtype):
    """Cache the pure shape-based dispatch decision used by ``matmul``."""
    strong_lds_plan = _strong_lds_plan(M, N, K)
    if strong_lds_plan is not None:
        return "lds", strong_lds_plan
    register_config = _register_plan_for_shape(M, N, K, dtype)
    if register_config is not None:
        return "register", register_config
    return "lds", _lds_plan_for_shape(M, N, K)


def _launch_lds(a, b, bias=None, SPLIT_K=None, TILE=None, K_LIMIT=None, DEFER_EPILOGUE=False, out=None):
    """Launch the shared gfx950 GEMM core, optionally with a fused bias."""
    M, input_k = a.shape
    b_k, N = b.shape
    assert input_k == b_k, "Incompatible dimensions"
    K = input_k if K_LIMIT is None else K_LIMIT
    assert 0 < K <= input_k, f"K_LIMIT={K} must be in (0, {input_k}]"
    if bias is not None:
        assert bias.shape == (M, N), f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}"
        assert bias.device == a.device, "Bias and matrix operands must be on the same device"
        assert bias.dtype == a.dtype, "Bias and matrix operands must have the same dtype"
        if _needs_i64_offsets(bias):
            raise ValueError("gfx950 inter-wave GEMM bias exceeds signed-i32 byte offsets; "
                             f"shape={tuple(bias.shape)}, strides={bias.stride()}")
    if TILE is not None:
        BM, BN = TILE
        grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        SPLIT_K = _lds_split_k_for(grid_mn, K) if SPLIT_K is None else SPLIT_K
    elif SPLIT_K is None:
        BM, BN, SPLIT_K = _lds_plan_for_shape(M, N, K)
    else:
        BM, BN = BLOCK_M, BLOCK_N  # explicit SPLIT_K override keeps the default tile
    uneven_split_k = K % SPLIT_K != 0
    KS = K // SPLIT_K
    # Each split is big enough for the 2-tile prologue and starts on a 16-byte
    # boundary. Full BLOCK_K tiles use direct-to-LDS; the remainder is handled by
    # the masked register tail in the kernel.
    if uneven_split_k:
        assert K % BLOCK_K == 0, (f"Uneven Split-K requires K={K} to be divisible by BLOCK_K={BLOCK_K}")
    min_ks = ((K // BLOCK_K // SPLIT_K) * BLOCK_K if uneven_split_k else KS)
    assert min_ks >= 2 * BLOCK_K, (f"K/SPLIT_K={min_ks} must be at least {2 * BLOCK_K}")
    assert min_ks * a.element_size() % 16 == 0, (f"K/SPLIT_K={min_ks} must preserve 16-byte split alignment")
    c = torch.empty((M, N), device=a.device, dtype=a.dtype) if out is None else out
    GRID_MN = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if SPLIT_K > 1 or DEFER_EPILOGUE:
        workspace_shape = (SPLIT_K * M, N)
        workspace_view = torch.empty(workspace_shape, device="meta", dtype=torch.float32)
        if _needs_i64_offsets(workspace_view):
            raise ValueError("gfx950 inter-wave GEMM FP32 workspace exceeds signed-i32 byte offsets; "
                             f"shape={workspace_shape}, SPLIT_K={SPLIT_K}, DEFER_EPILOGUE={DEFER_EPILOGUE}")
        # fp32 workspace: partials are stored without a rounding step, so the
        # split-K result matches a single fp32-accumulated GEMM (an fp16 workspace
        # would lose ~1e-1 near cancellation). The reduce sums in fp32 too.
        workspace = torch.empty(workspace_shape, device=a.device, dtype=torch.float32)
    else:
        workspace = c  # dummy; the kernel writes c_ptr directly when SPLIT_K==1
    bias_ptr = bias if bias is not None else c
    stride_bias_m = bias.stride(0) if bias is not None else 0
    stride_bias_n = bias.stride(1) if bias is not None else 0
    kernel_output = workspace if SPLIT_K > 1 or DEFER_EPILOGUE else c
    use_i64_c_offsets = _needs_i64_offsets(kernel_output)
    group_size_m, num_xcds = _lds_grid_mapping(M, N, K, a.dtype)
    a16w16_8wave[(GRID_MN * SPLIT_K, )](
        a,
        b,
        bias_ptr,
        c,
        workspace,
        c,
        c,
        M,
        N,
        K,
        KS,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_bias_m,
        stride_bias_n,
        kernel_output.stride(0),
        kernel_output.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=group_size_m,
        NUM_XCDS=num_xcds,
        SPLIT_K=SPLIT_K,
        ADD_BIAS=bias is not None,
        HAS_REGISTER_TAIL=(uneven_split_k or KS % (2 * BLOCK_K) != 0),
        USE_I64_A_OFFSETS=_needs_i64_offsets(a),
        USE_I64_B_OFFSETS=_needs_i64_offsets(b),
        USE_I64_C_OFFSETS=use_i64_c_offsets,
        UNEVEN_SPLIT_K=uneven_split_k,
        HAS_M_TAIL=M % BM != 0,
        HAS_N_TAIL=N % BN != 0,
        PIN_OFFSET_LAYOUT=K_LIMIT is not None,
        DEFER_EPILOGUE=DEFER_EPILOGUE,
        WRITE_STATS=False,
        IS_RMS_NORM=False,
        num_warps=4 if BM == 128 else NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma). Essential to match the reference perf.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        enable_sched_group_barrier_scheduler=((M, N, K, a.dtype) not in _LDS_DISABLE_SCHEDULER_SHAPES),
    )
    if SPLIT_K > 1:
        # Adaptive reduce tile: small outputs need many small CTAs to fill the CUs;
        # large outputs are BW-bound and prefer big tiles for burst efficiency
        # (measured: 32x32 -> 4.5 TB/s vs 128x128 -> 5.4 TB/s on Pooler).
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (64, 256, 4) if big else (32, 32, 4)
        reduce_grid = (triton.cdiv(M, rbm), triton.cdiv(N, rbn))
        _reduce_k_kernel[reduce_grid](
            workspace,
            bias_ptr,
            c,
            M,
            N,
            stride_bias_m,
            stride_bias_n,
            c.stride(0),
            c.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
            ADD_BIAS=bias is not None,
            num_warps=rw,
        )
    if DEFER_EPILOGUE:
        return workspace, c
    return c


def _lds_matmul(a, b, SPLIT_K=None):
    """C = A @ B. `a` is (M, K), `b` is (K, N).

    SPLIT_K partitions the K reduction across SPLIT_K programs per output tile
    (grid = GRID_MN*SPLIT_K), landing fp32 partials in a (SPLIT_K*M, N) workspace
    that a separate fp32 reduce kernel sums into C. This fills the CUs on small-N /
    small-tile-count shapes where the M/N tile grid alone can't. SPLIT_K is chosen
    automatically from the shape (pass an int to override); SPLIT_K=1 launches the
    plain kernel (no workspace, no reduce). The fp32 workspace keeps the result
    deterministic without atomics; as with any Split-K scheme, the changed
    reduction order may introduce normal fp32 rounding differences from the
    non-split kernel.
    """
    if SPLIT_K is None:
        M, K = a.shape
        N = b.shape[1]
        path, config = _matmul_plan(M, N, K, a.dtype)
        if path == "register":
            return _launch_register(a, b, config=config)
        block_m, block_n, split_k = config
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
        )
    return _launch_lds(a, b, SPLIT_K=SPLIT_K)


def _validate_streamk(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16),\
        "streamk_matmul requires matching FP16 or BF16 operands"
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0,\
        f"M and N must be multiples of {BLOCK_M}"
    assert K >= MIN_K, f"K must be at least {MIN_K}"
    assert K % (2 * BLOCK_K) == 0, f"K must be a multiple of {2 * BLOCK_K}"
    return M, N, K


def _choose_streamk_tile(M, N):
    """Use a smaller persistent tile only when the default grid underfills the GPU."""
    BM, BN = TILE_CANDIDATES[0]
    grid_mn = (M // BM) * (N // BN)
    if grid_mn < NUM_CU // 2:
        for candidate_m, candidate_n in TILE_CANDIDATES[1:]:
            candidate_grid = (M // candidate_m) * (N // candidate_n)
            if grid_mn < candidate_grid <= NUM_CU:
                BM, BN, grid_mn = candidate_m, candidate_n, candidate_grid
    return BM, BN


def streamk_matmul(a, b):
    """Run one persistent kernel with a variable-work Stream-K tail when profitable."""
    M, N, K = _validate_streamk(a, b)
    BM, BN = _choose_streamk_tile(M, N)
    schedule = _streamk_schedule(M, N, K, block_m=BM, block_n=BN)
    # Keep the extracted async pipeline out of an outer multi-wave tile loop;
    # the AMD warp-pipeline pass cannot nest its waits in that region. The
    # data-centric kernel keeps the same pipeline inline for this case.
    if not schedule["HAS_STREAMK"] and schedule["NUM_FULL_TILES"] > schedule["NUM_PROGRAMS"]:
        return _lds_matmul(a, b)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if schedule["HAS_STREAMK"]:
        partials = torch.empty((NUM_CU, BM, BN), device=a.device, dtype=torch.float32)
        locks = torch.empty((NUM_CU, ), device=a.device, dtype=torch.int32)
    else:
        partials = locks = c
    streamk_kernel[(schedule["NUM_PROGRAMS"], )](a, b, c, partials, locks, _READY_VALUE, K, a.stride(0), a.stride(1),
                                                 b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BM,
                                                 BLOCK_N=BN, BLOCK_K=BLOCK_K, NUM_XCDS=NUM_XCDS, NUM_CU=NUM_CU,
                                                 GROUP_SIZE_M=GROUP_SIZE_M, **schedule, num_warps=NUM_WARPS,
                                                 num_stages=1, matrix_instr_nonkdim=16, llvm_fn_attrs=_LLVM_ATTRS)
    return c


# Optimized inter-wave and Stream-K implementation.

# Optimized inter-wave and Stream-K kernels.
# Imported lazily by gfx950 after its register helpers are defined.  This
# direction keeps the optimized inter-wave implementation independent while
# avoiding a second copy of the register kernel.

_iw_BLOCK_M = 256
_iw_BLOCK_N = 256
_iw_BLOCK_K = 64
_iw_NUM_WARPS = 8
_iw_GROUP_SIZE_M = 4
_iw_NUM_XCDS = 8

_iw_MIN_K = (2 * _iw_BLOCK_K)  # pipeline prefetches 2 whole K-tiles; the rest goes to the masked tail
_iw_KERNEL_NAME = "a16w16_8wave"
_iw_LLVM_ATTRS = (("amdgpu-agpr-alloc", "0,0"), )
_iw_READY_VALUE = 3

# Coalesced SIMD layout for a 128x128 fp16 output quadrant.  Keeping the
# epilogue in this layout preserves dwordx4 stores on the eight-wave path.
_iw_C_STORE_SIMD_LAYOUT = tlx.layout(
    shape=((16, 4, 8), (8, 4)),
    stride=((8, 128, 512), (1, 4096)),
)


def _iw_swz_offset_bases(shape, contig_dim):
    """Padded-shared swizzle offset bases for a 2D fp16 half-tile, derived from the
    tile shape so both tile sizes share one path (no per-size branch).

    `contig_dim` is the K-contiguous axis (0 or 1); its bits come first (fastest),
    then the free axis contributes its high bits (>= bit 4) before its low bits --
    the row/col permutation that makes the direct-to-LDS ds_reads bank-conflict-free
    on the 128x64 / 64x128 halves. A 128-wide free axis simply carries the extra top
    bit ([64,0] resp. [0,64]) that a 64-wide one omits. Used for both operands: the
    a half-tile [HALF_M, BLOCK_K] has K on dim 1, the b half-tile [BLOCK_K, HALF_N]
    has K on dim 0."""

    def basis(dim, i):
        return [1 << i, 0] if dim == 0 else [0, 1 << i]

    free_dim = 1 - contig_dim
    # log2 of each extent: int(n).bit_length() - 1 == floor(log2(n)), exact for the
    # power-of-two tile extents here (integer math, no float log2).
    cb = int(shape[contig_dim]).bit_length() - 1
    fb = int(shape[free_dim]).bit_length() - 1
    contig = [basis(contig_dim, i) for i in range(cb)]
    free = [basis(free_dim, i) for i in range(4, fb)] + [basis(free_dim, i) for i in range(min(4, fb))]
    return contig + free


# Swizzle offset bases per (square) tile size, computed once from the tile shape by
# _swz_offset_bases. The @jit body can't call the generator (only constexpr module
# values are referenceable inside @jit), so precompute the base lists here and build
# the layout in-body, selecting by the constexpr tile size.
# The bases are built for a half-tile (2x2 quadrant tiling): HALF = tile // 2.
_iw_HALF_256 = 256 // 2  # half of the 256x256 tile
_iw_HALF_128 = 128 // 2  # half of the 128x128 tile
_iw_A_BASES_256 = tl.constexpr(_iw_swz_offset_bases([_iw_HALF_256, _iw_BLOCK_K], 1))
_iw_A_BASES_128 = tl.constexpr(_iw_swz_offset_bases([_iw_HALF_128, _iw_BLOCK_K], 1))
_iw_B_BASES_256 = tl.constexpr(_iw_swz_offset_bases([_iw_BLOCK_K, _iw_HALF_256], 0))
_iw_B_BASES_128 = tl.constexpr(_iw_swz_offset_bases([_iw_BLOCK_K, _iw_HALF_128], 0))
_iw_B_BASES_64 = tl.constexpr(_iw_swz_offset_bases([_iw_BLOCK_K, 32], 0))
# Direct-to-LDS offset layouts inferred by the aligned 256x256 path. Pinning
# these keeps a merely 16-byte-aligned leading stride from falling back to a
# blocked layout that the AMD buffer-load lowering cannot consume.
_iw_A_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((8, 1024, 64), (1, 512)))
_iw_B_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((1024, 16, 1), (128, 8)))
_iw_A_OFFSET_LAYOUT_64_8W = tlx.layout(
    shape=((8, 4, 8, 2), (8, )),
    stride=((8, 1024, 64, 512), (1, )),
)
_iw_B_OFFSET_LAYOUT_64_8W = tlx.layout(
    shape=((8, 4, 8, 2), (8, )),
    stride=((512, 16, 1, 8), (64, )),
)


def _iw_launch_register(a, b, bias=None, config=None, out=None):
    if config is None:
        config = _register_plan_for_shape(a.shape[0], b.shape[1], a.shape[1], a.dtype)
    return _launch_register_plan(a, b, bias=bias, config=config, out=out)


@triton.jit
def _iw_matmul_tile(
    a_ptr,
    b_ptr,
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    a_top_off,
    a_bot_off,
    b_left_off,
    b_right_off,
    ka,
    kb,
    n_steps,
    stride_ak,
    stride_bk,
    _iw_BLOCK_M: tl.constexpr,
    _iw_BLOCK_N: tl.constexpr,
    _iw_BLOCK_K: tl.constexpr,
):
    """Compute one output tile over an even contiguous range of K64 steps.

    ``ka`` and ``kb`` are the initial element offsets along K. ``n_steps`` must
    be even and at least two. Both the original data-centric kernel and the new
    Stream-K kernel use this same LDS/MFMA pipeline.
    """
    FIRST_M: tl.constexpr = 128 if _iw_BLOCK_M > 128 else _iw_BLOCK_M // 2
    SECOND_M: tl.constexpr = _iw_BLOCK_M - FIRST_M
    FIRST_N: tl.constexpr = 128 if _iw_BLOCK_N > 128 else _iw_BLOCK_N // 2
    SECOND_N: tl.constexpr = _iw_BLOCK_N - FIRST_N

    # Keep the direct-to-LDS producer contract local to this extracted helper.
    # K is contiguous in A's second tensor dimension and B's first tensor
    # dimension. The helper boundary otherwise hides those width/alignment
    # facts from AxisInfo and buffer-load lowering falls back to an illegal
    # scalar copy.
    a_top_off = tl.max_contiguous(tl.multiple_of(a_top_off, (1, 8)), (1, 8))
    a_bot_off = tl.max_contiguous(tl.multiple_of(a_bot_off, (1, 8)), (1, 8))
    b_left_off = tl.max_contiguous(tl.multiple_of(b_left_off, (8, 1)), (8, 1))
    b_right_off = tl.max_contiguous(tl.multiple_of(b_right_off, (8, 1)), (8, 1))

    k_step_a = _iw_BLOCK_K * stride_ak
    k_step_b = _iw_BLOCK_K * stride_bk
    a_top_off_n = a_top_off + k_step_a
    a_bot_off_n = a_bot_off + k_step_a
    b_left_off_n = b_left_off + k_step_b
    b_right_off_n = b_right_off + k_step_b
    acc_tl = tl.zeros((FIRST_M, FIRST_N), dtype=tl.float32)
    acc_bl = tl.zeros((SECOND_M, FIRST_N), dtype=tl.float32)
    acc_tr = tl.zeros((FIRST_M, SECOND_N), dtype=tl.float32)
    acc_br = tl.zeros((SECOND_M, SECOND_N), dtype=tl.float32)

    # ── Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ──
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
    tlx.async_load_commit_group()

    ka += _iw_BLOCK_K * stride_ak * 2
    kb += _iw_BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    # ── Main loop (2x unrolled): 8 (mfma + local_load + async refill) regions ──
    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
        # --- sub-iter 0 (buffer 0) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, _next offsets) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
            tlx.async_load_commit_group()
            ka += _iw_BLOCK_K * stride_ak * 2
            kb += _iw_BLOCK_K * stride_bk * 2

    # ── Epilogue: last 2 pipelined K-steps, drain LDS loads ──
    # iter n_steps-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (n_steps - 2) % 2, always 0 since n_steps is even
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1  # 1 - l_idx
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    # iter n_steps-1: finish ALL four mfmas before returning so the dot operands
    # die and the caller holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


def _iw_streamk_schedule(M, N, K, block_m=_iw_BLOCK_M, block_n=_iw_BLOCK_N):
    """Build a persistent or variable-work Stream-K schedule."""
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    total_tiles = num_pid_m * num_pid_n
    n_full = K // _iw_BLOCK_K
    k_pipe_steps = (n_full // 2) * 2
    k_pipe_pairs = k_pipe_steps // 2
    sparse_grid = total_tiles < _iw_NUM_CU
    streamk_tiles = total_tiles if sparse_grid else total_tiles % _iw_NUM_CU
    total_streamk_units = streamk_tiles * k_pipe_pairs
    # A sub-wave output grid distributes every K-pair across the resident
    # programs.  Larger grids keep complete device waves data-parallel and
    # distribute only the sparse final wave.
    use_streamk = (K == k_pipe_steps * _iw_BLOCK_K and (sparse_grid or total_tiles >= _iw_NUM_CU) and streamk_tiles > 0
                   and total_streamk_units >= _iw_NUM_CU)
    units_per_program = total_streamk_units // _iw_NUM_CU if use_streamk else 0
    remainder_units = total_streamk_units % _iw_NUM_CU if use_streamk else 0

    return {
        "HAS_STREAMK": use_streamk,
        "HAS_K_TAIL": K != k_pipe_steps * _iw_BLOCK_K,
        "NUM_PROGRAMS": _iw_NUM_CU if use_streamk else min(_iw_NUM_CU, total_tiles),
        "NUM_FULL_TILES": total_tiles - streamk_tiles if use_streamk else total_tiles,
        "NUM_PID_M": num_pid_m,
        "NUM_PID_N": num_pid_n,
        "K_PIPE_STEPS": k_pipe_steps,
        "K_PIPE_PAIRS": k_pipe_pairs,
        "UNITS_PER_PROGRAM": units_per_program,
        "REMAINDER_UNITS": remainder_units,
    }


@triton.jit
def _iw_a16w16_8wave(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    workspace_ptr,
    M,
    N,
    K,
    KS,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_m,
    stride_bias_n,
    stride_cm,
    stride_cn,
    _iw_BLOCK_M: tl.constexpr,
    _iw_BLOCK_N: tl.constexpr,
    _iw_BLOCK_K: tl.constexpr,
    _iw_GROUP_SIZE_M: tl.constexpr,
    WORKGROUP_MAPPING: tl.constexpr,
    _iw_NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    HAS_REGISTER_TAIL: tl.constexpr,
    USE_I64_A_OFFSETS: tl.constexpr,
    USE_I64_B_OFFSETS: tl.constexpr,
    USE_I64_C_OFFSETS: tl.constexpr,
    HAS_M_TAIL: tl.constexpr,
    HAS_N_TAIL: tl.constexpr,
    PIN_OFFSET_LAYOUT: tl.constexpr,
    DEFER_EPILOGUE: tl.constexpr,
    WARPS_M: tl.constexpr,
    WARPS_N: tl.constexpr,
):
    # ── Split-K: grid is GRID_MN*SPLIT_K. Peel off split_id, keep the MN pid for
    # the XCD/group remap below. Each split owns a contiguous K-slice of size KS.
    # We do NOT shift a_ptr/b_ptr (AMD buffer_load builds its resource descriptor
    # from the raw kernel-arg pointer, so an arith'd base fails to lower); instead
    # the split's K byte-offset is folded into the running ka/kb offset (used by
    # every buffer_load) and into the masked-tail addresses. Partials go to a
    # (SPLIT_K*M, N) workspace (row_base=split_id*M); a reduce kernel sums (fp32).
    #
    # KS (per-split K length) is passed as a runtime ARG, not computed as
    # K // SPLIT_K here: the in-kernel divide only proves divisibility 2 for large
    # SPLIT_K (K is known div-16, //8 -> div-2), which collapses the buffer_load
    # offset from the coalesced #linear layout to #blocked and fails to lower. As
    # an arg, KS gets Triton's div-by-16 specialization, so split_id*KS*stride
    # keeps enough divisibility for #linear.
    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    if USE_I64_A_OFFSETS:
        ak_split = split_id.to(tl.int64) * KS * stride_ak
    else:
        ak_split = split_id * KS * stride_ak
    if USE_I64_B_OFFSETS:
        bk_split = split_id.to(tl.int64) * KS * stride_bk
    else:
        bk_split = split_id * KS * stride_bk
    num_pid_m = tl.cdiv(M, _iw_BLOCK_M)
    num_pid_n = tl.cdiv(N, _iw_BLOCK_N)

    # ── Grid-level scheduling: XCD PID remap + GROUP_SIZE_M swizzle (v9-style) ──
    if _iw_NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + _iw_NUM_XCDS - 1) // _iw_NUM_XCDS
        tall_xcds = GRID_MN % _iw_NUM_XCDS
        tall_xcds = _iw_NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % _iw_NUM_XCDS
        local_pid = pid // _iw_NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

    if WORKGROUP_MAPPING > 0:
        # Tensile-style positive blocked workgroup mapping.  Hardware PIDs are
        # M-major; visit a bounded N band before advancing to the next band so
        # neighboring workgroups can reuse A through L2.  This is distinct
        # from Triton's GROUP_SIZE_M swizzle, which groups a full N row for a
        # bounded set of M tiles.
        source_m = pid % num_pid_m
        source_n = pid // num_pid_m
        n_block = source_n // WORKGROUP_MAPPING
        block_start_n = n_block * WORKGROUP_MAPPING
        block_width = min(WORKGROUP_MAPPING, num_pid_n - block_start_n)
        serial = source_m + (source_n % WORKGROUP_MAPPING) * num_pid_m
        pid_m = serial // block_width
        pid_n = serial % block_width + block_start_n
    elif _iw_GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = _iw_GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * _iw_GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, _iw_GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    if PIN_OFFSET_LAYOUT:
        stride_am = tl.multiple_of(stride_am, 8)
        stride_bn = tl.multiple_of(stride_bn, 8)

    FIRST_M: tl.constexpr = 128 if _iw_BLOCK_M > 128 else _iw_BLOCK_M // 2
    SECOND_M: tl.constexpr = _iw_BLOCK_M - FIRST_M
    FIRST_N: tl.constexpr = 128 if _iw_BLOCK_N > 128 else _iw_BLOCK_N // 2
    SECOND_N: tl.constexpr = _iw_BLOCK_N - FIRST_N

    # Four separate double-buffered LDS allocations — one per operand half-tile.
    # Pin the *swizzled* padded_shared layout (row/col-permuted offset bases) so
    # the ds_reads feeding the MFMAs are bank-conflict-free. The default inferred
    # padded layout ({order, shape})
    # conflicts on CDNA4 (measured 50M SQ_LDS_BANK_CONFLICT vs 0 for this one).
    # Swizzle bases are derived from the half-tile shape (_swz_offset_bases), so the
    # 256x256 (128x64 / 64x128 halves) and thin-N 128x128 (64x64 halves) tiles share
    # one path -- the 64-wide free axis just drops the top bit the 128-wide one adds.
    # TODO(perf): the 64x64 swizzle still shows ~1.5M SQ_LDS_BANK_CONFLICT (10%
    # LDS stall) vs 0 for 128x64. It can't be made conflict-free as a padded layout
    # (direct-to-LDS needs pad interval >=512, but 64x64 lacks a high offset bit for
    # the 4th MFMA row-bit); a swizzled_shared layout is conflict-free but slower
    # (gfx950 has no direct-to-LDS scattering -> extra write swizzle). Net: this
    # padded layout is the fastest option and still beats vendor -- the stall is the
    # price of the cheap direct-to-LDS write on a small square tile.
    a_top_bases: tl.constexpr = _iw_A_BASES_256 if FIRST_M == 128 else _iw_A_BASES_128
    a_bot_bases: tl.constexpr = _iw_A_BASES_256 if SECOND_M == 128 else _iw_A_BASES_128
    b_left_bases: tl.constexpr = (_iw_B_BASES_256 if FIRST_N == 128 else
                                  (_iw_B_BASES_128 if FIRST_N == 64 else _iw_B_BASES_64))
    b_right_bases: tl.constexpr = (_iw_B_BASES_256 if SECOND_N == 128 else
                                   (_iw_B_BASES_128 if SECOND_N == 64 else _iw_B_BASES_64))
    a_top_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_top_bases,
                                                                              [FIRST_M, _iw_BLOCK_K])
    a_bot_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bot_bases,
                                                                              [SECOND_M, _iw_BLOCK_K])
    b_left_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_left_bases,
                                                                               [_iw_BLOCK_K, FIRST_N])
    b_right_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_right_bases,
                                                                                [_iw_BLOCK_K, SECOND_N])
    smem_a_top = tlx.local_alloc((FIRST_M, _iw_BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_top_shared)
    smem_a_bot = tlx.local_alloc((SECOND_M, _iw_BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_bot_shared)
    smem_b_left = tlx.local_alloc((_iw_BLOCK_K, FIRST_N), tlx.dtype_of(b_ptr), 2, layout=b_left_shared)
    smem_b_right = tlx.local_alloc((_iw_BLOCK_K, SECOND_N), tlx.dtype_of(b_ptr), 2, layout=b_right_shared)

    # The direct-to-LDS buffer_load write is coalesced only when each offset
    # tensor's #linear layout matches the swizzled LDS layout above. We pin only
    # the shared layouts; the matching offset layouts are inferred from them by
    # tlx-insert-require-layout (no explicit offset_layout= needed).
    offs_am = pid_m * _iw_BLOCK_M + tl.arange(0, FIRST_M)
    offs_am_bot = pid_m * _iw_BLOCK_M + FIRST_M + tl.arange(0, SECOND_M)
    offs_bn = pid_n * _iw_BLOCK_N + tl.arange(0, FIRST_N)
    offs_bn_right = pid_n * _iw_BLOCK_N + FIRST_N + tl.arange(0, SECOND_N)
    offs_k = tl.arange(0, _iw_BLOCK_K)

    # Direct-to-LDS vectorizes its address construction before masked zero-fill
    # lowering. Redirect padded edge rows/columns to valid elements so every
    # source address is legal; the corresponding accumulator lanes are later
    # discarded by the output masks. Keep this coordinate work entirely inside
    # constexpr tail branches so complete tiles retain the original address IR.
    if HAS_M_TAIL:
        global_am = tl.where(offs_am < M, offs_am, 0)
        global_am_bot = tl.where(offs_am_bot < M, offs_am_bot, 0)
    if HAS_N_TAIL:
        global_bn = tl.where(offs_bn < N, offs_bn, 0)
        global_bn_right = tl.where(offs_bn_right < N, offs_bn_right, 0)

    # Widen coordinates before multiplying by strides so large tensors cannot
    # overflow while constructing the pointer offset.
    if USE_I64_A_OFFSETS:
        if HAS_M_TAIL:
            a_row_off = global_am.to(tl.int64)[:, None] * stride_am
            a_bot_row_off = global_am_bot.to(tl.int64)[:, None] * stride_am
        else:
            a_row_off = offs_am.to(tl.int64)[:, None] * stride_am
            a_bot_row_off = offs_am_bot.to(tl.int64)[:, None] * stride_am
        a_k_off = offs_k.to(tl.int64)[None, :] * stride_ak
    else:
        if HAS_M_TAIL:
            a_row_off = global_am[:, None] * stride_am
            a_bot_row_off = global_am_bot[:, None] * stride_am
        else:
            a_row_off = offs_am[:, None] * stride_am
            a_bot_row_off = offs_am_bot[:, None] * stride_am
        a_k_off = offs_k[None, :] * stride_ak
    if USE_I64_B_OFFSETS:
        if HAS_N_TAIL:
            b_col_off = global_bn.to(tl.int64)[None, :] * stride_bn
            b_right_col_off = global_bn_right.to(tl.int64)[None, :] * stride_bn
        else:
            b_col_off = offs_bn.to(tl.int64)[None, :] * stride_bn
            b_right_col_off = offs_bn_right.to(tl.int64)[None, :] * stride_bn
        b_k_off = offs_k.to(tl.int64)[:, None] * stride_bk
    else:
        if HAS_N_TAIL:
            b_col_off = global_bn[None, :] * stride_bn
            b_right_col_off = global_bn_right[None, :] * stride_bn
        else:
            b_col_off = offs_bn[None, :] * stride_bn
            b_right_col_off = offs_bn_right[None, :] * stride_bn
        b_k_off = offs_k[:, None] * stride_bk
    if PIN_OFFSET_LAYOUT:
        a_row_off = tl.multiple_of(a_row_off, (8, 8))
        b_col_off = tl.multiple_of(b_col_off, (8, 8))
        if HAS_M_TAIL:
            a_bot_row_off = tl.multiple_of(a_bot_row_off, (8, 8))
        if HAS_N_TAIL:
            b_right_col_off = tl.multiple_of(b_right_col_off, (8, 8))
    a_top_off = a_row_off + a_k_off
    a_bot_off = a_bot_row_off + a_k_off
    b_left_off = b_k_off + b_col_off
    b_right_off = b_k_off + b_right_col_off
    if PIN_OFFSET_LAYOUT:
        a_top_off = tlx.require_layout(
            a_top_off,
            _iw_A_OFFSET_LAYOUT_256 if FIRST_M == 128 else _iw_A_OFFSET_LAYOUT_64_8W,
        )
        a_bot_off = tlx.require_layout(
            a_bot_off,
            _iw_A_OFFSET_LAYOUT_256 if SECOND_M == 128 else _iw_A_OFFSET_LAYOUT_64_8W,
        )
        b_left_off = tlx.require_layout(
            b_left_off,
            _iw_B_OFFSET_LAYOUT_256 if FIRST_N == 128 else _iw_B_OFFSET_LAYOUT_64_8W,
        )
        b_right_off = tlx.require_layout(
            b_right_off,
            _iw_B_OFFSET_LAYOUT_256 if SECOND_N == 128 else _iw_B_OFFSET_LAYOUT_64_8W,
        )
    # Keep this pipeline inline: its K-contiguous B producer layout is inferred
    # together with the bank-conflict-free LDS layout. Moving it through a JIT
    # helper boundary loses that relationship on current layout propagation.
    a_top_off_n = a_top_off + _iw_BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + _iw_BLOCK_K * stride_ak
    b_left_off_n = b_left_off + _iw_BLOCK_K * stride_bk
    b_right_off_n = b_right_off + _iw_BLOCK_K * stride_bk

    ka = ak_split
    kb = bk_split

    acc_tl = tl.zeros((FIRST_M, FIRST_N), dtype=tl.float32)
    acc_bl = tl.zeros((SECOND_M, FIRST_N), dtype=tl.float32)
    acc_tr = tl.zeros((FIRST_M, SECOND_N), dtype=tl.float32)
    acc_br = tl.zeros((SECOND_M, SECOND_N), dtype=tl.float32)

    # The pipeline consumes K in pairs of BLOCK_K tiles (prologue prefetches 2,
    # the loop 2/iter, the epilogue drains 2), so it covers only an EVEN number of
    # whole K-tiles: n_pipe. Any leftover -- an odd whole tile and/or a partial
    # final tile (K not a multiple of BLOCK_K) -- is handled by the masked scalar
    # tail after the epilogue.
    n_full = KS // _iw_BLOCK_K
    n_pipe = (n_full // 2) * 2

    # Padded M/N coordinates were redirected to legal row/column zero above.
    # Load the complete physical operand windows without a predicate: free-axis
    # lanes do not mix in a GEMM reduction, and the corresponding output lanes
    # are discarded by the epilogue masks.  Only the separate K tail needs
    # zero-fill semantics.
    tlx.async_load(b_ptr + b_left_off + kb, smem_b_left[0])
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_top_off + ka, smem_a_top[0])
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_bot_off + ka, smem_a_bot[0])
    tlx.async_load_commit_group()
    tlx.async_load(b_ptr + b_right_off + kb, smem_b_right[0])
    tlx.async_load_commit_group()

    tlx.async_load(b_ptr + b_left_off_n + kb, smem_b_left[1])
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_top_off_n + ka, smem_a_top[1])
    tlx.async_load_commit_group()
    tlx.async_load(a_ptr + a_bot_off_n + ka, smem_a_bot[1])
    tlx.async_load_commit_group()
    tlx.async_load(b_ptr + b_right_off_n + kb, smem_b_right[1])
    tlx.async_load_commit_group()

    ka += _iw_BLOCK_K * stride_ak * 2
    kb += _iw_BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.async_load(b_ptr + b_left_off + kb, smem_b_left[0])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.async_load(a_ptr + a_top_off + ka, smem_a_top[0])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.async_load(a_ptr + a_bot_off + ka, smem_a_bot[0])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.async_load(b_ptr + b_right_off + kb, smem_b_right[0])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.async_load(b_ptr + b_left_off_n + kb, smem_b_left[1])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.async_load(a_ptr + a_top_off_n + ka, smem_a_top[1])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.async_load(a_ptr + a_bot_off_n + ka, smem_a_bot[1])
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.async_load(b_ptr + b_right_off_n + kb, smem_b_right[1])
            tlx.async_load_commit_group()
            ka += _iw_BLOCK_K * stride_ak * 2
            kb += _iw_BLOCK_K * stride_bk * 2

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)
    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)
    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)
    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)
    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)
    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)

    # ── Masked scalar tail: K columns past the pipelined region (an odd leftover
    # tile and/or a partial final tile). Plain masked tl.load + tl.dot -- no LDS,
    # no pipeline. The K-mask zeros the missing contraction elements (they add 0
    # to C = sum_k A*B), so this is correct for arbitrary K. Runs 0-2 iterations;
    # the whole-tile even hot path (n_pipe*BLOCK_K == K) skips it entirely.
    if HAS_REGISTER_TAIL:
        for kk in tl.range(n_pipe * _iw_BLOCK_K, KS, _iw_BLOCK_K, num_stages=1):
            offs_kt = kk + offs_k
            k_mask = offs_kt < KS
            a_top_t = tl.load(
                a_ptr + ak_split + offs_am[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                mask=(offs_am[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            a_bot_t = tl.load(
                a_ptr + ak_split + offs_am_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                mask=(offs_am_bot[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b_left_t = tl.load(
                b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=k_mask[:, None] & (offs_bn[None, :] < N),
                other=0.0,
            )
            b_right_t = tl.load(
                b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn_right[None, :] * stride_bn,
                mask=k_mask[:, None] & (offs_bn_right[None, :] < N),
                other=0.0,
            )
            acc_tl = tl.dot(a_top_t, b_left_t, acc_tl)
            acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl)
            acc_tr = tl.dot(a_top_t, b_right_t, acc_tr)
            acc_br = tl.dot(a_bot_t, b_right_t, acc_br)

    offs_cm_top = pid_m * _iw_BLOCK_M + tl.arange(0, FIRST_M)
    offs_cm_bot = pid_m * _iw_BLOCK_M + FIRST_M + tl.arange(0, SECOND_M)
    offs_cn_left = pid_n * _iw_BLOCK_N + tl.arange(0, FIRST_N)
    offs_cn_right = pid_n * _iw_BLOCK_N + FIRST_N + tl.arange(0, SECOND_N)
    m_top = offs_cm_top[:, None] < M
    m_bot = offs_cm_bot[:, None] < M
    n_left = offs_cn_left[None, :] < N
    n_right = offs_cn_right[None, :] < N
    if USE_I64_C_OFFSETS:
        c_row_top = offs_cm_top.to(tl.int64)[:, None] * stride_cm
        c_row_bot = offs_cm_bot.to(tl.int64)[:, None] * stride_cm
        c_col_left = offs_cn_left.to(tl.int64)[None, :] * stride_cn
        c_col_right = offs_cn_right.to(tl.int64)[None, :] * stride_cn
    else:
        c_row_top = offs_cm_top[:, None] * stride_cm
        c_row_bot = offs_cm_bot[:, None] * stride_cm
        c_col_left = offs_cn_left[None, :] * stride_cn
        c_col_right = offs_cn_right[None, :] * stride_cn
    c_top_left = c_row_top + c_col_left
    c_bot_left = c_row_bot + c_col_left
    c_top_right = c_row_top + c_col_right
    c_bot_right = c_row_bot + c_col_right

    if SPLIT_K == 1 and not DEFER_EPILOGUE:
        if ADD_BIAS:
            acc_tl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_top & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_bl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_bot & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_tr += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_top & n_right,
                other=0.0,
            ).to(tl.float32)
            acc_br += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_bot & n_right,
                other=0.0,
            ).to(tl.float32)

        # Direct store to C.
        et = c_ptr.dtype.element_ty
        if WARPS_M * WARPS_N == 8:
            # Stop the wide epilogue-store layout from propagating backward
            # through the extracted tile function. Split-K and the 128 tile keep
            # the original inferred accumulator layout.
            acc_layout: tl.constexpr = tlx.amd_mfma_layout(
                version=4,
                instr_shape=[16, 16, 32],
                transposed=True,
                warps_per_cta=[WARPS_M, WARPS_N],
            )
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
            # Pin each 128x128 quadrant to the coalesced SIMD #linear layout (no LDS
            # staging) so OptimizeEpilogue keeps the wide dwordx4 store.
            # _C_STORE_SIMD_LAYOUT is derived for the 128x128 quadrant, so it only
            # applies to the 256x256 tile; a smaller tile (64x64 quadrant) uses a
            # plain store.
            L: tl.constexpr = _iw_C_STORE_SIMD_LAYOUT
            if FIRST_M == 128 and FIRST_N == 128:
                c_tl = tlx.require_layout(acc_tl.to(et), L)
                # Static guard (no device code): the pin must survive
                # coalesce/remove-layout-conversions/AMD optimize-epilogue.
                tlx.assert_same_layout(c_tl, L)
                tl.store(c_ptr + c_top_left, c_tl, mask=m_top & n_left)
            else:
                tl.store(c_ptr + c_top_left, acc_tl.to(et), mask=m_top & n_left)
            if SECOND_M == 128 and FIRST_N == 128:
                tl.store(
                    c_ptr + c_bot_left,
                    tlx.require_layout(acc_bl.to(et), L),
                    mask=m_bot & n_left,
                )
            else:
                tl.store(c_ptr + c_bot_left, acc_bl.to(et), mask=m_bot & n_left)
            if FIRST_M == 128 and SECOND_N == 128:
                tl.store(
                    c_ptr + c_top_right,
                    tlx.require_layout(acc_tr.to(et), L),
                    mask=m_top & n_right,
                )
            else:
                tl.store(c_ptr + c_top_right, acc_tr.to(et), mask=m_top & n_right)
            if SECOND_M == 128 and SECOND_N == 128:
                tl.store(
                    c_ptr + c_bot_right,
                    tlx.require_layout(acc_br.to(et), L),
                    mask=m_bot & n_right,
                )
            else:
                tl.store(c_ptr + c_bot_right, acc_br.to(et), mask=m_bot & n_right)
        else:
            tl.store(c_ptr + c_top_left, acc_tl.to(et), mask=m_top & n_left)
            tl.store(c_ptr + c_bot_left, acc_bl.to(et), mask=m_bot & n_left)
            tl.store(c_ptr + c_top_right, acc_tr.to(et), mask=m_top & n_right)
            tl.store(c_ptr + c_bot_right, acc_br.to(et), mask=m_bot & n_right)
    else:
        # Split-K: every split writes its fp32 partial into its workspace slice
        # (rows [split_id*M, split_id*M+M)). Mask stays in relative-M coords; the
        # row offset is added only to the store index.
        rb = split_id * M
        tl.store(
            workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_left[None, :],
            acc_tl,
            mask=m_top & n_left,
        )
        tl.store(
            workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_left[None, :],
            acc_bl,
            mask=m_bot & n_left,
        )
        tl.store(
            workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_right[None, :],
            acc_tr,
            mask=m_top & n_right,
        )
        tl.store(
            workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_right[None, :],
            acc_br,
            mask=m_bot & n_right,
        )


@triton.jit
def _iw_matmul_full_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    pid_m,
    pid_n,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    _iw_BLOCK_M: tl.constexpr,
    _iw_BLOCK_N: tl.constexpr,
    _iw_BLOCK_K: tl.constexpr,
    K_PIPE_STEPS: tl.constexpr,
    HAS_K_TAIL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    c_layout: tl.constexpr,
):
    """Compute and store one complete output tile."""
    HALF_M: tl.constexpr = _iw_BLOCK_M // 2
    HALF_N: tl.constexpr = _iw_BLOCK_N // 2
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, _iw_BLOCK_K)
    offs_m_top = pid_m * _iw_BLOCK_M + offs_m
    offs_m_bot = offs_m_top + HALF_M
    offs_n_left = pid_n * _iw_BLOCK_N + offs_n
    offs_n_right = offs_n_left + HALF_N
    # Direct-to-LDS loads are intentionally unmasked.  Duplicate any padded
    # edge coordinate to row/column zero, then discard it at the output store.
    # This keeps the fast full-width load path without requiring host padding.
    load_m_top = tl.where(offs_m_top < M, offs_m_top, 0)
    load_m_bot = tl.where(offs_m_bot < M, offs_m_bot, 0)
    load_n_left = tl.where(offs_n_left < N, offs_n_left, 0)
    load_n_right = tl.where(offs_n_right < N, offs_n_right, 0)
    a_top_off = load_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = load_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_left_off = offs_k[:, None] * stride_bk + load_n_left[None, :] * stride_bn
    b_right_off = offs_k[:, None] * stride_bk + load_n_right[None, :] * stride_bn
    acc_tl, acc_bl, acc_tr, acc_br = _iw_matmul_tile(
        a_ptr,
        b_ptr,
        smem_a_top,
        smem_a_bot,
        smem_b_left,
        smem_b_right,
        a_top_off,
        a_bot_off,
        b_left_off,
        b_right_off,
        0,
        0,
        K_PIPE_STEPS,
        stride_ak,
        stride_bk,
        _iw_BLOCK_M,
        _iw_BLOCK_N,
        _iw_BLOCK_K,
    )
    if HAS_K_TAIL:
        # Mask odd full and/or partial K64 steps left after the even pipelined prefix.
        for kk in tl.range(K_PIPE_STEPS * _iw_BLOCK_K, K, _iw_BLOCK_K, num_stages=1):
            offs_kt = kk + offs_k
            k_mask = offs_kt < K
            a_top = tl.load(
                a_ptr + load_m_top[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                mask=k_mask[None, :],
                other=0.0,
            )
            a_bot = tl.load(
                a_ptr + load_m_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                mask=k_mask[None, :],
                other=0.0,
            )
            b_left = tl.load(
                b_ptr + offs_kt[:, None] * stride_bk + load_n_left[None, :] * stride_bn,
                mask=k_mask[:, None],
                other=0.0,
            )
            b_right = tl.load(
                b_ptr + offs_kt[:, None] * stride_bk + load_n_right[None, :] * stride_bn,
                mask=k_mask[:, None],
                other=0.0,
            )
            acc_tl = tl.dot(a_top, b_left, acc_tl)
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
            acc_tr = tl.dot(a_top, b_right, acc_tr)
            acc_br = tl.dot(a_bot, b_right, acc_br)
    et: tl.constexpr = c_ptr.dtype.element_ty
    tl.store(
        c_ptr + offs_m_top[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
        tlx.require_layout(acc_tl.to(et), c_layout),
        mask=(offs_m_top[:, None] < M) & (offs_n_left[None, :] < N),
    )
    tl.store(
        c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
        tlx.require_layout(acc_bl.to(et), c_layout),
        mask=(offs_m_bot[:, None] < M) & (offs_n_left[None, :] < N),
    )
    tl.store(
        c_ptr + offs_m_top[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
        tlx.require_layout(acc_tr.to(et), c_layout),
        mask=(offs_m_top[:, None] < M) & (offs_n_right[None, :] < N),
    )
    tl.store(
        c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
        tlx.require_layout(acc_br.to(et), c_layout),
        mask=(offs_m_bot[:, None] < M) & (offs_n_right[None, :] < N),
    )


@triton.jit
def _iw_grouped_tile_coords(
    tile_id,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    _iw_GROUP_SIZE_M: tl.constexpr,
):
    """Map a linear tile ID to the grouped M-major output grid."""
    tiles_per_group: tl.constexpr = _iw_GROUP_SIZE_M * NUM_PID_N
    group_id = tile_id // tiles_per_group
    first_pid_m = group_id * _iw_GROUP_SIZE_M
    group_size_m = min(NUM_PID_M - first_pid_m, _iw_GROUP_SIZE_M)
    tile_in_group = tile_id % tiles_per_group
    pid_m = first_pid_m + tile_in_group % group_size_m
    pid_n = tile_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _iw_wait_for_streamk_partial(locks_ptr, slot, ready_value):
    """Wait until one producer has published its partial tile."""
    while tl.load(locks_ptr + slot, cache_modifier=".cv", volatile=True) != ready_value:
        pass


@triton.jit
def _iw_reduce_and_store_streamk_quadrant(
    partials_ptr,
    c_ptrs,
    first_contributor,
    num_contributors: tl.constexpr,
    partial_off,
    tile_elems: tl.constexpr,
    acc_layout: tl.constexpr,
    c_layout: tl.constexpr,
    mask,
):
    """Reduce one quadrant across a tile's contributors and store it."""
    acc = tlx.require_layout(
        tl.load(
            partials_ptr + first_contributor * tile_elems + partial_off,
            cache_modifier=".cv",
        ),
        acc_layout,
        pin=False,
    )
    for peer in range(1, num_contributors):
        acc += tlx.require_layout(
            tl.load(
                partials_ptr + (first_contributor + peer) * tile_elems + partial_off,
                cache_modifier=".cv",
            ),
            acc_layout,
            pin=False,
        )
    tl.store(
        c_ptrs,
        tlx.require_layout(acc.to(c_ptrs.dtype.element_ty), c_layout),
        mask=mask,
    )


@triton.jit
def _iw_streamk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    partials_ptr,
    locks_ptr,
    ready_value,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    _iw_BLOCK_M: tl.constexpr,
    _iw_BLOCK_N: tl.constexpr,
    _iw_BLOCK_K: tl.constexpr,
    _iw_NUM_XCDS: tl.constexpr,
    _iw_NUM_CU: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    HAS_STREAMK: tl.constexpr,
    NUM_FULL_TILES: tl.constexpr,
    HAS_K_TAIL: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    _iw_GROUP_SIZE_M: tl.constexpr,
    K_PIPE_STEPS: tl.constexpr,
    K_PIPE_PAIRS: tl.constexpr,
    UNITS_PER_PROGRAM: tl.constexpr,
    REMAINDER_UNITS: tl.constexpr,
):
    """Persistent full tiles plus owner or distributed Stream-K fixup."""
    pid = tl.program_id(0)
    contributors_per_tile: tl.constexpr = K_PIPE_PAIRS // max(UNITS_PER_PROGRAM, 1)
    # When every flattened interval is one equal segment of one tile, all
    # contributors can participate in fixup instead of serializing it in the
    # tile owner. This is a reduction optimization, not a separate schedule.
    DISTRIBUTED_FIXUP: tl.constexpr = (HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS and REMAINDER_UNITS == 0
                                       and K_PIPE_PAIRS % max(UNITS_PER_PROGRAM, 1) == 0
                                       and (contributors_per_tile == 2 or contributors_per_tile == 4))
    HALF_M: tl.constexpr = _iw_BLOCK_M // 2
    HALF_N: tl.constexpr = _iw_BLOCK_N // 2
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                   warps_per_cta=[2, 4])
    a_bases: tl.constexpr = _iw_A_BASES_256 if _iw_BLOCK_M == 256 else _iw_A_BASES_128
    b_bases: tl.constexpr = _iw_B_BASES_256 if _iw_BLOCK_N == 256 else _iw_B_BASES_128
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [HALF_M, _iw_BLOCK_K])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [_iw_BLOCK_K, HALF_N])
    et: tl.constexpr = a_ptr.dtype.element_ty
    smem_a_top = tlx.local_alloc((HALF_M, _iw_BLOCK_K), et, 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((HALF_M, _iw_BLOCK_K), et, 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((_iw_BLOCK_K, HALF_N), et, 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((_iw_BLOCK_K, HALF_N), et, 2, layout=b_layout)
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, _iw_BLOCK_K)
    C: tl.constexpr = (_iw_C_STORE_SIMD_LAYOUT if _iw_BLOCK_M == 256 and _iw_BLOCK_N == 256 else acc_layout)
    tile_elems: tl.constexpr = _iw_BLOCK_M * _iw_BLOCK_N
    partial_tl_off = offs_m[:, None] * _iw_BLOCK_N + offs_n[None, :]
    partial_bl_off = partial_tl_off + HALF_M * _iw_BLOCK_N
    partial_tr_off = partial_tl_off + HALF_N
    partial_br_off = partial_bl_off + HALF_N
    stream_pid = pid
    if HAS_STREAMK:
        stream_pid = (pid % _iw_NUM_XCDS) * (_iw_NUM_CU // _iw_NUM_XCDS) + pid // _iw_NUM_XCDS

    if HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS:
        # Fast path: each Stream-K program owns one full tile, so no loop is needed.
        head_pid_m, head_pid_n = _iw_grouped_tile_coords(stream_pid, NUM_PID_M, NUM_PID_N, _iw_GROUP_SIZE_M)
        _iw_matmul_full_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            head_pid_m,
            head_pid_n,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            _iw_BLOCK_M,
            _iw_BLOCK_N,
            _iw_BLOCK_K,
            K_PIPE_STEPS,
            HAS_K_TAIL,
            M,
            N,
            C,
        )
    else:
        # General path for both persistent and generic Stream-K.
        pids_per_xcd: tl.constexpr = (NUM_FULL_TILES + _iw_NUM_XCDS - 1) // _iw_NUM_XCDS
        remainder_xcds: tl.constexpr = NUM_FULL_TILES % _iw_NUM_XCDS
        tall_xcds: tl.constexpr = _iw_NUM_XCDS if remainder_xcds == 0 else remainder_xcds
        for virtual_pid in range(pid, NUM_FULL_TILES, NUM_PROGRAMS):
            xcd = virtual_pid % _iw_NUM_XCDS
            local_pid = virtual_pid // _iw_NUM_XCDS
            if xcd < tall_xcds:
                tile_id = xcd * pids_per_xcd + local_pid
            else:
                tile_id = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
            pid_m, pid_n = _iw_grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, _iw_GROUP_SIZE_M)
            _iw_matmul_full_tile(
                a_ptr,
                b_ptr,
                c_ptr,
                smem_a_top,
                smem_a_bot,
                smem_b_left,
                smem_b_right,
                pid_m,
                pid_n,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                stride_cm,
                stride_cn,
                _iw_BLOCK_M,
                _iw_BLOCK_N,
                _iw_BLOCK_K,
                K_PIPE_STEPS,
                HAS_K_TAIL,
                M,
                N,
                C,
            )
    if not HAS_STREAMK:
        return

    if DISTRIBUTED_FIXUP:
        # Unlike owner-based (standard Stream-K) fixup, every contributor publishes
        # its partial tile, synchronizes, and reduces a disjoint output region.
        # This lets all contributing CTAs actively reduce instead of relying on a
        # single owner CTA, as in standard Stream-K.
        contributor_id = stream_pid % contributors_per_tile
        tail_tile = NUM_FULL_TILES + stream_pid // contributors_per_tile

        tail_pid_m, tail_pid_n = _iw_grouped_tile_coords(tail_tile, NUM_PID_M, NUM_PID_N, _iw_GROUP_SIZE_M)
        tail_offs_m_top = tail_pid_m * _iw_BLOCK_M + offs_m
        tail_offs_m_bot = tail_offs_m_top + HALF_M
        tail_offs_n_left = tail_pid_n * _iw_BLOCK_N + offs_n
        tail_offs_n_right = tail_offs_n_left + HALF_N
        load_m_top = tl.where(tail_offs_m_top < M, tail_offs_m_top, 0)
        load_m_bot = tl.where(tail_offs_m_bot < M, tail_offs_m_bot, 0)
        load_n_left = tl.where(tail_offs_n_left < N, tail_offs_n_left, 0)
        load_n_right = tl.where(tail_offs_n_right < N, tail_offs_n_right, 0)
        tail_a_top_off = load_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_a_bot_off = load_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_b_left_off = offs_k[:, None] * stride_bk + load_n_left[None, :] * stride_bn
        tail_b_right_off = (offs_k[:, None] * stride_bk + load_n_right[None, :] * stride_bn)
        segment_k_steps: tl.constexpr = UNITS_PER_PROGRAM * 2
        segment_k_offset = contributor_id * segment_k_steps * _iw_BLOCK_K
        acc_tl, acc_bl, acc_tr, acc_br = _iw_matmul_tile(
            a_ptr,
            b_ptr,
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            tail_a_top_off,
            tail_a_bot_off,
            tail_b_left_off,
            tail_b_right_off,
            segment_k_offset * stride_ak,
            segment_k_offset * stride_bk,
            segment_k_steps,
            stride_ak,
            stride_bk,
            _iw_BLOCK_M,
            _iw_BLOCK_N,
            _iw_BLOCK_K,
        )

        # Pin and publish all four partial quadrants before any contributor waits.
        # This avoids cyclic dependencies and lets the MFMA accumulators die before
        # fixup. Publishing a runtime-selected quadrant instead costs more VGPR and
        # select work than it saves in workspace traffic on gfx950.
        acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
        acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
        acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
        acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
        partial_base = partials_ptr + stream_pid * tile_elems
        partial_tl_ptr = tlx.require_layout(partial_base + partial_tl_off, acc_layout, pin=False)
        partial_bl_ptr = tlx.require_layout(partial_base + partial_bl_off, acc_layout, pin=False)
        partial_tr_ptr = tlx.require_layout(partial_base + partial_tr_off, acc_layout, pin=False)
        partial_br_ptr = tlx.require_layout(partial_base + partial_br_off, acc_layout, pin=False)
        tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
        tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
        tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
        tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks_ptr + stream_pid, ready_value, cache_modifier=".wt")

        # Cooperatively reduce the tile, assigning consecutive output quadrants to
        # each contributor so all programs remain useful during fixup.
        first_contributor = stream_pid - contributor_id
        for peer in range(contributors_per_tile):
            _iw_wait_for_streamk_partial(locks_ptr, first_contributor + peer, ready_value)
        # Two contributors own the left and right halves; four contributors
        # own one quadrant each.
        bottom_left_owner: tl.constexpr = 0 if contributors_per_tile == 2 else 1
        top_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 2
        bottom_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 3
        if contributor_id == 0:
            _iw_reduce_and_store_streamk_quadrant(
                partials_ptr,
                c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor,
                contributors_per_tile,
                partial_tl_off,
                tile_elems,
                acc_layout,
                C,
                (tail_offs_m_top[:, None] < M) & (tail_offs_n_left[None, :] < N),
            )
        if contributor_id == bottom_left_owner:
            _iw_reduce_and_store_streamk_quadrant(
                partials_ptr,
                c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor,
                contributors_per_tile,
                partial_bl_off,
                tile_elems,
                acc_layout,
                C,
                (tail_offs_m_bot[:, None] < M) & (tail_offs_n_left[None, :] < N),
            )
        if contributor_id == top_right_owner:
            _iw_reduce_and_store_streamk_quadrant(
                partials_ptr,
                c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor,
                contributors_per_tile,
                partial_tr_off,
                tile_elems,
                acc_layout,
                C,
                (tail_offs_m_top[:, None] < M) & (tail_offs_n_right[None, :] < N),
            )
        if contributor_id == bottom_right_owner:
            _iw_reduce_and_store_streamk_quadrant(
                partials_ptr,
                c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor,
                contributors_per_tile,
                partial_br_off,
                tile_elems,
                acc_layout,
                C,
                (tail_offs_m_bot[:, None] < M) & (tail_offs_n_right[None, :] < N),
            )
    else:
        # Standard Stream-K fixup path: Divide remaining flattened pairs of K
        # pipeline steps almost evenly across CUs.
        logical_pid = stream_pid
        streamk_base = NUM_FULL_TILES * K_PIPE_PAIRS
        start_unit = (streamk_base + logical_pid * UNITS_PER_PROGRAM + min(logical_pid, REMAINDER_UNITS))
        last_unit = (streamk_base + (logical_pid + 1) * UNITS_PER_PROGRAM + min(logical_pid + 1, REMAINDER_UNITS))

        while start_unit < last_unit:
            relative_tile_id = start_unit // K_PIPE_PAIRS
            tile_id = relative_tile_id
            tile_start = relative_tile_id * K_PIPE_PAIRS
            tile_end = tile_start + K_PIPE_PAIRS
            segment_end = min(last_unit, tile_end)
            pid_m, pid_n = _iw_grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, _iw_GROUP_SIZE_M)
            tile_offs_m_top = pid_m * _iw_BLOCK_M + offs_m
            tile_offs_m_bot = tile_offs_m_top + HALF_M
            tile_offs_n_left = pid_n * _iw_BLOCK_N + offs_n
            tile_offs_n_right = tile_offs_n_left + HALF_N
            load_m_top = tl.where(tile_offs_m_top < M, tile_offs_m_top, 0)
            load_m_bot = tl.where(tile_offs_m_bot < M, tile_offs_m_bot, 0)
            load_n_left = tl.where(tile_offs_n_left < N, tile_offs_n_left, 0)
            load_n_right = tl.where(tile_offs_n_right < N, tile_offs_n_right, 0)
            tile_a_top_off = (load_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak)
            tile_a_bot_off = (load_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak)
            tile_b_left_off = (offs_k[:, None] * stride_bk + load_n_left[None, :] * stride_bn)
            tile_b_right_off = (offs_k[:, None] * stride_bk + load_n_right[None, :] * stride_bn)
            k_step = (start_unit - tile_start) * 2 * _iw_BLOCK_K
            acc_tl, acc_bl, acc_tr, acc_br = _iw_matmul_tile(
                a_ptr,
                b_ptr,
                smem_a_top,
                smem_a_bot,
                smem_b_left,
                smem_b_right,
                tile_a_top_off,
                tile_a_bot_off,
                tile_b_left_off,
                tile_b_right_off,
                k_step * stride_ak,
                k_step * stride_bk,
                (segment_end - start_unit) * 2,
                stride_ak,
                stride_bk,
                _iw_BLOCK_M,
                _iw_BLOCK_N,
                _iw_BLOCK_K,
            )
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)

            if start_unit != tile_start:
                # A contributor publishes one partial. Its range can then start
                # the next tile, so continue instead of returning immediately.
                base = logical_pid * tile_elems
                partial_tl_ptr = tlx.require_layout(partials_ptr + base + partial_tl_off, acc_layout, pin=False)
                partial_bl_ptr = tlx.require_layout(partials_ptr + base + partial_bl_off, acc_layout, pin=False)
                partial_tr_ptr = tlx.require_layout(partials_ptr + base + partial_tr_off, acc_layout, pin=False)
                partial_br_ptr = tlx.require_layout(partials_ptr + base + partial_br_off, acc_layout, pin=False)
                tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
                tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
                tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
                tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
                tl.debug_barrier()
                tl.store(locks_ptr + logical_pid, ready_value, cache_modifier=".wt")
            else:
                # The program owning the first work unit of a tile collects the
                # following contributors, exactly as the TritonBLAS fixup does.
                covered_end = segment_end
                next_pid = logical_pid + 1
                while covered_end < tile_end:
                    _iw_wait_for_streamk_partial(locks_ptr, next_pid, ready_value)
                    peer_base = next_pid * tile_elems
                    acc_tl += tlx.require_layout(
                        tl.load(
                            partials_ptr + peer_base + partial_tl_off,
                            cache_modifier=".cv",
                        ),
                        acc_layout,
                        pin=False,
                    )
                    acc_bl += tlx.require_layout(
                        tl.load(
                            partials_ptr + peer_base + partial_bl_off,
                            cache_modifier=".cv",
                        ),
                        acc_layout,
                        pin=False,
                    )
                    acc_tr += tlx.require_layout(
                        tl.load(
                            partials_ptr + peer_base + partial_tr_off,
                            cache_modifier=".cv",
                        ),
                        acc_layout,
                        pin=False,
                    )
                    acc_br += tlx.require_layout(
                        tl.load(
                            partials_ptr + peer_base + partial_br_off,
                            cache_modifier=".cv",
                        ),
                        acc_layout,
                        pin=False,
                    )
                    covered_end += UNITS_PER_PROGRAM + (next_pid < REMAINDER_UNITS)
                    next_pid += 1
                tl.store(
                    c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                    tlx.require_layout(acc_tl.to(et), C),
                    mask=(tile_offs_m_top[:, None] < M)
                    & (tile_offs_n_left[None, :] < N),
                )
                tl.store(
                    c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                    tlx.require_layout(acc_bl.to(et), C),
                    mask=(tile_offs_m_bot[:, None] < M)
                    & (tile_offs_n_left[None, :] < N),
                )
                tl.store(
                    c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                    tlx.require_layout(acc_tr.to(et), C),
                    mask=(tile_offs_m_top[:, None] < M)
                    & (tile_offs_n_right[None, :] < N),
                )
                tl.store(
                    c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                    tlx.require_layout(acc_br.to(et), C),
                    mask=(tile_offs_m_bot[:, None] < M)
                    & (tile_offs_n_right[None, :] < N),
                )
            start_unit = segment_end


_iw_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def _iw_reduce_k_kernel(
    workspace_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    stride_bias_m,
    stride_bias_n,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    # Sum the SPLIT_K partials (each a contiguous (M, N) slab in workspace) into
    # C with fp32 accumulation. Small tiles (32x32) so small outputs still spawn
    # many CTAs -- else the reduce is CTA-starved and dominates (D97513062).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    if ADD_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
            mask=mask,
            other=0.0,
        )
        acc += bias.to(tl.float32)
    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


_iw_NUM_CU = 256  # gfx950 (CDNA4) compute units
# Minimum K-tiles per split. Two forces set this floor: (1) below it the per-split
# prologue/epilogue overhead dominates the shrinking K work; (2) more splits means
# a proportionally larger fp32 workspace for the reduce to stream back (reduce cost
# ~ SPLIT_K*M*N), so an over-split that only marginally improves the GEMM loses the
# gain to the reduce. Every measured production optimum uses >= 16 tiles/split
# (e.g. K=12288 wants SPLIT_K=12 (16 tiles) not 16 (12 tiles); the latter fills the
# CUs but its extra reduce traffic makes it net slower).
_iw_MIN_KTILES_PER_SPLIT = 16

# Tile candidates, largest first. The big tile is the tuned default; the smaller
# one is used only when the big tile can't fill the CUs (see choose_tile).
# choose_tile scans the fallbacks generically, so adding another tile here (e.g.
# (64, 64)) needs no logic change; today only the 128x128 fallback is used.
_iw_TILE_CANDIDATES = ((256, 256), (128, 128))


def _iw_split_k_for(grid_mn, K):
    """Largest SPLIT_K keeping grid_mn*SK within one CU wave and each split a whole,
    BLOCK_K-aligned chunk of >= MIN_KTILES_PER_SPLIT tiles.

    All divisors of K are considered, not just powers of two: for K with odd factors
    (e.g. 22272 = 64*348) a non-pow2 SPLIT_K divides K and fills the CUs far more
    precisely than the nearest pow2 (SPLIT_K=12 -> 192 CUs vs SPLIT_K=4 -> 64 CUs).
    The scan is <= NUM_CU/grid_mn (~20) iterations, negligible at compile time."""
    min_ks = _iw_MIN_KTILES_PER_SPLIT * _iw_BLOCK_K
    best = 1
    for sk in range(2, _iw_NUM_CU // grid_mn + 1):  # grid_mn*sk <= NUM_CU
        ks = K // sk
        if K % sk == 0 and ks >= min_ks and ks % _iw_BLOCK_K == 0:
            best = sk  # fill = grid_mn*sk grows with sk, so the last valid sk wins
    return best


@lru_cache(maxsize=None)
def _iw_choose_tile(M, N, K):
    """Pick (BLOCK_M, BLOCK_N, SPLIT_K) by CU fill -- no shape hardcoding.

    Prefer the tuned 256x256 tile; it is more MFMA-efficient per work-group than the
    128x128 tile. Fall back to the smaller tile only when the 256 grid leaves most of
    the machine idle even after split-K (fill < NUM_CU/2) -- the genuinely thin-N /
    small-tile-count shapes (e.g. N=256, gmn=8): the 4x-denser MN grid then reaches
    occupancy the big tile can't. When the big tile fills at least half the CUs, its
    efficiency beats a full grid of small tiles, so it is kept (comparing raw
    work-group counts across tile sizes is apples-to-oranges -- a 128 tile does 1/4
    the work -- so a bigger small-tile count does not mean it is faster)."""
    bm, bn = _iw_TILE_CANDIDATES[0]
    gmn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _iw_split_k_for(gmn, K)
    best_fill = gmn * sk
    if best_fill < _iw_NUM_CU // 2:  # big tile leaves most CUs idle even with split-K
        for cbm, cbn in _iw_TILE_CANDIDATES[1:]:
            g = triton.cdiv(M, cbm) * triton.cdiv(N, cbn)
            s = _iw_split_k_for(g, K)
            if g * s > best_fill:  # smaller tile fills the machine better
                bm, bn, sk, best_fill = cbm, cbn, s, g * s
    return bm, bn, sk


def _iw_choose_split_k(M, N, K):
    """Back-compat: SPLIT_K for the auto-chosen tile."""
    return _iw_choose_tile(M, N, K)[2]


def _iw_needs_i64_offsets(tensor):
    """Return whether this view can address beyond signed i32 byte offsets."""
    if any(stride < 0 for stride in tensor.stride()):
        return True
    max_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    max_byte_offset = max_element_offset * tensor.element_size()
    return max_byte_offset > (1 << 31) - 1


@lru_cache(maxsize=None)
def _iw_matmul_plan(M, N, K):
    """Cache the pure shape-based dispatch decision used by ``matmul``."""
    register_config = _register_plan_for_shape(M, N, K)
    if register_config is not None:
        return "register", register_config
    return "lds", _iw_choose_tile(M, N, K)


def _iw_deep_k_direct_group_size(M, N, K, block_m, block_n, split_k):
    """Choose the wider locality group outside the sparse Stream-K regime."""
    if block_m != 256 or block_n != 256 or split_k != 1:
        return None
    if K != 16384:
        return None
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    tail_tiles = grid_m * grid_n % _iw_NUM_CU
    if tail_tiles == 0 or tail_tiles > 5 * _iw_NUM_CU // 8:
        return 6
    return None


def _iw_launch(
    a,
    b,
    bias=None,
    out=None,
    SPLIT_K=None,
    TILE=None,
    K_LIMIT=None,
    DEFER_EPILOGUE=False,
    DISABLE_AGPR=True,
    REVERSE_LOCAL_ASSIGNMENT=False,
    SINK_INSTS_TO_AVOID_SPILLS=False,
    DISABLE_HIGH_RP_RESCHEDULE=False,
    WARP_GRID=(2, 4),
    SCHED_MFMA_PER_DWORDX4=4,
    GROUP_M_OVERRIDE=None,
    WORKGROUP_MAPPING=0,
    NUM_XCDS_OVERRIDE=None,
    WAVES_PER_EU=0,
    REGCLASS_PRIORITY=False,
    MATRIX_INSTR_NONKDIM=16,
):
    """Launch the shared gfx950 GEMM core, optionally with a fused bias."""
    M, input_k = a.shape
    b_k, N = b.shape
    assert input_k == b_k, "Incompatible dimensions"
    K = input_k if K_LIMIT is None else K_LIMIT
    assert 0 < K <= input_k, f"K_LIMIT={K} must be in (0, {input_k}]"
    if bias is not None:
        assert bias.shape == (
            M,
            N,
        ), f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}"
        assert bias.device == a.device, ("Bias and matrix operands must be on the same device")
        assert bias.dtype == a.dtype, ("Bias and matrix operands must have the same dtype")
        if _iw_needs_i64_offsets(bias):
            raise ValueError("gfx950 inter-wave GEMM bias exceeds signed-i32 byte offsets; "
                             f"shape={tuple(bias.shape)}, strides={bias.stride()}")
    if TILE is not None:
        supported_tiles = {
            (128, 128),
            (128, 256),
            (192, 256),
            (256, 64),
            (256, 128),
            (256, 256),
        }
        assert tuple(TILE) in supported_tiles, (f"unsupported inter-wave tile {TILE}; non-power-of-two axes "
                                                "must have an explicit fragment decomposition")
        BM, BN = TILE
        grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        SPLIT_K = _iw_split_k_for(grid_mn, K) if SPLIT_K is None else SPLIT_K
    elif SPLIT_K is None:
        BM, BN, SPLIT_K = _iw_choose_tile(M, N, K)
    else:
        BM, BN = _iw_BLOCK_M, _iw_BLOCK_N  # explicit SPLIT_K override keeps the default tile
    KS = K // SPLIT_K
    # Each split is big enough for the 2-tile prologue and starts on a 16-byte
    # boundary. Full BLOCK_K tiles use direct-to-LDS; the remainder is handled by
    # the masked register tail in the kernel.
    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    assert KS >= 2 * _iw_BLOCK_K, f"K/SPLIT_K={KS} must be at least {2 * BLOCK_K}"
    assert KS * a.element_size() % 16 == 0, (f"K/SPLIT_K={KS} must preserve 16-byte split alignment")
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (
            M,
            N,
        ), f"Output must have shape ({M}, {N}), got {tuple(out.shape)}"
        assert out.device == a.device and out.dtype == a.dtype
        c = out
    GRID_MN = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if SPLIT_K > 1 or DEFER_EPILOGUE:
        workspace_shape = (SPLIT_K * M, N)
        workspace_view = torch.empty(workspace_shape, device="meta", dtype=torch.float32)
        if _iw_needs_i64_offsets(workspace_view):
            raise ValueError("gfx950 inter-wave GEMM FP32 workspace exceeds signed-i32 byte offsets; "
                             f"shape={workspace_shape}, SPLIT_K={SPLIT_K}, DEFER_EPILOGUE={DEFER_EPILOGUE}")
        # fp32 workspace: partials are stored without a rounding step, so the
        # split-K result matches a single fp32-accumulated GEMM (an fp16 workspace
        # would lose ~1e-1 near cancellation). The reduce sums in fp32 too.
        workspace = torch.empty(workspace_shape, device=a.device, dtype=torch.float32)
    else:
        workspace = c  # dummy; the kernel writes c_ptr directly when SPLIT_K==1
    bias_ptr = bias if bias is not None else c
    stride_bias_m = bias.stride(0) if bias is not None else 0
    stride_bias_n = bias.stride(1) if bias is not None else 0
    use_i64_c_offsets = _iw_needs_i64_offsets(c)
    warps_m, warps_n = WARP_GRID
    assert warps_m * warps_n in (4, 8)
    launch_options = {}
    if DISABLE_AGPR:
        launch_options["llvm_fn_attrs"] = (("amdgpu-agpr-alloc", "0,0"), )
    if REVERSE_LOCAL_ASSIGNMENT:
        launch_options["reverse_local_assignment"] = True
    if SINK_INSTS_TO_AVOID_SPILLS:
        launch_options["sink_insts_to_avoid_spills"] = True
    if DISABLE_HIGH_RP_RESCHEDULE:
        launch_options["disable_unclustered_high_rp_reschedule"] = True
    _iw_a16w16_8wave[(GRID_MN * SPLIT_K, )](
        a,
        b,
        bias_ptr,
        c,
        workspace,
        M,
        N,
        K,
        KS,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_bias_m,
        stride_bias_n,
        c.stride(0),
        c.stride(1),
        _iw_BLOCK_M=BM,
        _iw_BLOCK_N=BN,
        _iw_BLOCK_K=_iw_BLOCK_K,
        _iw_GROUP_SIZE_M=(GROUP_M_OVERRIDE if GROUP_M_OVERRIDE is not None else
                          (4 if M == N and K >= 8192 else (2 if M <= 1024 and N >= 16384 else _iw_GROUP_SIZE_M))),
        WORKGROUP_MAPPING=WORKGROUP_MAPPING,
        _iw_NUM_XCDS=(NUM_XCDS_OVERRIDE if NUM_XCDS_OVERRIDE is not None else
                      (1 if M == N and K >= 8192 else _iw_NUM_XCDS)),
        GRID_MN=GRID_MN,
        SPLIT_K=SPLIT_K,
        ADD_BIAS=bias is not None,
        HAS_REGISTER_TAIL=KS % (2 * _iw_BLOCK_K) != 0,
        USE_I64_A_OFFSETS=_iw_needs_i64_offsets(a),
        USE_I64_B_OFFSETS=_iw_needs_i64_offsets(b),
        USE_I64_C_OFFSETS=use_i64_c_offsets,
        HAS_M_TAIL=M % BM != 0,
        HAS_N_TAIL=N % BN != 0,
        PIN_OFFSET_LAYOUT=K_LIMIT is not None,
        DEFER_EPILOGUE=DEFER_EPILOGUE,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        num_warps=warps_m * warps_n,
        num_stages=1,
        matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM,
        waves_per_eu=WAVES_PER_EU,
        regclass_priority_trumps_globalness=REGCLASS_PRIORITY,
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=SCHED_MFMA_PER_DWORDX4,
        **launch_options,
    )
    if SPLIT_K > 1:
        # Adaptive reduce tile: small outputs need many small CTAs to fill the CUs;
        # large outputs are BW-bound and prefer big tiles for burst efficiency
        # (measured: 32x32 -> 4.5 TB/s vs 128x128 -> 5.4 TB/s on Pooler).
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (128, 128, 8) if big else (32, 32, 4)
        reduce_grid = (triton.cdiv(M, rbm), triton.cdiv(N, rbn))
        _iw_reduce_k_kernel[reduce_grid](
            workspace,
            bias_ptr,
            c,
            M,
            N,
            stride_bias_m,
            stride_bias_n,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=_iw_TORCH_TO_TL[a.dtype],
            ADD_BIAS=bias is not None,
            num_warps=rw,
        )
    if DEFER_EPILOGUE:
        return workspace, c
    return c


def _iw_matmul(a, b, SPLIT_K=None, TILE=None, out=None):
    """C = A @ B. `a` is (M, K), `b` is (K, N).

    SPLIT_K partitions the K reduction across SPLIT_K programs per output tile
    (grid = GRID_MN*SPLIT_K), landing fp32 partials in a (SPLIT_K*M, N) workspace
    that a separate fp32 reduce kernel sums into C. This fills the CUs on small-N /
    small-tile-count shapes where the M/N tile grid alone can't. SPLIT_K is chosen
    automatically from the shape (pass an int to override); SPLIT_K=1 launches the
    plain kernel (no workspace, no reduce). The fp32 workspace keeps the result
    numerically identical to the non-split-K kernel; only an int-free fp32 sum is
    added, so there is no precision loss and the result is deterministic.
    """
    if SPLIT_K is None:
        M, K = a.shape
        N = b.shape[1]
        path, config = _iw_matmul_plan(M, N, K)
        if path == "register":
            return _iw_launch_register(a, b, config=config, out=out)
        block_m, block_n, split_k = config
        # For deep-K square tiles that remain data parallel, a six-row group
        # shortens operand-B reuse without creating a sparse Stream-K tail.
        # Use the same 5/8-device-wave boundary as the public Stream-K cost
        # model: below it Stream-K owns the tail; above it (or for an exact
        # device-wave multiple) the wider direct ordering wins consistently.
        direct_group_size = _iw_deep_k_direct_group_size(M, N, K, block_m, block_n, split_k)
        if direct_group_size is not None:
            return _iw_launch(
                a,
                b,
                out=out,
                SPLIT_K=1,
                TILE=(block_m, block_n),
                GROUP_M_OVERRIDE=direct_group_size,
            )
        # Large balanced, shallow-reduction GEMMs use the same MT256 square
        # dataflow as the general LDS kernel, but four M-oriented wave rows and
        # one resident wave per EU give its K64 pipeline the best load/MFMA
        # balance. Keep the policy geometric so neighboring square sizes share
        # one implementation and one schedule.
        if M == N and M >= 8192 and 1024 <= K <= 2048:
            tile_rows = triton.cdiv(M, block_m)
            return _iw_launch(
                a,
                b,
                out=out,
                SPLIT_K=split_k,
                TILE=(block_m, block_n),
                DISABLE_HIGH_RP_RESCHEDULE=True,
                WARP_GRID=(4, 2),
                SCHED_MFMA_PER_DWORDX4=1,
                # Once the grid has at least 64 tile rows, a wider M group
                # improves L2 locality without reducing CU coverage.
                GROUP_M_OVERRIDE=8 if tile_rows >= 64 else 6,
                NUM_XCDS_OVERRIDE=8,
                WAVES_PER_EU=0 if tile_rows >= 64 else 1,
            )
        return _iw_launch(
            a,
            b,
            out=out,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
        )
    return _iw_launch(a, b, out=out, SPLIT_K=SPLIT_K, TILE=TILE)


def _iw_ragged_n_matmul(a, b, out=None):
    """Compute a small N remainder with a narrower secondary launch."""
    M, K = a.shape
    _, N = b.shape
    main_n = N // _iw_BLOCK_N * _iw_BLOCK_N
    tail_n = N - main_n
    if main_n == 0 or tail_n == 0 or tail_n > _iw_BLOCK_N // 2:
        return _iw_matmul(a, b, out=out)
    default_tail_n = 64 if tail_n <= 64 else 128
    tail_tile = tuple(
        int(value) for value in os.environ.get("TLX_TMP_RAGGED_TAIL_TILE", f"256x{default_tail_n}").split("x"))
    default_tail_warps = "4x1" if default_tail_n == 64 else "4x2"
    tail_warp_grid = tuple(
        int(value) for value in os.environ.get("TLX_TMP_RAGGED_TAIL_WARP_GRID", default_tail_warps).split("x"))
    main_workgroup_mapping = int(os.environ.get("TLX_TMP_RAGGED_MAIN_MAPPING", "4"))
    main_num_xcds = int(os.environ.get("TLX_TMP_RAGGED_MAIN_XCDS", "4"))
    c = torch.empty((M, N), device=a.device, dtype=a.dtype) if out is None else out
    _iw_launch(
        a,
        b[:, :main_n],
        out=c[:, :main_n],
        SPLIT_K=1,
        TILE=(_iw_BLOCK_M, _iw_BLOCK_N),
        WORKGROUP_MAPPING=main_workgroup_mapping,
        NUM_XCDS_OVERRIDE=main_num_xcds,
    )
    _iw_launch(
        a,
        b[:, main_n:],
        out=c[:, main_n:],
        SPLIT_K=1,
        TILE=tail_tile,
        WARP_GRID=tail_warp_grid,
    )
    return c


def _iw_validate_streamk(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype and a.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "streamk_matmul requires matching FP16 or BF16 operands"
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    assert M > 0 and N > 0
    assert K >= _iw_MIN_K, f"K must be at least {MIN_K}"
    assert K % (2 * _iw_BLOCK_K) == 0, f"K must be a multiple of {2 * BLOCK_K}"
    return M, N, K


def _iw_choose_streamk_tile(M, N):
    """Use a smaller persistent tile only when the default grid underfills the GPU."""
    BM, BN = _iw_TILE_CANDIDATES[0]
    grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if grid_mn < _iw_NUM_CU // 2:
        for candidate_m, candidate_n in _iw_TILE_CANDIDATES[1:]:
            candidate_grid = triton.cdiv(M, candidate_m) * triton.cdiv(N, candidate_n)
            if grid_mn < candidate_grid <= _iw_NUM_CU:
                BM, BN, grid_mn = candidate_m, candidate_n, candidate_grid
    return BM, BN


def _iw_streamk_mapping(M, N, K, block_m, block_n):
    """Choose the bounded L2/XCD mapping for a persistent tile grid."""
    tile_rows = triton.cdiv(M, block_m)
    tile_columns = triton.cdiv(N, block_n)
    if tile_rows >= 128 and tile_columns >= 128:
        # On large two-dimensional grids, a wider M locality band and two-XCD
        # interleave reduce repeated operand traffic without starving an XCD.
        return 16, 2
    if K == 8192 and 16 <= tile_rows < 24 and tile_columns >= 64:
        return 6, _iw_NUM_XCDS
    return _iw_GROUP_SIZE_M, _iw_NUM_XCDS


def _iw_streamk_matmul(
    a,
    b,
    out=None,
    *,
    _group_size_m=None,
    _num_xcds=None,
    _tile=None,
):
    """Run a variable-work Stream-K grid with optional full-tile waves."""
    M, N, K = _iw_validate_streamk(a, b)
    BM, BN = _iw_choose_streamk_tile(M, N) if _tile is None else _tile
    schedule = _iw_streamk_schedule(M, N, K, block_m=BM, block_n=BN)
    # Keep the extracted async pipeline out of an outer multi-wave tile loop;
    # the AMD warp-pipeline pass cannot nest its waits in that region. The
    # data-centric kernel keeps the same pipeline inline for this case.
    if (not schedule["HAS_STREAMK"] and schedule["NUM_FULL_TILES"] > schedule["NUM_PROGRAMS"]):
        return _iw_matmul(a, b, out=out)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype) if out is None else out
    if schedule["HAS_STREAMK"]:
        partials = torch.empty((_iw_NUM_CU, BM, BN), device=a.device, dtype=torch.float32)
        # Owners may poll a contributor before that contributor starts.  Clear
        # locks in an earlier stream-ordered operation so recycled allocator
        # storage cannot expose a stale ready value across launches.
        locks = torch.zeros((_iw_NUM_CU, ), device=a.device, dtype=torch.int32)
    else:
        partials = locks = c
    default_group_size_m, default_num_xcds = _iw_streamk_mapping(M, N, K, BM, BN)
    group_size_m = (_group_size_m if _group_size_m is not None else int(
        os.environ.get(
            "TLX_TMP_STREAMK_GROUP_M",
            str(default_group_size_m),
        )))
    num_xcds = (_num_xcds if _num_xcds is not None else int(
        os.environ.get("TLX_TMP_STREAMK_NUM_XCDS", str(default_num_xcds))))
    _iw_streamk_kernel[(schedule["NUM_PROGRAMS"], )](
        a,
        b,
        c,
        partials,
        locks,
        _iw_READY_VALUE,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        _iw_BLOCK_M=BM,
        _iw_BLOCK_N=BN,
        _iw_BLOCK_K=_iw_BLOCK_K,
        _iw_NUM_XCDS=num_xcds,
        _iw_NUM_CU=_iw_NUM_CU,
        _iw_GROUP_SIZE_M=group_size_m,
        **schedule,
        num_warps=_iw_NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=_iw_LLVM_ATTRS,
    )
    return c


# Persistent N160/N192 paths.

TILE = tl.constexpr(32)
_PERSISTENT_BLOCK_K = tl.constexpr(64)
N_GROUP_FRAGMENTS = tl.constexpr(4)

# (BLOCK_M, BLOCK_N, NUM_PID_N, NUM_PROGRAMS, TILES_PER_PROGRAM)
_MT256X160_TILE_SPEC = (256, 160, 128, 256, 2)
_MT224X160_MTAIL_TILE_SPEC = (224, 160, 256, 256, 1)
_MT256X192_TILE_SPEC = (256, 192, 128, 256, 2)
_MT224X128_K4096_TILE_SPEC = (224, 128, 48, 240, 1)

# The pipeline schedule is compile-time data, separate from the reusable load,
# LDS, MFMA, persistent traversal, and epilogue machinery below.
_M8_A1_READ_PLAN = (
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 3),
    (3, 3),
    (6, 2),
    (0, 0),
)
_MT256X160_B_PUBLISH_PLAN = (
    ((2, 0, 3), (-1, 0, 0)),
    ((2, 3, 2), (3, 0, 1)),
    ((3, 1, 4), (-1, 0, 0)),
    ((4, 0, 3), (-1, 0, 0)),
    ((1, 0, 2), (4, 3, 2)),
)
_MT256X160_FINISH_MFMA_PLAN = (7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)
_MT256X160_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    4,
    _MT256X160_B_PUBLISH_PLAN,
    _MT256X160_FINISH_MFMA_PLAN,
)

_M7_A1_READ_PLAN = _M8_A1_READ_PLAN[:6] + ((6, 1), )
_MT224X160_B_PUBLISH_PLAN = _MT256X160_B_PUBLISH_PLAN[:4] + (((-1, 0, 0), (-1, 0, 0)), )
_MT224X160_FINISH_MFMA_PLAN = (tuple(range(5, 10)) + (23, 24) + tuple(range(25, 30)))
_MT224X160_MTAIL_PIPELINE_SPEC = (
    _M7_A1_READ_PLAN,
    4,
    _MT224X160_B_PUBLISH_PLAN,
    _MT224X160_FINISH_MFMA_PLAN,
)

# N192 reuses the same N128-plus-tail decomposition with two independent N32
# tail images.  B publishes cover rows 1-3 and the first four columns of row 4;
# the final LDS reads cover the rest of row 4 and rows 5-6.
_MT256X192_B_PUBLISH_PLAN = (
    ((1, 0, 4), (-1, 0, 0)),
    ((1, 4, 2), (2, 0, 2)),
    ((2, 2, 4), (-1, 0, 0)),
    ((3, 0, 4), (-1, 0, 0)),
    ((3, 4, 2), (4, 0, 2)),
    ((4, 2, 2), (-1, 0, 0)),
)
_MT256X192_FINISH_MFMA_PLAN = tuple(range(28, 42))
_MT256X192_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    5,
    _MT256X192_B_PUBLISH_PLAN,
    _MT256X192_FINISH_MFMA_PLAN,
)

# This K4096 row-major production shape needs 240 independent CTAs to keep
# all CUs occupied.  MT224x128 minimizes M-tail work while retaining the
# contiguous N128 LDS/store path used by the mature persistent kernel.
_MT224X128_K4096_PIPELINE_SPEC = (
    _M7_A1_READ_PLAN,
    3,
    (
        ((1, 0, 3), (-1, 0, 0)),
        ((1, 3, 1), (2, 0, 2)),
        ((2, 2, 2), (3, 0, 1)),
        ((-1, 0, 0), (-1, 0, 0)),
    ),
    (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
)

_SPECIALIZATIONS = {
    "mt256x160": (
        (1024, 20480, 6144),
        _MT256X160_TILE_SPEC,
        _MT256X160_PIPELINE_SPEC,
    ),
    "mt256x192": (
        (1024, 24576, 6144),
        _MT256X192_TILE_SPEC,
        _MT256X192_PIPELINE_SPEC,
    ),
    "mt224x128_k4096": (
        (1024, 6144, 4096),
        _MT224X128_K4096_TILE_SPEC,
        _MT224X128_K4096_PIPELINE_SPEC,
    ),
}
_SHAPE_DEFAULTS = {
    (1024, 20480, 6144): "mt256x160",
    (1024, 24576, 6144): "mt256x192",
    (1024, 6144, 4096): "mt224x128_k4096",
}

# Four waves jointly own the first four N32 accumulator fragments as one 32x128
# region.  Each lane gets sixteen contiguous N values, allowing four narrow
# stores to become two 128-bit stores after one layout conversion.
_C_STORE_32X128_LAYOUT = tlx.layout(
    shape=((16, 4, 2, 2), (16, )),
    stride=((128, 16, 2048, 64), (1, )),
)


@triton.jit
def _global_loads(source, kb, tile_spec: tl.constexpr):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    a_ptr += kb * _PERSISTENT_BLOCK_K * stride_ak
    b_ptr += kb * _PERSISTENT_BLOCK_K * stride_bk
    a = [tlx.buffer_load(a_ptr, a_offsets[mi], contiguity=8) for mi in range(tl.constexpr(tile_spec[0] // TILE))]
    b = [tlx.buffer_load(b_ptr, b_offsets[nj], contiguity=8) for nj in range(tl.constexpr(tile_spec[1] // TILE))]
    return tl.tuple(a + b)


@triton.jit
def _local_store_one(stage, value, index: tl.constexpr, tile_spec: tl.constexpr):
    a_buffers, b_buffers = stage
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    if index < m_fragments:
        tlx.local_store(tlx.local_view(a_buffers[index], 0), value)
    else:
        nj: tl.constexpr = index - m_fragments
        view = tlx.local_slice(
            tlx.local_view(
                b_buffers[nj // N_GROUP_FRAGMENTS if nj < n_main_fragments else n_main_groups + nj - n_main_fragments],
                0,
            ),
            [
                0,
                (nj % N_GROUP_FRAGMENTS) * TILE if nj < n_main_fragments else 0,
            ],
            [_PERSISTENT_BLOCK_K, TILE],
        )
        tlx.local_store(view, value)


@triton.jit
def _local_store_all(stage, values, tile_spec: tl.constexpr):
    load_groups: tl.constexpr = tl.constexpr(tile_spec[0] // TILE + tile_spec[1] // TILE)
    for index in tl.static_range(load_groups):
        _local_store_one(stage, values[index], index, tile_spec)


@triton.jit
def _local_load_a(stage, kh: tl.constexpr, mi: tl.constexpr, dot_a: tl.constexpr):
    a_buffers = stage[0]
    view = tlx.local_slice(
        tlx.local_view(a_buffers[mi], 0),
        [0, kh * TILE],
        [TILE, TILE],
    )
    return tlx.require_layout(tlx.local_load(view, relaxed=True), dot_a, pin=False)


@triton.jit
def _local_load_b(
    stage,
    kh: tl.constexpr,
    nj: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_buffers = stage[1]
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    view = tlx.local_slice(
        tlx.local_view(
            b_buffers[nj // N_GROUP_FRAGMENTS if nj < n_main_fragments else n_main_groups + nj - n_main_fragments],
            0,
        ),
        [
            kh * TILE,
            (nj % N_GROUP_FRAGMENTS) * TILE if nj < n_main_fragments else 0,
        ],
        [TILE, TILE],
    )
    value = tlx.local_load(view, relaxed=True)
    return tlx.require_layout(value, dot_b, pin=False)


@triton.jit
def _local_load_b_row(
    stage,
    kh: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return tl.tuple(
        [_local_load_b(stage, kh, nj, dot_b, tile_spec) for nj in range(tl.constexpr(tile_spec[1] // TILE))])


@triton.jit
def _mfma_part(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    first_nj: tl.constexpr,
    count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    accumulators: tl.constexpr = tl.constexpr(tile_spec[0] // TILE * tile_spec[1] // TILE)
    return tl.tuple([
        tlx.amd_scheduled_mfma(
            tlx.require_layout(a_operand, dot_a, pin=False),
            tlx.require_layout(b_operands[index % n_fragments], dot_b, pin=False),
            tlx.require_layout(acc[index], mma, pin=False),
            accumulator_role="persistent",
            resident_operand=None,
            initialize=initialize,
        ) if (index // n_fragments == mi and index % n_fragments >= first_nj and index % n_fragments < first_nj + count)
        else acc[index] for index in range(accumulators)
    ])


@triton.jit
def _mfma_row(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        0,
        tl.constexpr(tile_spec[1] // TILE),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )


@triton.jit
def _global_prefetch_one(source, kb, index: tl.constexpr, tile_spec: tl.constexpr):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    if index < m_fragments:
        return tlx.buffer_load(
            a_ptr + kb * _PERSISTENT_BLOCK_K * stride_ak,
            a_offsets[index],
            contiguity=8,
        )
    else:
        return tlx.buffer_load(
            b_ptr + kb * _PERSISTENT_BLOCK_K * stride_bk,
            b_offsets[index - m_fragments],
            contiguity=8,
        )


@triton.jit
def _publish_a_row(
    a_operand,
    b_operands,
    acc,
    old_value,
    next_stage,
    source,
    future_kb,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance one A fragment across three K64 pipeline generations.

    ``a_operand`` and ``b_operands`` compute one accumulator row for K(t),
    ``old_value`` is A(K(t+1)) already prefetched in VGPRs, and ``future``
    becomes A(K(t+2)) in VGPRs.  The K(t) MFMA row is split around the future
    global load to balance load-latency coverage against VGPR lifetime.
    """
    # One accumulator row contains BLOCK_N / 32 logical C[32, 32]
    # fragments.  This is 5 fragments for N160 and 6 for N192.
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # K(t+1): retire the previously prefetched A[32, 64] value from VGPRs
    # into the next LDS stage.
    _local_store_one(next_stage, old_value, mi, tile_spec)

    # K(t): compute columns [0, mfmas_before_prefetch) of accumulator row mi.
    acc = _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        0,
        tl.constexpr(pipeline_spec[1]),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )

    # K(t+2): issue this row's next global A[32, 64] load into VGPRs.
    future = _global_prefetch_one(source, future_kb, mi, tile_spec)

    # K(t): compute the remaining columns
    # [mfmas_before_prefetch, n_fragments).  For N160 the split is 4 + 1;
    # for N192 it is 5 + 1.
    acc = _mfma_part(
        a_operand,
        b_operands,
        acc,
        mi,
        tl.constexpr(pipeline_spec[1]),
        n_fragments - tl.constexpr(pipeline_spec[1]),
        mma,
        dot_a,
        dot_b,
        initialize,
        tile_spec,
    )
    return acc, future


@triton.jit
def _publish_b_fragment(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    nj: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    index: tl.constexpr = m_fragments + nj
    _local_store_one(next_stage, prefetched[index], index, tile_spec)
    acc = _mfma_part(
        a_operands[0],
        b_operands,
        acc,
        0,
        nj,
        1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )

    future = _global_prefetch_one(source, future_kb, index, tile_spec)

    # Spread the MFMA rows across publish groups according to a tile-specific
    # compile-time plan, so the pipeline core itself is independent of 8x5.
    for part in tl.static_range(tl.constexpr(len(pipeline_spec[2][nj]))):
        if tl.constexpr(pipeline_spec[2][nj][part][2]) > 0:
            acc = _mfma_part(
                a_operands[tl.constexpr(pipeline_spec[2][nj][part][0])],
                b_operands,
                acc,
                tl.constexpr(pipeline_spec[2][nj][part][0]),
                tl.constexpr(pipeline_spec[2][nj][part][1]),
                tl.constexpr(pipeline_spec[2][nj][part][2]),
                mma,
                dot_a,
                dot_b,
                False,
                tile_spec,
            )
    return acc, future


@triton.jit
def _publish_b_rows(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Publish B fragments and issue their tile-specific K1 MFMA spans."""
    future = tl.tuple([])
    for nj in tl.static_range(tl.constexpr(tile_spec[1] // TILE)):
        acc, value = _publish_b_fragment(
            a_operands,
            b_operands,
            acc,
            prefetched,
            next_stage,
            source,
            future_kb,
            nj,
            mma,
            dot_a,
            dot_b,
            pipeline_spec,
            tile_spec,
        )
        future += tl.tuple([value])
    return acc, future


@triton.jit
def _finish_read(
    next_stage,
    a_operands,
    b_operands,
    acc,
    read: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    if read < n_fragments:
        value = _local_load_b(next_stage, 0, read, dot_b, tile_spec)
    else:
        value = _local_load_a(next_stage, 0, read - n_fragments, dot_a)
    flat: tl.constexpr = tl.constexpr(pipeline_spec[3][read])
    acc = _mfma_part(
        a_operands[flat // n_fragments],
        b_operands,
        acc,
        flat // n_fragments,
        flat % n_fragments,
        1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )
    return acc, value


@triton.jit
def _finish_iteration(
    next_stage,
    a_operands,
    b_operands,
    acc,
    early_prefetched,
    late_prefetched,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Pair next-stage LDS reads with the tile-specific final MFMA sequence."""
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for read in tl.static_range(m_fragments + n_fragments):
        acc, value = _finish_read(
            next_stage,
            a_operands,
            b_operands,
            acc,
            read,
            mma,
            dot_a,
            dot_b,
            pipeline_spec,
            tile_spec,
        )
        if read < n_fragments:
            next_b += tl.tuple([value])
        else:
            next_a += tl.tuple([value])
    acc = _mfma_row(
        a_operands[m_fragments - 1],
        b_operands,
        acc,
        m_fragments - 1,
        mma,
        dot_a,
        dot_b,
        False,
        tile_spec,
    )
    return acc, early_prefetched + late_prefetched, next_a, next_b


@triton.jit
def _pipeline(
    source,
    future_kb,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance the manual pipeline by one K64 block.

    Entry state:
      * ``current_stage`` contains K(t), while ``current_a``/``current_b``
        already hold its first K32 half (kh=0) in operand VGPRs.
      * ``prefetched`` contains the complete K(t+1) A/B tile in VGPRs.
      * ``next_stage`` is available to receive K(t+1).

    Exit state:
      * K(t) has been fully accumulated.
      * ``next_stage`` contains K(t+1), whose kh=0 operands are returned.
      * the complete K(t+2) A/B tile is returned in prefetch VGPRs.
    """
    # Future global prefetches for K(t+2).  ``a1`` and ``b1`` are not K(t+1):
    # they are the second K32 half (kh=1) of the current K(t) LDS stage.
    future_a = tl.tuple([])
    a1 = tl.tuple([])
    b1 = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for mi in tl.static_range(m_fragments):
        # K(t).kh1: spread the smaller B operand window across the first
        # n_fragments loop iterations instead of issuing one late LDS burst.
        if mi < n_fragments:
            b1 += tl.tuple([_local_load_b(current_stage, 1, mi, dot_b, tile_spec)])

        # In one interleaved step: publish A(mi,K(t+1)) from VGPRs to the next
        # LDS stage, compute row mi of K(t).kh0, and prefetch A(mi,K(t+2)).
        acc, future = _publish_a_row(
            current_a[mi],
            current_b,
            acc,
            prefetched[mi],
            next_stage,
            source,
            future_kb,
            mi,
            mma,
            dot_a,
            dot_b,
            initialize,
            pipeline_spec,
            tile_spec,
        )
        future_a += tl.tuple([future])

        # K(t).kh1: read the configured A operand span from current LDS.  The
        # plan covers every A fragment exactly once while controlling lifetime.
        if tl.constexpr(pipeline_spec[0][mi][1]) > 0:
            a1 += tl.tuple([
                _local_load_a(current_stage, 1, index, dot_a) for index in range(
                    tl.constexpr(pipeline_spec[0][mi][0]),
                    tl.constexpr(pipeline_spec[0][mi][0]) + tl.constexpr(pipeline_spec[0][mi][1]),
                )
            ])

    # Publish B(K(t+1)) to next LDS and prefetch B(K(t+2)), while a1/b1
    # compute most of the current K(t).kh1 accumulator updates.
    acc, late_prefetched = _publish_b_rows(
        a1,
        b1,
        acc,
        prefetched,
        next_stage,
        source,
        future_kb,
        mma,
        dot_a,
        dot_b,
        pipeline_spec,
        tile_spec,
    )

    # All A/B fragments of K(t+1) are now in next_stage.  Make them visible
    # before reading its kh=0 operands; pair those reads with the remaining
    # K(t).kh1 MFMA updates in _finish_iteration.
    tl.debug_barrier()
    return _finish_iteration(
        next_stage,
        a1,
        b1,
        acc,
        future_a,
        late_prefetched,
        mma,
        dot_a,
        dot_b,
        pipeline_spec,
        tile_spec,
    )


@triton.jit
def _pipeline_pair(
    kb,
    source,
    stage0,
    stage1,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Consume K(kb) and K(kb+1), restoring the LDS stage orientation.

    On entry, stage0/current_a/current_b represent K(kb), and ``prefetched``
    is K(kb+1).  The first call advances to K(kb+1), swapping the current LDS
    stage from stage0 to stage1.  The second advances to K(kb+2), swapping it
    back to stage0.  The numeric argument passed to ``_pipeline`` is the
    future global-prefetch block, not the block currently being consumed.
    """
    # Consume K(kb), publish K(kb+1) into stage1, and prefetch K(kb+2).
    acc, prefetched, current_a, current_b = _pipeline(
        source,
        kb + 2,
        stage0,
        stage1,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        initialize,
        pipeline_spec,
        tile_spec,
    )
    # Consume K(kb+1), publish K(kb+2) into stage0, and prefetch K(kb+3).
    # Accumulators were initialized by the first call, so initialize=False.
    return _pipeline(
        source,
        kb + 3,
        stage1,
        stage0,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        False,
        pipeline_spec,
        tile_spec,
    )


@triton.jit
def _make_global_tile_load_addresses(
    a_ptr,
    b_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    M: tl.constexpr,
    tile_id,
    has_m_tail: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Build the A/B base pointers and fragment offsets for one output tile."""
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    rk = tl.arange(0, _PERSISTENT_BLOCK_K)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    block_m_offset = pid_m * block_m
    block_n_offset = pid_n * block_n
    a_offsets = tl.tuple([(tl.where(
        block_m_offset + mi * TILE + tl.arange(0, TILE) < M,
        block_m_offset + mi * TILE + tl.arange(0, TILE),
        0,
    ) if has_m_tail else block_m_offset + mi * TILE + tl.arange(0, TILE))[:, None] * stride_am + rk[None, :] * stride_ak
                          for mi in range(m_fragments)])
    b_offsets = tl.tuple([
        rk[:, None] * stride_bk + (block_n_offset + nj * TILE + tl.arange(0, TILE))[None, :] * stride_bn
        for nj in range(n_fragments)
    ])
    return tl.tuple([a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets])


@triton.jit
def _consume_k32(
    stage,
    kh: tl.constexpr,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_operands = _local_load_b_row(stage, kh, dot_b, tile_spec)
    for mi in tl.static_range(tl.constexpr(tile_spec[0] // TILE)):
        acc = _mfma_row(
            _local_load_a(stage, kh, mi, dot_a),
            b_operands,
            acc,
            mi,
            mma,
            dot_a,
            dot_b,
            False,
            tile_spec,
        )
    return acc


@triton.jit
def _compute_full_tile(
    source,
    preloaded_k0,
    stage0,
    stage1,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    k_blocks: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Compute one output tile across all compile-time K64 blocks.

    ``preloaded_k0`` is the already-issued global load of K0 held in VGPRs.
    The prologue publishes K0 to stage0 and prefetches K1. Pipeline pairs
    leave the final two blocks in stage0 and prefetch VGPRs; the explicit
    epilogue drains them without issuing an out-of-range prefetch pair.
    """
    tl.static_assert(k_blocks >= 4 and k_blocks % 2 == 0)
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # Prologue: publish the caller's preloaded K0 from VGPRs to stage0, read
    # K0.kh0 into operand VGPRs, and prefetch the complete K1 into VGPRs.
    _local_store_all(stage0, preloaded_k0, tile_spec)
    tl.debug_barrier()
    current_b = _local_load_b_row(stage0, 0, dot_b, tile_spec)
    current_a = tl.tuple([_local_load_a(stage0, 0, mi, dot_a) for mi in range(m_fragments)])
    prefetched = _global_loads(source, 1, tile_spec)

    # Create one persistent FP32 accumulator for every logical C[32, 32]
    # fragment.  The first pair consumes K0/K1 and prepares K2/K3.
    zero = tlx.zeros((TILE, TILE), tl.float32, layout=mma)
    acc = tl.tuple([zero for _ in range(m_fragments * n_fragments)])
    acc, prefetched, current_a, current_b = _pipeline_pair(
        0,
        source,
        stage0,
        stage1,
        current_a,
        current_b,
        prefetched,
        acc,
        mma,
        dot_a,
        dot_b,
        True,
        pipeline_spec,
        tile_spec,
    )

    # Steady state: pair(kb) consumes K(kb)/K(kb+1) and prepares
    # K(kb+2)/K(kb+3), stopping with two blocks for the explicit epilogue.
    for kb in tl.range(2, k_blocks - 2, 2, num_stages=1):
        acc, prefetched, current_a, current_b = _pipeline_pair(
            kb,
            source,
            stage0,
            stage1,
            current_a,
            current_b,
            prefetched,
            acc,
            mma,
            dot_a,
            dot_b,
            False,
            pipeline_spec,
            tile_spec,
        )

    # Epilogue: publish the final prefetched block into stage1 while consuming
    # the penultimate block from stage0, then drain both halves from stage1.
    _local_store_all(stage1, prefetched, tile_spec)
    for mi in tl.static_range(m_fragments):
        acc = _mfma_row(
            current_a[mi],
            current_b,
            acc,
            mi,
            mma,
            dot_a,
            dot_b,
            False,
            tile_spec,
        )
    acc = _consume_k32(stage0, 1, acc, mma, dot_a, dot_b, tile_spec)
    tl.debug_barrier()
    for kh in tl.static_range(2):
        acc = _consume_k32(stage1, kh, acc, mma, dot_a, dot_b, tile_spec)
    return acc


@triton.jit
def _global_store_output(
    c_ptr,
    acc,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    tile_id,
    has_m_tail: tl.constexpr,
    mma: tl.constexpr,
    tile_spec: tl.constexpr,
):
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = n_fragments - n_main_fragments
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    rm = pid_m * block_m + tl.arange(0, TILE)
    rn = pid_n * block_n + tl.arange(0, TILE)
    for mi in tl.static_range(m_fragments):
        for group in tl.static_range(n_main_groups):
            rn_wide = (pid_n * block_n + group * N_GROUP_FRAGMENTS * TILE + tl.arange(0, N_GROUP_FRAGMENTS * TILE))
            offsets = (c_ptr + (rm + mi * TILE)[:, None] * stride_cm + rn_wide[None, :] * stride_cn)
            lo = tl.cat(
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 1],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            hi = tl.cat(
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 2],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[mi * n_fragments + group * N_GROUP_FRAGMENTS + 3],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            value = tl.cat(lo, hi, dim=1)
            value = tlx.require_layout(value.to(c_ptr.dtype.element_ty), _C_STORE_32X128_LAYOUT)
            tlx.assert_same_layout(value, _C_STORE_32X128_LAYOUT)
            if has_m_tail:
                tl.store(
                    offsets,
                    value,
                    mask=(rm + mi * TILE)[:, None] < M,
                )
            else:
                tl.store(offsets, value)
        for tail in tl.static_range(n_tail_fragments):
            offsets = tlx.require_layout(
                c_ptr + (rm + mi * TILE)[:, None] * stride_cm + (rn +
                                                                 (n_main_fragments + tail) * TILE)[None, :] * stride_cn,
                mma,
                pin=False,
            )
            value = tlx.require_layout(
                acc[mi * n_fragments + n_main_fragments + tail],
                mma,
                pin=False,
            )
            if has_m_tail:
                tl.store(
                    offsets,
                    value,
                    mask=(rm + mi * TILE)[:, None] < M,
                )
            else:
                tl.store(offsets, value)


@triton.jit
def _commit_accumulators(acc, mma: tl.constexpr, tile_spec: tl.constexpr):
    accumulators: tl.constexpr = tl.constexpr(tile_spec[0] // TILE * tile_spec[1] // TILE)
    values = [tlx.require_layout(acc[index], mma, pin=False) for index in range(accumulators)]
    return tlx.amd_mfma_commit(tl.tuple(values))


@triton.jit
def _local_alloc_stage(
    a_layout: tl.constexpr,
    b_main_layout: tl.constexpr,
    b_tail_layout: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = (n_fragments - n_main_groups * N_GROUP_FRAGMENTS)
    a_buffers = tl.tuple(
        [tlx.local_alloc((TILE, _PERSISTENT_BLOCK_K), tl.float16, 1, layout=a_layout) for _ in range(m_fragments)])
    b_buffers = tl.tuple([
        tlx.local_alloc(
            (_PERSISTENT_BLOCK_K, N_GROUP_FRAGMENTS * TILE),
            tl.float16,
            1,
            layout=b_main_layout,
        ) for _ in range(n_main_groups)
    ] + [
        tlx.local_alloc((_PERSISTENT_BLOCK_K, TILE), tl.float16, 1, layout=b_tail_layout)
        for _ in range(n_tail_fragments)
    ])
    return tl.tuple([a_buffers, b_buffers])


@triton.jit
def _persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    HAS_M_TAIL: tl.constexpr,
    TILE_SPEC: tl.constexpr,
    PIPELINE_SPEC: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(64, 16)],
                                                                                  [TILE, _PERSISTENT_BLOCK_K],
                                                                                  order=[1, 0]))
    # Keep N contiguous in LDS. This turns each coalesced B load into one
    # ds_write_b128 per lane and lets the dot operand use ds_read_b64_tr_b16.
    b_main_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(512, 16)],
                                                                                       [_PERSISTENT_BLOCK_K, 128],
                                                                                       order=[1, 0]))
    b_tail_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_identity_for([(256, 16)],
                                                                                       [_PERSISTENT_BLOCK_K, TILE],
                                                                                       order=[1, 0]))
    stage0 = _local_alloc_stage(a_layout, b_main_layout, b_tail_layout, TILE_SPEC)
    stage1 = _local_alloc_stage(a_layout, b_main_layout, b_tail_layout, TILE_SPEC)

    # Persistently assign a compile-time group of adjacent N tiles to each
    # program.  For example, 128 N tiles with two tiles/program produce 64
    # programs per M row: program 0 -> tiles 0/1, ..., program 63 -> 126/127.
    program = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.constexpr(TILE_SPEC[2])
    tiles_per_program: tl.constexpr = tl.constexpr(TILE_SPEC[4])
    tl.static_assert(tiles_per_program > 0)
    tl.static_assert(num_pid_n % tiles_per_program == 0)
    programs_per_m = num_pid_n // tiles_per_program
    program_m = program // programs_per_m
    program_n = program % programs_per_m
    first_tile = (program_m * num_pid_n + program_n * tiles_per_program)

    # Prime the persistent traversal with the first tile's complete K0 slice.
    tile_load_addresses = _make_global_tile_load_addresses(
        a_ptr,
        b_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        M,
        first_tile,
        HAS_M_TAIL,
        TILE_SPEC,
    )
    prefetched_k0 = _global_loads(tile_load_addresses, 0, TILE_SPEC)

    for tile_offset in tl.static_range(tiles_per_program):
        tile_id = first_tile + tile_offset
        acc_tile = _compute_full_tile(
            tile_load_addresses,
            prefetched_k0,
            stage0,
            stage1,
            mma,
            dot_a,
            dot_b,
            K_BLOCKS,
            PIPELINE_SPEC,
            TILE_SPEC,
        )

        # Before the current tile's epilogue, issue the next tile's K0 loads.
        # At most one completed accumulator tile and one future K0 prefetch are
        # live together, independent of tiles_per_program.
        if tile_offset + 1 < tiles_per_program:
            next_tile_load_addresses = _make_global_tile_load_addresses(
                a_ptr,
                b_ptr,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                M,
                tile_id + 1,
                HAS_M_TAIL,
                TILE_SPEC,
            )
            next_prefetched_k0 = _global_loads(next_tile_load_addresses, 0, TILE_SPEC)

        acc_tile = _commit_accumulators(acc_tile, mma, TILE_SPEC)
        _global_store_output(
            c_ptr,
            acc_tile,
            stride_cm,
            stride_cn,
            M,
            tile_id,
            HAS_M_TAIL,
            mma,
            TILE_SPEC,
        )

        if tile_offset + 1 < tiles_per_program:
            tile_load_addresses = next_tile_load_addresses
            prefetched_k0 = next_prefetched_k0


def _validate_specialization(m, n, tile_spec, pipeline_spec):
    block_m, block_n, num_pid_n, num_programs, tiles_per_program = tile_spec
    m_fragments = block_m // int(TILE)
    n_fragments = block_n // int(TILE)
    a1_read_plan, mfmas_before_prefetch, b_publish_plan, finish_plan = (pipeline_spec)
    assert num_pid_n == (n + block_n - 1) // block_n
    assert tiles_per_program > 0
    assert num_pid_n % tiles_per_program == 0
    assert num_programs * tiles_per_program == ((m + block_m - 1) // block_m) * num_pid_n
    assert len(a1_read_plan) == m_fragments
    assert 0 <= mfmas_before_prefetch <= n_fragments
    assert len(b_publish_plan) == n_fragments
    assert len(finish_plan) == m_fragments + n_fragments

    a1_reads = []
    for first, count in a1_read_plan:
        assert 0 <= first <= m_fragments and 0 <= count <= m_fragments - first
        a1_reads.extend(range(first, first + count))
    assert sorted(a1_reads) == list(range(m_fragments))

    # K1 coverage consists of row0's mandatory B publishes, the extra spans
    # attached to each publish, the final-read plan, and the last MFMA row.
    mfma_coverage = list(range(n_fragments))
    for parts in b_publish_plan:
        for mi, first_nj, count in parts:
            if count == 0:
                continue
            assert 0 <= mi < m_fragments
            assert 0 <= first_nj < n_fragments
            assert first_nj + count <= n_fragments
            mfma_coverage.extend(mi * n_fragments + nj for nj in range(first_nj, first_nj + count))
    assert all(0 <= flat < m_fragments * n_fragments for flat in finish_plan)
    mfma_coverage.extend(finish_plan)
    mfma_coverage.extend(range((m_fragments - 1) * n_fragments, m_fragments * n_fragments))
    assert sorted(mfma_coverage) == list(range(m_fragments * n_fragments))


def _persistent_plan_for_shape(m, n, k, dtype):
    if dtype != torch.float16:
        return None
    return _SHAPE_DEFAULTS.get((m, n, k))


def _persistent_supports(a, b):
    """Return whether a and b select one of the persistent specializations."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return False
    m, k = a.shape
    _, n = b.shape
    return a.dtype == b.dtype and _persistent_plan_for_shape(m, n, k, a.dtype) is not None


def _launch_persistent_specialization(a, b, out, tile_spec, pipeline_spec):
    _persistent_kernel[(tile_spec[3], )](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        M=a.shape[0],
        K_BLOCKS=a.shape[1] // int(_PERSISTENT_BLOCK_K),
        HAS_M_TAIL=a.shape[0] % tile_spec[0] != 0,
        TILE_SPEC=tile_spec,
        PIPELINE_SPEC=pipeline_spec,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=1,
        regclass_priority_trumps_globalness=True,
        reverse_local_assignment=True,
    )


def _launch_persistent(a, b, out=None, specialization=None):
    """Run a compile-time gfx950 persistent GEMM specialization."""
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    kb, n = b.shape
    assert k == kb
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if specialization is None:
        specialization = os.environ.get("TLX_GFX950_TILE")
    if specialization is None:
        specialization = _persistent_plan_for_shape(m, n, k, a.dtype)
        assert specialization is not None
    assert specialization in _SPECIALIZATIONS
    expected_shape, tile_spec, pipeline_spec = _SPECIALIZATIONS[specialization]
    assert (m, n, k) == expected_shape
    _validate_specialization(m, n, tile_spec, pipeline_spec)
    _launch_persistent_specialization(
        a,
        b,
        out,
        tile_spec,
        pipeline_spec,
    )
    return out


def _launch_persistent_mtail_n160(a, b, out=None):
    """Run the tuned N160 pipeline for one partial 224-row tile."""
    m, k = a.shape
    kb, n = b.shape
    assert 0 < m <= 224 and k == kb == 6144 and n == 40960
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    _launch_persistent_specialization(
        a,
        b,
        out,
        _MT224X160_MTAIL_TILE_SPEC,
        _MT224X160_MTAIL_PIPELINE_SPEC,
    )
    return out


# LocalSplitU path and public dispatch.

__all__ = ["mm", "matmul", "supports"]


# The initial implementation deliberately exposes only measured plans. A
# follow-up change adds bounded plan generation for neighboring shapes without
# changing this register-staged execution mechanism.
class _Plan(NamedTuple):
    tile_m: int
    tile_n: int
    local_split_u: int
    wave_k: int
    k_width: int


_KNOWN_PLANS = {
    # Match the vendor's M16xN32xK512 geometry with four K128 partitions.
    # Three direct-to-LDS stages keep all four macros in flight and cut the
    # launch grid in half relative to the narrower M7 specialization.
    (7, 8192, 2048):
    _Plan(tile_m=16, tile_n=32, local_split_u=4, wave_k=128, k_width=8),
    # N16 gives two output tiles per CU. Keep the latest-master K256
    # partitions: their contraction order is required to match ATen on
    # cancellation-sensitive inputs.
    (7, 2048, 4096):
    _Plan(
        tile_m=16,
        tile_n=16,
        local_split_u=4,
        wave_k=256,
        k_width=8,
    ),
}

# Three direct-to-LDS stages match the depth of the vendor M7 kernels.  The
# N16 path uses 32 KiB per stage; N32 widens only B and uses 48 KiB.  Putting
# one LocalSplitU bit in the value mode gives every lane aligned fp16x8 vectors
# instead of the unsupported fp16x4 width of a plain four-wave mapping.
_LOCAL_SPLIT_U_A_RAW_LAYOUT = tlx.shared_linear_layout_encoding(
    [
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 4],
        [0, 0, 8],
        [0, 0, 16],
        [0, 0, 32],
        [0, 1, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 8, 0],
        [1, 0, 0],
        [2, 0, 0],
    ],
    [],
    16,
)
_LOCAL_SPLIT_U_B_RAW_LAYOUT = tlx.shared_linear_layout_encoding(
    [
        [0, 1, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 8, 0],
        [0, 16, 0],
        [0, 32, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 4],
        [0, 0, 8],
        [1, 0, 0],
        [2, 0, 0],
    ],
    [],
    16,
)
_LOCAL_SPLIT_U_A_DIRECT_LAYOUT = tlx.layout(
    shape=((8, 16, 2), (8, 2)),
    stride=((8, 64, 2048), (1, 1024)),
)
_LOCAL_SPLIT_U_B_DIRECT_LAYOUT = tlx.layout(
    shape=((8, 16, 2), (8, 2)),
    stride=((128, 1, 2048), (16, 1024)),
)
_LOCAL_SPLIT_U_B32_RAW_LAYOUT = tlx.shared_linear_layout_encoding(
    [
        [0, 1, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 8, 0],
        [0, 16, 0],
        [0, 32, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 4],
        [0, 0, 8],
        [0, 0, 16],
        [1, 0, 0],
        [2, 0, 0],
    ],
    [],
    16,
)
_LOCAL_SPLIT_U_B32_DIRECT_LAYOUT = tlx.layout(
    shape=((8, 16, 2), (8, 2, 2)),
    stride=((256, 1, 4096), (32, 2048, 16)),
)


@lru_cache(maxsize=None)
def _device_arch(device):
    """Return the AMD architecture for a CUDA device, or an empty string."""
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


@triton.jit
def _local_split_u_direct_load_half(
    a_ptr,
    b_ptr,
    a_buffer,
    b_buffer,
    pid_n,
    stage: tl.constexpr,
    macro: tl.constexpr,
    half: tl.constexpr,
    tile_n: tl.constexpr,
    m: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    a_direct_layout: tl.constexpr,
    b_direct_layout: tl.constexpr,
):
    splits = tl.arange(0, 4).to(tl.int32)
    rows = tl.arange(0, 16).to(tl.int32)
    rows = tl.where(rows < m, rows, 0)
    cols = pid_n * tile_n + tl.arange(0, tile_n).to(tl.int32)
    rk = tl.arange(0, 64).to(tl.int32)
    ks = macro * 512 + splits[:, None, None] * 128 + half * 64
    a_offsets = (rows[None, :, None] * stride_am + (ks + rk[None, None, :]) * stride_ak)
    b_offsets = ((ks + rk[None, :, None]) * stride_bk + cols[None, None, :] * stride_bn)
    a_offsets = tl.max_contiguous(tl.multiple_of(a_offsets, (1, 1, 8)), (1, 1, 8))
    b_offsets = tl.max_contiguous(tl.multiple_of(b_offsets, (1, 8, 1)), (1, 8, 1))
    a_offsets = tlx.require_layout(a_offsets, a_direct_layout)
    b_offsets = tlx.require_layout(b_offsets, b_direct_layout)
    a_token = tlx.buffer_load_to_local(
        tlx.local_view(a_buffer, stage),
        a_ptr,
        a_offsets,
        contiguity=8,
    )
    b_token = tlx.buffer_load_to_local(
        tlx.local_view(b_buffer, stage),
        b_ptr,
        b_offsets,
        cache_modifier=".cg",
        contiguity=8,
    )
    tlx.async_load_commit_group([a_token, b_token])


@triton.jit
def _local_split_u_lds_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    a_raw_layout: tl.constexpr,
    b_raw_layout: tl.constexpr,
    a_direct_layout: tl.constexpr,
    b_direct_layout: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Run the measured M7 LocalSplitU direct-to-LDS pipeline."""
    NUM_MACROS: tl.constexpr = K // 512
    tl.static_assert(M == 7
                     and ((TILE_N == 16 and N == 2048 and K == 4096) or (TILE_N == 32 and N == 8192 and K == 2048)))
    pid_n = tl.program_id(0).to(tl.int32)
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1, 1],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    element_type: tl.constexpr = tlx.dtype_of(a_ptr)
    a_lo = tlx.local_alloc(
        (4, 16, 64),
        element_type,
        3,
        layout=a_raw_layout,
    )
    a_hi = tlx.local_alloc(
        (4, 16, 64),
        element_type,
        3,
        layout=a_raw_layout,
    )
    b_lo = tlx.local_alloc(
        (4, 64, TILE_N),
        element_type,
        3,
        layout=b_raw_layout,
    )
    b_hi = tlx.local_alloc(
        (4, 64, TILE_N),
        element_type,
        3,
        layout=b_raw_layout,
    )
    for stage in tl.static_range(3):
        _local_split_u_direct_load_half(
            a_ptr,
            b_ptr,
            a_lo,
            b_lo,
            pid_n,
            stage,
            stage,
            0,
            TILE_N,
            M,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            a_direct_layout,
            b_direct_layout,
        )
        _local_split_u_direct_load_half(
            a_ptr,
            b_ptr,
            a_hi,
            b_hi,
            pid_n,
            stage,
            stage,
            1,
            TILE_N,
            M,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            a_direct_layout,
            b_direct_layout,
        )

    acc = tlx.zeros((4, 16, TILE_N), tl.float32, layout=mma)
    for macro in tl.static_range(NUM_MACROS):
        tlx.async_load_wait_group(2 * min(2, NUM_MACROS - 1 - macro))
        tl.debug_barrier()
        a0 = tlx.local_load(tlx.local_view(a_lo, macro % 3), layout=dot_a, relaxed=True)
        b0 = tlx.local_load(tlx.local_view(b_lo, macro % 3), layout=dot_b, relaxed=True)
        a1 = tlx.local_load(tlx.local_view(a_hi, macro % 3), layout=dot_a, relaxed=True)
        b1 = tlx.local_load(tlx.local_view(b_hi, macro % 3), layout=dot_b, relaxed=True)
        tl.debug_barrier()
        if macro + 3 < NUM_MACROS:
            _local_split_u_direct_load_half(
                a_ptr,
                b_ptr,
                a_lo,
                b_lo,
                pid_n,
                macro % 3,
                macro + 3,
                0,
                TILE_N,
                M,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                a_direct_layout,
                b_direct_layout,
            )
            _local_split_u_direct_load_half(
                a_ptr,
                b_ptr,
                a_hi,
                b_hi,
                pid_n,
                macro % 3,
                macro + 3,
                1,
                TILE_N,
                M,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                a_direct_layout,
                b_direct_layout,
            )
        acc = tl.dot(a0, b0, acc, allow_tf32=False, out_dtype=tl.float32)
        acc = tl.dot(a1, b1, acc, allow_tf32=False, out_dtype=tl.float32)

    partial_buffer = tlx.local_alloc((4, 16, TILE_N), tl.float32, 1)
    partial_view = tlx.local_view(partial_buffer, 0)
    tlx.local_store(partial_view, acc)
    tl.debug_barrier()
    result = tl.reshape(
        tlx.local_load(tlx.local_slice(partial_view, [0, 0, 0], [1, 16, TILE_N])),
        (16, TILE_N),
    )
    for split in tl.static_range(1, 4):
        partial = tlx.local_load(tlx.local_slice(partial_view, [split, 0, 0], [1, 16, TILE_N]))
        result += tl.reshape(partial, (16, TILE_N))

    rows = tl.arange(0, 16)
    cols = pid_n * TILE_N + tl.arange(0, TILE_N)
    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        result.to(c_ptr.dtype.element_ty),
        mask=rows[:, None] < M,
    )


@triton.jit
def _load_dot_operands(
    a_ptr,
    b_ptr,
    global_rows,
    global_cols,
    split_ids,
    rk,
    macro_k_base,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    WAVE_K: tl.constexpr,
    K_WIDTH: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
):
    """Load one macro-K slice directly into the two MFMA operand layouts."""
    split_k = macro_k_base + split_ids[:, None, None] * WAVE_K
    a_offsets = (global_rows[None, :, None] * stride_am + (split_k + rk[None, None, :]) * stride_ak)
    b_offsets = ((split_k + rk[None, :, None]) * stride_bk + global_cols[None, None, :] * stride_bn)
    # The offsets already have their dot-operand layouts, so the loaded values
    # reach MFMA registers without an intervening conversion through LDS.
    # K_WIDTH is also the largest contiguous run owned by one lane; claiming a
    # wider buffer vector would cross the lane's two disjoint K runs.
    a_offsets = tlx.require_layout(a_offsets, dot_a)
    b_offsets = tlx.require_layout(b_offsets, dot_b)
    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=K_WIDTH)
    b = tlx.buffer_load(b_ptr, b_offsets, cache=".cg", contiguity=K_WIDTH)
    return a, b


@triton.jit
def _local_split_u_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    K_WIDTH: tl.constexpr,
    WAVE_K: tl.constexpr,
    LOCAL_SPLIT_U: tl.constexpr,
):
    """Compute one MxN tile by partitioning K across the CTA's waves."""
    MACRO_K: tl.constexpr = WAVE_K * LOCAL_SPLIT_U
    tl.static_assert(M <= TILE_M)

    pid_n = tl.program_id(0).to(tl.int32)
    split_ids = tl.arange(0, LOCAL_SPLIT_U).to(tl.int32)
    rows = tl.arange(0, TILE_M).to(tl.int32)
    # Padded rows may read any valid A row because their results are discarded.
    output_rows = rows
    global_rows = tl.where(rows < M, rows, 0)
    local_cols = tl.arange(0, TILE_N).to(tl.int32)
    output_cols = pid_n * TILE_N + local_cols
    global_cols = tl.where(output_cols < N, output_cols, 0)
    rk = tl.arange(0, WAVE_K).to(tl.int32)

    # The leading batch axis is the LocalSplitU partition. Mapping that axis
    # one-to-one onto waves keeps the wave-specific K coordinate explicit.
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[LOCAL_SPLIT_U, 1, 1],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)

    tl.static_assert(K % MACRO_K == 0)
    acc = tlx.zeros(
        (LOCAL_SPLIT_U, TILE_M, TILE_N),
        tl.float32,
        layout=mma,
    )

    current_a, current_b = _load_dot_operands(
        a_ptr,
        b_ptr,
        global_rows,
        global_cols,
        split_ids,
        rk,
        0,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        WAVE_K,
        K_WIDTH,
        dot_a,
        dot_b,
    )

    # One-stage register pipeline: issue K(t+1)'s global loads before K(t)'s
    # dot. The chosen WAVE_K controls the prefetch lifetime and register cost.
    for macro in tl.range(0, K // MACRO_K - 1, num_stages=1):
        next_k = (macro + 1) * MACRO_K
        next_a, next_b = _load_dot_operands(
            a_ptr,
            b_ptr,
            global_rows,
            global_cols,
            split_ids,
            rk,
            next_k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            WAVE_K,
            K_WIDTH,
            dot_a,
            dot_b,
        )
        acc = tl.dot(
            current_a,
            current_b,
            acc,
            allow_tf32=False,
            out_dtype=tl.float32,
        )
        current_a = next_a
        current_b = next_b

    acc = tl.dot(
        current_a,
        current_b,
        acc,
        allow_tf32=False,
        out_dtype=tl.float32,
    )

    if LOCAL_SPLIT_U == 16 and TILE_N == 32:
        # This swizzle is tuned for the dense U16/N32 partial tile. Its encoding
        # depends on TILE_N, so other U16 widths deliberately use the default
        # layout instead of silently inheriting different swizzle parameters.
        partial_layout: tl.constexpr = tlx.swizzled_layout(2, 2, 3, order=[2, 1, 0])
        partial_buffer = tlx.local_alloc(
            (LOCAL_SPLIT_U, TILE_M, TILE_N),
            tl.float32,
            1,
            layout=partial_layout,
        )
    else:
        partial_buffer = tlx.local_alloc((LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, 1)
    partial_view = tlx.local_view(partial_buffer, 0)
    tlx.local_store(partial_view, acc)
    tl.debug_barrier()

    tl.static_assert(LOCAL_SPLIT_U == 2 or LOCAL_SPLIT_U == 4 or LOCAL_SPLIT_U == 8 or LOCAL_SPLIT_U == 16)
    result = tl.reshape(
        tlx.local_load(tlx.local_slice(
            partial_view,
            [0, 0, 0],
            [1, TILE_M, TILE_N],
        )),
        (TILE_M, TILE_N),
    )
    # Load and immediately consume each subsequent partial. This preserves
    # increasing split-id association without keeping every partial live.
    for split in tl.static_range(1, LOCAL_SPLIT_U):
        partial = tlx.local_load(tlx.local_slice(
            partial_view,
            [split, 0, 0],
            [1, TILE_M, TILE_N],
        ))
        result += tl.reshape(partial, (TILE_M, TILE_N))

    output_ptrs = (c_ptr + output_rows[:, None] * stride_cm + output_cols[None, :] * stride_cn)
    tl.store(
        output_ptrs,
        result.to(c_ptr.dtype.element_ty),
        mask=(output_rows[:, None] < M)
        & (output_cols[None, :] < N),
    )


def _problem_for(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    m, k = a.shape
    _, n = b.shape
    if not (a.dtype in (torch.float16, torch.bfloat16) and b.dtype == a.dtype and a.is_cuda and a.device == b.device
            and _device_arch(a.device) == "gfx950" and a.stride(1) == 1 and b.stride(0) == 1):
        return None
    return m, n, k


def _plan_for(a, b):
    """Return the measured LocalSplitU plan, otherwise ``None``."""
    problem = _problem_for(a, b)
    if problem is None or a.dtype != torch.float16:
        return None
    return _KNOWN_PLANS.get(problem)


def _uses_local_split_u_lds_for_shape(m, n, k):
    # K4096/N2048 must retain the register-staged K128 contraction order to
    # match ATen on cancellation-sensitive inputs. The K2048/N8192 promotion
    # is numerically equivalent and keeps its measured direct-to-LDS win.
    return (m, n, k) == (7, 8192, 2048)


def _launch_validated(a, b, out, plan):
    """Launch a plan after operand and output validation has completed."""
    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n, local_split_u, wave_k, k_width = plan
    if m > tile_m:
        raise InvalidInput(f"gfx950 LocalSplitU plan covers at most {tile_m} rows; got M={m}")
    if _uses_local_split_u_lds_for_shape(m, n, k):
        b_raw_layout = (_LOCAL_SPLIT_U_B32_RAW_LAYOUT if tile_n == 32 else _LOCAL_SPLIT_U_B_RAW_LAYOUT)
        b_direct_layout = (_LOCAL_SPLIT_U_B32_DIRECT_LAYOUT if tile_n == 32 else _LOCAL_SPLIT_U_B_DIRECT_LAYOUT)
        _local_split_u_lds_kernel[(triton.cdiv(n, tile_n), )](
            a,
            b,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            _LOCAL_SPLIT_U_A_RAW_LAYOUT,
            b_raw_layout,
            _LOCAL_SPLIT_U_A_DIRECT_LAYOUT,
            b_direct_layout,
            TILE_N=tile_n,
            num_warps=local_split_u,
            num_stages=1,
            matrix_instr_nonkdim=16,
            waves_per_eu=0,
        )
        return out
    _local_split_u_kernel[(triton.cdiv(n, tile_n), )](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_M=tile_m,
        TILE_N=tile_n,
        K_WIDTH=k_width,
        WAVE_K=wave_k,
        LOCAL_SPLIT_U=local_split_u,
        num_warps=local_split_u,
        num_stages=1,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
        sink_insts_to_avoid_spills=True,
    )
    return out


def _register_plan_for(a, b):
    problem = _problem_for(a, b)
    if problem is None:
        return None
    m, n, k = problem
    if min(m, n, k) <= 0:
        return None
    return _register_plan_for_shape(m, n, k, a.dtype)


def _lds_plan_for(a, b):
    problem = _problem_for(a, b)
    if problem is None:
        return None
    m, n, k = problem
    if min(m, n) <= 0 or k < 128 or k * a.element_size() % 16 != 0:
        return None
    block_m, block_n, split_k = _lds_plan_for_shape(m, n, k)
    if not _valid_lds_split(k, split_k, a.element_size()):
        return None
    return block_m, block_n, split_k


def _valid_lds_split(k, split_k, element_size):
    if k % split_k == 0:
        split_size = k // split_k
    elif k % BLOCK_K == 0:
        split_size = (k // BLOCK_K // split_k) * BLOCK_K
    else:
        return False
    return (split_size >= 2 * BLOCK_K and split_size * element_size % 16 == 0)


def _launch_optimized_inter_wave(a, b, out, kind, plan=None):
    """Lazily enter the promoted inter-wave implementation family."""
    if kind == "streamk":
        return _iw_streamk_matmul(a, b, out=out, **({} if plan is None else plan))
    if plan is not None:
        return _iw_launch(a, b, out=out, **plan)
    if kind == "ragged_n":
        return _iw_ragged_n_matmul(a, b, out=out)
    if kind == "m192n256":
        m, k = a.shape
        n = b.shape[1]
        wide_shallow = k == 512 and 768 < m <= 960 and n >= 96 * m
        tall_short = k == 768 and m >= 65536 and 768 < n <= 1664
        device_waves = (((m + 191) // 192) * ((n + 255) // 256) + _NUM_CU - 1) // _NUM_CU
        return _iw_launch(
            a,
            b,
            out=out,
            TILE=(192, 256),
            SCHED_MFMA_PER_DWORDX4=4 if tall_short else 2,
            WORKGROUP_MAPPING=(0 if tall_short else 8 if device_waves >= 3 else 0),
            GROUP_M_OVERRIDE=16 if tall_short else None,
            NUM_XCDS_OVERRIDE=2 if wide_shallow or tall_short else None,
        )
    return _iw_matmul(a, b, out=out)


def _launch_hybrid_n160(a, b, out):
    """Split one sparse N wave into full N256 and persistent N160 launches."""
    main_n = b.shape[1] - 160 * _NUM_CU
    _iw_matmul(a, b[:, :main_n], out=out[:, :main_n])
    _launch_persistent_mtail_n160(
        a,
        b[:, main_n:],
        out=out[:, main_n:],
    )
    return out


_BF16_INTER_WAVE_PROMOTIONS = {
    # This balanced MT256 family uses the same numerical reduction order as
    # the LDS fallback, but the extracted direct-to-LDS pipeline overlaps its
    # loads more effectively.  Paired 181-round runs on two GPUs measured
    # 1.0096x and 1.0097x over the previous LDS route.
    (2048, 25408, 10240):
    None,
}

_BF16_STREAMK_PROMOTIONS = {}


def _supports_short_k_register(m, n, k, dtype):
    """Select the one-K64 register kernel for sufficiently parallel BF16 GEMMs."""
    return (dtype == torch.bfloat16 and 0 < k <= 64 and m >= 256 and n >= 256)


@lru_cache(maxsize=None)
def _dispatch_plan(m, n, k, dtype, element_size):
    shape = (m, n, k)
    wave_grid_plan = _wave_grid_plan_for_shape(m, n, k, dtype)
    if dtype == torch.bfloat16:
        streamk_plan = _BF16_STREAMK_PROMOTIONS.get(shape)
        if streamk_plan is not None:
            return "streamk", streamk_plan
        promotion = _BF16_INTER_WAVE_PROMOTIONS.get(shape)
        if promotion is not None or shape in _BF16_INTER_WAVE_PROMOTIONS:
            return "inter_wave", promotion
        if _supports_short_k_register(m, n, k, dtype):
            return "short_k_register", None
    if dtype == torch.bfloat16 and wave_grid_plan is not None:
        return "wave_grid", wave_grid_plan
    if dtype == torch.float16:
        local_split_u_plan = _KNOWN_PLANS.get((m, n, k))
        if local_split_u_plan is not None:
            return "local_split_u", local_split_u_plan
        persistent_plan = _persistent_plan_for_shape(m, n, k, dtype)
        if persistent_plan is not None:
            return "persistent", persistent_plan
        if supports_hybrid_n160(m, n, k):
            return "hybrid_n160", None
        transposed_plan = transposed_wave_grid_plan(m, n, k)
        if transposed_plan is not None:
            return "transposed_wave_grid", transposed_plan
        if (prefer_tuned_wave_grid(m, n, k, wave_grid_plan) and not prefer_lds_over_wave_grid(m, n, k, wave_grid_plan)):
            return "wave_grid", wave_grid_plan
    strong_lds_plan = _strong_lds_plan(m, n, k)
    if strong_lds_plan is not None:
        return "lds", strong_lds_plan
    register_plan = _register_plan_for_shape(m, n, k, dtype)
    if register_plan is not None:
        return "register", register_plan
    if dtype == torch.float16:
        if prefer_streamk(m, n, k):
            return "streamk", None
        if prefer_ragged_n(m, n, k, wave_grid_plan):
            return "ragged_n", None
        if prefer_m192n256(m, n, k):
            return "m192n256", None
        if wave_grid_plan is not None:
            if prefer_lds_over_wave_grid(m, n, k, wave_grid_plan):
                return "inter_wave", None
            return "wave_grid", wave_grid_plan
    if min(m, n) <= 0 or k < 128 or k * element_size % 16 != 0:
        return None
    if dtype == torch.float16:
        return "inter_wave", None
    block_m, block_n, split_k = _lds_plan_for_shape(m, n, k)
    if not _valid_lds_split(k, split_k, element_size):
        return None
    return "lds", (block_m, block_n, split_k)


def _incumbent_heuristic_config(m, n, k, dtype, element_size, a_strides, b_strides):
    """Return one production plan selected from measured gfx950 families."""

    def register(block_m, block_n, block_k, group_m, num_xcds, waves_per_eu, num_warps, num_stages):
        return "register", {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": num_xcds,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    if _supports_short_k_register(m, n, k, dtype):
        return "short_k_register", None

    # Only genuinely strided operands retain the generic register fallback; either dense B orientation is eligible below.
    if a_strides[1] != 1 or (b_strides[0] != 1 and b_strides[1] != 1):
        return "register", _register_plan_for_shape(m, n, k) or _intermediate_register_config(m, n, k)

    # Measured fixed plans are layout-independent register kernels.  Consult
    # them before the dense-B orientation rules so a row-major problem can
    # use the same validated winner as its TN counterpart.
    if dtype == torch.float16:
        tuned = _FP16_TUNED_SHAPE_CONFIGS.get((m, n, k))
        if tuned is not None:
            return "register", tuned

    # Small row-major-B problems outside the explicit measured catalog retain
    # the generic register fallback. The inter-wave selector below is tuned
    # for K-contiguous B and is not a safe implicit route for this orientation.
    if b_strides[0] != 1 and m < 1024:
        return "register", _register_plan_for_shape(m, n, k) or _intermediate_register_config(m, n, k)

    # Dense row-major B favors a larger output tile than the TN families.
    # The wider tile reduces loop and launch overhead while its eight waves
    # preserve enough parallel memory work for the non-unit K stride.
    if b_strides[1] == 1:
        if m >= 4096 and n >= 1024 and k <= 4096:
            return register(256, 256, 64, 8, 8, 0, 8, 2)
        if m == 1024 and n >= 16384 and k >= 4096:
            return register(256, 256, 64, 8, 8, 0, 8, 2)

    # Sub-1024 row problems retain their measured LDS, persistent, and LocalSplitU algorithm choices.
    if m < 1024:
        return _dispatch_plan(m, n, k, dtype, element_size)

    # Small output grids and shallow medium-M reductions retain their measured incumbent algorithms.
    if m < 4096 and (m * n <= 4 * 1024 * 1024 or n <= 512 or k <= 512):
        return _dispatch_plan(m, n, k, dtype, element_size)

    # Extremely reduction-heavy matrices retain the incumbent split-K plan.
    if k >= 64 * n:
        return _dispatch_plan(m, n, k, dtype, element_size)

    # Short reductions favor the measured 128x256 K32 wave-limited family.
    if k <= 256:
        return register(128, 256, 32, 16, 1, 2, 4, 2)

    # Narrow outputs favor the measured 256x256 two-stage family despite the N tail.
    if n <= 256:
        return register(256, 256, 64, 4, 1, 0, 8, 2)

    # Extreme N-major shapes favor the wide-N K32 family.
    if n >= 8 * k:
        return register(128, 256, 32, 16, 1, 2, 4, 2)

    # Large-M throughput shapes favor the measured 256x256 two-stage family.
    if m >= 16384:
        return register(256, 256, 64, 4, 1, 0, 8, 2)

    # Low-M N-major shapes amortize best with the larger square tile.
    if n >= 2 * k and m <= 1024:
        return register(256, 256, 64, 4, 1, 0, 8, 2)

    # Low-M broad K-major shapes favor the measured XCD-swizzled square family.
    if m <= 1024 and k >= 2 * n and n >= 4096:
        return register(128, 128, 64, 8, 8, 0, 4, 2)

    # Remaining N-major shapes favor the XCD-swizzled 128x128 family.
    if n >= 2 * k:
        return register(128, 128, 64, 16, 8, 0, 4, 2)

    # Broad K-major outputs favor the non-swizzled 128x128 family.
    if k >= 2 * n and n >= 4096:
        return register(128, 128, 64, 16, 1, 0, 4, 2)

    # Narrower K-major outputs favor the XCD-swizzled 128x128 family with shorter grouping.
    if k >= 2 * n:
        return register(128, 128, 64, 8, 8, 0, 4, 2)

    # Balanced shapes with at least 4096 rows favor the wide-N K32 family.
    if m >= 4096:
        return register(128, 256, 32, 16, 1, 2, 4, 2)

    # Remaining balanced shapes use the measured XCD-swizzled square family.
    return register(128, 128, 64, 8, 8, 0, 4, 2)


def heuristic_config(m, n, k, dtype, element_size, a_strides, b_strides):
    """Return one gfx950 plan from the geometry default or incumbent policy."""
    candidates = _range_dispatch_candidates(m, n, k, dtype, element_size, a_strides, b_strides, include_incumbent=False)
    if candidates:
        return candidates[0]
    return _incumbent_heuristic_config(m, n, k, dtype, element_size, a_strides, b_strides)


def _dispatch_for(a, b):
    if a.dtype == torch.bfloat16:
        ranged = _range_dispatch_for(a, b)
        if ranged is not None:
            return ranged
    if (a.ndim == 2 and b.ndim == 2 and a.dtype == torch.float16 and b.dtype == a.dtype and a.is_cuda
            and a.device == b.device and a.shape[1] == b.shape[0] and a.stride(1) == 1 and b.stride(1) == 1
            and _wg_device_arch(a.device) == "gfx950"):
        # The persistent MT256x160 specialization is also legal for dense
        # row-major B.  Its two-output-tile traversal and tile-boundary
        # prefetch remain materially faster than the generic row-major path.
        persistent_plan = _persistent_plan_for_shape(a.shape[0], b.shape[1], a.shape[1], a.dtype)
        if persistent_plan is not None:
            return "persistent", persistent_plan
        plan = _ROW_MAJOR_WAVE_GRID_PLANS.get((a.shape[0], b.shape[1], a.shape[1]))
        if plan is not None:
            return "wave_grid", plan
        plan = _ROW_MAJOR_DIRECT_PLANS.get((a.shape[0], b.shape[1], a.shape[1]))
        if plan is not None:
            return "row_major_direct", plan
    problem = _problem_for(a, b)
    if problem is None:
        return _range_dispatch_for(a, b)
    return _dispatch_plan(*problem, a.dtype, a.element_size())


def supports(a, b):
    """Return whether a and b select a validated gfx950 GEMM plan."""
    return _dispatch_for(a, b) is not None


def _launch_dispatch(a, b, out, dispatch):
    path, plan = dispatch
    if path == "short_k_register":
        return _launch_short_k_register(a, b, out=out)
    if path == "row_major_direct":
        return _launch_row_major_direct(a, b, out, plan)
    if path == "hybrid_n160":
        return _launch_hybrid_n160(a, b, out)
    if path in ("inter_wave", "streamk", "ragged_n", "m192n256"):
        return _launch_optimized_inter_wave(a, b, out, path, plan)
    if path == "transposed_wave_grid":
        _launch_wave_grid(
            b.T,
            a.T,
            out=out.T,
            _candidate_plan=plan,
        )
        return out
    if path == "wave_grid":
        return _launch_wave_grid(a, b, out=out, _candidate_plan=plan)
    if path == "persistent":
        return _launch_persistent(a, b, out=out)
    if path == "range_register":
        return _launch_range_register_plan(a, b, config=plan, out=out)
    if path == "register":
        return _launch_register_plan(
            a,
            b,
            config=plan,
            out=out,
            _validated=True,
        )
    if path == "lds":
        block_m, block_n, split_k = plan
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
            out=out,
        )
    return _launch_validated(a, b, out, plan)


def matmul(a, b, out=None):
    """Run the selected gfx950 GEMM specialization."""
    dispatch = _dispatch_for(a, b)
    if dispatch is None:
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    m, _ = a.shape
    _, n = b.shape
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    elif not isinstance(out, torch.Tensor):
        raise InvalidInput("gfx950 mm output must be a torch.Tensor; "
                           f"got {type(out).__name__}")
    elif out.shape != (m, n):
        raise InvalidInput(f"gfx950 mm output shape must be {(m, n)}; "
                           f"got {tuple(out.shape)}")
    elif out.dtype != a.dtype:
        raise InvalidInput(f"gfx950 mm output dtype must be {a.dtype}; "
                           f"got {out.dtype}")
    elif out.device != a.device:
        raise InvalidInput(f"gfx950 mm output device must be {a.device}; "
                           f"got {out.device}")

    return _launch_dispatch(a, b, out, dispatch)


def mm(a, b, *, out=None, space="heuristic"):
    """Run the trusted gfx950 entry selected after ``tlx.ops.mm`` validation."""
    if space not in ("full", "heuristic"):
        raise InvalidInput(f"unknown gfx950 mm search space: {space}")
    if not a.is_cuda or any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    elif not isinstance(out, torch.Tensor):
        raise InvalidInput("gfx950 mm output must be a torch.Tensor; "
                           f"got {type(out).__name__}")
    elif out.shape != (m, n):
        raise InvalidInput(f"gfx950 mm output shape must be {(m, n)}; "
                           f"got {tuple(out.shape)}")
    elif out.dtype != a.dtype:
        raise InvalidInput(f"gfx950 mm output dtype must be {a.dtype}; "
                           f"got {out.dtype}")
    elif out.device != a.device:
        raise InvalidInput(f"gfx950 mm output device must be {a.device}; "
                           f"got {out.device}")
    dispatch = None
    use_compiled_cache = None
    if space == "full":
        if k >= 128:
            use_compiled_cache = _can_use_range_register_compiled_cache()
            if use_compiled_cache:
                dispatch = _RANGE_TUNED_PLAN_CACHE.get(_range_tuned_plan_cache_key(a, b, out))
        if dispatch is None:
            candidates = _range_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
            if len(candidates) == 1:
                dispatch = candidates[0]
            elif candidates:
                return _launch_range_autotuned(a, b, out, candidates)
            else:
                return _launch_register(a, b, out=out)
    else:
        # Preserve the measured family selector for dense TN inputs.  The broader
        # master heuristic remains the fallback for layouts such as row-major B
        # that are outside the promoted selector's validated domain.
        dispatch = _dispatch_for(a, b)
        if dispatch is None:
            dispatch = heuristic_config(
                m,
                n,
                k,
                a.dtype,
                a.element_size(),
                a.stride(),
                b.stride(),
            )
    if dispatch is None:
        raise InvalidInput("gfx950 mm does not support "
                           f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    # Keep this catalog hot path inline: ``tlx.ops.mm`` already validated the
    # inputs, and another Python call is material for the small-M kernels.
    path, plan = dispatch
    if path == "short_k_register":
        return _launch_short_k_register(a, b, out=out)
    if path == "row_major_direct":
        return _launch_row_major_direct(a, b, out, plan)
    if path == "hybrid_n160":
        return _launch_hybrid_n160(a, b, out)
    if path in ("inter_wave", "streamk", "ragged_n", "m192n256"):
        return _launch_optimized_inter_wave(a, b, out, path, plan)
    if path == "transposed_wave_grid":
        _launch_wave_grid(
            b.T,
            a.T,
            out=out.T,
            _candidate_plan=plan,
        )
        return out
    if path == "wave_grid":
        return _launch_wave_grid(a, b, out=out, _candidate_plan=plan)
    if path == "persistent":
        return _launch_persistent(a, b, out=out)
    if path == "range_register":
        return _launch_range_register_plan(a, b, config=plan, out=out, _use_compiled_cache=use_compiled_cache)
    if path == "register":
        return _launch_register_plan(
            a,
            b,
            config=plan,
            out=out,
            _validated=True,
        )
    if path == "lds":
        block_m, block_n, split_k = plan
        return _launch_lds(
            a,
            b,
            SPLIT_K=split_k,
            TILE=(block_m, block_n),
            out=out,
        )
    return _launch_validated(a, b, out, plan)
