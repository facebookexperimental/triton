"""gfx950 small-M GEMM implementation for :func:`triton.tlx.ops.mm`.

The logical M dimension is too small to expose enough output tiles. Each wave
therefore computes one block-cyclic K partition of the same output tile. The
FP32 partials are reduced in wave order inside the workgroup before the logical
result is stored.
"""

from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput
from ._shapes import GFX950_FOCUS

__all__ = ["mm", "matmul", "supports"]

PERF_SHAPES = GFX950_FOCUS


# Exact entries are a tuning cache, not the supported-shape list. Unseen shapes
# use the bounded plan generator below and benchmark at most four candidates.
class _Plan(NamedTuple):
    tile_m: int
    tile_n: int
    local_split_u: int
    wave_k: int
    k_width: int


_KNOWN_PLANS = {
    # Sixteen short K32 chains maximize wave-level latency hiding. N32 keeps
    # one workgroup per CU and gives every A load two independent MFMA users.
    (7, 8192, 2048): _Plan(
        tile_m=16, tile_n=32, local_split_u=16, wave_k=32, k_width=8
    ),
    # N16 gives two output tiles per CU. Four K256 partitions provide enough
    # wave-level latency hiding without the long-lived K1024 operands.
    (7, 2048, 4096): _Plan(
        tile_m=16, tile_n=16, local_split_u=4, wave_k=256,
        k_width=8,
    ),
}

@lru_cache(maxsize=None)
def _device_arch(device):
    """Return the AMD architecture for a CUDA device, or an empty string."""
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


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
    a_offsets = (
        global_rows[None, :, None] * stride_am
        + (split_k + rk[None, None, :]) * stride_ak
    )
    b_offsets = (
        (split_k + rk[None, :, None]) * stride_bk
        + global_cols[None, None, :] * stride_bn
    )
    # The offsets already have their dot-operand layouts, so the loaded values
    # reach MFMA registers without an intervening conversion through LDS.
    # K_WIDTH is also the largest contiguous run owned by one lane; claiming a
    # wider buffer vector would cross the lane's two disjoint K runs.
    a_offsets = tlx.require_layout(a_offsets, dot_a)
    b_offsets = tlx.require_layout(b_offsets, dot_b)
    a = tlx.buffer_load(a_ptr, a_offsets, contiguity=K_WIDTH)
    b = tlx.buffer_load(
        b_ptr, b_offsets, cache=".cg", contiguity=K_WIDTH
    )
    return a, b


_NUM_CU = 256
_TILE_N_CANDIDATES = (8, 16, 32)
_SPLIT_U_CANDIDATES = (2, 4, 8, 16)
_WAVE_K_CANDIDATES = (32, 64, 128, 256, 512, 1024)
_TARGET_SPLIT_U = {8: 2, 16: 8, 32: 16}
_TARGET_MACRO_K = {8: 2048, 16: 1024, 32: 512}
_MAX_AUTOTUNE_CONFIGS = 4


def _pow2_distance(lhs, rhs):
    return abs(lhs.bit_length() - rhs.bit_length())


@lru_cache(maxsize=128)
def _generate_plans(m, n, k):
    """Generate at most four legal, high-value LocalSplitU plans.

    First choose output widths that put the N grid near one workgroup per CU.
    Within each width, prefer 2--4 macro-K iterations and stay near the measured
    split/reduction balance for that width. The final choice is measured by a
    bounded host-side tuner rather than hard-coded by the cost model.
    """
    if not (0 < m <= 16 and n >= 512 and k >= 1024):
        return ()

    tile_ns = sorted(
        _TILE_N_CANDIDATES,
        key=lambda tile_n: (
            abs(triton.cdiv(n, tile_n) - _NUM_CU),
            tile_n,
        ),
    )[:2]
    closest_grid = triton.cdiv(n, tile_ns[0])
    if not _NUM_CU // 2 <= closest_grid <= 2 * _NUM_CU:
        return ()
    plans = []
    for tile_n in tile_ns:
        legal = []
        for split_u in _SPLIT_U_CANDIDATES:
            for wave_k in _WAVE_K_CANDIDATES:
                macro_k = split_u * wave_k
                if k % macro_k != 0:
                    continue
                macro_iterations = k // macro_k
                if not 2 <= macro_iterations <= 8:
                    continue
                score = (
                    _pow2_distance(macro_k, _TARGET_MACRO_K[tile_n]),
                    _pow2_distance(split_u, _TARGET_SPLIT_U[tile_n]),
                    min(
                        abs(macro_iterations - 2),
                        abs(macro_iterations - 4),
                    ),
                    -wave_k,
                )
                legal.append((score, _Plan(16, tile_n, split_u, wave_k, 4)))
        legal.sort()
        plans.extend(spec for _, spec in legal[:2])
    return tuple(plans[:_MAX_AUTOTUNE_CONFIGS])


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
    # Generated K1024 plans benefit from the shorter clamp instruction; the
    # measured exact plans preserve their row-zero duplication and load map.
    if WAVE_K == 1024:
        global_rows = tl.minimum(rows, M - 1)
    else:
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
    dot_a: tl.constexpr = tlx.dot_operand_layout(
        0, mma, k_width=K_WIDTH
    )
    dot_b: tl.constexpr = tlx.dot_operand_layout(
        1, mma, k_width=K_WIDTH
    )

    tl.static_assert(K % MACRO_K == 0)
    acc = tlx.zeros(
        (LOCAL_SPLIT_U, TILE_M, TILE_N),
        tl.float32,
        layout=mma,
    )

    current_a, current_b = _load_dot_operands(
        a_ptr, b_ptr, global_rows, global_cols, split_ids, rk, 0,
        stride_am, stride_ak, stride_bk, stride_bn,
        WAVE_K, K_WIDTH, dot_a, dot_b,
    )

    # One-stage register pipeline: issue K(t+1)'s global loads before K(t)'s
    # dot. The chosen WAVE_K controls the prefetch lifetime and register cost.
    for macro in tl.range(0, K // MACRO_K - 1, num_stages=1):
        next_k = (macro + 1) * MACRO_K
        next_a, next_b = _load_dot_operands(
            a_ptr, b_ptr, global_rows, global_cols, split_ids, rk, next_k,
            stride_am, stride_ak, stride_bk, stride_bn,
            WAVE_K, K_WIDTH, dot_a, dot_b,
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
        partial_layout: tl.constexpr = tlx.swizzled_layout(
            2, 2, 3, order=[2, 1, 0]
        )
        partial_buffer = tlx.local_alloc(
            (LOCAL_SPLIT_U, TILE_M, TILE_N),
            tl.float32,
            1,
            layout=partial_layout,
        )
    else:
        partial_buffer = tlx.local_alloc(
            (LOCAL_SPLIT_U, TILE_M, TILE_N), tl.float32, 1
        )
    partial_view = tlx.local_view(partial_buffer, 0)
    tlx.local_store(partial_view, acc)
    tl.debug_barrier()

    tl.static_assert(
        LOCAL_SPLIT_U == 2
        or LOCAL_SPLIT_U == 4
        or LOCAL_SPLIT_U == 8
        or LOCAL_SPLIT_U == 16
    )
    result = tl.reshape(
        tlx.local_load(
            tlx.local_slice(
                partial_view,
                [0, 0, 0],
                [1, TILE_M, TILE_N],
            )
        ),
        (TILE_M, TILE_N),
    )
    # Load and immediately consume each subsequent partial. This preserves
    # increasing split-id association without keeping every partial live.
    for split in tl.static_range(1, LOCAL_SPLIT_U):
        partial = tlx.local_load(
            tlx.local_slice(
                partial_view,
                [split, 0, 0],
                [1, TILE_M, TILE_N],
            )
        )
        result += tl.reshape(partial, (TILE_M, TILE_N))

    output_rows = tl.arange(0, TILE_M).to(tl.int32)
    output_ptrs = (
        c_ptr
        + output_rows[:, None] * stride_cm
        + output_cols[None, :] * stride_cn
    )
    tl.store(
        output_ptrs,
        result.to(c_ptr.dtype.element_ty),
        mask=(output_rows[:, None] < M)
        & (output_cols[None, :] < N),
    )


_PLAN_CACHE = {}


# This helper is used only during first-call tuning. Keep the steady-state
# launch inline in matmul(): another Python frame is measurable for a ~10 us
# kernel because host dispatch falls between the timing events.
def _launch_plan_for_tuning(a, b, out, plan):
    tile_m, tile_n, local_split_u, wave_k, k_width = plan
    launch_options = {"sink_insts_to_avoid_spills": True}
    if wave_k >= 512:
        # Iterative ILP controls register pressure for long operand windows;
        # keep the later generic high-RP pass from replacing its schedule.
        launch_options.update(
            llvm_fn_attrs=(
                ("amdgpu-sched-strategy", "iterative-ilp"),
            ),
            disable_unclustered_high_rp_reschedule=True,
        )
    m, k = a.shape
    _, n = b.shape
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
        **launch_options,
    )


def _plan_cache_key(a, b, out):
    return (
        a.device.type,
        a.device.index,
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        tuple(out.stride()),
    )


def _select_plan(a, b, out):
    m, k = a.shape
    _, n = b.shape
    known = _KNOWN_PLANS.get((m, n, k))
    if known is not None:
        return known

    key = _plan_cache_key(a, b, out)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = _generate_plans(m, n, k)
    assert candidates
    timings = [
        (
            triton.testing.do_bench(
                lambda plan=plan: _launch_plan_for_tuning(a, b, out, plan),
                warmup=25,
                rep=100,
            ),
            plan,
        )
        for plan in candidates
    ]
    selected = min(timings, key=lambda item: item[0])[1]
    _PLAN_CACHE[key] = selected
    return selected


def supports(a, b):
    """Return whether a and b fit the aligned small-M LocalSplitU family."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return False
    m, k = a.shape
    _, n = b.shape
    if (
        a.dtype != torch.float16
        or b.dtype != torch.float16
        or not a.is_cuda
        or a.device != b.device
        or _device_arch(a.device) != "gfx950"
        or a.stride(1) != 1
        or b.stride(0) != 1
    ):
        return False
    if (m, n, k) in _KNOWN_PLANS:
        return True
    return bool(_generate_plans(m, n, k))


def matmul(a, b, out=None):
    """Run a tuned gfx950 LocalSplitU GEMM specialization."""
    if not supports(a, b):
        raise InvalidInput(
            "gfx950 LocalSplitU matmul does not support "
            f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    m, k = a.shape
    _, n = b.shape
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    elif not isinstance(out, torch.Tensor):
        raise InvalidInput(
            "gfx950 LocalSplitU output must be a torch.Tensor; "
            f"got {type(out).__name__}"
        )
    elif out.shape != (m, n):
        raise InvalidInput(
            f"gfx950 LocalSplitU output shape must be {(m, n)}; "
            f"got {tuple(out.shape)}"
        )
    elif out.dtype != a.dtype:
        raise InvalidInput(
            f"gfx950 LocalSplitU output dtype must be {a.dtype}; "
            f"got {out.dtype}"
        )
    elif out.device != a.device:
        raise InvalidInput(
            f"gfx950 LocalSplitU output device must be {a.device}; "
            f"got {out.device}"
        )

    plan = _select_plan(a, b, out)
    tile_m, tile_n, local_split_u, wave_k, k_width = plan
    if m > tile_m:
        raise InvalidInput(
            f"gfx950 LocalSplitU plan covers at most {tile_m} rows; got M={m}"
        )
    launch_options = {"sink_insts_to_avoid_spills": True}
    if wave_k >= 512:
        # Generated long-window plans require the same register-pressure
        # controls used while timing their candidates.
        launch_options.update(
            llvm_fn_attrs=(
                ("amdgpu-sched-strategy", "iterative-ilp"),
            ),
            disable_unclustered_high_rp_reschedule=True,
        )
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
        **launch_options,
    )
    return out


def mm(a, b, *, space="heuristic"):
    """Run the gfx950 implementation selected by ``tlx.ops.mm``.

    The initial implementation has one measured plan per supported shape;
    the next stack revision broadens this into a bounded plan generator.
    """
    if space != "heuristic":
        raise InvalidInput(
            "gfx950 LocalSplitU mm currently supports space='heuristic' only"
        )
    return matmul(a, b)
