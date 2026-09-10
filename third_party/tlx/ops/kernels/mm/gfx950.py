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


def _plan_for(a, b):
    """Return the measured plan for supported operands, otherwise ``None``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    m, k = a.shape
    _, n = b.shape
    if not (
        a.dtype == torch.float16
        and b.dtype == torch.float16
        and a.is_cuda
        and a.device == b.device
        and _device_arch(a.device) == "gfx950"
        and a.stride(1) == 1
        and b.stride(0) == 1
    ):
        return None
    return _KNOWN_PLANS.get((m, n, k))


def supports(a, b):
    """Return whether a and b select a measured LocalSplitU plan."""
    return _plan_for(a, b) is not None


def _launch_validated(a, b, out, plan):
    """Launch a plan after operand and output validation has completed."""
    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n, local_split_u, wave_k, k_width = plan
    if m > tile_m:
        raise InvalidInput(
            f"gfx950 LocalSplitU plan covers at most {tile_m} rows; got M={m}"
        )
    launch_options = {"sink_insts_to_avoid_spills": True}
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


def matmul(a, b, out=None):
    """Run a tuned gfx950 LocalSplitU GEMM specialization."""
    plan = _plan_for(a, b)
    if plan is None:
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

    return _launch_validated(a, b, out, plan)


def mm(a, b, *, space="heuristic"):
    """Run the gfx950 implementation selected by ``tlx.ops.mm``.

    The initial implementation has one measured plan per supported shape;
    the next stack revision broadens this into a bounded plan generator.
    """
    if space != "heuristic":
        raise InvalidInput(
            "gfx950 LocalSplitU mm currently supports space='heuristic' only"
        )
    plan = _plan_for(a, b)
    if plan is None:
        raise InvalidInput(
            "gfx950 LocalSplitU mm has no legal plan for "
            f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    m, _ = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    return _launch_validated(a, b, out, plan)
