"""Grouped GEMM kernels for AMD/gfx1250.

This file starts with the correctness-oriented pointer-table baseline from
``third_party/tlx/doc/gfx1250_grouped_gemm.md``.  It supports a fully ragged
group of FP16 GEMMs in one persistent launch:

    C_i = A_i @ B_i

Each group may have different M/N/K.  A, B, and C are expected to be row-major
with inner stride 1.  The kernel masks M, N, and K tails and uses ordinary
global loads/stores, not gfx1250 TDM.  The optimized TDM packed-ragged-M path
should use this kernel as a reference.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip_gfx1250


def _active_num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _grouped_gemm_tflops(ms: float, m_list: list[int], n: int, k: int) -> float:
    total_flops = sum(2 * m * n * k for m in m_list)
    return total_flops / (ms * 1e-3) / 1e12


_XCD_REMAP_MODES = {
    "none": 0,
    "balanced": 1,
    "chunked": 2,
}

# Relative saturated rates for the validated gfx1250 TDM seeds. The large
# 256x256 tile is the reference; asymmetric rates reflect matched full-tile
# measurements in the local profiling notes.
_GROUPED_GEMM_CONFIGS = (
    dict(name="256x256-alias", block_m=256, block_n=256, block_k=128, group_m=4, tdm_pipeline_depth=2, c_staging_mode=0,
         cross_tile_prefetch=False, relative_rate=1.0),
    dict(name="128x256-dedicated", block_m=128, block_n=256, block_k=128, group_m=4, tdm_pipeline_depth=2,
         c_staging_mode=1, cross_tile_prefetch=True, relative_rate=0.995),
    dict(name="256x128-dedicated", block_m=256, block_n=128, block_k=128, group_m=4, tdm_pipeline_depth=2,
         c_staging_mode=1, cross_tile_prefetch=True, relative_rate=0.98),
    dict(name="128x128-alias", block_m=128, block_n=128, block_k=128, group_m=4, tdm_pipeline_depth=2, c_staging_mode=0,
         cross_tile_prefetch=False, relative_rate=0.90),
)


def _cdiv_int(a: int, b: int) -> int:
    return -(-a // b)


def _rank_grouped_gemm_configs(m_list: list[int], n: int, k: int, num_sms: int):
    num_sms = max(1, int(num_sms))
    useful = sum(m * n * k for m in m_list)
    ranked = []
    for seed in _GROUPED_GEMM_CONFIGS:
        cfg = dict(seed)
        bm = int(cfg["block_m"])
        bn = int(cfg["block_n"])
        bk = int(cfg["block_k"])
        if n % bn != 0 or k % bk != 0 or any(m % bm != 0 for m in m_list):
            continue

        group_tiles = [(m // bm) * (n // bn) for m in m_list]
        total_tiles = sum(group_tiles)
        if total_tiles == 0:
            continue
        launch_programs = min(num_sms, total_tiles)
        slots = _cdiv_int(total_tiles, num_sms) * num_sms
        utilization = total_tiles / slots
        padded = sum(_cdiv_int(m, bm) * bm * _cdiv_int(n, bn) * bn * k for m in m_list)
        padding_efficiency = useful / padded

        cross_tile_prefetch = bool(cfg["cross_tile_prefetch"])
        cross_tile_prefetch &= (k // bk) % int(cfg["tdm_pipeline_depth"]) == 0
        cross_tile_prefetch &= any(tiles > launch_programs for tiles in group_tiles)
        if cfg["cross_tile_prefetch"] and not cross_tile_prefetch:
            # The measured asymmetric rate includes the cross-tile overlap.
            cfg["relative_rate"] = float(cfg["relative_rate"]) / 1.02
        cfg["cross_tile_prefetch"] = cross_tile_prefetch

        score = float(cfg["relative_rate"]) * utilization * padding_efficiency
        intensity = bm * bn / (bm + bn)
        ranked.append((score, intensity, utilization, cfg))
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return ranked


def _pick_grouped_gemm_config(m_list: list[int], n: int, k: int, num_sms: int):
    ranked = _rank_grouped_gemm_configs(m_list, n, k, num_sms)
    if not ranked:
        raise ValueError("no validated gfx1250 grouped GEMM config supports these full-tile shapes")
    top_score = ranked[0][0]
    near_top = [entry for entry in ranked if entry[0] >= 0.9 * top_score]
    return dict(max(near_top, key=lambda entry: (entry[1], entry[2]))[3])


def _remap_program_id_reference(pid: int, num_programs: int, mode: str, num_xcds: int, chunk_size: int) -> int:
    if mode == "none" or num_xcds == 1:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if mode == "balanced":
        min_per_xcd = num_programs // num_xcds
        extra = num_programs % num_xcds
        return xcd * min_per_xcd + min(xcd, extra) + local_pid
    aligned = (num_programs // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def grouped_gemm_phase0_kernel(
    # Device tensors of per-group matrix base pointers.
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # Flattened [group_size, 3] metadata: [M, N, K] per group.
    group_gemm_sizes,
    # Flattened [group_size, 3] metadata: [lda, ldb, ldc] per group.
    group_lds,
    group_size,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3 + 0)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            lda = tl.load(group_lds + g * 3 + 0)
            ldb = tl.load(group_lds + g * 3 + 1)
            ldc = tl.load(group_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

            row_mask = offs_m[:, None] < gm
            col_mask = offs_n[None, :] < gn
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                k_remaining = gk - kk * BLOCK_K
                k_mask = offs_k < k_remaining

                a = tl.load(a_ptrs, mask=row_mask & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask, other=0.0)
                acc += tl.dot(a, b)

                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb

            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
            tl.store(c_ptrs, c, mask=row_mask & col_mask)

            tile_idx += NUM_PROGRAMS

        last_problem_end += num_tiles


def grouped_gemm_phase0(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    group_c: Optional[list[torch.Tensor]] = None,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    num_programs: Optional[int] = None,
) -> list[torch.Tensor]:
    """Compute a fully ragged group of FP16 row-major GEMMs.

    ``group_a[i]`` has shape ``[M_i, K_i]`` and ``group_b[i]`` has shape
    ``[K_i, N_i]``.  The returned ``group_c[i]`` has shape ``[M_i, N_i]``.
    """
    group_size = len(group_a)
    assert len(group_b) == group_size
    if group_c is not None:
        assert len(group_c) == group_size
    if group_size == 0:
        return []

    device = group_a[0].device
    dtype = group_a[0].dtype
    assert dtype == torch.float16, "baseline currently supports FP16"

    a_addrs: list[int] = []
    b_addrs: list[int] = []
    c_addrs: list[int] = []
    sizes: list[int] = []
    lds: list[int] = []
    out: list[torch.Tensor] = []
    total_tiles = 0

    for i, (a, b) in enumerate(zip(group_a, group_b)):
        assert a.device == device and b.device == device, "all inputs must be on the same device"
        assert a.dtype == dtype and b.dtype == dtype, "all inputs must have the same dtype"
        assert a.dim() == 2 and b.dim() == 2
        assert a.shape[1] == b.shape[0], f"K mismatch in group {i}: {tuple(a.shape)} x {tuple(b.shape)}"
        assert a.stride(1) == 1 and b.stride(1) == 1, "A and B must have inner stride 1"

        m, k = a.shape
        _, n = b.shape
        c = group_c[i] if group_c is not None else torch.empty((m, n), device=device, dtype=dtype)
        assert c.device == device and c.dtype == dtype
        assert c.shape == (m, n)
        assert c.stride(1) == 1, "C must have inner stride 1"

        a_addrs.append(a.data_ptr())
        b_addrs.append(b.data_ptr())
        c_addrs.append(c.data_ptr())
        sizes += [m, n, k]
        lds += [a.stride(0), b.stride(0), c.stride(0)]
        out.append(c)
        total_tiles += triton.cdiv(m, block_m) * triton.cdiv(n, block_n)

    if total_tiles == 0:
        return out

    if num_programs is None:
        num_programs = _active_num_cus(device)
    num_programs = max(1, min(int(num_programs), total_tiles))

    d_a_ptrs = torch.tensor(a_addrs, device=device, dtype=torch.int64)
    d_b_ptrs = torch.tensor(b_addrs, device=device, dtype=torch.int64)
    d_c_ptrs = torch.tensor(c_addrs, device=device, dtype=torch.int64)
    d_sizes = torch.tensor(sizes, device=device, dtype=torch.int32)
    d_lds = torch.tensor(lds, device=device, dtype=torch.int32)

    grouped_gemm_phase0_kernel[(num_programs, )](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_sizes,
        d_lds,
        group_size,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    return out


@triton.jit
def _tdm_load_subtile(
    a_buf,
    b_buf,
    consumer,
    start: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
):
    slot = consumer % NUM_BUFFERS
    a_view = tlx.local_slice(a_buf[slot], [0, start], [BLOCK_M, SUBTILE_LEN])
    b_view = tlx.local_slice(b_buf[slot], [0, start], [BLOCK_N, SUBTILE_LEN])
    return tlx.local_load(a_view), tlx.local_load(tlx.local_trans(b_view))


@triton.jit
def _tdm_issue_loads(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    pred,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    slot = producer % NUM_BUFFERS
    tlx.async_amd_descriptor_load_group(
        [a_desc, b_desc],
        [a_buf[slot], b_buf[slot]],
        [[off_m, producer * BLOCK_K], [off_n, producer * BLOCK_K]],
        [0b0011, 0b1100],
        preds=[pred, pred],
    )
    return producer + 1


@triton.jit
def _tdm_issue_loads_unpredicated(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    slot = producer % NUM_BUFFERS
    tlx.async_amd_descriptor_load_group(
        [a_desc, b_desc],
        [a_buf[slot], b_buf[slot]],
        [[off_m, producer * BLOCK_K], [off_n, producer * BLOCK_K]],
        [0b0011, 0b1100],
    )
    return producer + 1


@triton.jit
def _tdm_prefetch_unpredicated(a_desc, b_desc, prefetch_iter, off_m, off_n, BLOCK_K: tl.constexpr):
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_iter * BLOCK_K])
    tlx.amd_descriptor_prefetch_tensor(b_desc, [off_n, prefetch_iter * BLOCK_K])


@triton.jit
def _tdm_accumulate_subtiles(
    acc,
    a0,
    b0,
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    consumer,
    producer,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
):
    a1, b1 = _tdm_load_subtile(a_buf, b_buf, consumer, SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a0, b0, acc)

    if L2_PREFETCH_DISTANCE > 0:
        prefetch_iter = producer + L2_PREFETCH_DISTANCE - 1
        _tdm_prefetch_unpredicated(a_desc, b_desc, prefetch_iter, 0, 0, BLOCK_K)

    a2, b2 = _tdm_load_subtile(a_buf, b_buf, consumer, 2 * SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a1, b1, acc)

    a3, b3 = _tdm_load_subtile(a_buf, b_buf, consumer, 3 * SUBTILE_LEN, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a2, b2, acc)
    return acc, a3, b3


@triton.jit
def _tdm_wait_and_finish_k_block(
    acc,
    a3,
    b3,
    a_buf,
    b_buf,
    consumer,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
):
    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)
    a0, b0 = _tdm_load_subtile(a_buf, b_buf, consumer, 0, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
    acc = tl.dot(a3, b3, acc)
    return acc, a0, b0


@triton.jit
def _grouped_tile_offsets(
    tile_idx,
    last_problem_end,
    num_m_tiles,
    num_n_tiles,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_idx_in_gemm = tile_idx - last_problem_end
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = tile_idx_in_gemm // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_m_tiles - first_pid_m, GROUP_M)
    tile_m_idx = first_pid_m + ((tile_idx_in_gemm % num_pid_in_group) % group_size_m)
    tile_n_idx = (tile_idx_in_gemm % num_pid_in_group) // group_size_m
    return tile_m_idx * BLOCK_M, tile_n_idx * BLOCK_N


@triton.jit
def _remap_grouped_program_id(
    pid,
    num_programs,
    XCD_REMAP_MODE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    if XCD_REMAP_MODE == 0 or NUM_XCDS == 1:
        return pid
    elif XCD_REMAP_MODE == 1:
        # Physical workgroups are assigned round-robin across XCDs. Map each
        # XCD's local sequence onto one balanced contiguous logical range.
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        min_per_xcd = num_programs // NUM_XCDS
        extra = num_programs % NUM_XCDS
        return xcd * min_per_xcd + min(xcd, extra) + local_pid
    else:
        tl.static_assert(XCD_REMAP_MODE == 2, "XCD_REMAP_MODE must be none, balanced, or chunked")
        tl.static_assert(XCD_CHUNK >= 1, "XCD_CHUNK must be positive")
        aligned = (num_programs // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
        if pid >= aligned:
            return pid
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        return ((local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + (local_pid % XCD_CHUNK))


@triton.jit
def grouped_gemm_tdm_kernel(
    a_packed,
    b_t,
    c_packed,
    group_offsets,
    group_size,
    N,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    C_STAGING_MODE: tl.constexpr,
    CROSS_TILE_PREFETCH: tl.constexpr,
    XCD_REMAP_MODE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    """Packed ragged-M grouped GEMM using gfx1250 TDM descriptor loads/stores.

    ``b_t`` is [G, N, K], so each B tile is loaded as [BLOCK_N, BLOCK_K] and
    locally transposed before ``tl.dot``.
    """
    tl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    tl.static_assert(C_STAGING_MODE == 0 or C_STAGING_MODE == 1, "C_STAGING_MODE must be 0 or 1")
    tl.static_assert(not CROSS_TILE_PREFETCH or C_STAGING_MODE == 1, "cross-tile prefetch requires dedicated C staging")
    tl.static_assert(not CROSS_TILE_PREFETCH or NUM_BUFFERS == 2, "cross-tile prefetch currently requires depth 2")
    tl.static_assert(NUM_XCDS >= 1, "NUM_XCDS must be positive")
    NUM_SUBTILES: tl.constexpr = 4
    SUBTILE_LEN: tl.constexpr = BLOCK_K // NUM_SUBTILES
    K_ITERS: tl.constexpr = K // BLOCK_K
    USE_FULL_C_TILE: tl.constexpr = (NUM_BUFFERS * BLOCK_K) % BLOCK_N == 0
    tl.static_assert(K % BLOCK_K == 0, "K must be divisible by BLOCK_K")
    tl.static_assert(K_ITERS >= NUM_BUFFERS, "K must cover the TDM pipeline")
    tl.static_assert(SUBTILE_LEN == 32, "BLOCK_K must be 128 for the first TDM schedule")

    pid = _remap_grouped_program_id(tl.program_id(0), NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK)
    tile_idx = pid
    last_problem_end = 0
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_packed), NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_t), NUM_BUFFERS)
    # TLXRewriteLocalAlias only accepts size-mismatched shared aliases when
    # the backing allocation is an integer multiple of the alias.  Reuse A only
    # when that ratio is legal.  Otherwise store C in BLOCK_K-wide chunks that
    # are each small enough to alias A.  Reusing B can satisfy the size check
    # for asymmetric tiles, but B's TDM-load/dot-transpose layout constraints
    # conflict with C's TDM-store layout constraints.
    if C_STAGING_MODE == 1:
        tl.static_assert(USE_FULL_C_TILE, "dedicated C staging currently requires a full C buffer")
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_packed), 1)
    elif USE_FULL_C_TILE:
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_packed), 1, reuse=a_buf)
    else:
        tl.static_assert(BLOCK_N == 2 * BLOCK_K, "split C store currently supports BLOCK_N == 2 * BLOCK_K")
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(c_packed), 1, reuse=a_buf)

    for g in range(group_size):
        m_start = tl.load(group_offsets + g)
        m_end = tl.load(group_offsets + g + 1)
        gm = m_end - m_start

        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        group_base_m = m_start.to(tl.int64) * stride_am
        group_base_c = m_start.to(tl.int64) * stride_cm
        group_base_b = g.to(tl.int64) * stride_bg

        a_desc_base = tl.make_tensor_descriptor(
            a_packed + group_base_m,
            # The launcher validates exact BLOCK_K coverage. Keeping the K
            # extent static avoids rebuilding dynamic bounds at every TDM issue.
            shape=[gm, K],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc_base = tl.make_tensor_descriptor(
            b_t + group_base_b,
            shape=[N, K],
            strides=[stride_bn, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        if USE_FULL_C_TILE:
            c_desc_base = tl.make_tensor_descriptor(
                c_packed + group_base_c,
                shape=[gm, N],
                strides=[stride_cm, tl.constexpr(1)],
                block_shape=[BLOCK_M, BLOCK_N],
            )
        else:
            c_desc_base = tl.make_tensor_descriptor(
                c_packed + group_base_c,
                shape=[gm, N],
                strides=[stride_cm, tl.constexpr(1)],
                # Match the BLOCK_K-wide LDS alias above: when a full C tile
                # cannot legally alias the A ring, store its two halves. This
                # is an aliasing constraint, not a store-efficiency choice.
                block_shape=[BLOCK_M, BLOCK_K],
            )

        if CROSS_TILE_PREFETCH:
            # Prime only the first tile this persistent program handles in the
            # group. Every later tile receives K0/K1 from its predecessor's
            # peeled tail, so no per-tile predicated prologue is required.
            if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                first_off_m, first_off_n = _grouped_tile_offsets(tile_idx, last_problem_end, num_m_tiles, num_n_tiles,
                                                                 GROUP_M, BLOCK_M, BLOCK_N)
                _tdm_issue_loads_unpredicated(
                    a_desc_base,
                    b_desc_base,
                    a_buf,
                    b_buf,
                    0,
                    first_off_m,
                    first_off_n,
                    BLOCK_K,
                    NUM_BUFFERS,
                )
                # Build tile-adjusted descriptors after K0 is issued, then use
                # them for K1 so descriptor setup overlaps K0 movement.
                first_a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[first_off_m, 0])
                first_b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[first_off_n, 0])
                _tdm_issue_loads_unpredicated(
                    first_a_desc,
                    first_b_desc,
                    a_buf,
                    b_buf,
                    1,
                    0,
                    0,
                    BLOCK_K,
                    NUM_BUFFERS,
                )
                tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            off_m, off_n = _grouped_tile_offsets(tile_idx, last_problem_end, num_m_tiles, num_n_tiles, GROUP_M, BLOCK_M,
                                                 BLOCK_N)
            a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[off_m, 0])
            b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[off_n, 0])

            producer = 0
            consumer = 0

            if C_STAGING_MODE == 0:
                # The previous tile's async C store may still be reading from
                # c_buf, which aliases a_buf. Delay the drain until the next
                # actual LDS overwrite hazard instead of waiting immediately
                # after the store.
                tlx.async_amd_descriptor_wait(0)

            if not CROSS_TILE_PREFETCH:
                # Prime all ring slots up front so the first consumer load only
                # waits until the oldest TDM op is complete, leaving newer
                # slots in flight.
                for _ in tl.static_range(NUM_BUFFERS):
                    producer = _tdm_issue_loads_unpredicated(
                        a_desc,
                        b_desc,
                        a_buf,
                        b_buf,
                        producer,
                        0,
                        0,
                        BLOCK_K,
                        NUM_BUFFERS,
                    )
                # With dedicated C staging, the previous C store is older than
                # these input groups. This count-based wait retires that store
                # and the oldest input group before the first LDS read.
                tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)
            a0, b0 = _tdm_load_subtile(a_buf, b_buf, consumer, 0, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)

            if CROSS_TILE_PREFETCH:
                next_tile_idx = tile_idx + NUM_PROGRAMS
                group_end = last_problem_end + num_tiles
                has_next = next_tile_idx < group_end
                safe_next_tile_idx = min(next_tile_idx, group_end - 1)
                next_off_m, next_off_n = _grouped_tile_offsets(safe_next_tile_idx, last_problem_end, num_m_tiles,
                                                               num_n_tiles, GROUP_M, BLOCK_M, BLOCK_N)
                next_a_desc = tlx.update_tensor_descriptor(a_desc_base, add_offsets=[next_off_m, 0])
                next_b_desc = tlx.update_tensor_descriptor(b_desc_base, add_offsets=[next_off_n, 0])

                steady_end = K_ITERS - NUM_BUFFERS
                # Producer and consumer are affine functions of the canonical
                # loop IV; deriving them avoids two scalar loop-carried values.
                # Unroll explicit ring-sized chunks instead of using
                # loop_unroll_factor: the generic unroller emits a full scalar
                # remainder loop. The cross-tile host contract makes
                # steady_end divisible by NUM_BUFFERS.
                chunk_end = steady_end // NUM_BUFFERS
                for chunk in tl.range(0, chunk_end):
                    for phase in tl.static_range(NUM_BUFFERS):
                        i = chunk * NUM_BUFFERS + phase
                        producer_i = i + NUM_BUFFERS
                        acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, i, producer_i,
                                                               L2_PREFETCH_DISTANCE, BLOCK_K, NUM_BUFFERS, BLOCK_M,
                                                               BLOCK_N, SUBTILE_LEN)
                        _tdm_issue_loads_unpredicated(
                            a_desc,
                            b_desc,
                            a_buf,
                            b_buf,
                            producer_i,
                            0,
                            0,
                            BLOCK_K,
                            NUM_BUFFERS,
                        )
                        acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, i + 1, NUM_BUFFERS,
                                                                   BLOCK_M, BLOCK_N, SUBTILE_LEN)

                # Peel one full ring rotation. Once a current-tile slot has
                # been read into registers, recycle it for the matching K
                # block of the next tile handled by this persistent program.
                # Prefer the late unroll hint over static_range. Early
                # expansion lets LICM hoist both iterations' LDS view setup
                # across the group loop, increasing C-store barrier skew.
                for j in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
                    consumer_j = steady_end + j
                    acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, consumer_j, j, 0,
                                                           BLOCK_K, NUM_BUFFERS, BLOCK_M, BLOCK_N, SUBTILE_LEN)
                    _tdm_issue_loads(
                        next_a_desc,
                        next_b_desc,
                        a_buf,
                        b_buf,
                        j,
                        0,
                        0,
                        has_next,
                        BLOCK_K,
                        NUM_BUFFERS,
                    )
                    acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, consumer_j + 1, NUM_BUFFERS,
                                                               BLOCK_M, BLOCK_N, SUBTILE_LEN)
            else:
                for i in tl.range(0, K_ITERS):
                    acc, a3, b3 = _tdm_accumulate_subtiles(acc, a0, b0, a_desc, b_desc, a_buf, b_buf, consumer,
                                                           producer, L2_PREFETCH_DISTANCE, BLOCK_K, NUM_BUFFERS,
                                                           BLOCK_M, BLOCK_N, SUBTILE_LEN)
                    consumer += 1
                    pred = (i + 1) - epilogue_lb
                    pred = (pred >> 31) & 1
                    producer = _tdm_issue_loads(
                        a_desc,
                        b_desc,
                        a_buf,
                        b_buf,
                        producer,
                        0,
                        0,
                        pred,
                        BLOCK_K,
                        NUM_BUFFERS,
                    )
                    # Issue first so the next producer group can overlap the
                    # final WMMA, then wait until the older consumer group is
                    # complete.
                    acc, a0, b0 = _tdm_wait_and_finish_k_block(acc, a3, b3, a_buf, b_buf, consumer, NUM_BUFFERS,
                                                               BLOCK_M, BLOCK_N, SUBTILE_LEN)

            c_view = c_buf[0]
            c = acc.to(tlx.dtype_of(c_packed))
            # The descriptor update is pure, so LICM otherwise hoists it into
            # the tile preheader and lengthens its live range through the K
            # loop. Keep it in the epilogue to avoid an extra gfx1250 VGPR-MSB
            # mode transition in the steady body.
            tlx.amd_sched_barrier()
            c_desc = tlx.update_tensor_descriptor(c_desc_base, add_offsets=[off_m, off_n])
            tlx.amd_sched_barrier()
            if USE_FULL_C_TILE:
                tlx.local_store(c_view, c)
                tlx.async_amd_descriptor_store(c_desc, c_view, [0, 0])
            else:
                c_pair = tl.reshape(c, (BLOCK_M, 2, BLOCK_K)).permute(0, 2, 1)
                c0, c1 = tl.split(c_pair)
                tlx.local_store(c_view, c0)
                tlx.async_amd_descriptor_store(c_desc, c_view, [0, 0])
                # c1 reuses the same LDS view immediately, so this wait is
                # still required; the second-half store is drained lazily.
                tlx.async_amd_descriptor_wait(0)
                tlx.local_store(c_view, c1)
                tlx.async_amd_descriptor_store(c_desc, c_view, [0, BLOCK_K])

            tile_idx += NUM_PROGRAMS

        last_problem_end += num_tiles

    # At kernel exit, rely on s_endpgm to drain the final async TDM
    # local-to-global store. Keep the explicit wait here as a marker in case
    # target semantics require restoring it.
    # tlx.async_amd_descriptor_wait(0)


def grouped_gemm_tdm(
    a_packed: torch.Tensor,
    b_t: torch.Tensor,
    group_offsets: torch.Tensor,
    c_packed: Optional[torch.Tensor] = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    group_m: int = 4,
    tdm_pipeline_depth: int = 2,
    l2_prefetch_distance: int = 0,
    num_programs: Optional[int] = None,
    benchmark: Optional[str] = None,
    benchmark_num_iters: int = 32,
    c_staging_mode: int = 0,
    cross_tile_prefetch: bool = False,
    auto_config: bool = False,
    xcd_remap_mode: str = "none",
    num_xcds: int = 8,
    xcd_chunk: int = 2,
) -> torch.Tensor:
    """Compute packed ragged-M grouped GEMM with gfx1250 TDM.

    ``a_packed`` is ``[sum(M_g), K]`` and ``b_t`` is ``[G, N, K]``.  The
    current TDM path is full-tile only: every ``M_g`` must be divisible by
    ``block_m``, and ``N``/``K`` must be divisible by ``block_n``/``block_k``.
    The default 128x128x128 shape follows the larger f16 gfx1250 GEMM tile
    family and exposes enough static WMMAs to make scheduler gaps visible.
    ``tdm_pipeline_depth`` maps to the TDM LDS ring-buffer depth; the f16 and
    MXFP gfx1250 kernels use the same 2/3/4-buffer tuning space.
    ``cross_tile_prefetch`` is the dedicated-C, depth-2 schedule that recycles
    the final two K-loop slots for the next tile in the same group.
    ``auto_config`` scores the validated tile seeds using saturated rate, CU
    utilization, and padding efficiency.
    """
    assert a_packed.dtype == torch.float16 and b_t.dtype == torch.float16
    assert a_packed.device == b_t.device == group_offsets.device
    assert a_packed.dim() == 2 and b_t.dim() == 3
    assert a_packed.stride(1) == 1 and b_t.stride(2) == 1
    assert group_offsets.dtype == torch.int32
    assert xcd_remap_mode in _XCD_REMAP_MODES
    assert num_xcds >= 1 and xcd_chunk >= 1

    group_size, n, k = b_t.shape
    total_m, a_k = a_packed.shape
    assert a_k == k
    assert group_offsets.numel() == group_size + 1

    offsets_host = group_offsets.detach().cpu().tolist()
    assert offsets_host[0] == 0 and offsets_host[-1] == total_m
    m_list = [end - start for start, end in zip(offsets_host, offsets_host[1:])]

    model_sms = max(1, int(num_programs)) if num_programs is not None else _active_num_cus(a_packed.device)
    if auto_config:
        selected = _pick_grouped_gemm_config(m_list, n, k, model_sms)
        block_m = int(selected["block_m"])
        block_n = int(selected["block_n"])
        block_k = int(selected["block_k"])
        group_m = int(selected["group_m"])
        tdm_pipeline_depth = int(selected["tdm_pipeline_depth"])
        c_staging_mode = int(selected["c_staging_mode"])
        cross_tile_prefetch = bool(selected["cross_tile_prefetch"])

    assert block_k == 128, "first TDM schedule requires BLOCK_K=128"
    assert tdm_pipeline_depth in (2, 3, 4), "tdm_pipeline_depth must be 2, 3, or 4"
    assert c_staging_mode in (0, 1)
    assert (not cross_tile_prefetch or c_staging_mode == 1), "cross_tile_prefetch requires dedicated C staging"
    assert (not cross_tile_prefetch or tdm_pipeline_depth == 2), "cross_tile_prefetch currently requires depth 2"
    assert n % block_n == 0 and k % block_k == 0
    assert k // block_k >= tdm_pipeline_depth
    assert (not cross_tile_prefetch or (k // block_k) % tdm_pipeline_depth
            == 0), "cross_tile_prefetch requires an even number of K blocks so K0 maps back to ring slot 0"

    total_tiles = 0
    for i, m in enumerate(m_list):
        assert m >= 0, "group_offsets must be monotonically increasing"
        assert m % block_m == 0, f"group {i} M={m} must be divisible by block_m={block_m}"
        total_tiles += (m // block_m) * (n // block_n)

    if c_packed is None:
        c_packed = torch.empty((total_m, n), device=a_packed.device, dtype=a_packed.dtype)
    assert c_packed.device == a_packed.device and c_packed.dtype == a_packed.dtype
    assert c_packed.shape == (total_m, n)
    assert c_packed.stride(1) == 1

    if total_tiles == 0:
        return c_packed

    if num_programs is None:
        num_programs = _active_num_cus(a_packed.device)
    num_programs = max(1, min(int(num_programs), total_tiles))

    def run_kernel():
        return grouped_gemm_tdm_kernel[(num_programs, )](
            a_packed,
            b_t,
            c_packed,
            group_offsets,
            group_size,
            n,
            a_packed.stride(0),
            b_t.stride(0),
            b_t.stride(1),
            c_packed.stride(0),
            K=k,
            NUM_PROGRAMS=num_programs,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            NUM_BUFFERS=tdm_pipeline_depth,
            L2_PREFETCH_DISTANCE=l2_prefetch_distance,
            C_STAGING_MODE=c_staging_mode,
            CROSS_TILE_PREFETCH=cross_tile_prefetch,
            XCD_REMAP_MODE=_XCD_REMAP_MODES[xcd_remap_mode],
            NUM_XCDS=num_xcds,
            XCD_CHUNK=xcd_chunk,
            num_warps=4,
            waves_per_eu=1,
        )

    if benchmark == "graph":
        ms = triton.testing.do_bench_cudagraph(run_kernel, rep=benchmark_num_iters)
        print(f"execution time: {ms} ms, {_grouped_gemm_tflops(ms, m_list, n, k):.2f} TFLOPS")
    elif benchmark == "eager":
        ms = triton.testing.do_bench(run_kernel, warmup=30, rep=benchmark_num_iters)
        print(f"execution time: {ms} ms, {_grouped_gemm_tflops(ms, m_list, n, k):.2f} TFLOPS")
    else:
        run_kernel()
    return c_packed


def _make_groups(shapes: list[tuple[int, int, int]], device: torch.device):
    group_a = []
    group_b = []
    for m, n, k in shapes:
        group_a.append(torch.randn((m, k), device=device, dtype=torch.float16))
        group_b.append(torch.randn((k, n), device=device, dtype=torch.float16))
    return group_a, group_b


def _make_packed_ragged_m(m_list: list[int], n: int, k: int, device: torch.device):
    group_a = [torch.randn((m, k), device=device, dtype=torch.float16) for m in m_list]
    b_t = torch.randn((len(m_list), n, k), device=device, dtype=torch.float16)
    a_packed = torch.cat(group_a, dim=0).contiguous()
    group_offsets = torch.tensor([0] + list(torch.tensor(m_list).cumsum(0).tolist()), device=device, dtype=torch.int32)
    return a_packed, b_t, group_offsets, group_a


def _grouped_gemm_tdm_compile_signature() -> dict[str, str]:
    return {
        "a_packed": "*fp16",
        "b_t": "*fp16",
        "c_packed": "*fp16",
        "group_offsets": "*i32",
        "group_size": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_bg": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
    }


def _grouped_gemm_tdm_compile_attrs() -> dict[tuple[int, ...], list[list[int | str]]]:
    # Match runtime specialization for this packed contiguous benchmark shape:
    # D+S for pointers and D for aligned integer shape/stride arguments.
    names = grouped_gemm_tdm_kernel.arg_names
    attrs: dict[tuple[int, ...], list[list[int | str]]] = {}
    for name in ("a_packed", "b_t", "c_packed", "group_offsets"):
        attrs[(names.index(name), )] = [["tt.divisibility", 16], ["tt.pointer_range", 32]]
    for name in ("N", "stride_am", "stride_bg", "stride_bn", "stride_cm"):
        attrs[(names.index(name), )] = [["tt.divisibility", 16]]
    return attrs


@pytest.mark.parametrize("mode", ["none", "balanced", "chunked"])
@pytest.mark.parametrize("num_programs", [4, 10, 32, 37])
def test_grouped_gemm_xcd_remap_is_permutation(mode, num_programs):
    mapped = [
        _remap_program_id_reference(pid, num_programs, mode, num_xcds=8, chunk_size=2) for pid in range(num_programs)
    ]
    assert sorted(mapped) == list(range(num_programs))


def test_grouped_gemm_cost_model_selects_large_saturated_tile():
    cfg = _pick_grouped_gemm_config([4096] * 16, n=4096, k=4096, num_sms=256)
    assert (cfg["block_m"], cfg["block_n"]) == (256, 256)


def test_grouped_gemm_cost_model_selects_small_m_tile():
    cfg = _pick_grouped_gemm_config([128] * 16, n=4096, k=4096, num_sms=256)
    assert (cfg["block_m"], cfg["block_n"]) == (128, 256)


@pytest.mark.parametrize("TDM_PIPELINE_DEPTH", [2, 3, 4])
def test_grouped_gemm_tdm_compiles_gfx1250(TDM_PIPELINE_DEPTH):
    """The packed ragged-M path should lower to gfx1250 TDM ops."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=grouped_gemm_tdm_kernel,
        signature=_grouped_gemm_tdm_compile_signature(),
        constexprs={
            "NUM_PROGRAMS": 8,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "NUM_BUFFERS": TDM_PIPELINE_DEPTH,
            "L2_PREFETCH_DISTANCE": 0,
            "C_STAGING_MODE": 0,
            "CROSS_TILE_PREFETCH": False,
            "XCD_REMAP_MODE": 0,
            "NUM_XCDS": 8,
            "XCD_CHUNK": 2,
            "K": 2048,
        },
        attrs=_grouped_gemm_tdm_compile_attrs(),
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_group_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    assert "tt.dot" in ttgir

    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "tensor_store_from_lds" in amdgcn or "tensor.store.from.lds" in amdgcn
    assert amdgcn.count("v_wmma_f32_16x16x32_f16") >= 64


def test_grouped_gemm_tdm_asymmetric_alias_compiles_gfx1250():
    """Asymmetric C tiles should pick a legal reuse buffer for local aliasing."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=grouped_gemm_tdm_kernel,
        signature=_grouped_gemm_tdm_compile_signature(),
        constexprs={
            "NUM_PROGRAMS": 8,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "NUM_BUFFERS": 3,
            "L2_PREFETCH_DISTANCE": 0,
            "C_STAGING_MODE": 0,
            "CROSS_TILE_PREFETCH": False,
            "XCD_REMAP_MODE": 0,
            "NUM_XCDS": 8,
            "XCD_CHUNK": 2,
            "K": 2048,
        },
        attrs=_grouped_gemm_tdm_compile_attrs(),
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_group_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "tt.dot" in ttgir


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(128, 256), (256, 128)])
def test_grouped_gemm_tdm_asymmetric_dedicated_c_compiles_gfx1250(BLOCK_M, BLOCK_N):
    """Asymmetric depth-2 tiles should fit a full C buffer without aliasing A."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=grouped_gemm_tdm_kernel,
        signature=_grouped_gemm_tdm_compile_signature(),
        constexprs={
            "NUM_PROGRAMS": 4,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "NUM_BUFFERS": 2,
            "L2_PREFETCH_DISTANCE": 0,
            "C_STAGING_MODE": 1,
            "CROSS_TILE_PREFETCH": False,
            "XCD_REMAP_MODE": 0,
            "NUM_XCDS": 8,
            "XCD_CHUNK": 2,
            "K": 2048,
        },
        attrs=_grouped_gemm_tdm_compile_attrs(),
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})

    assert compiled.metadata.shared < 320 * 1024
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_group_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "tt.dot" in ttgir


@pytest.mark.parametrize("XCD_REMAP_MODE", [0, 1, 2])
def test_grouped_gemm_tdm_cross_tile_prefetch_compiles_gfx1250(XCD_REMAP_MODE):
    """The peeled depth-2 tail should retain the next-tile TDM load path."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=grouped_gemm_tdm_kernel,
        signature=_grouped_gemm_tdm_compile_signature(),
        constexprs={
            "NUM_PROGRAMS": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "NUM_BUFFERS": 2,
            "L2_PREFETCH_DISTANCE": 0,
            "C_STAGING_MODE": 1,
            "CROSS_TILE_PREFETCH": True,
            "XCD_REMAP_MODE": XCD_REMAP_MODE,
            "NUM_XCDS": 8,
            "XCD_CHUNK": 2,
            "K": 2048,
        },
        attrs=_grouped_gemm_tdm_compile_attrs(),
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})

    assert compiled.metadata.shared < 320 * 1024
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdg.async_tdm_group_copy_global_to_local") >= 4
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert "tt.dot" in ttgir
    last_dot = ttgir.rfind("tt.dot")
    late_barrier = ttgir.find("rocdl.sched.barrier", last_dot)
    late_c_desc = ttgir.find("amdg.update_tensor_descriptor %c_desc_base", late_barrier)
    closing_barrier = ttgir.find("rocdl.sched.barrier", late_c_desc)
    assert last_dot < late_barrier < late_c_desc < closing_barrier

    amdgcn = compiled.asm["amdgcn"]
    assert amdgcn.count("tensor_load_to_lds") >= 4
    assert "tensor_store_from_lds" in amdgcn


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_grouped_gemm_phase0_ragged_gfx1250():
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    shapes = [
        (17, 33, 31),
        (64, 70, 64),
        (95, 128, 96),
        (128, 65, 129),
    ]
    group_a, group_b = _make_groups(shapes, device)

    actual = grouped_gemm_phase0(group_a, group_b, block_m=32, block_n=32, block_k=32)
    expected = [a @ b for a, b in zip(group_a, group_b)]

    for ref, out in zip(expected, actual):
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("TDM_PIPELINE_DEPTH", [2, 3, 4])
def test_grouped_gemm_tdm_packed_ragged_m_gfx1250(TDM_PIPELINE_DEPTH):
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    m_list = [128, 256, 384]
    n = 256
    k = 512
    a_packed, b_t, group_offsets, group_a = _make_packed_ragged_m(m_list, n, k, device)

    actual = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=128,
        block_n=128,
        block_k=128,
        group_m=4,
        tdm_pipeline_depth=TDM_PIPELINE_DEPTH,
        l2_prefetch_distance=0,
    )

    start = 0
    for i, m in enumerate(m_list):
        ref = group_a[i] @ b_t[i].T
        torch.testing.assert_close(actual[start:start + m], ref, atol=1e-2, rtol=1e-2)
        start += m


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_grouped_gemm_tdm_asymmetric_depth3_gfx1250():
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    m_list = [128, 256]
    n = 512
    k = 384
    a_packed, b_t, group_offsets, group_a = _make_packed_ragged_m(m_list, n, k, device)

    actual = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=128,
        block_n=256,
        block_k=128,
        group_m=4,
        tdm_pipeline_depth=3,
        l2_prefetch_distance=0,
    )

    start = 0
    for i, m in enumerate(m_list):
        ref = group_a[i] @ b_t[i].T
        torch.testing.assert_close(actual[start:start + m], ref, atol=1e-2, rtol=1e-2)
        start += m


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_grouped_gemm_tdm_cross_tile_prefetch_gfx1250():
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    # The first group has four tiles for two persistent programs, so each
    # program consumes one tile whose first two K blocks were prefetched.
    m_list = [256, 128]
    n = 512
    k = 512
    a_packed, b_t, group_offsets, group_a = _make_packed_ragged_m(m_list, n, k, device)

    actual = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=128,
        block_n=256,
        block_k=128,
        group_m=4,
        tdm_pipeline_depth=2,
        l2_prefetch_distance=0,
        num_programs=2,
        c_staging_mode=1,
        cross_tile_prefetch=True,
    )

    start = 0
    for i, m in enumerate(m_list):
        ref = group_a[i] @ b_t[i].T
        torch.testing.assert_close(actual[start:start + m], ref, atol=1e-2, rtol=1e-2)
        start += m


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("XCD_REMAP_MODE", ["balanced", "chunked"])
def test_grouped_gemm_tdm_xcd_remap_gfx1250(XCD_REMAP_MODE):
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    m_list = [1024]
    n = 512
    k = 512
    a_packed, b_t, group_offsets, group_a = _make_packed_ragged_m(m_list, n, k, device)
    actual = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=128,
        block_n=256,
        block_k=128,
        num_programs=16,
        c_staging_mode=1,
        xcd_remap_mode=XCD_REMAP_MODE,
        num_xcds=8,
        xcd_chunk=2,
    )
    torch.testing.assert_close(actual, group_a[0] @ b_t[0].T, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import argparse

    if not is_hip_gfx1250():
        raise SystemExit("Requires gfx1250 hardware")

    parser = argparse.ArgumentParser(description="Benchmark the TLX AMD f16 grouped GEMM TDM kernel")
    parser.add_argument("--m_list", type=str, default="512,512,512,512", help="Comma-separated M sizes per group")
    parser.add_argument("-N", type=int, default=1024, help="problem N size")
    parser.add_argument("-K", type=int, default=2048, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=128, help="BLOCK_K")
    parser.add_argument("--group_m", type=int, default=4, help="GROUP_M launch-order swizzle")
    parser.add_argument("--tdm_pipeline_depth", type=int, default=2, choices=[2, 3, 4],
                        help="TDM LDS ring-buffer depth")
    parser.add_argument("--l2_prefetch_distance", type=int, default=0)
    parser.add_argument("--num_programs", type=int, default=None,
                        help="persistent program count; defaults to the active CU count")
    parser.add_argument("--dedicated_c_buffer", action=argparse.BooleanOptionalAction, default=False,
                        help="stage C in distinct LDS instead of aliasing the A ring")
    parser.add_argument("--cross_tile_prefetch", action=argparse.BooleanOptionalAction, default=False,
                        help="prefetch the next same-group tile in the peeled K-loop tail")
    parser.add_argument("--auto_config", action=argparse.BooleanOptionalAction, default=False,
                        help="select a validated tile config using the host cost model")
    parser.add_argument("--xcd_remap", choices=tuple(_XCD_REMAP_MODES), default="none",
                        help="persistent program-id mapping across gfx1250 XCDs")
    parser.add_argument("--num_xcds", type=int, default=8)
    parser.add_argument("--xcd_chunk", type=int, default=2)
    parser.add_argument("--benchmark_mode", choices=["eager", "graph", "none"], default="eager")
    parser.add_argument("--benchmark_num_iters", type=int, default=32)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    m_list = [int(x) for x in args.m_list.split(",") if x]
    if not m_list:
        raise SystemExit("--m_list must contain at least one group")

    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(args.seed)
    if args.check:
        a_packed, b_t, group_offsets, group_a = _make_packed_ragged_m(m_list, args.N, args.K, device)
    else:
        total_m = sum(m_list)
        a_packed = torch.randn((total_m, args.K), device=device, dtype=torch.float16)
        b_t = torch.randn((len(m_list), args.N, args.K), device=device, dtype=torch.float16)
        group_offsets = torch.tensor([0] + list(torch.tensor(m_list).cumsum(0).tolist()), device=device,
                                     dtype=torch.int32)
        group_a = None

    if args.auto_config:
        model_sms = args.num_programs if args.num_programs is not None else _active_num_cus(device)
        selected = _pick_grouped_gemm_config(m_list, args.N, args.K, model_sms)
        print("selected config:", selected["name"], f"BM={selected['block_m']}", f"BN={selected['block_n']}",
              f"cross_tile_prefetch={selected['cross_tile_prefetch']}")

    benchmark = None if args.benchmark_mode == "none" else args.benchmark_mode
    out = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=args.BM,
        block_n=args.BN,
        block_k=args.BK,
        group_m=args.group_m,
        tdm_pipeline_depth=args.tdm_pipeline_depth,
        l2_prefetch_distance=args.l2_prefetch_distance,
        num_programs=args.num_programs,
        c_staging_mode=int(args.dedicated_c_buffer),
        cross_tile_prefetch=args.cross_tile_prefetch,
        auto_config=args.auto_config,
        xcd_remap_mode=args.xcd_remap,
        num_xcds=args.num_xcds,
        xcd_chunk=args.xcd_chunk,
        benchmark=benchmark,
        benchmark_num_iters=args.benchmark_num_iters,
    )

    if args.check:
        assert group_a is not None
        start = 0
        max_diff = 0.0
        for i, m in enumerate(m_list):
            ref = group_a[i] @ b_t[i].T
            diff = (out[start:start + m] - ref).abs().max().item()
            max_diff = max(max_diff, diff)
            torch.testing.assert_close(out[start:start + m], ref, atol=1e-2, rtol=1e-2)
            start += m
        print(f"max abs diff: {max_diff:.4f}")
