"""Hopper (sm90) ``tlx.ops.addmm_mul_add``: ``out = x0 * (bias + a @ b) + res``.

This is the cross layer of a low-rank DCN, ``x_{l+1} = x0 * U(V x_l) + x_l``.
The up-projection GEMM has a small K (256-512) and a wide N, so its output is
as large as ``x0``, and the pointwise kernel that applies the mul/add after it
costs more than the GEMM. Fusing the two removes one write and one read of the
GEMM output.

Two kernels. For K <= 512 the heuristic launch uses the A-resident kernel:

* each CTA owns one 128-row M block (A is loaded into SMEM once per block and
  only B streams through the TMA ring), and the CTAs of different M blocks walk
  the same N columns at the same time, so B is read from DRAM once;
* two wgmma consumers add the bias (TMA-loaded into SMEM), round the linear
  output to the input dtype (as the unfused ``linear`` -> ``mul`` -> ``add``
  sequence does) and hand their 64-row half-tile to SMEM;
* a separate epilogue warpgroup streams ``x0``/``res`` with plain loads one
  64-column chunk ahead, applies the mul/add in fp32 and stores ``out`` with
  plain coalesced stores. These warps run no wgmma and no proxy fence, which
  both wait for every outstanding register load; when the wgmma warps load
  ``x0``/``res`` themselves, each tile's mainloop waits the full load latency.

Larger K, and the ``full`` autotuned space, use the persistent cooperative
GEMM of ``kernels/mm/sm90.py``: a producer task TMA-loads A/B tiles into a
ring, and two replicated wgmma consumers each own half of the 128-row tile and
load their ``x0``/``res`` half-tile before the mainloop. In both kernels every
output element is written after the CTA that writes it has read its ``x0`` and
``res``, so ``out`` may alias either.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

from ...._catalog import InvalidInput
from ...mm._sm90_core import _allocate_buffers, _consume_tile

_MAX_SMEM = 232448 - 1024
# Layout-conversion scratch the compiler adds for the epilogue (measured).
_SCRATCH_SMEM = 20 * 1024
# The A-resident kernel keeps a 128 x K slice of A in SMEM; at K = 512 that leaves room for four B stages.
_RESIDENT_MAX_K = 512
_RESIDENT_BN = 128


@functools.lru_cache(maxsize=None)
def _get_num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _set_block_shapes(nargs):
    bm_split = nargs["BM"] // 2
    bn, bk = nargs["BN"], nargs["BK"]
    nargs["a_desc"].block_shape = [bm_split, bk]
    nargs["b_desc"].block_shape = [bk, bn] if nargs["B_ROW_MAJOR"] else [bn, bk]
    nargs["c_desc"].block_shape = [bm_split, bn]


def _smem_bytes(bm, bn, bk, num_stages, elem_bytes=2):
    return (bm + bn) * bk * elem_bytes * num_stages + bm * bn * elem_bytes + _SCRATCH_SMEM


def _config(bm, bn, bk, num_stages, group_size_m=8):
    return triton.Config(
        {
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "GROUP_SIZE_M": group_size_m,
            "NUM_STAGES": num_stages,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_set_block_shapes,
    )


def _all_configs():
    return [_config(128, 128, bk, s) for bk in (64, 128) for s in (2, 3, 4, 5)]


def _prune(configs, nargs, **kwargs):
    kept = []
    for c in configs:
        kw = c.kwargs
        if _smem_bytes(kw["BM"], kw["BN"], kw["BK"], kw["NUM_STAGES"]) > _MAX_SMEM:
            continue
        if kw["NUM_STAGES"] > triton.cdiv(nargs["K"], kw["BK"]) + 1:
            continue
        kept.append(c)
    return kept or [configs[0]]


def heuristic_config(M, N, K):
    # At K = 512 the ring must cover a tile's eight K steps through the
    # epilogue; at K <= 256 four stages (one tile of lookahead) measured best.
    k_tiles = triton.cdiv(K, 64)
    num_stages = 5 if k_tiles > 4 else max(2, min(4, k_tiles))
    return [_config(128, 128, 64, num_stages)]


@triton.jit
def _tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _producer(M, N, K, a_desc, b_desc, a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b,
              NUM_SMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
              NUM_STAGES: tl.constexpr, B_ROW_MAJOR: tl.constexpr):
    BM_SPLIT: tl.constexpr = BM // 2
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    phase = 1
    buf = 0
    for tile_id in range(tl.program_id(0), num_pid_m * num_pid_n, NUM_SMS):
        pid_m, pid_n = _tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
        offset_am = (pid_m * BM).to(tl.int32)
        offset_bn = (pid_n * BN).to(tl.int32)
        for k in range(0, tl.cdiv(K, BK)):
            offset_k = (k * BK).to(tl.int32)
            for half in tl.static_range(2):
                a_idx = buf + NUM_STAGES * half
                empty_a = tlx.local_view(bars_empty_a, a_idx)
                full_a = tlx.local_view(bars_full_a, a_idx)
                # Triton TR051: bounded unrolling misses the matching release
                # after the nested persistent loop revisits this ring slot.
                tlx.barrier_wait(empty_a, phase)  # noqa: TR051
                tlx.barrier_expect_bytes(full_a, BM_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                tlx.async_descriptor_load(a_desc, tlx.local_view(a, a_idx), [offset_am + half * BM_SPLIT, offset_k],
                                          full_a)
                if half == 0:
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(empty_b, phase)
                    tlx.barrier_expect_bytes(full_b, BN * BK * tlx.size_of(tlx.dtype_of(b_desc)))
                    if B_ROW_MAJOR:
                        tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf), [offset_k, offset_bn], full_b)
                    else:
                        tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf), [offset_bn, offset_k], full_b)
            phase = phase ^ (buf == NUM_STAGES - 1)
            buf = (buf + 1) % NUM_STAGES


@triton.jit
# Triton TR001: `_tuned` applies the caller-selected autotune space lazily.
# triton-lint: assume NUM_STAGES=4
def addmm_mul_add_kernel(  # noqa: TR001
    a_desc,
    b_desc,
    c_desc,
    bias_ptr,
    x0_ptr,
    res_ptr,
    stride_x0,
    stride_res,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BM // 2
    a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b = _allocate_buffers(a_desc, b_desc, BM, BN, BK,
                                                                                   NUM_STAGES, 2, True, B_ROW_MAJOR)
    c_smem = tlx.local_alloc((BM_SPLIT, BN), tlx.dtype_of(c_desc), 2)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            _producer(M, N, K, a_desc, b_desc, a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b, NUM_SMS, BM,
                      BN, BK, GROUP_SIZE_M, NUM_STAGES, B_ROW_MAJOR)

        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            consumer_id: tl.constexpr = tlx.async_task_replica_id()
            phase = 0
            buf = 0
            for tile_id in range(tl.program_id(0), num_pid_m * num_pid_n, NUM_SMS):
                pid_m, pid_n = _tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
                offset_cm = pid_m * BM + BM_SPLIT * consumer_id
                offset_bn = pid_n * BN
                rows = tl.arange(0, BM_SPLIT)
                offs_n = offset_bn + tl.arange(0, BN)
                mask = (offset_cm + rows[:, None] < M) & (offs_n[None, :] < N)
                # The tile's first row in int64 (row * stride can exceed INT32_MAX); in-tile offsets stay int32.
                first = offset_cm.to(tl.int64)
                x0 = tl.load(x0_ptr + first * stride_x0 + (rows[:, None] * stride_x0 + offs_n[None, :]), mask=mask,
                             other=0.0)
                if HAS_RES:
                    res = tl.load(res_ptr + first * stride_res + (rows[:, None] * stride_res + offs_n[None, :]),
                                  mask=mask, other=0.0)
                else:
                    res = x0

                acc, buf, phase = _consume_tile(K, a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b, buf,
                                                phase, consumer_id, BK, NUM_STAGES, True, B_ROW_MAJOR)
                if HAS_BIAS:
                    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
                    acc = acc + bias.to(tl.float32)[None, :]
                lin = acc.to(tlx.dtype_of(c_desc)).to(tl.float32)
                out = x0.to(tl.float32) * lin + res.to(tl.float32)

                slot = tlx.local_view(c_smem, consumer_id)
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(slot, out.to(tlx.dtype_of(c_desc)))
                tlx.fence("async_shared")
                tlx.async_descriptor_store(c_desc, slot, [offset_cm, offset_bn])
            tlx.async_descriptor_store_wait(0)


@triton.jit
def _resident_chunk(t, c: tl.constexpr, j: tl.constexpr, t_hi, num_pid_n, M, N, BM: tl.constexpr, BN: tl.constexpr,
                    EPI_N: tl.constexpr):
    BM_SPLIT: tl.constexpr = BM // 2
    pid_m = t // num_pid_n
    pid_n = t % num_pid_n
    first = pid_m * BM + c * BM_SPLIT
    rows = tl.arange(0, BM_SPLIT)
    offs_n = pid_n * BN + j * EPI_N + tl.arange(0, EPI_N)
    mask = (first + rows[:, None] < M) & (offs_n[None, :] < N) & (t < t_hi)
    # The chunk's first row in int64 (row * stride can exceed INT32_MAX); in-chunk offsets stay int32.
    return first.to(tl.int64), rows[:, None], offs_n, mask


@triton.jit
def addmm_mul_add_resident_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    DescSlots,
    bias_ptr,
    x0_ptr,
    res_ptr,
    stride_am,
    stride_bn,
    stride_cm,
    stride_x0,
    stride_res,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EPI_N: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BM // 2
    NCH: tl.constexpr = BN // EPI_N
    dt = tlx.dtype_of(c_ptr)
    a = tlx.local_alloc((BM_SPLIT, BK), tlx.dtype_of(a_ptr), 2 * K_TILES)
    if B_ROW_MAJOR:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_ptr), NUM_STAGES)
    else:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_ptr), NUM_STAGES)
    a_full = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    a_empty = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    b_empty = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)
    hb = tlx.local_alloc((BM_SPLIT, BN), dt, 2)
    h_full = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    h_empty = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    if HAS_BIAS:
        bs = tlx.local_alloc((1, BN), dt, 2)
        bs_full = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
        bs_empty = tlx.alloc_barriers(num_barriers=2, arrive_count=2)
    slots_base = DescSlots + tl.program_id(0) * 3 * 128
    M32 = tl.cast(M, tl.int32)
    N32 = tl.cast(N, tl.int32)
    K32 = tl.cast(K, tl.int32)

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if num_pid_m <= nprog:
        # CTA pid takes M block g = pid % num_pid_m and the r-th of G contiguous N ranges of it.
        g = pid % num_pid_m
        r = pid // num_pid_m
        G = (nprog - g + num_pid_m - 1) // num_pid_m
        t_lo = g * num_pid_n + (r * num_pid_n) // G
        t_hi = g * num_pid_n + ((r + 1) * num_pid_n) // G
    else:
        # More M blocks than CTAs: contiguous chunks of the M-major tile order.
        num_tiles = num_pid_m * num_pid_n
        base = num_tiles // nprog
        extra = num_tiles % nprog
        t_lo = pid * base + tl.minimum(pid, extra)
        t_hi = t_lo + base + (pid < extra).to(tl.int32)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            slots = tlx.tensor_descriptor_ptr(slots_base.handle, 3, 128)
            a_desc = tlx.make_tensor_descriptor(slots[0], a_ptr, shape=[M32, K32], strides=[stride_am, 1],
                                                block_shape=[BM_SPLIT, BK])
            if B_ROW_MAJOR:
                b_desc = tlx.make_tensor_descriptor(slots[1], b_ptr, shape=[K32, N32], strides=[stride_bn, 1],
                                                    block_shape=[BK, BN])
            else:
                b_desc = tlx.make_tensor_descriptor(slots[1], b_ptr, shape=[N32, K32], strides=[stride_bn, 1],
                                                    block_shape=[BN, BK])
            if HAS_BIAS:
                # A one-row view; its row stride is never used but must be a multiple of 16 bytes.
                bias_desc = tlx.make_tensor_descriptor(slots[2], bias_ptr, shape=[1, N32],
                                                       strides=[(N32 + 7) // 8 * 8, 1], block_shape=[1, BN])
            phase = 1
            buf = 0
            a_phase = 1
            prev_m = -1
            for t in range(t_lo, t_hi):
                pid_m = t // num_pid_n
                pid_n = t % num_pid_n
                offset_bn = (pid_n * BN).to(tl.int32)
                if HAS_BIAS:
                    bslot = (t - t_lo) % 2
                    tlx.barrier_wait(tlx.local_view(bs_empty, bslot), (((t - t_lo) // 2) & 1) ^ 1)
                    tlx.barrier_expect_bytes(tlx.local_view(bs_full, bslot), BN * tlx.size_of(dt))
                    tlx.async_descriptor_load(bias_desc, tlx.local_view(bs, bslot), [0, offset_bn],
                                              tlx.local_view(bs_full, bslot))
                if pid_m != prev_m:
                    for half in tl.static_range(2):
                        tlx.barrier_wait(tlx.local_view(a_empty, half), a_phase)
                        tlx.barrier_expect_bytes(tlx.local_view(a_full, half),
                                                 K_TILES * BM_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_ptr)))
                        for kk in tl.static_range(K_TILES):
                            tlx.async_descriptor_load(a_desc, tlx.local_view(a, half * K_TILES + kk),
                                                      [(pid_m * BM + half * BM_SPLIT).to(tl.int32), kk * BK],
                                                      tlx.local_view(a_full, half))
                    a_phase = a_phase ^ 1
                    prev_m = pid_m
                for kk in range(0, K_TILES):
                    # Triton TR051: bounded unrolling misses the matching release after the persistent loop wraps.
                    tlx.barrier_wait(tlx.local_view(b_empty, buf), phase)  # noqa: TR051
                    tlx.barrier_expect_bytes(tlx.local_view(b_full, buf), BN * BK * tlx.size_of(tlx.dtype_of(b_ptr)))
                    if B_ROW_MAJOR:
                        tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf), [kk * BK, offset_bn],
                                                  tlx.local_view(b_full, buf))
                    else:
                        tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf), [offset_bn, kk * BK],
                                                  tlx.local_view(b_full, buf))
                    phase = phase ^ (buf == NUM_STAGES - 1)
                    buf = (buf + 1) % NUM_STAGES

        with tlx.async_task(num_warps=4):
            first, rows, offs_n, mask = _resident_chunk(t_lo, 0, 0, t_hi, num_pid_n, M, N, BM, BN, EPI_N)
            xa = tl.load(x0_ptr + first * stride_x0 + (rows * stride_x0 + offs_n[None, :]), mask=mask, other=0.0)
            if HAS_RES:
                ra = tl.load(res_ptr + first * stride_res + (rows * stride_res + offs_n[None, :]), mask=mask, other=0.0)
            else:
                ra = xa
            hphase = 0
            for t in range(t_lo, t_hi):
                for c in tl.static_range(2):
                    for j in tl.static_range(NCH):
                        first, rows, offs_n, mask = _resident_chunk(t, c, j, t_hi, num_pid_n, M, N, BM, BN, EPI_N)
                        if c == 1 and j == NCH - 1:
                            nfirst, nrows, noffs_n, nmask = _resident_chunk(t + 1, 0, 0, t_hi, num_pid_n, M, N, BM, BN,
                                                                            EPI_N)
                        elif j == NCH - 1:
                            nfirst, nrows, noffs_n, nmask = _resident_chunk(t, c + 1, 0, t_hi, num_pid_n, M, N, BM, BN,
                                                                            EPI_N)
                        else:
                            nfirst, nrows, noffs_n, nmask = _resident_chunk(t, c, j + 1, t_hi, num_pid_n, M, N, BM, BN,
                                                                            EPI_N)
                        xn = tl.load(x0_ptr + nfirst * stride_x0 + (nrows * stride_x0 + noffs_n[None, :]), mask=nmask,
                                     other=0.0)
                        if HAS_RES:
                            rn = tl.load(res_ptr + nfirst * stride_res + (nrows * stride_res + noffs_n[None, :]),
                                         mask=nmask, other=0.0)
                        else:
                            rn = xn
                        if j == 0:
                            tlx.barrier_wait(tlx.local_view(h_full, c), hphase)
                        if NCH == 1:
                            lin = tlx.local_load(tlx.local_view(hb, c))
                        else:
                            lin = tlx.local_load(
                                tlx.local_slice(tlx.local_view(hb, c), [0, j * EPI_N], [BM_SPLIT, EPI_N]))
                        out = xa.to(tl.float32) * lin.to(tl.float32) + ra.to(tl.float32)
                        tl.store(c_ptr + first * stride_cm + (rows * stride_cm + offs_n[None, :]), out.to(dt),
                                 mask=mask)
                        if j == NCH - 1:
                            # Release the handoff slot only after its values have been used.
                            tlx.barrier_arrive(tlx.local_view(h_empty, c))
                        xa = xn
                        ra = rn
                hphase = hphase ^ 1

        with tlx.async_task(num_warps=4, replicate=2):
            cid: tl.constexpr = tlx.async_task_replica_id()
            phase = 0
            buf = 0
            a_phase = 0
            hphase = 1
            prev_m = -1
            for t in range(t_lo, t_hi):
                pid_m = t // num_pid_n
                if pid_m != prev_m:
                    tlx.barrier_wait(tlx.local_view(a_full, cid), a_phase)
                    a_phase = a_phase ^ 1
                    prev_m = pid_m
                last_buf = buf
                tlx.barrier_wait(tlx.local_view(b_full, buf), phase)
                b_op = tlx.local_view(b, buf) if B_ROW_MAJOR else tlx.local_trans(tlx.local_view(b, buf))
                acc = tlx.async_dot(tlx.local_view(a, cid * K_TILES), b_op)
                phase = phase ^ (buf == NUM_STAGES - 1)
                buf = (buf + 1) % NUM_STAGES
                for kk in range(1, K_TILES):
                    # Triton TR051: bounded unrolling misses the producer's TMA completion after the ring wraps.
                    tlx.barrier_wait(tlx.local_view(b_full, buf), phase)  # noqa: TR051
                    b_op = tlx.local_view(b, buf) if B_ROW_MAJOR else tlx.local_trans(tlx.local_view(b, buf))
                    acc = tlx.async_dot(tlx.local_view(a, cid * K_TILES + kk), b_op, acc)
                    acc = tlx.async_dot_wait(1, acc)
                    tlx.barrier_arrive(tlx.local_view(b_empty, last_buf))
                    last_buf = buf
                    phase = phase ^ (buf == NUM_STAGES - 1)
                    buf = (buf + 1) % NUM_STAGES
                acc = tlx.async_dot_wait(0, acc)
                tlx.barrier_arrive(tlx.local_view(b_empty, last_buf))
                # Release A as soon as the last MMA of this M block has retired.
                if (t + 1 == t_hi) | ((t + 1) // num_pid_n != pid_m):
                    tlx.barrier_arrive(tlx.local_view(a_empty, cid))
                if HAS_BIAS:
                    bslot = (t - t_lo) % 2
                    tlx.barrier_wait(tlx.local_view(bs_full, bslot), ((t - t_lo) // 2) & 1)
                    bias = tlx.local_load(tlx.local_view(bs, bslot))
                    acc = acc + bias.to(tl.float32)
                tlx.barrier_wait(tlx.local_view(h_empty, cid), hphase)
                tlx.local_store(tlx.local_view(hb, cid), acc.to(dt))
                tlx.barrier_arrive(tlx.local_view(h_full, cid))
                if HAS_BIAS:
                    # The bias slot is released after the value has been used (the stored half-tile depends on it).
                    tlx.barrier_arrive(tlx.local_view(bs_empty, bslot))
                hphase = hphase ^ 1


@functools.lru_cache(maxsize=None)
def _tuned(space):
    assert space == "full", space
    return triton.autotune(configs=_all_configs(), key=["M", "N", "K", "HAS_RES", "B_ROW_MAJOR"],
                           prune_configs_by={"early_config_prune": _prune})(addmm_mul_add_kernel)


# (M, N, K, has_res, b_row_major, dtype) keys already autotuned in the full space.
_FULL_TUNED: set = set()


def _has_storage(t):
    # Fake/functional tensors (torch.export tracing) have no data pointer to check.
    from torch._subclasses.fake_tensor import is_fake

    return not is_fake(t)


def _aligned(t):
    """Whether ``t`` starts on a 16-byte boundary.

    A traced tensor has no address, so its storage offset decides: Inductor treats an input as aligned only when the
    traced input was, and passes an input traced at an unaligned offset as it is.
    """
    if _has_storage(t):
        return t.data_ptr() % 16 == 0
    return t.storage_offset() * t.element_size() % 16 == 0


def _check_matrix(name, t, shape, dtype, device):
    if t.dim() != 2 or tuple(t.shape) != tuple(shape) or t.dtype != dtype or t.device != device:
        raise InvalidInput(f"{name} must be a {dtype} tensor of shape {tuple(shape)} on {device}")
    if t.stride(1) != 1 or t.stride(0) % 8 != 0 or not _aligned(t):
        raise InvalidInput(f"{name} must be row-major with a 16-byte-aligned base and row stride")
    # The epilogue adds row offsets within a 64-row half-tile in int32.
    if t.stride(0) >= 2**31 // 64:
        raise InvalidInput(f"{name} row stride must be below {2**31 // 64} elements")


def _validate(bias, a, b, x0, res, out):
    """Checks the operands; returns ``(M, N, K, b_src, b_row_major)``."""
    if a.dim() != 2 or b.dim() != 2:
        raise InvalidInput("sm90 tlx.ops.addmm_mul_add expects two-dimensional a and b")
    M, K = a.shape
    K2, N = b.shape
    if K2 != K or min(N, K) <= 0 or M <= 0:
        raise InvalidInput(f"sm90 tlx.ops.addmm_mul_add got incompatible a {tuple(a.shape)} and b {tuple(b.shape)}")
    dtype, device = a.dtype, a.device
    if dtype not in (torch.float16, torch.bfloat16):
        raise InvalidInput(f"sm90 tlx.ops.addmm_mul_add does not support {dtype}; supported: float16, bfloat16")
    _check_matrix("a", a, (M, K), dtype, device)
    b_row_major = b.stride(1) == 1
    b_src = b if b_row_major else b.t()
    _check_matrix("b", b_src, b_src.shape, dtype, device)
    _check_matrix("x0", x0, (M, N), dtype, device)
    if res is not None:
        _check_matrix("res", res, (M, N), dtype, device)
    if bias is not None and (tuple(bias.shape) != (N, ) or bias.dtype != dtype or bias.device != device
                             or not bias.is_contiguous()):
        raise InvalidInput(f"bias must be a contiguous {dtype} tensor of shape ({N},) on {device}")
    if out is not None:
        _check_matrix("out", out, (M, N), dtype, device)
    return M, N, K, b_src, b_row_major


class KernelLaunch(NamedTuple):
    """``kernel[grid](*args, **kwargs)``; ``kernel`` is a plain ``triton.jit`` function."""
    kernel: object
    grid: tuple
    args: tuple
    kwargs: dict


def _resident_config(K, elem_bytes=2):
    """``(BN, BK, num_stages)`` of the A-resident kernel, or ``None`` when the 128 x K slice of A does not fit."""
    if K > _RESIDENT_MAX_K:
        return None
    bk = 64
    a_bytes = 128 * triton.cdiv(K, bk) * bk * elem_bytes
    fixed = a_bytes + 2 * 64 * _RESIDENT_BN * elem_bytes + 2 * _RESIDENT_BN * elem_bytes + 1024
    num_stages = min(8, (_MAX_SMEM - fixed) // (_RESIDENT_BN * bk * elem_bytes))
    return _RESIDENT_BN, bk, num_stages


def _grid(device, M, N, bm, bn):
    # Only depends on M when N has fewer column tiles than the GPU has SMs, so a symbolic M is fine.
    num_sms = _get_num_sms(device)
    n_tiles = triton.cdiv(N, bn)
    return (num_sms if n_tiles >= num_sms else torch.sym_min(num_sms, triton.cdiv(M, bm) * n_tiles), )


def _resident_launch(bias, a, b_src, b_row_major, x0, res, out, M, N, K):
    bn, bk, num_stages = _resident_config(K)
    has_res = res is not None
    desc_slots = torch.empty(_get_num_sms(a.device) * 3 * 128, device=a.device, dtype=torch.int8)
    args = (
        a,
        b_src,
        out,
        desc_slots,
        bias if bias is not None else x0,
        x0,
        res if has_res else x0,
        a.stride(0),
        b_src.stride(0),
        out.stride(0),
        x0.stride(0),
        res.stride(0) if has_res else x0.stride(0),
        M,
        N,
        K,
    )
    kwargs = dict(BM=128, BN=bn, BK=bk, K_TILES=triton.cdiv(K, bk), NUM_STAGES=num_stages, EPI_N=64, HAS_RES=has_res,
                  HAS_BIAS=bias is not None, B_ROW_MAJOR=b_row_major, num_warps=4, num_stages=1)
    return KernelLaunch(addmm_mul_add_resident_kernel, _grid(a.device, M, N, 128, bn), args, kwargs)


def _cooperative_launch(bias, a, b_src, b_row_major, x0, res, out, M, N, K):
    config = heuristic_config(M, N, K)[0]
    meta = dict(config.kwargs)
    bm_split, bn, bk = meta["BM"] // 2, meta["BN"], meta["BK"]
    has_res = res is not None
    args = (
        TensorDescriptor.from_tensor(a, [bm_split, bk]),
        TensorDescriptor.from_tensor(b_src, [bk, bn] if b_row_major else [bn, bk]),
        TensorDescriptor.from_tensor(out, [bm_split, bn]),
        bias if bias is not None else x0,
        x0,
        res if has_res else x0,
        x0.stride(0),
        res.stride(0) if has_res else x0.stride(0),
        M,
        N,
        K,
    )
    kwargs = dict(meta, NUM_SMS=_get_num_sms(a.device), HAS_RES=has_res, HAS_BIAS=bias is not None,
                  B_ROW_MAJOR=b_row_major, num_warps=config.num_warps, num_stages=config.num_stages)
    return KernelLaunch(addmm_mul_add_kernel, _grid(a.device, M, N, meta["BM"], bn), args, kwargs)


def addmm_mul_add_launch(bias, a, b, x0, res=None, *, out):
    """The heuristic-space launch of :func:`addmm_mul_add` as a :class:`KernelLaunch`.

    For callers that launch the kernel themselves, such as a
    ``torch.library.triton_op`` that calls ``wrap_triton(kernel)[grid](...)``
    so that torch.export / AOTInductor compile the kernel into the artifact.
    There is no autotuner and no pre-hook. For K <= 512 this is the A-resident
    kernel, which builds its TMA descriptors on the device in a small scratch
    tensor that the launch allocates; for larger K it is the cooperative kernel
    with ``TensorDescriptor.from_tensor`` descriptors. The grid only depends on
    M when N has fewer column tiles than the GPU has SMs, so a symbolic M is
    fine. The output is the same as ``addmm_mul_add(..., out=out,
    space="heuristic")``.
    """
    M, N, K, b_src, b_row_major = _validate(bias, a, b, x0, res, out)
    # The resident kernel TMA-loads the bias, which needs a 16-byte-aligned base.
    bias_tma_ok = bias is None or _aligned(bias)
    resident = _resident_config(K) is not None and bias_tma_ok
    launch = _resident_launch if resident else _cooperative_launch
    return launch(bias, a, b_src, b_row_major, x0, res, out, M, N, K)


def addmm_mul_add(bias, a, b, x0, res=None, *, out=None, space="heuristic"):
    """Compute ``x0 * (bias + a @ b) + res``; ``res`` defaults to ``x0``.

    ``a`` is ``(M, K)`` row-major; ``b`` is ``(K, N)`` in either layout (pass
    ``linear.weight.t()`` for an ``nn.Linear``); ``bias`` is ``(N,)`` or
    ``None``; ``x0``, ``res`` and ``out`` are ``(M, N)`` row-major. ``out`` may
    alias ``x0`` or ``res``. The linear output is rounded to the input dtype
    before the mul/add, as in the unfused sequence.
    """
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"sm90 tlx.ops.addmm_mul_add does not provide search space {space!r}; "
                           "expected 'heuristic' or 'full'")
    if space == "heuristic":
        # addmm_mul_add_launch validates the operands, including this out.
        if out is None:
            out = torch.empty((a.shape[0], b.shape[-1]), device=a.device, dtype=a.dtype)
        launch = addmm_mul_add_launch(bias, a, b, x0, res, out=out)
        # Triton launches on the current device.
        with torch.cuda.device(a.device):
            launch.kernel[launch.grid](*launch.args, **launch.kwargs)
        return out

    M, N, K, b_src, b_row_major = _validate(bias, a, b, x0, res, out)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    has_res = res is not None
    kernel = _tuned(space)
    num_sms = _get_num_sms(a.device)

    def launch(c):
        dummy = [1, 1]
        kernel[lambda meta: (min(num_sms,
                                 triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"])), )](
                                     TensorDescriptor(a, a.shape, a.stride(), dummy),
                                     TensorDescriptor(b_src, b_src.shape, b_src.stride(), dummy),
                                     TensorDescriptor(c, c.shape, c.stride(), dummy),
                                     bias if bias is not None else x0,
                                     x0,
                                     res if has_res else x0,
                                     x0.stride(0),
                                     res.stride(0) if has_res else x0.stride(0),
                                     M,
                                     N,
                                     K,
                                     NUM_SMS=num_sms,
                                     HAS_RES=has_res,
                                     HAS_BIAS=bias is not None,
                                     B_ROW_MAJOR=b_row_major,
                                 )

    key = (M, N, K, has_res, b_row_major, a.dtype)
    with torch.cuda.device(a.device):
        if key not in _FULL_TUNED:
            # Autotuning reruns the kernel; tune into scratch so an output that
            # aliases x0 or res is not updated more than once.
            launch(torch.empty_like(out))
            _FULL_TUNED.add(key)
        launch(out)
    return out
