"""MXFP6 GEMM backend for gfx950 (CDNA4): ``a @ b.T`` as BF16.

``a_format`` and ``b_format`` independently select E2M3 or E3M2.
Both operands are dense-packed uint8 (groups of
16 codes per 12 bytes, K padded to the unpacked width; shapes count
logical values in multiples of 16). Shapes with M a multiple of 256,
N a multiple of 128 and at least two K tiles per split use the
LDS-pipelined interwave kernels (256x256 or 256x128 tiles); other shapes use the register-staged native kernel,
whose launch config ``space`` selects ("heuristic" single config,
"full" autotuned). Only the "natural" scale layout is supported.

Registered in ``tlx.ops`` for gfx950: the native scaled-MFMA path
produces correct numerics for K=128 tiles, so MI350 callers dispatch
here through the catalog instead of getting ``UnsupportedOp``.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


@triton.jit
def _fp6_codes_from_bytes(b0, b1, b2):
    v0 = b0 & 63
    v1 = ((b0 >> 6) & 3) | ((b1 & 15) << 2)
    v2 = ((b1 >> 4) & 15) | ((b2 & 3) << 4)
    v3 = (b2 >> 2) & 63
    return v0, v1, v2, v3


@triton.jit
def _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset, BLOCK_SIZE_K: tl.constexpr):
    valid_k = k_offset + tl.arange(0, BLOCK_SIZE_K) < K
    valid_sk = k_offset + 32 * tl.arange(0, BLOCK_SIZE_K // 32) < K
    a = tl.load(a_bptrs, mask=valid_m[:, None] & valid_k[None, :], other=0)
    b = tl.load(b_bptrs, mask=valid_k[:, None] & valid_n[None, :], other=0)
    # E8M0 byte 127 encodes 1.0, the neutral scale for zero-padded codes.
    unit_scale = tl.full((), 127, tl.uint8).to(tl.float8e8m0fnu, bitcast=True)
    sa = tl.load(as_ptrs, mask=valid_m[:, None] & valid_sk[None, :], other=unit_scale)
    sb = tl.load(bs_ptrs, mask=valid_n[:, None] & valid_sk[None, :], other=unit_scale)
    return a, b, sa, sb


@triton.jit
def _gemm_mxfp6_native_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    A_FORMAT: tl.constexpr = "e2m3",
    B_FORMAT: tl.constexpr = "e3m2",
):
    tl.static_assert(A_FORMAT == "e2m3" or A_FORMAT == "e3m2", "unsupported A_FORMAT")
    tl.static_assert(B_FORMAT == "e2m3" or B_FORMAT == "e3m2", "unsupported B_FORMAT")
    # Native scaled-MFMA path (manual 96-byte packed K, 128 logical values): load each dense 96-byte
    # tile, unpack its 128 codes in-kernel, and feed unpacked codes to
    # tl.dot_scaled; the compiler dense-packs per lane for the
    # 16x16x128 / 32x32x64 instruction. No padded storage is used.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sk = tl.arange(0, BLOCK_SIZE_K // 32)
    valid_m = offs_m < M
    valid_n = offs_n < N
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    if SPLIT_K > 1:
        pid_k = tl.program_id(axis=1)
        first_k_tile = pid_k * num_k_tiles // SPLIT_K
        last_k_tile = (pid_k + 1) * num_k_tiles // SPLIT_K
        k_start = first_k_tile * BLOCK_SIZE_K
        num_k_tiles = last_k_tile - first_k_tile
    else:
        k_start = 0

    # Per-segment packed loads: slot j holds dense byte
    # (j // 32) * 24 + (j % 32) (24 packed bytes + 8 bytes the
    # lowering ignores per 32-code segment, matching the kWidth=32
    # fragment). No byte->code unpack / code->byte repack.
    byte_offs = k_start * 3 // 4 + (offs_k // 32) * 24 + (offs_k % 32)
    a_bptrs = a_ptr + offs_m[:, None] * K + byte_offs[None, :]
    b_bptrs = b_ptr + offs_n[None, :] * K + byte_offs[:, None]
    as_ptrs = as_ptr + offs_m[:, None] * (K // 32) + k_start // 32 + offs_sk[None, :]
    bs_ptrs = bs_ptr + offs_n[:, None] * (K // 32) + k_start // 32 + offs_sk[None, :]

    # Manual staging: at most two tiles live (the compiler pipeliner
    # kept three and spilled), so compiler pipelining is off.
    k_offset = k_start
    a_codes, b_codes, sa, sb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                          BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Manual unroll x2: two staged steps per body, plus at most one
    # remainder step; the final tile dots after the loop.
    for _ in tl.range(1, num_k_tiles - 1, 2, num_stages=1):
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    if (num_k_tiles - 1) % 2 == 1:
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)

    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    if SPLIT_K > 1:
        tl.store(c_ptrs + pid_k.to(tl.int64) * M * N, acc, mask=valid_m[:, None] & valid_n[None, :])
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid_m[:, None] & valid_n[None, :])


@triton.jit
def _reduce_split_k(partials_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    splits = tl.arange(0, SPLIT_K)
    partials = tl.load(partials_ptr + splits.to(tl.int64)[:, None] * MN + offsets[None, :], mask=offsets[None, :] < MN,
                       other=0)
    tl.store(out_ptr + offsets, tl.sum(partials, axis=0).to(tl.bfloat16), mask=offsets < MN)


# Interwave 256x256 kernel: four 128x128 quadrants over eight waves, LDS
# double-buffered with direct-to-LDS copies. Each LDS row holds a K tile in the
# per-segment packed form consumed by ``fp6_packed``: 24 data bytes plus the 8
# bytes that follow them for each 32-code segment.
_MMA_LAYOUT = tlx.amd_mfma_layout(4, [16, 16, 128], True, [2, 4])
_COPY_SCALE = tlx.layout(shape=((128, 4), (4, )), stride=((4, 0), (1, )))
_SCALE_A = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
    stride=((4, 8, 16, 32, 1, 2, 0, 0, 64), (128, 256)),
)
_SCALE_B = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, )),
    stride=((4, 8, 16, 32, 1, 2, 64, 128, 0), (256, )),
)
_STORE = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((8, 16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 4096, 8192)),
)
_STORE_F32 = tlx.layout(
    shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    stride=((4, 8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 2048, 4096, 8192)),
)


@triton.jit
def _store_quadrant(acc, C, rows, cols, N: tl.constexpr, SPLIT: tl.constexpr, CACHE: tl.constexpr = None):
    acc = tlx.require_layout(acc, _MMA_LAYOUT)
    if SPLIT:
        result = tlx.require_layout(acc, _STORE_F32, late_address_compute=True)
        tlx.buffer_store(result, C, rows[:, None] * N + cols[None, :], cache=CACHE)
    else:
        # Streaming stores: the BF16 output is never re-read by the kernel.
        result = tlx.require_layout(acc.to(tl.bfloat16), _STORE, late_address_compute=True)
        tlx.buffer_store(result, C, rows[:, None] * N + cols[None, :], cache=".cs")


@triton.jit
def _finish_quadrant(acc, WS, C, rows, cols, M: tl.constexpr, N: tl.constexpr, NPART, own):
    # Add the NPART partials in slot order, using this program's accumulator for
    # its own slot, so the sum does not depend on which program arrived last.
    acc = tlx.require_layout(tlx.require_layout(acc, _MMA_LAYOUT), _STORE_F32, late_address_compute=True)
    offs = tlx.require_layout(rows[:, None] * N + cols[None, :], _STORE_F32)
    if own == 0:
        total = acc
    else:
        total = tlx.buffer_load(WS, offs, cache=".cg", contiguity=4)
    for part in tl.range(1, NPART):
        if part == own:
            total += acc
        else:
            total += tlx.buffer_load(WS + part * M * N, offs, cache=".cg", contiguity=4)
    tlx.buffer_store(total.to(tl.bfloat16), C, offs, cache=".cs")


@triton.jit
def _publish_partial(lock):
    # The release below is issued by one lane and covers only its own wave's
    # stores, and tl.debug_barrier only synchronizes execution. Every wave
    # first waits for its own .wt partial stores, so all of them have reached
    # L2 before the barrier and the release writes them back.
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=s,~{memory}", [], dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    tl.atomic_add(lock + 1, 1, sem="release", scope="gpu")


@triton.jit
def _wait_partials(lock, NPART):
    # Only the last split to arrive waits, and only on splits that have already
    # arrived, so co-residency is not required. Spin relaxed and acquire once:
    # an acquiring poll invalidates L2 on every iteration.
    while tl.atomic_add(lock + 1, 0, sem="relaxed", scope="gpu") < NPART:
        pass
    tl.atomic_add(lock + 1, 0, sem="acquire", scope="gpu")
    # Order every wave's partial loads after the acquire: the atomic is
    # not guaranteed to be executed by (or visible to) all waves, so
    # sync the block before _finish_partials reads the workspace. The
    # publish side drains every wave's .wt partial stores before its
    # barrier and release atomic.
    tl.debug_barrier()


@triton.jit
def _reset_lock(lock):
    tl.atomic_xchg(lock, 0, sem="relaxed", scope="gpu")
    tl.atomic_xchg(lock + 1, 0, sem="relaxed", scope="gpu")


@triton.jit
def _iw_load(dst, src, data_off, sdst, ssrc, scale_off):
    tlx.buffer_load_to_local(dst, src, data_off, contiguity=16)
    tlx.buffer_load_to_local(sdst, ssrc, scale_off)
    tlx.async_load_commit_group()


@triton.jit
def _iw256_segment(A, B, SA, SB, a_top_s, a_bot_s, b_left_s, b_right_s, sa_top_s, sa_bot_s, sb_left_s, sb_right_s, pm,
                   pn, first_tile, n_tiles, K: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr):
    """One 256x256 tile over K tiles ``[first_tile, first_tile + n_tiles)``; ``n_tiles`` is even and >= 2."""
    # K-tile byte step and per-segment byte offsets within a K tile.
    TB: tl.constexpr = 96
    rows = tl.arange(0, 128)
    slots = tl.arange(0, 128)
    data_off = rows[:, None] * K + ((slots // 32) * 24 + slots % 32)[None, :]
    scale_off = tlx.require_layout(rows[:, None] * (K // 32) + tl.arange(0, 4)[None, :], _COPY_SCALE)
    a_base = A + pm * 256 * K + first_tile * TB
    a_bot_base = a_base + 128 * K
    b_base = B + pn * 256 * K + first_tile * TB
    b_right_base = b_base + 128 * K
    sa_base = SA + pm * 256 * (K // 32) + first_tile * 4
    sa_bot_base = sa_base + 128 * (K // 32)
    sb_base = SB + pn * 256 * (K // 32) + first_tile * 4
    sb_right_base = sb_base + 128 * (K // 32)

    acc_tl = tl.zeros((128, 128), tl.float32)
    acc_bl = tl.zeros((128, 128), tl.float32)
    acc_tr = tl.zeros((128, 128), tl.float32)
    acc_br = tl.zeros((128, 128), tl.float32)

    for slot in tl.static_range(2):
        _iw_load(b_left_s[slot], b_base + slot * TB, data_off, sb_left_s[slot], sb_base + slot * 4, scale_off)
        _iw_load(a_top_s[slot], a_base + slot * TB, data_off, sa_top_s[slot], sa_base + slot * 4, scale_off)
        _iw_load(a_bot_s[slot], a_bot_base + slot * TB, data_off, sa_bot_s[slot], sa_bot_base + slot * 4, scale_off)
        _iw_load(b_right_s[slot], b_right_base + slot * TB, data_off, sb_right_s[slot], sb_right_base + slot * 4,
                 scale_off)
    a_base += 2 * TB
    a_bot_base += 2 * TB
    b_base += 2 * TB
    b_right_base += 2 * TB
    sa_base += 8
    sa_bot_base += 8
    sb_base += 8
    sb_right_base += 8

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), relaxed=True)
    sb_l = tlx.local_load(sb_left_s[0], layout=_SCALE_B, relaxed=True)
    a_top = tlx.local_load(a_top_s[0], relaxed=True)
    sa_t = tlx.local_load(sa_top_s[0], layout=_SCALE_A, relaxed=True)

    for _ in tl.range(0, n_tiles - 2, 2, num_stages=1):
        for slot in tl.static_range(2):
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_left, sb_l, B_FORMAT, acc_tl, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(a_bot_s[slot], relaxed=True)
                sa_b = tlx.local_load(sa_bot_s[slot], layout=_SCALE_A, relaxed=True)
                tlx.amd_sched_barrier(0)
                _iw_load(b_left_s[slot], b_base + slot * TB, data_off, sb_left_s[slot], sb_base + slot * 4, scale_off)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_left, sb_l, B_FORMAT, acc_bl, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(b_right_s[slot]), relaxed=True)
                sb_r = tlx.local_load(sb_right_s[slot], layout=_SCALE_B, relaxed=True)
                tlx.amd_sched_barrier(0)
                _iw_load(a_top_s[slot], a_base + slot * TB, data_off, sa_top_s[slot], sa_base + slot * 4, scale_off)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_right, sb_r, B_FORMAT, acc_tr, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(b_left_s[1 - slot]), relaxed=True)
                sb_l = tlx.local_load(sb_left_s[1 - slot], layout=_SCALE_B, relaxed=True)
                tlx.amd_sched_barrier(0)
                _iw_load(a_bot_s[slot], a_bot_base + slot * TB, data_off, sa_bot_s[slot], sa_bot_base + slot * 4,
                         scale_off)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_right, sb_r, B_FORMAT, acc_br, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(a_top_s[1 - slot], relaxed=True)
                sa_t = tlx.local_load(sa_top_s[1 - slot], layout=_SCALE_A, relaxed=True)
                tlx.amd_sched_barrier(0)
                _iw_load(b_right_s[slot], b_right_base + slot * TB, data_off, sb_right_s[slot],
                         sb_right_base + slot * 4, scale_off)
                if slot == 1:
                    a_base += 2 * TB
                    a_bot_base += 2 * TB
                    b_base += 2 * TB
                    b_right_base += 2 * TB
                    sa_base += 8
                    sa_bot_base += 8
                    sb_base += 8
                    sb_right_base += 8

    # Drain: no refills, wait(5) .. wait(0) completes the remaining groups in issue order.
    acc_tl = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_left, sb_l, B_FORMAT, acc_tl, fp6_packed=True)
    tlx.async_load_wait_group(5)
    a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
    sa_b = tlx.local_load(sa_bot_s[0], layout=_SCALE_A, relaxed=True)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_left, sb_l, B_FORMAT, acc_bl, fp6_packed=True)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
    sb_r = tlx.local_load(sb_right_s[0], layout=_SCALE_B, relaxed=True)
    acc_tr = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_right, sb_r, B_FORMAT, acc_tr, fp6_packed=True)
    tlx.async_load_wait_group(3)
    b_left = tlx.local_load(tlx.local_trans(b_left_s[1]), relaxed=True)
    sb_l = tlx.local_load(sb_left_s[1], layout=_SCALE_B, relaxed=True)
    acc_br = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_right, sb_r, B_FORMAT, acc_br, fp6_packed=True)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(a_top_s[1], relaxed=True)
    sa_t = tlx.local_load(sa_top_s[1], layout=_SCALE_A, relaxed=True)
    acc_tl = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_left, sb_l, B_FORMAT, acc_tl, fp6_packed=True)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(a_bot_s[1], relaxed=True)
    sa_b = tlx.local_load(sa_bot_s[1], layout=_SCALE_A, relaxed=True)
    acc_bl = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_left, sb_l, B_FORMAT, acc_bl, fp6_packed=True)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(b_right_s[1]), relaxed=True)
    sb_r = tlx.local_load(sb_right_s[1], layout=_SCALE_B, relaxed=True)
    acc_tr = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_right, sb_r, B_FORMAT, acc_tr, fp6_packed=True)
    acc_br = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_right, sb_r, B_FORMAT, acc_br, fp6_packed=True)

    return acc_tl, acc_bl, acc_tr, acc_br


@triton.jit
def _gemm_mxfp6_interwave(A, B, C, WS, LOCKS, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          K_TILES: tl.constexpr, SPLIT_K: tl.constexpr, FUSED_SPLIT: tl.constexpr,
                          GROUP_M: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr):
    tl.static_assert(K_TILES >= 2)
    # A 16-byte pad per 1024-byte direct-to-LDS chunk spreads the 16 rows of an
    # MFMA tile over distinct 16-byte LDS bank offsets.
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    a_top_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    a_bot_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    b_left_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    b_right_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    sa_top_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sa_bot_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sb_left_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sb_right_s = tlx.local_alloc((128, 4), tl.uint8, 2)

    grid_mn: tl.constexpr = M // 256 * (N // 256)
    program = tl.program_id(0)
    split_id = program // grid_mn
    pid = program % grid_mn
    if grid_mn % 8 == 0:
        # Consecutive programs are dispatched round-robin to the eight XCDs. Give each XCD
        # a contiguous run of tiles so the grouped order below shares panels within one L2.
        pid = (pid % 8) * (grid_mn // 8) + pid // 8
    first_m = pid // (GROUP_M * (N // 256)) * GROUP_M
    group_m = tl.minimum(M // 256 - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * (N // 256)) % group_m
    pn = pid % (GROUP_M * (N // 256)) // group_m
    extra: tl.constexpr = (K // 128) % SPLIT_K
    first_tile = split_id * K_TILES + tl.minimum(split_id, extra)

    main_k_tiles: tl.constexpr = K_TILES - K_TILES % 2
    acc_tl, acc_bl, acc_tr, acc_br = _iw256_segment(A, B, SA, SB, a_top_s, a_bot_s, b_left_s, b_right_s, sa_top_s,
                                                    sa_bot_s, sb_left_s, sb_right_s, pm, pn, first_tile, main_k_tiles,
                                                    K, A_FORMAT, B_FORMAT)

    rows = tl.arange(0, 128)
    slots = tl.arange(0, 128)
    data_off = rows[:, None] * K + ((slots // 32) * 24 + slots % 32)[None, :]
    scale_off = tlx.require_layout(rows[:, None] * (K // 32) + tl.arange(0, 4)[None, :], _COPY_SCALE)
    TB: tl.constexpr = 96
    # Odd tile count, plus one extra tile for each of the first ``extra`` splits.
    tail_tiles = K_TILES % 2 + (split_id < extra).to(tl.int32)
    for t in tl.range(0, tail_tiles, num_stages=1):
        tl.debug_barrier()
        tile = first_tile + main_k_tiles + t
        a_t = A + pm * 256 * K + tile * TB
        b_t = B + pn * 256 * K + tile * TB
        sa_tp = SA + pm * 256 * (K // 32) + tile * 4
        sb_tp = SB + pn * 256 * (K // 32) + tile * 4
        _iw_load(b_left_s[0], b_t, data_off, sb_left_s[0], sb_tp, scale_off)
        _iw_load(a_top_s[0], a_t, data_off, sa_top_s[0], sa_tp, scale_off)
        _iw_load(a_bot_s[0], a_t + 128 * K, data_off, sa_bot_s[0], sa_tp + 128 * (K // 32), scale_off)
        _iw_load(b_right_s[0], b_t + 128 * K, data_off, sb_right_s[0], sb_tp + 128 * (K // 32), scale_off)
        tlx.async_load_wait_group(0)
        b_left = tlx.local_load(tlx.local_trans(b_left_s[0]), relaxed=True)
        sb_l = tlx.local_load(sb_left_s[0], layout=_SCALE_B, relaxed=True)
        a_top = tlx.local_load(a_top_s[0], relaxed=True)
        sa_t = tlx.local_load(sa_top_s[0], layout=_SCALE_A, relaxed=True)
        a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
        sa_b = tlx.local_load(sa_bot_s[0], layout=_SCALE_A, relaxed=True)
        b_right = tlx.local_load(tlx.local_trans(b_right_s[0]), relaxed=True)
        sb_r = tlx.local_load(sb_right_s[0], layout=_SCALE_B, relaxed=True)
        acc_tl = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_left, sb_l, B_FORMAT, acc_tl, fp6_packed=True)
        acc_bl = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_left, sb_l, B_FORMAT, acc_bl, fp6_packed=True)
        acc_tr = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_right, sb_r, B_FORMAT, acc_tr, fp6_packed=True)
        acc_br = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_right, sb_r, B_FORMAT, acc_br, fp6_packed=True)

    cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
    cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
    if FUSED_SPLIT:
        lock = LOCKS + 2 * (pm * (N // 256) + pn)
        arrived = tl.atomic_add(lock, 1, sem="relaxed", scope="gpu")
        if arrived < SPLIT_K - 1:
            ws = WS + split_id.to(tl.int64) * M * N
            _store_quadrant(acc_tl, ws, cm, cn, N, True, ".wt")
            _store_quadrant(acc_bl, ws, cm + 128, cn, N, True, ".wt")
            _store_quadrant(acc_tr, ws, cm, cn + 128, N, True, ".wt")
            _store_quadrant(acc_br, ws, cm + 128, cn + 128, N, True, ".wt")
            _publish_partial(lock)
        else:
            _wait_partials(lock, SPLIT_K - 1)
            _finish_quadrant(acc_tl, WS, C, cm, cn, M, N, SPLIT_K, split_id)
            _finish_quadrant(acc_bl, WS, C, cm + 128, cn, M, N, SPLIT_K, split_id)
            _finish_quadrant(acc_tr, WS, C, cm, cn + 128, M, N, SPLIT_K, split_id)
            _finish_quadrant(acc_br, WS, C, cm + 128, cn + 128, M, N, SPLIT_K, split_id)
            _reset_lock(lock)
    else:
        out = C
        if SPLIT_K > 1:
            out = WS + split_id.to(tl.int64) * M * N
        _store_quadrant(acc_tl, out, cm, cn, N, SPLIT_K > 1)
        _store_quadrant(acc_bl, out, cm + 128, cn, N, SPLIT_K > 1)
        _store_quadrant(acc_tr, out, cm, cn + 128, N, SPLIT_K > 1)
        _store_quadrant(acc_br, out, cm + 128, cn + 128, N, SPLIT_K > 1)


@triton.jit
def _sk_program_of(it, UNITS: tl.constexpr, GRID: tl.constexpr):
    # The Stream-K program whose iteration range holds K-tile iteration ``it``.
    return ((it // 2 + 1) * GRID - 1) // UNITS


@triton.jit
def _gemm_mxfp6_interwave_sk(A, B, C, WS, LOCKS, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                             GRID: tl.constexpr, GROUP_M: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr):
    # Stream-K over 256x256 tiles. Program p owns an equal contiguous range of
    # the (tile, K tile) iterations, in 2-iteration units, and runs it as one
    # segment per tile it touches. Tile pieces are combined by the same
    # arrival-counter fixup as the fused split-K path, one slot per piece.
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    a_top_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    a_bot_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    b_left_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    b_right_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    sa_top_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sa_bot_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sb_left_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sb_right_s = tlx.local_alloc((128, 4), tl.uint8, 2)

    KT: tl.constexpr = K // 128
    UNITS: tl.constexpr = M // 256 * (N // 256) * KT // 2
    program = tl.program_id(0)
    if GRID % 8 == 0:
        program = (program % 8) * (GRID // 8) + program // 8
    it = program * UNITS // GRID * 2
    end = (program + 1) * UNITS // GRID * 2
    while it < end:
        tile = it // KT
        k0 = it - tile * KT
        n_tiles = tl.minimum(KT - k0, end - it)
        first_m = tile // (GROUP_M * (N // 256)) * GROUP_M
        group_m = tl.minimum(M // 256 - first_m, GROUP_M)
        pm = first_m + tile % (GROUP_M * (N // 256)) % group_m
        pn = tile % (GROUP_M * (N // 256)) // group_m
        # The previous segment's last LDS reads must finish before this one refills the slots.
        tl.debug_barrier()
        acc_tl, acc_bl, acc_tr, acc_br = _iw256_segment(A, B, SA, SB, a_top_s, a_bot_s, b_left_s, b_right_s, sa_top_s,
                                                        sa_bot_s, sb_left_s, sb_right_s, pm, pn, k0, n_tiles, K,
                                                        A_FORMAT, B_FORMAT)
        pieces = _sk_program_of(tile * KT + KT - 1, UNITS, GRID) - _sk_program_of(tile * KT, UNITS, GRID) + 1
        cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=n_tiles)
        cn = pn * 256 + tlx.rematerialized_range(0, 128, 11, placement=n_tiles)
        if pieces == 1:
            _store_quadrant(acc_tl, C, cm, cn, N, False)
            _store_quadrant(acc_bl, C, cm + 128, cn, N, False)
            _store_quadrant(acc_tr, C, cm, cn + 128, N, False)
            _store_quadrant(acc_br, C, cm + 128, cn + 128, N, False)
        else:
            lock = LOCKS + 2 * tile
            own = program - _sk_program_of(tile * KT, UNITS, GRID)
            arrived = tl.atomic_add(lock, 1, sem="relaxed", scope="gpu")
            if arrived < pieces - 1:
                ws = WS + own.to(tl.int64) * M * N
                _store_quadrant(acc_tl, ws, cm, cn, N, True, ".wt")
                _store_quadrant(acc_bl, ws, cm + 128, cn, N, True, ".wt")
                _store_quadrant(acc_tr, ws, cm, cn + 128, N, True, ".wt")
                _store_quadrant(acc_br, ws, cm + 128, cn + 128, N, True, ".wt")
                _publish_partial(lock)
            else:
                _wait_partials(lock, pieces - 1)
                _finish_quadrant(acc_tl, WS, C, cm, cn, M, N, pieces, own)
                _finish_quadrant(acc_bl, WS, C, cm + 128, cn, M, N, pieces, own)
                _finish_quadrant(acc_tr, WS, C, cm, cn + 128, M, N, pieces, own)
                _finish_quadrant(acc_br, WS, C, cm + 128, cn + 128, M, N, pieces, own)
                _reset_lock(lock)
        it += n_tiles


@triton.jit
def _gemm_mxfp6_interwave_n128(A, B, C, WS, LOCKS, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               K_TILES: tl.constexpr, SPLIT_K: tl.constexpr, FUSED_SPLIT: tl.constexpr,
                               GROUP_M: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr):
    # 256x128 variant: two 128x128 quadrants share one B tile. Per K-tile slot, B
    # is one async group and the two A halves are another.
    tl.static_assert(K_TILES >= 2)
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [128, 128],
    )
    a_top_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    a_bot_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    b_s = tlx.local_alloc((128, 128), tl.uint8, 2, layout=tile_layout)
    sa_top_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sa_bot_s = tlx.local_alloc((128, 4), tl.uint8, 2)
    sb_s = tlx.local_alloc((128, 4), tl.uint8, 2)

    grid_mn: tl.constexpr = M // 256 * (N // 128)
    program = tl.program_id(0)
    split_id = program // grid_mn
    pid = program % grid_mn
    if grid_mn % 8 == 0:
        pid = (pid % 8) * (grid_mn // 8) + pid // 8
    first_m = pid // (GROUP_M * (N // 128)) * GROUP_M
    group_m = tl.minimum(M // 256 - first_m, GROUP_M)
    pm = first_m + pid % (GROUP_M * (N // 128)) % group_m
    pn = pid % (GROUP_M * (N // 128)) // group_m
    extra: tl.constexpr = (K // 128) % SPLIT_K
    first_tile = split_id * K_TILES + tl.minimum(split_id, extra)

    TB: tl.constexpr = 96
    rows = tl.arange(0, 128)
    slots = tl.arange(0, 128)
    data_off = rows[:, None] * K + ((slots // 32) * 24 + slots % 32)[None, :]
    scale_off = tlx.require_layout(rows[:, None] * (K // 32) + tl.arange(0, 4)[None, :], _COPY_SCALE)
    a_base = A + pm * 256 * K + first_tile * TB
    b_base = B + pn * 128 * K + first_tile * TB
    sa_base = SA + pm * 256 * (K // 32) + first_tile * 4
    sb_base = SB + pn * 128 * (K // 32) + first_tile * 4

    acc_t = tl.zeros((128, 128), tl.float32)
    acc_b = tl.zeros((128, 128), tl.float32)
    main_k_tiles: tl.constexpr = K_TILES - K_TILES % 2

    for slot in tl.static_range(2):
        _iw_load(b_s[slot], b_base + slot * TB, data_off, sb_s[slot], sb_base + slot * 4, scale_off)
        tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * TB, data_off, contiguity=16)
        tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * 4, scale_off)
        _iw_load(a_bot_s[slot], a_base + 128 * K + slot * TB, data_off, sa_bot_s[slot],
                 sa_base + 128 * (K // 32) + slot * 4, scale_off)
    a_base += 2 * TB
    b_base += 2 * TB
    sa_base += 8
    sb_base += 8

    tlx.async_load_wait_group(2)
    b_cur = tlx.local_load(tlx.local_trans(b_s[0]), relaxed=True)
    sb_c = tlx.local_load(sb_s[0], layout=_SCALE_B, relaxed=True)
    a_top = tlx.local_load(a_top_s[0], relaxed=True)
    sa_t = tlx.local_load(sa_top_s[0], layout=_SCALE_A, relaxed=True)

    for _ in tl.range(0, main_k_tiles - 2, 2, num_stages=1):
        for slot in tl.static_range(2):
            tlx.async_load_wait_group(2)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_t = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_t, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(a_bot_s[slot], relaxed=True)
                sa_b = tlx.local_load(sa_bot_s[slot], layout=_SCALE_A, relaxed=True)
                tlx.amd_sched_barrier(0)
                _iw_load(b_s[slot], b_base + slot * TB, data_off, sb_s[slot], sb_base + slot * 4, scale_off)
            tlx.async_load_wait_group(1)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_b = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_b, fp6_packed=True)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_cur = tlx.local_load(tlx.local_trans(b_s[1 - slot]), relaxed=True)
                sb_c = tlx.local_load(sb_s[1 - slot], layout=_SCALE_B, relaxed=True)
                a_top = tlx.local_load(a_top_s[1 - slot], relaxed=True)
                sa_t = tlx.local_load(sa_top_s[1 - slot], layout=_SCALE_A, relaxed=True)
                tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(a_top_s[slot], a_base + slot * TB, data_off, contiguity=16)
                tlx.buffer_load_to_local(sa_top_s[slot], sa_base + slot * 4, scale_off)
                _iw_load(a_bot_s[slot], a_base + 128 * K + slot * TB, data_off, sa_bot_s[slot],
                         sa_base + 128 * (K // 32) + slot * 4, scale_off)
                if slot == 1:
                    a_base += 2 * TB
                    b_base += 2 * TB
                    sa_base += 8
                    sb_base += 8

    # Drain the last two tiles: slot 0 is in registers except A-bottom, slot 1 is in LDS.
    tlx.async_load_wait_group(0)
    acc_t = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_t, fp6_packed=True)
    a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
    sa_b = tlx.local_load(sa_bot_s[0], layout=_SCALE_A, relaxed=True)
    acc_b = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_b, fp6_packed=True)
    b_cur = tlx.local_load(tlx.local_trans(b_s[1]), relaxed=True)
    sb_c = tlx.local_load(sb_s[1], layout=_SCALE_B, relaxed=True)
    a_top = tlx.local_load(a_top_s[1], relaxed=True)
    sa_t = tlx.local_load(sa_top_s[1], layout=_SCALE_A, relaxed=True)
    acc_t = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_t, fp6_packed=True)
    a_bot = tlx.local_load(a_bot_s[1], relaxed=True)
    sa_b = tlx.local_load(sa_bot_s[1], layout=_SCALE_A, relaxed=True)
    acc_b = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_b, fp6_packed=True)

    tail_tiles = K_TILES % 2 + (split_id < extra).to(tl.int32)
    for t in tl.range(0, tail_tiles, num_stages=1):
        tl.debug_barrier()
        tile = first_tile + main_k_tiles + t
        a_t = A + pm * 256 * K + tile * TB
        sa_tp = SA + pm * 256 * (K // 32) + tile * 4
        _iw_load(b_s[0], B + pn * 128 * K + tile * TB, data_off, sb_s[0], SB + pn * 128 * (K // 32) + tile * 4,
                 scale_off)
        _iw_load(a_top_s[0], a_t, data_off, sa_top_s[0], sa_tp, scale_off)
        _iw_load(a_bot_s[0], a_t + 128 * K, data_off, sa_bot_s[0], sa_tp + 128 * (K // 32), scale_off)
        tlx.async_load_wait_group(0)
        b_cur = tlx.local_load(tlx.local_trans(b_s[0]), relaxed=True)
        sb_c = tlx.local_load(sb_s[0], layout=_SCALE_B, relaxed=True)
        a_top = tlx.local_load(a_top_s[0], relaxed=True)
        sa_t = tlx.local_load(sa_top_s[0], layout=_SCALE_A, relaxed=True)
        a_bot = tlx.local_load(a_bot_s[0], relaxed=True)
        sa_b = tlx.local_load(sa_bot_s[0], layout=_SCALE_A, relaxed=True)
        acc_t = tlx.dot_scaled(a_top, sa_t, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_t, fp6_packed=True)
        acc_b = tlx.dot_scaled(a_bot, sa_b, A_FORMAT, b_cur, sb_c, B_FORMAT, acc_b, fp6_packed=True)

    cm = pm * 256 + tlx.rematerialized_range(0, 128, 10, placement=K_TILES)
    cn = pn * 128 + tlx.rematerialized_range(0, 128, 11, placement=K_TILES)
    if FUSED_SPLIT:
        lock = LOCKS + 2 * (pm * (N // 128) + pn)
        arrived = tl.atomic_add(lock, 1, sem="relaxed", scope="gpu")
        if arrived < SPLIT_K - 1:
            ws = WS + split_id.to(tl.int64) * M * N
            _store_quadrant(acc_t, ws, cm, cn, N, True, ".wt")
            _store_quadrant(acc_b, ws, cm + 128, cn, N, True, ".wt")
            _publish_partial(lock)
        else:
            _wait_partials(lock, SPLIT_K - 1)
            _finish_quadrant(acc_t, WS, C, cm, cn, M, N, SPLIT_K, split_id)
            _finish_quadrant(acc_b, WS, C, cm + 128, cn, M, N, SPLIT_K, split_id)
            _reset_lock(lock)
    else:
        out = C
        if SPLIT_K > 1:
            out = WS + split_id.to(tl.int64) * M * N
        _store_quadrant(acc_t, out, cm, cn, N, SPLIT_K > 1)
        _store_quadrant(acc_b, out, cm + 128, cn, N, SPLIT_K > 1)


def _interwave_block_n(M, N, K, split_k):
    """Interwave tile width for this problem, or None when only the native kernel applies."""
    if M % 256 or N % 128 or K % 128 or K // 128 // split_k < 2:
        return None
    # 256x128 tiles when 256x256 tiles would occupy at most half of the CUs.
    cu_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if N % 256 or M // 256 * (N // 256) * split_k <= cu_count // 2:
        return 128
    return 256


# Largest split_k whose partials the last-arriving split reduces in-kernel; wider
# splits reduce in ``_reduce_split_k``, as one CTA summing them all is slower.
_FUSED_SPLIT_K_MAX = 4
# Fewest K-tile iterations per program for which split_k=1 uses Stream-K.
_STREAM_K_MIN_ITERS = 32
# Per-(device, stream) tile arrival counters. The reducing split zeroes its
# tile's counters, so a buffer is reusable by every later launch on its stream.
# A buffer only grows, keeping the largest grid seen on its stream; at 8 bytes
# per tile (8 KiB for 1024 tiles) that is cheaper than reallocating.
_SPLIT_LOCKS = {}


def _split_locks(device, tiles):
    # During stream capture, torch.zeros is only recorded, not executed:
    # caching that buffer would hand a later eager launch uninitialized
    # counters if it runs before the graph is replayed. Captured graphs
    # get a fresh (recorded) zeroing on every replay instead, and only
    # eager allocations enter the reusable cache.
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(2 * tiles, device=device, dtype=torch.int32)
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    locks = _SPLIT_LOCKS.get(key)
    if locks is None or locks.numel() < 2 * tiles:
        locks = torch.zeros(2 * tiles, device=device, dtype=torch.int32)
        _SPLIT_LOCKS[key] = locks
    return locks


def _stream_k_grid(M, N, K, cu_count):
    """Stream-K program count for split_k=1, or None to keep one CTA per tile.

    Stream-K pays when 256x256 tiles leave CUs idle and every program still
    gets enough K iterations to amortize its tile fixups.
    """
    if M % 256 or N % 256 or K % 256:
        return None
    tiles = M // 256 * (N // 256)
    if tiles >= cu_count or tiles * (K // 128) < _STREAM_K_MIN_ITERS * cu_count:
        return None
    return cu_count


def _launch_stream_k(a, a_scale, b, b_scale, out, a_format, b_format, grid, group_m=4):
    M, K = a.shape
    N = b.shape[0]
    tiles, kt = M // 256 * (N // 256), K // 128
    # Program p owns 2 * (floor((p + 1) * U / grid) - floor(p * U / grid)) >=
    # 2 * floor(U / grid) = min_iters iterations (U = tiles * kt // 2 units), so
    # a tile spans at most ceil(kt / min_iters) + 1 programs, each with its own
    # partial slot. _stream_k_grid guarantees min_iters >= _STREAM_K_MIN_ITERS.
    min_iters = tiles * kt // 2 // grid * 2
    workspace = torch.empty((triton.cdiv(kt, min_iters) + 1, M, N), device=a.device, dtype=torch.float32)
    _gemm_mxfp6_interwave_sk[(grid, )](a, b, out, workspace, _split_locks(a.device, tiles), a_scale.view(torch.uint8),
                                       b_scale.view(torch.uint8), M, N, K, grid, group_m, a_format, b_format,
                                       num_warps=8, num_stages=1, matrix_instr_nonkdim=16,
                                       disable_unclustered_high_rp_reschedule=True, reverse_local_assignment=True,
                                       llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ))


def _launch_interwave(a, a_scale, b, b_scale, out, split_k, a_format, b_format, block_n, group_m=4):
    M, K = a.shape
    N = b.shape[0]
    if split_k == 1:
        cu_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        grid = _stream_k_grid(M, N, K, cu_count)
        if grid is not None:
            _launch_stream_k(a, a_scale, b, b_scale, out, a_format, b_format, grid, group_m)
            return
    tiles = M // 256 * (N // block_n)
    fused = 1 < split_k <= _FUSED_SPLIT_K_MAX
    locks = _split_locks(a.device, tiles) if fused else out
    workspace = out
    if split_k > 1:
        workspace = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    kernel = _gemm_mxfp6_interwave if block_n == 256 else _gemm_mxfp6_interwave_n128
    kernel[(tiles * split_k, )](a, b, out, workspace, locks, a_scale.view(torch.uint8), b_scale.view(torch.uint8), M, N,
                                K, K // 128 // split_k, split_k, fused, group_m, a_format, b_format, num_warps=8,
                                num_stages=1, matrix_instr_nonkdim=16, disable_unclustered_high_rp_reschedule=True,
                                reverse_local_assignment=True, llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ))
    if split_k > 1 and not fused:
        _reduce_split_k[(triton.cdiv(M * N, 1024), )](workspace, out, M * N, split_k, 1024)


def _full_configs():
    configs = []
    for block_m, block_n, block_k in (
        (128, 128, 128),
        (128, 256, 128),
        (256, 128, 128),
    ):
        for num_stages in (1, ):
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 8,
                    },
                    num_warps=8,
                    num_stages=num_stages,
                ))
    # 128x128x128 is fastest at 4 warps (the heuristic default).
    configs.append(
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=1,
        ))
    return configs


_kernel_full = triton.autotune(configs=_full_configs(), key=["M", "N", "K", "SPLIT_K", "A_FORMAT",
                                                             "B_FORMAT"])(_gemm_mxfp6_native_kernel)


def _launch_full(a, a_scale, b, b_scale, out, split_k=1, a_format="e2m3", b_format="e3m2"):
    M, K = a.shape
    N = b.shape[0]

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), split_k)

    _kernel_full[grid](a, b, out, a_scale, b_scale, M, N, K, SPLIT_K=split_k, A_FORMAT=a_format, B_FORMAT=b_format)
    return out


def mm_mxfp6(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full", split_k=1, a_format="e2m3",
             b_format="e3m2"):
    """``a @ b.T`` for pre-quantized MXFP6 operands on gfx950, as BF16.

    ``a_format`` and ``b_format`` independently select "e2m3" or "e3m2"
    (defaulting to E2M3 A and E3M2 B), both dense-packed uint8 (groups
    of 16 codes per 12 bytes, K padded to the unpacked width). Inputs
    are validated
    by ``tlx.ops.mm_mxfp6``. ``space`` is "full" (autotune, the
    default) or "heuristic" (one shape-picked config); it applies only
    to shapes the interwave kernel does not cover. Only the "natural"
    scale layout is supported on this backend.

    ``split_k`` partitions whole MFMA K tiles. FP32 partials are
    reduced before the single BF16 conversion. With ``split_k=1``, an
    interwave grid that would leave CUs idle may instead be balanced
    with Stream-K, which splits K at iteration granularity internally.
    """
    if a_format not in ("e2m3", "e3m2") or b_format not in ("e2m3", "e3m2"):
        raise InvalidInput("a_format and b_format must each be 'e2m3' or 'e3m2'")
    if sf_layout != "natural":
        raise InvalidInput(f"gfx950 mm_mxfp6 only supports sf_layout='natural'; got {sf_layout!r}")
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"unsupported space {space!r}; expected 'heuristic' or 'full'")
    M, K = a.shape
    N = b.shape[0]
    block_k = 128 if K % 128 == 0 else 64
    if not isinstance(split_k, int) or split_k not in (1, 2, 4, 8, 16) or split_k > K // block_k:
        raise InvalidInput("split_k must be one of (1, 2, 4, 8, 16) and not exceed the number of K tiles")
    if space == "full" and K % 128:
        raise InvalidInput("space='full' requires K to be a multiple of 128")
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    with torch.cuda.device(a.device):
        block_n = _interwave_block_n(M, N, K, split_k)
        if block_n is not None:
            _launch_interwave(a, a_scale, b, b_scale, out, split_k, a_format, b_format, block_n)
            return out
        target = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32) if split_k > 1 else out
        if space == "heuristic":
            # Default from the MI350 warp sweep: 128x128x128 at 4
            # warps spills zero and beats the 256-wide tile; K=64
            # shapes keep the 64-deep tile (whole 32x32x64 steps).
            meta = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
            }
            grid = (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), split_k)
            _gemm_mxfp6_native_kernel[grid](
                a,
                b,
                target,
                a_scale,
                b_scale,
                M,
                N,
                K,
                BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
                GROUP_SIZE_M=meta["GROUP_SIZE_M"],
                SPLIT_K=split_k,
                A_FORMAT=a_format,
                B_FORMAT=b_format,
                num_warps=meta["num_warps"],
                num_stages=1,
            )
        else:
            _launch_full(a, a_scale, b, b_scale, target, split_k, a_format, b_format)
        if split_k > 1:
            _reduce_split_k[(triton.cdiv(M * N, 1024), )](target, out, M * N, split_k, 1024)
        return out
