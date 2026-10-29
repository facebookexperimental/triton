"""Packed variable-length flash attention forward for sm90 (H100). FP16/BF16, head dim 64 or 128, inference only.

Q/K/V are packed ``(tokens, heads, head_dim)``; sequence ``b`` owns query rows
``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` and key rows
``cu_seqlens_k[b]:cu_seqlens_k[b + 1]``. Causal masking is bottom-right
aligned: query row ``i`` of a sequence sees keys ``j <= i + k_len - q_len``,
and a row that sees no key writes 0.

This is the warp-specialized pingpong forward of ``flash_attn/sm90.py`` made
ragged. A producer warp group loads Q, K and V with TMA; two consumer warp
groups each own 64 rows of a 128-row query tile and alternate their QK GEMMs.
The persistent grid draws ``(sequence, head, query tile)`` items, so Q, K and
V are 2-D TMA descriptors over the packed token axis with the head folded into
the column coordinate.

- Tiles are claimed dynamically: the producer takes the next tile index from a
  global atomic counter, two key blocks before it finishes issuing the current
  tile, and hands it to the consumers through a small shared memory ring. Programs
  that draw short key lengths therefore take more tiles. A tile whose first row
  is past its sequence's end is never handed out.
- A key block that straddles a sequence end reads the next sequence's rows;
  those scores are masked to ``-inf`` before the softmax, so they weigh 0.
- The last query tile of a sequence is written with a row mask, because a TMA
  store would spill into the next sequence's rows.
- The output is a single pointer argument. Full tiles are stored through a TMA
  descriptor that the kernel builds from that pointer, so a traced launch, where
  Inductor copies each mutated argument separately, keeps the TMA store. The
  descriptor is written to a slot of a small buffer the launch passes in rather
  than to Triton's global scratch, so no launch path needs a Triton allocator.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

from ..._holdout import num_sms as _num_sms

BLOCK_M = 128
NUM_MMA_GROUPS = 2
NUM_MMA_WARPS = 8
# The dense forward's profile-selected register split: it avoids consumer
# spills without reducing CTA residency.
CONSUMER_REGISTERS = 240
LOG2E = 1.4426950408889634
NUM_TILE_BUFFERS = 3
CLAIM_AHEAD = 2
TMA_STORE = True


@tl.core.builtin
def _descriptor_slots(ptr, num, _semantic=None):
    """``num`` 128-byte tensor descriptor slots starting at ``ptr``, indexed like ``tlx.allocate_tensor_descriptor``."""
    return tlx.tensor_descriptor_ptr(ptr.handle, tl.core._unwrap_if_constexpr(num), 128)


@triton.jit
def _tile_info(tile_idx, CuQ, CuK, H, kv_group, num_m_blocks, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               CAUSAL: tl.constexpr):
    m_block = tile_idx % num_m_blocks
    bh = tile_idx // num_m_blocks
    off_h = bh % H
    b = bh // H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(CuQ + b + 1) - q_start
    k_start = tl.load(CuK + b)
    k_len = tl.load(CuK + b + 1) - k_start
    hi = k_len
    if CAUSAL:
        hi = tl.maximum(tl.minimum(k_len, (m_block + 1) * BLOCK_M + k_len - q_len), 0)
    return off_h, off_h // kv_group, q_start, q_len, k_start, k_len, m_block, tl.cdiv(hi, BLOCK_N)


@triton.jit
def _mask_scores(qk, kv_idx, k_len, offs_m, causal_off, full_cols, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    # Columns below `full_cols` are visible to every row of the tile, so a block that ends there needs no mask.
    if kv_idx + BLOCK_N > full_cols:
        offs_n = kv_idx + tl.arange(0, BLOCK_N)
        valid = offs_n[None, :] < k_len
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + causal_off)
        qk = tl.where(valid, qk, -float("inf"))
    return qk


@triton.jit
def _softmax_step(qk, m_i, l_i, qk_scale):
    # m_i is in scaled (log2) units. A row with no visible key keeps m = -inf,
    # so the shift is 0 there and its probabilities stay 0 instead of NaN.
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    m_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
    p = tl.math.exp2(qk * qk_scale - m_safe[:, None])
    alpha = tl.math.exp2(m_i - m_safe)
    return m_ij, l_i * alpha + tl.sum(p, 1), p, alpha


@triton.jit
def _publish_tile(tile_idx, tile_id_smem, tile_fulls, tile_empties, accum_cnt, NUM_TILE_BUFFERS: tl.constexpr):
    buf, phase = get_bufidx_phase(accum_cnt, NUM_TILE_BUFFERS)
    tlx.barrier_wait(tile_empties[buf], phase ^ 1)
    tlx.local_store(tile_id_smem[buf], tl.full((1, ), tile_idx, tl.int32))
    tlx.barrier_arrive(tile_fulls[buf], 1)


@triton.jit
def _next_tile(tile_id_smem, tile_fulls, tile_empties, accum_cnt, NUM_TILE_BUFFERS: tl.constexpr):
    buf, phase = get_bufidx_phase(accum_cnt, NUM_TILE_BUFFERS)
    tlx.barrier_wait(tile_fulls[buf], phase)
    tile_idx = tl.sum(tl.reshape(tlx.local_load(tile_id_smem[buf]), (1, )))
    tlx.barrier_arrive(tile_empties[buf], 1)
    return tile_idx


@triton.jit
def _flash_attn_varlen_fwd(
    desc_q,
    desc_k,
    desc_v,
    Out,
    DescSlots,
    CuQ,
    CuK,
    Counter,
    tokens_q,
    stride_ot,
    stride_oh,
    qk_scale,
    H,
    kv_group,
    num_m_blocks,
    total_tiles,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_TILE_BUFFERS: tl.constexpr,
    CLAIM_AHEAD: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    CONSUMER_REGISTERS: tl.constexpr,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    # Pingpong: the consumers alternate QK issue so the tensor core is never contended.
    SERIALIZE_QK: tl.constexpr = CAUSAL or HEAD_DIM == 128
    Q_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q)) * BLOCK_M_SPLIT * HEAD_DIM
    K_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k)) * BLOCK_N * HEAD_DIM
    V_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v)) * BLOCK_N * HEAD_DIM

    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    q_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_MMA_GROUPS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=1,
    )
    k_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_BUFFERS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=NUM_MMA_GROUPS,
    )
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_BUFFERS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=NUM_MMA_GROUPS,
    )
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    tile_id_smem = tlx.local_alloc((1, ), tl.int32, NUM_TILE_BUFFERS)
    tile_fulls = tlx.alloc_barriers(num_barriers=NUM_TILE_BUFFERS, arrive_count=1)
    tile_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_TILE_BUFFERS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=NUM_MMA_GROUPS,
    )

    with tlx.async_tasks(exclusive=True):
        # producer
        with tlx.async_task("default"):
            accum_cnt_kv = 0
            accum_cnt_q = 0
            accum_cnt_tile = 0
            tile_idx = prog_id
            while tile_idx < total_tiles:
                off_h, off_kh, q_start, q_len, k_start, k_len, m_block, n_kv = _tile_info(
                    tile_idx, CuQ, CuK, H, kv_group, num_m_blocks, BLOCK_M, BLOCK_N, CAUSAL)
                if m_block * BLOCK_M < q_len:
                    _publish_tile(tile_idx, tile_id_smem, tile_fulls, tile_empties, accum_cnt_tile, NUM_TILE_BUFFERS)
                    accum_cnt_tile += 1
                next_tile = -1
                if m_block * BLOCK_M < q_len and n_kv > 0:
                    q_phase = accum_cnt_q & 1
                    for cid in tl.static_range(NUM_MMA_GROUPS):
                        tlx.barrier_wait(q_empties[cid], q_phase ^ 1)
                        tlx.barrier_expect_bytes(q_fulls[cid], Q_BYTES)
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[cid],
                            [q_start + m_block * BLOCK_M + cid * BLOCK_M_SPLIT, off_h * HEAD_DIM],
                            q_fulls[cid],
                        )
                    for blk in range(0, n_kv):
                        if blk == tl.maximum(n_kv - CLAIM_AHEAD, 0):
                            next_tile = num_progs + tl.atomic_add(Counter, 1, sem="relaxed")
                        buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                        kv_row = k_start + blk * BLOCK_N
                        tlx.barrier_wait(k_empties[buf], phase ^ 1)
                        tlx.barrier_expect_bytes(k_fulls[buf], K_BYTES)
                        tlx.async_descriptor_load(desc_k, k_tiles[buf], [kv_row, off_kh * HEAD_DIM], k_fulls[buf])
                        tlx.barrier_wait(v_empties[buf], phase ^ 1)
                        tlx.barrier_expect_bytes(v_fulls[buf], V_BYTES)
                        tlx.async_descriptor_load(desc_v, v_tiles[buf], [kv_row, off_kh * HEAD_DIM], v_fulls[buf])
                        accum_cnt_kv += 1
                    accum_cnt_q += 1
                if next_tile < 0:
                    next_tile = num_progs + tl.atomic_add(Counter, 1, sem="relaxed")
                tile_idx = next_tile
            _publish_tile(-1, tile_id_smem, tile_fulls, tile_empties, accum_cnt_tile, NUM_TILE_BUFFERS)

        # consumers
        with tlx.async_task(
                num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
                registers=CONSUMER_REGISTERS,
                replicate=NUM_MMA_GROUPS,
        ):
            accum_cnt_kv = 0
            accum_cnt_q = 0
            accum_cnt_tile = 0
            cid: tl.constexpr = tlx.async_task_replica_id()
            if TMA_STORE:
                slots = _descriptor_slots(DescSlots + prog_id * NUM_MMA_GROUPS * 128, NUM_MMA_GROUPS)
                desc_o = tlx.make_tensor_descriptor(slots[cid], Out, shape=[tokens_q, H * HEAD_DIM],
                                                    strides=[stride_ot, 1], block_shape=[BLOCK_M_SPLIT, HEAD_DIM])

            # Bootstrap the pingpong sequence once before the persistent tile loop.
            if SERIALIZE_QK and cid == 1:
                tlx.named_barrier_arrive(9, 256)

            tile_idx = _next_tile(tile_id_smem, tile_fulls, tile_empties, accum_cnt_tile, NUM_TILE_BUFFERS)
            accum_cnt_tile += 1
            while tile_idx >= 0:
                off_h, off_kh, q_start, q_len, k_start, k_len, m_block, n_kv = _tile_info(
                    tile_idx, CuQ, CuK, H, kv_group, num_m_blocks, BLOCK_M, BLOCK_N, CAUSAL)
                offs_m = m_block * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                o_ptrs = (Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh +
                          tl.arange(0, HEAD_DIM)[None, :])
                row_ok = (offs_m < q_len)[:, None]
                if n_kv == 0:
                    zeros = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=Out.dtype.element_ty)
                    tl.store(o_ptrs, zeros, mask=row_ok)
                else:
                    causal_off = k_len - q_len
                    full_cols = k_len
                    if CAUSAL:
                        full_cols = tl.minimum(k_len, m_block * BLOCK_M + causal_off + 1)
                    m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                    l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32)
                    acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

                    # wait for this tile's Q, then for K[0]
                    q_phase = accum_cnt_q & 1
                    tlx.barrier_wait(q_fulls[cid], q_phase)
                    k_buf_id, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                    tlx.barrier_wait(k_fulls[k_buf_id], k_phase)

                    # -- qk[0] --
                    k_tile = tlx.local_trans(k_tiles[k_buf_id])
                    if SERIALIZE_QK:
                        if cid == 0:
                            tlx.named_barrier_wait(9, 256)
                        else:
                            tlx.named_barrier_wait(10, 256)
                    qk = tlx.async_dot(q_tiles[cid], k_tile)
                    if SERIALIZE_QK:
                        if cid == 0:
                            tlx.named_barrier_arrive(10, 256)
                        else:
                            tlx.named_barrier_arrive(9, 256)
                    qk = tlx.async_dot_wait(0, qk)
                    qk = _mask_scores(qk, 0, k_len, offs_m, causal_off, full_cols, BLOCK_N, CAUSAL)
                    tlx.barrier_arrive(k_empties[k_buf_id], 1)
                    # A single key block has finished its last QK read of Q.
                    if n_kv == 1:
                        tlx.barrier_arrive(q_empties[cid], 1)
                    m_i, l_i, p, alpha = _softmax_step(qk, m_i, l_i, qk_scale)
                    accum_cnt_kv += 1

                    for blk in range(1, n_kv):
                        k_buf_id, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                        tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                        k_tile = tlx.local_trans(k_tiles[k_buf_id])
                        if SERIALIZE_QK:
                            if cid == 0:
                                tlx.named_barrier_wait(9, 256)
                            else:
                                tlx.named_barrier_wait(10, 256)
                        qk = tlx.async_dot(q_tiles[cid], k_tile)
                        if SERIALIZE_QK:
                            if cid == 0:
                                tlx.named_barrier_arrive(10, 256)
                            else:
                                tlx.named_barrier_arrive(9, 256)

                        # pv of the previous block overlaps this block's qk
                        v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS)
                        tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                        acc = tlx.async_dot(p.to(tlx.dtype_of(desc_k)), v_tiles[v_buf_id], acc)

                        qk = tlx.async_dot_wait(1, qk)
                        qk = _mask_scores(qk, blk * BLOCK_N, k_len, offs_m, causal_off, full_cols, BLOCK_N, CAUSAL)
                        tlx.barrier_arrive(k_empties[k_buf_id], 1)
                        m_i, l_i, p, alpha = _softmax_step(qk, m_i, l_i, qk_scale)

                        acc = tlx.async_dot_wait(0, acc)
                        tlx.barrier_arrive(v_empties[v_buf_id], 1)
                        acc = acc * alpha[:, None]
                        accum_cnt_kv += 1

                    # pv of the last block
                    v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS)
                    tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                    acc = tlx.async_dot(p.to(tlx.dtype_of(desc_k)), v_tiles[v_buf_id], acc)
                    acc = tlx.async_dot_wait(1, acc)
                    if n_kv != 1:
                        tlx.barrier_arrive(q_empties[cid], 1)
                    acc = tlx.async_dot_wait(0, acc)
                    tlx.barrier_arrive(v_empties[v_buf_id], 1)
                    accum_cnt_q += 1

                    # A row that sees no key has l_i = 0 and writes 0.
                    inv_l = 1.0 / tl.where(l_i == 0.0, 1.0, l_i)
                    o_tile = (acc * inv_l[:, None]).to(Out.dtype.element_ty)
                    if TMA_STORE and m_block * BLOCK_M + (cid + 1) * BLOCK_M_SPLIT <= q_len:
                        desc_o.store([q_start + m_block * BLOCK_M + cid * BLOCK_M_SPLIT, off_h * HEAD_DIM], o_tile)
                    else:
                        tl.store(o_ptrs, o_tile, mask=row_ok)
                tile_idx = _next_tile(tile_id_smem, tile_fulls, tile_empties, accum_cnt_tile, NUM_TILE_BUFFERS)
                accum_cnt_tile += 1


def _has_address(x):
    try:
        x.data_ptr()
    except RuntimeError:
        return False
    return True


def _unaligned(x):
    if _has_address(x):
        return x.data_ptr() % 16 != 0
    # A traced tensor has no address, but its storage offset still places it. Inductor realigns a misaligned
    # runtime input only when the traced one was aligned, so an unaligned traced input must be copied here.
    return (x.storage_offset() * x.element_size()) % 16 != 0


def _rows(x):
    """``x`` as a TMA-addressable ``(tokens, heads * head_dim)`` matrix: heads dense in a row, rows 16-byte aligned."""
    tokens, heads, head_dim = x.shape
    if (heads > 1 and x.stride(1) != head_dim) or x.stride(2) != 1 or (x.stride(0) * x.element_size()) % 16 != 0:
        x = x.contiguous()
    if _unaligned(x):
        # `contiguous()` returns a contiguous input as is, so only a copy moves the base.
        x = x.clone()
    return x.view(tokens, heads * head_dim)


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, sm_scale, causal):
    """Launch the forward; inputs are validated by ``tlx.ops.flash_attn_varlen``.

    The launch goes through ``wrap_triton`` and every host descriptor is built with ``from_tensor`` over a 2-D view,
    so ``torch.compile`` and AOTInductor can trace it. The output descriptor is built inside the kernel from the
    output pointer, in a descriptor slot this launch allocates, so it needs no Triton allocator.
    """
    if isinstance(q, FakeTensor):
        return torch.empty_like(q)
    tokens_q, heads, head_dim = q.shape
    kv_heads = k.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    num_m_blocks = triton.cdiv(max_seqlen_q, BLOCK_M)
    total_tiles = batch * heads * num_m_blocks
    if tokens_q == 0 or total_tiles == 0:
        return out
    if k.shape[0] == 0:
        return out.zero_()
    block_n = 128
    num_buffers = 3 if head_dim == 64 and not causal else 2
    split = BLOCK_M // NUM_MMA_GROUPS
    out_rows = out.view(tokens_q, heads * head_dim)
    desc_q = TensorDescriptor.from_tensor(_rows(q), [split, head_dim])
    desc_k = TensorDescriptor.from_tensor(_rows(k), [block_n, head_dim])
    desc_v = TensorDescriptor.from_tensor(_rows(v), [block_n, head_dim])
    counter = torch.zeros(1, device=q.device, dtype=torch.int32)
    # AOTI captures the build device's SM count. On smaller GPUs, queued CTAs own their first tiles until scheduled.
    num_sms = _num_sms(q.device)
    desc_slots = torch.empty(num_sms * NUM_MMA_GROUPS * 128, device=q.device, dtype=torch.int8)
    kernel = wrap_triton(_flash_attn_varlen_fwd)[(num_sms, )]
    with torch.cuda.device(q.device):
        kernel(desc_q, desc_k, desc_v, out_rows, desc_slots, cu_seqlens_q, cu_seqlens_k, counter, tokens_q,
               out_rows.stride(0), head_dim, sm_scale * LOG2E, heads, heads // kv_heads, num_m_blocks, total_tiles,
               HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=block_n, CAUSAL=causal, TMA_STORE=TMA_STORE,
               NUM_BUFFERS=num_buffers, NUM_TILE_BUFFERS=NUM_TILE_BUFFERS, CLAIM_AHEAD=CLAIM_AHEAD,
               NUM_MMA_WARPS=NUM_MMA_WARPS, NUM_MMA_GROUPS=NUM_MMA_GROUPS, CONSUMER_REGISTERS=CONSUMER_REGISTERS,
               num_warps=4, num_stages=1)
    return out
