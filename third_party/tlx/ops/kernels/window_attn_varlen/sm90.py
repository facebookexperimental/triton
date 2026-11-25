"""Sliding-window self-attention over packed variable-length sequences, forward, for sm90 (H100). FP16/BF16, head dim
64 or 128, inference only.

Q/K/V are packed ``(tokens, heads, head_dim)``; sequence ``b`` owns rows ``cu_seqlens[b]:cu_seqlens[b + 1]``, and
row ``i`` attends the keys ``i - left .. i + right`` of its own sequence, both ends inclusive.

Work is tiled over the packed token axis, not per sequence: a tile is 128 consecutive rows, which may span sequence
boundaries. Its keys all lie in rows ``r0 - left .. r0 + 127 + right``, so the tile reads that band and masks each
score to its row's window clipped to its row's sequence. Every tile then costs the same, there are no partial tiles
at sequence ends, and the schedule needs no sequence lookup.

- A persistent program per SM walks tiles ``program_id, program_id + num_programs, ...``. A producer warp group loads
  with TMA the two 64-row query blocks of a tile, into a ring of two tiles, and its key band, in 64-row K and V blocks,
  into rings of shared memory buffers; it never waits on the consumers except for a free buffer, so it runs ahead into
  the next tile.
- Two consumer warp groups each own 64 rows of the tile. A consumer computes only the key blocks its rows can see,
  known at compile time from the window; it still waits for and releases the others, so both consumers walk the
  same ring.
- Rows look up their sequence by counting sequence starts at or before them, so a row never sees a key of another
  sequence, and every real row sees at least itself.
- Q, K, V and the output reach the kernels as pointers. A three-program kernel launched first builds the Q, K and V
  TMA descriptors on the GPU, one per program, in a small buffer the launch passes in; the attention kernel only reads
  them, so its producer starts loading at once instead of first building and fencing three descriptors. Each consumer
  builds its output descriptor in its own slot of that buffer, off the producer's path, which also keeps the output
  write visible to Inductor's analysis of the kernel. No launch path needs a host descriptor or a Triton allocator.
  Bands that run past either end of the token axis are zero-filled by TMA and masked; the output store is clipped to
  the token axis.
- The running maximum is kept in unscaled score units, and the float32 ``sm_scale * log2(e)`` multiplies only the
  differences from it, which are at most 0: no finite scale overflows to ``inf - inf``. A scale below the smallest
  normal float32 is treated as 0 by a separate specialization that gives every visible key weight 1.
"""

import functools
import struct

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import guard_or_false

BLOCK_M = 128
BLOCK_N = 64
NUM_MMA_GROUPS = 2
NUM_MMA_WARPS = 8
CONSUMER_REGISTERS = 240
NUM_BUFFERS = 4
NUM_Q_BUFFERS = 2
LOG2E = 1.4426950408889634
# Smallest normal float32: a smaller ``sm_scale * log2(e)`` selects the zero-scale specialization.
MIN_QK_SCALE = 1.1754943508222875e-38
# Slots of the descriptor buffer: the query, key and value descriptors shared by all programs, a marker slot, then one
# output descriptor per consumer of each program.
NUM_LOAD_DESCS = 3
DESC_BYTES = 128
OUT_DESC_SLOT = NUM_LOAD_DESCS + 1


@triton.jit
def _row_key_range(Cu, num_seqs, rows, LEFT: tl.constexpr, RIGHT: tl.constexpr):
    """First and last key of each row: its window clipped to its sequence. A row's sequence is the number of sequence
    starts after the first that are at or before it, counted 32 starts per load."""
    SEQ_CHUNK: tl.constexpr = 32
    j = tl.arange(0, SEQ_CHUNK)
    b = tl.zeros(rows.shape, dtype=tl.int32)
    for c in range(1, num_seqs, SEQ_CHUNK):
        starts = tl.load(Cu + c + j, mask=c + j < num_seqs, other=2147483647)
        b += tl.sum((starts[None, :] <= rows[:, None]).to(tl.int32), 1)
    b = tl.minimum(b, num_seqs - 1)
    lo = tl.maximum(tl.load(Cu + b), rows - LEFT)
    hi = tl.minimum(tl.load(Cu + b + 1) - 1, rows + RIGHT)
    return lo, hi


@triton.jit
def _window_attn_varlen_descs(Q, K, V, DescSlots, tokens, stride_qt, stride_kt, stride_vt, H: tl.constexpr,
                              KV_GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M_SPLIT: tl.constexpr,
                              BLOCK_N: tl.constexpr, NUM_LOAD_DESCS: tl.constexpr, DESC_BYTES: tl.constexpr):
    """Program ``i`` builds descriptor ``i`` of Q, K and V into ``DescSlots``."""
    which = tl.program_id(0)
    # Inductor's analysis of a user kernel does not count descriptor creation as a write. This plain store into the
    # marker slot shows that the kernel writes DescSlots, so a compiled graph keeps the launch and orders it before the
    # attention kernel.
    if which == 0:
        tl.store(DescSlots + NUM_LOAD_DESCS * DESC_BYTES, 1)
    tokens = tl.cast(tokens, tl.int32)
    slots = tlx.tensor_descriptor_ptr(DescSlots.handle, NUM_LOAD_DESCS, DESC_BYTES)
    if which == 0:
        tlx.make_tensor_descriptor(slots[0], Q, shape=[tokens, H * HEAD_DIM], strides=[stride_qt, 1],
                                   block_shape=[BLOCK_M_SPLIT, HEAD_DIM])
    elif which == 1:
        tlx.make_tensor_descriptor(slots[1], K, shape=[tokens, H // KV_GROUP * HEAD_DIM], strides=[stride_kt, 1],
                                   block_shape=[BLOCK_N, HEAD_DIM])
    else:
        tlx.make_tensor_descriptor(slots[2], V, shape=[tokens, H // KV_GROUP * HEAD_DIM], strides=[stride_vt, 1],
                                   block_shape=[BLOCK_N, HEAD_DIM])


@triton.jit
def _window_attn_varlen_fwd(
    Q,
    K,
    V,
    Out,
    DescSlots,
    Cu,
    tokens,
    num_seqs,
    stride_ot,
    qk_scale,
    H: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_Q_BUFFERS: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    CONSUMER_REGISTERS: tl.constexpr,
    OUT_DESC_SLOT: tl.constexpr,
    NUM_LOAD_DESCS: tl.constexpr,
    DESC_BYTES: tl.constexpr,
    ZERO_SCALE: tl.constexpr,
):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    # Key blocks of a tile's band, rows r0 - LEFT .. r0 + BLOCK_M - 1 + RIGHT.
    N_KV: tl.constexpr = (BLOCK_M + LEFT + RIGHT + BLOCK_N - 1) // BLOCK_N
    Q_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(Q)) * BLOCK_M_SPLIT * HEAD_DIM
    KV_BYTES: tl.constexpr = tlx.size_of(tlx.dtype_of(K)) * BLOCK_N * HEAD_DIM
    # With a data-dependent token count Inductor passes it as int64; TMA coordinates must be int32.
    tokens = tl.cast(tokens, tl.int32)
    num_tiles = tl.cdiv(tokens, BLOCK_M) * H
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(Q), NUM_Q_BUFFERS * NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), NUM_BUFFERS)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(Out), NUM_MMA_GROUPS)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_Q_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    q_empties = tlx.alloc_warp_barrier(num_barriers=NUM_Q_BUFFERS * NUM_MMA_GROUPS,
                                       num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, num_arrivals=1)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    k_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS, num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
                                       num_arrivals=NUM_MMA_GROUPS)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS, num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
                                       num_arrivals=NUM_MMA_GROUPS)

    with tlx.async_tasks(exclusive=True):
        # producer
        with tlx.async_task("default"):
            slots = tlx.tensor_descriptor_ptr(DescSlots.handle, NUM_LOAD_DESCS, DESC_BYTES)
            desc_q = tlx.reinterpret_tensor_descriptor(slots[0], [BLOCK_M_SPLIT, HEAD_DIM], tlx.dtype_of(Q))
            desc_k = tlx.reinterpret_tensor_descriptor(slots[1], [BLOCK_N, HEAD_DIM], tlx.dtype_of(K))
            desc_v = tlx.reinterpret_tensor_descriptor(slots[2], [BLOCK_N, HEAD_DIM], tlx.dtype_of(V))
            cnt_q = 0
            cnt_kv = 0
            for tile in range(prog_id, num_tiles, num_progs):
                r0 = (tile // H) * BLOCK_M
                off_h = tile % H
                q_buf = (cnt_q % NUM_Q_BUFFERS) * NUM_MMA_GROUPS
                q_phase = (cnt_q // NUM_Q_BUFFERS) & 1
                for cid in tl.static_range(NUM_MMA_GROUPS):
                    tlx.barrier_wait(q_empties[q_buf + cid], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf + cid], Q_BYTES)
                    tlx.async_descriptor_load(desc_q, q_tiles[q_buf + cid],
                                              [r0 + cid * BLOCK_M_SPLIT, off_h * HEAD_DIM], q_fulls[q_buf + cid])
                cnt_q += 1
                for j in tl.static_range(N_KV):
                    buf = cnt_kv % NUM_BUFFERS
                    phase = (cnt_kv // NUM_BUFFERS) & 1
                    row = r0 - LEFT + j * BLOCK_N
                    tlx.barrier_wait(k_empties[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(k_fulls[buf], KV_BYTES)
                    tlx.async_descriptor_load(desc_k, k_tiles[buf], [row, off_h // KV_GROUP * HEAD_DIM], k_fulls[buf])
                    tlx.barrier_wait(v_empties[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(v_fulls[buf], KV_BYTES)
                    tlx.async_descriptor_load(desc_v, v_tiles[buf], [row, off_h // KV_GROUP * HEAD_DIM], v_fulls[buf])
                    cnt_kv += 1

        # consumers
        with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=CONSUMER_REGISTERS,
                            replicate=NUM_MMA_GROUPS):
            cid: tl.constexpr = tlx.async_task_replica_id()
            out_slots = tlx.tensor_descriptor_ptr(
                (DescSlots + (OUT_DESC_SLOT + prog_id * NUM_MMA_GROUPS) * DESC_BYTES).handle, NUM_MMA_GROUPS,
                DESC_BYTES)
            desc_o = tlx.make_tensor_descriptor(out_slots[cid], Out, shape=[tokens, H * HEAD_DIM],
                                                strides=[stride_ot, 1], block_shape=[BLOCK_M_SPLIT, HEAD_DIM])
            cnt_q = 0
            cnt_kv = 0
            for tile in range(prog_id, num_tiles, num_progs):
                r0 = (tile // H) * BLOCK_M
                off_h = tile % H
                rows = r0 + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                lo, hi = _row_key_range(Cu, num_seqs, rows, LEFT, RIGHT)
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32)
                acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
                q_buf = (cnt_q % NUM_Q_BUFFERS) * NUM_MMA_GROUPS + cid
                tlx.barrier_wait(q_fulls[q_buf], (cnt_q // NUM_Q_BUFFERS) & 1)
                for j in tl.static_range(N_KV):
                    buf = cnt_kv % NUM_BUFFERS
                    phase = (cnt_kv // NUM_BUFFERS) & 1
                    tlx.barrier_wait(k_fulls[buf], phase)
                    # Key block j holds keys this consumer's rows can see (known at compile time).
                    if ((j + 1) * BLOCK_N > cid * BLOCK_M_SPLIT) and (j * BLOCK_N <= cid * BLOCK_M_SPLIT +
                                                                      BLOCK_M_SPLIT - 1 + LEFT + RIGHT):
                        qk = tlx.async_dot(q_tiles[q_buf], tlx.local_trans(k_tiles[buf]))
                        qk = tlx.async_dot_wait(0, qk)
                    tlx.barrier_arrive(k_empties[buf], 1)
                    tlx.barrier_wait(v_fulls[buf], phase)
                    if ((j + 1) * BLOCK_N > cid * BLOCK_M_SPLIT) and (j * BLOCK_N <= cid * BLOCK_M_SPLIT +
                                                                      BLOCK_M_SPLIT - 1 + LEFT + RIGHT):
                        cols = r0 - LEFT + j * BLOCK_N + tl.arange(0, BLOCK_N)
                        visible = (cols[None, :] >= lo[:, None]) & (cols[None, :] <= hi[:, None])
                        qk = tl.where(visible, qk, -float("inf"))
                        if ZERO_SCALE:
                            # Every visible key weighs 1. The weights are derived from the scores, not from the mask,
                            # so they keep the wgmma accumulator layout and reach the PV dot without a shared-memory
                            # conversion.
                            p = tl.where(qk == -float("inf"), 0.0, 1.0)
                            l_i = l_i + tl.sum(p, 1)
                        else:
                            # m_i is the running maximum of the unscaled scores. A row that sees no key in a block (a
                            # padding row past the last token may see none at all) keeps m = -inf, so the shift is 0
                            # and its weights stay 0.
                            m_ij = tl.maximum(m_i, tl.max(qk, 1))
                            m_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
                            p = tl.math.exp2((qk - m_safe[:, None]) * qk_scale)
                            alpha = tl.math.exp2((m_i - m_safe) * qk_scale)
                            l_i = l_i * alpha + tl.sum(p, 1)
                            acc = acc * alpha[:, None]
                            m_i = m_ij
                        acc = tlx.async_dot(p.to(tlx.dtype_of(V)), v_tiles[buf], acc)
                        acc = tlx.async_dot_wait(0, acc)
                    tlx.barrier_arrive(v_empties[buf], 1)
                    cnt_kv += 1
                tlx.barrier_arrive(q_empties[q_buf], 1)
                cnt_q += 1
                o_tile = (acc * (1.0 / tl.where(l_i == 0.0, 1.0, l_i))[:, None]).to(tlx.dtype_of(Out))
                # The previous tile's store has had a whole tile to finish reading this buffer.
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(o_tiles[cid], o_tile)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(desc_o, o_tiles[cid], [r0 + cid * BLOCK_M_SPLIT, off_h * HEAD_DIM])
            tlx.async_descriptor_store_wait(0)


@functools.lru_cache(maxsize=None)
def _num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _has_address(x):
    try:
        x.data_ptr()
    except RuntimeError:
        return False
    return True


def _unaligned(x):
    if _has_address(x):
        return x.data_ptr() % 16 != 0
    # A traced tensor has no address, but its storage offset still places it.
    return (x.storage_offset() * x.element_size()) % 16 != 0


def _rows(x):
    """``x`` with heads dense in a row and 16-byte aligned rows, as its TMA descriptor needs."""
    _, heads, head_dim = x.shape
    if (heads > 1 and x.stride(1) != head_dim) or x.stride(2) != 1 or (x.stride(0) * x.element_size()) % 16 != 0:
        x = x.contiguous()
    if _unaligned(x):
        # `contiguous()` returns a contiguous input as is, so only a copy moves the base.
        x = x.clone()
    return x


def window_attn_varlen(q, k, v, cu_seqlens, window_size, sm_scale):
    """Launch the forward; inputs are validated by ``tlx.ops.window_attn_varlen``.

    The launch goes through ``wrap_triton`` with tensor and scalar arguments only, so ``torch.compile`` and
    AOTInductor can trace it.
    """
    if isinstance(q, FakeTensor):
        return torch.empty(q.shape, device=q.device, dtype=q.dtype)
    tokens, heads, head_dim = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if guard_or_false(tokens == 0):
        return out
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # The float32 value of sm_scale * log2(e) that the kernel multiplies score differences by.
    qk_scale = struct.unpack("f", struct.pack("f", sm_scale * LOG2E))[0]
    zero_scale = qk_scale < MIN_QK_SCALE
    q, k, v = _rows(q), _rows(k), _rows(v)
    cu_seqlens = cu_seqlens.contiguous()
    num_sms = _num_sms(q.device)
    kv_group = heads // k.shape[1]
    # AOTI captures the build device's SM count; on a smaller GPU the extra programs queue.
    desc_slots = torch.empty((OUT_DESC_SLOT + num_sms * NUM_MMA_GROUPS) * DESC_BYTES, device=q.device, dtype=torch.int8)
    with torch.cuda.device(q.device):
        wrap_triton(_window_attn_varlen_descs)[(NUM_LOAD_DESCS, )](q, k, v, desc_slots, tokens, q.stride(0),
                                                                   k.stride(0), v.stride(0), H=heads, KV_GROUP=kv_group,
                                                                   HEAD_DIM=head_dim,
                                                                   BLOCK_M_SPLIT=BLOCK_M // NUM_MMA_GROUPS,
                                                                   BLOCK_N=BLOCK_N, NUM_LOAD_DESCS=NUM_LOAD_DESCS,
                                                                   DESC_BYTES=DESC_BYTES, num_warps=1)
        wrap_triton(_window_attn_varlen_fwd)[(num_sms, )](
            q, k, v, out, desc_slots, cu_seqlens, tokens, cu_seqlens.shape[0] - 1, heads * head_dim,
            1.0 if zero_scale else qk_scale, H=heads, KV_GROUP=kv_group, HEAD_DIM=head_dim, LEFT=window_size[0],
            RIGHT=window_size[1], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_BUFFERS=NUM_BUFFERS,
            NUM_Q_BUFFERS=NUM_Q_BUFFERS, NUM_MMA_WARPS=NUM_MMA_WARPS, NUM_MMA_GROUPS=NUM_MMA_GROUPS,
            CONSUMER_REGISTERS=CONSUMER_REGISTERS, OUT_DESC_SLOT=OUT_DESC_SLOT, NUM_LOAD_DESCS=NUM_LOAD_DESCS,
            DESC_BYTES=DESC_BYTES, ZERO_SCALE=zero_scale, num_warps=4, num_stages=1)
    return out
