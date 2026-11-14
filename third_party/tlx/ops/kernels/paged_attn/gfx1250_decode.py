"""BF16 page64 GQA decode for gfx1250.

One CTA handles a sequence, KV head and context partition. Two TDM stages
feed explicitly laid out WMMA operands. Split reduction retains FP32 raw
numerators, maxima and denominators, including for empty partitions.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


@triton.jit
def _dot(a, b, c, LAYOUT: tl.constexpr):
    a = tlx.require_layout(a, tlx.dot_operand_layout(0, LAYOUT, k_width=8))
    b = tlx.require_layout(b, tlx.dot_operand_layout(1, LAYOUT, k_width=8))
    c = tlx.require_layout(c, LAYOUT)
    return tl.dot(a, b, c)


@triton.jit
def _load_packed_operand(buffer, slot, LAYOUT: tl.constexpr):
    # View the physical [8, 64, 8] page as a matrix before loading it.
    # Keeping the load in the WMMA operand layout avoids a live register copy.
    view = tlx.local_reshape(tlx.local_view(buffer, slot), (8, 64, 8))
    view = tlx.local_reshape(tlx.local_trans(view, (0, 2, 1)), (64, 64))
    return tlx.local_load(view, layout=tlx.dot_operand_layout(1, LAYOUT, k_width=8), relaxed=True)


@triton.jit
def _partition(Q, K, V, Table, Lengths, Part, Stats, Out, QLEN: tl.constexpr, TS: tl.constexpr, NS: tl.constexpr,
               BN: tl.constexpr, BM: tl.constexpr, WINDOW: tl.constexpr, PACKED: tl.constexpr, LAYOUT: tl.constexpr,
               SCALE: tl.constexpr, NP: tl.constexpr, REG: tl.constexpr, KH: tl.constexpr, ROTATE: tl.constexpr,
               BATCH: tl.constexpr, HEAD_FIRST: tl.constexpr):
    b = tl.program_id(1) if HEAD_FIRST else tl.program_id(0)
    h = tl.program_id(0) if HEAD_FIRST else tl.program_id(1)
    s = tl.program_id(2)
    # Each (head, split) rotates a complete permutation of the batch.
    # Output/workspace addresses always use the resulting logical sequence.
    if ROTATE:
        b = (b + 17 * h + 3 * s) % BATCH
    length = tl.load(Lengths + b)
    start = tl.maximum(length - QLEN - WINDOW + 1, 0) // 64 * 64 if WINDOW else 0
    tiles = tl.cdiv(length - start, BN)
    step = tl.cdiv(tiles, NS)
    first = start + s * step * BN
    end = tl.minimum(length, first + step * BN)
    m = tl.arange(0, BM)
    d = tl.arange(0, 64)
    n = tl.arange(0, BN)
    qoff = ((b.to(tl.int64) * QLEN + m // 8) * (KH * 8) + h * 8 + m % 8)[:, None] * 64 + d[None, :]
    q = tl.load(Q + qoff, m[:, None] < QLEN * 8, 0)
    maximum = tl.full((BM, ), -float('inf'), tl.float32)
    denom = tl.zeros((BM, ), tl.float32)
    acc = tl.zeros((BM, 64), tl.float32)
    if PACKED:
        kd = tl.make_tensor_descriptor(K, [NP * KH * 8, 512], [512, 1], [8, 512])
        vd = tl.make_tensor_descriptor(V, [NP * KH * 8, 512], [512, 1], [8, 512])
        kb = tlx.local_alloc((8, 512), tl.bfloat16, 2)
        vb = tlx.local_alloc((8, 512), tl.bfloat16, 2)
    else:
        kd = tl.make_tensor_descriptor(K, [NP * KH * 64, 64], [64, 1], [64, 64])
        vd = tl.make_tensor_descriptor(V, [NP * KH * 64, 64], [64, 1], [64, 64])
        kb = tlx.local_alloc((64, 64), tl.bfloat16, 2)
        vb = tlx.local_alloc((64, 64), tl.bfloat16, 2)
    # Packed pages are streamed once per partition; avoid retaining them in
    # preference to the query, page table and reduction workspace.
    if first < end:
        physical = tl.load(Table + b.to(tl.int64) * TS + first // 64)
        tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [(physical * KH + h) * (8 if PACKED else 64), 0],
                                      clamp_bounds=False, cache_modifier=".cs" if PACKED else "")
        tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [(physical * KH + h) * (8 if PACKED else 64), 0],
                                      clamp_bounds=False, cache_modifier=".cs" if PACKED else "")
    for base in range(first, end, BN):
        pos = base + n
        slot = ((base - first) // BN) % 2
        tlx.async_amd_descriptor_wait(0)
        if PACKED and QLEN > 1:
            kr = _load_packed_operand(kb, slot, LAYOUT)
        elif PACKED:
            kr = tlx.local_load(tlx.local_view(kb, slot), layout=REG, relaxed=True)
        else:
            kr = tlx.local_load(tlx.local_view(kb, slot), relaxed=True)
        if base + BN < end:
            next_phys = tl.load(Table + b.to(tl.int64) * TS + (base + BN) // 64)
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 1 - slot),
                                          [(next_phys * KH + h) * (8 if PACKED else 64), 0], clamp_bounds=False,
                                          cache_modifier=".cs" if PACKED else "")
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 1 - slot),
                                          [(next_phys * KH + h) * (8 if PACKED else 64), 0], clamp_bounds=False,
                                          cache_modifier=".cs" if PACKED else "")
        if PACKED and QLEN > 1:
            k = kr
        elif PACKED:
            k = tl.reshape(tl.permute(tl.reshape(kr, (8, 64, 8)), (0, 2, 1)), (64, 64))
        else:
            k = tl.trans(kr)
        scores = _dot(q, k, tl.zeros((BM, BN), tl.float32), LAYOUT) * (SCALE * 1.4426950408889634)
        full = base + BN <= length - QLEN + 1
        if WINDOW:
            full &= base >= length - WINDOW
        if not full:
            qpos = length - QLEN + m // 8
            valid = (pos[None, :] <= qpos[:, None]) & (pos[None, :] < length)
            if WINDOW:
                valid &= pos[None, :] > qpos[:, None] - WINDOW
            scores = tl.where(valid, scores, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(new_max == -float('inf'), 0., new_max)
        p = tl.exp2(scores - safe_max[:, None])
        alpha = tl.exp2(maximum - safe_max)
        denom = denom * alpha + tl.sum(p, 1)
        # The wait at the top of this iteration orders both K and V loads.
        # Delay reading V into registers until QK and softmax have consumed K.
        if PACKED and QLEN > 1:
            vr = _load_packed_operand(vb, slot, LAYOUT)
        elif PACKED:
            vr = tlx.local_load(tlx.local_view(vb, slot), layout=REG, relaxed=True)
        else:
            vr = tlx.local_load(tlx.local_view(vb, slot), relaxed=True)
        if PACKED and QLEN > 1:
            v = vr
        elif PACKED:
            v = tl.reshape(tl.permute(tl.reshape(vr, (8, 64, 8)), (0, 2, 1)), (64, 64))
        else:
            v = vr
        # Both K and V reads must finish before this stage can be refilled.
        tlx.workgroup_barrier()
        if PACKED and QLEN > 1:
            needs_v_mask = base + BN > length
            if WINDOW:
                needs_v_mask |= base < tl.maximum(length - QLEN - WINDOW + 1, 0)
            if needs_v_mask:
                valid_v = pos < length
                if WINDOW:
                    valid_v &= pos >= tl.maximum(length - QLEN - WINDOW + 1, 0)
                # Convert the small row mask before broadcasting it. All-ones
                # preserves every BF16 bit; zero clears poisoned page padding.
                mask_v = tlx.require_layout(-valid_v.to(tl.int16),
                                            tlx.slice_layout(tlx.dot_operand_layout(1, LAYOUT, k_width=8), 1))
                mask_v = tl.broadcast_to(mask_v[:, None], (BN, 64))
                v = (v.to(tl.int16, bitcast=True) & mask_v).to(tl.bfloat16, bitcast=True)
        elif not full:
            valid_v = pos < length
            if WINDOW:
                valid_v &= pos >= tl.maximum(length - QLEN - WINDOW + 1, 0)
            v = tl.where(valid_v[:, None], v, 0.)
        # BF16 matrix operands, with FP32 score/output and normalizer accumulation.
        acc = _dot(p.to(tl.bfloat16), v, acc * alpha[:, None], LAYOUT)
        maximum = new_max
    if NS == 1:
        inv = tl.div_rn(1., tl.where(denom > 0, denom, 1.))
        result = acc * inv[:, None]
        tl.store(Out + qoff, result, m[:, None] < QLEN * 8)
    else:
        # Store only real query rows, including for the padded q_len=1 tile.
        # Raw FP32 numerators avoid a divide/multiply pair across the merge.
        offset = ((b.to(tl.int64) * KH + h) * NS + s) * (QLEN * 8) + m
        tl.store(Part + offset[:, None] * 64 + d[None, :], acc, m[:, None] < QLEN * 8)
        # Keep max and sum separate: max + log2(sum) loses the sum when
        # logits have very large magnitude, causing incorrect split weights.
        tl.store(Stats + offset * 2, maximum, m < QLEN * 8)
        tl.store(Stats + offset * 2 + 1, denom, m < QLEN * 8)


@triton.jit
def _reduce(Part, Stats, Out, QLEN: tl.constexpr, BM: tl.constexpr, NS: tl.constexpr, BS: tl.constexpr,
            RM: tl.constexpr, KH: tl.constexpr, RD: tl.constexpr = 64):
    group = tl.program_id(0)
    h = tl.program_id(1) % KH
    b = tl.program_id(1) // KH
    m = group * RM + tl.arange(0, RM)
    s = tl.arange(0, BS)
    if RD == 64:
        d = tl.arange(0, 64)
    else:
        d = tl.program_id(2) * RD + tl.arange(0, RD)
    offset = ((b.to(tl.int64) * KH + h) * NS + s[None, :]) * BM + m[:, None]
    maxima = tl.load(Stats + offset * 2, (s[None, :] < NS) & (m[:, None] < QLEN * 8), -float('inf'))
    denom = tl.load(Stats + offset * 2 + 1, (s[None, :] < NS) & (m[:, None] < QLEN * 8), 0.)
    mx = tl.max(maxima, 1)
    mx = tl.where(mx == -float('inf'), 0., mx)
    weights = tl.exp2(maxima - mx[:, None])
    total = tl.sum(weights * denom, 1)
    part = tl.load(Part + offset[:, :, None] * 64 + d[None, None, :],
                   (s[None, :, None] < NS) & (m[:, None, None] < QLEN * 8), 0.)
    result = tl.sum(part * weights[:, :, None], 1) / tl.where(total[:, None] > 0, total[:, None], 1.)
    outoff = ((b.to(tl.int64) * QLEN + m // 8) * (KH * 8) + h * 8 + m % 8)[:, None] * 64 + d[None, :]
    tl.store(Out + outoff, result, m[:, None] < QLEN * 8)


def _select_config(batch, qlen, max_context_len, window=0, kv_heads=2):
    """Frozen dispatch rules; no on-device tuning or data-dependent host reads."""
    active = min(max_context_len, window + qlen + 62) if window else max_context_len
    short = active <= 2176
    # Occupancy depends on sequence/head groups, not batch size alone.
    groups = batch * kv_heads
    if qlen == 1:
        if groups <= 4:
            budget = 512 if active > 32768 else 256
        elif groups <= 16:
            budget = 256 if short else (512 if groups == 16 and active <= 16384 else 1024)
        else:
            budget = 1024 if short and groups <= 64 else 2048
        warps = 1
    else:
        if groups <= 4:
            budget = 512 if groups == 4 and active > 32768 else 256
        elif groups <= 16:
            budget = 256 if short else 512
        elif groups <= 64:
            budget = 512 if short else 1024
        else:
            budget = 1024
        warps = 4
    if qlen > 1:
        # Long speculative partitions need more CTAs at small group counts.
        # Short windowed partitions need fewer CTAs to limit reduction overhead.
        if not window and 4 < groups <= 16 and (active >= 98304 or (groups == 16 and active >= 65536)):
            budget = 1024
        elif window and short and 64 < groups <= 128:
            budget = 512
    # Medium global workloads spend a significant fraction of their time writing
    # and merging partials. Limit their CTA count without rounding up to a power
    # of two: that can nearly double the work for intermediate batch sizes.
    medium_global = not window and 16 < groups <= 64 and 2176 < active and groups * active <= 524288
    if medium_global:
        if qlen == 1:
            budget = 1024
        else:
            budget = 1024 if groups == 64 and active > 4096 else 512
        splits = max(1, min(triton.cdiv(active, 64), triton.cdiv(budget, groups)))
    else:
        splits = max(1, min(triton.cdiv(active, 64), triton.next_power_of_2(max(1, budget // groups))))
    if qlen > 1 and not window and splits >= 16:
        # Rounding up for intermediate group counts can exceed the intended
        # work budget by almost 2x.
        splits = min(splits, triton.cdiv(budget, groups))
    if qlen > 1 and not window and splits == 8 and groups > 128:
        # Short dense batches need fewer partials. Longer contexts benefit
        # from more parallel reads once there are over 160 sequence/head groups.
        if short:
            splits = 4
        elif active <= 8192:
            splits = triton.cdiv(1024, groups)
        elif groups > 160:
            splits = triton.cdiv(2048, groups)
    reduce_rows = min(8, max(1, 128 // triton.next_power_of_2(splits))) if groups >= 64 and qlen > 1 else 1
    # Pair q1 head rows; each reduction CTA covers 32 output channels.
    if qlen == 1 and 1 < splits <= 128:
        reduce_rows = 2
    reduce_warps = 4 if splits >= 256 else 1
    return splits, warps, reduce_rows, reduce_warps


def _plan(query, key_cache, value_cache, block_tables, query_length, sliding_window, max_context_len):
    tensors = (query, key_cache, value_cache, block_tables)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise InvalidInput("paged decode inputs must be tensors")
    if (query_length not in (1, 7, 8) or query.ndim != 3 or query.shape[2] != 64 or query.shape[1] < 8
            or query.shape[1] % 8):
        raise InvalidInput(
            "gfx1250 paged decode requires query [batch * query_length, Q_heads, 64], GQA8, query_length 1, 7 or 8")
    if query.shape[0] == 0 or query.shape[0] % query_length:
        raise InvalidInput("query rows must contain a positive whole number of sequences")
    batch = query.shape[0] // query_length
    kv_heads = query.shape[1] // 8
    if (key_cache.ndim not in (4, 5) or key_cache.shape != value_cache.shape or key_cache.shape[0] < 1
            or key_cache.shape[1:] not in ((kv_heads, 64, 64), (kv_heads, 8, 64, 8))):
        raise InvalidInput("gfx1250 paged decode requires page64, GQA8, D64 in standard 4-D or packed 5-D caches")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch or block_tables.shape[1] < 1:
        raise InvalidInput("block_tables must have shape [batch, capacity_pages >= 1]")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]) or block_tables.dtype != torch.int32:
        raise InvalidInput("gfx1250 paged decode requires BF16 Q/K/V and int32 block_tables")
    if not query.is_cuda or any(t.device != query.device or not t.is_contiguous() for t in tensors):
        raise InvalidInput("paged decode inputs must be contiguous and on one GPU")
    if key_cache.data_ptr() % 16 or value_cache.data_ptr() % 16:
        raise InvalidInput("paged decode cache pointers must be 16-byte aligned")
    if max_context_len is None:
        max_context_len = block_tables.shape[1] * 64
    # Reserve 512 page64 tiles for rounded split ends and inactive partitions.
    if type(max_context_len) is not int or not 0 <= max_context_len <= min(block_tables.shape[1] * 64, 2**31 - 32768):
        raise InvalidInput("max_context_len must fit block_tables and be in [0, 2**31 - 32768]")
    descriptor_rows = key_cache.shape[0] * kv_heads * (8 if key_cache.ndim == 5 else 64)
    if descriptor_rows >= 2**31:
        raise InvalidInput("cache exceeds the tensor descriptor coordinate range")
    if sliding_window >= max_context_len:
        sliding_window = 0
    return batch, sliding_window, _select_config(batch, query_length, max_context_len, sliding_window, kv_heads)


def _workspace_shapes(batch, query_length, splits, kv_heads):
    return (batch, kv_heads, splits, query_length * 8, 64), (batch, kv_heads, splits, query_length * 8, 2)


def _allocate_workspace(query, batch, query_length, splits):
    if splits == 1:
        return None
    return tuple(
        torch.empty(shape, dtype=torch.float32, device=query.device)
        for shape in _workspace_shapes(batch, query_length, splits, query.shape[1] // 8))


def paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, query_length, sliding_window,
                                     max_context_len):
    batch, _, config = _plan(query, key_cache, value_cache, block_tables, query_length, sliding_window, max_context_len)
    return _allocate_workspace(query, batch, query_length, config[0])


def _check_writes(out, workspace, query, batch, query_length, splits, inputs):
    if (not isinstance(out, torch.Tensor) or out.shape != query.shape or out.dtype != query.dtype
            or out.device != query.device or not out.is_contiguous() or out.requires_grad):
        raise InvalidInput("out must be a contiguous BF16 inference tensor matching query")
    writes = [out]
    if splits == 1:
        if workspace is not None:
            raise InvalidInput("this decode plan needs no workspace; use paged_attention_decode_workspace")
    else:
        if not isinstance(workspace, (tuple, list)) or len(workspace) != 2:
            raise InvalidInput("workspace must come from paged_attention_decode_workspace with matching arguments")
        for tensor, shape in zip(workspace, _workspace_shapes(batch, query_length, splits, query.shape[1] // 8)):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32
                    or tensor.device != query.device or not tensor.is_contiguous() or tensor.requires_grad):
                raise InvalidInput(
                    "workspace tensors must match the decode plan, be contiguous FP32 and on query's GPU")
        writes.extend(workspace)
    storage = {t.untyped_storage().data_ptr() for t in inputs}
    for tensor in writes:
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in storage:
            raise InvalidInput("out and workspace must have separate storage from inputs and each other")
        storage.add(ptr)


def paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables, sm_scale, query_length,
                           sliding_window, max_context_len, out, workspace):
    batch, window, config = _plan(query, key_cache, value_cache, block_tables, query_length, sliding_window,
                                  max_context_len)
    if (context_lens.shape != (batch, ) or context_lens.dtype != torch.int32 or context_lens.device != query.device
            or not context_lens.is_contiguous()):
        raise InvalidInput("context_lens must be contiguous int32 [batch] on query's GPU")
    splits, warps, reduce_rows, reduce_warps = config
    if out is None:
        out = torch.empty_like(query)
    if workspace is None:
        workspace = _allocate_workspace(query, batch, query_length, splits)
    _check_writes(out, workspace, query, batch, query_length, splits,
                  (query, key_cache, value_cache, context_lens, block_tables))
    partial, stats = workspace if workspace is not None else (None, None)
    layout = tlx.amd_wmma_layout([[2**i, 0] for i in range(warps.bit_length() - 1)], instr_shape=(16, 16, 32))
    # Inverse of [8,64,8] -> [8,8,64] -> [64,64] packing, in flat offsets.
    # Loads land directly in WMMA B operands; all row waves share K/V.
    registers = tlx.layout(shape=((16, 2, warps), (8, 2, 2, 2, 2)), stride=((8, 512, 0), (1, 1024, 2048, 128, 256)))
    kv_heads = key_cache.shape[1]
    # Spread long sequences over the device without reading lengths on the host.
    # With many heads, dispatch adjacent heads together; with fewer heads,
    # rotate sequences between head/split planes to avoid repeated length order.
    # Keep partial groups of 16 sequences in their original order: adjacent
    # heads can otherwise concentrate the final long sequence on one engine.
    reorder = batch >= 16
    # Windowed head-first dispatch needs enough groups to amortize its cost.
    head_first = reorder and kv_heads >= 16 and batch % 16 == 0 and (window == 0 or batch * kv_heads >= 512)
    # Short q1 windows and incomplete batch groups benefit from the original
    # order; speculative queries have more work to hide the rotation overhead.
    rotate = (reorder and kv_heads < 16 and batch * kv_heads >= 128
              and (query_length > 1 or (window == 0 and batch % 16 == 0)))
    grid = (kv_heads, batch, splits) if head_first else (batch, kv_heads, splits)
    _partition[grid](query, key_cache, value_cache, block_tables, context_lens, partial, stats,
                     out, query_length, block_tables.shape[1], splits, 64,
                     max(16, triton.next_power_of_2(query_length * 8)), window, key_cache.ndim == 5, layout, sm_scale,
                     key_cache.shape[0], registers, kv_heads, rotate, batch, head_first, num_warps=warps)
    if splits > 1:
        reduce_columns = 32 if query_length == 1 and splits <= 128 else 64
        _reduce[(triton.cdiv(query_length * 8, reduce_rows), batch * kv_heads,
                 64 // reduce_columns)](partial, stats, out, query_length, query_length * 8, splits,
                                        triton.next_power_of_2(splits), reduce_rows, kv_heads, reduce_columns,
                                        num_warps=reduce_warps)
    return out
