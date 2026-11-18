"""Explicit-layout BF16 paged prefill fast path for gfx1250 TLX.

Normalize the same BF16 probabilities used by PV; both sums accumulate in
FP32. Abnormal ranges replay stable online softmax in smaller query chunks.
The interior omits masking when causal and window validity follow from its
bounds, peeling both possible left-edge window pages. Physical K/V pages are
complete; Q descriptor bounds cover ragged tails.

Q is contiguous [B,S,8*KVH,64]; K/V are [physical_pages,64,KVH,64]. The
int32 page table is [B,logical_pages] and must contain valid physical IDs.
The packed entry reads per-sequence lengths at runtime and supports AITER's
shuffled 5-D cache directly. Window 2048 includes self; window 0 means global.
Cached prefixes use bottom-right causal alignment.
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


# Keep compatibility with validated gfx1250 builds predating amd_wmma_layout.
class WmmaLayout(tlx.types.layout_encoding):

    def __init__(self, transposed, warp_bases, reg_bases):
        super().__init__()
        self.transposed = transposed
        self.warp_bases = warp_bases
        self.reg_bases = reg_bases

    def to_ir(self, builder, shape=None, element_type=None):
        return builder.make_amd_wmma_encoding_attr(3, self.transposed, self.warp_bases, self.reg_bases, [16, 16, 32], 2)


# Layout values include every field in the compilation cache key.
WMMA_TWO = tl.constexpr(WmmaLayout(True, [[1, 0]], []))
WMMA_FOUR = tl.constexpr(WmmaLayout(True, [[2, 0], [4, 0]], [[0, 1], [1, 0]]))


@triton.jit
def _dot(a, b, c):
    layout: tl.constexpr = WMMA_TWO if tlx.num_warps() == 2 else WMMA_FOUR
    aa = tlx.require_layout(a, tlx.dot_operand_layout(0, layout, k_width=8))
    bb = tlx.require_layout(b, tlx.dot_operand_layout(1, layout, k_width=8))
    cc = tlx.require_layout(c, layout)
    return tl.dot(aa, bb, cc)


@triton.jit
def _scale_online(scores):
    # The max and exponent must use the same rounded FP32 scores. Prevent
    # fusion into score*scale-center after the maximum has been rounded.
    return tl.inline_asm_elementwise("v_mul_f32 $0, 0x3e38aa3b, $1;", constraints="=v,v", args=[scores],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _consume(
    q,
    k,
    v,
    acc,
    maximum,
    denom,
    rows,
    key_start,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    FIXED: tl.constexpr,
    MASK: tl.constexpr = True,
):
    scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32))
    if FIXED:
        scores *= 0.125 * 1.4426950408889634
    else:
        scores = _scale_online(scores)
    if MASK:
        keys = key_start + tl.arange(0, 64)
        valid = (keys[None, :] <= rows[:, None] + SK - SQ) & (keys[None, :] < SK)
        if WINDOW:
            valid &= keys[None, :] > rows[:, None] + SK - SQ - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
    if SK % 64:
        # Zero masked values too: an unwritten final-page slot may hold NaN,
        # and zero probability alone does not prevent 0 * NaN in PV.
        v = tl.where((key_start + tl.arange(0, 64))[:, None] < SK, v, 0.0)
    if FIXED:
        p = tl.exp2(scores - maximum[:, None])
        denom = _dot(p.to(q.dtype), tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p.to(q.dtype), v, acc)
        return acc, maximum, denom
    else:
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        # Only an all-masked row needs a finite center. Clamping finite
        # negative maxima would underflow every probability during recovery.
        center = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - center[:, None])
        alpha = tl.exp2(maximum - center)
        denom = denom * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        acc = _dot(p.to(q.dtype), v, acc)
        return acc, new_max, denom


@triton.jit
def _scores(
    q,
    k,
    rows,
    page,
    first_q,
    last_q,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32)) * (0.125 * 1.4426950408889634)
    partial = True
    if SPLIT:
        full = page * 64 + 63 <= first_q
        if WINDOW:
            full &= page * 64 > last_q - 1 - WINDOW
        partial = not full
    if partial:
        keys = page * 64 + tl.arange(0, 64)
        valid = (keys[None, :] <= rows[:, None] + SK - SQ) & (keys[None, :] < SK)
        if WINDOW:
            valid &= keys[None, :] > rows[:, None] + SK - SQ - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
    return scores


@triton.jit
def _finish(scores, denom, maximum, BM: tl.constexpr, DTYPE: tl.constexpr):
    p = tl.exp2(scores - maximum[:, None])
    return p.to(DTYPE), denom


@triton.jit
def _prob(
    q,
    k,
    denom,
    maximum,
    rows,
    page,
    first_q,
    last_q,
    SQ,
    SK,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # This helper consumes only the final two pages. They are entirely
    # above a 2048-token window's lower edge; only causality can mask them.
    scores = _scores(q, k, rows, page, first_q, last_q, SQ, SK, 0, BM, SPLIT)
    return _finish(scores, denom, maximum, BM, q.dtype)


@triton.jit
def _view_k(kb, slot, SHUFFLED: tl.constexpr):
    if SHUFFLED:
        view = tlx.local_reshape(tlx.local_view(kb, slot), [8, 64, 8])
        view = tlx.local_trans(view, [1, 0, 2])
        view = tlx.local_reshape(view, [64, 64])
        return tlx.local_load(tlx.local_trans(view))
    else:
        return tlx.local_load(tlx.local_trans(tlx.local_view(kb, slot)))


@triton.jit
def _view_v(vb, slot, SHUFFLED: tl.constexpr):
    if SHUFFLED:
        view = tlx.local_reshape(tlx.local_view(vb, slot), [8, 64, 8])
        view = tlx.local_trans(view, [0, 2, 1])
        return tlx.local_load(tlx.local_reshape(view, [64, 64]))
    else:
        return tlx.local_load(tlx.local_view(vb, slot))


@triton.jit
def _get_k(kb, slot, wait: tl.constexpr, SHUFFLED: tl.constexpr):
    tlx.async_amd_descriptor_wait(wait)
    return _view_k(kb, slot, SHUFFLED)


@triton.jit
def _get_v(vb, slot, wait: tl.constexpr, SHUFFLED: tl.constexpr):
    tlx.async_amd_descriptor_wait(wait)
    v = _view_v(vb, slot, SHUFFLED)
    # Finish every wave's LDS reads before reusing a stage.
    tlx.workgroup_barrier()
    return v


@triton.jit
def _put_k_id(desc, buf, slot, phys, PAGE_ROWS: tl.constexpr):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * PAGE_ROWS, 0], clamp_bounds=False)


@triton.jit
def _put_v_id(desc, buf, slot, phys, PAGE_ROWS: tl.constexpr):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * PAGE_ROWS, 0], clamp_bounds=False)


@triton.jit
def _pass(
    Q,
    K,
    V,
    Tables,
    Out,
    SQ,
    SK,
    NPAGES,
    KH: tl.constexpr,
    TABLE_STRIDE,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    PIPE: tl.constexpr,
    PAD: tl.constexpr,
    SPLIT_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
    FIXED: tl.constexpr,
    DEEP: tl.constexpr,
    FACTOR: tl.constexpr = 1,
    CHUNK=0,
    Q_START=0,
    PACKED: tl.constexpr = False,
    SHUFFLED: tl.constexpr = False,
    SAMPLED: tl.constexpr = True,
):
    batch = tl.program_id(2)
    blocks = tl.num_programs(0) if PACKED else tl.cdiv(SQ * GROUP, BM * FACTOR)
    linear = tl.program_id(0) + blocks * tl.program_id(1)
    block = linear // (KH * 8 // GROUP)
    if REVERSE:
        block = blocks - 1 - block
    block = block * FACTOR + CHUNK
    head = (linear % (KH * 8 // GROUP)) * GROUP
    kv_head = head // 8
    row_ids = block * BM + tl.arange(0, BM)
    rows = row_ids // GROUP
    heads = head + row_ids % GROUP
    dims = tl.arange(0, 64)
    q_start = Q_START.to(tl.int64) if PACKED else batch.to(tl.int64) * SQ
    offsets = ((q_start + rows[:, None]) * (KH * 8) + heads[:, None]) * 64 + dims[None, :]
    qd = tl.make_tensor_descriptor(
        Q + q_start * KH * 8 * 64 + head * 64,
        [SQ, GROUP * 64],
        [KH * 8 * 64, 1],
        [BM // GROUP, GROUP * 64],
    )
    qb = tlx.local_alloc((BM // GROUP, GROUP * 64), tl.bfloat16, 1)
    tlx.async_amd_descriptor_load(qd, tlx.local_view(qb, 0), [block * (BM // GROUP), 0], clamp_bounds=True)
    if SHUFFLED:
        # K [P,H,D/8,64,8], V [P,H,64/8,D,8]: each physical
        # head occupies 4096 contiguous BF16 elements, read without a gather.
        kd = tl.make_tensor_descriptor(K + kv_head * 4096, [NPAGES * KH * 8, 512], [512, 1], [8, 512])
        vd = tl.make_tensor_descriptor(V + kv_head * 4096, [NPAGES * KH * 8, 512], [512, 1], [8, 512])
        kb = tlx.local_alloc((8, 512), tl.bfloat16, 2 if PIPE or FIXED else 1)
        vb = tlx.local_alloc((8, 512), tl.bfloat16, 2 if PIPE else 1)
        PAGE_ROWS: tl.constexpr = KH * 8
    else:
        kd = tl.make_tensor_descriptor(K + kv_head * 64, [NPAGES * 64, 64], [KH * 64, 1], [64, 64])
        vd = tl.make_tensor_descriptor(V + kv_head * 64, [NPAGES * 64, 64], [KH * 64, 1], [64, 64])
        if PAD:
            layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(64, PAD)], [64, 64], [1, 0])
            kb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE or FIXED else 1, layout=layout)
        else:
            kb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE or FIXED else 1)
        vb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE else 1)
        PAGE_ROWS: tl.constexpr = 64
    first_q = block * (BM // GROUP) + SK - SQ
    last_q = tl.minimum(first_q + BM // GROUP, SK)
    first_page = 0
    if WINDOW:
        first_page = tl.maximum(0, (first_q - WINDOW + 1) // 64)
    end_page = tl.cdiv(last_q, 64)
    maximum = tl.full([BM], 0.0 if FIXED else -float("inf"), tl.float32)
    if FIXED and DEEP:
        use_deep = end_page - first_page >= (4 if WINDOW else 3)
    else:
        use_deep: tl.constexpr = False
    if FIXED and SAMPLED and not use_deep:
        # Sample both ends of the visible range. A fixed center avoids online
        # rescaling in the main loop while keeping ordinary large logits away
        # from the replay threshold. Unseen extremes still use stable replay.
        reference_page = tl.maximum(first_page, (first_q + 1) // 64 - 1)
        reference_first = first_page
        if WINDOW:
            reference_first = tl.minimum(reference_page, tl.cdiv(tl.maximum(0, last_q - WINDOW), 64))
        _put_k_id(kd, kb, 0, tl.load(Tables + batch * TABLE_STRIDE + reference_first), PAGE_ROWS)
        _put_k_id(kd, kb, 1, tl.load(Tables + batch * TABLE_STRIDE + reference_page), PAGE_ROWS)
        tlx.async_amd_descriptor_wait(2)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        k_first = _get_k(kb, 0, 1, SHUFFLED)
        first_scores = _scores(q, k_first, rows, reference_first, first_q, last_q, SQ, SK, WINDOW, BM, True)
        first_max = tl.max(first_scores, 1)
        k_last = _get_k(kb, 1, 0, SHUFFLED)
        last_scores = _scores(q, k_last, rows, reference_page, first_q, last_q, SQ, SK, WINDOW, BM, True)
        maximum = tl.maximum(first_max, tl.max(last_scores, 1))
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        # Bound the center before contraction of dot*scale-center into FMA.
        # Large equal logits otherwise acquire different rounding residues
        # in sampled and interior pages. Extreme ranges still trigger replay.
        # Extra exponent headroom covers sparse peaks missed by both samples.
        maximum = tl.minimum(tl.maximum(maximum + 16.0, -64.0), 64.0)
        tlx.workgroup_barrier()
    if FIXED:
        denom = tl.zeros([BM, 16], tl.float32)
    else:
        denom = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, 64], tl.float32)
    if use_deep:
        phys0 = tl.load(Tables + batch * TABLE_STRIDE + first_page)
        phys1 = tl.load(Tables + batch * TABLE_STRIDE + first_page + 1)
        phys2 = tl.load(Tables + batch * TABLE_STRIDE + first_page + 2)
        _put_k_id(kd, kb, 0, phys0, PAGE_ROWS)
        _put_k_id(kd, kb, 1, phys1, PAGE_ROWS)
        _put_v_id(vd, vb, 0, phys0, PAGE_ROWS)
        tlx.async_amd_descriptor_wait(3)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        tlx.workgroup_barrier()
        k = _get_k(kb, 0, 2, SHUFFLED)
        tlx.workgroup_barrier()
        if BM // GROUP <= 64:
            first_scores = _dot(q, k, tl.zeros((BM, 64), tl.float32)) * (0.125 * 1.4426950408889634)
            if WINDOW:
                keys = first_page * 64 + tl.arange(0, 64)
                first_scores = tl.where(keys[None, :] > rows[:, None] + SK - SQ - WINDOW, first_scores, -float("inf"))
        else:
            first_scores = _scores(q, k, rows, first_page, first_q, last_q, SQ, SK, WINDOW, BM, SPLIT_MASK)
        if SAMPLED:
            first_max = tl.max(first_scores, 1)
            if WINDOW:
                # Both left-edge pages are already queued. Reuse their QK
                # scores for the reference and the two peeled iterations.
                reference_page = first_page + 1
                reference_k = _get_k(kb, 1, 1, SHUFFLED)
            else:
                reference_page = tl.maximum(first_page, (first_q + 1) // 64 - 1)
                _put_k_id(kd, kb, 0, tl.load(Tables + batch * TABLE_STRIDE + reference_page), PAGE_ROWS)
                reference_k = _get_k(kb, 0, 0, SHUFFLED)
            if BM // GROUP <= 64:
                reference_scores = _dot(q, reference_k, tl.zeros((BM, 64), tl.float32)) * (0.125 * 1.4426950408889634)
            else:
                reference_scores = _scores(q, reference_k, rows, reference_page, first_q, last_q, SQ, SK, WINDOW, BM,
                                           True)
            if WINDOW:
                keys = reference_page * 64 + tl.arange(0, 64)
                reference_scores = tl.where(keys[None, :] > rows[:, None] + SK - SQ - WINDOW, reference_scores,
                                            -float("inf"))
                next_scores = reference_scores
            maximum = tl.maximum(first_max, tl.max(reference_scores, 1))
            maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
            maximum = tl.minimum(tl.maximum(maximum + 16.0, -64.0), 64.0)
            tlx.workgroup_barrier()
        p, denom = _finish(first_scores, denom, maximum, BM, q.dtype)
        _put_k_id(kd, kb, 0, phys2, PAGE_ROWS)
        _put_v_id(vd, vb, 1, phys1, PAGE_ROWS)
        if not (WINDOW and SAMPLED):
            k = _get_k(kb, 1, 3, SHUFFLED)
        slot = 0
        body_begin = first_page
        if WINDOW:
            # Peel the second possible lower-edge page. The steady loop
            # then has no masks or per-iteration visibility branches, even
            # when runtime lengths place a query tile across a page boundary.
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + first_page + 3)
            if not SAMPLED:
                if BM // GROUP <= 64:
                    next_scores = _dot(q, k, tl.zeros((BM, 64), tl.float32)) * (0.125 * 1.4426950408889634)
                    keys = (first_page + 1) * 64 + tl.arange(0, 64)
                    next_scores = tl.where(keys[None, :] > rows[:, None] + SK - SQ - WINDOW, next_scores, -float("inf"))
                else:
                    next_scores = _scores(q, k, rows, first_page + 1, first_q, last_q, SQ, SK, WINDOW, BM, True)
            v = _get_v(vb, 0, 2, SHUFFLED)
            _put_k_id(kd, kb, 1, next_phys, PAGE_ROWS)
            denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
            acc = _dot(p, v, acc)
            p, denom = _finish(next_scores, denom, maximum, BM, q.dtype)
            k = _get_k(kb, 0, 2, SHUFFLED)
            _put_v_id(vd, vb, 0, phys2, PAGE_ROWS)
            phys2 = next_phys
            slot = 1
            body_begin += 1
        for page in range(body_begin, end_page - 3):
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + page + 3)
            # With <=64 distinct query positions, every key in this
            # prologue-excluded iteration is strictly before the earliest query.
            # At most two pages intersect the left window boundary, even
            # when a runtime prefix places the query tile across a page.
            # The prologue and peeled step covered both lower-edge pages.
            if BM // GROUP <= 64:
                next_scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32)) * (0.125 * 1.4426950408889634)
            else:
                next_scores = _scores(
                    q,
                    k,
                    rows,
                    page + 1,
                    first_q,
                    last_q,
                    SQ,
                    SK,
                    WINDOW,
                    BM,
                    SPLIT_MASK,
                )
            v = _get_v(vb, slot, 2, SHUFFLED)
            _put_k_id(kd, kb, 1 - slot, next_phys, PAGE_ROWS)
            denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
            acc = _dot(p, v, acc)
            p, denom = _finish(next_scores, denom, maximum, BM, q.dtype)
            k = _get_k(kb, slot, 2, SHUFFLED)
            _put_v_id(vd, vb, slot, phys2, PAGE_ROWS)
            phys2 = next_phys
            slot = 1 - slot
        v = _get_v(vb, slot, 2, SHUFFLED)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
            maximum,
            rows,
            end_page - 2,
            first_q,
            last_q,
            SQ,
            SK,
            WINDOW,
            BM,
            SPLIT_MASK,
        )
        k = _get_k(kb, slot, 1, SHUFFLED)
        _put_v_id(vd, vb, slot, phys2, PAGE_ROWS)
        v = _get_v(vb, 1 - slot, 1, SHUFFLED)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
            maximum,
            rows,
            end_page - 1,
            first_q,
            last_q,
            SQ,
            SK,
            WINDOW,
            BM,
            SPLIT_MASK,
        )
        v = _get_v(vb, slot, 0, SHUFFLED)
        if SK % 64:
            v = tl.where(((end_page - 1) * 64 + tl.arange(0, 64))[:, None] < SK, v, 0.0)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
    elif PIPE:
        phys = tl.load(Tables + batch * TABLE_STRIDE + first_page)
        tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * PAGE_ROWS, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * PAGE_ROWS, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_wait(2)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        tlx.workgroup_barrier()
        for page in range(first_page, end_page):
            slot = (page - first_page) % 2
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + page + 1, page + 1 < end_page, 0)
            tlx.async_amd_descriptor_load(
                kd,
                tlx.local_view(kb, 1 - slot),
                [next_phys * PAGE_ROWS, 0],
                clamp_bounds=False,
            )
            tlx.async_amd_descriptor_load(
                vd,
                tlx.local_view(vb, 1 - slot),
                [next_phys * PAGE_ROWS, 0],
                clamp_bounds=False,
            )
            tlx.async_amd_descriptor_wait(2)
            k = _view_k(kb, slot, SHUFFLED)
            v = _view_v(vb, slot, SHUFFLED)
            tlx.workgroup_barrier()
            if SPLIT_MASK:
                full = page * 64 + 63 <= first_q
                if WINDOW:
                    full &= page * 64 > last_q - 1 - WINDOW
                if full:
                    acc, maximum, denom = _consume(
                        q,
                        k,
                        v,
                        acc,
                        maximum,
                        denom,
                        rows,
                        page * 64,
                        SQ,
                        SK,
                        WINDOW,
                        BM,
                        FIXED,
                        False,
                    )
                else:
                    acc, maximum, denom = _consume(
                        q,
                        k,
                        v,
                        acc,
                        maximum,
                        denom,
                        rows,
                        page * 64,
                        SQ,
                        SK,
                        WINDOW,
                        BM,
                        FIXED,
                    )
            else:
                acc, maximum, denom = _consume(
                    q,
                    k,
                    v,
                    acc,
                    maximum,
                    denom,
                    rows,
                    page * 64,
                    SQ,
                    SK,
                    WINDOW,
                    BM,
                    FIXED,
                )
        tlx.async_amd_descriptor_wait(0)
    else:
        tlx.async_amd_descriptor_wait(0)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        tlx.workgroup_barrier()
        for page in range(first_page, end_page):
            phys = tl.load(Tables + batch * TABLE_STRIDE + page)
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * PAGE_ROWS, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * PAGE_ROWS, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_wait(0)
            k = _view_k(kb, 0, SHUFFLED)
            v = _view_v(vb, 0, SHUFFLED)
            tlx.workgroup_barrier()
            acc, maximum, denom = _consume(q, k, v, acc, maximum, denom, rows, page * 64, SQ, SK, WINDOW, BM, FIXED)
    if FIXED:
        denom = tl.sum(denom, 1) * 0.0625
    result = acc / tl.maximum(denom[:, None], 1.0e-30)
    tl.store(Out + offsets, result.to(Out.dtype.element_ty), rows[:, None] < SQ)
    if FIXED:
        bad = (denom < 1.0e-20) | ~(denom < 1.0e20)
        bad |= tl.sum((~(tl.abs(acc) < 3.402823466e38)).to(tl.int32), 1) > 0
        return tl.sum(((rows < SQ) & bad).to(tl.int32), 0) > 0
    return False


@triton.jit
def _replay(
    Q,
    K,
    V,
    Tables,
    Out,
    SQ,
    SK,
    NPAGES,
    KH: tl.constexpr,
    TABLE_STRIDE,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    PIPE: tl.constexpr,
    PAD: tl.constexpr,
    SPLIT_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    FACTOR: tl.constexpr = 2 if BM >= 64 else 1
    for chunk in tl.range(FACTOR, loop_unroll_factor=1):
        _pass(
            Q,
            K,
            V,
            Tables,
            Out,
            SQ,
            SK,
            NPAGES,
            KH,
            TABLE_STRIDE,
            WINDOW,
            BM // FACTOR,
            GROUP,
            PIPE,
            PAD,
            SPLIT_MASK,
            REVERSE,
            False,
            False,
            FACTOR,
            chunk,
        )


@triton.jit
def paged_prefill_kernel(
    Q,
    K,
    V,
    Tables,
    Out,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    NPAGES: tl.constexpr,
    KH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    PIPE: tl.constexpr,
    PAD: tl.constexpr,
    SPLIT_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
    FIXED: tl.constexpr,
    DEEP: tl.constexpr,
):
    retry = _pass(
        Q,
        K,
        V,
        Tables,
        Out,
        SQ,
        SK,
        NPAGES,
        KH,
        TABLE_STRIDE,
        WINDOW,
        BM,
        GROUP,
        PIPE,
        PAD,
        SPLIT_MASK,
        REVERSE,
        FIXED,
        DEEP,
        # Preserve the established full-prefill fast path. Serving calls and
        # cached-prefix specializations use the sampled reference.
        SAMPLED=SK != SQ,
    )
    if FIXED:
        if retry:
            _replay(
                Q,
                K,
                V,
                Tables,
                Out,
                SQ,
                SK,
                NPAGES,
                KH,
                TABLE_STRIDE,
                WINDOW,
                BM,
                GROUP,
                PIPE,
                PAD,
                SPLIT_MASK,
                REVERSE,
            )


@triton.jit(do_not_specialize=["NPAGES", "TABLE_STRIDE"])
def packed_prefill_kernel(Q, K, V, Tables, Out, CuQ, UsedK, NPAGES, TABLE_STRIDE, KH: tl.constexpr,
                          WINDOW: tl.constexpr, BM: tl.constexpr, SHUFFLED: tl.constexpr):
    batch = tl.program_id(2)
    q_start = tl.load(CuQ + batch)
    sq = tl.load(CuQ + batch + 1) - q_start
    linear = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    block = tl.num_programs(0) - 1 - linear // KH
    if block * (BM // 8) < sq:
        sk = tl.load(UsedK + batch).to(tl.int32)
        retry = _pass(Q, K, V, Tables, Out, sq, sk, NPAGES, KH, TABLE_STRIDE, WINDOW, BM, 8, True, 8, True, True, True,
                      True, Q_START=q_start, PACKED=True, SHUFFLED=SHUFFLED)
        if retry:
            for chunk in tl.range(2, loop_unroll_factor=1):
                _pass(Q, K, V, Tables, Out, sq, sk, NPAGES, KH, TABLE_STRIDE, WINDOW, BM // 2, 8, True, 8, True, True,
                      False, False, 2, chunk, q_start, True, SHUFFLED)


def launch_packed(q, k, v, tables, out, cu_q, used_k, max_q, *, window=0, bm=128, shuffled=False, warps=None):
    kh = q.shape[1] // 8
    warps = (4 if bm >= 256 else 2) if warps is None else warps
    return packed_prefill_kernel[(triton.cdiv(max_q * 8,
                                              bm), kh, used_k.numel())](q, k, v, tables, out, cu_q, used_k, k.shape[0],
                                                                        tables.stride(0), kh, window, bm, shuffled,
                                                                        num_warps=warps, waves_per_eu=1,
                                                                        reverse_local_assignment=True,
                                                                        llvm_fn_attrs="amdgpu-sched-strategy=coexec")


def launch(
    q,
    k,
    v,
    tables,
    out,
    *,
    sk=None,
    window=0,
    bm=64,
    group=1,
    pipe=False,
    pad=0,
    split_mask=False,
    reverse=False,
    warps=4,
    waves=1,
    fixed=False,
    deep=False,
    reverse_local_assignment=False,
    enable_wmma_queuing=False,
    llvm_fn_attrs="",
):
    if warps not in (2, 4):
        raise ValueError("The explicit WMMA pipeline supports two or four waves")
    b, sq, q_heads, _ = q.shape
    sk = sq if sk is None else sk
    # A window covering the full KV length is exactly global causal attention.
    window = 0 if window and sk <= window else window
    return paged_prefill_kernel[(triton.cdiv(sq * group, bm), q_heads // group, b)](
        q,
        k,
        v,
        tables,
        out,
        sq,
        sk,
        k.shape[0],
        k.shape[2],
        tables.stride(0),
        window,
        bm,
        group,
        pipe,
        pad,
        split_mask,
        reverse,
        fixed,
        deep,
        num_warps=warps,
        waves_per_eu=waves,
        reverse_local_assignment=reverse_local_assignment,
        enable_wmma_queuing=enable_wmma_queuing,
        llvm_fn_attrs=llvm_fn_attrs,
    )
