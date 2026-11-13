"""Explicit-layout BF16 paged prefill fast path for gfx1250 TLX.

Normalize the same BF16 probabilities used by PV; both sums accumulate in
FP32. Abnormal ranges replay stable online softmax in smaller query chunks.
The interior omits masking only when causal and window validity follow from
its bounds. SWA uses this path only for tiles aligned within a KV page.
Physical K/V pages are complete; Q descriptor bounds still cover ragged tails.

Q is contiguous [B,S,8*KVH,64]; K/V are [physical_pages,64,KVH,64]. The
int32 page table is [B,logical_pages] and must contain valid physical IDs.
All sequences in a batch share Q/KV lengths. Window 2048 includes self;
window 0 means global. Cached prefixes use bottom-right causal alignment.
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
def _consume(
    q,
    k,
    v,
    acc,
    maximum,
    denom,
    rows,
    key_start,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    FIXED: tl.constexpr,
    MASK: tl.constexpr = True,
):
    scores = _dot(q, k, tl.zeros((q.shape[0], k.shape[1]), tl.float32)) * (0.125 * 1.4426950408889634)
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
        p = tl.exp2(scores)
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
    SQ: tl.constexpr,
    SK: tl.constexpr,
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
def _finish(scores, denom, BM: tl.constexpr, DTYPE: tl.constexpr):
    p = tl.exp2(scores)
    return p.to(DTYPE), denom


@triton.jit
def _prob(
    q,
    k,
    denom,
    rows,
    page,
    first_q,
    last_q,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    scores = _scores(q, k, rows, page, first_q, last_q, SQ, SK, WINDOW, BM, SPLIT)
    return _finish(scores, denom, BM, q.dtype)


@triton.jit
def _get_k(kb, slot, wait: tl.constexpr):
    tlx.async_amd_descriptor_wait(wait)
    return tlx.local_load(tlx.local_trans(tlx.local_view(kb, slot)))


@triton.jit
def _get_v(vb, slot, wait: tl.constexpr):
    tlx.async_amd_descriptor_wait(wait)
    v = tlx.local_load(tlx.local_view(vb, slot))
    # Finish every wave's LDS reads before reusing a stage.
    tlx.workgroup_barrier()
    return v


@triton.jit
def _put_k(desc, buf, slot, tables, batch, page, STRIDE: tl.constexpr):
    phys = tl.load(tables + batch * STRIDE + page)
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=False)


@triton.jit
def _put_v(desc, buf, slot, tables, batch, page, STRIDE: tl.constexpr):
    phys = tl.load(tables + batch * STRIDE + page)
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=False)


@triton.jit
def _put_k_id(desc, buf, slot, phys):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=False)


@triton.jit
def _put_v_id(desc, buf, slot, phys):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=False)


@triton.jit
def _pass(
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
    FACTOR: tl.constexpr = 1,
    CHUNK=0,
):
    batch = tl.program_id(2)
    linear = tl.program_id(0) + tl.cdiv(SQ * GROUP, BM * FACTOR) * tl.program_id(1)
    block = linear // (KH * 8 // GROUP)
    if REVERSE:
        block = tl.cdiv(SQ * GROUP, BM * FACTOR) - 1 - block
    block = block * FACTOR + CHUNK
    head = (linear % (KH * 8 // GROUP)) * GROUP
    kv_head = head // 8
    row_ids = block * BM + tl.arange(0, BM)
    rows = row_ids // GROUP
    heads = head + row_ids % GROUP
    dims = tl.arange(0, 64)
    offsets = ((batch.to(tl.int64) * SQ + rows[:, None]) * (KH * 8) + heads[:, None]) * 64 + dims[None, :]
    qd = tl.make_tensor_descriptor(
        Q + batch.to(tl.int64) * SQ * KH * 8 * 64 + head * 64,
        [SQ, GROUP * 64],
        [KH * 8 * 64, 1],
        [BM // GROUP, GROUP * 64],
    )
    qb = tlx.local_alloc((BM // GROUP, GROUP * 64), tl.bfloat16, 1)
    tlx.async_amd_descriptor_load(qd, tlx.local_view(qb, 0), [block * (BM // GROUP), 0], clamp_bounds=True)
    kd = tl.make_tensor_descriptor(K + kv_head * 64, [NPAGES * 64, 64], [KH * 64, 1], [64, 64])
    vd = tl.make_tensor_descriptor(V + kv_head * 64, [NPAGES * 64, 64], [KH * 64, 1], [64, 64])
    if PAD:
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(64, PAD)], [64, 64], [1, 0])
        kb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE else 1, layout=layout)
    else:
        kb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE else 1)
    vb = tlx.local_alloc((64, 64), tl.bfloat16, 2 if PIPE else 1)
    first_q = block * (BM // GROUP) + SK - SQ
    last_q = tl.minimum(first_q + BM // GROUP, SK)
    first_page = 0
    if WINDOW:
        first_page = tl.maximum(0, (first_q - WINDOW + 1) // 64)
    end_page = tl.cdiv(last_q, 64)
    maximum = tl.full([BM], -float("inf"), tl.float32)
    if FIXED:
        denom = tl.zeros([BM, 16], tl.float32)
    else:
        denom = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, 64], tl.float32)
    if FIXED and DEEP:
        use_deep = end_page - first_page >= 3
    else:
        use_deep: tl.constexpr = False
    if use_deep:
        phys0 = tl.load(Tables + batch * TABLE_STRIDE + first_page)
        phys1 = tl.load(Tables + batch * TABLE_STRIDE + first_page + 1)
        phys2 = tl.load(Tables + batch * TABLE_STRIDE + first_page + 2)
        _put_k_id(kd, kb, 0, phys0)
        _put_k_id(kd, kb, 1, phys1)
        _put_v_id(vd, vb, 0, phys0)
        tlx.async_amd_descriptor_wait(3)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        tlx.workgroup_barrier()
        k = _get_k(kb, 0, 2)
        tlx.workgroup_barrier()
        p, denom = _prob(
            q,
            k,
            denom,
            rows,
            first_page,
            first_q,
            last_q,
            SQ,
            SK,
            WINDOW,
            BM,
            SPLIT_MASK,
        )
        _put_k_id(kd, kb, 0, phys2)
        _put_v_id(vd, vb, 1, phys1)
        k = _get_k(kb, 1, 3)
        slot = 0
        for page in range(first_page, end_page - 3):
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + page + 3)
            # With <=64 distinct query positions, every key in this
            # prologue-excluded iteration is strictly before the earliest query.
            # A tile aligned within a KV page has at most one partially
            # visible page at the left SWA boundary; the prologue handled it.
            if BM // GROUP <= 64 and (WINDOW == 0 or (64 % (BM // GROUP) == 0 and (SK - SQ) %
                                                      (BM // GROUP) == 0 and WINDOW % 64 == 0)):
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
            v = _get_v(vb, slot, 2)
            _put_k_id(kd, kb, 1 - slot, next_phys)
            denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
            acc = _dot(p, v, acc)
            p, denom = _finish(next_scores, denom, BM, q.dtype)
            k = _get_k(kb, slot, 2)
            _put_v_id(vd, vb, slot, phys2)
            phys2 = next_phys
            slot = 1 - slot
        v = _get_v(vb, slot, 2)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
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
        k = _get_k(kb, slot, 1)
        _put_v_id(vd, vb, slot, phys2)
        v = _get_v(vb, 1 - slot, 1)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
        p, denom = _prob(
            q,
            k,
            denom,
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
        v = _get_v(vb, slot, 0)
        if SK % 64:
            v = tl.where(((end_page - 1) * 64 + tl.arange(0, 64))[:, None] < SK, v, 0.0)
        denom = _dot(p, tl.full((64, 16), 1.0, q.dtype), denom)
        acc = _dot(p, v, acc)
    elif PIPE:
        phys = tl.load(Tables + batch * TABLE_STRIDE + first_page)
        tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * 64, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * 64, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_wait(2)
        q = tlx.local_load(tlx.local_reshape(tlx.local_view(qb, 0), [BM, 64]))
        tlx.workgroup_barrier()
        for page in range(first_page, end_page):
            slot = (page - first_page) % 2
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + page + 1, page + 1 < end_page, 0)
            tlx.async_amd_descriptor_load(
                kd,
                tlx.local_view(kb, 1 - slot),
                [next_phys * 64, 0],
                clamp_bounds=False,
            )
            tlx.async_amd_descriptor_load(
                vd,
                tlx.local_view(vb, 1 - slot),
                [next_phys * 64, 0],
                clamp_bounds=False,
            )
            tlx.async_amd_descriptor_wait(2)
            k = tlx.local_load(tlx.local_trans(tlx.local_view(kb, slot)))
            v = tlx.local_load(tlx.local_view(vb, slot))
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
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * 64, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * 64, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_wait(0)
            k = tlx.local_load(tlx.local_trans(tlx.local_view(kb, 0)))
            v = tlx.local_load(tlx.local_view(vb, 0))
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
