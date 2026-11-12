"""BF16 paged causal 16Q/2KV D64 prefill for gfx1250 TLX.

Q is contiguous [B,S,16,64]; K/V are [physical_pages,64,2,64]. The
int32 page table is [B,logical_pages] and must contain valid physical IDs.
All sequences in a batch share Q/KV lengths. Window 2048 includes self;
window 0 means global. Cached prefixes use bottom-right causal alignment.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


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
    scores = tl.dot(q, k) * (0.125 * 1.4426950408889634)
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
        parts = tl.reshape(p, [BM, 2, 4, 8])
        denom += tl.sum(tl.sum(parts, 3), 1)
        acc = tl.dot(p.to(q.dtype), v, acc)
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
        acc = tl.dot(p.to(q.dtype), v, acc)
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
    scores = tl.dot(q, k) * (0.125 * 1.4426950408889634)
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
    parts = tl.reshape(p, [BM, 2, 4, 8])
    denom += tl.sum(tl.sum(parts, 3), 1)
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
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=True)


@triton.jit
def _put_v(desc, buf, slot, tables, batch, page, STRIDE: tl.constexpr):
    phys = tl.load(tables + batch * STRIDE + page)
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=True)


@triton.jit
def _put_k_id(desc, buf, slot, phys):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=True)


@triton.jit
def _put_v_id(desc, buf, slot, phys):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [phys * 64, 0], clamp_bounds=True)


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
    batch = tl.program_id(2)
    block = tl.program_id(0)
    if REVERSE:
        block = tl.cdiv(SQ * GROUP, BM) - 1 - block
    head = tl.program_id(1) * GROUP
    kv_head = head // 8
    row_ids = block * BM + tl.arange(0, BM)
    rows = row_ids // GROUP
    heads = head + row_ids % GROUP
    dims = tl.arange(0, 64)
    offsets = ((batch.to(tl.int64) * SQ + rows[:, None]) * 16 + heads[:, None]) * 64 + dims[None, :]
    q = tl.load(Q + offsets, rows[:, None] < SQ, 0)
    kd = tl.make_tensor_descriptor(K + kv_head * 64, [NPAGES * 64, 64], [128, 1], [64, 64])
    vd = tl.make_tensor_descriptor(V + kv_head * 64, [NPAGES * 64, 64], [128, 1], [64, 64])
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
        denom = tl.zeros([BM, 4], tl.float32)
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
            next_scores = _scores(q, k, rows, page + 1, first_q, last_q, SQ, SK, WINDOW, BM, SPLIT_MASK)
            v = _get_v(vb, slot, 2)
            _put_k_id(kd, kb, 1 - slot, next_phys)
            acc = tl.dot(p, v, acc)
            p, denom = _finish(next_scores, denom, BM, q.dtype)
            k = _get_k(kb, slot, 2)
            _put_v_id(vd, vb, slot, phys2)
            phys2 = next_phys
            slot = 1 - slot
        v = _get_v(vb, slot, 2)
        acc = tl.dot(p, v, acc)
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
        acc = tl.dot(p, v, acc)
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
        acc = tl.dot(p, v, acc)
    elif PIPE:
        phys = tl.load(Tables + batch * TABLE_STRIDE + first_page)
        tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * 64, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * 64, 0], clamp_bounds=True)
        for page in range(first_page, end_page):
            slot = (page - first_page) % 2
            next_phys = tl.load(Tables + batch * TABLE_STRIDE + page + 1, page + 1 < end_page, 0)
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 1 - slot), [next_phys * 64, 0], clamp_bounds=True)
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 1 - slot), [next_phys * 64, 0], clamp_bounds=True)
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
        for page in range(first_page, end_page):
            phys = tl.load(Tables + batch * TABLE_STRIDE + page)
            tlx.async_amd_descriptor_load(kd, tlx.local_view(kb, 0), [phys * 64, 0], clamp_bounds=True)
            tlx.async_amd_descriptor_load(vd, tlx.local_view(vb, 0), [phys * 64, 0], clamp_bounds=True)
            tlx.async_amd_descriptor_wait(0)
            k = tlx.local_load(tlx.local_trans(tlx.local_view(kb, 0)))
            v = tlx.local_load(tlx.local_view(vb, 0))
            tlx.workgroup_barrier()
            acc, maximum, denom = _consume(q, k, v, acc, maximum, denom, rows, page * 64, SQ, SK, WINDOW, BM, FIXED)
    if FIXED:
        denom = tl.sum(denom, 1)
    result = acc / tl.maximum(denom[:, None], 1.0e-30)
    tl.store(Out + offsets, result.to(Out.dtype.element_ty), rows[:, None] < SQ)
    if FIXED:
        bad = (denom < 1.0e-20) | ~(denom < 1.0e20)
        bad |= tl.sum((~(tl.abs(acc) < 3.402823466e38)).to(tl.int32), 1) > 0
        return tl.sum(((rows < SQ) & bad).to(tl.int32), 0) > 0
    return False


@triton.jit(noinline=True)
def _replay(
    Q,
    K,
    V,
    Tables,
    Out,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    NPAGES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    PIPE: tl.constexpr,
    PAD: tl.constexpr,
    SPLIT_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    _pass(
        Q,
        K,
        V,
        Tables,
        Out,
        SQ,
        SK,
        NPAGES,
        TABLE_STRIDE,
        WINDOW,
        BM,
        GROUP,
        PIPE,
        PAD,
        SPLIT_MASK,
        REVERSE,
        False,
        False,
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
                TABLE_STRIDE,
                WINDOW,
                BM,
                GROUP,
                PIPE,
                PAD,
                SPLIT_MASK,
                REVERSE,
            )


def _launch(
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
    fast=False,
):
    # TDM requires aligned Q; unaligned contiguous views retain the general path.
    if fast and q.data_ptr() % 16 == 0:
        from .gfx1250_wmma import launch as fast_launch

        return fast_launch(
            q,
            k,
            v,
            tables,
            out,
            sk=sk,
            window=window,
            bm=bm,
            group=group,
            pipe=pipe,
            pad=pad,
            split_mask=split_mask,
            reverse=reverse,
            warps=warps,
            waves=waves,
            fixed=fixed,
            deep=deep,
            reverse_local_assignment=reverse_local_assignment,
            enable_wmma_queuing=enable_wmma_queuing,
            llvm_fn_attrs=llvm_fn_attrs,
        )
    b, sq, _, _ = q.shape
    sk = sq if sk is None else sk
    # A window covering the full KV length is exactly global causal attention.
    window = 0 if window and sk <= window else window
    return paged_prefill_kernel[(triton.cdiv(sq * group, bm), 16 // group, b)](
        q,
        k,
        v,
        tables,
        out,
        sq,
        sk,
        k.shape[0],
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


def paged_prefill(q, k, v, tables, *, kv_length=None, window=0):
    """Return BF16 causal attention; default configurations do not autotune.

    This is a uniform-length prefill API. It does not append to the KV cache.
    Callers provide valid page IDs and cache contents, including current tokens.
    """
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, tables)):
        raise InvalidInput("Expected tensor inputs")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise InvalidInput("Q/K/V must be BF16")
    if (q.ndim != 4 or q.shape[2:] != (16, 64) or k.ndim != 4 or k.shape[1:] != (64, 2, 64) or v.shape != k.shape):
        raise InvalidInput("Expected Q [B,S,16,64] and K/V [pages,64,2,64]")
    if not all(t.is_contiguous() for t in (q, k, v, tables)):
        raise InvalidInput("Expected contiguous tensors")
    if any(t.device != q.device for t in (k, v, tables)) or not q.is_cuda:
        raise InvalidInput("Inputs must share one GPU")
    if min(q.shape) <= 0 or k.shape[0] <= 0 or tables.ndim != 2:
        raise InvalidInput("Expected nonempty tensors and a rank-2 page table")
    if k.data_ptr() % 16 or v.data_ptr() % 16:
        raise InvalidInput("K/V bases must be 16-byte aligned for TDM")
    sk = q.shape[1] if kv_length is None else kv_length
    if (type(sk) is not int or sk < q.shape[1] or type(window) is not int or window not in (0, 2048)
            or tables.shape[0] != q.shape[0] or tables.shape[1] < triton.cdiv(sk, 64)):
        raise InvalidInput("Invalid KV length, window or page table shape")
    if tables.dtype != torch.int32:
        raise InvalidInput("Page table must be int32")
    config = _select_config(q.shape[1], sk, window)
    out = torch.empty_like(q)
    _launch(q, k, v, tables, out, sk=sk, window=window, **config)
    return out


def _select_config(query_length, kv_length, window):
    """Frozen choices from the BF16 prefill sweep; q_len=8 is not tuned."""
    effective_window = 0 if window and kv_length <= window else window
    if query_length > 4096 or (query_length > 768 and effective_window == 0):
        return dict(
            pipe=True,
            pad=8,
            reverse=True,
            fixed=True,
            split_mask=True,
            group=8,
            deep=True,
            fast=True,
            enable_wmma_queuing=False,
            bm=(64 if query_length <= 4096 else (128 if effective_window or query_length <= 8192 else 256)),
            warps=2 if effective_window or query_length <= 8192 else 4,
            reverse_local_assignment=True,
            llvm_fn_attrs="amdgpu-sched-strategy=coexec",
        )
    base = dict(pipe=True, pad=8, reverse=True, fixed=True, split_mask=True, warps=4)
    if query_length < 256:
        return dict(base, bm=32, group=1, warps=2, fixed=False, split_mask=False)
    if query_length <= 768:
        return dict(base, bm=64, group=1, deep=False)
    if query_length <= 4096:
        return dict(
            base,
            bm=128,
            group=4,
            deep=False,
            enable_wmma_queuing=True,
            llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp",
        )
