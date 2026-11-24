"""sm100 paged attention over packed E4M3FN K/V with per-KV-head scales.

Cache layout
------------
K is [blocks, kv_heads, D/16, page_size, 16] and V is [blocks, kv_heads, page_size/16, D, 16]; ``page_size`` is 64
or 128. One page of one KV head is contiguous in both, and its bytes are already tcgen05's no-swizzle K-major order
for S^T = K . Q^T (K as [page, D], an FP8 operand). Each page of K and V lands with one TMA box of 256-byte rows in
an unswizzled buffer; K is read through an explicit ``shared_linear`` view, and V is converted to FP16 in the
16-bit form of the same order for O^T = V^T . P^T (V^T as [D, page]).

Execution
---------
A CTA handles one slice of one (request, row tile, KV head); a row tile is up to 64 (query position, query head)
rows of one KV head's group. Four warp-specialized roles:

* load warp: the slice's page ids first (one round trip), then K and V for every page, each page completing its
  own barrier. Pages past the slice get an out-of-range coordinate, which TMA fills with zeros.
* MMA warp: S^T per page as soon as it lands (FP8; BF16 Q rows are split exactly into two E4M3 terms with a
  power-of-two row scale, accumulated by two MMAs), and O^T = V^T . P^T per page in FP16 as soon as the softmax
  publishes that page's P.
* convert warps: V^T of every page to FP16 as soon as it lands (tokens past the slice zeroed, since cache tails
  may hold NaN), in parallel with QK and the softmax.
* softmax warps: the row max over the chunk's pages, then P per page as FP16 scaled by 2^15, which keeps 11 bits
  down to probabilities of 2^-39 relative to the row max. PV on page i overlaps the softmax of page i+1. Long
  slices loop over chunks with an online softmax.

Slices of one row tile merge in a thread-block cluster through distributed shared memory when they fit one (at most
eight); otherwise every slice writes a normalized FP32 partial and log2-sum-exp to the workspace and the last one
to finish (a per-row-tile counter) merges them. A row tile's partials are contiguous, so the merging slice brings
them into shared memory with bulk copies. The counters live in a per-device, per-stream buffer that every merge
resets, so workspaces stay opaque and graph-replay safe. Automatic split counts stay within one wave of CTAs.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

_SMEM_BUDGET = 200 * 1024
_TMEM_COLUMNS = 512
_MAX_AUTO_SPLITS = 128
_COUNTERS: dict = {}
_SMS: dict = {}
_MERGE_BUDGET = 80 * 1024


@triton.jit
def _slice_bounds(LEN, b, s, QL: tl.constexpr, WINDOW: tl.constexpr, PAGE: tl.constexpr, GR: tl.constexpr,
                  NS: tl.constexpr):
    length = tl.load(LEN + b).to(tl.int32)
    if WINDOW > 0:
        lo = tl.maximum(length - QL - WINDOW + 1, 0)
    else:
        lo = 0
    lo_p = lo // PAGE * PAGE
    per = tl.maximum(tl.cdiv(tl.cdiv(length - lo_p, NS), GR) * GR, GR)
    n_eff = tl.maximum(tl.cdiv(length - lo_p, per), 1)
    start = lo_p + s * per
    end = tl.minimum(start + per, length)
    return length, lo, start, end, n_eff


@triton.jit
def _decode_kernel(Q, stride_qt, stride_qh, stride_qd, desc_k, desc_v, KS, VS, QS, BT, stride_bt, LEN, PART, LSE, CNT,
                   OUT, sm_scale, KROWS, D: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, QL: tl.constexpr,
                   PAGE: tl.constexpr, ROWS: tl.constexpr, RN: tl.constexpr, NPG: tl.constexpr, TA: tl.constexpr,
                   NS: tl.constexpr, WS: tl.constexpr, MS: tl.constexpr, CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
                   CLUSTER: tl.constexpr, MULTI: tl.constexpr, Q_FP8: tl.constexpr, KLAY: tl.constexpr,
                   KBOX: tl.constexpr, V16: tl.constexpr, V16T: tl.constexpr, ROWMAJ: tl.constexpr):
    G: tl.constexpr = HQ // HKV
    TILES: tl.constexpr = (ROWS + RN - 1) // RN
    NP: tl.constexpr = NPG
    CHUNK: tl.constexpr = NPG * PAGE
    # A cluster balances its slices by page; otherwise whole chunks keep the active slices (and the merge) few.
    GR: tl.constexpr = PAGE if CLUSTER else CHUNK
    R: tl.constexpr = PAGE * D // 256
    b = tl.program_id(0) // TILES
    tile = tl.program_id(0) % TILES
    h = tl.program_id(1)
    s = tl.program_id(2)
    # Launched with PDL: lengths, page table, Q and the KV just written may come from earlier kernels, and the next
    # kernel may start launching right away (it still waits for this one before reading the output).
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    # The context length and the first chunk's page ids are loaded first, so their two dependent round trips overlap
    # the prologue (allocations, TMEM, barrier init, cluster rendezvous).
    length, lo, start, end, n_eff = _slice_bounds(LEN, b, s, QL, WINDOW, PAGE, GR, NS)
    if not CLUSTER:
        # The split capacity covers max_context_len; slices past this request's context leave before allocating.
        if s >= n_eff:
            return
    pages0 = ()
    for i in tl.static_range(NPG):
        pages0 = pages0 + (tl.load(BT + b.to(tl.int64) * stride_bt +
                                   (start + i * PAGE) // PAGE, mask=start + i * PAGE < end, other=0).to(tl.int32), )
    tlx.prefetch(desc_k, tensormap=True)
    tlx.prefetch(desc_v, tensormap=True)

    q_s0 = tlx.local_alloc((RN, D), tl.float8e4nv, 1)
    if not Q_FP8:
        q_s1 = tlx.local_alloc((RN, D), tl.float8e4nv, 1)
    k3 = tlx.local_alloc((1, R, 256), tl.float8e4nv, NP, layout=KBOX)
    v3 = tlx.local_alloc((1, R, 256), tl.float8e4nv, NP, layout=KBOX)
    # FP16 PV operands: V^T per page and P as [rows, tokens].
    vt16 = tlx.local_alloc((1, R, 256), tl.float16, NP, layout=V16)
    p16 = tlx.local_alloc((RN, TA), tl.float16, 1)
    s_t = tlx.local_alloc((PAGE, RN), tl.float32, NP, tlx.storage_kind.tmem)
    o_t = tlx.local_alloc((D, RN), tl.float32, 1, tlx.storage_kind.tmem)
    kv_full = tlx.alloc_barriers(num_barriers=NP)
    qk_full = tlx.alloc_barriers(num_barriers=NP)
    p_full = tlx.alloc_barriers(num_barriers=NP)
    v_ready = tlx.alloc_barriers(num_barriers=NP)
    q_full = tlx.alloc_barriers(num_barriers=1)
    o_full = tlx.alloc_barriers(num_barriers=1)
    # Chunk-to-chunk reuse: K/V buffers free (MMAs done), S read, O read.
    kv_empty = tlx.alloc_barriers(num_barriers=1)
    s_empty = tlx.alloc_barriers(num_barriers=1)
    o_empty = tlx.alloc_barriers(num_barriers=1)
    if not CLUSTER and NS > 1:
        # The merging slice copies MS workspace slots (partials and log-sum-exps) per round.
        mrg_o = tlx.local_alloc((RN * D, ), tl.float32, MS)
        mrg_l = tlx.local_alloc((RN, ), tl.float32, MS)
        mrg_bar = tlx.alloc_barriers(num_barriers=1)
    if CLUSTER:
        # Ranks 1.. send their unnormalized O^T, max and sum to rank 0 with st.async, counted on rank 0's mbarrier.
        # Peers write these at any time, so a store here keeps them live for the whole kernel.
        po_s = tlx.local_alloc((D, RN), tl.float32, NS)
        pm_s = tlx.local_alloc((1, RN), tl.float32, NS)
        pl_s = tlx.local_alloc((1, RN), tl.float32, NS)
        tlx.local_store(po_s[0], tl.zeros([D, RN], tl.float32))
        tlx.local_store(pm_s[0], tl.zeros([1, RN], tl.float32))
        tlx.local_store(pl_s[0], tl.zeros([1, RN], tl.float32))
        mbar = tlx.alloc_barriers(num_barriers=1)
        if s == 0:
            tlx.barrier_expect_bytes(mbar[0], (NS - 1) * (D + 2) * RN * 4)
        # The compiler publishes rank 0's mbarrier init to the peers (fence and relaxed cluster arrive/wait at kernel
        # entry) because the peers signal it with st.async.
        tlx.fence_mbarrier_init_cluster()

    # The first chunk's K/V copies go out from the prologue, once the page ids arrive.
    if CLUSTER or s < n_eff:
        for i in tl.static_range(NP):
            p0 = start + i * PAGE
            hp = tl.where(p0 < end, pages0[i] * HKV + h, KROWS)
            tlx.barrier_expect_bytes(kv_full[i], 2 * PAGE * D)
            tlx.async_descriptor_load(desc_k, k3[i], [hp, 0, 0], kv_full[i])
            tlx.async_descriptor_load(desc_v, v3[i], [hp, 0, 0], kv_full[i])
    with tlx.async_tasks():
        # softmax and epilogue
        with tlx.async_task("default"):
            offs_d = tl.arange(0, D)
            offs_t = tl.arange(0, PAGE)
            rows = tile * RN + tl.arange(0, RN)
            row_ok = rows < ROWS
            qi = rows // G
            head = h * G + rows % G
            token = b * QL + qi
            m = tl.full([RN], float("-inf"), tl.float32)
            l = tl.zeros([RN], tl.float32)
            acc = tl.zeros([D, RN], tl.float32)
            if CLUSTER or s < n_eff:
                q_ptrs = Q + token.to(
                    tl.int64)[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :] * stride_qd
                if Q_FP8:
                    tlx.local_store(q_s0[0], tl.load(q_ptrs, mask=row_ok[:, None], other=0.0))
                    q_scale = tl.load(QS + head, mask=row_ok, other=1.0)
                else:
                    # A power-of-two row scale puts the row maximum in [224, 448] and keeps every element a BF16
                    # value, whose 8 significant bits two E4M3 terms hold exactly (above the E4M3 subnormal range);
                    # both accumulate into one S.
                    qf = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0).to(tl.float32)
                    amax = tl.max(tl.abs(qf), axis=1) / 448.0
                    pow2 = ((amax.to(tl.int32, bitcast=True) & 0x7F800000) + 0x00800000).to(tl.float32, bitcast=True)
                    q_scale = tl.where(amax > 0, pow2, 1.0)
                    qn = qf / q_scale[:, None]
                    q0 = qn.to(tl.float8e4nv)
                    tlx.local_store(q_s0[0], q0)
                    tlx.local_store(q_s1[0], (qn - q0.to(tl.float32)).to(tl.float8e4nv))
                tlx.fence_async_shared()
                tlx.barrier_arrive(q_full[0])
                row_scale = sm_scale * tl.load(KS + h) * 1.4426950408889634 * q_scale
                qpos = length - QL + qi
                ci = 0
                for c0 in range(start, end, CHUNK):
                    ph = ci & 1
                    cend = tl.minimum(c0 + CHUNK, end)
                    scs = ()
                    mx = tl.full([PAGE, RN], float("-inf"), tl.float32)
                    for i in tl.static_range(NP):
                        tlx.barrier_wait(qk_full[i], ph)
                        sc = tlx.local_load(s_t[i]) * row_scale[None, :]
                        toks = c0 + i * PAGE + offs_t
                        ok = (toks[:, None] >= lo) & (toks[:, None] < cend) & row_ok[None, :]
                        if CAUSAL:
                            ok = ok & (toks[:, None] <= qpos[None, :])
                        if WINDOW > 0:
                            ok = ok & (toks[:, None] > qpos[None, :] - WINDOW)
                        sc = tl.where(ok, sc, float("-inf"))
                        mx = tl.maximum(mx, sc)
                        scs = scs + (sc, )
                    if MULTI:
                        tlx.barrier_arrive(s_empty[0])
                    m_new = tl.maximum(m, tl.max(mx, axis=0))
                    m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
                    alpha = tl.exp2(m - m_ref)
                    psum = tl.zeros([PAGE, RN], tl.float32)
                    for i in tl.static_range(NP):
                        p = tl.exp2(scs[i] - m_ref[None, :])
                        psum += p
                        tlx.local_store(tlx.local_slice(p16[0], [0, i * PAGE], [RN, PAGE]),
                                        tl.trans(p * 32768.0).to(tl.float16))
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(p_full[i])
                    if MULTI:
                        l = l * alpha + tl.sum(psum, axis=0)
                    else:
                        l = tl.sum(psum, axis=0)
                    m = m_new
                    tlx.barrier_wait(o_full[0], ph)
                    o_c = tlx.local_load(o_t[0]) * (1.0 / 32768)
                    if MULTI:
                        tlx.barrier_arrive(o_empty[0])
                        acc = acc * alpha[None, :] + o_c
                    else:
                        acc = o_c
                    ci += 1

            vs = tl.load(VS + h)
            vrow = tl.arange(0, D)[:, None]
            l_safe = tl.where(l > 0, l, 1.0)
            inv_l = tl.where(l > 0, 1.0 / l_safe, 0.0)
            row_id = token.to(tl.int64) * HQ + head
            o_rows = row_id[None, :] * D + vrow  # O^T [D, RN]
            o_mask = row_ok[None, :] & (vrow < D)
            if CLUSTER:
                if s != 0:
                    tlx.async_remote_shmem_store(dst=po_s[s], src=acc, remote_cta_rank=0, barrier=mbar[0])
                    tlx.async_remote_shmem_store(dst=pm_s[s], src=m[None, :], remote_cta_rank=0, barrier=mbar[0])
                    tlx.async_remote_shmem_store(dst=pl_s[s], src=l[None, :], remote_cta_rank=0, barrier=mbar[0])
                else:
                    tlx.barrier_wait(mbar[0], 0)
                    # Rank 0's own partial goes to slot 0, so every slot is read back in one layout. The max comes
                    # first, so every weight is independent and the loads overlap.
                    tlx.local_store(po_s[0], acc)
                    tlx.local_store(pm_s[0], m[None, :])
                    tlx.local_store(pl_s[0], l[None, :])
                    mx = tlx.local_load(pm_s[0])
                    for r in tl.static_range(1, NS):
                        mx = tl.maximum(mx, tlx.local_load(pm_s[r]))
                    m_ref = tl.where(mx == float("-inf"), 0.0, mx)
                    w = tl.exp2(tlx.local_load(pm_s[0]) - m_ref)
                    acc = tlx.local_load(po_s[0]) * w
                    l2 = tlx.local_load(pl_s[0]) * w
                    for r in tl.static_range(1, NS):
                        w = tl.exp2(tlx.local_load(pm_s[r]) - m_ref)
                        acc += tlx.local_load(po_s[r]) * w
                        l2 += tlx.local_load(pl_s[r]) * w
                    o = acc * (vs * tl.where(l2 > 0, 1.0 / tl.where(l2 > 0, l2, 1.0), 0.0))
                    tl.store(OUT + o_rows, o.to(tl.bfloat16), mask=o_mask)
            elif NS == 1:
                tl.store(OUT + o_rows, (acc * (inv_l * vs)[None, :]).to(tl.bfloat16), mask=o_mask)
            elif s < n_eff:
                # Partials are [row tile, slot, RN, D] and log-sum-exps [row tile, slot, RN].
                rt = (b.to(tl.int64) * TILES + tile) * HKV + h
                rr = tl.arange(0, RN)
                tl.store(PART + ((rt * WS + s) * RN + rr[None, :]) * D + vrow, acc * inv_l[None, :], mask=o_mask)
                tl.store(LSE + (rt * WS + s) * RN + rr, tl.where(l > 0, m + tl.log2(l_safe), float("-inf")),
                         mask=row_ok)
                tl.debug_barrier()
                counter = CNT + rt
                done = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
                if done == n_eff - 1:
                    # Rounds of MS slots, each brought into shared memory with two bulk copies; the reference moves
                    # with the running max. Everything stays in the output's row layout, so the merge needs no
                    # cross-thread reductions. Workspaces hold whole rounds, so every copy stays in its row tile;
                    # slots past n_eff are excluded with selects, since they may hold anything.
                    ref = tl.full([RN], float("-inf"), tl.float32)
                    wsum = tl.zeros([RN], tl.float32)
                    o = tl.zeros([RN, D], tl.float32)
                    mph = 0
                    for c0 in range(0, n_eff, MS):
                        if c0 > 0:
                            # The previous round's reads finish before the copies overwrite its buffers.
                            tl.debug_barrier()
                            tlx.fence_async_shared()
                        tlx.barrier_expect_bytes(mrg_bar[0], MS * (RN * D + RN) * 4)
                        tlx.async_load(PART + (rt * WS + c0) * (RN * D), mrg_o[0], bulk=True,
                                       bulk_size=c0 * 0 + MS * RN * D * 4, barrier=mrg_bar[0])
                        tlx.async_load(LSE + (rt * WS + c0) * RN, mrg_l[0], bulk=True, bulk_size=c0 * 0 + MS * RN * 4,
                                       barrier=mrg_bar[0])
                        tlx.barrier_wait(mrg_bar[0], mph)
                        lses = ()
                        m_r = ref
                        for j in tl.static_range(MS):
                            lj = tl.where(c0 + j < n_eff, tlx.local_load(mrg_l[j]), float("-inf"))
                            lses = lses + (lj, )
                            m_r = tl.maximum(m_r, lj)
                        r_safe = tl.where(m_r == float("-inf"), 0.0, m_r)
                        scale = tl.exp2(ref - r_safe)
                        o = o * scale[:, None]
                        wsum = wsum * scale
                        for j in tl.static_range(MS):
                            w = tl.exp2(lses[j] - r_safe)
                            part = tlx.local_load(tlx.local_reinterpret(mrg_o[j], tl.float32, [RN, D], layout=ROWMAJ))
                            o += tl.where(c0 + j < n_eff, part * w[:, None], 0.0)
                            wsum += w
                        ref = m_r
                        mph ^= 1
                    o = o * (vs / tl.where(wsum > 0, wsum, 1.0))[:, None]
                    tl.store(OUT + row_id[:, None] * D + offs_d[None, :], o.to(tl.bfloat16), mask=row_ok[:, None])
                    tl.atomic_xchg(counter, 0, sem="relaxed")

        # V^T to FP16, page by page as it lands
        with tlx.async_task(num_warps=4):
            if CLUSTER or s < n_eff:
                # Token of each element of a page's [1, R, 256] rows ([page/16][D][16] bytes).
                vtok = (tl.arange(0, R) // (D // 16))[:, None] * 16 + (tl.arange(0, 256) % 16)[None, :]
                ci = 0
                for c0 in range(start, end, CHUNK):
                    ph = ci & 1
                    if MULTI and ci > 0:
                        # The previous chunk's PV (the last reader of vt16) is done.
                        tlx.barrier_wait(kv_empty[0], ph ^ 1)
                    cend = tl.minimum(c0 + CHUNK, end)
                    for i in tl.static_range(NP):
                        tlx.barrier_wait(kv_full[i], ph)
                        p0 = c0 + i * PAGE
                        # Pages past the slice are skipped by PV too; only the page holding the slice end is masked.
                        if p0 + PAGE <= cend:
                            tlx.local_store(vt16[i], tlx.local_load(v3[i]).to(tl.float16))
                            tlx.fence_async_shared()
                        elif p0 < cend:
                            vt = tlx.local_load(v3[i]).to(tl.float16)
                            tlx.local_store(vt16[i], tl.where((p0 + vtok < cend)[None, :, :], vt, 0.0))
                            tlx.fence_async_shared()
                        tlx.barrier_arrive(v_ready[i])
                    ci += 1

        # loads
        with tlx.async_task(num_warps=1):
            if CLUSTER or s < n_eff:
                ci = 1
                for c0 in range(start + CHUNK, end, CHUNK):
                    tlx.barrier_wait(kv_empty[0], (ci - 1) & 1)
                    # Page ids are loaded before any TMA so they cost one round trip.
                    pages = ()
                    for i in tl.static_range(NP):
                        p0 = c0 + i * PAGE
                        pages = pages + (tl.load(BT + b.to(tl.int64) * stride_bt + p0 // PAGE, mask=p0 < end,
                                                 other=0).to(tl.int32), )
                    for i in tl.static_range(NP):
                        p0 = c0 + i * PAGE
                        hp = tl.where(p0 < end, pages[i] * HKV + h, KROWS)
                        tlx.barrier_expect_bytes(kv_full[i], 2 * PAGE * D)
                        tlx.async_descriptor_load(desc_k, k3[i], [hp, 0, 0], kv_full[i])
                        tlx.async_descriptor_load(desc_v, v3[i], [hp, 0, 0], kv_full[i])
                    ci += 1

        # MMAs
        with tlx.async_task(num_warps=1):
            if CLUSTER or s < n_eff:
                tlx.barrier_wait(q_full[0], 0)
                ci = 0
                for c0 in range(start, end, CHUNK):
                    ph = ci & 1
                    if MULTI and ci > 0:
                        tlx.barrier_wait(s_empty[0], ph ^ 1)
                    for i in tl.static_range(NP):
                        tlx.barrier_wait(kv_full[i], ph)
                        k_op = tlx.local_reinterpret(k3[i], tl.float8e4nv, [PAGE, D], layout=KLAY)
                        if Q_FP8:
                            tlx.async_dot(k_op, tlx.local_trans(q_s0[0]), s_t[i], use_acc=False, mBarriers=[qk_full[i]])
                        else:
                            tlx.async_dot(k_op, tlx.local_trans(q_s0[0]), s_t[i], use_acc=False, force_async=True)
                            tlx.async_dot(k_op, tlx.local_trans(q_s1[0]), s_t[i], use_acc=True, mBarriers=[qk_full[i]])
                    if MULTI and ci > 0:
                        tlx.barrier_wait(o_empty[0], ph ^ 1)
                    cend = tl.minimum(c0 + CHUNK, end)
                    for i in tl.static_range(NP):
                        tlx.barrier_wait(v_ready[i], ph)
                        tlx.barrier_wait(p_full[i], ph)
                        if c0 + i * PAGE < cend:
                            tlx.async_dot(tlx.local_reinterpret(vt16[i], tl.float16, [D, PAGE], layout=V16T),
                                          tlx.local_trans(tlx.local_slice(p16[0], [0, i * PAGE], [RN, PAGE])), o_t[0],
                                          use_acc=i > 0, force_async=True)
                    tlx.tcgen05_commit(o_full[0])
                    if MULTI:
                        tlx.tcgen05_commit(kv_empty[0])
                    ci += 1


def _interleaved_kmajor(rows, cols, inner=16):
    """shared_linear layout of a [rows, cols] tile stored as [cols / inner][rows][inner] (16 bytes per inner run)."""
    bases = [[0, 1 << i] for i in range(inner.bit_length() - 1)]
    bases += [[1 << i, 0] for i in range(rows.bit_length() - 1)]
    bases += [[0, inner << i] for i in range((cols // inner).bit_length() - 1)]
    return tlx.shared_linear_layout_encoding(bases)


def _v16_rows(page, dim):
    """shared_linear layout for a page of V converted to FP16 in place of its packed [1, R, 256] FP8 rows: element b
    (token t, dim d, b = (t // 16) * 16 * dim + d * 16 + t % 16) goes to (t // 8) * 8 * dim + d * 8 + t % 8, the
    16-bit no-swizzle K-major order of V^T. That is a permutation of b's bits: token bit 3 moves above the dim bits."""
    log_d = dim.bit_length() - 1
    perm = list(range(3)) + [j + 1 for j in range(3, 3 + log_d)] + [3]
    perm += list(range(4 + log_d, (page * dim).bit_length() - 1))
    return tlx.shared_linear_layout_encoding([[0, 0, 1 << k] if k < 8 else [0, 1 << (k - 8), 0] for k in perm])


def _unswizzled(shape):
    rank = len(shape)
    order = list(reversed(range(rank)))
    return tlx.nv_mma_shared_layout_encoding(shape, order, tl.float8e4nv, [1] * rank, [1] * rank, order, False, False)


def _bound(max_context_len, query_length, sliding_window, page):
    """Tokens a request can attend to, plus a page of alignment slack."""
    return (min(max_context_len, sliding_window + query_length - 1) if sliding_window else max_context_len) + page - 1


def _merge_slots(splits, rn, dim):
    """Workspace slots the merging slice copies into shared memory per round."""
    return max(1, min(splits, _MERGE_BUDGET // (rn * dim * 4)))


def _fits(npg, page, dim, rn, q_fp8, cluster, splits):
    ta = triton.next_power_of_2(npg * page)
    smem = rn * dim * (1 if q_fp8 else 2) + 4 * npg * page * dim + 2 * rn * ta
    if cluster:
        smem += splits * (dim + 2) * rn * 4
    elif splits > 1:
        smem += _merge_slots(splits, rn, dim) * (dim + 1) * rn * 4
    return smem <= _SMEM_BUDGET and (npg + 1) * max(rn, 32) <= _TMEM_COLUMNS


def _plan(query, key_cache, *, max_context_len, query_length, sliding_window, num_splits):
    """(row tile, pages per chunk, splits, cluster). Automatic plans run a short bound as one cluster of up to eight
    slices of at most 512 tokens; longer bounds use chunks of 256 tokens and up to one wave of slices (at most 128)
    merged through the workspace. An explicit split count is kept and only sizes the chunks."""
    tokens, heads, dim = query.shape
    page = key_cache.shape[3]
    if page not in (64, 128):
        raise ValueError(f"sm100 supports page_size 64 and 128, got {page}")
    rows = query_length * (heads // key_cache.shape[1])
    rn = min(64, max(8, triton.next_power_of_2(rows)))
    q_fp8 = query.dtype == torch.float8_e4m3fn
    pages = triton.cdiv(_bound(max_context_len, query_length, sliding_window, page), page)
    max_npg = 512 // page
    # Slices past a request's context exit at once, but every launched CTA still costs a launch and a prologue, and
    # a second wave of them outlasts the real work.
    if query.device not in _SMS:
        _SMS[query.device] = torch.cuda.get_device_properties(query.device).multi_processor_count
    row_tiles = tokens // query_length * triton.cdiv(rows, rn) * key_cache.shape[1]
    max_splits = min(_MAX_AUTO_SPLITS, max(1, _SMS[query.device] // row_tiles))
    if num_splits is None:
        npg = max(128 // page, triton.cdiv(pages, 8))
        splits = triton.cdiv(pages, npg)
        cluster = npg <= max_npg and splits <= 8
        if not cluster:
            npg = 256 // page
            splits = min(max_splits, triton.cdiv(pages, npg))
    else:
        splits = num_splits
        npg = min(max_npg, max(1, triton.cdiv(pages, splits)))
        cluster = 1 < splits <= 8
    if cluster and splits > 1 and not _fits(npg, page, dim, rn, q_fp8, True, splits):
        cluster = False
        if num_splits is None:
            npg = 256 // page
            splits = min(max_splits, triton.cdiv(pages, npg))
    cluster = cluster and splits > 1
    while npg > 1 and not _fits(npg, page, dim, rn, q_fp8, cluster, splits):
        npg -= 1
    if not _fits(npg, page, dim, rn, q_fp8, cluster, splits):
        raise ValueError("no sm100 decode plan fits shared and tensor memory for this geometry")
    return rn, npg, splits, cluster


def _merge_layout(query, key_cache, query_length, rn, splits):
    """(row tiles, slots per row tile) of the workspace as the kernel addresses it: partials [row tile, slot, RN, D]
    and log-sum-exps [row tile, slot, RN], in whole rounds of merge slots so the merging slice's fixed-size copies
    stay inside its row tile."""
    tokens, heads, dim = query.shape
    rows = query_length * (heads // key_cache.shape[1])
    ms = _merge_slots(splits, rn, dim)
    return tokens // query_length * triton.cdiv(rows, rn) * key_cache.shape[1], triton.cdiv(splits, ms) * ms


def _counters(device, count):
    """Zeroed int32 merge counters for the current stream; every merge resets its counter to zero."""
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < count:
        buf = torch.zeros(max(count, 1024), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


def paged_attention_decode_fp8_workspace(query, key_cache, value_cache, block_tables, *, max_context_len,
                                         query_length=1, sliding_window=0, num_splits=None):
    rn, _, splits, cluster = _plan(query, key_cache, max_context_len=max_context_len, query_length=query_length,
                                   sliding_window=sliding_window, num_splits=num_splits)
    if splits == 1:
        return None
    tokens, heads, dim = query.shape
    row_tiles, slots = _merge_layout(query, key_cache, query_length, rn, splits)
    if not cluster:
        _counters(query.device, row_tiles)
    # The public [tokens, heads, splits', D] shape, with splits' large enough to hold the kernel's layout.
    per = triton.cdiv(row_tiles * slots * rn, tokens * heads)
    return (torch.empty((tokens, heads, per, dim), dtype=torch.float32,
                        device=query.device), torch.empty((tokens, heads, per), dtype=torch.float32,
                                                          device=query.device))


def paged_attention_decode_fp8(query, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale, *,
                               max_context_len, q_scale=None, query_length=1, causal=True, sliding_window=0,
                               sm_scale=None, num_splits=None, out=None, workspace=None):
    tokens, heads, dim = query.shape
    blocks, hkv, page = key_cache.shape[0], key_cache.shape[1], key_cache.shape[3]
    rn, npg, splits, cluster = _plan(query, key_cache, max_context_len=max_context_len, query_length=query_length,
                                     sliding_window=sliding_window, num_splits=num_splits)
    if out is None:
        out = torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)
    rows = query_length * (heads // hkv)
    batch = tokens // query_length
    tiles = triton.cdiv(rows, rn)
    q_fp8 = query.dtype == torch.float8_e4m3fn
    if splits > 1 and not cluster:
        if workspace is None:
            workspace = paged_attention_decode_fp8_workspace(query, key_cache, value_cache, block_tables,
                                                             max_context_len=max_context_len, query_length=query_length,
                                                             sliding_window=sliding_window, num_splits=num_splits)
        part, lse = workspace
        row_tiles, ws = _merge_layout(query, key_cache, query_length, rn, splits)
        if part.numel() < row_tiles * ws * rn * dim or lse.numel() < row_tiles * ws * rn:
            raise ValueError("workspace is too small for this plan; allocate it with the matching workspace call")
        cnt = _counters(query.device, batch * tiles * hkv)
    else:
        part = lse = cnt = out
        ws = 1
    sm_scale = dim**-0.5 if sm_scale is None else sm_scale
    krows = blocks * hkv
    r = page * dim // 256
    desc_k = TensorDescriptor(key_cache.view(krows, r, 256), [krows, r, 256], [r * 256, 256, 1], [1, r, 256])
    desc_v = TensorDescriptor(value_cache.view(krows, r, 256), [krows, r, 256], [r * 256, 256, 1], [1, r, 256])
    _decode_kernel[(batch * tiles, hkv, splits)](
        query, query.stride(0), query.stride(1),
        query.stride(2), desc_k, desc_v, k_scale, v_scale, q_scale if q_fp8 else k_scale, block_tables,
        block_tables.stride(0), context_lens, part, lse, cnt, out, sm_scale, krows, D=dim, HQ=heads, HKV=hkv,
        QL=query_length, PAGE=page, ROWS=rows, RN=rn, NPG=npg, TA=triton.next_power_of_2(npg * page), NS=splits, WS=ws,
        MS=_merge_slots(splits, rn, dim) if splits > 1 else 1, CAUSAL=causal, WINDOW=sliding_window, CLUSTER=cluster,
        MULTI=splits * npg * page < _bound(max_context_len, query_length, sliding_window, page), Q_FP8=q_fp8,
        KLAY=_interleaved_kmajor(page, dim), KBOX=_unswizzled([1, r, 256]), V16=_v16_rows(page, dim),
        ROWMAJ=tlx.swizzled_shared_layout_encoding.make_default(2), V16T=_interleaved_kmajor(dim, page, 8), num_warps=4,
        num_stages=1, ctas_per_cga=(1, 1, splits) if cluster else None, launch_pdl=True)
    return out


def _per_head(scale, heads):
    """A [heads] FP32 scale vector: one-element (per-tensor) scales broadcast to every head."""
    return scale.reshape(1).expand(heads).contiguous() if scale.numel() == 1 else scale


def paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables, sm_scale, query_length,
                           sliding_window, max_context_len, out, workspace, kv_layout, k_descale=None, v_descale=None,
                           q_descale=None, causal=True, num_splits=None):
    """``tlx.ops.paged_attention_decode`` on sm100: packed 5-D FP8 K/V (the catalog admits nothing else here)."""
    if max_context_len is None:
        raise ValueError("packed FP8 decode needs max_context_len")
    kv_heads, q_heads = key_cache.shape[1], query.shape[1]
    q_scale = None if q_descale is None else _per_head(q_descale, q_heads)
    return paged_attention_decode_fp8(query, key_cache, value_cache, context_lens, block_tables,
                                      _per_head(k_descale, kv_heads), _per_head(v_descale, kv_heads),
                                      max_context_len=max_context_len, q_scale=q_scale, query_length=query_length,
                                      causal=causal, sliding_window=sliding_window, sm_scale=sm_scale,
                                      num_splits=num_splits, out=out, workspace=workspace)


def paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, query_length, sliding_window,
                                     max_context_len, kv_layout, norm_gate=False, num_splits=None):
    if max_context_len is None:
        raise ValueError("packed FP8 decode needs max_context_len")
    return paged_attention_decode_fp8_workspace(query, key_cache, value_cache, block_tables,
                                                max_context_len=max_context_len, query_length=query_length,
                                                sliding_window=sliding_window, num_splits=num_splits)
