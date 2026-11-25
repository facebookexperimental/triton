"""Decode engine of ``moe_mxfp4_a16w4``: the same GEMMs with the MXFP4 weights loaded straight into registers.

At a few tokens per expert the stage-1 and stage-2 GEMMs only stream weights, so the cost per routed expert is set by
how fast a wave turns packed E2M1 words into its BF16 MFMA operand. ``tl.dot_scaled`` stages the weight tile through
LDS and spends about 21 VALU per MFMA on the upcast; here each lane loads 16 bytes of its own column along K, moves them
into MFMA lane order with two permlane swaps and upcasts them exactly with ``v_cvt_scalef32_pk_bf16_fp4``. Weight and
activation loads run one K tile ahead.

The MFMA (16x16x32 BF16), the K order (32 at a time, from a zero accumulator) and the SiLU are those of
``_a16w4_gemm_kernel``, so both engines give the same bits; ``moe_mxfp4_a16w4`` picks one by token count.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.jit import constexpr_function

from triton.tlx.ops.kernels.moe_mxfp.gfx950 import _silu


@constexpr_function
def _linear(shape, lane, warp, reg):
    strides = []
    for d in range(len(shape)):
        s = 1
        for later in shape[d + 1:]:
            s *= later
        strides.append(s)

    def flat(c):
        return sum(ci * si for ci, si in zip(c, strides))

    th = tuple(flat(c) for c in list(lane) + list(warp))
    va = tuple(flat(c) for c in reg)
    return tlx.layout(shape=((2, ) * len(th), (2, ) * len(va)), stride=(th, va))


@constexpr_function
def _bits(n):
    return [1 << i for i in range(n.bit_length() - 1)]


@constexpr_function
def _b_layout(BN, BKW, WN, kind):
    """Packed-weight layouts for a [BN, BKW] int32 tile (BKW words of eight E2M1 values along K).

    kind 0: load layout, one lane reads four consecutive words (32 values) of its column.
    kind 1: MFMA layout, lane group g of a 32-wide K step holds word g of that step.
    kind 2: kind 1 with a minor dim of 8 decoded values per word ([BN, BKW, 8]).
    kind 3: halfway between 0 and 1, one lane/register swap away from each, so going from the load layout to the MFMA
    layout is two single permlane swaps (``_to_mfma``).
    """
    NR = BN // (16 * WN)
    U = BKW // 16
    n_lane = [(b, 0) for b in (1, 2, 4, 8)]
    n_warp = [(16 * b, 0) for b in _bits(WN)]
    n_rep = [(16 * WN * b, 0) for b in _bits(NR)]
    kw_hi = [(0, 16 * b) for b in _bits(U)]
    if kind == 0:
        return _linear((BN, BKW), n_lane + [(0, 4), (0, 8)], n_warp, [(0, 1), (0, 2)] + kw_hi + n_rep)
    if kind == 1:
        return _linear((BN, BKW), n_lane + [(0, 1), (0, 2)], n_warp, [(0, 4), (0, 8)] + kw_hi + n_rep)
    if kind == 3:
        return _linear((BN, BKW), n_lane + [(0, 1), (0, 8)], n_warp, [(0, 4), (0, 2)] + kw_hi + n_rep)
    lane = [(a, b, 0) for a, b in n_lane] + [(0, 1, 0), (0, 2, 0)]
    warp = [(a, b, 0) for a, b in n_warp]
    reg = [(0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 4, 0), (0, 8, 0)] + [(a, b, 0) for a, b in kw_hi + n_rep]
    return _linear((BN, BKW, 8), lane, warp, reg)


@constexpr_function
def _a_ld_layout(BM, BK, WN):
    """Coalesced [BM, BK] activation load: eight BF16 per lane along K, lanes along K then rows, waves over rows."""
    rest = [(0, 8 * b) for b in _bits(BK // 8)] + [(b, 0) for b in _bits(BM)]
    nw = len(_bits(WN))
    rest += [(0, 0)] * max(0, 6 + nw - len(rest))
    return _linear((BM, BK), rest[:6], rest[6:6 + nw], [(0, 1), (0, 2), (0, 4)] + rest[6 + nw:])


@constexpr_function
def _slice2(BN, BKW, WN):
    """``_b_layout(..., 2)`` sliced along its decoded dim, so a [BN, BKW] tile can expand_dims into it."""
    parent = _b_layout(BN, BKW, WN, 2)

    class _Slice(tlx.slice_layout):

        def to_ir(self, builder, shape=None, element_type=None):
            return builder.make_slice_encoding_attr(2, self.parent.to_ir(builder, (BN, BKW, 8), element_type))

    return _Slice(parent, 2)


@triton.jit
def _to_mfma(w, l_mid: tl.constexpr, l_pack: tl.constexpr):
    """Load layout -> MFMA layout as two single lane/register transpositions (one permlane swap each); the empty asm
    keeps the layout passes from fusing them into one general conversion."""
    w = tlx.require_layout(w, l_mid)
    w = tl.inline_asm_elementwise("", "=v,0", [w], dtype=tl.int32, is_pure=True, pack=1)
    return tlx.require_layout(w, l_pack)


@triton.jit
def _decode(w, sc, BN: tl.constexpr, BKW: tl.constexpr, WN: tl.constexpr, b_op: tl.constexpr):
    """[BN, BKW] packed words + E8M0 bytes (MFMA layout) -> BF16 [BKW * 8, BN] B operand, exact."""
    l_dec: tl.constexpr = _b_layout(BN, BKW, WN, 2)
    l_src: tl.constexpr = _slice2(BN, BKW, WN)
    sf = (sc.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    # The anchors after the broadcasts keep them in front of the asm, which needs all eight copies per word. Each
    # convert writes byte b's two values (low nibble first) as one BF16 pair: eight consecutive K values per word.
    w8 = tlx.require_layout(tl.broadcast_to(tl.expand_dims(tlx.require_layout(w, l_src), 2), (BN, BKW, 8)), l_dec)
    s8 = tlx.require_layout(tl.broadcast_to(tl.expand_dims(tlx.require_layout(sf, l_src), 2), (BN, BKW, 8)), l_dec)
    v = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf16_fp4 $0, $4, $12\n"
        "v_cvt_scalef32_pk_bf16_fp4 $1, $4, $12 op_sel:[1,0,0]\n"
        "v_cvt_scalef32_pk_bf16_fp4 $2, $4, $12 op_sel:[0,1,0]\n"
        "v_cvt_scalef32_pk_bf16_fp4 $3, $4, $12 op_sel:[1,1,0]", "=&v,=&v,=&v,=&v" + ",v" * 16, [w8, s8],
        dtype=tl.bfloat16, is_pure=True, pack=8)
    v = tl.trans(tl.reshape(v, (BN, BKW * 8)))
    return tlx.require_layout(v, b_op)


@triton.jit
def _load_tile(a_ptr, a_off, a_mask, b_base, b2_base, b_off, s_base, s2_base, s_off, k, BLOCK_K: tl.constexpr,
               BKW: tl.constexpr, BKS: tl.constexpr, DUAL_B: tl.constexpr):
    """K tile ``k``: activations, packed weight words and E8M0 bytes (twice for the gate and up halves)."""
    SV: tl.constexpr = BKS if BKS < 16 else 16
    a = tlx.buffer_load(a_ptr, a_off + k * BLOCK_K, mask=a_mask, other=0.0, contiguity=8)
    w = tlx.buffer_load(b_base, b_off + k * BKW, contiguity=4)
    sc = tlx.buffer_load(s_base, s_off + k * BKS, contiguity=SV)
    if DUAL_B:
        w2 = tlx.buffer_load(b2_base, b_off + k * BKW, contiguity=4)
        sc2 = tlx.buffer_load(s2_base, s_off + k * BKS, contiguity=SV)
    else:
        w2 = w
        sc2 = sc
    return a, w, sc, w2, sc2


@triton.jit
def _mma_tile(a, t, acc, acc2, l_mid: tl.constexpr, l_pack: tl.constexpr, b_op: tl.constexpr, BLOCK_N: tl.constexpr,
              BKW: tl.constexpr, WN: tl.constexpr, DUAL_B: tl.constexpr):
    _, w, sc, w2, sc2 = t
    acc = tl.dot(a, _decode(_to_mfma(w, l_mid, l_pack), sc, BLOCK_N, BKW, WN, b_op), acc)
    if DUAL_B:
        acc2 = tl.dot(a, _decode(_to_mfma(w2, l_mid, l_pack), sc2, BLOCK_N, BKW, WN, b_op), acc2)
    return acc, acc2


@triton.jit
def _decode_kernel(
    a_ptr,
    b_ptr,  # int32 view of the packed weights [E, ROWS, K / 8], unit stride along K
    s_ptr,  # uint8 E8M0 [E, ROWS, K / 32], unit stride along K
    c_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    num_valid,
    # Strides are compile-time constants (one compile per weight layout): as runtime values they cost about 1% at 16
    # tokens.
    stride_be: tl.constexpr,  # in int32 words
    stride_bn: tl.constexpr,
    stride_se: tl.constexpr,  # in bytes
    stride_sn: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_DIV_TOPK: tl.constexpr,
    DUAL_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WN: tl.constexpr,
    A_LDS: tl.constexpr,
):
    BKW: tl.constexpr = BLOCK_K // 8
    BKS: tl.constexpr = BLOCK_K // 32
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=False,
                                            warps_per_cta=[1, WN])
    a_op: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    b_op: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    l_load: tl.constexpr = _b_layout(BLOCK_N, BKW, WN, 0)
    l_pack: tl.constexpr = _b_layout(BLOCK_N, BKW, WN, 1)
    l_mid: tl.constexpr = _b_layout(BLOCK_N, BKW, WN, 3)

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_N
    num_pid_m = tl.cdiv(tl.load(num_post_ptr), BLOCK_M)
    if pid >= num_pid_m * num_pid_n:
        return
    # Blocks run last to first: an alignment that groups skipped routes into leading blocks with a negative expert id
    # then starts the real work at once, and the skipped blocks exit at the end, when their ids are in L2.
    pid_m = num_pid_m - 1 - pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return
    tok = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    valid = tok < num_valid
    if A_ROW_DIV_TOPK:
        row = tl.where(valid, tok // TOP_K, 0)
    else:
        row = tl.where(valid, tok, 0)
    if A_LDS:
        a_reg: tl.constexpr = _a_ld_layout(BLOCK_M, BLOCK_K, WN)
        smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, 2,
                                 layout=tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]],
                                                                                            [BLOCK_M, BLOCK_K]))
    else:
        a_reg: tl.constexpr = a_op
    a_off = tlx.require_layout((row[:, None] * K + tl.arange(0, BLOCK_K)[None, :]).to(tl.int32), a_reg)
    # Padding rows issue no loads: alignment pads every expert to BLOCK_M, so at decode most rows are padding.
    a_mask = tlx.require_layout(tl.broadcast_to(valid[:, None], (BLOCK_M, BLOCK_K)), a_reg)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kw = tl.arange(0, BKW)
    b_off = tlx.require_layout((n_idx[:, None] * stride_bn + kw[None, :]).to(tl.int32), l_load)
    s_off = tlx.require_layout((n_idx[:, None] * stride_sn + kw[None, :] // 4).to(tl.int32), l_pack)
    b_base = b_ptr + expert.to(tl.int64) * stride_be
    s_base = s_ptr + expert.to(tl.int64) * stride_se
    acc = tlx.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=mma)
    acc2 = tlx.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=mma)
    b2_base = b_base + N * stride_bn
    s2_base = s_base + N * stride_sn
    n_k: tl.constexpr = K // BLOCK_K
    tl.static_assert(n_k % 2 == 0)
    # Two register sets, one K tile apart: each tile's loads are issued before the previous tile is unpacked. The loop
    # is unrolled by two so the sets swap roles without register copies (a copy of a pending load would wait for it).
    # With A_LDS the activation tile goes through double-buffered LDS, loaded once per workgroup instead of per wave.
    cur = _load_tile(a_ptr, a_off, a_mask, b_base, b2_base, b_off, s_base, s2_base, s_off, 0, BLOCK_K, BKW, BKS, DUAL_B)
    if A_LDS:
        tlx.local_store(tlx.local_view(smem_a, 0), cur[0])
        tlx.workgroup_barrier()
    for k in range(0, n_k, 2):
        nxt = _load_tile(a_ptr, a_off, a_mask, b_base, b2_base, b_off, s_base, s2_base, s_off, k + 1, BLOCK_K, BKW, BKS,
                         DUAL_B)
        tlx.amd_sched_barrier(0)
        if A_LDS:
            a = tlx.local_load(tlx.local_view(smem_a, 0), layout=a_op)
        else:
            a = cur[0]
        acc, acc2 = _mma_tile(a, cur, acc, acc2, l_mid, l_pack, b_op, BLOCK_N, BKW, WN, DUAL_B)
        if A_LDS:
            tlx.local_store(tlx.local_view(smem_a, 1), nxt[0])
            tlx.workgroup_barrier()
        cur = _load_tile(a_ptr, a_off, a_mask, b_base, b2_base, b_off, s_base, s2_base, s_off,
                         tl.minimum(k + 2, n_k - 1), BLOCK_K, BKW, BKS, DUAL_B)
        tlx.amd_sched_barrier(0)
        if A_LDS:
            a = tlx.local_load(tlx.local_view(smem_a, 1), layout=a_op)
        else:
            a = nxt[0]
        acc, acc2 = _mma_tile(a, nxt, acc, acc2, l_mid, l_pack, b_op, BLOCK_N, BKW, WN, DUAL_B)
        if A_LDS:
            tlx.local_store(tlx.local_view(smem_a, 0), cur[0])
            tlx.workgroup_barrier()
    if DUAL_B:
        acc = _silu(acc) * acc2
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + tok[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid[:, None])


# Tuned on MI350X under HIP graph replay with cold weights on one MoE layer of the served model (H = I = K = 4096, 32
# local experts at EP8), stage by stage over uniform and skewed routings. With BLOCK_M=16 each hit local expert owns
# one M block, so a launch streams each hit expert's weights once and its parallelism is (experts hit) x N / BLOCK_N.
# One token hits at most eight local experts (two to four on the busiest rank at EP8), so 16-wide one-wave tiles with
# the activation row in registers give twice the programs: 1.2-1.6x per stage over 32-wide tiles. From 12 to 24 tokens
# about 11-17 experts are hit and the down GEMM is faster with 128-wide tiles (fewer programs re-reading the shared
# intermediate rows); with fewer or more hit experts the 64-wide tile wins. Every config accumulates K in the same
# order on the same MFMA, so the choice never changes the result.
_ONE_TOKEN = dict(BLOCK_N=16, BLOCK_K=512, WN=1, A_LDS=0)
_GATE_UP = dict(BLOCK_N=32, BLOCK_K=512, WN=2, A_LDS=1)
_DOWN = dict(BLOCK_N=64, BLOCK_K=512, WN=4, A_LDS=1)
_DOWN_WIDE = dict(BLOCK_N=128, BLOCK_K=256, WN=4, A_LDS=1)
# Expert tensor parallelism gives every rank a short intermediate slice (I = 512 at TP8) of all 256 experts, so one
# token already hits eight experts and 16 tokens about 100. Tuned the same way on the TP8 slice of the served layer:
# the 16-wide gate/up tile stays best up to four tokens (1.2x at four), and the down GEMM, whose K is only two 256-wide
# tiles, is fastest with 128-wide tiles from two tokens on (1.1x at eight).
_SHORT_INTER = 512


def pick_config(tokens: int, inter: int = 4096) -> tuple[dict, dict]:
    """(gate/up, down) configs of the decode engine for ``tokens`` tokens and per-expert intermediate size ``inter``;
    BLOCK_M is 16."""
    if tokens == 1:
        return _ONE_TOKEN, _ONE_TOKEN
    if inter <= _SHORT_INTER:
        return (_ONE_TOKEN if tokens <= 4 else _GATE_UP), _DOWN_WIDE
    if 12 <= tokens <= 24:
        return _GATE_UP, _DOWN_WIDE
    return _GATE_UP, _DOWN


def _fit(cfg: dict, N: int, K: int) -> dict:
    """Halve the N and K tiles until they divide this stage (K must hold an even number of tiles)."""
    cfg = dict(cfg)
    while N % cfg["BLOCK_N"]:
        cfg["BLOCK_N"] //= 2
    while cfg["BLOCK_N"] < 16 * cfg["WN"]:
        cfg["WN"] //= 2
    while K % (2 * cfg["BLOCK_K"]):
        cfg["BLOCK_K"] //= 2
    return cfg


def launch(a, w, ws, c, sorted_ids, expert_ids, num_post, num_valid, N, K, topk, dual, cfg, block_m):
    """One stage over ``block_m``-aligned rows: DUAL reads token rows (slot // topk) and writes bf16(silu(gate) * up)
    per slot; otherwise it reads and writes per slot."""
    cfg = _fit(cfg, N, K)
    bn = cfg["BLOCK_N"]
    # Every block that holds a route is a prefix of sorted_ids and has at least one route, so at most num_valid
    # blocks do work; the padded capacity would launch many more that only exit.
    num_m_blocks = min(triton.cdiv(sorted_ids.numel(), block_m), max(num_valid, 1))
    w32 = w.view(torch.int32)
    _decode_kernel[(num_m_blocks * (N // bn), )](a, w32, ws, c, sorted_ids, expert_ids, num_post, num_valid,
                                                 w32.stride(0), w32.stride(1), ws.stride(0), ws.stride(1), N=N, K=K,
                                                 TOP_K=topk, A_ROW_DIV_TOPK=dual, DUAL_B=dual, BLOCK_M=block_m,
                                                 BLOCK_N=bn, BLOCK_K=cfg["BLOCK_K"], WN=cfg["WN"],
                                                 A_LDS=cfg["A_LDS"] == 1, num_warps=cfg["WN"], num_stages=1,
                                                 matrix_instr_nonkdim=16)
