"""MXFP4 A16W4 routed MoE on gfx950: BF16 activations, MXFP4 weights.

Same contract as ``moe_mxfp4`` except that activations stay BF16: the stage-1 gate/up GEMM, the SiLU and the stage-2
down GEMM all take BF16 inputs, and only the stage-1 output is rounded to BF16 between the GEMMs. ``tl.dot_scaled``
with a BF16 left operand upcasts each MXFP4 weight tile to BF16 in registers (``v_cvt_scalef32_pk_bf16_fp4``) and runs
the BF16 MFMA. Routing alignment and the finalize (optional post-norm and routing-weighted sum) are shared with
``moe_mxfp4``.

Weights are read in the checkpoint's natural (Quark) layout, so there is no load-time preparation: ``w13`` uint8
``[E, 2I, K/2]`` with ``w13_scale`` uint8 ``[E, 2I, K/32]`` (gate rows first), and ``w2`` uint8 ``[E, H, I/2]`` with
``w2_scale`` uint8 ``[E, H, I/32]``. Packed bytes hold the even element in the low nibble.

Up to ``DECODE_MAX_TOKENS`` tokens both GEMMs run on the decode engine (``gfx950_a16w4_decode``), which loads the
weights straight into registers instead of through ``tl.dot_scaled``; it uses the same MFMA, K order and SiLU, so the
output does not depend on which engine ran.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton.tlx.ops.kernels.moe.gfx950 import _align, _finalize_kernel
from triton.tlx.ops.kernels.moe_mxfp import gfx950_a16w4_decode as _decode
from triton.tlx.ops.kernels.moe_mxfp.gfx950 import _align_quad, _silu

# Largest token count that runs on the decode engine (measured crossover on MI350X, H = I = K = 4096, EP8).
DECODE_MAX_TOKENS = 64


@triton.jit
def _a16w4_gemm_kernel(
    a_ptr,  # DUAL: bf16 tokens [T, K]; else bf16 sorted rows [cap, K]
    b_ptr,  # uint8 [E, N_rows, K/2]
    bs_ptr,  # uint8 [E, N_rows, K/32]
    c_ptr,  # DUAL: bf16 sorted rows [cap, N]; else bf16 slots [num_valid, N]
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    num_valid,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bse,
    stride_bsn,
    stride_cm,
    TOP_K: tl.constexpr,
    DUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    UNIT_AK: tl.constexpr,
):
    """One [BLOCK_M, BLOCK_N] output tile of one expert's sorted rows. DUAL computes gate (rows n) and up (rows
    N + n) and writes bf16(silu(gate) * up) to the sorted intermediate; otherwise writes bf16 rows per slot."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(tl.load(num_post_ptr), BLOCK_M)
    num_pid_n = N // BLOCK_N
    # The grid is sized from a host-side bound; tiles past the routed blocks exit.
    if pid >= num_pid_m * num_pid_n:
        return
    group = GROUP_M * num_pid_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slots = tl.load(sorted_ids_ptr + rows)
    valid = slots < num_valid
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // 32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if DUAL:
        a_rows = (slots // TOP_K).to(tl.int64)
    else:
        a_rows = rows.to(tl.int64)
    # A runtime column stride would hide the contiguity the vectorized load relies on.
    a_step = BLOCK_K if UNIT_AK else BLOCK_K * stride_ak
    if UNIT_AK:
        a_ptrs = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :]
    else:
        a_ptrs = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
    # B tile as [K/2, N] (packed along K) and scales as [N, K/32], straight from the [N, K/2] / [N, K/32] layout.
    b_base = b_ptr + expert * stride_be
    bs_base = bs_ptr + expert * stride_bse
    g_ptrs = b_base + offs_n[None, :].to(tl.int64) * stride_bn + offs_kp[:, None]
    gs_ptrs = bs_base + offs_n[:, None].to(tl.int64) * stride_bsn + offs_ks[None, :]
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if DUAL:
        u_ptrs = g_ptrs + N * stride_bn
        us_ptrs = gs_ptrs + N * stride_bsn
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        g = tl.load(g_ptrs)
        gs = tl.load(gs_ptrs)
        acc_g = tl.dot_scaled(a, None, "bf16", g, gs, "e2m1", acc_g)
        if DUAL:
            u = tl.load(u_ptrs)
            us = tl.load(us_ptrs)
            acc_u = tl.dot_scaled(a, None, "bf16", u, us, "e2m1", acc_u)
            u_ptrs += BLOCK_K // 2
            us_ptrs += BLOCK_K // 32
        a_ptrs += a_step
        g_ptrs += BLOCK_K // 2
        gs_ptrs += BLOCK_K // 32

    if DUAL:
        h = (_silu(acc_g) * acc_u).to(tl.bfloat16)
        c_ptrs = c_ptr + rows[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, h, mask=valid[:, None])
    else:
        c_ptrs = c_ptr + slots[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, acc_g.to(tl.bfloat16), mask=valid[:, None])


def pick_down_config(num_valid: int, cfg: dict) -> dict:
    """Down-GEMM config given the gate/up one (same BLOCK_M, so both stages share one alignment). Past 8192 slots the
    down GEMM (N = H, K = I, no second accumulator) is 1.25x faster with 256-wide N tiles (MI350X, H = I = K = 4096,
    2048 and 8192 tokens at EP8); the tile does not change the K order, so the result is the same."""
    if num_valid > 8192:
        return dict(cfg, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=2)
    return cfg


def pick_config(num_valid: int) -> dict:
    """Tile config by routed-slot count (tuned on MI350, H = I = K = 4096, 64 local experts). Small counts use
    16-row tiles and narrow N so enough programs stream the expert weights; the decode tier is picked for 4-8 local
    routes, the slowest rank's load at EP4 (it sets the layer time), not the 2-route average."""
    if num_valid <= 64:
        return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=512, GROUP_M=1, num_warps=4, num_stages=2)
    if num_valid <= 2048:
        return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, GROUP_M=1, num_warps=4, num_stages=2)
    if num_valid <= 4096:
        return dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2)
    if num_valid <= 8192:
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, num_warps=4, num_stages=2)
    return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, num_warps=8, num_stages=2)


_DECODE_BLOCK_M = 16


def _launch(a, b, bs, c, sorted_ids, expert_ids, num_post, num_valid, N, K, topk, dual, cfg):
    bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
    num_m_blocks = min(triton.cdiv(sorted_ids.numel(), bm), max(num_valid, 1))
    _a16w4_gemm_kernel[(num_m_blocks * (N // bn), )](a, b, bs, c, sorted_ids, expert_ids, num_post, num_valid, N, K,
                                                     a.stride(0), a.stride(1), b.stride(0), b.stride(1), bs.stride(0),
                                                     bs.stride(1), c.stride(0), TOP_K=topk, DUAL=dual, BLOCK_M=bm,
                                                     BLOCK_N=bn, BLOCK_K=cfg["BLOCK_K"], GROUP_M=cfg["GROUP_M"],
                                                     UNIT_AK=a.stride(1) == 1, num_warps=cfg["num_warps"],
                                                     num_stages=cfg["num_stages"], matrix_instr_nonkdim=16)


def _word_strided(w):
    """True if the packed weights can be viewed as int32 words (byte offset and expert and row strides in words)."""
    return w.storage_offset() % 4 == 0 and w.stride(0) % 4 == 0 and w.stride(1) % 4 == 0


def _check_inputs(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale):
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError(f"moe_mxfp4_a16w4: x must be bf16 [T, K]; got {x.dtype} {tuple(x.shape)}")
    T, K = x.shape
    if topk_ids.ndim != 2 or topk_ids.shape[0] != T or topk_weights.shape != topk_ids.shape:
        raise ValueError(f"moe_mxfp4_a16w4: topk_ids {tuple(topk_ids.shape)} and topk_weights "
                         f"{tuple(topk_weights.shape)} must both be [T={T}, topk]")
    for name, w in (("w13", w13), ("w13_scale", w13_scale), ("w2", w2), ("w2_scale", w2_scale)):
        if w.dtype != torch.uint8 or w.ndim != 3:
            raise ValueError(f"moe_mxfp4_a16w4: {name} must be uint8 rank 3; got {w.dtype} {tuple(w.shape)}")
        if w.stride(2) != 1:
            raise ValueError(f"moe_mxfp4_a16w4: {name} must be contiguous along its last (K) dim")
    E, two_i, kp = w13.shape
    if K % 32 or kp * 2 != K or w13_scale.shape != (E, two_i, K // 32):
        raise ValueError(f"moe_mxfp4_a16w4: w13 {tuple(w13.shape)} / w13_scale {tuple(w13_scale.shape)} do not match "
                         f"K={K}; expected [E, 2I, {K // 2}] / [E, 2I, {K // 32}]")
    I = two_i // 2
    H = w2.shape[1]
    if two_i % 2 or I % 32 or w2.shape != (E, H, I // 2) or w2_scale.shape != (E, H, I // 32):
        raise ValueError(f"moe_mxfp4_a16w4: w2 {tuple(w2.shape)} / w2_scale {tuple(w2_scale.shape)} do not match "
                         f"E={E}, I={I}; expected [E, H, {I // 2}] / [E, H, {I // 32}]")


def moe_mxfp4_a16w4(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, expert_start=0, post_norm_weight=None,
                    post_norm_eps=1e-6, post_norm_gain=0.0, out=None, cfg=None):
    """BF16 x [T, K] (any strides) through the local experts [expert_start, expert_start + E) with natural-layout
    MXFP4 weights. Returns BF16 [T, H]: the routing-weighted sum over local experts (slots routed elsewhere contribute
    nothing). ``cfg`` overrides ``pick_config``'s tile choice, whose cutoffs were tuned for one shape, and forces the
    ``tl.dot_scaled`` engine."""
    _check_inputs(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale)
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return out
    # The decode engine runs an even number of K tiles of at least 128 and N tiles of at least 16, and reads the
    # weights as 32-bit words.
    use_decode = (cfg is None and T <= DECODE_MAX_TOKENS and K % 256 == 0 and I % 256 == 0 and H % 16 == 0
                  and _word_strided(w13) and _word_strided(w2))
    if cfg is None:
        cfg = pick_config(num_valid)
        cfg2 = pick_down_config(num_valid, cfg)
        if H % cfg2["BLOCK_N"] or I % cfg2["BLOCK_K"]:
            cfg2 = cfg
    else:
        cfg2 = cfg
    if not use_decode:
        for dim, size, tile, c in (("K", K, "BLOCK_K", cfg), ("I", I, "BLOCK_N", cfg), ("I", I, "BLOCK_K", cfg2),
                                   ("H", H, "BLOCK_N", cfg2)):
            if size % c[tile]:
                raise ValueError(f"moe_mxfp4_a16w4: {dim}={size} is not a multiple of {tile}={c[tile]}")

    ids = topk_ids.contiguous()
    scores32 = topk_weights.to(torch.float32).contiguous()
    y = torch.empty((num_valid, H), dtype=torch.bfloat16, device=dev)
    if use_decode:
        sorted_ids, expert_ids, num_post = _align(ids, E, expert_start, _DECODE_BLOCK_M)
        c1, c2 = _decode.pick_config(T, I)
        inter = torch.empty((num_valid, I), dtype=torch.bfloat16, device=dev)
        _decode.launch(x.contiguous(), w13, w13_scale, inter, sorted_ids, expert_ids, num_post, num_valid, I, K, topk,
                       True, c1, _DECODE_BLOCK_M)
        _decode.launch(inter, w2, w2_scale, y, sorted_ids, expert_ids, num_post, num_valid, H, I, topk, False, c2,
                       _DECODE_BLOCK_M)
    else:
        # Same sorted order either way: one program per expert wins up to ~8K slots, the chunked version past that.
        align = _align if num_valid <= 8192 else _align_quad
        sorted_ids, expert_ids, num_post = align(ids, E, expert_start, cfg["BLOCK_M"])
        inter = torch.empty((sorted_ids.numel(), I), dtype=torch.bfloat16, device=dev)
        _launch(x, w13, w13_scale, inter, sorted_ids, expert_ids, num_post, num_valid, I, K, topk, True, cfg)
        _launch(inter, w2, w2_scale, y, sorted_ids, expert_ids, num_post, num_valid, H, I, topk, False, cfg2)

    has_norm = post_norm_weight is not None
    _finalize_kernel[(T, )](
        y,
        post_norm_weight.contiguous() if has_norm else out,
        scores32,
        ids,
        out,
        H,
        expert_start,
        E,
        float(post_norm_eps),
        float(post_norm_gain),
        0,
        TOP_K=topk,
        TOP_K_BLOCK=triton.next_power_of_2(topk),
        HAS_NORM=has_norm,
        BLOCK_H=triton.next_power_of_2(H),
        SPLIT_K=1,
        num_warps=4,
    )
    return out
