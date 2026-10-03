# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TLX HSTU cross attention targeting the gfx950 (AMD MI350X) architecture.

Standalone OSS copy of the Hammer v2 kernel at D120435766. The kernel and
variant bodies are preserved; only fbcode imports and custom-op dispatch are
replaced with sibling tutorial helpers and a direct Python call.

Extracted from `hammer/v2/ops/triton/template/triton_bw_cross_attention.py`,
the v3 cross attention kernel; the previous diff landed that as a forward-only,
TMA-free copy plus deletions. This is the optimization pass: the KV load path
is staged through LDS, following what D114539697 did to the self attention
kernel and D118162155 did to the v2 cross attention kernel.

The load path is backported from the TLX tutorial
`third-party/triton/beta/triton/third_party/tlx/tutorials/amd_hstu_attn.py` by
way of `tlx_gfx950_hstu_attention.py`:

  - **K and V are staged through LDS.** `lds_k` / `lds_v` are
    `tlx.local_alloc`'d once outside the KV loop; each block runs
    `tlx.async_load` -> `async_load_commit_group` -> `async_load_wait_group(0)`
    -> `tlx.local_load`. Both copies go into one async group, so their global
    latencies overlap instead of being paid back to back.
  - **K is transposed in LDS.** It is fetched row-major `(BLOCK_N, HEAD_DIM)`
    and transposed with `tlx.local_trans`, a metadata-only view, replacing the
    register-level `tl.trans(k)`. Note this is a smaller lever here than it was
    on the v2 kernel: there, K came through a block pointer with
    `shape=(BLOCK_D_Q, seq_len_kv), strides=(1, stride_kn)`, so the staging
    also turned a strided transposed global read into a contiguous one. v3
    already read K contiguously, so what LDS buys on this base is the register
    round trip, not the access pattern.
  - **XCD remapping.** The grid becomes 1-D
    (`cdiv(max_q_len, BLOCK_M) * Z * H`) and `remap_xcd` regroups tiles so they
    are contiguous within each of the 8 XCDs instead of round-robin across
    them. Adds a `MAX_Q_LEN` runtime arg for the in-kernel tile count, and
    moves the program-id decode out of `_compute_offsets`, which now takes
    `off_z` / `off_h` as arguments.

The main-loop and last-block bodies, which the source repeated inline, are
merged into one `_tlx_gfx950_cross_attn_fwd_one_block` taking a `MASKED`
constexpr. They differed only in the bounds mask, and the staging sequence
would otherwise have been written twice. The masked and unmasked forms are
numerically identical to the source's two bodies: for softmax the source set
invalid entries to `-inf` before scaling by `alpha` where
`forward_softmax_activation_scaled_alpha` scales first and masks second, and
for silu it zeroed `qk` before scaling where the helper zeroes `alpha` -- both
land on the same value.

`TRANS` is dropped, along with `forward_valid_mask_trans`,
`forward_epilogue_trans` and `forward_softmax_activation_trans_scaled_alpha`.
All 144 configs in the AMD list set it False, so `TRANS=True` was unreachable
on gfx950, and staging it would have meant a second copy of the load sequence
for a path nothing selects. `SHARED_KV` and `IS_DYNAMIC` are kept.

`SHARED_KV` no longer specialises the staging. The source reuses the
register-resident K tile as the PV operand; through LDS that would mean reading
one `local_alloc` through both a `tlx.local_trans`'d and an untransposed
consumer, which TLX cannot express -- `TlxPropagateLayout` fails the module. V
is staged unconditionally instead, and under `SHARED_KV` both copies read one
address, so the second is an L2 hit rather than a second trip to DRAM. The
constexpr is left unread here; it stays in the signature for `shared_kv` one
level up and for the backward port.

Output is no longer bit-identical to the upstream kernel, and cannot be:
`local_trans` changes the MFMA operand layout, which changes the accumulation
order.
"""

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

# @manual=//triton:triton
import triton.language.extra.tlx as tlx
from stubs import (
    autotune_max_seq_len,
    maybe_register_custom_op,
    next_power_of_2,
    triton_autotune,
)
from tlx_gfx950_cross_attention_v3_baseline import (
    _tlx_gfx950_cross_attn_v3_ttgir_bwd, )
from triton_attention_utils import (
    backward_softmax_activation_scaled_alpha,
    fast_silu,
    forward_softmax_activation_scaled_alpha,
)
from triton_hstu_cross_attention import (
    _attn_bwd_preprocess,
    backward_common_preprocess,
    backward_d_silu_activation,
    backward_d_softmax_activation,
    backward_silu_activation,
    backward_valid_mask,
    forward_custom_vars,
    forward_epilogue,
    forward_softmax_common_preprocess,
    forward_valid_mask,
    target_common_preprocess,
    uih_common_preprocess,
)


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Pack only the innermost dimension without replacing writable outputs."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# MI300X/MI350X expose 8 XCDs; round-robin pid assignment across them hurts L2
# reuse, so tiles are re-grouped to be contiguous within an XCD.
NUM_XCDS: tl.constexpr = tl.constexpr(8)

# `autotune_max_seq_len` is the wrong quantizer for the q-length autotune key:
# its runtime path is `prev_power_of_2`, which rounds *down* and so merges
# max_q_len=160 into the same entry as 128 -- the config tuned for 128 then
# tiles 160 badly. Its static path rounds up, but into KV-sized buckets
# (1024+), which collapses every target count into one entry.
#
# 32 is the smallest BLOCK_M in `get_fwd_triton_spec_configs`, and rounding up
# to a multiple of it leaves `cdiv(max_q_len, BLOCK_M)` unchanged for every
# BLOCK_M in the sweep -- so the bucket never merges two lengths that tile
# differently.
AUTOTUNE_Q_LEN_GRANULARITY: int = 32

SPLIT_SOFTMAX_DKDV_BLOCK_M: int = 64
SPLIT_SOFTMAX_DKDV_BLOCK_N: int = 128
SPLIT_SOFTMAX_DKDV_NUM_WARPS: int = 4
SPLIT_SOFTMAX_DKDV_MATRIX_INSTR_NONKDIM: int = 16
SPLIT_SOFTMAX_DKDV_WAVES_PER_EU: int = 1
SPLIT_SOFTMAX_DKDV_NUM_STAGES: int = 1
SPLIT_SOFTMAX_DQ_BLOCK_M: int = 64
SPLIT_SOFTMAX_DQ_BLOCK_N: int = 128
SPLIT_SOFTMAX_DQ_NUM_WARPS: int = 4
SPLIT_SOFTMAX_DQ_MATRIX_INSTR_NONKDIM: int = 16
SPLIT_SOFTMAX_DQ_WAVES_PER_EU: int = 1
SPLIT_SOFTMAX_DQ_NUM_STAGES: int = 1


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN is not divisible by NUM_XCDS, the first `tall_xcds` XCDs get
    # pids_per_xcd tiles and the rest get one fewer.
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    return pid


@triton.jit
def _compute_offsets(
    off_z,
    off_h,
    G,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_h_kv = off_h // G
    seq_start_kv = tl.load(seq_offsets + off_z)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        # KEEP_LAST: skip early tokens; KEEP_FIRST: keep start unchanged
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
    )


@triton.jit
def _tlx_gfx950_cross_attn_fwd_one_block(
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n_0,
    q,
    K_base,
    V_base,
    stride_kn,
    stride_vn,
    lds_k,
    lds_v,
    acc,
    m_i,
    l_i,
    alpha,
    uih_len_q,
    HEAD_DIM: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    MASKED: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_n = offs_n_0 + start_n
    offs_d_q = tl.arange(0, HEAD_DIM)
    offs_d_v = tl.arange(0, BLOCK_D_V)
    mask_n = offs_n < seq_len_kv

    # K and V are copied straight into LDS instead of being read into registers
    # and handed to the MFMA from there. Unlike the self-attention kernel, the
    # global read this replaces was already contiguous -- v3 fetches K row-major
    # and transposes in registers -- so what LDS buys here is the register-file
    # round trip and the `tl.trans`, not a strided access.
    #
    # Both copies go into one async group. V's address does not depend on the
    # QK dot, so issuing it here costs the block one global latency rather than
    # two. Only the LDS->register read of V is deferred to its use, to keep the
    # tile out of registers across the activation.
    k_ptrs = K_base + offs_n.to(tl.int64)[:, None] * stride_kn + offs_d_q[None, :]
    k_local = tlx.local_view(lds_k, 0)
    v_local = tlx.local_view(lds_v, 0)
    # Staged the same way whether or not V aliases K -- see the note on
    # SHARED_KV in the module docstring.
    v_ptrs = V_base + offs_n.to(tl.int64)[:, None] * stride_vn + offs_d_v[None, :]
    if MASKED:
        tok_k = tlx.async_load(k_ptrs, k_local, mask=mask_n[:, None])
        tok_v = tlx.async_load(v_ptrs, v_local, mask=mask_n[:, None])
    else:
        tok_k = tlx.async_load(k_ptrs, k_local)
        tok_v = tlx.async_load(v_ptrs, v_local)
    tlx.async_load_commit_group([tok_k, tok_v])
    # The wait token is threaded into both `local_load`s below. Without it AMD
    # lowering keeps its own conservative producer-to-consumer wait-count
    # tracking on top of this explicit wait.
    # pyrefly: ignore [bad-argument-type]
    wait_tok = tlx.async_load_wait_group(0)
    kt_local = tlx.local_trans(k_local)
    k_dot_op = tlx.local_load(kt_local, token=wait_tok)

    qk = tl.dot(q, k_dot_op)

    if MASKED:
        valid_mask = forward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL)
    elif HAS_CAUSAL:
        # Interior blocks are wholly inside seq_len_kv, so only the causal cut
        # can bite. q rows past seq_len_q are left unmasked because the store
        # and the `M` write are both masked on them.
        shifted_offs_m = offs_m + seq_len_kv - uih_len_q
        valid_mask = shifted_offs_m[:, None] >= offs_n[None, :]
    else:
        valid_mask = tl.full([BLOCK_M, BLOCK_N], 1, tl.int1)

    if IS_SOFTMAX:
        act_qk, acc, l_i, m_i = forward_softmax_activation_scaled_alpha(qk, alpha, valid_mask, m_i, acc, l_i)
    else:
        masked_alpha = tl.where(valid_mask, alpha, 0.0)
        qk = qk * masked_alpha
        # pyrefly: ignore [bad-argument-type]
        act_qk = fast_silu(qk, MULT_BY_X=True)

    v = tlx.local_load(v_local, token=wait_tok)

    act_qk = act_qk.to(v.dtype)
    acc = tl.dot(act_qk, v, acc)
    return acc, l_i, m_i


@triton.jit
def _tlx_gfx950_cross_attn_fwd_compute(
    acc,
    alpha,
    q,
    K,
    V,
    offs_m,
    offs_n_0,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    attn_scale,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_mm,
    m_i,
    l_i,
    M,
    start_m,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    if HAS_CAUSAL:
        high = start_m + BLOCK_M + seq_len_kv - uih_len_q
        high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
        last_block_start_n = tl.maximum(0, (tl.cdiv(high, BLOCK_N) - 1) * BLOCK_N)
    else:
        last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    if IS_SOFTMAX:
        alpha = alpha * 1.44269504

    # Head- and sequence-offset the bases once; the per-block pointers below are
    # then a BLOCK_N stride off these. `tlx.async_load` takes a pointer tensor,
    # so the source's inline `(seq_start_kv + offs_n)` arithmetic is hoisted
    # rather than repeated per block.
    K_base = K + seq_start_kv.to(tl.int64) * stride_kn + off_h_kv * stride_kh
    V_base = V + seq_start_kv.to(tl.int64) * stride_vn + off_h_kv * stride_vh

    # Single-buffered LDS staging, reserved once and reused by every KV block.
    # pyrefly: ignore [bad-argument-type]
    lds_k = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), 1)
    # pyrefly: ignore [bad-argument-type]
    lds_v = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), 1)

    # Main loop - no masking needed for non-last kv blocks
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_one_block(
            start_n=start_n,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            offs_m=offs_m,
            offs_n_0=offs_n_0,
            q=q,
            K_base=K_base,
            V_base=V_base,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            lds_k=lds_k,
            lds_v=lds_v,
            acc=acc,
            m_i=m_i,
            l_i=l_i,
            alpha=alpha,
            uih_len_q=uih_len_q,
            HEAD_DIM=HEAD_DIM,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SHARED_KV=SHARED_KV,
            IS_SOFTMAX=IS_SOFTMAX,
            # pyrefly: ignore [bad-argument-type]
            MASKED=False,
            HAS_CAUSAL=HAS_CAUSAL,
        )

    # Last kv block - needs masking for out-of-bounds elements;
    # q masking is not needed here because we mask when storing results.
    acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_one_block(
        start_n=last_block_start_n,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        offs_m=offs_m,
        offs_n_0=offs_n_0,
        q=q,
        K_base=K_base,
        V_base=V_base,
        stride_kn=stride_kn,
        stride_vn=stride_vn,
        lds_k=lds_k,
        lds_v=lds_v,
        acc=acc,
        m_i=m_i,
        l_i=l_i,
        alpha=alpha,
        uih_len_q=uih_len_q,
        HEAD_DIM=HEAD_DIM,
        BLOCK_D_V=BLOCK_D_V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SHARED_KV=SHARED_KV,
        IS_SOFTMAX=IS_SOFTMAX,
        # pyrefly: ignore [bad-argument-type]
        MASKED=True,
        HAS_CAUSAL=HAS_CAUSAL,
    )

    # epilogue
    if IS_SOFTMAX:
        acc = forward_epilogue(
            acc,
            l_i,
            m_i,
            offs_m,
            seq_start_q,
            stride_mm,
            seq_len_q,
            M,
            off_h,
            num_softmax_heads,
        )
    else:
        scale = tl.load(attn_scale).to(tl.float32)
        acc = acc * scale
    return acc


@triton.jit
def _tlx_gfx950_cross_attn_fwd_inner(
    alpha,
    Z,
    Q,
    K,
    V,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    start_m,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    M,
    stride_mm,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    # initialize offsets
    if start_m < seq_len_q:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n_0 = tl.arange(0, BLOCK_N)
        qo_offset_y_split = seq_start_q + start_m
        offs_qk_d = tl.arange(0, HEAD_DIM)
        q_ptrs = (Q + (qo_offset_y_split + tl.arange(0, BLOCK_M)).to(tl.int64)[:, None] * stride_qm +
                  (off_h * stride_qh + offs_qk_d)[None, :])
        q = tl.load(
            q_ptrs,
            mask=(offs_m < seq_len_q)[:, None],
            other=0.0,
        )
        m_i, l_i = forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M)
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        # when seq_len_kv is 0, we can skip computation but still needs to write acc to out
        # to avoid NaN.
        if seq_len_kv > 0:
            if off_h < num_softmax_heads:
                acc = _tlx_gfx950_cross_attn_fwd_compute(
                    acc,
                    alpha,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_STAGES,
                    SHARED_KV,
                    # pyrefly: ignore [bad-argument-type]
                    IS_SOFTMAX=True,
                    HAS_CAUSAL=HAS_CAUSAL,
                )
            else:
                acc = _tlx_gfx950_cross_attn_fwd_compute(
                    acc,
                    alpha,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_STAGES,
                    SHARED_KV,
                    # pyrefly: ignore [bad-argument-type]
                    IS_SOFTMAX=False,
                    HAS_CAUSAL=HAS_CAUSAL,
                )
        off_o = Out + seq_start_q * stride_om + off_h * stride_oh
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_D_V)
        out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
        acc = acc.to(Out.dtype.element_ty)
        tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])


def get_fwd_triton_spec_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64]:
            for num_stages in [1, 2]:
                for num_warps in [4, 8]:
                    for matrix_instr_nonkdim in [16, 32]:
                        for waves_per_eu in [1, 2, 3]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "BLOCK_M1": BLOCK_M,
                                        "BLOCK_N1": BLOCK_N,
                                        "NUM_STAGES": num_stages,
                                        "NUM_STAGES1": num_stages,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": waves_per_eu,
                                        "IS_DYNAMIC": False,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                ))
    return configs


@triton_autotune(
    configs=get_fwd_triton_spec_configs(),
    key=[
        "AUTOTUNE_Z",
        "HEAD_DIM",
        # max_q_len sets BLOCK_M's tiling and the grid, so a config tuned at one
        # target count is not transferable to another. Bucketed by
        # AUTOTUNE_Q_LEN_GRANULARITY, not autotune_max_seq_len -- see the note
        # on that constant.
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "num_softmax_heads",
    ],
)
@triton.jit
def _tlx_gfx950_cross_attn_fwd(
    alpha,
    Z,
    H,
    G,
    Q,
    K,
    V,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    M,
    stride_mm,
    num_targets,
    MAX_Q_LEN,
    AUTOTUNE_Z,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    NUM_STAGES1: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_DYNAMIC: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
):
    # 1-D grid so tiles can be regrouped per XCD. The 2-D (m_tile, Z*H) launch
    # this replaces hands consecutive tiles to different XCDs, which scatters
    # workgroups that share K/V across all 8 L2s.
    tpid = tl.program_id(0)
    num_tiles = tl.cdiv(MAX_Q_LEN, BLOCK_M)
    tpid = remap_xcd(tpid, num_tiles * Z * H, NUM_XCDS)
    off_h = tpid % H
    off_nz = tpid // H
    start_m = (off_nz % num_tiles) * BLOCK_M
    off_z = off_nz // num_tiles
    (
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
    ) = _compute_offsets(off_z, off_h, G, seq_offsets_q, seq_offsets, max_seq_len, TRUNCATE_METHOD)
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    # DO NOT merge IS_DYNAMIC into a compound boolean expression
    # (e.g. "if IS_DYNAMIC and condition:"). The Triton CC compiler
    # passes constexprs as i32 and cannot bitcast i32 to i1 for the
    # AND operator. Keeping IS_DYNAMIC as a standalone constexpr
    # branch allows dead-code elimination without the bitcast.
    if IS_DYNAMIC:
        if start_m + BLOCK_M1 >= seq_len_q:
            # Use the smaller (BLOCK_M1, BLOCK_N1) tiling for tail handling
            _tlx_gfx950_cross_attn_fwd_inner(
                alpha,
                Z,
                Q,
                K,
                V,
                Out,
                seq_offsets_q,
                seq_offsets,
                max_seq_len,
                attn_scale,
                stride_qm,
                stride_qh,
                stride_kn,
                stride_kh,
                stride_vn,
                stride_vh,
                stride_om,
                stride_oh,
                start_m,
                off_h,
                off_h_kv,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                M,
                stride_mm,
                uih_len_q,
                num_softmax_heads,
                HEAD_DIM,
                BLOCK_D_V,
                BLOCK_M1,
                BLOCK_N1,
                NUM_STAGES1,
                SHARED_KV=SHARED_KV,
                HAS_CAUSAL=HAS_CAUSAL,
            )
            return
    _tlx_gfx950_cross_attn_fwd_inner(
        alpha,
        Z,
        Q,
        K,
        V,
        Out,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        attn_scale,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_om,
        stride_oh,
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        M,
        stride_mm,
        uih_len_q,
        num_softmax_heads,
        HEAD_DIM,
        BLOCK_D_V,
        BLOCK_M,
        BLOCK_N,
        NUM_STAGES,
        SHARED_KV=SHARED_KV,
        HAS_CAUSAL=HAS_CAUSAL,
    )


@maybe_register_custom_op("hammer::tlx_gfx950_cross_attn_v2_fwd", mutates_args=())
def tlx_gfx950_cross_attn_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = False,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert not enable_tma, "TMA is CUDA-only and not supported on gfx950"
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dv.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"
    # The silu epilogue reads the scale with a scalar `tl.load(attn_scale)`, so
    # a per-row tensor would silently apply element 0 to every row.
    assert attn_scale.ndim == 0, ("per-row attn_scale is not supported; pass a 0-dim tensor")

    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        return out, M

    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]) * Z * H, )

    # Create M tensor for softmax heads if needed
    if num_softmax_heads > 0:
        M, stride_mm = forward_custom_vars(q, num_softmax_heads, [])
    else:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        stride_mm = 0

    _tlx_gfx950_cross_attn_fwd[grid](
        alpha=alpha,
        Z=Z,
        H=H,
        G=G,
        Q=q,
        K=k,
        V=v,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        seq_offsets_q=seq_offsets_q,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        attn_scale=attn_scale,
        M=M,
        stride_mm=stride_mm,
        num_targets=num_targets,
        MAX_Q_LEN=max_q_len,
        AUTOTUNE_Z=next_power_of_2(Z),
        AUTOTUNE_MAX_Q_LEN=triton.cdiv(max_q_len, AUTOTUNE_Q_LEN_GRANULARITY) * AUTOTUNE_Q_LEN_GRANULARITY,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        num_softmax_heads=num_softmax_heads,
        HEAD_DIM=DimQ,
        BLOCK_D_V=DimV,
        SHARED_KV=shared_kv,
        TRUNCATE_METHOD=truncate_method,
        HAS_CAUSAL=causal,
        HAS_NUM_TARGETS=num_targets is not None,
    )
    return out, M


@triton.jit
def _compute_bwd_reduce_dq_offsets(
    H,
    G,
    BLOCK_N,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    # Truncate to last max_seq_len KV tokens if actual seq len exceeds max_seq_len
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    )


def _bwd_pre_hook_redq(nargs):
    """
    The redq variant reduces DQ through memory:
    - DQ: many KV blocks accumulate into the same Q rows, so it must be zeroed.
    - DK, DV: each instance owns distinct K/V rows and uses tl.store, except
      under GQA without PER_KV_HEAD, where Q heads sharing a KV head collide.
    """
    if "DQ" in nargs:
        nargs["DQ"].zero_()
        if nargs["TRUNCATE_METHOD"] != "none":
            # When truncating KV, we need to zero out the entire dk/dv tensor
            nargs["DK"].zero_()
            if not nargs["SHARED_KV"]:
                nargs["DV"].zero_()
    # When G > 1 (GQA), multiple Q heads write to the same K/V head,
    # so dk/dv must be zero-initialized for atomic accumulation
    if nargs["G"] > 1 and not nargs.get("PER_KV_HEAD", False) and "DK" in nargs:
        nargs["DK"].zero_()
        if not nargs["SHARED_KV"]:
            nargs["DV"].zero_()


def _get_bw_redq_configs() -> List[triton.Config]:
    configs = []
    for M in [32, 64]:
        for N in [64, 128]:
            for waves_per_eu in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": M,
                            "BLOCK_N": N,
                            "matrix_instr_nonkdim": 16,
                            "waves_per_eu": waves_per_eu,
                        },
                        num_stages=1,
                        num_warps=4,
                        pre_hook=_bwd_pre_hook_redq,
                    ))
    return configs


@triton.jit
def _bwd_load_2d(
    BASE,
    row_start,
    head_col,
    row_stride,
    row_end,
    BLOCK: tl.constexpr,
    DIM: tl.constexpr,
):
    """Load a [BLOCK, DIM] tile by pointer arithmetic.

    The backward kernels operate on tensors packed as [n_rows, H * DIM] (row =
    flattened sequence position, columns = heads concatenated along the last
    dim). ``head_col`` is the column offset of the current head within that
    packed layout (i.e. ``off_h * stride_head``). ``row_start`` is the absolute
    starting row and ``row_end`` is the absolute end row (exclusive) used to
    mask out-of-bounds rows.
    """
    offs_r = row_start + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, DIM)
    ptrs = (BASE + offs_r.to(tl.int64)[:, None] * row_stride.to(tl.int64) + (head_col + offs_d)[None, :])
    mask = (offs_r < row_end)[:, None]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def _tlx_gfx950_cross_attn_bwd_inner(  # noqa C901
    start_n,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    DV,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    M,
    Delta,
    stride_mm,
    off_h,
    off_h_kv,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    SHARED_KV: tl.constexpr,
    MASK_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
):
    kv_offset = (seq_start_kv + start_n).to(tl.int32)
    end_kv = seq_start_kv + seq_len_kv
    end_dq = seq_start_q + seq_len_q
    k = _bwd_load_2d(
        K,
        kv_offset,
        off_h_kv * stride_kh,
        stride_kn,
        end_kv,
        BLOCK=BLOCK_N,
        DIM=DimQ,
    )
    if SHARED_KV:
        v = k
    else:
        v = _bwd_load_2d(
            V,
            kv_offset,
            off_h_kv * stride_vh,
            stride_vn,
            end_kv,
            BLOCK=BLOCK_N,
            DIM=DimV,
        )

    dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
    if not SHARED_KV:
        dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
    # Offset DV pointers by sequence start and head offset
    if not SHARED_KV:
        DV = (DV + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64) +
              off_h_kv * tl.cast(stride_dvh, tl.int64))
    scale = tl.load(attn_scale).to(tl.float32)

    M_off, Delta_off = backward_common_preprocess(M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    if off_h < num_softmax_heads:
        scaled_alpha = alpha * 1.44269504
    else:
        scaled_alpha = alpha
    if HAS_CAUSAL:
        low_q = max(0, start_n + uih_len_q - seq_len_kv)
    else:
        low_q = 0
    for start_m in tl.range(low_q, seq_len_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        q_offset = seq_start_q + start_m
        q = _bwd_load_2d(
            Q,
            q_offset,
            off_h * stride_qh,
            stride_qm,
            end_dq,
            BLOCK=BLOCK_M,
            DIM=DimQ,
        )
        qk_trans = tl.dot(
            k,
            tl.trans(q),
        )
        if MASK_KV or HAS_CAUSAL:
            valid_mask_trans = backward_valid_mask(
                offs_m,
                offs_n,
                uih_len_q,
                seq_len_q,
                seq_len_kv,
                HAS_CAUSAL,
            )
        else:
            valid_mask_trans = offs_m[None, :] < seq_len_q
        if off_h < num_softmax_heads:
            qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                qk_trans,
                scaled_alpha,
                valid_mask_trans,
                M_off,
                offs_m,
                stride_mm,
                mask_m,
                k,
            )
        else:
            qk_trans, act_qk_trans, pT = backward_silu_activation(qk_trans, alpha, valid_mask_trans, k.dtype, scale)
        do = _bwd_load_2d(
            DO,
            q_offset,
            off_h * stride_doh,
            stride_dom,
            end_dq,
            BLOCK=BLOCK_M,
            DIM=DimV,
        )

        if SHARED_KV:
            dk = tl.dot(act_qk_trans, do, dk, allow_tf32=ALLOW_TF32)
        else:
            # pyrefly: ignore [unbound-name]
            dv = tl.dot(act_qk_trans, do, dv, allow_tf32=ALLOW_TF32)

        dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
        if off_h < num_softmax_heads:
            dqk_trans = backward_d_softmax_activation(dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT)
        else:
            dqk_trans = backward_d_silu_activation(dact_qk_trans, pT, qk_trans, scale, valid_mask_trans)
        dqk_trans = dqk_trans.to(k.dtype)
        if SHARED_KV:
            dk_attn = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32)
            dk = dk + dk_attn * alpha
        else:
            dk = tl.dot(dqk_trans, q, dk, allow_tf32=ALLOW_TF32)
        dq_trans = tl.dot(tl.trans(k), dqk_trans)
        dq_trans = dq_trans * alpha
        dq = tl.trans(dq_trans)
        offs_qk_d = tl.arange(0, DimQ)
        dq_ptrs = (DQ + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_dqm.to(tl.int64) +
                   (DimQ * off_h + offs_qk_d)[None, :])
        cur = tl.load(dq_ptrs, mask=mask_m[:, None], other=0.0)
        tl.store(
            dq_ptrs,
            cur + dq.to(DQ.dtype.element_ty),
            mask=mask_m[:, None],
        )

    offs_n = start_n + tl.arange(0, BLOCK_N)
    if not SHARED_KV:
        offs_v_d = tl.arange(0, DimV)
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        if MASK_KV:
            mask_n = offs_n < seq_len_kv
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                )
        else:
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                )

    if not SHARED_KV:
        dk = dk * alpha

    offs_qk_d = tl.arange(0, DimQ)
    dk_ptrs = (DK + (kv_offset + tl.arange(0, BLOCK_N)).to(tl.int64)[:, None] * stride_dkn.to(tl.int64) +
               (DimQ * off_h_kv + offs_qk_d)[None, :])
    mask_n = offs_n < seq_len_kv
    if GQA_ATOMIC_ADD:
        tl.atomic_add(
            dk_ptrs,
            dk.to(DK.dtype.element_ty),
            mask=mask_n[:, None],
            sem="relaxed",
        )
    else:
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])


@triton_autotune(
    configs=_get_bw_redq_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "max_q_len",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
    ],
)
@triton.jit
def _tlx_gfx950_cross_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    DO,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G: tl.constexpr,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    PER_KV_HEAD: tl.constexpr = False,
):
    if PER_KV_HEAD:
        # Per-KV-head (atomic-free GQA) variant: one program owns a KV head and
        # walks its G query heads, so dk/dv reduce in registers.
        H_kv = H // G
        off_hz = tl.program_id(0)
        off_z = off_hz // H_kv
        off_h_kv = off_hz % H_kv
        seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
        seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
        seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
        if TRUNCATE_METHOD != "none":
            truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
            if TRUNCATE_METHOD == "keep_last":
                seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
            seq_len_kv = truncated_len
        seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
        seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
        seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
        if HAS_CAUSAL:
            n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
            uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        else:
            uih_len_q = seq_len_q
        if seq_len_kv == 0 or seq_len_q == 0:
            return

        end_dq = seq_start_q + seq_len_q
        offs_qk_d = tl.arange(0, DimQ)
        offs_v_d = tl.arange(0, DimV)
        scale = tl.load(attn_scale).to(tl.float32)

        # pyrefly: ignore [bad-argument-type]
        for start_n in tl.range(0, seq_len_kv, BLOCK_N, num_stages=1):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv
            k_offset = (seq_start_kv + start_n).to(tl.int32)
            k = _bwd_load_2d(
                K,
                k_offset,
                off_h_kv * stride_kh,
                stride_kn,
                seq_start_kv + seq_len_kv,
                BLOCK=BLOCK_N,
                DIM=DimQ,
            )
            if SHARED_KV:
                v = k
            else:
                v = _bwd_load_2d(
                    V,
                    k_offset,
                    off_h_kv * stride_vh,
                    stride_vn,
                    seq_start_kv + seq_len_kv,
                    BLOCK=BLOCK_N,
                    DIM=DimV,
                )
            dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
            if not SHARED_KV:
                dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
            for g in range(G):
                off_h = off_h_kv * G + g
                if off_h < num_softmax_heads:
                    scaled_alpha = alpha * 1.44269504
                else:
                    scaled_alpha = alpha
                M_off, Delta_off = backward_common_preprocess(M, Delta, off_h, num_softmax_heads, seq_start_q,
                                                              stride_mm)
                # pyrefly: ignore [bad-argument-type]
                for start_m in tl.range(0, seq_len_q, BLOCK_M, num_stages=1):
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    valid_mask_trans = backward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL)
                    q_offset = (seq_start_q + start_m).to(tl.int32)
                    q = _bwd_load_2d(
                        Q,
                        q_offset,
                        off_h * stride_qh,
                        stride_qm,
                        end_dq,
                        BLOCK=BLOCK_M,
                        DIM=DimQ,
                    )
                    do = _bwd_load_2d(
                        DO,
                        q_offset,
                        off_h * stride_doh,
                        stride_dom,
                        end_dq,
                        BLOCK=BLOCK_M,
                        DIM=DimV,
                    )
                    qk_trans = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (backward_softmax_activation_scaled_alpha(
                            qk_trans,
                            scaled_alpha,
                            valid_mask_trans,
                            M_off,
                            offs_m,
                            stride_mm,
                            mask_m,
                            k,
                        ))
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(qk_trans, alpha, valid_mask_trans,
                                                                              k.dtype, scale)
                    if SHARED_KV:
                        dk += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    else:
                        # pyrefly: ignore [unbound-name]
                        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m,
                                                                  pT)
                    else:
                        dqk_trans = backward_d_silu_activation(dact_qk_trans, pT, qk_trans, scale, valid_mask_trans)
                    dqk_trans = dqk_trans.to(k.dtype)
                    dk += tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
                    dq_trans = (tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha)
                    dq = tl.trans(dq_trans)
                    offs_d = tl.arange(0, DimQ)
                    dq_ptrs = (DQ + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_dqm.to(tl.int64) +
                               (DimQ * off_h + offs_d)[None, :])
                    cur = tl.load(dq_ptrs, mask=mask_m[:, None], other=0.0)
                    tl.store(
                        dq_ptrs,
                        cur + dq.to(DQ.dtype.element_ty),
                        mask=mask_m[:, None],
                    )
            DK_off = (DK + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64) +
                      off_h_kv * tl.cast(stride_dkh, tl.int64))
            dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            if not SHARED_KV:
                DV_off = (DV + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64) +
                          off_h_kv * tl.cast(stride_dvh, tl.int64))
                dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
                # pyrefly: ignore [unbound-name]
                tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
        return
    (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_reduce_dq_offsets(
        H,
        G,
        BLOCK_N,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    if seq_len_kv == 0:
        return
    last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        _tlx_gfx950_cross_attn_bwd_inner(
            start_n,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            off_h,
            off_h_kv,
            uih_len_q,
            num_softmax_heads,
            DimQ,
            DimV,
            ALLOW_TF32,
            BLOCK_M,
            BLOCK_N,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            # pyrefly: ignore [bad-argument-type]
            MASK_KV=False,
            HAS_CAUSAL=HAS_CAUSAL,
            Q=Q,
            K=K,
            V=V,
            DO=DO,
            DQ=DQ,
            DK=DK,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
        )
    # Handle last block with masking
    _tlx_gfx950_cross_attn_bwd_inner(
        last_block_start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        DV,
        stride_qh,
        stride_kh,
        stride_vh,
        stride_doh,
        stride_dvn,
        stride_dvh,
        alpha,
        attn_scale,
        M,
        Delta,
        stride_mm,
        off_h,
        off_h_kv,
        uih_len_q,
        num_softmax_heads,
        DimQ,
        DimV,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        GQA_ATOMIC_ADD=G > 1,
        SHARED_KV=SHARED_KV,
        # pyrefly: ignore [bad-argument-type]
        MASK_KV=True,
        HAS_CAUSAL=HAS_CAUSAL,
        Q=Q,
        K=K,
        V=V,
        DO=DO,
        DQ=DQ,
        DK=DK,
        stride_qm=stride_qm,
        stride_kn=stride_kn,
        stride_vn=stride_vn,
        stride_dom=stride_dom,
        stride_dqm=stride_dqm,
        stride_dkn=stride_dkn,
    )


@triton.jit
def _tlx_gfx950_cross_attn_bwd_split_softmax_dkdv(  # noqa: TR001
    Q,
    K,
    DOut,
    DK,
    seq_offsets,
    seq_offsets_q,
    M,
    Delta,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_dom,
    stride_doh,
    stride_dkn,
    stride_dkh,
    stride_mm,
    alpha,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    if start_n >= seq_len_kv:
        return
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    Q += seq_start_q * stride_qm + off_h * stride_qh
    K += seq_start_kv * stride_kn + off_h * stride_kh
    DOut += seq_start_q * stride_dom + off_h * stride_doh
    DK += seq_start_kv * stride_dkn + off_h * stride_dkh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < seq_len_kv
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    scaled_alpha = alpha * 1.44269504
    offs_m_base = tl.arange(0, BLOCK_M)
    for start_m in tl.range(0, seq_len_q, BLOCK_M):
        offs_m = start_m + offs_m_base
        mask_m = offs_m < seq_len_q
        q = tl.load(
            Q + offs_m[:, None] * stride_qm + offs_d[None, :],
            mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        q_trans = tl.trans(q)
        do = tl.load(
            DOut + offs_m[:, None] * stride_dom + offs_d[None, :],
            mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        do_trans = tl.trans(do)
        valid = mask_n[:, None] & mask_m[None, :]
        qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
        m = tl.load(
            M + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
            mask=mask_m,
        )
        p = tl.math.exp2(qk_trans * scaled_alpha - m[None, :])
        p = tl.where(valid, p, 0.0)
        p_bf16 = p.to(k.dtype)
        dk += tl.dot(p_bf16, do, allow_tf32=ALLOW_TF32)
        dact = tl.dot(k, do_trans, allow_tf32=ALLOW_TF32)
        delta = tl.load(
            Delta + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
            mask=mask_m,
        )
        dqk = (p * (dact - delta[None, :])).to(k.dtype)
        dk += tl.dot(dqk, q, allow_tf32=ALLOW_TF32) * alpha

    tl.store(
        DK + offs_n[:, None] * stride_dkn + offs_d[None, :],
        dk.to(k.dtype),
        mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
    )


@triton.jit
def _tlx_gfx950_cross_attn_bwd_split_softmax_dq(  # noqa: TR001
    Q,
    K,
    DOut,
    DQ,
    seq_offsets,
    seq_offsets_q,
    M,
    Delta,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_mm,
    alpha,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    start_m = tl.program_id(1) * BLOCK_M
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
    if start_m >= seq_len_q:
        return

    Q += seq_start_q * stride_qm + off_h * stride_qh
    K += seq_start_kv * stride_kn + off_h * stride_kh
    DOut += seq_start_q * stride_dom + off_h * stride_doh
    DQ += seq_start_q * stride_dqm + off_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len_q
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    do = tl.load(
        DOut + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    m = tl.load(
        M + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
        mask=mask_m,
    )
    delta = tl.load(
        Delta + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
        mask=mask_m,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    scaled_alpha = alpha * 1.44269504
    offs_n_base = tl.arange(0, BLOCK_N)
    for start_n in tl.range(0, seq_len_kv, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < seq_len_kv
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        k_trans = tl.trans(k)
        valid = mask_m[:, None] & mask_n[None, :]
        qk = tl.dot(q, k_trans, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(qk * scaled_alpha - m[:, None])
        p = tl.where(valid, p, 0.0)
        dact = tl.dot(do, k_trans, allow_tf32=ALLOW_TF32)
        dqk = (p * (dact - delta[:, None])).to(k.dtype)
        dq += tl.dot(dqk, k, allow_tf32=ALLOW_TF32) * alpha

    tl.store(
        DQ + offs_m[:, None] * stride_dqm + offs_d[None, :],
        dq.to(q.dtype),
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
    )


def tlx_gfx950_cross_attn_bwd(  # noqa C901
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    G: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    truncate_method: str = "none",
    enable_tma: bool = False,
    v3_ttgir_variant: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not enable_tma, "TMA is CUDA-only and not supported on gfx950"
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets

    # redq only. The v3 source picks between three backward kernels through
    # `BwdVariant`; this port carries the reduce-dq one alone, so the dtype
    # choice the source spreads across four branches collapses: dq is reduced
    # through memory and is always fp32, and dk/dv are each written once by the
    # program that owns them, so they stay in k's dtype for both G == 1 (no
    # collision) and G > 1 (PER_KV_HEAD groups the query heads in registers).
    use_per_kv_head = G > 1
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=k.dtype)
    if shared_kv:
        dv = dk
    else:
        dv = torch.empty_like(v, dtype=k.dtype)

    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    if not shared_kv:
        dv = switch_to_contiguous_if_needed(dv)

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dk.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"
    # Both backward paths read the scale with a scalar `tl.load`, matching the
    # forward.
    assert attn_scale.ndim == 0, ("per-row attn_scale is not supported; pass a 0-dim tensor")

    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape

    # Pointer loads read DO using q's folded row layout. In the GQA fold,
    # forward saves q as [total * G, H_kv, D] but returns out as [total, H, D],
    # so reshape dout back to the saved-q layout before deriving DO strides.
    dout = dout.reshape(total_seq_len_q, H, dout.shape[-1])

    if v3_ttgir_variant == "split_softmax":
        torch._assert(G == 1, "Split-softmax requires G=1")
        torch._assert(H == 1, "Split-softmax requires one attention head")
        torch._assert(
            DimQ == 128 and DimV == 128,
            "Split-softmax requires D=128",
        )
        torch._assert(
            num_softmax_heads == H,
            "Split-softmax requires all heads to use softmax",
        )
        torch._assert(not causal, "Split-softmax is non-causal")
        torch._assert(
            attn_scale.ndim == 0,
            "Split-softmax requires scalar attention scale",
        )
        torch._assert(shared_kv, "Split-softmax requires shared K/V")
        torch._assert(
            seq_offsets_q is not None,
            "Split-softmax requires query offsets",
        )
        dq_softmax = torch.empty_like(q)
        dk_softmax = torch.empty_like(k)
        _tlx_gfx950_cross_attn_bwd_split_softmax_dkdv[(Z * H, triton.cdiv(max_seq_len, SPLIT_SOFTMAX_DKDV_BLOCK_N))](
            Q=q,
            K=k,
            DOut=dout,
            DK=dk_softmax,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            M=M,
            Delta=Delta,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dkn=dk_softmax.stride(0),
            stride_dkh=dk_softmax.stride(1),
            stride_mm=stride_mm,
            alpha=alpha,
            # pyrefly: ignore [bad-argument-type]
            H=H,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_M=SPLIT_SOFTMAX_DKDV_BLOCK_M,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_N=SPLIT_SOFTMAX_DKDV_BLOCK_N,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_D=DimQ,
            # pyrefly: ignore [bad-argument-type]
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            # pyrefly: ignore [unexpected-keyword]
            num_warps=SPLIT_SOFTMAX_DKDV_NUM_WARPS,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=SPLIT_SOFTMAX_DKDV_NUM_STAGES,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=SPLIT_SOFTMAX_DKDV_MATRIX_INSTR_NONKDIM,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=SPLIT_SOFTMAX_DKDV_WAVES_PER_EU,
        )
        _tlx_gfx950_cross_attn_bwd_split_softmax_dq[(Z * H, triton.cdiv(max_q_len, SPLIT_SOFTMAX_DQ_BLOCK_M))](
            Q=q,
            K=k,
            DOut=dout,
            DQ=dq_softmax,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            M=M,
            Delta=Delta,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq_softmax.stride(0),
            stride_dqh=dq_softmax.stride(1),
            stride_mm=stride_mm,
            alpha=alpha,
            # pyrefly: ignore [bad-argument-type]
            H=H,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_M=SPLIT_SOFTMAX_DQ_BLOCK_M,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_N=SPLIT_SOFTMAX_DQ_BLOCK_N,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_D=DimQ,
            # pyrefly: ignore [bad-argument-type]
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            # pyrefly: ignore [unexpected-keyword]
            num_warps=SPLIT_SOFTMAX_DQ_NUM_WARPS,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=SPLIT_SOFTMAX_DQ_NUM_STAGES,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=SPLIT_SOFTMAX_DQ_MATRIX_INSTR_NONKDIM,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=SPLIT_SOFTMAX_DQ_WAVES_PER_EU,
        )
        return (
            dq_softmax,
            dk_softmax,
            torch.empty(0, dtype=v.dtype, device=v.device),
        )

    if v3_ttgir_variant is not None:
        assert v3_ttgir_variant in (
            "v3_ttgir",
            "v3_ttgir_direct_dq",
            "v3_ttgir_atomic_dq",
            "v3_ttgir_bf16_dq",
            "v3_ttgir_bf16_prefetch_dq",
            "v3_ttgir_bf16_stage_qdo",
            "v3_ttgir_fp32_stage_qdo",
            "v3_ttgir_fp32_stage_qdo_prefetch_dq",
            "v3_ttgir_fp32_pipeline_qdo",
        )
        torch._assert(G == 1, "V3 TTGIR baseline requires G=1")
        torch._assert(
            H == 1 and k.shape[1] == 1 and v.shape[1] == 1,
            "V3 TTGIR baseline requires one Q/K/V head",
        )
        torch._assert(
            DimQ == 128 and DimV == 128,
            "V3 TTGIR baseline requires D=128",
        )
        torch._assert(
            q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
            and dout.dtype == torch.bfloat16,
            "V3 TTGIR baseline requires BF16 tensors",
        )
        torch._assert(max_q_len == 256, "V3 TTGIR baseline requires max Q=256")
        torch._assert(
            num_softmax_heads == 1,
            "V3 TTGIR baseline requires one softmax head",
        )
        torch._assert(not causal, "V3 TTGIR baseline is non-causal")
        torch._assert(
            attn_scale.ndim == 0,
            "V3 TTGIR baseline requires scalar attention scale",
        )
        torch._assert(shared_kv, "V3 TTGIR baseline requires shared K/V")
        torch._assert(
            k.data_ptr() == v.data_ptr(),
            "V3 TTGIR baseline requires aliased K/V storage",
        )
        torch._assert(
            num_targets is not None,
            "V3 TTGIR baseline requires target counts",
        )
        torch._assert(
            seq_offsets_q is not None,
            "V3 TTGIR baseline requires query offsets",
        )
        if v3_ttgir_variant in (
                "v3_ttgir_atomic_dq",
                "v3_ttgir_bf16_dq",
                "v3_ttgir_bf16_prefetch_dq",
                "v3_ttgir_bf16_stage_qdo",
        ):
            dq_v3 = torch.empty_like(q)
        elif v3_ttgir_variant != "v3_ttgir":
            dq_v3 = torch.empty_like(q, dtype=torch.float32)
        else:
            dq_v3 = torch.zeros_like(q, dtype=torch.float32)
        dk_v3 = torch.empty_like(k)
        dv_v3 = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
        _tlx_gfx950_cross_attn_v3_ttgir_bwd[(Z * H, 1)](
            q,
            k,
            v,
            dout,
            q.shape[0],
            k.shape[0],
            seq_offsets,
            seq_offsets_q,
            dq_v3,
            dk_v3,
            dv_v3,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            dout.stride(0),
            dout.stride(1),
            dq_v3.stride(0),
            dq_v3.stride(1),
            dk_v3.stride(0),
            dk_v3.stride(1),
            dv_v3.stride(0),
            dv_v3.stride(1),
            alpha,
            max_seq_len,
            attn_scale,
            M,
            Delta,
            num_targets,
            Z,
            triton.next_power_of_2(Z),
            max_q_len,
            autotune_max_seq_len(max_seq_len),
            # pyrefly: ignore [bad-argument-type]
            DIRECT_FIRST_DQ_STORE=v3_ttgir_variant != "v3_ttgir",
            # pyrefly: ignore [bad-argument-type]
            ATOMIC_DQ=v3_ttgir_variant == "v3_ttgir_atomic_dq",
            # pyrefly: ignore [bad-argument-type]
            PREFETCH_DQ=v3_ttgir_variant in (
                "v3_ttgir_bf16_prefetch_dq",
                "v3_ttgir_bf16_stage_qdo",
                "v3_ttgir_fp32_stage_qdo_prefetch_dq",
                "v3_ttgir_fp32_pipeline_qdo",
            ),
            # pyrefly: ignore [bad-argument-type]
            STAGE_QDO=v3_ttgir_variant in (
                "v3_ttgir_bf16_stage_qdo",
                "v3_ttgir_fp32_stage_qdo",
                "v3_ttgir_fp32_stage_qdo_prefetch_dq",
                "v3_ttgir_fp32_pipeline_qdo",
            ),
            # pyrefly: ignore [bad-argument-type]
            PIPELINE_QDO=v3_ttgir_variant == "v3_ttgir_fp32_pipeline_qdo",
            # pyrefly: ignore [unexpected-keyword]
            num_warps=4,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=1,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=16,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=1,
        )
        return (
            dq_v3.to(q.dtype),
            dk_v3,
            dv_v3,
        )

    if use_per_kv_head:
        H_kv = H // G
        grid = lambda meta: (Z * H_kv, )  # noqa E731
    else:
        grid = lambda meta: (Z * H, 1)  # noqa E731
    _tlx_gfx950_cross_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        DO=dout,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DQ=dq,
        DK=dk,
        DV=dv,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        max_seq_len=max_seq_len,
        attn_scale=attn_scale,
        M=M,
        Delta=Delta,
        stride_mm=stride_mm,
        num_targets=num_targets,
        Z=Z,
        AUTOTUNE_Z=next_power_of_2(Z),
        H=H,
        G=G,
        num_softmax_heads=num_softmax_heads,
        max_q_len=next_power_of_2(max_q_len),
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        SHARED_KV=shared_kv,
        TRUNCATE_METHOD=truncate_method,
        HAS_CAUSAL=causal,
        HAS_NUM_TARGETS=num_targets is not None,
        PER_KV_HEAD=use_per_kv_head,
    )

    # When shared_kv=True, dv aliases dk. Return a placeholder so the autograd
    # wrapper can expose the combined K/V gradient through the K input only.
    if shared_kv:
        dv = torch.empty(0)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class _TlxGfx950CrossAttentionFunction(torch.autograd.Function):

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor],
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        seq_offsets_q: torch.Tensor,
        max_q_len: int,
        shared_kv: bool,
        num_softmax_heads: int,
        enable_tma: bool,
        truncate_method: str,
        num_targets: Optional[torch.Tensor] = None,
        causal: bool = False,
        v3_ttgir_variant: Optional[str] = None,
    ):
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        if not shared_kv:
            assert v is not None, "v is required when shared_kv is False"
            v = switch_to_contiguous_if_needed(v)
        else:
            v = k
        total_seq_len_q, H, DimQ = q.shape
        _, H_kv, DimV = v.shape
        assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
        G = H // H_kv  # GQA group size (number of Q heads per KV head)
        if total_seq_len_q == 0:
            out = torch.zeros(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
            return out

        if (G > 1 and H_kv == 1 and (num_softmax_heads == 0 or num_softmax_heads == H) and not causal):
            # GQA can not enabled with causal masking
            seq_offsets_q = seq_offsets_q * G
            max_q_len = max_q_len * G
            num_softmax_heads = num_softmax_heads // G
            q = q.view(total_seq_len_q * G, H_kv, DimQ)
            G = 1

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}

        if v3_ttgir_variant == "split_softmax":
            torch._assert(
                truncate_method == "none",
                "Split-softmax does not support truncation",
            )
            torch._assert_async(
                torch.all(seq_offsets[1:] - seq_offsets[:-1] <= max_seq_len),
                "Split-softmax KV lengths must fit max_seq_len",
            )
            torch._assert_async(
                torch.all(seq_offsets_q[1:] - seq_offsets_q[:-1] <= max_q_len),
                "Split-softmax query lengths must fit max_q_len",
            )

        ctx.v3_ttgir_variant = v3_ttgir_variant
        out, M = tlx_gfx950_cross_attn_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            max_q_len=max_q_len,
            attn_scale=attn_scale,
            shared_kv=shared_kv,
            num_softmax_heads=num_softmax_heads,
            G=G,
            enable_tma=enable_tma,
            truncate_method=truncate_method,
            num_targets=num_targets,
            causal=causal,
        )

        saved_tensors = [q, k, v, seq_offsets, attn_scale, seq_offsets_q]
        if num_softmax_heads > 0:
            saved_tensors.extend([M, out])
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.causal = causal
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        ctx.truncate_method = truncate_method
        ctx.shared_kv = shared_kv
        ctx.enable_tma = enable_tma
        ctx.num_softmax_heads = num_softmax_heads
        ctx.stride_mm = M.stride(0)
        ctx.total_seq_len_q = total_seq_len_q
        ctx.G = G
        ctx.H = H
        ctx.H_kv = H_kv
        return out.view(total_seq_len_q, H, DimV)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(  # noqa C901
        ctx, dout: torch.Tensor) -> Tuple[
            None,
            None,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale, seq_offsets_q = saved_tensors[:6]
        idx = 6
        num_softmax_heads = ctx.num_softmax_heads
        # Reshape dout from [T, H, DimV] to folded [T*G, H_kv, DimV].
        #
        # This check is inherited from the source and does not fire for the
        # all-softmax fold: forward stores the post-fold `num_softmax_heads`
        # (H // G, so 1) while `ctx.H` is the pre-fold count, so the
        # `num_softmax_heads == ctx.H` arm is never true there. It is inert
        # either way. `dout` is contiguous, so viewing [T, H, DimV] as
        # [T*G, H_kv, DimV] moves no data, and the op below reshapes to
        # exactly that using the saved folded q's shape whether or not this
        # branch ran. Verified on H=4, H_kv=1, non-causal with
        # num_softmax_heads in {0, 4}: gradients match the reference either
        # way.
        if (ctx.H > 1 and ctx.H_kv == 1 and (num_softmax_heads == 0 or num_softmax_heads == ctx.H) and not ctx.causal):
            dout = dout.view(ctx.total_seq_len_q * ctx.H, -1, dout.shape[-1])
        if num_softmax_heads > 0:
            M = saved_tensors[idx]
            idx += 1
            out = saved_tensors[idx]
            idx += 1
            Delta = torch.empty_like(M)
            pre_grid = (triton.cdiv(out.shape[0], 128), num_softmax_heads)
            _attn_bwd_preprocess[pre_grid](
                out, dout, Delta, out.shape[0], H=out.shape[1], softmax_heads=num_softmax_heads,
                BLOCK_M=128,  # pyrefly: ignore [bad-argument-type]
                HEAD_DIM=out.shape[2],  # pyrefly: ignore [bad-argument-type]
            )
        else:
            M = torch.empty(0, device=q.device, dtype=torch.float32)
            Delta = torch.empty(0, device=q.device, dtype=torch.float32)
        if ctx.has_num_targets:
            num_targets = saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        dq, dk, dv = tlx_gfx950_cross_attn_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            max_seq_len=ctx.max_seq_len,
            alpha=ctx.alpha,
            max_q_len=ctx.max_q_len,
            seq_offsets_q=seq_offsets_q,
            num_targets=num_targets,
            causal=ctx.causal,
            shared_kv=ctx.shared_kv,
            num_softmax_heads=num_softmax_heads,
            M=M,
            Delta=Delta,
            stride_mm=ctx.stride_mm,
            G=ctx.G,
            truncate_method=ctx.truncate_method,
            enable_tma=ctx.enable_tma,
            v3_ttgir_variant=ctx.v3_ttgir_variant,
        )
        dq = dq.view(ctx.total_seq_len_q, ctx.H, -1)
        return (
            None,
            None,
            dq,
            dk,
            dv if not ctx.shared_kv else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.jit.unused
@torch.fx.wrap
def tlx_gfx950_cross_attn_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = False,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
    v3_ttgir_variant: Optional[str] = None,
) -> torch.Tensor:
    # Self-attention-style usage: q and kv are the same sequence. The v3 source
    # resolves this in its backward op, which this forward-only copy drops, so
    # it has to be resolved here instead -- `tlx_gfx950_cross_attention_v1.py`,
    # whose public name this file took over, did the same in its forward.
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets

    # Dynamo cannot trace autograd.Function.apply when the same tensor object
    # is passed to multiple forward inputs. Preserve aliasing with a zero-copy
    # view while giving k and v distinct Python objects.
    if v is k:
        v = v.view(v.shape)
    return _TlxGfx950CrossAttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v if not shared_kv else None,
        seq_offsets,
        attn_scale,
        seq_offsets_q,
        max_q_len,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
        v3_ttgir_variant,
    )


def tlx_gfx950_cross_attn_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = False,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
    v3_ttgir_variant: Optional[str] = None,
) -> torch.Tensor:
    assert not sort_by_length, ("sort_by_length not supported by the v3 cross attention kernel")
    return tlx_gfx950_cross_attn_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        shared_kv=shared_kv,
        enable_tma=enable_tma,
        actual_max_seq_len=actual_max_seq_len,
        truncate_method=truncate_method,
        v3_ttgir_variant=v3_ttgir_variant,
    )


@tlx_gfx950_cross_attn_fwd.register_fake
@tlx_gfx950_cross_attn_fwd.register_kernel("cpu")
def _tlx_gfx950_cross_attn_fwd_fake(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = False,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L, H, _ = q.shape
    if num_softmax_heads > 0:
        M = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=v.device)
    else:
        # The real op returns a 0-element M here. The v3 source's fake returns
        # `torch.empty(1)`, which reports the wrong size to `torch.compile` /
        # `torch.export` and fails `torch.library.opcheck`.
        M = torch.empty(0, dtype=torch.float32, device=v.device)
    return (
        torch.empty(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype),
        M,
    )
