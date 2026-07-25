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

# pyre-unsafe

#!/usr/bin/env python3

from typing import List, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef, fast_expf,
)

HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9  # >= H100
)

if HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def tanh_approx_fp32(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.f32 $0, $1;
            """,
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return output

    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        # Replace divf(1, 1 + expf(-x)) with (1 + tanhf(x/2)) / 2
        # If an approximate instruction exists.
        x = x * 0.5
        if MULT_BY_X:
            return x * (tanh_approx_fp32(x) + 1)
        else:
            return (1 + tanh_approx_fp32(x)) * 0.5

else:
    # Don't approximate without an instruction for hardware compatibility
    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        if MULT_BY_X:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(x, 1.0 + fast_expf(-x))
        else:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.jit
def _attn_bwd_preprocess(
    Out,
    DOut,
    Delta,
    total_seq_len_q,
    H,
    softmax_heads,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_h = tl.program_id(1)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    masks = offs_m < total_seq_len_q
    offs_n = tl.arange(0, HEAD_DIM)
    o = tl.load(
        Out + off_h * HEAD_DIM + offs_m[:, None] * HEAD_DIM * H + offs_n[None, :],
        mask=masks[:, None],
    ).to(tl.float32)
    do = tl.load(
        DOut + off_h * HEAD_DIM + offs_m[:, None] * HEAD_DIM * H + offs_n[None, :],
        mask=masks[:, None],
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_h + offs_m * softmax_heads, delta, mask=masks)


@triton.jit
def backward_common_preprocess(M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm):
    if off_h < num_softmax_heads:
        M_off = M + off_h + seq_start_q * stride_mm
        Delta_off = Delta + off_h + seq_start_q * stride_mm
    else:  # placeholder
        M_off = M
        Delta_off = Delta
    return M_off, Delta_off


@triton.jit
def backward_d_silu_activation(
    dact_qk_trans,
    pT,  # pT represents sig_trans
    qk_trans,
    scale,
    valid_mask_trans,
):
    dqk_trans = dact_qk_trans * pT * (1 + qk_trans * (1 - pT)) * scale[None, :]
    dqk_trans = tl.where(valid_mask_trans, dqk_trans, 0)
    return dqk_trans


@triton.jit
def backward_d_softmax_activation(dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT,  # pT represents sig_trans
                                  ):
    Di = tl.load(Delta_off + offs_m * stride_mm, mask=mask_m)
    dqk_trans = pT * (dact_qk_trans - Di[None, :])
    return dqk_trans


@triton.jit
def backward_silu_activation(qk_trans, alpha, valid_mask_trans, dtype_k, scale):
    masked_alpha = tl.where(valid_mask_trans, alpha, 0.0)
    qk_trans = qk_trans * masked_alpha
    pT = fast_silu(qk_trans, MULT_BY_X=False)
    silu_trans = qk_trans * pT * scale[None, :]
    act_qk_trans = silu_trans.to(dtype_k)
    return qk_trans, act_qk_trans, pT


@triton.jit
def backward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr):
    valid_mask_trans = (offs_m[None, :] < seq_len_q) & (offs_n[:, None] < seq_len_kv)
    if HAS_CAUSAL:
        offs_m = offs_m + seq_len_kv - uih_len_q
        causal_mask_trans = offs_m[None, :] >= offs_n[:, None]
        valid_mask_trans = valid_mask_trans & causal_mask_trans
    return valid_mask_trans


def forward_custom_vars(q: torch.Tensor, num_softmax_heads: int,
                        saved_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, int]:
    assert num_softmax_heads <= q.shape[1]
    M = torch.empty((q.shape[0], num_softmax_heads), device=q.device, dtype=torch.float32)
    saved_tensors.append(M)
    stride_mm = M.stride(0)
    return M, stride_mm


@triton.jit
def forward_epilogue(
    acc,
    l_i,
    m_i,
    offs_m,
    seq_start_q,
    stride_mm,
    seq_len_q,
    M,
    off_h,
    num_softmax_heads: tl.constexpr,
):
    if off_h + 1 < num_softmax_heads + 1:  # For AOTInductor lowering walkaround.
        acc = acc / l_i[:, None]
        m_i += tl.math.log2(l_i)
        m_ptrs = M + (offs_m + seq_start_q) * stride_mm + off_h
        tl.store(m_ptrs, m_i, mask=offs_m < seq_len_q)
    return acc


@triton.jit
def forward_softmax_activation(qk, alpha, valid_mask, m_i, acc, l_i):
    qk = qk * alpha * 1.44269504
    qk = tl.where(valid_mask, qk, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    act_qk = tl.math.exp2(qk)
    corr = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(act_qk, 1)
    acc = acc * corr[:, None]
    l_i = l_i * corr + l_ij
    m_i = m_ij
    return act_qk, acc, l_i, m_i


@triton.jit
def forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M):
    if off_h < num_softmax_heads:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    else:  # placeholder
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    return m_i, l_i


@triton.jit
def forward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr):
    valid_mask = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
    if HAS_CAUSAL:
        offs_m = offs_m + seq_len_kv - uih_len_q
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = valid_mask & causal_mask
    return valid_mask


@triton.jit
def target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS: tl.constexpr):
    if HAS_NUM_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    return n_targets


@triton.jit
def uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS: tl.constexpr):
    if HAS_NUM_TARGETS:
        uih_len_q = seq_len_q - n_targets
    else:
        uih_len_q = seq_len_q
    return uih_len_q
