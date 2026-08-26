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

#!/usr/bin/env python3
"""
TLX HSTU ragged attention targeting the gfx950 (AMD MI350X) architecture.

Standalone tutorial copy of the hammer template kernel of the same name. It has
no fbcode dependencies: the handful of leaf helpers it used from
`hammer/ops/triton/utils.py` live in the sibling `stubs.py`, and the backward is
a plain function instead of a `hammer::` custom op (the autograd `Function`
calls it directly). Everything else -- kernels, autotune configs, variant
selection -- is carried over unchanged.

The forward is a port of the hammer ragged HSTU attention kernel. Relative to
that kernel this copy:

  - drops the attention-bias backward (the relative-bias variant is
    forward-only here); the plain backward is present and is where the TLX
    scheduling work lives
  - drops the CUDA autotune configs from `_get_fw_configs`, keeping the AMD
    list only
  - drops `forward_silu_activation`, which had no callers; the live silu is
    `fast_silu`
  - drops the `_get_named_specs` TritonCC AOT table and its
    `register_tritoncc_specs` registrations. AOT pre-compilation exists to
    remove JIT warmup from inference serving (the `standalone_cint_*` /
    `amd_standalone_cint_*` packages) and only ever covered the relative-bias
    variant. It is metadata-only, so dropping it costs no performance, and
    `tritoncc_specs` asserts that spec keys exactly match the kernel signature
    -- which would break on every signature change during the TLX port.

  - drops the `_persistent` forward variant, which had no caller here or
    upstream; it existed only to be AOT-registered
  - drops the delta-q path (`IS_DELTA_Q`, `delta_x_offsets`, `DeltaSize`), an
    incremental-decoding optimization for inference. The public wrappers never
    exposed it and passed `IS_DELTA_Q=False`, so Triton was already folding it
    away; removing it is readability-only.

Everything else -- softmax/silu head split (`num_softmax_heads`), attention
bias, contextual prefix, max_attn_len, targets -- is carried over
unchanged. Porting the load path to TLX is a follow-up.
"""

from typing import List, Optional, Tuple

import torch

import triton

import triton.language as tl

import triton.language.extra.tlx as tlx

from stubs import (
    autotune_max_seq_len,
    get_use_rtz,
    maybe_register_custom_op,
    pid_swizzle,
    prev_power_of_2,
    triton_autotune,
)

try:
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        from triton.language.math import fast_dividef, fast_expf


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    # Same contract as the hammer helper: only the last dim has to be packed.
    # It must NOT copy a tensor that is merely non-contiguous in an outer dim --
    # dq/dk/dv are written in place, so a copy here would silently drop the
    # gradients. `stubs.switch_to_contiguous_if_needed` is the stricter (always
    # `.contiguous()`) variant used by the sibling kernels; do not use it here.
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# kpack is deprecated on gfx950; only gfx942 benefited from kpack=2.
KPACK: int = 1


def _get_fw_configs() -> List[triton.Config]:
    # gfx950 (MI350X) only: the CUDA config list from the source kernel is
    # dropped in this port.
    configs = []
    for BLOCK_M, num_warps in [(32, 2), (64, 2), (64, 4), (128, 4)]:
        for num_stages in [1, 2]:
            for matrix_instr_nonkdim in [16, 32]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": 32,
                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                            "waves_per_eu": 0,
                            "kpack": KPACK,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    ))
    return configs


HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9  # >= H100
)

# AMD CDNA3/4: polynomial tanh avoids Trans unit (exp at 1/4 VALU rate).
# Uses degree-9 minimax odd polynomial on [-4.5, 4.5], clamped outside.
# Max error < 5e-4, invisible at bf16 precision (eps ~0.0078).
HAS_AMD_POLY_SIGMOID = torch.version.hip is not None

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

elif HAS_AMD_POLY_SIGMOID:

    @triton.jit
    def _tanh_poly(x):
        """Degree-9 minimax polynomial approximation of tanh(x).

        Uses Horner form: tanh(x) ~= x * (a0 + u*(a1 + u*(a2 + u*(a3 + u*a4))))
        where u = x*x. All ops are FMA on VALU (full throughput on CDNA3/4),
        avoiding the Trans unit used by exp/tanh hardware instructions.

        Clamped to [-1, 1] for |x| > 4.5 where |tanh(x)| > 0.9999.
        """
        # Minimax coefficients for tanh on [-4.5, 4.5]
        u = x * x
        # Horner evaluation: p(u) = a0 + u*(a1 + u*(a2 + u*(a3 + u*a4)))
        p = -0.000198527 + u * 0.00972515  # a4*u + a3
        p = p * u + (-0.0533740)  # ... + a2
        p = p * u + 0.133392  # ... + a1
        p = p * u + 1.0  # ... + a0
        result = x * p
        # Clamp for |x| > 4.5 where polynomial diverges
        result = tl.where(result > 1.0, 1.0, result)
        result = tl.where(result < -1.0, -1.0, result)
        return result

    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        # sigmoid(x) = 0.5 * (1 + tanh(x/2))
        # SiLU(x) = x * sigmoid(x)
        x_half = x * 0.5
        if MULT_BY_X:
            return x_half * (_tanh_poly(x_half) + 1.0)
        else:
            return (_tanh_poly(x_half) + 1.0) * 0.5

else:
    # Fallback for non-AMD, non-H100 hardware
    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        if MULT_BY_X:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(x, 1.0 + fast_expf(-x))
        else:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.jit
def forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M):
    if off_h < num_softmax_heads:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    return m_i, l_i


@triton.jit
def forward_softmax_activation(qk, alpha, valid_mask, m_i, acc, l_i):
    qk = qk * alpha * 1.44269504
    qk = tl.where(valid_mask, qk, -1e6)
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
def forward_epilogue(
    acc,
    l_i,
    m_i,
    offs_m,
    seq_len,
    num_softmax_heads: tl.constexpr,
    off_h,
    M_buffer,
    seq_start,
    BLOCK_M: tl.constexpr,
):
    if off_h + 1 < num_softmax_heads + 1:
        acc = acc / l_i[:, None]
        M_i = m_i + tl.math.log2(l_i)
        mask_m = offs_m < seq_len
        M_ptrs = M_buffer + (seq_start + offs_m) * num_softmax_heads + off_h
        tl.store(M_ptrs, M_i, mask=mask_m)
    return acc


@triton.jit
def backward_softmax_activation_recompute(
    qk_trans,
    alpha,
    invalid_mask_trans,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    qk_scaled = qk_trans * alpha * 1.44269504
    qk_scaled = tl.where(invalid_mask_trans, qk_scaled, -1e6)
    m = tl.max(qk_scaled, axis=0)
    qk_shifted = qk_scaled - m[None, :]
    p_trans = tl.math.exp2(qk_shifted)
    p_trans = tl.where(invalid_mask_trans, p_trans, 0.0)
    l = tl.sum(p_trans, axis=0)
    p_trans = p_trans / l[None, :]
    return p_trans, m, l


@triton.jit
def backward_softmax_activation(qk_trans, alpha, valid_mask_trans, M_i, k):
    qk_trans = qk_trans * 1.44269504
    pT = tl.math.exp2(qk_trans - M_i[None, :])
    pT = tl.where(valid_mask_trans, pT, 0.0)
    act_qk_trans = pT.to(k.dtype)
    return qk_trans, act_qk_trans, pT


@triton.jit
def backward_d_softmax_activation(dact_qk_trans, pT, Delta_block):
    dqk_trans = pT * (dact_qk_trans - Delta_block[None, :])
    return dqk_trans


@triton.jit
# Triton TR001: the caller supplies the selected backward tile dimensions.
def _attn_bwd_preprocess(  # noqa: TR001
    Out,
    DOut,
    Delta,
    seq_offsets,
    stride_om,
    stride_oh,
    stride_dom,
    stride_doh,
    H,
    num_softmax_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    if off_h >= num_softmax_heads:
        return
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    if tl.program_id(0) * BLOCK_M >= seq_len:
        return
    mask_m = offs_m < seq_len
    offs_d = tl.arange(0, BLOCK_D_V)
    # Triton TR003: BLOCK_D_V is the exact output hidden dimension.
    o = tl.load(  # noqa: TR003
        Out + seq_start * stride_om + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(  # noqa: TR003
        DOut + seq_start * stride_dom + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + seq_start * num_softmax_heads + offs_m * num_softmax_heads + off_h,
        delta,
        mask=mask_m,
    )


@triton.jit
def _tlx_gfx950_ragged_hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    mask_m,
    mask_n,
    q,
    K_base,
    V_base,
    stride_kn,
    stride_vn,
    lds_k,
    lds_v,
    n_targets,
    ts_1_ptrs,
    ts_0,
    TW,
    PW,
    alpha,
    scale,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    bias_ptrs,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    m_i,
    acc,
    l_i,
    IS_MASKLESS: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_D_Q)
    offs_d_v = tl.arange(0, BLOCK_D_V)
    mask_kv = offs_n < seq_len
    k_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d_q[None, :]
    v_ptrs = V_base + offs_n[:, None] * stride_vn + offs_d_v[None, :]
    k_local = tlx.local_view(lds_k, 0)
    v_local = tlx.local_view(lds_v, 0)

    # -- compute qk ----
    # K is fetched row-major and transposed as an LDS view, so the global access
    # stays contiguous (a strided/transposed global load is far slower here).
    tok_k = tlx.async_load(k_ptrs, k_local, mask=mask_kv[:, None])
    tlx.async_load_commit_group([tok_k])
    # pyrefly: ignore [bad-argument-type]
    tlx.async_load_wait_group(0)
    kt_local = tlx.local_trans(k_local)
    k = tlx.local_load(kt_local)
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    if not IS_MASKLESS:
        invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if not IS_MASKLESS:
        global_attn_mask = tl.full((BLOCK_M, BLOCK_N), True, tl.int1)
        if HAS_MAX_ATTN_LEN:
            if HAS_FULL_ATTN_SIZE:
                if INVALID_MASK_TYPE == "lower_triangular":
                    global_attn_mask = (offs_m[:, None] >= max_ids - full_attn_size) & (offs_n_minus_m < 0)
                elif INVALID_MASK_TYPE == "upper_triangular":
                    global_attn_mask = (offs_m[:, None] < full_attn_size) & (offs_n_minus_m > 0)

                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | global_attn_mask

            if INVALID_MASK_TYPE == "lower_triangular":
                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | ((offs_n_minus_m < 0) & (offs_n_minus_m >= -max_attn_len))
            elif INVALID_MASK_TYPE == "none":
                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | ((offs_n_minus_m <= max_attn_len)
                                               & (offs_n_minus_m >= -max_attn_len)
                                               & (offs_n[None, :] < max_ids))

            if HAS_FULL_ATTN_SIZE:
                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | global_attn_mask
        else:
            if INVALID_MASK_TYPE == "lower_triangular":
                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | (offs_n_minus_m < 0)
            elif INVALID_MASK_TYPE == "none":
                # pyrefly: ignore [unbound-name]
                invalid_mask = invalid_mask | (offs_n[None, :] < max_ids)
        if HAS_CONTEXTUAL_SEQ_LEN:
            # pyrefly: ignore [unbound-name]
            invalid_mask = invalid_mask | ((offs_m[:, None] == 0) & (offs_n[None, :] < max_ids))
            invalid_mask = invalid_mask | (offs_n[None, :] == 0)
    # Compute load mask for attention bias
    if IS_MASKLESS:
        bias_load_mask = mask_m[:, None]
    else:
        bias_load_mask = mask_m[:, None] & mask_n[None, :]
    if ATTN_BIAS_TYPE == "fused":
        attn_bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                if IS_MASKLESS:
                    ts_1 = tl.load(ts_1_ptrs + start_n)
                else:
                    ts_1 = tl.load(ts_1_ptrs + start_n, mask=mask_n)
            else:
                if IS_MASKLESS:
                    ts_1 = tl.load(ts_1_ptrs + start_n + 1)
                else:
                    ts_1 = tl.load(ts_1_ptrs + start_n + 1, mask=mask_n)
            ts = ts_0[:, None] - ts_1[None, :]
            ts = ts + time_delta
            ts = tl.where(ts > 1e-6, ts, 1e-6)
            ts = ts * (1.0 / time_bucket_incr)
            if BUCKET_FN == "log":
                ts = tl.log(ts)
            elif BUCKET_FN == "sqrt":
                ts = tl.sqrt(ts)
            ts = ts * (1.0 / time_bucket_div)
            ts = ts.to(tl.int32)
            ts = tl.where(ts > 0, ts, 0)
            ts = tl.where(ts < num_buckets, ts, num_buckets)
            ts_w = tl.load(
                TW + ts,
                mask=bias_load_mask,
                latency=0,
            )
            attn_bias = attn_bias + ts_w
        if USE_POS_BIAS:
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n_minus_m + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(
                    offs_pos_w < 2 * max_pos_ind - 2,
                    offs_pos_w,
                    2 * max_pos_ind - 2,
                )
            else:
                offs_pos_w = offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w = tl.load(
                PW + offs_pos_w,
                mask=bias_load_mask,
                latency=0,
            )
            attn_bias = attn_bias + pos_w
        qk = qk + attn_bias
    elif ATTN_BIAS_TYPE == "separate":
        attn_bias = tl.load(
            bias_ptrs + start_n,
            mask=bias_load_mask,
            other=0.0,
            latency=0,
        )
        qk = qk + attn_bias
    # Apply activation
    if IS_SOFTMAX:
        qk = qk * 1.44269504
        if not IS_MASKLESS:
            # pyre-fixme[61]: `invalid_mask` is defined in a matching `if not IS_MASKLESS` branch above.
            qk = tl.where(invalid_mask, qk, -1e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        act_qk = tl.math.exp2(qk)
        corr = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(act_qk, 1)
        acc = acc * corr[:, None]
        l_i = l_i * corr + l_ij
        m_i = m_ij
    else:
        # pyrefly: ignore [bad-argument-type]
        silu = fast_silu(qk, True) * scale[:, None]
        if IS_MASKLESS:
            act_qk = silu
        else:
            # pyre-fixme[61]: `invalid_mask` is defined in a matching `if not IS_MASKLESS` branch above.
            act_qk = tl.where(invalid_mask, silu, 0)
    # doing LDS staging explicitly lets the backend use the ds_read_b64_tr path and keep
    # the accumulators in AGPRs instead of Loading V straight to registers
    tok_v = tlx.async_load(v_ptrs, v_local, mask=mask_kv[:, None])
    tlx.async_load_commit_group([tok_v])
    # pyrefly: ignore [bad-argument-type]
    tlx.async_load_wait_group(0)
    v = tlx.local_load(v_local)
    if get_use_rtz() is True:
        act_qk = act_qk.to(v.dtype, "rtz")
    else:
        act_qk = act_qk.to(v.dtype)
    acc += tl.dot(act_qk, v, allow_tf32=ALLOW_TF32)
    return acc, l_i, m_i


@triton.jit
def _tlx_gfx950_ragged_hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    num_targets,
    attn_scale,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    M_buffer,
    alpha,
    Z,
    H,
    MAX_SEQ_LEN,
    DimQ,
    DimV,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    off_z,
    off_h,
    pid,
    num_softmax_heads: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if ATTN_SCALE_TYPE == "none":
            scale = 1.0 / MAX_SEQ_LEN
        elif ATTN_SCALE_TYPE == "scalar":
            scale = tl.load(attn_scale).to(tl.float32)
        else:
            tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
            scale = tl.load(attn_scale + seq_start + offs_m, mask=offs_m < seq_len).to(tl.float32)

        Q_block_ptr = tl.make_block_ptr(
            base=Q + off_h * stride_qh + seq_start * stride_qm,
            shape=(seq_len, BLOCK_D_Q),
            strides=(stride_qm, 1),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_D_Q),
            order=(1, 0),
        )
        mask_m = offs_m < seq_len
        if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS:
            ts_0_ptrs = TS + off_z * stride_ts + offs_m
            ts_1_ptrs = TS + off_z * stride_ts + offs_n
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
        elif ATTN_BIAS_TYPE == "separate":
            seq2_start = tl.load(seq2_offsets + off_z)
            bias_start = seq2_start * H + off_h * seq_len * seq_len
            off_bias = offs_m[:, None] * seq_len + offs_n[None, :]
            bias_ptrs = Bias + bias_start + off_bias

        q = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option="zero")
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        m_i, l_i = forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M)

        # K/V are addressed with raw pointer arithmetic off these bases; every
        # block derives its own offsets from start_n, so there is no pointer to
        # advance between iterations.
        K_base = K + off_h * stride_kh + seq_start * stride_kn
        V_base = V + off_h * stride_vh + seq_start * stride_vn
        # Single-buffered LDS staging, allocated once and reused by every KV
        # block across all of the loops below.
        # pyrefly: ignore [bad-argument-type]
        lds_k = tlx.local_alloc((BLOCK_N, BLOCK_D_Q), tlx.dtype_of(K), 1)
        # pyrefly: ignore [bad-argument-type]
        lds_v = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), 1)
        if INVALID_MASK_TYPE == "lower_triangular":
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len

            if HAS_CONTEXTUAL_SEQ_LEN > 0 and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len
            else:
                low = 0
                high = start_m + BLOCK_M
                if HAS_MAX_ATTN_LEN:
                    if (HAS_FULL_ATTN_SIZE > 0 and start_m + BLOCK_M >= uih_end - full_attn_size):
                        low = 0
                    else:
                        if start_m > uih_end:
                            low = uih_end - max_attn_len
                        else:
                            low = start_m - max_attn_len
                    if HAS_CONTEXTUAL_SEQ_LEN:
                        low = low if low > contextual_seq_len else 0
                    else:
                        low = low if low > 0 else 0
                if HAS_MULTIPLE_TARGETS:
                    uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                    if uih_end < start_m:
                        high = seq_len - n_targets
        elif INVALID_MASK_TYPE == "none":
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len
            low = 0
            high = uih_end
            if HAS_MAX_ATTN_LEN:
                low = start_m - max_attn_len
                low = low if low > 0 else 0
                high = start_m + BLOCK_M + max_attn_len
                high = high if high < uih_end else uih_end
        else:
            low = start_m
            high = seq_len
        if HAS_MAX_ATTN_LEN and HAS_CONTEXTUAL_SEQ_LEN:
            ctx_block_end = tl.cdiv(contextual_seq_len, BLOCK_N) * BLOCK_N
            # pyre-ignore[61]
            if low < ctx_block_end:
                low = ctx_block_end
            for start_n in range(0, contextual_seq_len, BLOCK_N):
                cur_offs_n = offs_n + start_n
                mask_n = cur_offs_n < seq_len
                if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=True,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                else:  # SiLU heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=False,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
        end_n = low
        if INVALID_MASK_TYPE == "lower_triangular":
            # Blocks fully below diagonal: start_n + BLOCK_N <= start_m
            # These blocks need no mask and no boundary check
            maskless_end = low + ((start_m - low) // BLOCK_N) * BLOCK_N
            # Clamp: maskless_end must be >= low and <= high
            # pyre-ignore[61]
            if maskless_end < low:
                # pyre-ignore[61]
                maskless_end = low
            # pyre-ignore[61]
            if maskless_end > high:
                # pyre-ignore[61]
                maskless_end = high
            if HAS_MULTIPLE_TARGETS:
                # When HAS_MULTIPLE_TARGETS, positions >= (seq_len - n_targets)
                # get clamped, which can make offs_n_minus_m == 0 for positions that
                # should be invalid. Limit maskless range so all KV positions are
                # below the clamping boundary.
                # pyre-ignore[61]
                maskless_end_mt = (low + ((seq_len - n_targets - low) // BLOCK_N) * BLOCK_N)
                # pyre-ignore[61]
                if maskless_end > maskless_end_mt:
                    # pyre-ignore[61]
                    maskless_end = maskless_end_mt

            if HAS_MAX_ATTN_LEN:
                # Tightest max_attn_len constraint from bottom row of Q block:
                # start_n >= start_m + BLOCK_M - 1 - max_attn_len
                max_attn_left = start_m + BLOCK_M - 1 - max_attn_len
                if max_attn_left <= 0:
                    # pyre-ignore[61]
                    maskless_left = low
                else:
                    maskless_left = (low + ((max_attn_left - low + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
                    # pyre-ignore[61]
                    if maskless_left < low:
                        # pyre-ignore[61]
                        maskless_left = low
            else:
                # pyre-ignore[61]
                maskless_left = low
            # Ensure maskless_left does not exceed maskless_end
            # pyre-ignore[61]
            if maskless_left > maskless_end:
                # pyre-ignore[61]
                maskless_left = maskless_end

            # Phase 1: Pre-maskless (low to maskless_left) - masked
            # pyre-ignore[61]
            for start_n in range(low, maskless_left, BLOCK_N):
                cur_offs_n = offs_n + start_n
                mask_n = cur_offs_n < seq_len
                if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=True,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                else:  # SiLU heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=False,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                end_n += BLOCK_N

            # Phase 2: Maskless (maskless_left to maskless_end) - no masks needed
            # pyre-ignore[61]
            for start_n in range(maskless_left, maskless_end, BLOCK_N):
                cur_offs_n = offs_n + start_n
                if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_m,  # placeholder, unused when IS_MASKLESS=True
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=True,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=True,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                else:  # SiLU heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_m,  # placeholder, unused when IS_MASKLESS=True
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=True,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=False,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                end_n += BLOCK_N

            # Phase 3: Post-maskless (maskless_end to high) - masked (diagonal region)
            # pyre-ignore[61]
            for start_n in range(maskless_end, high, BLOCK_N):
                cur_offs_n = offs_n + start_n
                mask_n = cur_offs_n < seq_len
                if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=True,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                else:  # SiLU heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=False,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                end_n += BLOCK_N
        else:
            # upper_triangular or other: existing loop unchanged
            # pyre-ignore[61]
            for start_n in range(low, high, BLOCK_N):
                cur_offs_n = offs_n + start_n
                mask_n = cur_offs_n < seq_len
                if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=True,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                else:  # SiLU heads
                    acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        lds_k=lds_k,
                        lds_v=lds_v,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                        # pyre-ignore[61]
                        ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        scale=scale,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        full_attn_size=full_attn_size,
                        m_i=m_i,
                        acc=acc,
                        l_i=l_i,
                        # pyrefly: ignore [bad-argument-type]
                        IS_MASKLESS=False,
                        # pyrefly: ignore [bad-argument-type]
                        IS_SOFTMAX=False,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )
                end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS and INVALID_MASK_TYPE == "lower_triangular":
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                # When high (= seq_len - n_targets) is not block-aligned, the main
                # loop's last block may extend past high into the target key region,
                # potentially overlapping with the diagonal. Avoid reprocessing keys
                # already covered by the main loop by starting from end_n instead.
                if end_n > start_m:
                    low_delta = end_n
                high_delta = start_m + BLOCK_M
                for start_delta in tl.range(
                        # pyrefly: ignore [bad-argument-type]
                        low_delta,
                        high_delta,
                        BLOCK_N,
                        # pyrefly: ignore [bad-argument-type]
                        num_stages=1,
                ):
                    cur_offs_n = offs_n + start_delta
                    mask_n = cur_offs_n < seq_len
                    if off_h + 1 < num_softmax_heads + 1:  # Softmax heads
                        acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                            start_n=start_delta,
                            seq_len=seq_len,
                            offs_m=offs_m,
                            offs_n=cur_offs_n,
                            mask_m=mask_m,
                            mask_n=mask_n,
                            q=q,
                            K_base=K_base,
                            V_base=V_base,
                            stride_kn=stride_kn,
                            stride_vn=stride_vn,
                            lds_k=lds_k,
                            lds_v=lds_v,
                            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                            ts_1_ptrs=(
                                # pyre-ignore[61]
                                ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                            ts_0=(
                                # pyre-ignore[61]
                                ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                            TW=TW,
                            PW=PW,
                            alpha=alpha,
                            scale=scale,
                            MAX_SEQ_LEN=MAX_SEQ_LEN,
                            num_buckets=num_buckets,
                            max_pos_ind=max_pos_ind,
                            time_bucket_incr=time_bucket_incr,
                            time_bucket_div=time_bucket_div,
                            time_delta=time_delta,
                            # pyre-ignore[61]
                            bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                            contextual_seq_len=contextual_seq_len,
                            max_attn_len=max_attn_len,
                            full_attn_size=full_attn_size,
                            m_i=m_i,
                            acc=acc,
                            l_i=l_i,
                            # pyrefly: ignore [bad-argument-type]
                            IS_MASKLESS=False,
                            # pyrefly: ignore [bad-argument-type]
                            IS_SOFTMAX=True,
                            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                            CAUSAL=CAUSAL,
                            BUCKET_FN=BUCKET_FN,
                            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                            USE_TIME_BIAS=USE_TIME_BIAS,
                            USE_POS_BIAS=USE_POS_BIAS,
                            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                            ALLOW_TF32=ALLOW_TF32,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            BLOCK_D_Q=BLOCK_D_Q,
                            BLOCK_D_V=BLOCK_D_V,
                        )
                    else:  # SiLU heads
                        acc, l_i, m_i = _tlx_gfx950_ragged_hstu_attn_fwd_one_block(
                            start_n=start_delta,
                            seq_len=seq_len,
                            offs_m=offs_m,
                            offs_n=cur_offs_n,
                            mask_m=mask_m,
                            mask_n=mask_n,
                            q=q,
                            K_base=K_base,
                            V_base=V_base,
                            stride_kn=stride_kn,
                            stride_vn=stride_vn,
                            lds_k=lds_k,
                            lds_v=lds_v,
                            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                            ts_1_ptrs=(
                                # pyre-ignore[61]
                                ts_1_ptrs if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                            ts_0=(
                                # pyre-ignore[61]
                                ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None),
                            TW=TW,
                            PW=PW,
                            alpha=alpha,
                            scale=scale,
                            MAX_SEQ_LEN=MAX_SEQ_LEN,
                            num_buckets=num_buckets,
                            max_pos_ind=max_pos_ind,
                            time_bucket_incr=time_bucket_incr,
                            time_bucket_div=time_bucket_div,
                            time_delta=time_delta,
                            # pyre-ignore[61]
                            bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                            contextual_seq_len=contextual_seq_len,
                            max_attn_len=max_attn_len,
                            full_attn_size=full_attn_size,
                            m_i=m_i,
                            acc=acc,
                            l_i=l_i,
                            # pyrefly: ignore [bad-argument-type]
                            IS_MASKLESS=False,
                            # pyrefly: ignore [bad-argument-type]
                            IS_SOFTMAX=False,
                            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                            CAUSAL=CAUSAL,
                            BUCKET_FN=BUCKET_FN,
                            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                            USE_TIME_BIAS=USE_TIME_BIAS,
                            USE_POS_BIAS=USE_POS_BIAS,
                            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                            ALLOW_TF32=ALLOW_TF32,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            BLOCK_D_Q=BLOCK_D_Q,
                            BLOCK_D_V=BLOCK_D_V,
                        )

        # Finalize accumulator based on head type
        if num_softmax_heads > 0:
            acc = forward_epilogue(
                acc,
                l_i,
                m_i,
                offs_m,
                seq_len,
                num_softmax_heads,
                off_h,
                M_buffer,
                seq_start,
                BLOCK_M,
            )

        # rematerialize offsets to save registers
        start_m = pid * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_D_V)
        off_o = Out + seq_start * stride_om + off_h * DimV
        out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
        tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
    ],
)
@triton.jit
def _tlx_gfx950_ragged_hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    num_targets,
    attn_scale,
    Out,
    M_buffer,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    num_softmax_heads,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)

    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    pid, off_hz = pid_swizzle(pid, off_hz, n_tile_num, H * Z)

    off_z = off_hz // H
    off_h = off_hz % H
    _tlx_gfx950_ragged_hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        TS=TS,
        TW=TW,
        PW=PW,
        Bias=Bias,
        seq2_offsets=seq2_offsets,
        num_targets=num_targets,
        attn_scale=attn_scale,
        Out=Out,
        M_buffer=M_buffer,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_ts=stride_ts,
        stride_om=stride_om,
        alpha=alpha,
        Z=Z,
        H=H,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DimQ=DimQ,
        DimV=DimV,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        num_softmax_heads=num_softmax_heads,
        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
        CAUSAL=CAUSAL,
        BUCKET_FN=BUCKET_FN,
        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
        ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
        USE_TIME_BIAS=USE_TIME_BIAS,
        USE_POS_BIAS=USE_POS_BIAS,
        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


_tlx_gfx950_ragged_hstu_attn_fwd = triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
    ],
)(_tlx_gfx950_ragged_hstu_attn_fwd.fn)


@maybe_register_custom_op("hammer::tlx_gfx950_ragged_hstu_attn_fwd", mutates_args=())
def tlx_gfx950_ragged_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert invalid_attn_mask_type in [
        "lower_triangular",
        "upper_triangular",
        "none",
    ], f"Invalid {invalid_attn_mask_type=}"
    if invalid_attn_mask_type != "lower_triangular":
        assert contextual_seq_len == 0
        assert full_attn_size == 0
        if invalid_attn_mask_type != "none":
            assert max_attn_len == 0
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape

    out_buffer: Optional[torch.Tensor] = None
    M_buffer: torch.Tensor
    if num_softmax_heads > 0:
        out = torch.empty((L, H, DimV), dtype=v.dtype, device=v.device)
        out_buffer = out.view(-1)
        M_buffer = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=v.device)
    else:
        out = torch.empty_like(v)
        out_buffer = out.view(-1)
        M_buffer = torch.empty(1, dtype=torch.float32, device=v.device)

    max_attn_len = max_attn_len or 0
    contextual_seq_len = contextual_seq_len or 0
    full_attn_size = full_attn_size or 0
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_full_attn_size = full_attn_size > 0
    attn_scale_type: str = "none"
    if attn_scale is not None:
        if attn_scale.ndim == 0:
            attn_scale_type = "scalar"
        else:
            attn_scale_type = "dynamic"

    if L == 0:
        if num_softmax_heads > 0:
            M = torch.empty(0, num_softmax_heads, dtype=torch.float32, device=v.device)
        else:
            M = torch.empty(1, dtype=torch.float32, device=v.device)
        return out, M

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )
    _tlx_gfx950_ragged_hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        seq_offsets=seq_offsets,
        TS=None,
        TW=None,
        PW=None,
        Bias=None,
        seq2_offsets=None,
        num_targets=num_targets,
        attn_scale=attn_scale,
        Out=out_buffer,
        M_buffer=M_buffer,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=None,
        stride_om=H * DimV,
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        num_buckets=None,
        max_pos_ind=None,
        time_bucket_incr=None,
        time_bucket_div=None,
        time_delta=None,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        num_softmax_heads=num_softmax_heads,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=None,
        BUCKET_FN="none",
        ATTN_BIAS_TYPE="none",
        ATTN_SCALE_TYPE=attn_scale_type,
        USE_TIME_BIAS=False,
        USE_POS_BIAS=False,
        HAS_MAX_POS_IND=False,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_FULL_ATTN_SIZE=has_full_attn_size,
    )

    if num_softmax_heads > 0:
        M = M_buffer
    else:
        # Create a dummy M tensor with valid memory for kernel pointer arithmetic
        M = torch.empty(1, dtype=torch.float32, device=v.device)

    return out, M


@triton.jit
def _tlx_gfx950_hstu_native_dq_8wave(
    dqk_trans,
    k_native,
    dq_old,
    dq_addresses,
    dq_mask,
    dq_scratch,
    alpha,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
):
    # pyre-ignore[6]
    ds = tlx.require_layout(tl.trans(dqk_trans), DS_MD_LAYOUT, pin=False)
    # pyre-ignore[6]
    k_native = tlx.require_layout(k_native, K_MD_LAYOUT, pin=False)
    ds0 = tlx.extract_slice(ds, [BLOCK_M, 32], [0, 0])
    ds1 = tlx.extract_slice(ds, [BLOCK_M, 32], [0, 32])
    k00 = tlx.extract_slice(k_native, [32, 64], [0, 0])
    k01 = tlx.extract_slice(k_native, [32, 64], [0, 64])
    k10 = tlx.extract_slice(k_native, [32, 64], [32, 0])
    k11 = tlx.extract_slice(k_native, [32, 64], [32, 64])
    dq0 = tlx.zeros((BLOCK_M, 64), tl.float32, layout=MMA_MD)
    dq1 = tlx.zeros((BLOCK_M, 64), tl.float32, layout=MMA_MD)
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds0,
        k00,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
        # pyre-ignore[6]
        initialize=True,
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds0,
        k01,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
        # pyre-ignore[6]
        initialize=True,
    )
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds1,
        k10,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds1,
        k11,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    dq0, dq1, _ = tlx.amd_mfma_commit((dq0, dq1), k11)
    dq = tl.join(dq0, dq1)
    dq = tl.permute(dq, (0, 2, 1))
    dq = tl.reshape(dq, (BLOCK_M, 128))
    tlx.local_store(dq_scratch, dq.to(tl.bfloat16))
    tl.debug_barrier()
    dq = tlx.local_load(dq_scratch).to(tl.float32)
    tl.store(
        dq_addresses,
        (dq_old + dq * alpha).to(tl.bfloat16),
        mask=dq_mask,
        eviction_policy="evict_last",
    )


@triton.jit
def _tlx_gfx950_hstu_native_dq_4wave(
    dqk_trans,
    k_native,
    dq_addresses,
    dq_mask,
    alpha,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
):
    # pyre-ignore[6]
    ds = tlx.require_layout(tl.trans(dqk_trans), DS_MD_LAYOUT, pin=False)
    # pyre-ignore[6]
    k_native = tlx.require_layout(k_native, K_MD_LAYOUT, pin=False)
    ds0 = tlx.extract_slice(ds, [BLOCK_M, 16], [0, 0])
    ds1 = tlx.extract_slice(ds, [BLOCK_M, 16], [0, 16])
    ds2 = tlx.extract_slice(ds, [BLOCK_M, 16], [0, 32])
    ds3 = tlx.extract_slice(ds, [BLOCK_M, 16], [0, 48])
    k0 = tlx.extract_slice(k_native, [16, 128], [0, 0])
    k1 = tlx.extract_slice(k_native, [16, 128], [16, 0])
    k2 = tlx.extract_slice(k_native, [16, 128], [32, 0])
    k3 = tlx.extract_slice(k_native, [16, 128], [48, 0])
    # pyre-ignore[6]
    dq_addresses = tlx.require_layout(dq_addresses, MMA_MD, pin=False)
    # pyre-ignore[6]
    dq_mask = tlx.require_layout(dq_mask, MMA_MD, pin=False)
    dq_zero = tlx.zeros((BLOCK_M, 128), tl.bfloat16, layout=MMA_MD)
    dq_old = tl.load(
        dq_addresses,
        mask=dq_mask,
        other=dq_zero,
        eviction_policy="evict_last",
    ).to(tl.float32)
    dq = tlx.zeros((BLOCK_M, 128), tl.float32, layout=MMA_MD)
    # pyre-ignore[6]
    dq = tlx.amd_scheduled_mfma(
        ds0,
        k0,
        dq,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
        # pyre-ignore[6]
        initialize=True,
    )
    # pyre-ignore[6]
    dq = tlx.amd_scheduled_mfma(
        ds1,
        k1,
        dq,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq = tlx.amd_scheduled_mfma(
        ds2,
        k2,
        dq,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq = tlx.amd_scheduled_mfma(
        ds3,
        k3,
        dq,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    dq, _ = tlx.amd_mfma_commit(dq, k3)
    # pyre-ignore[6]
    alpha_native = tlx.require_layout(
        tl.broadcast_to(alpha, (BLOCK_M, 128)),
        MMA_MD,
        # pyre-ignore[6]
        pin=False,
    )
    tl.store(
        dq_addresses,
        (dq_old + dq * alpha_native).to(tl.bfloat16),
        mask=dq_mask,
        eviction_policy="evict_last",
    )


@triton.jit
def _tlx_gfx950_hstu_native_dq_atomic_4wave(
    ds_stage,
    k_native,
    DQ_ACC,
    start_m,
    mask_m,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 128)
    # pyre-ignore[6]
    k_native = tlx.require_layout(k_native, K_MD_LAYOUT, pin=False)
    dq0 = tlx.zeros((BLOCK_M, 64), tl.float32, layout=MMA_MD)
    dq1 = tlx.zeros((BLOCK_M, 64), tl.float32, layout=MMA_MD)
    ds0 = tlx.local_load(
        tlx.local_slice(ds_stage, [0, 0], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    ds1 = tlx.local_load(
        tlx.local_slice(ds_stage, [0, 32], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    k00 = tlx.extract_slice(k_native, [32, 64], [0, 0])
    k01 = tlx.extract_slice(k_native, [32, 64], [0, 64])
    k10 = tlx.extract_slice(k_native, [32, 64], [32, 0])
    k11 = tlx.extract_slice(k_native, [32, 64], [32, 64])
    ds2 = tlx.local_load(
        tlx.local_slice(ds_stage, [0, 64], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds0,
        k00,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
        # pyre-ignore[6]
        initialize=True,
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds0,
        k01,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
        # pyre-ignore[6]
        initialize=True,
    )
    ds3 = tlx.local_load(
        tlx.local_slice(ds_stage, [0, 96], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds1,
        k10,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds1,
        k11,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    k20 = tlx.extract_slice(k_native, [32, 64], [64, 0])
    k21 = tlx.extract_slice(k_native, [32, 64], [64, 64])
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds2,
        k20,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds2,
        k21,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    k30 = tlx.extract_slice(k_native, [32, 64], [96, 0])
    k31 = tlx.extract_slice(k_native, [32, 64], [96, 64])
    # pyre-ignore[6]
    dq0 = tlx.amd_scheduled_mfma(
        ds3,
        k30,
        dq0,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    # pyre-ignore[6]
    dq1 = tlx.amd_scheduled_mfma(
        ds3,
        k31,
        dq1,
        # pyre-ignore[6]
        resident_operand=1,
        # pyre-ignore[6]
        accumulator_role="transient",
    )
    dq0, dq1, _ = tlx.amd_mfma_commit((dq0, dq1), k_native)
    dq = tl.join(dq0, dq1)
    dq = tl.permute(dq, (0, 2, 1))
    dq = tl.reshape(dq, (BLOCK_M, 128))
    # pyre-ignore[6]
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    # pyre-ignore[6]
    dq_scale = tlx.require_layout(
        tl.broadcast_to(alpha, (BLOCK_M, 128)),
        MMA_MD,
        # pyre-ignore[6]
        pin=False,
    )
    dq = dq * dq_scale

    local_m = tlx.rematerialized_range(0, BLOCK_M, 14, placement=start_m)
    offs_d = tlx.rematerialized_range(0, 128, 13, placement=start_m)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    swizzled = (start_m * 128 + ((local_m[:, None] << 1) | d_swizzled[None, :])).to(tl.int32)
    swizzled = tl.max_contiguous(swizzled, [1, 2])
    # pyre-ignore[6]
    swizzled = tlx.require_layout(swizzled, MMA_MD, pin=False)
    # pyre-ignore[6]
    atomic_mask = tlx.require_layout(mask_m[:, None], MMA_MD, pin=False)
    tlx.buffer_atomic_add(
        DQ_ACC,
        swizzled,
        dq.to(tl.bfloat16),
        mask=atomic_mask,
        sem="relaxed",
        contiguity=2,
    )


@triton.jit
# Triton TR001: this conversion has one fixed 128x128 layout and no tuning axis.
def _tlx_gfx950_hstu_native_dq_convert(  # noqa: TR001
    DQ_ACC,
    DQ,
    seq_offsets,
    stride_dqm,
    stride_dqh,
    PADDED_L,
    H: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tl.static_assert(BLOCK_D_Q == 128)
    tl.static_assert(BLOCK_M == 128)
    # pyre-ignore[9]
    native_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (256, 512, 1024, 8, 2, 2048, 16, 32),
            (1, 128, 4, 64, 4096, 8192),
        ),
    )
    # pyre-ignore[9]
    store_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (256, 512, 1024, 8, 128, 2048, 16, 32),
            (1, 2, 4, 64, 4096, 8192),
        ),
    )
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    pid_m = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    tile_m_base = pid_m * BLOCK_M
    native_m = tile_m_base + tl.arange(0, BLOCK_M)
    native_d = tl.arange(0, BLOCK_D_Q)
    local_m = native_m & 15
    tile_m = native_m - local_m
    d_swizzled = ((native_d & 1)
                  | ((native_d & 2) << 6)
                  | ((native_d & 12) << 3)
                  | ((native_d & 48) << 5)
                  | ((native_d & 64) << 2))
    native_offsets = (tile_m[:, None] * BLOCK_D_Q + (local_m[:, None] << 1) + d_swizzled[None, :]).to(tl.int32)
    native_offsets = tl.max_contiguous(native_offsets, [1, 4])
    # pyre-ignore[6]
    native_offsets = tlx.require_layout(native_offsets, native_layout, pin=False)
    # pyre-ignore[6]
    valid = tlx.require_layout(
        tl.broadcast_to(native_m[:, None] < seq_len, (BLOCK_M, BLOCK_D_Q)),
        native_layout,
        # pyre-ignore[6]
        pin=False,
    )
    scratch_seq_start = seq_start + 15 * off_z
    scratch_base = (off_h.to(tl.int64) * PADDED_L + scratch_seq_start) * BLOCK_D_Q
    values = tlx.buffer_load(
        DQ_ACC + scratch_base,
        native_offsets,
        mask=valid,
        other=0.0,
        contiguity=4,
    )
    # pyre-ignore[6]
    values = tlx.require_layout(values, store_layout, pin=False)

    store_m = tile_m_base + tl.arange(0, BLOCK_M)
    store_d = tl.arange(0, BLOCK_D_Q)
    store_offsets = store_m[:, None] * stride_dqm + store_d[None, :]
    # pyre-ignore[6]
    store_offsets = tlx.require_layout(store_offsets, store_layout, pin=False)
    # pyre-ignore[6]
    store_mask = tlx.require_layout(
        tl.broadcast_to(store_m[:, None] < seq_len, (BLOCK_M, BLOCK_D_Q)),
        store_layout,
        # pyre-ignore[6]
        pin=False,
    )
    DQ = DQ + seq_start * stride_dqm + off_h.to(tl.int64) * stride_dqh
    tlx.buffer_store(values, DQ, store_offsets.to(tl.int32), mask=store_mask)


@triton.jit
def _tlx_gfx950_hstu_fa_dv_fragmented_half(
    lhs,
    rhs,
    dv,
    MMA_ND: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    FIRST_HALF: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    lhs = tlx.require_layout(lhs, P_ND_LAYOUT, pin=False)
    rhs = tlx.require_layout(rhs, Q_OUT_LAYOUT, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    lhs0 = tlx.extract_slice(lhs, [128, 16], [0, 0])
    lhs1 = lhs0
    if BLOCK_N == 256:
        lhs1 = tlx.extract_slice(lhs, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(rhs, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(rhs, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(rhs, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(rhs, [16, 32], [0, 96])
    c0 = tlx.extract_slice(dv, [128, 32], [0, 0])
    c1 = tlx.extract_slice(dv, [128, 32], [0, 32])
    c2 = tlx.extract_slice(dv, [128, 32], [0, 64])
    c3 = tlx.extract_slice(dv, [128, 32], [0, 96])
    c10, c11, c12, c13 = c0, c1, c2, c3
    if BLOCK_N == 256:
        c10 = tlx.extract_slice(dv, [128, 32], [128, 0])
        c11 = tlx.extract_slice(dv, [128, 32], [128, 32])
        c12 = tlx.extract_slice(dv, [128, 32], [128, 64])
        c13 = tlx.extract_slice(dv, [128, 32], [128, 96])
    if FIRST_HALF:
        c0 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs0,
            c0,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c1 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs1,
            c1,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        if BLOCK_N == 256:
            c10 = tlx.amd_scheduled_mfma(
                lhs1,
                rhs0,
                c10,
                accumulator_role="persistent",
                accumulator_register_class="vgpr",
            )
            c11 = tlx.amd_scheduled_mfma(
                lhs1,
                rhs1,
                c11,
                accumulator_role="persistent",
                accumulator_register_class="vgpr",
            )
    else:
        c2 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs2,
            c2,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c3 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs3,
            c3,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        if BLOCK_N == 256:
            c12 = tlx.amd_scheduled_mfma(
                lhs1,
                rhs2,
                c12,
                accumulator_role="persistent",
                accumulator_register_class="vgpr",
            )
            c13 = tlx.amd_scheduled_mfma(
                lhs1,
                rhs3,
                c13,
                accumulator_role="persistent",
                accumulator_register_class="vgpr",
            )
    row0 = tl.cat(tl.cat(c0, c1, dim=1), tl.cat(c2, c3, dim=1), dim=1)
    if BLOCK_N == 256:
        row1 = tl.cat(tl.cat(c10, c11, dim=1), tl.cat(c12, c13, dim=1), dim=1)
        result = tl.cat(row0, row1, dim=0)
    else:
        result = row0
    return tlx.require_layout(result, MMA_ND, pin=False)


@triton.jit
def _tlx_gfx950_hstu_fa_dk_fragmented_prefetch(
    lhs,
    rhs,
    dk,
    previous_dr,
    MMA_ND: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    DR_MD_LAYOUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    lhs = tlx.require_layout(lhs, P_ND_LAYOUT, pin=False)
    rhs = tlx.require_layout(rhs, Q_OUT_LAYOUT, pin=False)
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    lhs0 = tlx.extract_slice(lhs, [128, 16], [0, 0])
    lhs1 = lhs0
    if BLOCK_N == 256:
        lhs1 = tlx.extract_slice(lhs, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(rhs, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(rhs, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(rhs, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(rhs, [16, 32], [0, 96])
    c0 = tlx.extract_slice(dk, [128, 32], [0, 0])
    c1 = tlx.extract_slice(dk, [128, 32], [0, 32])
    c2 = tlx.extract_slice(dk, [128, 32], [0, 64])
    c3 = tlx.extract_slice(dk, [128, 32], [0, 96])
    c10, c11, c12, c13 = c0, c1, c2, c3
    if BLOCK_N == 256:
        c10 = tlx.extract_slice(dk, [128, 32], [128, 0])
        c11 = tlx.extract_slice(dk, [128, 32], [128, 32])
        c12 = tlx.extract_slice(dk, [128, 32], [128, 64])
        c13 = tlx.extract_slice(dk, [128, 32], [128, 96])
    c0 = tlx.amd_scheduled_mfma(lhs0, rhs0, c0, accumulator_role="persistent")
    if BLOCK_N == 256:
        c10 = tlx.amd_scheduled_mfma(lhs1, rhs0, c10, accumulator_role="persistent")
    dr0 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 0], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    c1 = tlx.amd_scheduled_mfma(lhs0, rhs1, c1, accumulator_role="persistent")
    if BLOCK_N == 256:
        c11 = tlx.amd_scheduled_mfma(lhs1, rhs1, c11, accumulator_role="persistent")
    c2 = tlx.amd_scheduled_mfma(lhs0, rhs2, c2, accumulator_role="persistent")
    if BLOCK_N == 256:
        c12 = tlx.amd_scheduled_mfma(lhs1, rhs2, c12, accumulator_role="persistent")
    dr1 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 32], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    c3 = tlx.amd_scheduled_mfma(lhs0, rhs3, c3, accumulator_role="persistent")
    if BLOCK_N == 256:
        c13 = tlx.amd_scheduled_mfma(lhs1, rhs3, c13, accumulator_role="persistent")
    row0 = tl.cat(tl.cat(c0, c1, dim=1), tl.cat(c2, c3, dim=1), dim=1)
    if BLOCK_N == 256:
        row1 = tl.cat(tl.cat(c10, c11, dim=1), tl.cat(c12, c13, dim=1), dim=1)
        result = tl.cat(row0, row1, dim=0)
    else:
        result = row0
    result = tlx.require_layout(result, MMA_ND, pin=False)
    return result, dr0, dr1


@triton.jit
def _tlx_gfx950_hstu_fa_dq_prefetched(
    previous_dr,
    dr0,
    dr1,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_resident,
    MMA_MD: tl.constexpr,
    DR_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dr0 = tlx.require_layout(dr0, DR_MD_LAYOUT, pin=False)
    dr1 = tlx.require_layout(dr1, DR_MD_LAYOUT, pin=False)
    k_resident_lo = tlx.require_layout(k_resident_lo, K_MD_LAYOUT, pin=False)
    v_resident = tlx.require_layout(v_resident, V_LAYOUT, pin=False)
    dq0 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    dq1 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    dr2 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 64], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    k00 = tlx.extract_slice(k_resident_lo, [32, 64], [0, 0])
    k01 = tlx.extract_slice(k_resident_lo, [32, 64], [0, 64])
    k10 = tlx.extract_slice(k_resident_lo, [32, 64], [32, 0])
    k11 = tlx.extract_slice(k_resident_lo, [32, 64], [32, 64])
    dq0 = tlx.amd_scheduled_mfma(
        dr0,
        k00,
        dq0,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    dq1 = tlx.amd_scheduled_mfma(
        dr0,
        k01,
        dq1,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    dr3 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 96], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    dq0 = tlx.amd_scheduled_mfma(dr1, k10, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(dr1, k11, dq1, resident_operand=1, accumulator_role="transient")
    k20 = tlx.extract_slice(k_resident_lo, [32, 64], [64, 0])
    k21 = tlx.extract_slice(k_resident_lo, [32, 64], [64, 64])
    dq0 = tlx.amd_scheduled_mfma(dr2, k20, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(dr2, k21, dq1, resident_operand=1, accumulator_role="transient")
    k30 = tlx.extract_slice(k_resident_lo, [32, 64], [96, 0])
    k31 = tlx.extract_slice(k_resident_lo, [32, 64], [96, 64])
    dq0 = tlx.amd_scheduled_mfma(dr3, k30, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(dr3, k31, dq1, resident_operand=1, accumulator_role="transient")
    if BLOCK_N == 256:
        k_resident_mid = tlx.require_layout(k_resident_mid, K_MD_LAYOUT, pin=False)
        k_resident_band6 = tlx.require_layout(k_resident_band6, K_MD_LAYOUT, pin=False)
        dr4 = tlx.local_load(
            tlx.local_slice(previous_dr, [0, 128], [16, 32]),
            layout=DR_MD_LAYOUT,
            relaxed=True,
        )
        dr5 = tlx.local_load(
            tlx.local_slice(previous_dr, [0, 160], [16, 32]),
            layout=DR_MD_LAYOUT,
            relaxed=True,
        )
        k40 = tlx.extract_slice(k_resident_mid, [32, 64], [0, 0])
        k41 = tlx.extract_slice(k_resident_mid, [32, 64], [0, 64])
        dq0 = tlx.amd_scheduled_mfma(dr4, k40, dq0, resident_operand=1, accumulator_role="transient")
        dq1 = tlx.amd_scheduled_mfma(dr4, k41, dq1, resident_operand=1, accumulator_role="transient")
        dr6 = tlx.local_load(
            tlx.local_slice(previous_dr, [0, 192], [16, 32]),
            layout=DR_MD_LAYOUT,
            relaxed=True,
        )
        k50 = tlx.extract_slice(k_resident_mid, [32, 64], [32, 0])
        k51 = tlx.extract_slice(k_resident_mid, [32, 64], [32, 64])
        dq0 = tlx.amd_scheduled_mfma(dr5, k50, dq0, resident_operand=1, accumulator_role="transient")
        dq1 = tlx.amd_scheduled_mfma(dr5, k51, dq1, resident_operand=1, accumulator_role="transient")
        dr7 = tlx.local_load(
            tlx.local_slice(previous_dr, [0, 224], [16, 32]),
            layout=DR_MD_LAYOUT,
            relaxed=True,
        )
        k60 = tlx.extract_slice(k_resident_band6, [32, 64], [0, 0])
        k61 = tlx.extract_slice(k_resident_band6, [32, 64], [0, 64])
        dq0 = tlx.amd_scheduled_mfma(dr6, k60, dq0, resident_operand=1, accumulator_role="transient")
        dq1 = tlx.amd_scheduled_mfma(dr6, k61, dq1, resident_operand=1, accumulator_role="transient")
        k7 = tlx.local_load(
            tlx.local_slice(k_buffer, [224, 0], [32, 128]),
            layout=K_MD_LAYOUT,
            relaxed=True,
        )
        k70 = tlx.extract_slice(k7, [32, 64], [0, 0])
        k71 = tlx.extract_slice(k7, [32, 64], [0, 64])
        dq0 = tlx.amd_scheduled_mfma(dr7, k70, dq0, resident_operand=1, accumulator_role="transient")
        dq1 = tlx.amd_scheduled_mfma(dr7, k71, dq1, resident_operand=1, accumulator_role="transient")
    dq0, dq1, v_resident = tlx.amd_mfma_commit((dq0, dq1), v_resident)
    dq = tl.join(dq0, dq1)
    dq = tl.permute(dq, (0, 2, 1))
    dq = tl.reshape(dq, (16, 128))
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    return dq, v_resident


@triton.jit
def _tlx_gfx950_hstu_fa_dq(
    previous_dr,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_resident,
    MMA_MD: tl.constexpr,
    DR_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dr0 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 0], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    dr1 = tlx.local_load(
        tlx.local_slice(previous_dr, [0, 32], [16, 32]),
        layout=DR_MD_LAYOUT,
        relaxed=True,
    )
    return _tlx_gfx950_hstu_fa_dq_prefetched(
        previous_dr,
        dr0,
        dr1,
        k_buffer,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_resident,
        MMA_MD,
        DR_MD_LAYOUT,
        K_MD_LAYOUT,
        V_LAYOUT,
        BLOCK_N,
    )


@triton.jit
def _tlx_gfx950_hstu_fa_store_dq_native(
    dq,
    DQ_ACC,
    start_m,
    seq_len,
    alpha,
    MMA_MD: tl.constexpr,
):
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    dq_scale = tlx.require_layout(tl.broadcast_to(alpha, (16, 128)), MMA_MD, pin=False)
    dq = dq * dq_scale
    local_m = tlx.rematerialized_range(0, 16, 14, placement=start_m)
    offs_d = tlx.rematerialized_range(0, 128, 13, placement=start_m)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    swizzled = (start_m * 128 + ((local_m[:, None] << 1) | d_swizzled[None, :])).to(tl.int32)
    swizzled = tl.max_contiguous(swizzled, [1, 2])
    swizzled = tlx.require_layout(swizzled, MMA_MD, pin=False)
    mask = tlx.require_layout(
        tl.broadcast_to(start_m + local_m[:, None] < seq_len, (16, 128)),
        MMA_MD,
        pin=False,
    )
    tlx.buffer_atomic_add(
        DQ_ACC,
        swizzled,
        dq.to(tl.bfloat16),
        mask=mask,
        sem="relaxed",
        contiguity=2,
    )


@triton.jit
def _tlx_gfx950_hstu_fa_front(
    dv,
    q_slice,
    do_slice,
    k_buffer,
    k_operand,
    v_operand,
    dr_stage,
    start_m,
    start_n,
    seq_len,
    history_end,
    alpha,
    MAX_SEQ_LEN: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    IS_MASKLESS: tl.constexpr,
    RESIDENT_K_SCORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    v_operand = tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False)
    if RESIDENT_K_SCORE:
        k_operand = tlx.require_layout(k_operand, K_NM_LAYOUT, pin=False)
    else:
        k_operand = tlx.local_load(k_buffer, layout=K_NM_LAYOUT, relaxed=True)
    q_t = tlx.local_load(tlx.local_trans(q_slice), layout=QT_LAYOUT, relaxed=True)
    # Triton TR011: pin tensor-core behavior across compiler default changes.
    scores = tl.dot(
        k_operand,
        q_t,
        tlx.zeros((BLOCK_N, 16), tl.float32, layout=MMA_NM),
        allow_tf32=True,
    )
    score_registers_per_group: tl.constexpr = 16 if BLOCK_N == 256 else 8
    scores = tlx.amd_register_resident(
        scores,
        register_class="vgpr",
        registers_per_group=score_registers_per_group,
    )
    if IS_MASKLESS:
        do_t = tlx.local_load(tlx.local_trans(do_slice), layout=QT_LAYOUT, relaxed=True)
        # Triton TR011: pin tensor-core behavior across compiler default changes.
        da = tl.dot(
            v_operand,
            do_t,
            tlx.zeros((BLOCK_N, 16), tl.float32, layout=MMA_NM),
            allow_tf32=True,
        )
        da = tlx.amd_register_resident(
            da,
            register_class="vgpr",
            registers_per_group=score_registers_per_group,
        )
        q_out = tlx.local_load(q_slice, layout=Q_OUT_LAYOUT, relaxed=True)
        q_out = tlx.amd_register_resident(q_out, register_class="vgpr", registers_per_group=4)
        do_out = tlx.local_load(do_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    alpha_full = tlx.require_layout(tl.broadcast_to(alpha, (BLOCK_N, 16)), MMA_NM, pin=False)
    r = scores * alpha_full
    one = tlx.require_layout(tl.full((BLOCK_N, 16), 1.0, tl.float32), MMA_NM, pin=False)
    half = tlx.require_layout(tl.full((BLOCK_N, 16), 0.5, tl.float32), MMA_NM, pin=False)
    x_half = r * half
    u = x_half * x_half
    p = tlx.require_layout(tl.full((BLOCK_N, 16), -0.000198527, tl.float32), MMA_NM, pin=False)
    p = p + u * tlx.require_layout(tl.full((BLOCK_N, 16), 0.00972515, tl.float32), MMA_NM, pin=False)
    p = p * u + tlx.require_layout(tl.full((BLOCK_N, 16), -0.0533740, tl.float32), MMA_NM, pin=False)
    p = p * u + tlx.require_layout(tl.full((BLOCK_N, 16), 0.133392, tl.float32), MMA_NM, pin=False)
    p = p * u + one
    tanh_half = x_half * p
    # Triton TR012: tl.clamp rejects the explicit MFMA-layout bound encoding.
    tanh_half = tl.maximum(tanh_half, -one)  # noqa: TR012
    tanh_half = tl.minimum(tanh_half, one)
    sig = (tanh_half + one) * half
    scale = tlx.require_layout(
        tl.full((BLOCK_N, 16), 1.0 / MAX_SEQ_LEN, tl.float32),
        MMA_NM,
        pin=False,
    )
    zero = tlx.zeros((BLOCK_N, 16), tl.float32, layout=MMA_NM)
    scaled_sig = sig * scale
    if IS_MASKLESS:
        a = r * scaled_sig
    else:
        offs_n = start_n + tlx.rematerialized_range(0, BLOCK_N, 32, placement=start_m)
        offs_m = start_m + tlx.rematerialized_range(0, 16, 33, placement=start_m)
        valid = ((offs_n[:, None] < seq_len)
                 & (offs_m[None, :] < seq_len)
                 & ((offs_m[None, :] >= history_end) | (offs_n[:, None] <= offs_m[None, :])))
        valid = tlx.require_layout(valid, MMA_NM, pin=False)
        a = tl.where(valid, r * scaled_sig, zero)

    if not IS_MASKLESS:
        do_t = tlx.local_load(tlx.local_trans(do_slice), layout=QT_LAYOUT, relaxed=True)
        # Triton TR011: pin tensor-core behavior across compiler default changes.
        da = tl.dot(
            v_operand,
            do_t,
            tlx.zeros((BLOCK_N, 16), tl.float32, layout=MMA_NM),
            allow_tf32=True,
        )
        da = tlx.amd_register_resident(
            da,
            register_class="vgpr",
            registers_per_group=score_registers_per_group,
        )
    if not IS_MASKLESS:
        q_out = tlx.local_load(q_slice, layout=Q_OUT_LAYOUT, relaxed=True)
        q_out = tlx.amd_register_resident(q_out, register_class="vgpr", registers_per_group=4)
        do_out = tlx.local_load(do_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    dr = da * (scaled_sig + a * (one - sig))
    if not IS_MASKLESS:
        dr = tl.where(valid, dr, zero)

    a_nd = tl.reshape(a.to(tl.bfloat16), (BLOCK_N // 128, 2, 2, 2, 16, 16))
    a_nd = tl.permute(a_nd, (0, 2, 3, 1, 4, 5))
    a_nd = tl.reshape(a_nd, (BLOCK_N, 16))
    a_nd = tlx.require_layout(a_nd, P_ND_LAYOUT, pin=False)
    dv = _tlx_gfx950_hstu_fa_dv_fragmented_half(
        a_nd,
        do_out,
        dv,
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        True,
        BLOCK_N,
    )

    dr_bf16 = dr.to(tl.bfloat16)
    tlx.local_store(dr_stage, tl.trans(dr_bf16))
    dr_nd = tl.reshape(dr_bf16, (BLOCK_N // 128, 2, 2, 2, 16, 16))
    dr_nd = tl.permute(dr_nd, (0, 2, 3, 1, 4, 5))
    dr_nd = tl.reshape(dr_nd, (BLOCK_N, 16))
    dr_nd = tlx.require_layout(dr_nd, P_ND_LAYOUT, pin=False)
    return dv, dr_nd, q_out, a_nd, do_out


@triton.jit
def _tlx_gfx950_hstu_fa_phase(
    q_tiles,
    do_tiles,
    dr_buffers,
    k_buffer,
    k_operand,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_operand,
    dk,
    dv,
    DQ_ACC,
    start_n,
    seq_len,
    history_end,
    query_tile,
    phase: tl.constexpr,
    q_outer,
    do_outer,
    Q,
    DOut,
    next_outer_block,
    stride_qm,
    stride_dom,
    alpha,
    MAX_SEQ_LEN: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    MMA_MD: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    DR_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    ASYNC_LAYOUT: tl.constexpr,
    PREFETCH_NEXT: tl.constexpr,
    DIRECT_QDO_G2L: tl.constexpr,
    IS_MASKLESS: tl.constexpr,
    RESIDENT_K_SCORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = query_tile * 16 + phase * 16
    stage: tl.constexpr = phase % 2
    q_slice = tlx.local_view(q_tiles, phase)
    do_slice = tlx.local_view(do_tiles, phase)
    dv, dk_lhs, dk_rhs, dv_lhs, dv_rhs = _tlx_gfx950_hstu_fa_front(
        dv,
        q_slice,
        do_slice,
        k_buffer,
        k_operand,
        v_operand,
        tlx.local_view(dr_buffers, stage),
        start_m,
        start_n,
        seq_len,
        history_end,
        alpha,
        MAX_SEQ_LEN,
        MMA_NM,
        MMA_ND,
        K_NM_LAYOUT,
        QT_LAYOUT,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        IS_MASKLESS,
        RESIDENT_K_SCORE,
        BLOCK_N,
    )
    if PREFETCH_NEXT:
        _tlx_gfx950_hstu_fa_issue_qdo_async(
            q_outer,
            do_outer,
            Q,
            DOut,
            next_outer_block,
            seq_len,
            stride_qm,
            stride_dom,
            ASYNC_LAYOUT,
            DIRECT_QDO_G2L,
        )
    dv = _tlx_gfx950_hstu_fa_dv_fragmented_half(
        dv_lhs,
        dv_rhs,
        dv,
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        False,
        BLOCK_N,
    )
    previous_stage: tl.constexpr = 1 - stage
    dk, dr0, dr1 = _tlx_gfx950_hstu_fa_dk_fragmented_prefetch(
        dk_lhs,
        dk_rhs,
        dk,
        tlx.local_view(dr_buffers, previous_stage),
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        DR_MD_LAYOUT,
        BLOCK_N,
    )
    dq, v_operand = _tlx_gfx950_hstu_fa_dq_prefetched(
        tlx.local_view(dr_buffers, previous_stage),
        dr0,
        dr1,
        k_buffer,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_operand,
        MMA_MD,
        DR_MD_LAYOUT,
        K_MD_LAYOUT,
        K_NM_LAYOUT,
        BLOCK_N,
    )
    first_query_tile = start_n // 16
    dq_start_m = tl.maximum(start_m - 16, first_query_tile * 16)
    _tlx_gfx950_hstu_fa_store_dq_native(
        dq,
        DQ_ACC,
        dq_start_m,
        seq_len,
        alpha,
        MMA_MD,
    )
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    v_operand = tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False)
    if PREFETCH_NEXT:
        # Phase 3 publishes both current dR and the next Q/dOut slot.
        tlx.async_load_wait_group(1)
    tl.debug_barrier()
    return dk, dv, v_operand


@triton.jit
def _tlx_gfx950_hstu_fa_issue_qdo_async(
    q_outer,
    do_outer,
    Q,
    DOut,
    outer_block,
    seq_len,
    stride_qm,
    stride_dom,
    ASYNC_LAYOUT: tl.constexpr,
    DIRECT_QDO_G2L: tl.constexpr,
):
    outer_slice = tl.arange(0, 4)
    inner_m = tlx.rematerialized_range(0, 16, 10, placement=outer_block)
    offs_d = tlx.rematerialized_range(0, 128, 11, placement=outer_block)
    offs_m = outer_block * 64 + outer_slice[:, None] * 16 + inner_m[None, :]
    safe_m = tl.minimum(offs_m, seq_len - 1)
    q_offsets = safe_m[:, :, None] * stride_qm + offs_d[None, None, :]
    do_offsets = safe_m[:, :, None] * stride_dom + offs_d[None, None, :]
    q_offsets = tlx.require_layout(q_offsets.to(tl.int32), ASYNC_LAYOUT, pin=False)
    do_offsets = tlx.require_layout(do_offsets.to(tl.int32), ASYNC_LAYOUT, pin=False)
    if DIRECT_QDO_G2L:
        q_token = tlx.buffer_load_to_local(q_outer, Q, q_offsets)
        do_token = tlx.buffer_load_to_local(do_outer, DOut, do_offsets)
    else:
        q_token = tlx.async_load(Q + q_offsets, q_outer)
        do_token = tlx.async_load(DOut + do_offsets, do_outer)
    tlx.async_load_commit_group([q_token, do_token])


@triton.jit
def _tlx_gfx950_hstu_fa_outer_block(
    outer_offset,
    first_outer_block,
    q_buffers,
    do_buffers,
    dr_buffers,
    k_buffer,
    k_operand,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_operand,
    dk,
    dv,
    DQ_ACC,
    start_n,
    seq_len,
    history_end,
    Q,
    DOut,
    stride_qm,
    stride_dom,
    alpha,
    MAX_SEQ_LEN: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    MMA_MD: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    DR_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    QDO_ASYNC_LAYOUT: tl.constexpr,
    QDO_SLICE_SMEM_LAYOUT: tl.constexpr,
    DIRECT_QDO_G2L: tl.constexpr,
    IS_MASKLESS: tl.constexpr,
    RESIDENT_K_SCORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    outer_stage = outer_offset % 2
    outer_block = first_outer_block + outer_offset
    q_outer = tlx.local_view(q_buffers, outer_stage)
    do_outer = tlx.local_view(do_buffers, outer_stage)
    q_tiles = tlx.local_reinterpret(
        q_outer,
        tl.bfloat16,
        [4, 16, 128],
        layout=QDO_SLICE_SMEM_LAYOUT,
    )
    do_tiles = tlx.local_reinterpret(
        do_outer,
        tl.bfloat16,
        [4, 16, 128],
        layout=QDO_SLICE_SMEM_LAYOUT,
    )
    for phase in tl.static_range(0, 3):
        dk, dv, v_operand = _tlx_gfx950_hstu_fa_phase(
            q_tiles,
            do_tiles,
            dr_buffers,
            k_buffer,
            k_operand,
            k_resident_lo,
            k_resident_mid,
            k_resident_band6,
            v_operand,
            dk,
            dv,
            DQ_ACC,
            start_n,
            seq_len,
            history_end,
            outer_block * 4,
            phase,
            q_outer,
            do_outer,
            Q,
            DOut,
            outer_block + 2,
            stride_qm,
            stride_dom,
            alpha,
            MAX_SEQ_LEN,
            MMA_NM,
            MMA_ND,
            MMA_MD,
            K_NM_LAYOUT,
            QT_LAYOUT,
            P_ND_LAYOUT,
            Q_OUT_LAYOUT,
            DR_MD_LAYOUT,
            K_MD_LAYOUT,
            QDO_ASYNC_LAYOUT,
            False,
            DIRECT_QDO_G2L,
            IS_MASKLESS,
            RESIDENT_K_SCORE,
            BLOCK_N,
        )
    dk, dv, v_operand = _tlx_gfx950_hstu_fa_phase(
        q_tiles,
        do_tiles,
        dr_buffers,
        k_buffer,
        k_operand,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_operand,
        dk,
        dv,
        DQ_ACC,
        start_n,
        seq_len,
        history_end,
        outer_block * 4,
        3,
        q_outer,
        do_outer,
        Q,
        DOut,
        outer_block + 2,
        stride_qm,
        stride_dom,
        alpha,
        MAX_SEQ_LEN,
        MMA_NM,
        MMA_ND,
        MMA_MD,
        K_NM_LAYOUT,
        QT_LAYOUT,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        DR_MD_LAYOUT,
        K_MD_LAYOUT,
        QDO_ASYNC_LAYOUT,
        True,
        DIRECT_QDO_G2L,
        IS_MASKLESS,
        RESIDENT_K_SCORE,
        BLOCK_N,
    )
    return dk, dv, v_operand


@triton.jit
# Triton TR001: callers select the separately benchmarked BLOCK_N variant.
def _tlx_gfx950_hstu_fa_schedule_bwd_kernel(  # noqa: TR001
    Q,
    K,
    V,
    DOut,
    DQ_ACC,
    DK,
    DV,
    seq_offsets,
    num_targets,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    PADDED_L,
    H: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIRECT_QDO_G2L: tl.constexpr,
    MASK_PEEL: tl.constexpr,
    RESIDENT_K_SCORE: tl.constexpr,
):
    BLOCK_M: tl.constexpr = 16
    OUTER_M: tl.constexpr = 64
    D: tl.constexpr = 128
    off_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_z = tl.program_id(2)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    start_n = pid_n * BLOCK_N
    if start_n >= seq_len:
        return
    n_targets = tl.load(num_targets + off_z).to(tl.int32)
    history_end = seq_len - n_targets

    Q = Q + seq_start * stride_qm + off_h.to(tl.int64) * stride_qh
    K = K + seq_start * stride_kn + off_h.to(tl.int64) * stride_kh
    V = V + seq_start * stride_vn + off_h.to(tl.int64) * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h.to(tl.int64) * stride_doh
    DK = DK + seq_start * stride_dkn + off_h.to(tl.int64) * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h.to(tl.int64) * stride_dvh
    scratch_seq_start = seq_start + 15 * off_z
    DQ_ACC = DQ_ACC + (off_h.to(tl.int64) * PADDED_L + scratch_seq_start) * D

    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    k_nm_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_nd_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_out_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    dr_md_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_md_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    qdo_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 128, 64, 512, 256, 1024), (1, 2, 4, 2048, 4096)),
    )
    qdo_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 4],
            [0, 0, 8],
            [0, 0, 16],
            [0, 0, 32],
            [0, 1, 0],
            [0, 0, 64],
            [0, 4, 0],
            [0, 2, 0],
            [0, 8, 0],
            [1, 0, 0],
            [2, 0, 0],
        ],
        [4, BLOCK_M, D],
    )
    qdo_slice_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [0, 64],
            [4, 0],
            [2, 0],
            [8, 0],
        ],
        [BLOCK_M, D],
    )
    if BLOCK_N == 256:
        dr_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)],
            [
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [4, 0],
                [0, 8],
                [8, 0],
                [0, 32],
                [0, 16],
                [0, 4],
                [0, 64],
                [0, 128],
            ],
            [BLOCK_M, BLOCK_N],
        )
        k_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 64],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 64],
                [0, 16],
                [0, 32],
                [16, 0],
                [32, 0],
                [64, 0],
                [128, 0],
            ],
            block_bases=[],
            alignment=16,
        )
        k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 4],
                [0, 1, 0],
                [0, 2, 0],
                [0, 4, 0],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [16, 0, 0],
                [32, 0, 0],
                [64, 0, 0],
                [128, 0, 0],
            ],
            block_bases=[],
            alignment=16,
        )
        k_raw_async_layout: tl.constexpr = tlx.layout(
            shape=((64, 4), (8, 8, 2)),
            stride=((8, 512), (1, 2048, 16384)),
        )
        kv_native_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
            stride=(
                (128, 256, 512, 1024, 8192, 4, 2048, 4096),
                (1, 2, 8, 16, 32, 64, 16384),
            ),
        )
    else:
        dr_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)],
            [
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [4, 0],
                [0, 8],
                [8, 0],
                [0, 32],
                [0, 16],
                [0, 4],
                [0, 64],
            ],
            [BLOCK_M, BLOCK_N],
        )
        k_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 64],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 64],
                [0, 16],
                [0, 32],
                [16, 0],
                [32, 0],
                [64, 0],
            ],
            block_bases=[],
            alignment=16,
        )
        k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 4],
                [0, 1, 0],
                [0, 2, 0],
                [0, 4, 0],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [16, 0, 0],
                [32, 0, 0],
                [64, 0, 0],
            ],
            block_bases=[],
            alignment=16,
        )
        k_raw_async_layout: tl.constexpr = tlx.layout(
            shape=((64, 4), (8, 8)),
            stride=((8, 512), (1, 2048)),
        )
        kv_native_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
            stride=(
                (128, 256, 512, 1024, 8192, 4, 2048, 4096),
                (1, 2, 8, 16, 32, 64),
            ),
        )

    k_raw_buffer = tlx.local_alloc((BLOCK_N, D // 8, 8), tl.bfloat16, 1, layout=k_raw_smem_layout)
    k_buffer = tlx.local_reinterpret(
        tlx.local_view(k_raw_buffer, 0),
        tl.bfloat16,
        [BLOCK_N, D],
        layout=k_smem_layout,
    )
    q_buffers = tlx.local_alloc((4, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((4, BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    dr_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 2, layout=dr_smem_layout)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    raw_n = tl.arange(0, BLOCK_N)
    raw_dg = tl.arange(0, D // 8)
    raw_v = tl.arange(0, 8)
    k_phys = raw_n[:, None, None] * D + raw_dg[None, :, None] * 8
    k_d_base = ((k_phys & 0x8) | (((k_phys >> 9) & 0x3) << 4) | ((((k_phys >> 4) ^ (k_phys >> 8)) & 0x1) << 6))
    k_n = (((k_phys >> 5) & 0x7) | (((k_phys >> 8) & 0x1) << 3) | (((k_phys >> 11) & 0xF) << 4))
    safe_k_n = tl.minimum(start_n + k_n, seq_len - 1)
    k_offsets = safe_k_n * stride_kn + k_d_base + raw_v[None, None, :]
    k_offsets = tl.multiple_of(k_offsets, [1, 1, 8])
    k_offsets = tl.max_contiguous(k_offsets, [1, 1, 8])
    k_offsets = tlx.require_layout(k_offsets.to(tl.int32), k_raw_async_layout, pin=False)
    k_token = tlx.buffer_load_to_local(tlx.local_view(k_raw_buffer, 0), K, k_offsets)
    tlx.async_load_commit_group([k_token])

    first_outer_block = start_n // OUTER_M
    _tlx_gfx950_hstu_fa_issue_qdo_async(
        tlx.local_view(q_buffers, 0),
        tlx.local_view(do_buffers, 0),
        Q,
        DOut,
        first_outer_block,
        seq_len,
        stride_qm,
        stride_dom,
        qdo_async_layout,
        DIRECT_QDO_G2L,
    )
    _tlx_gfx950_hstu_fa_issue_qdo_async(
        tlx.local_view(q_buffers, 1),
        tlx.local_view(do_buffers, 1),
        Q,
        DOut,
        first_outer_block + 1,
        seq_len,
        stride_qm,
        stride_dom,
        qdo_async_layout,
        DIRECT_QDO_G2L,
    )
    initial_wait = tlx.async_load_wait_group(1)
    tl.debug_barrier()

    safe_n = tl.minimum(offs_n, seq_len - 1)
    v_offsets = safe_n[:, None] * stride_vn + offs_d[None, :]
    v_offsets = tlx.require_layout(v_offsets.to(tl.int32), k_nm_layout, pin=False)
    v_operand = tlx.buffer_load(V, v_offsets)
    v_operand = tlx.require_layout(v_operand, k_nm_layout, pin=False)
    if RESIDENT_K_SCORE:
        k_operand = tlx.local_load(
            k_buffer,
            token=initial_wait,
            layout=k_nm_layout,
            relaxed=True,
        )
    else:
        k_operand = v_operand
    k_resident_lo = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 0], [128, D]),
        token=initial_wait,
        layout=k_md_layout,
        relaxed=True,
    )
    if BLOCK_N == 256:
        k_resident_mid = tlx.local_load(
            tlx.local_slice(k_buffer, [128, 0], [64, D]),
            token=initial_wait,
            layout=k_md_layout,
            relaxed=True,
        )
        k_resident_band6 = tlx.local_load(
            tlx.local_slice(k_buffer, [192, 0], [32, D]),
            token=initial_wait,
            layout=k_md_layout,
            relaxed=True,
        )
    else:
        k_resident_mid = k_resident_lo
        k_resident_band6 = k_resident_lo
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    tlx.local_store(
        tlx.local_view(dr_buffers, 1),
        tl.zeros((BLOCK_M, BLOCK_N), tl.bfloat16),
    )
    tl.debug_barrier()

    n_outer_blocks = tl.cdiv(seq_len, OUTER_M)
    active_outer_blocks = n_outer_blocks - first_outer_block
    if MASK_PEEL:
        maskless_row = tl.minimum(history_end, start_n + BLOCK_N - 1)
        maskless_begin_block = tl.maximum(first_outer_block, tl.cdiv(maskless_row, OUTER_M))
        maskless_end_block = tl.maximum(maskless_begin_block, seq_len // OUTER_M)
        full_k_tile = start_n + BLOCK_N <= seq_len
        maskless_begin = tl.where(
            full_k_tile,
            tl.minimum(maskless_begin_block - first_outer_block, active_outer_blocks),
            active_outer_blocks,
        )
        maskless_end = tl.where(
            full_k_tile,
            tl.minimum(maskless_end_block - first_outer_block, active_outer_blocks),
            active_outer_blocks,
        )
        for outer_offset in tl.range(0, maskless_begin, loop_unroll_factor=1):
            dk, dv, v_operand = _tlx_gfx950_hstu_fa_outer_block(
                outer_offset,
                first_outer_block,
                q_buffers,
                do_buffers,
                dr_buffers,
                k_buffer,
                k_operand,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                dk,
                dv,
                DQ_ACC,
                start_n,
                seq_len,
                history_end,
                Q,
                DOut,
                stride_qm,
                stride_dom,
                alpha,
                MAX_SEQ_LEN,
                mma_nm,
                mma_nd,
                mma_md,
                k_nm_layout,
                qt_layout,
                p_nd_layout,
                q_out_layout,
                dr_md_layout,
                k_md_layout,
                qdo_async_layout,
                qdo_slice_smem_layout,
                DIRECT_QDO_G2L,
                False,
                RESIDENT_K_SCORE,
                BLOCK_N,
            )
        for outer_offset in tl.range(maskless_begin, maskless_end, loop_unroll_factor=1):
            dk, dv, v_operand = _tlx_gfx950_hstu_fa_outer_block(
                outer_offset,
                first_outer_block,
                q_buffers,
                do_buffers,
                dr_buffers,
                k_buffer,
                k_operand,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                dk,
                dv,
                DQ_ACC,
                start_n,
                seq_len,
                history_end,
                Q,
                DOut,
                stride_qm,
                stride_dom,
                alpha,
                MAX_SEQ_LEN,
                mma_nm,
                mma_nd,
                mma_md,
                k_nm_layout,
                qt_layout,
                p_nd_layout,
                q_out_layout,
                dr_md_layout,
                k_md_layout,
                qdo_async_layout,
                qdo_slice_smem_layout,
                DIRECT_QDO_G2L,
                True,
                RESIDENT_K_SCORE,
                BLOCK_N,
            )
        for outer_offset in tl.range(maskless_end, active_outer_blocks, loop_unroll_factor=1):
            dk, dv, v_operand = _tlx_gfx950_hstu_fa_outer_block(
                outer_offset,
                first_outer_block,
                q_buffers,
                do_buffers,
                dr_buffers,
                k_buffer,
                k_operand,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                dk,
                dv,
                DQ_ACC,
                start_n,
                seq_len,
                history_end,
                Q,
                DOut,
                stride_qm,
                stride_dom,
                alpha,
                MAX_SEQ_LEN,
                mma_nm,
                mma_nd,
                mma_md,
                k_nm_layout,
                qt_layout,
                p_nd_layout,
                q_out_layout,
                dr_md_layout,
                k_md_layout,
                qdo_async_layout,
                qdo_slice_smem_layout,
                DIRECT_QDO_G2L,
                False,
                RESIDENT_K_SCORE,
                BLOCK_N,
            )
    else:
        for outer_offset in tl.range(0, active_outer_blocks, loop_unroll_factor=1):
            dk, dv, v_operand = _tlx_gfx950_hstu_fa_outer_block(
                outer_offset,
                first_outer_block,
                q_buffers,
                do_buffers,
                dr_buffers,
                k_buffer,
                k_operand,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                dk,
                dv,
                DQ_ACC,
                start_n,
                seq_len,
                history_end,
                Q,
                DOut,
                stride_qm,
                stride_dom,
                alpha,
                MAX_SEQ_LEN,
                mma_nm,
                mma_nd,
                mma_md,
                k_nm_layout,
                qt_layout,
                p_nd_layout,
                q_out_layout,
                dr_md_layout,
                k_md_layout,
                qdo_async_layout,
                qdo_slice_smem_layout,
                DIRECT_QDO_G2L,
                False,
                RESIDENT_K_SCORE,
                BLOCK_N,
            )

    dq, v_operand = _tlx_gfx950_hstu_fa_dq(
        tlx.local_view(dr_buffers, 1),
        k_buffer,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_operand,
        mma_md,
        dr_md_layout,
        k_md_layout,
        k_nm_layout,
        BLOCK_N,
    )
    last_start_m = (n_outer_blocks * 4 - 1) * BLOCK_M
    _tlx_gfx950_hstu_fa_store_dq_native(
        dq,
        DQ_ACC,
        last_start_m,
        seq_len,
        alpha,
        mma_md,
    )
    tlx.async_load_wait_group(0)

    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    dk = tl.reshape(dk, (BLOCK_N // 128, 2, 2, 2, 16, D))
    dk = tl.permute(dk, (0, 3, 1, 2, 4, 5))
    dk = tl.reshape(dk, (BLOCK_N, D))
    dk = tlx.require_layout(dk, kv_native_layout, pin=False)
    dv = tl.reshape(dv, (BLOCK_N // 128, 2, 2, 2, 16, D))
    dv = tl.permute(dv, (0, 3, 1, 2, 4, 5))
    dv = tl.reshape(dv, (BLOCK_N, D))
    dv = tlx.require_layout(dv, kv_native_layout, pin=False)
    dk_store_offsets = offs_n[:, None] * stride_dkn + offs_d[None, :]
    dv_store_offsets = offs_n[:, None] * stride_dvn + offs_d[None, :]
    dk_store_offsets = tlx.require_layout(dk_store_offsets.to(tl.int32), kv_native_layout, pin=False)
    dv_store_offsets = tlx.require_layout(dv_store_offsets.to(tl.int32), kv_native_layout, pin=False)
    store_mask = tlx.require_layout(
        tl.broadcast_to(offs_n[:, None] < seq_len, (BLOCK_N, D)),
        kv_native_layout,
        pin=False,
    )
    tlx.buffer_store((dk * alpha).to(tl.bfloat16), DK, dk_store_offsets, mask=store_mask)
    tlx.buffer_store(dv.to(tl.bfloat16), DV, dv_store_offsets, mask=store_mask)


@triton.jit
def _tlx_gfx950_ragged_hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs,
    dq_ptrs_trans,
    dq_ptrs,
    DQ_ACC,
    mask_n,
    ts_0_ptrs,
    ts_1,
    bias_ptrs_trans,
    dbias_ptrs_trans,
    do_ptrs,
    lds_q,
    lds_do,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    n_targets,
    max_ids,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    TW,
    PW,
    DTW,
    DPW,
    M_block,
    Delta_block,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    attn_scale,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    off_h,
    num_softmax_heads: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NATIVE_MFMA_DQ_WARPS: tl.constexpr,
    MMA_MD_4W: tl.constexpr,
    DS_MD_LAYOUT_4W: tl.constexpr,
    K_MD_LAYOUT_4W: tl.constexpr,
    MMA_MD_ATOMIC_4W: tl.constexpr,
    DS_MD_LAYOUT_ATOMIC_4W: tl.constexpr,
    K_MD_LAYOUT_ATOMIC_4W: tl.constexpr,
    MMA_MD_8W: tl.constexpr,
    DS_MD_LAYOUT_8W: tl.constexpr,
    K_MD_LAYOUT_8W: tl.constexpr,
    ATOMIC_DQ: tl.constexpr,
    IS_MASKLESS: tl.constexpr = False,
    QDO_PRELOADED: tl.constexpr = False,
    q_view=None,
    do_view=None,
    qdo_wait=None,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    if IS_MASKLESS:
        invalid_mask_trans = tl.full((BLOCK_N, BLOCK_M), True, tl.int1)
    else:
        invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    if ATTN_SCALE_TYPE == "none":
        scale = 1.0 / MAX_SEQ_LEN
    elif ATTN_SCALE_TYPE == "scalar":
        scale = tl.load(attn_scale).to(tl.float32)
    else:
        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
        scale = tl.load(attn_scale + start_m + offs_m, mask=mask_m).to(tl.float32)

    # q and dOut are each consumed in both orientations. Stage them in LDS once
    # and take the second orientation as a tlx.local_trans view, which is
    # metadata-only: gfx950 has no in-thread transpose, so tl.trans() on a
    # register tile round-trips through LDS anyway and holds registers while it
    # does. Loading q row-major also makes its global read contiguous; the old
    # q_ptrs_trans read was strided along the fastest-varying axis.
    if QDO_PRELOADED:
        q_trans = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait)
    else:
        q_view = tlx.local_view(lds_q, 0)
        do_view = tlx.local_view(lds_do, 0)
        tok_q = tlx.async_load(q_ptrs + start_m * stride_qm, q_view, mask=mask_m[:, None])
        tok_do = tlx.async_load(do_ptrs + start_m * stride_dom, do_view, mask=mask_m[:, None])
        tlx.async_load_commit_group([tok_q, tok_do])
        # pyrefly: ignore [bad-argument-type]
        tlx.async_load_wait_group(0)
        q_trans = tlx.local_load(tlx.local_trans(q_view))

    # dQ is a read/modify/write tile owned by the current K/V block. Start its
    # global read before the qk/dV/dP/dK dot chain so that memory latency is
    # hidden behind those MFMAs. Keeping this bf16 tile live costs little next
    # to the fp32 dK/dV accumulators and avoids a late vmcnt bubble immediately
    # before the dQ dot.
    if NATIVE_MFMA_DQ_WARPS == 0 and not ATOMIC_DQ:
        dq_trans = tl.load(
            dq_ptrs_trans + start_m * stride_dqm,
            mask=mask_m[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
    elif NATIVE_MFMA_DQ_WARPS == 8:
        dq_addresses = dq_ptrs + start_m * stride_dqm
        dq_mask = mask_m[:, None]
        dq_old = tl.load(
            dq_addresses,
            mask=dq_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    pos_offs_n_minus_m = pos_offs_n[:, None] - pos_offs_m[None, :]
    if ATTN_BIAS_TYPE == "fused":
        attn_bias_trans = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + start_m + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs + start_m, mask=mask_m)
            ts_trans = ts_0[None, :] - ts_1[:, None]
            ts_trans = ts_trans + time_delta
            ts_trans = tl.where(ts_trans > 1e-6, ts_trans, 1e-6)
            ts_trans = ts_trans * (1.0 / time_bucket_incr)
            if BUCKET_FN == "log":
                ts_trans = tl.log(ts_trans)
            elif BUCKET_FN == "sqrt":
                ts_trans = tl.sqrt(ts_trans)
            ts_trans = ts_trans * (1.0 / time_bucket_div)
            ts_trans = ts_trans.to(tl.int32)
            ts_trans = tl.where(ts_trans > 0, ts_trans, 0)
            ts_trans = tl.where(ts_trans < num_buckets, ts_trans, num_buckets)
            ts_w_trans = tl.load(
                TW + ts_trans,
                mask=mask_m[None, :] & mask_n[:, None],
            )
            attn_bias_trans = attn_bias_trans + ts_w_trans
        if USE_POS_BIAS:
            offs_pos_w_trans = None
            if HAS_MAX_POS_IND:
                offs_pos_w_trans = pos_offs_n_minus_m + max_pos_ind - 1
                offs_pos_w_trans = tl.where(offs_pos_w_trans > 0, offs_pos_w_trans, 0)
                offs_pos_w_trans = tl.where(
                    offs_pos_w_trans < 2 * max_pos_ind - 2,
                    offs_pos_w_trans,
                    2 * max_pos_ind - 2,
                )
            else:
                offs_pos_w_trans = pos_offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w_trans = tl.load(
                PW + offs_pos_w_trans,
                mask=mask_m[None, :] & mask_n[:, None],
            )
            attn_bias_trans = attn_bias_trans + pos_w_trans
        qk_trans = qk_trans + attn_bias_trans
    elif ATTN_BIAS_TYPE == "separate":
        attn_bias_trans = tl.load(
            bias_ptrs_trans + start_m * seq_len,
            mask=mask_m[None, :] & mask_n[:, None],
            other=0.0,
        )
        qk_trans = qk_trans + attn_bias_trans

    # Delay the dOut orientations until qT has fed qk. This shortens the
    # simultaneous register lifetime of the four LDS-backed tiles while the
    # DS reads can overlap the mask/activation scalar work below.
    if QDO_PRELOADED:
        do = tlx.local_load(do_view, token=qdo_wait)
        do_trans = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait)
    else:
        do = tlx.local_load(do_view)
        do_trans = tlx.local_load(tlx.local_trans(do_view))

    if not IS_MASKLESS:
        global_invalid_mask_trans = tl.full((BLOCK_N, BLOCK_M), True, tl.int1)
        if HAS_MAX_ATTN_LEN:
            if HAS_FULL_ATTN_SIZE:
                if HAS_MULTIPLE_TARGETS:
                    if INVALID_MASK_TYPE == "lower_triangular":
                        global_invalid_mask_trans = (pos_offs_n_minus_m < 0) & (pos_offs_m[None, :]
                                                                                >= max_ids - full_attn_size)
                    elif INVALID_MASK_TYPE == "upper_triangular":
                        global_invalid_mask_trans = (pos_offs_n_minus_m > 0) & (pos_offs_m[None, :]
                                                                                < n_targets + full_attn_size)
                else:
                    if INVALID_MASK_TYPE == "lower_triangular":
                        global_invalid_mask_trans = (pos_offs_n_minus_m < 0) & (pos_offs_m[None, :]
                                                                                >= max_ids - full_attn_size)
                    elif INVALID_MASK_TYPE == "upper_triangular":
                        global_invalid_mask_trans = (pos_offs_n_minus_m > 0) & (pos_offs_m[None, :] < full_attn_size)
                global_invalid_mask_trans = (invalid_mask_trans | global_invalid_mask_trans)

            if INVALID_MASK_TYPE == "lower_triangular":
                invalid_mask_trans = invalid_mask_trans | ((pos_offs_n_minus_m < 0) &
                                                           (pos_offs_n_minus_m >= -max_attn_len))
            elif INVALID_MASK_TYPE == "none":
                invalid_mask_trans = invalid_mask_trans | ((pos_offs_n_minus_m <= max_attn_len)
                                                           & (pos_offs_n_minus_m >= -max_attn_len)
                                                           & (pos_offs_n[:, None] < max_ids))
            if HAS_FULL_ATTN_SIZE:
                # pyre-fixme[61]: Local variable `global_invalid_mask_trans` is undefined, or not always defined.
                invalid_mask_trans = invalid_mask_trans | global_invalid_mask_trans
        else:
            if INVALID_MASK_TYPE == "lower_triangular":
                invalid_mask_trans = invalid_mask_trans | (pos_offs_m[None, :] > pos_offs_n[:, None])
            elif INVALID_MASK_TYPE == "none":
                invalid_mask_trans = invalid_mask_trans | (pos_offs_n[:, None] < max_ids)
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask_trans = invalid_mask_trans | ((pos_offs_m[None, :] == 0) & (pos_offs_n[:, None] < max_ids))
            invalid_mask_trans = invalid_mask_trans | (pos_offs_n[:, None] == 0)

    is_softmax_head = off_h < num_softmax_heads
    if is_softmax_head:
        qk_trans, act_qk_trans, pT = backward_softmax_activation(qk_trans, alpha, invalid_mask_trans, M_block, k)
        # compute dv
        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)

        # compute dk and dq
        dact_qk_trans = tl.dot(v, do_trans, allow_tf32=ALLOW_TF32)
        if QDO_PRELOADED:
            q_tile = tlx.local_load(q_view, token=qdo_wait)
        else:
            q_tile = tlx.local_load(q_view)
        dqk_trans = backward_d_softmax_activation(dact_qk_trans, pT, Delta_block)
        dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
        dqk_trans = dqk_trans.to(k.dtype)
    else:
        # pyrefly: ignore [bad-argument-type]
        sig_trans = fast_silu(qk_trans, False)
        silu_trans = qk_trans * sig_trans * scale
        silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
        silu_trans = silu_trans.to(k.dtype)

        # compute dv
        dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)

        # compute dk and dq
        dqk_trans = tl.dot(v, do_trans, allow_tf32=ALLOW_TF32)
        if QDO_PRELOADED:
            q_tile = tlx.local_load(q_view, token=qdo_wait)
        else:
            q_tile = tlx.local_load(q_view)
        dqk_trans = (dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * scale[None, :])
        dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
        dqk_trans = dqk_trans.to(k.dtype)

    if ATTN_BIAS_TYPE == "fused" and FUSED_BIAS_BWD:
        if USE_TIME_BIAS:
            tl.atomic_add(
                # pyre-ignore[61]
                DTW + ts_trans,
                dqk_trans,
                mask=mask_m[None, :] & mask_n[:, None] & invalid_mask_trans,
                sem="relaxed",
            )
        if USE_POS_BIAS:
            tl.atomic_add(
                # pyre-ignore[61]
                DPW + offs_pos_w_trans,
                dqk_trans,
                mask=mask_m[None, :] & mask_n[:, None] & invalid_mask_trans,
                sem="relaxed",
            )
    elif ATTN_BIAS_TYPE == "separate":
        tl.store(
            dbias_ptrs_trans + start_m * seq_len,
            dqk_trans,
            mask=mask_m[None, :] & mask_n[:, None],
        )
    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    if NATIVE_MFMA_DQ_WARPS == 4 and ATOMIC_DQ:
        # q_tile is already in registers, so its LDS slot can stage dS while
        # the independent dK dot executes before the native dQ fragment reads.
        # pyre-ignore[6]
        tlx.local_store(q_view, tl.trans(dqk_trans))
    dk += tl.dot(dqk_trans, q_tile, allow_tf32=ALLOW_TF32)
    if NATIVE_MFMA_DQ_WARPS == 8:
        _tlx_gfx950_hstu_native_dq_8wave(
            dqk_trans,
            k,
            # pyre-ignore[61]
            dq_old,
            # pyre-ignore[61]
            dq_addresses,
            # pyre-ignore[61]
            dq_mask,
            q_view,
            alpha,
            BLOCK_M,
            MMA_MD_8W,
            DS_MD_LAYOUT_8W,
            K_MD_LAYOUT_8W,
        )
    elif NATIVE_MFMA_DQ_WARPS == 4:
        if ATOMIC_DQ:
            _tlx_gfx950_hstu_native_dq_atomic_4wave(
                q_view,
                k,
                DQ_ACC,
                start_m,
                mask_m,
                alpha,
                BLOCK_M,
                BLOCK_N,
                MMA_MD_ATOMIC_4W,
                DS_MD_LAYOUT_ATOMIC_4W,
                K_MD_LAYOUT_ATOMIC_4W,
            )
        else:
            _tlx_gfx950_hstu_native_dq_4wave(
                dqk_trans,
                k,
                dq_ptrs + start_m * stride_dqm,
                mask_m[:, None],
                alpha,
                BLOCK_M,
                MMA_MD_4W,
                DS_MD_LAYOUT_4W,
                K_MD_LAYOUT_4W,
            )
    else:
        if ATOMIC_DQ:
            dq_update = tl.dot(tl.trans(dqk_trans), k, allow_tf32=ALLOW_TF32) * alpha
            tl.atomic_add(
                dq_ptrs + start_m * stride_dqm,
                dq_update.to(k.dtype),
                mask=mask_m[:, None],
                sem="relaxed",
            )
        else:
            dq_update = tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
            # pyre-ignore[61]
            dq_trans += dq_update
            dq_trans = dq_trans.to(k.dtype)
            tl.store(
                dq_ptrs_trans + start_m * stride_dqm,
                dq_trans,
                mask=mask_m[None, :],
                eviction_policy="evict_last",
            )
    return dk, dv


@triton.jit
def _tlx_gfx950_ragged_hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    Q,
    K,
    V,
    TS,
    TW,
    PW,
    Bias,
    DOut,
    DQ,
    DQ_ACC,
    DK,
    DV,
    DBias,
    DTW,
    DPW,
    M,
    Delta,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_mm,
    alpha,
    attn_scale,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    off_h,
    num_softmax_heads: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    NATIVE_MFMA_DQ_WARPS: tl.constexpr,
    ATOMIC_DQ: tl.constexpr,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    has_contextual_columns = 0
    if INVALID_MASK_TYPE == "lower_triangular":
        if HAS_MULTIPLE_TARGETS:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                if HAS_FULL_ATTN_SIZE:
                    high = seq_len
                else:
                    high = start_n + max_attn_len + BLOCK_N
                high = high if high + n_targets < seq_len else seq_len
            else:
                high = seq_len
        else:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                if HAS_FULL_ATTN_SIZE:
                    high = seq_len
                else:
                    high = start_n + max_attn_len + BLOCK_N
                high = high if high < seq_len else seq_len
            else:
                high = seq_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
            if start_n < contextual_block_end:
                high = seq_len
                has_contextual_columns = 1
    elif INVALID_MASK_TYPE == "none":
        low = 0
        high = seq_len
        if HAS_MAX_ATTN_LEN:
            low = start_n - max_attn_len
            low = low if low > 0 else 0
            high = start_n + BLOCK_N + max_attn_len
            high = high if high < seq_len else seq_len
    else:
        low = 0
        high = start_n + BLOCK_N

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_qk_d[None, :])
    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_qk_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    mask_n = offs_n < seq_len
    # LDS staging for q / dOut. The base vectors are copied from the TLX
    # tutorial third_party/tlx/tutorials/amd_fa_bwd.py; they must match the tile
    # shape exactly (a row bit past the tile makes the memdesc reinterpret
    # verifier reject the layout), so this only covers BLOCK_M in {16, 32, 64}
    # at head dim 128. _get_bw_configs is restricted to match.
    tl.static_assert(BLOCK_M == 16 or BLOCK_M == 32 or BLOCK_M == 64)
    tl.static_assert(BLOCK_D_Q == 128 and BLOCK_D_V == 128)
    if BLOCK_M == 16:
        # pyrefly: ignore [bad-assignment]
        shared_offset_bases: tl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [8, 0],
            [4, 0],
            [1, 0],
            [2, 0],
        ]
    elif BLOCK_M == 32:
        # pyrefly: ignore [bad-assignment]
        shared_offset_bases: tl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [8, 0],
            [1, 0],
            [2, 0],
            [4, 0],
        ]
    else:
        # pyrefly: ignore [bad-assignment]
        shared_offset_bases: tl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ]
    # pyrefly: ignore [missing-attribute]
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 32)], shared_offset_bases,
                                                                               [BLOCK_M, BLOCK_D_Q])
    tl.static_assert(NATIVE_MFMA_DQ_WARPS == 0 or NATIVE_MFMA_DQ_WARPS == 4 or NATIVE_MFMA_DQ_WARPS == 8)
    # pyre-ignore[9]
    mma_md_4w: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    # pyre-ignore[9]
    ds_md_layout_4w: tl.constexpr = tlx.dot_operand_layout(0, mma_md_4w, k_width=8)
    # pyre-ignore[9]
    k_md_layout_4w: tl.constexpr = tlx.dot_operand_layout(1, mma_md_4w, k_width=8)
    # pyre-ignore[9]
    mma_md_atomic_4w: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    # pyre-ignore[9]
    ds_md_layout_atomic_4w: tl.constexpr = tlx.dot_operand_layout(0, mma_md_atomic_4w, k_width=8)
    # pyre-ignore[9]
    k_md_layout_atomic_4w: tl.constexpr = tlx.dot_operand_layout(1, mma_md_atomic_4w, k_width=8)
    # pyre-ignore[9]
    mma_md_8w: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    # pyre-ignore[9]
    ds_md_layout_8w: tl.constexpr = tlx.dot_operand_layout(0, mma_md_8w, k_width=8)
    # pyre-ignore[9]
    k_md_layout_8w: tl.constexpr = tlx.dot_operand_layout(1, mma_md_8w, k_width=8)
    use_qdo_pipeline: tl.constexpr = (INVALID_MASK_TYPE == "lower_triangular" and HAS_MULTIPLE_TARGETS
                                      and not HAS_CONTEXTUAL_SEQ_LEN and not HAS_MAX_ATTN_LEN
                                      and not HAS_FULL_ATTN_SIZE)
    if use_qdo_pipeline:
        lds_q = tlx.local_alloc(
            (BLOCK_M, BLOCK_D_Q),
            tlx.dtype_of(Q),
            2,
            # pyrefly: ignore [bad-argument-type]
            layout=shared_layout,
        )
        lds_do = tlx.local_alloc(
            (BLOCK_M, BLOCK_D_V),
            tlx.dtype_of(DOut),
            2,
            # pyrefly: ignore [bad-argument-type]
            layout=shared_layout,
        )
    else:
        lds_q = tlx.local_alloc(
            (BLOCK_M, BLOCK_D_Q),
            tlx.dtype_of(Q),
            1,
            # pyrefly: ignore [bad-argument-type]
            layout=shared_layout,
        )
        lds_do = tlx.local_alloc(
            (BLOCK_M, BLOCK_D_V),
            tlx.dtype_of(DOut),
            1,
            # pyrefly: ignore [bad-argument-type]
            layout=shared_layout,
        )

    ts_0_ptrs = None
    ts_1_ptrs = None
    ts_1 = None
    off_bias_trans = None
    bias_ptrs_trans = None
    dbias_ptrs_trans = None
    if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS:
        ts_0_ptrs = TS + offs_m
        ts_1_ptrs = TS + offs_n
        if CAUSAL:
            ts_1 = tl.load(ts_1_ptrs, mask=mask_n)
        else:
            ts_1 = tl.load(ts_1_ptrs + 1, mask=mask_n)
    elif ATTN_BIAS_TYPE == "separate":
        off_bias_trans = offs_m[None, :] * seq_len + offs_n[:, None]
        bias_ptrs_trans = Bias + off_bias_trans
        dbias_ptrs_trans = DBias + off_bias_trans
    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    qdo_slot = 0
    q_view = tlx.local_view(lds_q, 0)
    do_view = tlx.local_view(lds_do, 0)
    qdo_wait = None
    if use_qdo_pipeline:
        first_m = low + tl.arange(0, BLOCK_M)
        first_mask = first_m < seq_len
        first_q = tlx.async_load(
            q_ptrs + low * stride_qm,
            q_view,
            mask=first_mask[:, None],
        )
        first_do = tlx.async_load(
            do_ptrs + low * stride_dom,
            do_view,
            mask=first_mask[:, None],
        )
        tlx.async_load_commit_group([first_q, first_do])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    # Triton TR003: BLOCK_D_Q/BLOCK_D_V exactly match these tensor dimensions.
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # noqa: TR003
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # noqa: TR003
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN and INVALID_MASK_TYPE == "lower_triangular":
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            if num_softmax_heads > 0:
                M_block_mask = (start_m + tl.arange(0, BLOCK_M)) < seq_len
                M_block = tl.load(
                    M + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
                Delta_block = tl.load(
                    Delta + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
            else:
                M_block = tl.zeros([BLOCK_M], tl.float32)
                Delta_block = tl.zeros([BLOCK_M], tl.float32)
            dk, dv = _tlx_gfx950_ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs=q_ptrs,
                dq_ptrs_trans=dq_ptrs_trans,
                dq_ptrs=dq_ptrs,
                DQ_ACC=DQ_ACC,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                lds_q=lds_q,
                lds_do=lds_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                M_block=M_block,
                Delta_block=Delta_block,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                off_h=off_h,
                num_softmax_heads=num_softmax_heads,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
                ATOMIC_DQ=ATOMIC_DQ,
                MMA_MD_4W=mma_md_4w,
                DS_MD_LAYOUT_4W=ds_md_layout_4w,
                K_MD_LAYOUT_4W=k_md_layout_4w,
                MMA_MD_ATOMIC_4W=mma_md_atomic_4w,
                DS_MD_LAYOUT_ATOMIC_4W=ds_md_layout_atomic_4w,
                K_MD_LAYOUT_ATOMIC_4W=k_md_layout_atomic_4w,
                MMA_MD_8W=mma_md_8w,
                DS_MD_LAYOUT_8W=ds_md_layout_8w,
                K_MD_LAYOUT_8W=k_md_layout_8w,
            )

    if (HAS_MAX_ATTN_LEN and HAS_FULL_ATTN_SIZE) and has_contextual_columns == 0:
        high1 = (low + max_attn_len + BLOCK_N + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        for start_m in tl.range(low, high1, BLOCK_M, loop_unroll_factor=UNROLL):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            if num_softmax_heads > 0:
                M_block_mask = (start_m + tl.arange(0, BLOCK_M)) < seq_len
                M_block = tl.load(
                    M + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
                Delta_block = tl.load(
                    Delta + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
            else:
                M_block = tl.zeros([BLOCK_M], tl.float32)
                Delta_block = tl.zeros([BLOCK_M], tl.float32)
            dk, dv = _tlx_gfx950_ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs=q_ptrs,
                dq_ptrs_trans=dq_ptrs_trans,
                dq_ptrs=dq_ptrs,
                DQ_ACC=DQ_ACC,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                lds_q=lds_q,
                lds_do=lds_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                M_block=M_block,
                Delta_block=Delta_block,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                off_h=off_h,
                num_softmax_heads=num_softmax_heads,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
                ATOMIC_DQ=ATOMIC_DQ,
                MMA_MD_4W=mma_md_4w,
                DS_MD_LAYOUT_4W=ds_md_layout_4w,
                K_MD_LAYOUT_4W=k_md_layout_4w,
                MMA_MD_ATOMIC_4W=mma_md_atomic_4w,
                DS_MD_LAYOUT_ATOMIC_4W=ds_md_layout_atomic_4w,
                K_MD_LAYOUT_ATOMIC_4W=k_md_layout_atomic_4w,
                MMA_MD_8W=mma_md_8w,
                DS_MD_LAYOUT_8W=ds_md_layout_8w,
                K_MD_LAYOUT_8W=k_md_layout_8w,
            )

        low2 = max_ids - full_attn_size

        if low2 < high1:
            low2 = high1
        for start_m in range(low2, high, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            if num_softmax_heads > 0:
                M_block_mask = (start_m + tl.arange(0, BLOCK_M)) < seq_len
                M_block = tl.load(
                    M + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
                Delta_block = tl.load(
                    Delta + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
            else:
                M_block = tl.zeros([BLOCK_M], tl.float32)
                Delta_block = tl.zeros([BLOCK_M], tl.float32)
            dk, dv = _tlx_gfx950_ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs=q_ptrs,
                dq_ptrs_trans=dq_ptrs_trans,
                dq_ptrs=dq_ptrs,
                DQ_ACC=DQ_ACC,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                lds_q=lds_q,
                lds_do=lds_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                M_block=M_block,
                Delta_block=Delta_block,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                off_h=off_h,
                num_softmax_heads=num_softmax_heads,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
                ATOMIC_DQ=ATOMIC_DQ,
                MMA_MD_4W=mma_md_4w,
                DS_MD_LAYOUT_4W=ds_md_layout_4w,
                K_MD_LAYOUT_4W=k_md_layout_4w,
                MMA_MD_ATOMIC_4W=mma_md_atomic_4w,
                DS_MD_LAYOUT_ATOMIC_4W=ds_md_layout_atomic_4w,
                K_MD_LAYOUT_ATOMIC_4W=k_md_layout_atomic_4w,
                MMA_MD_8W=mma_md_8w,
                DS_MD_LAYOUT_8W=ds_md_layout_8w,
                K_MD_LAYOUT_8W=k_md_layout_8w,
            )
    else:
        maskless_start = high
        if (INVALID_MASK_TYPE == "lower_triangular" and HAS_MULTIPLE_TARGETS and not HAS_CONTEXTUAL_SEQ_LEN
                and not HAS_MAX_ATTN_LEN and not HAS_FULL_ATTN_SIZE):
            # A K block wholly before the target boundary has the ordinary
            # lower-triangular interior. Keep its diagonal BLOCK_N rows in the
            # masked loop and compile the remaining Q rows as a separate,
            # mask-free loop. K blocks that touch the target region stay fully
            # masked because target columns clamp pos_offs_n to max_ids.
            if start_n + BLOCK_N <= max_ids:
                maskless_start = start_n + BLOCK_N
        if use_qdo_pipeline:
            qdo_wait = tlx.async_load_wait_group(0)
        # pyre-ignore[61]
        for start_m in tl.range(low, maskless_start, BLOCK_M, loop_unroll_factor=UNROLL):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            if use_qdo_pipeline:
                next_slot = 1 - qdo_slot
                next_m = start_m + BLOCK_M
                next_mask = next_m + tl.arange(0, BLOCK_M) < seq_len
                next_q = tlx.async_load(
                    q_ptrs + next_m * stride_qm,
                    tlx.local_view(lds_q, next_slot),
                    mask=next_mask[:, None],
                )
                next_do = tlx.async_load(
                    do_ptrs + next_m * stride_dom,
                    tlx.local_view(lds_do, next_slot),
                    mask=next_mask[:, None],
                )
                tlx.async_load_commit_group([next_q, next_do])
                qdo_wait = tlx.async_load_wait_group(1)
                q_view = tlx.local_view(lds_q, qdo_slot)
                do_view = tlx.local_view(lds_do, qdo_slot)
            if num_softmax_heads > 0:
                M_block_mask = (start_m + tl.arange(0, BLOCK_M)) < seq_len
                M_block = tl.load(
                    M + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
                Delta_block = tl.load(
                    Delta + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
            else:
                M_block = tl.zeros([BLOCK_M], tl.float32)
                Delta_block = tl.zeros([BLOCK_M], tl.float32)
            dk, dv = _tlx_gfx950_ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs=q_ptrs,
                dq_ptrs_trans=dq_ptrs_trans,
                dq_ptrs=dq_ptrs,
                DQ_ACC=DQ_ACC,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                lds_q=lds_q,
                lds_do=lds_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                M_block=M_block,
                Delta_block=Delta_block,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                off_h=off_h,
                num_softmax_heads=num_softmax_heads,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
                ATOMIC_DQ=ATOMIC_DQ,
                MMA_MD_4W=mma_md_4w,
                DS_MD_LAYOUT_4W=ds_md_layout_4w,
                K_MD_LAYOUT_4W=k_md_layout_4w,
                MMA_MD_ATOMIC_4W=mma_md_atomic_4w,
                DS_MD_LAYOUT_ATOMIC_4W=ds_md_layout_atomic_4w,
                K_MD_LAYOUT_ATOMIC_4W=k_md_layout_atomic_4w,
                MMA_MD_8W=mma_md_8w,
                DS_MD_LAYOUT_8W=ds_md_layout_8w,
                K_MD_LAYOUT_8W=k_md_layout_8w,
                QDO_PRELOADED=use_qdo_pipeline,
                q_view=q_view,
                do_view=do_view,
                qdo_wait=qdo_wait,
            )
            if use_qdo_pipeline:
                qdo_slot = next_slot

        # pyre-ignore[61]
        for start_m in tl.range(maskless_start, high, BLOCK_M, loop_unroll_factor=UNROLL):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            if use_qdo_pipeline:
                next_slot = 1 - qdo_slot
                next_m = start_m + BLOCK_M
                next_mask = next_m + tl.arange(0, BLOCK_M) < seq_len
                next_q = tlx.async_load(
                    q_ptrs + next_m * stride_qm,
                    tlx.local_view(lds_q, next_slot),
                    mask=next_mask[:, None],
                )
                next_do = tlx.async_load(
                    do_ptrs + next_m * stride_dom,
                    tlx.local_view(lds_do, next_slot),
                    mask=next_mask[:, None],
                )
                tlx.async_load_commit_group([next_q, next_do])
                qdo_wait = tlx.async_load_wait_group(1)
                q_view = tlx.local_view(lds_q, qdo_slot)
                do_view = tlx.local_view(lds_do, qdo_slot)
            if num_softmax_heads > 0:
                M_block_mask = (start_m + tl.arange(0, BLOCK_M)) < seq_len
                M_block = tl.load(
                    M + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
                Delta_block = tl.load(
                    Delta + (start_m + tl.arange(0, BLOCK_M)) * stride_mm,
                    mask=M_block_mask,
                    other=0.0,
                )
            else:
                M_block = tl.zeros([BLOCK_M], tl.float32)
                Delta_block = tl.zeros([BLOCK_M], tl.float32)
            dk, dv = _tlx_gfx950_ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs=q_ptrs,
                dq_ptrs_trans=dq_ptrs_trans,
                dq_ptrs=dq_ptrs,
                DQ_ACC=DQ_ACC,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                lds_q=lds_q,
                lds_do=lds_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                M_block=M_block,
                Delta_block=Delta_block,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                off_h=off_h,
                num_softmax_heads=num_softmax_heads,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
                ATOMIC_DQ=ATOMIC_DQ,
                MMA_MD_4W=mma_md_4w,
                DS_MD_LAYOUT_4W=ds_md_layout_4w,
                K_MD_LAYOUT_4W=k_md_layout_4w,
                MMA_MD_ATOMIC_4W=mma_md_atomic_4w,
                DS_MD_LAYOUT_ATOMIC_4W=ds_md_layout_atomic_4w,
                K_MD_LAYOUT_ATOMIC_4W=k_md_layout_atomic_4w,
                MMA_MD_8W=mma_md_8w,
                DS_MD_LAYOUT_8W=ds_md_layout_8w,
                K_MD_LAYOUT_8W=k_md_layout_8w,
                IS_MASKLESS=True,
                QDO_PRELOADED=use_qdo_pipeline,
                q_view=q_view,
                do_view=do_view,
                qdo_wait=qdo_wait,
            )
            if use_qdo_pipeline:
                qdo_slot = next_slot

        if use_qdo_pipeline:
            tlx.async_load_wait_group(0)

    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    dk = dk * alpha
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    nargs["DQ_ACC"].zero_()
    if nargs["DTW"] is not None:
        nargs["DTW"].zero_()
    if nargs["DPW"] is not None:
        nargs["DPW"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    # gfx950 (MI350X) only: the CUDA and MI300 config lists from the source
    # kernel are dropped in this port.
    #
    # With DimQ=DimV=128 the fp32 dk/dv accumulators are register-heavy. The
    # eight-wave native-MFMA variant retains the production num_warps=8 configs,
    # while ordinary dot uses the profiled four-wave BLOCK_M=64 config at N=4096.
    # The other four-wave configs expose the D116875188-style native fragment
    # geometry as a separate experimental path.
    # Long N=2048/4096 cases benefit from unrolling two M tiles despite higher
    # register pressure; larger unroll factors lose to spill/schedule overhead.
    configs = []
    for BLOCK_M, BLOCK_N, num_warps, num_stages, waves_per_eu, unroll in [
        (16, 128, 4, 1, 0, 1),
        (16, 128, 4, 2, 0, 1),
        (16, 128, 4, 1, 0, 2),
        (16, 128, 4, 2, 0, 2),
        (32, 64, 4, 1, 0, 1),
        (32, 64, 4, 2, 0, 1),
        (32, 64, 4, 1, 1, 1),
        (64, 64, 4, 1, 0, 1),
        (64, 64, 4, 2, 0, 1),
        (32, 64, 4, 1, 0, 2),
        (32, 64, 4, 2, 0, 2),
        (32, 128, 4, 1, 0, 1),
        (32, 128, 4, 2, 0, 1),
        (32, 64, 8, 2, 0, 1),
        (32, 64, 8, 1, 0, 1),
        (32, 64, 8, 2, 1, 1),
        (32, 64, 8, 1, 1, 1),
        (32, 64, 8, 2, 2, 1),
        (32, 64, 8, 1, 0, 2),
        (32, 64, 8, 2, 0, 2),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_N": BLOCK_N,
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": waves_per_eu,
                    "kpack": KPACK,
                    "UNROLL": unroll,
                },
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=_bwd_pre_hook,
            ))
    return configs


def _prune_bw_configs(configs, named_args, **kwargs):
    native_mfma_dq_warps = kwargs["NATIVE_MFMA_DQ_WARPS"]
    if kwargs["KV_PARALLEL"]:
        return [
            config for config in configs if config.num_warps == 4 and config.kwargs["BLOCK_M"] == (
                16 if native_mfma_dq_warps == 4 else 32) and config.kwargs["BLOCK_N"] == 128
        ]
    use_four_waves = native_mfma_dq_warps == 4 or (native_mfma_dq_warps == 0 and kwargs["AUTOTUNE_MAX_SEQ_LEN"] >= 4096)
    num_warps = 4 if use_four_waves else 8
    selected = [config for config in configs if config.num_warps == num_warps and config.kwargs["BLOCK_N"] == 64]
    if native_mfma_dq_warps == 0 and use_four_waves:
        return [
            config for config in selected
            if config.kwargs["BLOCK_M"] == 64 and config.num_stages == 2 and config.kwargs["waves_per_eu"] == 0
        ]
    if native_mfma_dq_warps == 4:
        # Two-row unrolling benefits the 32-row tile at long sequences, while
        # the 64-row tile reduces dQ read-modify-write and loop overhead.
        unroll = 1 if kwargs["AUTOTUNE_MAX_SEQ_LEN"] <= 1024 else 2
        selected = [
            config for config in selected if config.kwargs["UNROLL"] == unroll or config.kwargs["BLOCK_M"] == 64
        ]
    return selected


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "NATIVE_MFMA_DQ_WARPS",
        "KV_PARALLEL",
    ],
    prune_configs_by={"early_config_prune": _prune_bw_configs},
)
@triton.jit
def _tlx_gfx950_ragged_hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    num_targets,
    attn_scale,
    Out,
    DOut,
    DQ,
    DQ_ACC,
    DK,
    DV,
    DBias,
    DTW,
    DPW,
    M,
    Delta,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    stride_oh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    stride_mm,
    alpha,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    PADDED_L,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    num_softmax_heads: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    NATIVE_MFMA_DQ_WARPS: tl.constexpr,
    KV_PARALLEL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr,
):
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(stride_qm >= 0)
        tl.assume(stride_qh >= 0)
        tl.assume(stride_kn >= 0)
        tl.assume(stride_kh >= 0)
        tl.assume(stride_vn >= 0)
        tl.assume(stride_vh >= 0)
        if USE_TIME_BIAS:
            tl.assume(stride_ts >= 0)
        tl.assume(stride_dom >= 0)
        tl.assume(stride_doh >= 0)
        tl.assume(stride_dqm >= 0)
        tl.assume(stride_dqh >= 0)
        tl.assume(stride_dkn >= 0)
        tl.assume(stride_dkh >= 0)
        tl.assume(stride_dvn >= 0)
        tl.assume(stride_dvh >= 0)
        tl.assume(contextual_seq_len >= 0)
        tl.assume(H > 0)
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    V = V + seq_start * stride_vn + off_h * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h * stride_doh
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    if KV_PARALLEL and NATIVE_MFMA_DQ_WARPS == 4:
        scratch_seq_start = seq_start + 15 * off_z
        DQ_ACC = DQ_ACC + (off_h * PADDED_L + scratch_seq_start) * BLOCK_D_Q
    DK = DK + seq_start * stride_dkn + off_h * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h * stride_dvh
    if ATTN_SCALE_TYPE == "dynamic":
        attn_scale = attn_scale + seq_start
    if ATTN_BIAS_TYPE == "fused":
        if USE_TIME_BIAS:
            TS = TS + off_z * stride_ts
        if FUSED_BIAS_BWD:
            if USE_TIME_BIAS:
                DTW = DTW + off_hz * (num_buckets + 1)
            if USE_POS_BIAS:
                if HAS_MAX_POS_IND:
                    DPW = DPW + off_hz * (2 * max_pos_ind - 1)
                else:
                    DPW = DPW + off_hz * (2 * MAX_SEQ_LEN - 1)
    elif ATTN_BIAS_TYPE == "separate":
        seq2_start = tl.load(seq2_offsets + off_z)
        bias_start = seq2_start * H + off_h * seq_len * seq_len
        Bias = Bias + bias_start
        DBias = DBias + bias_start

    first_n = tl.program_id(1) * BLOCK_N if KV_PARALLEL else 0
    last_n = tl.minimum(first_n + BLOCK_N, seq_len) if KV_PARALLEL else seq_len
    for start_n in range(first_n, last_n, BLOCK_N):
        _tlx_gfx950_ragged_hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
            Q=Q,
            K=K,
            V=V,
            TS=TS,
            TW=TW,
            PW=PW,
            Bias=Bias,
            DOut=DOut,
            DQ=DQ,
            DQ_ACC=DQ_ACC,
            DK=DK,
            DV=DV,
            DBias=DBias,
            DTW=DTW,
            DPW=DPW,
            M=M + seq_start * stride_mm + off_h,
            Delta=Delta + seq_start * stride_mm + off_h,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            stride_mm=stride_mm,
            alpha=alpha,
            attn_scale=attn_scale,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            off_h=off_h,
            num_softmax_heads=num_softmax_heads,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
            CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS,
            FUSED_BIAS_BWD=FUSED_BIAS_BWD,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
            ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            NATIVE_MFMA_DQ_WARPS=NATIVE_MFMA_DQ_WARPS,
            ATOMIC_DQ=KV_PARALLEL,
        )


def _validate_hstu_fa_schedule(
    fa_schedule: bool,
    fa_schedule_block_n: int,
    kv_parallel: bool,
    native_mfma_dq_warps: int,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    contextual_seq_len: int,
    max_attn_len: int,
    full_attn_size: int,
    sort_by_length_indices: Optional[torch.Tensor],
    num_softmax_heads: int,
    dim_q: int,
    dim_v: int,
) -> None:
    if not fa_schedule:
        return
    if fa_schedule_block_n not in (128, 256):
        raise ValueError("fa_schedule_block_n must be 128 or 256")
    if not kv_parallel or native_mfma_dq_warps != 4:
        raise ValueError("fa_schedule requires four-wave kv_parallel backward")
    if invalid_attn_mask_type != "lower_triangular" or num_targets is None:
        raise ValueError("fa_schedule requires target-aware lower-triangular attention")
    if (attn_scale is not None or contextual_seq_len != 0 or max_attn_len != 0 or full_attn_size != 0
            or sort_by_length_indices is not None or num_softmax_heads != 0):
        raise ValueError("fa_schedule only supports the plain causal HSTU path")
    if dim_q != 128 or dim_v != 128:
        raise ValueError("fa_schedule requires Dq=Dv=128")


# In hammer this is a `hammer::tlx_gfx950_ragged_attention_bwd` custom op
# (mutates_args=("dq", "dk", "dv")) so it can be traced by torch.compile/AOTI.
# The tutorial copy keeps it a plain function -- it is only ever called from
# `TlxGfx950RaggedAttentionFunction.backward` under `torch.inference_mode()`, and
# registering into a namespace here would collide with hammer if both are loaded.
def tlx_gfx950_ragged_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    invalid_attn_mask_type: str,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    full_attn_size: int,
    native_mfma_dq_warps: int = 0,
    kv_parallel: bool = False,
    fa_schedule: bool = False,
    fa_schedule_block_n: int = 128,
    fa_schedule_direct_qdo_g2l: bool = False,
    fa_schedule_mask_peel: bool = False,
    fa_schedule_resident_k_score: bool = False,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
) -> None:
    if native_mfma_dq_warps not in (0, 4, 8):
        raise ValueError(f"native_mfma_dq_warps must be 0, 4, or 8; got {native_mfma_dq_warps}")
    if kv_parallel and native_mfma_dq_warps not in (0, 4):
        raise ValueError("kv_parallel supports ordinary-dot or four-wave native dQ")
    dout = _switch_to_contiguous_if_needed(dout)
    dq = _switch_to_contiguous_if_needed(dq)
    dk = _switch_to_contiguous_if_needed(dk)
    dv = _switch_to_contiguous_if_needed(dv)
    max_attn_len = max_attn_len or 0
    full_attn_size = full_attn_size or 0
    _validate_hstu_fa_schedule(
        fa_schedule,
        fa_schedule_block_n,
        kv_parallel,
        native_mfma_dq_warps,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        contextual_seq_len,
        max_attn_len,
        full_attn_size,
        sort_by_length_indices,
        num_softmax_heads,
        q.shape[2],
        v.shape[2],
    )
    if dout.shape[0] == 0:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        return
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    L, _, DimV = v.shape
    native_dq_acc = kv_parallel and native_mfma_dq_warps == 4
    padded_l = L + 15 * Z if native_dq_acc else L
    dq_acc = (torch.empty((H, padded_l, DimQ), dtype=dq.dtype, device=dq.device) if native_dq_acc else dq)
    stride_mm = num_softmax_heads
    if M is None:
        M = torch.empty(1, dtype=torch.float32, device=q.device)

    if num_softmax_heads > 0 and Out is not None:
        Delta = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=q.device)
        BLOCK_M_PRE = 128
        pre_grid = (triton.cdiv(N, BLOCK_M_PRE), Z * H)
        _attn_bwd_preprocess[pre_grid](
            Out=Out,
            DOut=dout,
            Delta=Delta,
            seq_offsets=seq_offsets,
            stride_om=Out.stride(0),
            stride_oh=Out.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            H=H,
            num_softmax_heads=num_softmax_heads,
            # pyre-ignore[6]
            BLOCK_M=BLOCK_M_PRE,
            # pyre-ignore[6]
            BLOCK_D_V=DimV,
        )
    else:
        Delta = torch.empty(1, dtype=torch.float32, device=q.device)
        Out = torch.empty(1, dtype=v.dtype, device=q.device)

    grid = lambda META: (  # noqa: E731
        Z * H,
        triton.cdiv(N, META["BLOCK_N"]) if kv_parallel else 1,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    attn_scale_type: str = "none"
    if attn_scale is not None:
        if attn_scale.ndim == 0:
            attn_scale_type = "scalar"
        else:
            attn_scale_type = "dynamic"
    # Check for all non-negative strides as a common case guard
    enable_buffer_ops_assumes = (q.stride(0) >= 0 and q.stride(1) >= 0 and k.stride(0) >= 0 and k.stride(1) >= 0
                                 and v.stride(0) >= 0 and v.stride(1) >= 0 and dout.stride(0) >= 0
                                 and dout.stride(1) >= 0 and dq.stride(0) >= 0 and dq.stride(1) >= 0
                                 and dk.stride(0) >= 0 and dk.stride(1) >= 0 and dv.stride(0) >= 0
                                 and dv.stride(1) >= 0)
    if fa_schedule:
        dq.zero_()
        dq_acc.zero_()
        _tlx_gfx950_hstu_fa_schedule_bwd_kernel[(H, triton.cdiv(N, fa_schedule_block_n), Z)](
            Q=q,
            K=k,
            V=v,
            DOut=dout,
            DQ_ACC=dq_acc,
            DK=dk,
            DV=dv,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            alpha=alpha,
            PADDED_L=padded_l,
            H=H,
            MAX_SEQ_LEN=N,
            BLOCK_N=fa_schedule_block_n,
            DIRECT_QDO_G2L=fa_schedule_direct_qdo_g2l,
            MASK_PEEL=fa_schedule_mask_peel,
            RESIDENT_K_SCORE=fa_schedule_resident_k_score,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=16,
            waves_per_eu=0,
            reverse_local_assignment=not fa_schedule_direct_qdo_g2l,
        )
    else:
        _tlx_gfx950_ragged_hstu_attn_bwd[grid](
            Q=q,
            K=k,
            V=v,
            sort_by_length_indices=sort_by_length_indices,
            seq_offsets=seq_offsets,
            TS=None,
            TW=None,
            PW=None,
            Bias=None,
            seq2_offsets=None,
            num_targets=num_targets,
            attn_scale=attn_scale,
            Out=Out,
            DOut=dout,
            DQ=dq,
            DQ_ACC=dq_acc,
            DK=dk,
            DV=dv,
            DBias=None,
            DTW=None,
            DPW=None,
            M=M,
            Delta=Delta,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_ts=None,
            stride_om=Out.stride(0) if num_softmax_heads > 0 else 0,
            stride_oh=Out.stride(1) if num_softmax_heads > 0 else 0,
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            stride_mm=stride_mm,
            alpha=alpha,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
            PADDED_L=padded_l,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            MAX_SEQ_LEN=N,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
            DimQ=DimQ,
            DimV=DimV,
            num_buckets=None,
            max_pos_ind=None,
            time_bucket_incr=None,
            time_bucket_div=None,
            time_delta=None,
            num_softmax_heads=num_softmax_heads,
            INVALID_MASK_TYPE=invalid_attn_mask_type,
            CAUSAL=None,
            BUCKET_FN="none",
            ATTN_BIAS_TYPE="none",
            ATTN_SCALE_TYPE=attn_scale_type,
            USE_TIME_BIAS=False,
            USE_POS_BIAS=False,
            FUSED_BIAS_BWD=None,
            HAS_MAX_POS_IND=False,
            HAS_MULTIPLE_TARGETS=num_targets is not None,
            HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
            HAS_MAX_ATTN_LEN=max_attn_len > 0,
            HAS_FULL_ATTN_SIZE=full_attn_size > 0,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            NATIVE_MFMA_DQ_WARPS=native_mfma_dq_warps,
            KV_PARALLEL=kv_parallel,
            HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
            ENABLE_BUFFER_OPS_ASSUMES=enable_buffer_ops_assumes,
        )
    if native_dq_acc:
        convert_block_m = 128
        _tlx_gfx950_hstu_native_dq_convert[(Z * H, triton.cdiv(N, convert_block_m))](
            DQ_ACC=dq_acc,
            DQ=dq,
            seq_offsets=seq_offsets,
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            PADDED_L=padded_l,
            H=H,
            BLOCK_D_Q=DimQ,
            BLOCK_M=convert_block_m,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )


class TlxGfx950RaggedAttentionFunction(torch.autograd.Function):

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        invalid_attn_mask_type: str,
        num_targets: Optional[torch.Tensor],
        attn_scale: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        seq2_offsets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
        full_attn_size: int,
        num_softmax_heads: int,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(seq_lengths, descending=True, stable=False)
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if attn_scale is not None:
            saved_tensors.append(attn_scale)
        assert attn_bias is None
        assert seq2_offsets is None
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_attn_scale = attn_scale is not None
        ctx.max_attn_len = max_attn_len
        ctx.full_attn_size = full_attn_size
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.num_softmax_heads = num_softmax_heads
        ctx.save_for_backward(*saved_tensors)
        out, M = tlx_gfx950_ragged_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            num_targets=num_targets,
            attn_scale=attn_scale,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            full_attn_size=full_attn_size,
            num_softmax_heads=num_softmax_heads,
        )
        ctx.M = M
        ctx.out = out if num_softmax_heads > 0 else None
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
            None,
            None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
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
            None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.has_attn_scale:
                attn_scale = ctx.saved_tensors[idx]
                idx += 1
            else:
                attn_scale = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            kv_parallel = getattr(ctx, "kv_parallel", None)
            if kv_parallel is None:
                kv_parallel = (ctx.N >= 2048 and ctx.invalid_attn_mask_type == "lower_triangular"
                               and ctx.has_multiple_targets and ctx.contextual_seq_len == 0 and ctx.max_attn_len == 0
                               and ctx.full_attn_size == 0 and ctx.num_softmax_heads == 0 and not ctx.sort_by_length
                               and getattr(ctx, "native_mfma_dq_warps", 0) == 0)
            tlx_gfx950_ragged_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                attn_scale=attn_scale,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                invalid_attn_mask_type=ctx.invalid_attn_mask_type,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
                full_attn_size=ctx.full_attn_size,
                native_mfma_dq_warps=getattr(ctx, "native_mfma_dq_warps", 0),
                kv_parallel=kv_parallel,
                fa_schedule=getattr(ctx, "fa_schedule", False),
                fa_schedule_block_n=getattr(ctx, "fa_schedule_block_n", 128),
                fa_schedule_direct_qdo_g2l=getattr(ctx, "fa_schedule_direct_qdo_g2l", False),
                fa_schedule_mask_peel=getattr(ctx, "fa_schedule_mask_peel", False),
                fa_schedule_resident_k_score=getattr(ctx, "fa_schedule_resident_k_score", False),
                num_softmax_heads=ctx.num_softmax_heads,
                M=ctx.M,
                Out=ctx.out,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
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
                None,
            )


class TlxGfx950RaggedAttentionNativeMFMA4WaveFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionNativeMFMA8WaveFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 8
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.kv_parallel = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelNativeMFMA4WaveFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleDirectQDOG2LFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        ctx.fa_schedule_direct_qdo_g2l = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleMaskPeelFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        ctx.fa_schedule_direct_qdo_g2l = True
        ctx.fa_schedule_mask_peel = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleMaskPeelResidentKFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        ctx.fa_schedule_direct_qdo_g2l = True
        ctx.fa_schedule_mask_peel = True
        ctx.fa_schedule_resident_k_score = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleBN256Function(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        ctx.fa_schedule_block_n = 256
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


class TlxGfx950RaggedAttentionKVParallelFAScheduleBN256DirectQDOG2LFunction(TlxGfx950RaggedAttentionFunction):

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor):
        ctx.native_mfma_dq_warps = 4
        ctx.kv_parallel = True
        ctx.fa_schedule = True
        ctx.fa_schedule_block_n = 256
        ctx.fa_schedule_direct_qdo_g2l = True
        return TlxGfx950RaggedAttentionFunction.backward(ctx, dout)


def tlx_gfx950_ragged_attention_relative_bias_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    timestamps: torch.Tensor,
    ts_weights: torch.Tensor,
    pos_weights: torch.Tensor,
    causal: bool,
    num_buckets: int,
    time_bucket_fn: str,
    time_bucket_incr: float,
    time_bucket_div: float,
    time_delta: float,
    max_pos_ind: Optional[int],
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    relative_bias_type: str,
    max_attn_len: int,
    use_time_bias: bool,
    use_pos_bias: bool,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    Z = timestamps.size(0)
    AUTOTUNE_Z = prev_power_of_2(Z)
    N = timestamps.size(1) - 1
    has_multiple_targets = num_targets is not None
    has_max_pos_id = max_pos_ind is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_full_attn_size = full_attn_size > 0
    attn_scale_type: str = "none"
    if attn_scale is not None:
        if attn_scale.ndim == 0:
            attn_scale_type = "scalar"
        else:
            attn_scale_type = "dynamic"

    L, H, DimQ = q.shape
    _, _, DimV = v.shape

    out_buffer: Optional[torch.Tensor] = None
    M_buffer: torch.Tensor
    if num_softmax_heads > 0:
        out = torch.empty((L, H, DimV), dtype=v.dtype, device=v.device)
        out_buffer = out.view(-1)
        M_buffer = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=v.device)
    else:
        out = torch.empty_like(v)
        out_buffer = out.view(-1)
        M_buffer = torch.empty(1, dtype=torch.float32, device=v.device)

    if L == 0:
        if num_softmax_heads > 0:
            M = torch.empty(0, num_softmax_heads, dtype=torch.float32, device=v.device)
        else:
            M = torch.empty(1, dtype=torch.float32, device=v.device)
        return out, M

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _tlx_gfx950_ragged_hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        seq_offsets=seq_offsets,
        TS=timestamps,
        TW=ts_weights,
        PW=pos_weights,
        Bias=None,
        seq2_offsets=None,
        num_targets=num_targets,
        attn_scale=attn_scale,
        Out=out_buffer,
        M_buffer=M_buffer,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=timestamps.stride(0),
        stride_om=H * DimV,
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=causal,
        BUCKET_FN=time_bucket_fn,
        ATTN_BIAS_TYPE="fused",
        ATTN_SCALE_TYPE=attn_scale_type,
        USE_TIME_BIAS=use_time_bias,
        USE_POS_BIAS=use_pos_bias,
        HAS_MAX_POS_IND=has_max_pos_id,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_FULL_ATTN_SIZE=has_full_attn_size,
        num_softmax_heads=num_softmax_heads,
    )

    if num_softmax_heads > 0:
        M = M_buffer
    else:
        M = torch.empty(1, dtype=torch.float32, device=v.device)

    return out, M


class TlxGfx950RaggedAttentionRelativeBiasFunction(torch.autograd.Function):

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        invalid_attn_mask_type: str,
        timestamps: torch.Tensor,
        ts_weights: torch.Tensor,
        pos_weights: torch.Tensor,
        causal: bool,
        num_buckets: int,
        time_bucket_fn: str,
        time_bucket_incr: float,
        time_bucket_div: float,
        time_delta: float,
        max_pos_ind: Optional[int],
        num_targets: Optional[torch.Tensor],
        attn_scale: Optional[torch.Tensor],
        relative_bias_type: str,
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
        full_attn_size: int,
        num_softmax_heads: int = 0,
    ) -> torch.Tensor:
        use_time_bias = relative_bias_type == "TIME" or relative_bias_type == "ALL"
        use_pos_bias = relative_bias_type == "POSITION" or relative_bias_type == "ALL"
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(seq_lengths, descending=True, stable=False)
        saved_tensors: List[torch.Tensor] = [
            timestamps,
            ts_weights,
            pos_weights,
            q,
            k,
            v,
            seq_offsets,
        ]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if attn_scale is not None:
            saved_tensors.append(attn_scale)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_attn_scale = attn_scale is not None
        ctx.max_pos_ind = max_pos_ind
        ctx.N = N
        ctx.num_buckets = num_buckets
        ctx.time_bucket_fn = time_bucket_fn
        ctx.time_bucket_incr = time_bucket_incr
        ctx.time_bucket_div = time_bucket_div
        ctx.causal = causal
        ctx.time_delta = time_delta
        ctx.use_time_bias = use_time_bias
        ctx.use_pos_bias = use_pos_bias
        ctx.max_attn_len = max_attn_len
        ctx.full_attn_size = full_attn_size
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.num_softmax_heads = num_softmax_heads
        out, M = tlx_gfx950_ragged_attention_relative_bias_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            timestamps=timestamps,
            ts_weights=ts_weights,
            pos_weights=pos_weights,
            causal=causal,
            num_buckets=num_buckets,
            time_bucket_fn=time_bucket_fn,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            max_pos_ind=max_pos_ind,
            num_targets=num_targets,
            attn_scale=attn_scale,
            relative_bias_type=relative_bias_type,
            max_attn_len=max_attn_len,
            use_time_bias=use_time_bias,
            use_pos_bias=use_pos_bias,
            contextual_seq_len=contextual_seq_len,
            full_attn_size=full_attn_size,
            num_softmax_heads=num_softmax_heads,
        )
        ctx.M = M
        ctx.out = out if num_softmax_heads > 0 else None
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *args):  # pyre-ignore[3]
        raise NotImplementedError("the relative-bias backward is not ported yet: it needs "
                                  "_attn_bias_bwd and triton_ragged_attention_relative_bias_bwd "
                                  "from hammer/ops/triton/triton_ragged_hstu_attention.py.")


@tlx_gfx950_ragged_attention_fwd.register_fake
@tlx_gfx950_ragged_attention_fwd.register_kernel("cpu")
def _tlx_gfx950_ragged_attention_fwd_fake(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L, H, _ = q.shape
    _, _, DimV = v.shape
    out = torch.empty((L, H, DimV), dtype=v.dtype, device=v.device)
    if num_softmax_heads > 0:
        M = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=v.device)
    else:
        M = torch.empty(1, dtype=torch.float32, device=v.device)
    return out, M


# -----------------------------------------------------------------------------
# Tutorial / benchmark entry point
# -----------------------------------------------------------------------------
#
# The autograd Functions above differ only in which backward schedule they pick,
# so callers that just want "the gfx950 TLX HSTU kernel" go through
# `tlx_gfx950_hstu_mha` and name a variant. Same shape as the sibling
# `tlx_bw_hstu_attention.tlx_bw_hstu_mha` so both can be driven from one harness.

BWD_VARIANTS = {
    "default": TlxGfx950RaggedAttentionFunction,
    "native_mfma_4wave": TlxGfx950RaggedAttentionNativeMFMA4WaveFunction,
    "native_mfma_8wave": TlxGfx950RaggedAttentionNativeMFMA8WaveFunction,
    "kv_parallel": TlxGfx950RaggedAttentionKVParallelFunction,
    "kv_parallel_native_mfma_4wave": (TlxGfx950RaggedAttentionKVParallelNativeMFMA4WaveFunction),
    "kv_parallel_fa_schedule": TlxGfx950RaggedAttentionKVParallelFAScheduleFunction,
    "kv_parallel_fa_schedule_direct_qdo_g2l": (TlxGfx950RaggedAttentionKVParallelFAScheduleDirectQDOG2LFunction),
    "kv_parallel_fa_schedule_mask_peel": (TlxGfx950RaggedAttentionKVParallelFAScheduleMaskPeelFunction),
    "kv_parallel_fa_schedule_mask_peel_resident_k":
    (TlxGfx950RaggedAttentionKVParallelFAScheduleMaskPeelResidentKFunction),
    "kv_parallel_fa_schedule_bn256": (TlxGfx950RaggedAttentionKVParallelFAScheduleBN256Function),
    "kv_parallel_fa_schedule_bn256_direct_qdo_g2l":
    (TlxGfx950RaggedAttentionKVParallelFAScheduleBN256DirectQDOG2LFunction),
}

# BN128 resident-K mask-peel is the best backward at N >= 2048; the BN256
# direct-load variant still wins at N == 1024.
DEFAULT_BWD_VARIANT: str = "kv_parallel_fa_schedule_mask_peel_resident_k"


def tlx_gfx950_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: Optional[torch.Tensor] = None,
    num_targets: Optional[torch.Tensor] = None,
    invalid_attn_mask_type: str = "lower_triangular",
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    bwd_variant: str = DEFAULT_BWD_VARIANT,
) -> torch.Tensor:
    """gfx950 TLX ragged HSTU attention (fwd + bwd) as a single callable.

    `bwd_variant` selects the backward schedule; it has no effect on the
    forward. See `BWD_VARIANTS` for the list.
    """
    try:
        autograd_fn = BWD_VARIANTS[bwd_variant]
    except KeyError:
        raise ValueError(f"unknown bwd_variant {bwd_variant!r}; "
                         f"expected one of {sorted(BWD_VARIANTS)}") from None
    return autograd_fn.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        None,  # attn_bias
        None,  # seq2_offsets
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
        full_attn_size,
        num_softmax_heads,
    )
