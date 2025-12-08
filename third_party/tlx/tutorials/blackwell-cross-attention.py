# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List
from typing import Optional

import pytest
import torch
import os

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton._internal_testing import is_blackwell

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore[attr-defined]

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False

# @manual=//triton:triton
from triton.language.extra.libdevice import fast_dividef

# @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    DimV = nargs["BLOCK_D_V"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, DimV]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]


def _host_descriptor_pre_hook_ws(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    DimV = nargs["BLOCK_D_V"]
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, DimV]


def _host_descriptor_pre_hook_spec(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    DimV = nargs["BLOCK_D_V"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, DimV]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_q1"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v1"].block_shape = [BLOCK_N1, DimV]
    nargs["desc_k1"].block_shape = [BLOCK_N1, HEAD_DIM]


def get_fwd_pipeline_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "NUM_MMA_GROUPS": 2,
                "NUM_BUFFERS_KV": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "NUM_MMA_GROUPS": 2,
                "NUM_BUFFERS_KV": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "NUM_MMA_GROUPS": 2,
                "NUM_BUFFERS_KV": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        ),
    ]
    return configs


@triton.jit
def forward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr):
    valid_mask = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
    if HAS_CAUSAL:
        offs_m = offs_m + seq_len_kv - uih_len_q
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = valid_mask & causal_mask
    return valid_mask


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    buf_id = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return buf_id, phase


@triton.jit
def _compute_offsets(H, BLOCK_M: tl.constexpr, seq_offsets_q, seq_offsets):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    start_m = tl.program_id(0) * BLOCK_M
    seq_start_kv = tl.load(seq_offsets + off_z)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return start_m, off_h, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q


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
def fast_silu(x):
    # Replace divf(1, 1 + expf(-x)) with (1 + tanhf(x/2)) / 2
    # If an approximate instruction exists.
    x = x * 0.5
    return x * (1 + tanh_approx_fp32(x))


def get_fwd_triton_single() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "NUM_MMA_GROUPS": 1,
                "REMAT_OFF": off,
                "OPT_MASK": mask,
                "TMA_STORE": tma,
                "TRANS": trans,
                "NUM_STAGES": ns,
            },
            num_stages=0,
            num_warps=nw,
            pre_hook=_host_descriptor_pre_hook,
        )
        for bm in [128]  # 32, 64, 128]
        for bn in [64]  # 32, 64, 128]
        for nw in [4]  # 2, 4, 8]
        for ns in [2]  # 2
        for off in [False]
        for mask in [True]  #True]
        for tma in [False]  # False]
        for trans in [True]  #True]
    ]
    return configs


def get_fwd_triton_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "NUM_MMA_GROUPS": 1,
                "REMAT_OFF": off,
                "OPT_MASK": mask,
                "TMA_STORE": tma_trans[0],
                "TRANS": tma_trans[1],
                "NUM_STAGES": ns,
            },
            num_stages=0,
            num_warps=nw,
            pre_hook=_host_descriptor_pre_hook,
        )
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for nw in [2, 4]
        for ns in [2, 4]
        for off in [False]
        for mask in [True]
        for tma_trans in [(True, False), (False, True), (False, False)]
        # trans doesn't work with TMA
    ]
    return configs


@triton.jit
def forward_valid_mask_trans(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr):
    valid_mask = (offs_m[None, :] < seq_len_q) & (offs_n[:, None] < seq_len_kv)
    return valid_mask


@triton.jit
def _attn_fwd_triton_inner(
    alpha,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_om,
    stride_oh,
    start_m,
    off_h,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,
    REMAT_OFF: tl.constexpr,
    TRANS: tl.constexpr,
    OPT_MASK: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    WITH_MASK: tl.constexpr,
    WITH_ACT: tl.constexpr,  # when this is false, WITH_MASK should be false too
    WITH_STORE: tl.constexpr,
):
    # initialize offsets
    if start_m < seq_len_q:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n_0 = tl.arange(0, BLOCK_N)
        qo_offset_y_split = seq_start_q + start_m
        q = desc_q.load([qo_offset_y_split.to(tl.int32), off_h * stride_qh])
        if TRANS:
            acc = tl.zeros([BLOCK_D_V, BLOCK_M], dtype=tl.float32)
        else:
            acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        for start_n in tl.range(0, seq_len_kv, BLOCK_N, num_stages=NUM_STAGES):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = desc_k.load([(seq_start_kv + start_n).to(tl.int32), off_h * stride_kh])
            v = desc_v.load([(seq_start_kv + start_n).to(tl.int32), off_h * stride_vh])

            if WITH_MASK:
                if REMAT_OFF:
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                else:
                    offs_n = offs_n_0 + start_n
            if TRANS:
                qk = tl.dot(k, tl.trans(q))  # BM by BN
                if WITH_MASK:
                    valid_mask = forward_valid_mask_trans(
                        offs_m,
                        offs_n,
                        0,  # uih_len_q
                        seq_len_q,
                        seq_len_kv,
                        False,
                    )
            else:
                qk = tl.dot(q, tl.trans(k))
                if WITH_MASK:
                    valid_mask = forward_valid_mask(
                        offs_m,
                        offs_n,
                        0,  # uih_len_q
                        seq_len_q,
                        seq_len_kv,
                        False,
                    )
            if OPT_MASK:
                if WITH_MASK:
                    masked_alpha = tl.where(valid_mask, alpha, 0.0)
                    qk = qk * masked_alpha
                else:
                    if WITH_ACT:
                        qk = qk * alpha
                if WITH_ACT:
                    # silu = fast_dividef(qk, 1.0 + tl.exp(-qk))
                    silu = fast_silu(qk)
                    act_qk = silu.to(v.dtype)
                else:
                    act_qk = qk.to(v.dtype)
            else:
                if WITH_ACT:
                    qk = qk * alpha
                    silu = fast_dividef(qk, 1.0 + tl.exp(-qk))
                if WITH_MASK:
                    act_qk = tl.where(valid_mask, silu, 0.0)  # triton
                    act_qk = act_qk.to(v.dtype)
                else:
                    if WITH_ACT:
                        act_qk = silu.to(v.dtype)
                    else:
                        act_qk = qk.to(v.dtype)
            if TRANS:
                acc += tl.dot(tl.trans(v), act_qk)
            else:
                acc += tl.dot(act_qk, v)

        # epilogue
        if WITH_STORE:
            acc = acc / max_seq_len
            out_offset = off_h.to(tl.int64) * stride_oh
            end_o = seq_start_q + seq_len_q
            # we are writing out Out.T which is hDim x BM
            if TMA_STORE:
                if TRANS:  # This does not work
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[HEAD_DIM * H, end_o.to(tl.int32)],
                        strides=[1, HEAD_DIM * H],
                        block_shape=[BLOCK_D_V, BLOCK_M],
                    )
                    o_desc.store(
                        [
                            (out_offset).to(tl.int32),
                            (seq_start_q + start_m).to(tl.int32),
                        ],
                        acc.to(Out.dtype.element_ty),
                    )
                else:
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M, BLOCK_D_V],
                    )
                    o_desc.store(
                        [
                            (seq_start_q + start_m).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                        acc.to(Out.dtype.element_ty),
                    )
            else:
                if TRANS:
                    off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    offs_v_d = tl.arange(0, BLOCK_D_V)
                    out_ptrs = off_o + offs_m[None, :] * stride_om + offs_v_d[:, None]
                    acc = acc.to(Out.dtype.element_ty)
                    tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[None, :])
                else:
                    off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    offs_v_d = tl.arange(0, BLOCK_D_V)
                    out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
                    acc = acc.to(Out.dtype.element_ty)
                    tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])


fwd_triton_configs_sel = get_fwd_triton_configs()
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = get_fwd_triton_single()


# BLOCK_M: 32, BLOCK_N: 32, NUM_MMA_GROUPS: 1, REMAT_OFF: False, OPT_MASK: True, TMA_STORE: False, TRANS: True, NUM_STAGES: 1, num_warps: 4
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    TRANS = conf.kwargs["TRANS"]
    return not (BLOCK_M >= 32 and BLOCK_N == 32 and TRANS and conf.num_warps >= 4)


@triton.autotune(configs=list(filter(keep, fwd_triton_configs_sel)), key=["Z", "HEAD_DIM", "AUTOTUNE_MAX_Q_LEN"])
@triton.jit
def _attn_fwd_triton(
    alpha,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_om,
    stride_oh,
    AUTOTUNE_MAX_Q_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,
    REMAT_OFF: tl.constexpr,
    TRANS: tl.constexpr,
    OPT_MASK: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    start_m, off_h, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = _compute_offsets(
        H, BLOCK_M, seq_offsets_q, seq_offsets)
    _attn_fwd_triton_inner(
        alpha,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        Out,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        stride_qh,
        stride_kh,
        stride_vh,
        stride_om,
        stride_oh,
        start_m,
        off_h,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        HEAD_DIM,
        BLOCK_D_V,
        BLOCK_M,
        BLOCK_N,
        NUM_MMA_GROUPS,
        REMAT_OFF,
        TRANS,
        OPT_MASK,
        TMA_STORE,
        NUM_STAGES,
        WITH_MASK=True,
        WITH_ACT=True,
        WITH_STORE=True,
    )


def get_fwd_triton_spec_single() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_M1": 64,
                "BLOCK_N1": 64,
                "NUM_MMA_GROUPS": 1,
                "REMAT_OFF": False,
                "OPT_MASK": True,
                "TMA_STORE": False,
                "TRANS": True,
                "NUM_STAGES": 2,
                "NUM_STAGES1": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_spec,
        )
    ]
    return configs


def get_fwd_triton_spec_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_M1": bm1,
                "BLOCK_N1": bn1,
                "NUM_MMA_GROUPS": 1,
                "REMAT_OFF": off,
                "OPT_MASK": mask,
                "TMA_STORE": tma_trans[0],
                "TRANS": tma_trans[1],
                "NUM_STAGES": ns,
                "NUM_STAGES1": ns1,
            },
            num_stages=0,
            num_warps=nw,
            pre_hook=_host_descriptor_pre_hook_spec,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bm1 in [32, 64]
        for bn1 in [32, 64]
        for nw in [2, 4]
        for ns in [2, 4]
        for ns1 in [2, 4]
        for off in [False]
        for mask in [True]
        for tma_trans in [(True, False), (False, True), (False, False)]
    ]
    return configs


fwd_triton_spec_configs_sel = get_fwd_triton_spec_configs()
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = get_fwd_triton_spec_single()


# BLOCK_M: 32, BLOCK_N: 32, NUM_MMA_GROUPS: 1, REMAT_OFF: False, OPT_MASK: True, TMA_STORE: False, TRANS: True, NUM_STAGES: 1, num_warps: 4
def keep_spec(conf):
    BLOCK_N = conf.kwargs["BLOCK_N"]
    BLOCK_N1 = conf.kwargs["BLOCK_N1"]
    TRANS = conf.kwargs["TRANS"]
    return not ((BLOCK_N1 == 32 or BLOCK_N == 32) and TRANS and conf.num_warps >= 4)


@triton.autotune(configs=list(filter(keep_spec, fwd_triton_spec_configs_sel)),
                 key=["Z", "HEAD_DIM", "AUTOTUNE_MAX_Q_LEN"])
@triton.jit
def _attn_fwd_triton_spec(
    alpha,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_q1,
    desc_k1,
    desc_v1,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_om,
    stride_oh,
    AUTOTUNE_MAX_Q_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,
    REMAT_OFF: tl.constexpr,
    TRANS: tl.constexpr,
    OPT_MASK: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_STAGES1: tl.constexpr,
):
    start_m, off_h, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = _compute_offsets(
        H, BLOCK_M, seq_offsets_q, seq_offsets)
    # grid is using BLOCK_M, we need to make sure seq_len_q is handled in the thread block.
    if seq_len_q <= BLOCK_M1:
        _attn_fwd_triton_inner(
            alpha,
            Z,
            H,
            desc_q1,
            desc_k1,
            desc_v1,
            Out,
            seq_offsets_q,
            seq_offsets,
            max_seq_len,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_om,
            stride_oh,
            start_m,
            off_h,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            HEAD_DIM,
            BLOCK_D_V,
            BLOCK_M1,
            BLOCK_N1,
            NUM_MMA_GROUPS,
            REMAT_OFF,
            TRANS,
            OPT_MASK,
            TMA_STORE,
            NUM_STAGES1,
            WITH_MASK=True,
            WITH_ACT=True,
            WITH_STORE=True,
        )
    else:
        _attn_fwd_triton_inner(
            alpha,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            Out,
            seq_offsets_q,
            seq_offsets,
            max_seq_len,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_om,
            stride_oh,
            start_m,
            off_h,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            HEAD_DIM,
            BLOCK_D_V,
            BLOCK_M,
            BLOCK_N,
            NUM_MMA_GROUPS,
            REMAT_OFF,
            TRANS,
            OPT_MASK,
            TMA_STORE,
            NUM_STAGES,
            WITH_MASK=True,
            WITH_ACT=True,
            WITH_STORE=True,
        )


def get_fwd_single() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
    ]
    return configs


def get_fwd_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "NUM_BUFFERS_KV": 2,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook_ws,
        ),
    ]
    return configs


@triton.autotune(configs=get_fwd_configs(), key=["Z", "HEAD_DIM", "AUTOTUNE_MAX_Q_LEN"])
@triton.jit
def _attn_fwd_single_q(alpha, Z, H, desc_q, desc_k, desc_v, Out, seq_offsets_q, seq_offsets, max_seq_len, stride_qh,
                       stride_kh, stride_vh, stride_om, stride_oh, AUTOTUNE_MAX_Q_LEN: tl.constexpr,
                       HEAD_DIM: tl.constexpr,  #
                       BLOCK_D_V: tl.constexpr, BLOCK_M: tl.constexpr,  #
                       BLOCK_N: tl.constexpr,  #
                       NUM_BUFFERS_KV: tl.constexpr,  #
                       NUM_MMA_GROUPS: tl.constexpr,  #
                       ):
    """
    Single Q, multiple K/V pipeline
    """
    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_q), 1)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)

    q_fulls = tlx.alloc_barriers(num_barriers=1)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )
    # p_tiles is in bf16/fp6, when reusing qk_tiles which is fp32,
    # we need to create 2xNUM_MMA_GROUPS of p_tiles and use the
    # lower half for p1 so that  so that
    # q0k won't overwrite p1.
    p_tiles = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc(
        (BLOCK_M, BLOCK_D_V),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_fulls = tlx.alloc_barriers(num_barriers=1)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            # initialize offsets
            start_m, off_h, _, seq_len_kv, seq_start_q, seq_len_q = _compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets)
            if start_m < seq_len_q:
                tlx.barrier_wait(acc_fulls[0], 0)
                off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                offs_m = start_m + tl.arange(0, BLOCK_M)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
                acc = tlx.local_load(acc_tiles[0])
                # TODO: using 1/ max_seq_len as attn_scale for now, need to fix later
                acc = acc / max_seq_len
                acc = acc.to(tlx.dtype_of(desc_v))
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])
        # silu groups
        with tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS):
            # initialize offsets
            start_m, off_h, _, seq_len_kv, seq_start_q, seq_len_q = _compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets)
            if start_m < seq_len_q:
                phase = 0
                cid = tlx.async_task_replica_id()
                for start_n in tl.range(cid * BLOCK_N, seq_len_kv, BLOCK_N * NUM_MMA_GROUPS):
                    tlx.barrier_wait(qk_fulls[cid], phase)
                    qk = tlx.local_load(qk_tiles[cid])

                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                    valid_mask = forward_valid_mask(
                        offs_m,
                        offs_n,
                        0,  # uih_len_q
                        seq_len_q,
                        seq_len_kv,
                        False,
                    )
                    qk = qk * alpha
                    silu = fast_dividef(qk, 1.0 + tl.exp(-qk))
                    act_qk = tl.where(valid_mask, silu, 0.0)
                    act_qk = act_qk.to(tlx.dtype_of(desc_v))
                    tlx.local_store(p_tiles[cid * 2], act_qk)
                    tlx.barrier_arrive(p_fulls[cid])
                    phase ^= 1
        # mma group
        with tlx.async_task(num_warps=1, registers=32):
            start_m, off_h, _, seq_len_kv, seq_start_q, seq_len_q = _compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets)
            if start_m < seq_len_q:
                # wait for the Q buffer to be populated by the producer
                tlx.barrier_wait(q_fulls[0], 0)

                kv_cnt = 0
                # Q @ K0
                k_buff_id, phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                k_tile = tlx.local_trans(kv_tiles[k_buff_id])
                tlx.barrier_wait(kv_fulls[k_buff_id], phase)
                qk_cnt = 0
                qk_id, p_phase = _get_bufidx_phase(qk_cnt, NUM_MMA_GROUPS)
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[qk_id],
                    use_acc=False,
                    mBarriers=[qk_fulls[qk_id], kv_empties[k_buff_id]],
                )
                acc_pv = False
                # loop over k, v and update accumulator
                for start_n in tl.range(BLOCK_N, seq_len_kv, BLOCK_N):
                    qk_cnt += 1
                    kv_cnt += 1
                    # -- compute q @ k(i) ----
                    # wait for the K buffer to be populated by the producer
                    k_buff_id, phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_fulls[k_buff_id], phase)
                    k_tile = tlx.local_trans(kv_tiles[k_buff_id])
                    qk_id_prev = qk_id
                    p_phase_prev = p_phase
                    qk_id, p_phase = _get_bufidx_phase(qk_cnt, NUM_MMA_GROUPS)
                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[qk_id],
                        use_acc=False,
                        mBarriers=[qk_fulls[qk_id], kv_empties[k_buff_id]],
                    )

                    # -- compute p(i-1) @ v ----
                    # wait for the V buffer to be populated by the producer
                    kv_cnt += 1
                    v_buf_id, phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_fulls[v_buf_id], phase)
                    tlx.barrier_wait(p_fulls[qk_id_prev], p_phase_prev)
                    # Use p[0] for cid=0, and p[2] for cid=1
                    tlx.async_dot(
                        p_tiles[qk_id_prev * 2],
                        kv_tiles[v_buf_id],
                        acc_tiles[0],
                        use_acc=acc_pv,
                        mBarriers=[
                            kv_empties[v_buf_id],
                        ],
                    )
                    acc_pv = True
                # -- compute p(i) @ v ----
                kv_cnt += 1
                v_buf_id, phase = _get_bufidx_phase(kv_cnt, NUM_BUFFERS_KV)
                tlx.barrier_wait(kv_fulls[v_buf_id], phase)
                tlx.barrier_wait(p_fulls[qk_id], p_phase)
                # Use p[0] for cid=0, and p[2] for cid=1
                tlx.async_dot(
                    p_tiles[qk_id * 2],
                    kv_tiles[v_buf_id],
                    acc_tiles[0],
                    use_acc=acc_pv,
                    mBarriers=[
                        acc_fulls[0],
                        kv_empties[v_buf_id],
                    ],
                )

        # load
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            start_m, off_h, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (_compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets))
            if start_m < seq_len_q:
                # load q: it will stay in SRAM throughout
                tlx.barrier_expect_bytes(q_fulls[0], 2 * BLOCK_M * HEAD_DIM)  # float16
                qo_offset_y_split = seq_start_q + start_m
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[0],
                    [qo_offset_y_split.to(tl.int32), off_h * stride_qh],
                    q_fulls[0],
                )
                # load k0
                accum_cnt = 0
                k_buff_id, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV)
                k_tile = tlx.local_view(kv_tiles, k_buff_id)
                tlx.barrier_expect_bytes(kv_fulls[k_buff_id], 2 * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(
                    desc_k,
                    k_tile,
                    [seq_start_kv.to(tl.int32), off_h * stride_kh],
                    kv_fulls[k_buff_id],
                )
                for _ in tl.range(BLOCK_N, seq_len_kv, BLOCK_N):
                    accum_cnt += 1
                    seq_start_kv += BLOCK_N
                    k_buff_id, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_empties[k_buff_id], phase ^ 1)
                    # load k(i)
                    k_tile = tlx.local_view(kv_tiles, k_buff_id)
                    tlx.barrier_expect_bytes(kv_fulls[k_buff_id], 2 * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [seq_start_kv.to(tl.int32), off_h * stride_kh],
                        kv_fulls[k_buff_id],
                    )
                    # load v(i - 1)
                    accum_cnt += 1
                    v_buf_id, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_empties[v_buf_id], phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_buf_id)
                    v_tile = tlx.local_view(kv_tiles, v_buf_id)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D_V)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [(seq_start_kv - BLOCK_N).to(tl.int32), off_h * stride_vh],
                        v_full,
                    )
                # load last V
                accum_cnt += 1
                v_buf_id, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV)
                tlx.barrier_wait(kv_empties[v_buf_id], phase ^ 1)
                v_full = tlx.local_view(kv_fulls, v_buf_id)
                v_tile = tlx.local_view(kv_tiles, v_buf_id)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D_V)  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tile,
                    [seq_start_kv.to(tl.int32), off_h * stride_vh],
                    v_full,
                )


@triton.autotune(configs=get_fwd_pipeline_configs(), key=["Z", "HEAD_DIM", "AUTOTUNE_MAX_Q_LEN"])
@triton.jit
def _attn_fwd_pipeline(alpha, Z, H, desc_q, desc_k, desc_v, Out, seq_offsets_q, seq_offsets, max_seq_len, stride_qh,
                       stride_kh, stride_vh, stride_om, stride_oh, AUTOTUNE_MAX_Q_LEN: tl.constexpr,
                       HEAD_DIM: tl.constexpr,  #
                       BLOCK_D_V: tl.constexpr, BLOCK_M: tl.constexpr,  #
                       BLOCK_N: tl.constexpr,  #
                       NUM_MMA_GROUPS: tl.constexpr,  #
                       NUM_BUFFERS_KV: tl.constexpr,  #
                       ):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )
    # p_tiles is in bf16/fp6, when reusing qk_tiles which is fp32,
    # we need to create 2xNUM_MMA_GROUPS of p_tiles and use the
    # lower half for p1 so that  so that
    # q0k won't overwrite p1.
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D_V),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            # initialize offsets
            start_m, off_h, kv_offset_y, seq_len_kv, seq_start_q, seq_len_q = (_compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets))
            if start_m < seq_len_q:
                for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    tlx.barrier_wait(acc_empties[cid], 0)
                    # epilogue
                    off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                    offs_m = start_m + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    offs_v_d = tl.arange(0, BLOCK_D_V)
                    out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
                    acc = tlx.local_load(acc_tiles[cid])
                    acc = acc / max_seq_len
                    acc = acc.to(tlx.dtype_of(desc_v))
                    tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])
        # silu groups
        with tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS):
            # initialize offsets
            start_m, off_h, kv_offset_y, seq_len_kv, seq_start_q, seq_len_q = (_compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets))
            if start_m < seq_len_q:
                accum_cnt_qk = 0
                cid = tlx.async_task_replica_id()
                for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                    tlx.barrier_wait(qk_fulls[cid], accum_cnt_qk & 1)
                    qk = tlx.local_load(qk_tiles[cid])

                    offs_m = start_m + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                    valid_mask = forward_valid_mask(
                        offs_m,
                        offs_n,
                        0,  # uih_len_q
                        seq_len_q,
                        seq_len_kv,
                        False,
                    )
                    masked_alpha = tl.where(valid_mask, alpha, 0.0)
                    qk = qk * masked_alpha
                    # silu = fast_dividef(qk, 1.0 + tl.exp(-qk))
                    silu = fast_silu(qk)
                    act_qk = silu.to(tlx.dtype_of(desc_v))
                    tlx.local_store(p_tiles[cid * 2], act_qk)
                    tlx.barrier_arrive(p_fulls[cid])
                    accum_cnt_qk += 1
        # mma group
        with tlx.async_task(num_warps=1, registers=32):
            start_m, off_h, kv_offset_y, seq_len_kv, seq_start_q, seq_len_q = (_compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets))
            if start_m < seq_len_q:
                # compute q0 @ k
                # wait for the Q buffer to be populated by the producer
                accum_cnt_kv = 0
                kv_buf_id, kv_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                tlx.barrier_wait(q_fulls[0], 0)
                tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                k_tile = tlx.local_trans(k_tiles[kv_buf_id])
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )
                # compute q1 @ k
                tlx.barrier_wait(q_fulls[1], 0)
                tlx.async_dot(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    use_acc=False,
                    mBarriers=[qk_fulls[1], k_empties[0]],
                )

                # compute p0 @ v
                tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(p_fulls[0], 0)
                tlx.async_dot(
                    p_tiles[0],
                    v_tiles[kv_buf_id],
                    acc_tiles[0],
                    use_acc=False,
                )

                # loop over k, v and update accumulator
                accum_cnt_kv += 1
                acc1 = False
                phase = 1
                for _ in tl.range(BLOCK_N, seq_len_kv, BLOCK_N):
                    # -- compute q0 @ k ----
                    # wait for the K buffer to be populated by the producer
                    kv_buf_id_prev = kv_buf_id
                    kv_buf_id, kv_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                    k_tile = tlx.local_trans(k_tiles[kv_buf_id])
                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )
                    # compute p1 @ v
                    tlx.barrier_wait(p_fulls[1], phase ^ 1)
                    tlx.async_dot(
                        p_tiles[2],
                        v_tiles[kv_buf_id_prev],
                        acc_tiles[1],
                        use_acc=acc1,
                        mBarriers=[v_empties[kv_buf_id_prev]],
                    )
                    acc1 = True
                    # compute q1 @ k
                    tlx.async_dot(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        use_acc=False,
                        mBarriers=[qk_fulls[1], k_empties[kv_buf_id]],
                    )

                    # compute p0 @ v
                    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)
                    tlx.barrier_wait(p_fulls[0], phase)
                    tlx.async_dot(
                        p_tiles[0],
                        v_tiles[kv_buf_id],
                        acc_tiles[0],
                        use_acc=True,
                    )
                    phase = phase ^ 1
                    accum_cnt_kv += 1
                tlx.tcgen05_commit(acc_empties[0])
                # compute p1 @ v
                tlx.barrier_wait(p_fulls[1], phase ^ 1)
                tlx.async_dot(
                    p_tiles[2],
                    v_tiles[kv_buf_id],
                    acc_tiles[1],
                    use_acc=acc1,
                    mBarriers=[acc_empties[1], v_empties[kv_buf_id]],
                )

        # load
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            start_m, off_h, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (_compute_offsets(
                H, BLOCK_M, seq_offsets_q, seq_offsets))
            if start_m < seq_len_q:
                # load q: it will stay in SRAM throughout
                # load Q0
                tlx.barrier_expect_bytes(q_fulls[0], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                q_offset_split = seq_start_q + start_m
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[0],
                    [q_offset_split.to(tl.int32), off_h * stride_qh],
                    q_fulls[0],
                )
                # load K
                accum_cnt_kv = 0
                k_buff_id, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                k_full = tlx.local_view(k_fulls, k_buff_id)
                k_tile = tlx.local_view(k_tiles, k_buff_id)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(
                    desc_k,
                    k_tile,
                    [seq_start_kv.to(tl.int32), off_h * stride_kh],
                    k_full,
                )
                # load Q1
                tlx.barrier_expect_bytes(q_fulls[1], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                q_offset_split = seq_start_q + start_m + BLOCK_M_SPLIT
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[1],
                    [q_offset_split.to(tl.int32), off_h * stride_qh],
                    q_fulls[1],
                )

                # load V
                v_buf_id, v_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_full = tlx.local_view(v_fulls, v_buf_id)
                v_tile = tlx.local_view(v_tiles, v_buf_id)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D_V)  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tile,
                    [seq_start_kv.to(tl.int32), off_h * stride_vh],
                    v_full,
                )

                accum_cnt_kv += 1
                # loop over loading k, v
                for _ in tl.range(BLOCK_N, seq_len_kv, BLOCK_N):
                    seq_start_kv += BLOCK_N
                    # wait for the K buffer to be released by the consumer
                    kv_buf_id, kv_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    k_empty = tlx.local_view(k_empties, kv_buf_id)
                    tlx.barrier_wait(k_empty, kv_phase ^ 1)
                    # load K
                    k_full = tlx.local_view(k_fulls, kv_buf_id)
                    k_tile = tlx.local_view(k_tiles, kv_buf_id)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [seq_start_kv.to(tl.int32), off_h * stride_kh],
                        k_full,
                    )

                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(v_empties, kv_buf_id)
                    tlx.barrier_wait(v_empty, kv_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(v_fulls, kv_buf_id)
                    v_tile = tlx.local_view(v_tiles, kv_buf_id)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D_V)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [seq_start_kv.to(tl.int32), off_h * stride_vh],
                        v_full,
                    )

                    accum_cnt_kv += 1


def triton_hstu_cross_attn_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    Z = seq_offsets.numel() - 1
    # Previously this is AUTOTUNE_Z=prev_power_of_2(Z)
    # We rollback to Z to avoid the .item() call in prev_power_of_2
    # TODO: remove this once we have a better way to handle the .item() call
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.zeros(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        return out

    total_seq_len_kv, _, _ = k.shape
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(
        q,
        shape=[total_seq_len_q, H * DimQ],
        strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        v,
        shape=[total_seq_len_kv, H * DimV],
        strides=[H * DimV, 1],
        block_shape=dummy_block,
    )
    desc_k = TensorDescriptor(
        k,
        shape=[total_seq_len_kv, H * DimQ],
        strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_q1 = TensorDescriptor(
        q,
        shape=[total_seq_len_q, H * DimQ],
        strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_v1 = TensorDescriptor(
        v,
        shape=[total_seq_len_kv, H * DimV],
        strides=[H * DimV, 1],
        block_shape=dummy_block,
    )
    desc_k1 = TensorDescriptor(
        k,
        shape=[total_seq_len_kv, H * DimQ],
        strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]),
        Z * H,
    )
    #variant = "triton"  # "triton", "tlx_single_q", "triton_dyn_spec", "tlx_pipeline"
    if variant == "triton":
        _attn_fwd_triton[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            Out=out,
            stride_qh=q.stride(1),
            stride_kh=k.stride(1),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            AUTOTUNE_MAX_Q_LEN=triton.next_power_of_2(max_q_len),
            HEAD_DIM=DimQ,  #
            BLOCK_D_V=DimV,
        )
    if variant == "triton_dyn_spec":
        _attn_fwd_triton_spec[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            desc_q1=desc_q1,
            desc_k1=desc_k1,
            desc_v1=desc_v1,
            Out=out,
            stride_qh=q.stride(1),
            stride_kh=k.stride(1),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            AUTOTUNE_MAX_Q_LEN=triton.next_power_of_2(max_q_len),
            HEAD_DIM=DimQ,  #
            BLOCK_D_V=DimV,
        )
    if variant == "tlx_single_q":
        _attn_fwd_single_q[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            Out=out,
            stride_qh=q.stride(1),
            stride_kh=k.stride(1),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            AUTOTUNE_MAX_Q_LEN=triton.next_power_of_2(max_q_len),
            HEAD_DIM=DimQ,  #
            BLOCK_D_V=DimV,
        )
    if variant == "tlx_pipeline":
        _attn_fwd_pipeline[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            Out=out,
            stride_qh=q.stride(1),
            stride_kh=k.stride(1),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            AUTOTUNE_MAX_Q_LEN=triton.next_power_of_2(max_q_len),
            HEAD_DIM=DimQ,  #
            BLOCK_D_V=DimV,
        )
    return out


class AttentionFunction(torch.autograd.Function):

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_offsets_q: torch.Tensor,
        max_q_len: int,
        variant: str,
    ):
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)
        # Z = seq_offsets.numel() - 1
        total_seq_len_q, H, DimQ = q.shape
        _, _, DimV = v.shape
        if total_seq_len_q == 0:
            out = torch.zeros(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
            return out

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}

        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        out = triton_hstu_cross_attn_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            max_q_len=max_q_len,
            attn_scale=torch.tensor(1.0, device=q.device, dtype=torch.float32),
            variant=variant,
        )

        ctx.save_for_backward(q, k, v, seq_offsets, None, seq_offsets_q)
        ctx.alpha = alpha
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        return out


def triton_bw_hstu_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    variant: str,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    return AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        max_q_len,
        variant,
    )


def hstu_cross_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    attn_scale: torch.Tensor,
    variant: str,
    sort_by_length: bool = False,
    enable_tma: bool = False,
    num_softmax_heads: int = -1,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    max_q_len: Optional[int] = None,
    training: bool = True,
    targets_in_kv: bool = False,
) -> torch.Tensor:
    max_q_len = max_seq_len if max_q_len is None else max_q_len
    _, H, _ = q.shape
    if num_softmax_heads < 0:
        num_softmax_heads = H
    assert targets_in_kv is False, "targets_in_kv not implemented for TLX path"
    return triton_bw_hstu_mha_wrapper(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        variant=variant,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
    )


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    if sparsity == 0.0:
        return torch.zeros(size=(size, ), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size, ), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size, ),
            device=device,
            dtype=torch.int,
        )
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size, ),
            device=device,
            dtype=torch.int,
        )


dtype = torch.bfloat16
seq_sparsity = 0.95
batch_size = 1600
heads = 2


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
@pytest.mark.parametrize("max_uih_len_kv", [1024, 2048])
@pytest.mark.parametrize("max_targets", [32, 128, 160, 256])
def test_op(max_uih_len_kv, max_targets):
    torch.manual_seed(1001)  # for reproducibility
    num_softmax_heads = 0
    attn_dim = 128
    hidden_dim = 128
    sparsity = seq_sparsity
    max_uih_len_q = 0
    has_targets = True
    enable_tma = False
    causal = False

    alpha = 1.0 / (attn_dim**0.5)
    if sparsity > 0.0:
        lengths_kv = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len_kv,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        lengths_kv = torch.randint(1, max_uih_len_kv + 1, size=(batch_size, ), device=torch.device("cuda"))
    uih_lengths_q = torch.where(lengths_kv >= max_uih_len_q, max_uih_len_q, lengths_kv)
    num_targets = torch.randint(
        1,
        max_targets + 1,
        size=(batch_size, ),
        device=torch.device("cuda"),
    )
    max_seq_len = max_uih_len_kv + (max_targets if has_targets else 0)
    seq_offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
    seq_offsets[1:] = torch.cumsum(lengths_kv, dim=0)
    seq_offsets_q = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
    if has_targets:
        seq_offsets_q[1:] = torch.cumsum(uih_lengths_q + num_targets, dim=0)
    else:
        seq_offsets_q[1:] = torch.cumsum(uih_lengths_q, dim=0)
    total_seq_len_q = int(seq_offsets_q[-1].item())
    total_seq_len_kv = int(seq_offsets[-1].item())
    q = torch.empty((total_seq_len_q, heads, attn_dim), dtype=dtype, device=torch.device("cuda")).uniform_(-0.1, 0.1)
    k = torch.empty(
        (total_seq_len_kv, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)
    v = torch.empty(
        (total_seq_len_kv, heads, hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)

    fn = lambda: hstu_cross_mha(
        max_seq_len=max_seq_len,
        max_q_len=max_uih_len_q + (max_targets if has_targets else 0),
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        num_softmax_heads=num_softmax_heads,
        attn_scale=torch.tensor(1.0 / max_seq_len).to(q.device),
        variant="triton_dyn_spec",  # triton_dyn_spec or triton
        causal=causal,
        num_targets=num_targets if has_targets else None,
    )
    ref_out = fn()
    fn2 = lambda: hstu_cross_mha(
        max_seq_len=max_seq_len,
        max_q_len=max_uih_len_q + (max_targets if has_targets else 0),
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        num_softmax_heads=num_softmax_heads,
        attn_scale=torch.tensor(1.0 / max_seq_len).to(q.device),
        variant="tlx_single_q",
        causal=causal,
        num_targets=num_targets if has_targets else None,
    )
    tri_out = fn2()
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


line_vals = ["triton", "triton_dyn_spec", "tlx_single_q"]
line_names = ["Triton", "DynSpec", "tlx"]
modes = ["fwd"]
configs: List[triton.testing.Benchmark] = [
    triton.testing.Benchmark(
        x_names=["max_uih_len_kv"],
        x_vals=[1024, 2048, 4096, 6144],  # shape for IGR LSR
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=
        f"hstu-attn-b{batch_size}-h{heads}-d{attn_dim}-v{hidden_dim}--sparsity{seq_sparsity}-{mode}-softmax{num_softmax_heads}-{dtype}-max_uih_len_q{max_uih_len_q}-max_targets{max_targets}",
        args={
            "mode": mode,
            "batch_size": batch_size,
            "heads": heads,
            "attn_dim": attn_dim,
            "hidden_dim": hidden_dim,
            "dtype": dtype,
            "sparsity": seq_sparsity,
            "bench_backward": False,  #bench_backward,
            "max_uih_len_q": max_uih_len_q,
            "max_targets": max_targets,
        },
    )
    for mode in modes
    for max_uih_len_q in [0]
    for max_targets in [32, 128, 160, 256]
    for num_softmax_heads in [0]
    for attn_dim in [128]
    for hidden_dim in [128]
]


@triton.testing.perf_report(configs)
def bench_cross_attention(
    mode: str,
    provider: str,
    bench_backward: bool,
    batch_size: int,
    heads: int,
    max_uih_len_kv: int,
    max_uih_len_q: int,
    attn_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    max_targets: int = 80,
    sparsity: float = -1.0,
    enable_tma: bool = False,
    num_softmax_heads: int = 0,
    causal: bool = False,
    has_targets: bool = True,
) -> float:
    assert mode in ["fwd", "bwd"]
    warmup = 25  #2000 25
    rep = 1000  #2000 1000
    torch.manual_seed(1001)  # for reproducibility

    alpha = 1.0 / (attn_dim**0.5)
    if sparsity > 0.0:
        lengths_kv = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len_kv,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        lengths_kv = torch.randint(1, max_uih_len_kv + 1, size=(batch_size, ), device=torch.device("cuda"))
    uih_lengths_q = torch.where(lengths_kv >= max_uih_len_q, max_uih_len_q, lengths_kv)
    num_targets = torch.randint(
        1,
        max_targets + 1,
        size=(batch_size, ),
        device=torch.device("cuda"),
    )
    max_seq_len = max_uih_len_kv + (max_targets if has_targets else 0)
    seq_offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
    seq_offsets[1:] = torch.cumsum(lengths_kv, dim=0)
    seq_offsets_q = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
    if has_targets:
        seq_offsets_q[1:] = torch.cumsum(uih_lengths_q + num_targets, dim=0)
    else:
        seq_offsets_q[1:] = torch.cumsum(uih_lengths_q, dim=0)
    total_seq_len_q = int(seq_offsets_q[-1].item())
    total_seq_len_kv = int(seq_offsets[-1].item())
    q = torch.empty((total_seq_len_q, heads, attn_dim), dtype=dtype, device=torch.device("cuda")).uniform_(-0.1, 0.1)
    k = torch.empty(
        (total_seq_len_kv, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)
    v = torch.empty(
        (total_seq_len_kv, heads, hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)

    fn = lambda: hstu_cross_mha(
        max_seq_len=max_seq_len,
        max_q_len=max_uih_len_q + (max_targets if has_targets else 0),
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        num_softmax_heads=num_softmax_heads,
        attn_scale=torch.tensor(1.0 / max_seq_len).to(q.device),
        variant=provider,
        causal=causal,
        num_targets=num_targets if has_targets else None,
    )
    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    if is_blackwell():
        print("Running benchmarks...")
        bench_cross_attention.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
