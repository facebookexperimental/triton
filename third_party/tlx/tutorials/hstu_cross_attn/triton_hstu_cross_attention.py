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

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from stubs import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from stubs import acc_dq
from stubs import maybe_register_custom_op
from stubs import NamedSpecType, tritoncc_specs, VersionedSpec
from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef,
    fast_expf,
)

HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
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


def _get_named_specs() -> List[VersionedSpec]:
    s: int = 16

    def _common_specs(
        dtype: str = "*bf16", attn_scale_dtype: str = "*fp32"
    ) -> NamedSpecType:
        return {
            "Q": (dtype, s),
            "K": (dtype, s),
            "V": (dtype, s),
            "workspace_ptr": (dtype, s),
            "sort_by_length_indices": ("*i64", s),
            "seq_offsets": ("*i64", s),
            "seq_offsets_q": ("*i64", s),
            "Out": (dtype, s),
            "alpha": "fp32",
            "stride_qm": ("i32", s),
            "stride_qh": ("i32", s),
            "stride_kn": ("i32", s),
            "stride_kh": ("i32", s),
            "stride_vn": ("i32", s),
            "stride_vh": ("i32", s),
            "stride_om": ("i32", s),
            "stride_oh": ("i32", s),
            "Z": "i32",
            "AUTOTUNE_Z": "i32",
            "H": "i32",
            "attn_scale": (attn_scale_dtype, s, True),
            "AUTOTUNE_MAX_SEQ_LEN": "i32",
            "DimQ": "i32",
            "DimV": "i32",
            "M": ("*fp32", s),
            "stride_mm": "i32",
            "num_softmax_heads": "i32",
            "num_targets": ("*i64", s),
        }

    return [
        VersionedSpec(
            spec={
                "causal": CAUSAL,
                "HAS_NUM_TARGETS": HAS_NUM_TARGETS,
                "HAS_CAUSAL": CAUSAL,
                "ATTN_SCALE_TYPE": ATTN_SCALE_TYPE,
                "ALLOW_TF32": ALLOW_TF32,
                "BLOCK_D_Q": BLOCK_D_Q,
                "BLOCK_D_V": BLOCK_D_V,
                "BLOCK_M": -1,
                "BLOCK_N": -1,
                "HAS_SORT_BY_LENGTH_INDICES": HAS_SORT_BY_LENGTH_INDICES,
                "ENABLE_TMA": ENABLE_TMA,
                "TMA_DESC_SIZE": TMA_DESC_SIZE,
                "G": "i32",
                **_common_specs(dtype=dtype, attn_scale_dtype=attn_scale_dtype),
            },
        )
        for CAUSAL in [True, False]
        for HAS_NUM_TARGETS in [True, False]
        for ATTN_SCALE_TYPE in ["scalar", "dynamic"]
        for ALLOW_TF32 in [True, False]
        for BLOCK_D_Q in [64, 128]
        for BLOCK_D_V in [64, 128]
        for HAS_SORT_BY_LENGTH_INDICES in [False]
        for dtype in ["*bf16", "*fp32", "*fp16"]
        for ENABLE_TMA in [False]
        for TMA_DESC_SIZE in [0]
        for attn_scale_dtype in ["*fp32", "*bf16", "*fp16"]
    ]


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
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
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                            "waves_per_eu": waves_per_eu,
                                            "kpack": 2,
                                        },
                                        num_stages=num_stages,
                                        num_warps=num_warps,
                                    )
                                )
    else:
        configs = [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
        ]
    return configs


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    # GQA: When G > 1, multiple Q heads accumulate gradients into same KV head
    # so dk/dv must be zero-initialized for atomic accumulation
    if nargs["G"] > 1:
        nargs["DK"].zero_()
        nargs["DV"].zero_()
    if nargs["SEQUENCE_PARALLEL"] is True:
        nargs["LOCK"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2, 4]:
                                for sp in [True, False]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "SEQUENCE_PARALLEL": sp,
                                                "UNROLL": 1,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_pre_hook,
                                        )
                                    )
        return configs

    configs = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 4},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
    ]
    if torch.cuda.is_available():
        configs += [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=3,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=2,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 128,
                    "SEQUENCE_PARALLEL": False,
                    "UNROLL": 2,
                },
                num_stages=2,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
        ]
    else:
        print("WARNING: temporarily disabled some autotune configs for CUDA 12.8+")
    return configs


def _bwd_two_phases_pre_hook(nargs):
    nargs["DQ"].zero_()
    # GQA: When G > 1, multiple Q head groups write to the same KV head's
    # dK/dV via tl.atomic_add, so dk/dv must be zero-initialized
    if nargs["GQA_ATOMIC_ADD"]:
        nargs["DK"].zero_()
        nargs["DV"].zero_()


def _get_bw_two_phases_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        # Triton TR001: sweep BLOCK_KV (Phase-2 KV inner-loop block) over
        # {32, 64, 128}, decoupled from BLOCK_M. waves_per_eu trimmed from
        # [0, 2, 4] to [0, 2] so the added BLOCK_KV dim keeps the total config
        # count reasonable for autotuning.
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64, 128]:
                for BLOCK_KV in [32, 64, 128]:
                    for num_stages in [1, 2]:
                        for num_warps in [4, 8]:
                            for matrix_instr_nonkdim in [16]:
                                for waves_per_eu in [0, 2]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "BLOCK_KV": BLOCK_KV,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "kpack": 2,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_two_phases_pre_hook,
                                        )
                                    )
        return configs
    # Triton TR001: off-HIP fallback defaults BLOCK_KV to BM, preserving the
    # prior behavior where the Phase-2 KV inner loop stepped by BLOCK_M.
    configs = [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_KV": BM},
            num_stages=ns,
            num_warps=nw,
            pre_hook=_bwd_two_phases_pre_hook,
        )
        for BM in [16, 32, 64]
        for BN in [32, 64]
        for ns in [1, 2]
        for nw in [4, 8]
    ]
    return configs


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
def backward_common_preprocess(
    M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
):
    if off_h < num_softmax_heads:
        M_off = M + off_h + seq_start_q * stride_mm
        Delta_off = Delta + off_h + seq_start_q * stride_mm
    else:  # placeholder
        M_off = M
        Delta_off = Delta
    return M_off, Delta_off


def backward_custom_vars(
    dout: torch.Tensor,
    num_softmax_heads: int,
    idx: int,
    saved_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    M = saved_tensors[idx]
    idx += 1
    out = saved_tensors[idx]
    Delta = torch.empty_like(M)
    pre_grid = (triton.cdiv(out.shape[0], 128), num_softmax_heads)
    _attn_bwd_preprocess[pre_grid](
        out,
        dout,
        Delta,
        out.shape[0],
        H=out.shape[1],
        softmax_heads=num_softmax_heads,
        BLOCK_M=128,  # pyre-ignore [6]
        HEAD_DIM=out.shape[2],  # pyre-ignore [6]
    )
    stride_mm = M.stride(0)
    return M, Delta, stride_mm


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
def backward_d_softmax_activation(
    dact_qk_trans,
    Delta_off,
    offs_m,
    stride_mm,
    mask_m,
    pT,  # pT represents sig_trans
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
def backward_softmax_activation(
    qk_trans, alpha, valid_mask_trans, M_off, offs_m, stride_mm, mask_m, k
):
    qk_trans = qk_trans * alpha * 1.44269504
    m = tl.load(M_off + offs_m * stride_mm, mask=mask_m)
    pT = tl.math.exp2(qk_trans - m[None, :])
    pT = tl.where(valid_mask_trans, pT, 0.0)
    act_qk_trans = pT.to(k.dtype)
    return qk_trans, act_qk_trans, pT


@triton.jit
def backward_valid_mask(
    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr
):
    valid_mask_trans = (offs_m[None, :] < seq_len_q) & (offs_n[:, None] < seq_len_kv)
    if HAS_CAUSAL:
        offs_m = offs_m + seq_len_kv - uih_len_q
        causal_mask_trans = offs_m[None, :] >= offs_n[:, None]
        valid_mask_trans = valid_mask_trans & causal_mask_trans
    return valid_mask_trans


def forward_custom_vars(
    q: torch.Tensor, num_softmax_heads: int, saved_tensors: List[torch.Tensor]
) -> Tuple[torch.Tensor, int]:
    assert num_softmax_heads <= q.shape[1]
    M = torch.empty(
        (q.shape[0], num_softmax_heads), device=q.device, dtype=torch.float32
    )
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
def forward_silu_activation(qk, alpha, valid_mask, m_i, acc, l_i, scale):
    masked_alpha = tl.where(valid_mask, alpha, 0.0)
    qk = qk * masked_alpha
    silu = fast_silu(qk, MULT_BY_X=True) * scale[:, None]
    return silu, acc, l_i, m_i


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
def forward_valid_mask(
    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr
):
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


@triton.jit
def _hstu_attn_fwd_one_block_1(  # noqa: C901
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n,
    off_h,
    q,
    K,
    V,
    acc,
    K_block_ptr,
    V_block_ptr,
    device_desc_k,
    device_desc_v,
    offset_kh,
    offset_vh,
    seq_start_q,
    seq_start_kv,
    alpha,
    scale,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    n_targets,
    uih_len_q,
    m_i,
    l_i,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = None
    qk = None
    if ENABLE_TMA:
        k = tl._experimental_descriptor_load(
            device_desc_k,
            [(seq_start_kv + start_n).to(tl.int32), offset_kh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_Q],
            K.dtype.element_ty,
        )
        # tma can only be loaded in one order, use trans afterwards
        qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32)
    valid_mask = forward_valid_mask(
        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
    )
    act_qk, acc, l_i, m_i = forward_softmax_activation(
        qk, alpha, valid_mask, m_i, acc, l_i
    )
    v = None
    if ENABLE_TMA:
        v = tl._experimental_descriptor_load(
            device_desc_v,
            [(seq_start_kv + start_n).to(tl.int32), offset_vh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_V],
            V.dtype.element_ty,
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    act_qk = act_qk.to(v.dtype)
    acc += tl.dot(act_qk, v, allow_tf32=ALLOW_TF32)
    return acc, l_i, m_i


@triton.jit
def _hstu_attn_fwd_one_block_0(  # noqa: C901
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n,
    off_h,
    q,
    K,
    V,
    acc,
    K_block_ptr,
    V_block_ptr,
    device_desc_k,
    device_desc_v,
    offset_kh,
    offset_vh,
    seq_start_q,
    seq_start_kv,
    alpha,
    scale,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    n_targets,
    uih_len_q,
    m_i,
    l_i,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = None
    qk = None
    if ENABLE_TMA:
        k = tl._experimental_descriptor_load(
            device_desc_k,
            [(seq_start_kv + start_n).to(tl.int32), offset_kh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_Q],
            K.dtype.element_ty,
        )
        # tma can only be loaded in one order, use trans afterwards
        qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32)
    valid_mask = forward_valid_mask(
        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
    )
    act_qk, acc, l_i, m_i = forward_silu_activation(
        qk, alpha, valid_mask, m_i, acc, l_i, scale
    )
    v = None
    if ENABLE_TMA:
        v = tl._experimental_descriptor_load(
            device_desc_v,
            [(seq_start_kv + start_n).to(tl.int32), offset_vh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_V],
            V.dtype.element_ty,
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    act_qk = act_qk.to(v.dtype)
    acc += tl.dot(act_qk, v, allow_tf32=ALLOW_TF32)
    return acc, l_i, m_i


@triton.jit
def _hstu_attn_bwd_one_block_1(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    do_ptrs,
    device_desc_q,
    device_desc_do,
    dk,
    dv,
    k,
    v,
    seq_len_q,
    seq_len_kv,
    LOCK,
    off_h,
    stride_qh,
    stride_doh,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    attn_scale,
    max_q_len,
    M,
    Delta,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    n_targets,
    uih_len_q,
    M_off,
    Delta_off,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    offs_m = offs_m + start_m
    mask_m = offs_m < seq_len_q
    if ATTN_SCALE_TYPE == "scalar":
        scale = tl.load(attn_scale).to(tl.float32)
    else:
        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
        scale = tl.load(attn_scale + offs_m, mask=mask_m).to(tl.float32)
    # recompute qk and silu
    if ENABLE_TMA:
        q = tl._experimental_descriptor_load(
            device_desc_q,
            [start_m, (off_h * stride_qh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_Q],
            k.dtype,
        )
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(
            q_ptrs_trans + start_m * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
    valid_mask_trans = backward_valid_mask(
        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
    )
    qk_trans, act_qk_trans, pT = backward_softmax_activation(
        qk_trans, alpha, valid_mask_trans, M_off, offs_m, stride_mm, mask_m, k
    )
    # compute dv
    if ENABLE_TMA:
        do = tl._experimental_descriptor_load(
            device_desc_do,
            [start_m, (off_h * stride_doh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_V],
            k.dtype,
        )
    else:
        do = tl.load(
            do_ptrs + start_m * stride_dom,
            mask=mask_m[:, None],
            other=0.0,
        )
    dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = backward_d_softmax_activation(
        dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
    )
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(
        dq_ptrs_trans=dq_ptrs_trans,
        start_m=start_m,
        stride_dqm=stride_dqm,
        k=k,
        dqk_trans=dqk_trans,
        alpha=alpha,
        mask_m=mask_m,
        MAX_SEQ_LEN=max_q_len,
        LOCK=LOCK,
        BLOCK_M=BLOCK_M,
        ATOMIC_ADD=ATOMIC_ADD,
        ALLOW_TF32=ALLOW_TF32,
    )
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_block_0(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    do_ptrs,
    device_desc_q,
    device_desc_do,
    dk,
    dv,
    k,
    v,
    seq_len_q,
    seq_len_kv,
    LOCK,
    off_h,
    stride_qh,
    stride_doh,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    attn_scale,
    max_q_len,
    M,
    Delta,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    n_targets,
    uih_len_q,
    M_off,
    Delta_off,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    offs_m = offs_m + start_m
    mask_m = offs_m < seq_len_q
    if ATTN_SCALE_TYPE == "scalar":
        scale = tl.load(attn_scale).to(tl.float32)
    else:
        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
        scale = tl.load(attn_scale + offs_m, mask=mask_m).to(tl.float32)
    # recompute qk and silu
    if ENABLE_TMA:
        q = tl._experimental_descriptor_load(
            device_desc_q,
            [start_m, (off_h * stride_qh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_Q],
            k.dtype,
        )
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(
            q_ptrs_trans + start_m * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
    valid_mask_trans = backward_valid_mask(
        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
    )
    qk_trans, act_qk_trans, pT = backward_silu_activation(
        qk_trans, alpha, valid_mask_trans, k.dtype, scale
    )
    # compute dv
    if ENABLE_TMA:
        do = tl._experimental_descriptor_load(
            device_desc_do,
            [start_m, (off_h * stride_doh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_V],
            k.dtype,
        )
    else:
        do = tl.load(
            do_ptrs + start_m * stride_dom,
            mask=mask_m[:, None],
            other=0.0,
        )
    dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = backward_d_silu_activation(
        dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
    )
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(
        dq_ptrs_trans=dq_ptrs_trans,
        start_m=start_m,
        stride_dqm=stride_dqm,
        k=k,
        dqk_trans=dqk_trans,
        alpha=alpha,
        mask_m=mask_m,
        MAX_SEQ_LEN=max_q_len,
        LOCK=LOCK,
        BLOCK_M=BLOCK_M,
        ATOMIC_ADD=ATOMIC_ADD,
        ALLOW_TF32=ALLOW_TF32,
    )
    return dk, dv


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    workspace_ptr,
    seq_offsets,
    seq_offsets_q,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    attn_scale,
    off_z,
    off_h,
    off_h_kv,
    pid,
    M,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    G: tl.constexpr,
):
    off_h = off_h.to(tl.int64)
    off_h_kv = off_h_kv.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_o = None
    H_kv = H // G
    if ENABLE_TMA:
        workspace_base = workspace_ptr + TMA_DESC_SIZE * 4 * (
            tl.program_id(1) + tl.program_id(0) * tl.num_programs(1)
        )
        device_desc_q = workspace_base
        device_desc_k = workspace_base + 1 * TMA_DESC_SIZE
        device_desc_v = workspace_base + 2 * TMA_DESC_SIZE
        device_desc_o = workspace_base + 3 * TMA_DESC_SIZE

        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_k,
            global_address=K,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_end_kv.to(tl.int32), H_kv * DimQ],
            element_ty=K.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_v,
            global_address=V,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_end_kv.to(tl.int32), H_kv * DimV],
            element_ty=V.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_k)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_v)

    start_m = pid * BLOCK_M
    if start_m < seq_len_q:
        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        if ATTN_SCALE_TYPE == "scalar":
            scale = tl.load(attn_scale).to(tl.float32)
        else:
            tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
            scale = tl.load(
                attn_scale + seq_start_q + offs_m, mask=offs_m < seq_len_q
            ).to(tl.float32)

        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + seq_start_q * stride_qm,
                shape=(seq_len_q, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

            K_block_ptr = tl.make_block_ptr(
                base=K + off_h_kv * stride_kh + seq_start_kv * stride_kn,
                shape=(BLOCK_D_Q, seq_len_kv),
                strides=(1, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_D_Q, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=V + off_h_kv * stride_vh + seq_start_kv * stride_vn,
                shape=(seq_len_kv, BLOCK_D_V),
                strides=(stride_vn, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D_V),
                order=(1, 0),
            )
        else:
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                # pyrefly: ignore [bad-argument-type]
                desc_ptr=device_desc_q,
                global_address=Q,
                load_size=[
                    BLOCK_M,
                    BLOCK_D_Q,
                ],
                global_size=[seq_end_q.to(tl.int32), H * DimQ],
                element_ty=Q.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)

            q = tl._experimental_descriptor_load(
                device_desc_q,
                [
                    (seq_start_q + start_m).to(tl.int32),
                    (off_h * stride_qh).to(tl.int32),
                ],
                [
                    BLOCK_M,
                    BLOCK_D_Q,
                ],
                Q.dtype.element_ty,
            )
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        end_n = 0
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        m_i, l_i = forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M)
        if off_h + 1 < num_softmax_heads + 1:  # For AOTInductor lowering walkaround.
            low = 0
            offset = low
            if HAS_CAUSAL:
                low = 0
                high = start_m + BLOCK_M + seq_len_kv - uih_len_q
                high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
            else:
                high = tl.cdiv(seq_len_kv, BLOCK_N) * BLOCK_N
            # single loop compute
            if offset > 0:
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
            end_n = low
            for start_n in tl.range(low, high, BLOCK_N):
                acc, l_i, m_i = _hstu_attn_fwd_one_block_1(
                    start_n=start_n,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    offs_m=offs_m,
                    offs_n=offs_n + start_n,
                    off_h=off_h,
                    q=q,
                    K=K,
                    V=V,
                    acc=acc,
                    K_block_ptr=K_block_ptr,
                    V_block_ptr=V_block_ptr,
                    device_desc_k=device_desc_k,
                    device_desc_v=device_desc_v,
                    offset_kh=off_h_kv * stride_kh,
                    offset_vh=off_h_kv * stride_vh,
                    seq_start_q=seq_start_q,
                    seq_start_kv=seq_start_kv,
                    alpha=alpha,
                    scale=scale,
                    num_softmax_heads=num_softmax_heads,
                    num_targets=num_targets,
                    causal=causal,
                    n_targets=n_targets,
                    uih_len_q=uih_len_q,
                    m_i=m_i,
                    l_i=l_i,
                    HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                    HAS_CAUSAL=HAS_CAUSAL,
                    ALLOW_TF32=ALLOW_TF32,
                    BLOCK_D_Q=BLOCK_D_Q,
                    BLOCK_D_V=BLOCK_D_V,
                    BLOCK_N=BLOCK_N,
                    ENABLE_TMA=ENABLE_TMA,
                )
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                end_n += BLOCK_N
        else:
            low = 0
            offset = low
            if HAS_CAUSAL:
                low = 0
                high = start_m + BLOCK_M + seq_len_kv - uih_len_q
                high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
            else:
                high = tl.cdiv(seq_len_kv, BLOCK_N) * BLOCK_N
            # single loop compute
            if offset > 0:
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
            end_n = low
            for start_n in tl.range(low, high, BLOCK_N):
                acc, l_i, m_i = _hstu_attn_fwd_one_block_0(
                    start_n=start_n,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    offs_m=offs_m,
                    offs_n=offs_n + start_n,
                    off_h=off_h,
                    q=q,
                    K=K,
                    V=V,
                    acc=acc,
                    K_block_ptr=K_block_ptr,
                    V_block_ptr=V_block_ptr,
                    device_desc_k=device_desc_k,
                    device_desc_v=device_desc_v,
                    offset_kh=off_h_kv * stride_kh,
                    offset_vh=off_h_kv * stride_vh,
                    seq_start_q=seq_start_q,
                    seq_start_kv=seq_start_kv,
                    alpha=alpha,
                    scale=scale,
                    num_softmax_heads=num_softmax_heads,
                    num_targets=num_targets,
                    causal=causal,
                    n_targets=n_targets,
                    uih_len_q=uih_len_q,
                    m_i=m_i,
                    l_i=l_i,
                    HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                    HAS_CAUSAL=HAS_CAUSAL,
                    ALLOW_TF32=ALLOW_TF32,
                    BLOCK_D_Q=BLOCK_D_Q,
                    BLOCK_D_V=BLOCK_D_V,
                    BLOCK_N=BLOCK_N,
                    ENABLE_TMA=ENABLE_TMA,
                )
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                end_n += BLOCK_N
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
        if not ENABLE_TMA:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start_q * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])
        else:
            # Important: must cast to proper dtype. If acc is float32, but
            # TMA descriptor specifies float16, the program will run
            # without crashes but produce wrong results.
            acc = acc.to(Out.dtype.element_ty)
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                # pyrefly: ignore [bad-argument-type]
                desc_ptr=device_desc_o,
                global_address=Out,
                load_size=[BLOCK_M, BLOCK_D_V],
                global_size=[seq_end_q.to(tl.int32), H * DimV],
                element_ty=Out.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_o)
            tl._experimental_descriptor_store(
                device_desc_o,
                acc,
                [
                    (seq_start_q + pid * BLOCK_M).to(tl.int32),
                    (off_h * stride_oh).to(tl.int32),
                ],
            )


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
@tritoncc_specs(versioned_specs=_get_named_specs())
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    seq_offsets_q,
    num_targets,
    Out,
    alpha,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    Z,
    AUTOTUNE_Z,
    H,
    attn_scale,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    M,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    causal: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    G,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h_kv = off_h // G
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        H=H,
        DimQ=DimQ,
        DimV=DimV,
        workspace_ptr=workspace_ptr,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        attn_scale=attn_scale,
        off_z=off_z,
        off_h=off_h,
        off_h_kv=off_h_kv,
        pid=pid,
        M=M,
        stride_mm=stride_mm,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_CAUSAL=HAS_CAUSAL,
        ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        ENABLE_TMA=ENABLE_TMA,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        G=G,
    )


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len_q,
    seq_len_kv,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    device_desc_q,
    device_desc_k,
    device_desc_v,
    device_desc_do,
    device_desc_dk,
    device_desc_dv,
    LOCK,
    off_h,
    off_h_kv,
    off_z,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dkh,
    stride_dvh,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    attn_scale,
    max_q_len,
    seq_start_q,
    M,
    Delta,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    if ENABLE_TMA:
        q_ptrs_trans = None
        do_ptrs = None
        k = tl._experimental_descriptor_load(
            device_desc_k,
            [start_n, (off_h_kv * stride_kh).to(tl.int32)],
            [BLOCK_N, BLOCK_D_Q],
            K.dtype.element_ty,
        )
        v = tl._experimental_descriptor_load(
            device_desc_v,
            [start_n, (off_h_kv * stride_vh).to(tl.int32)],
            [BLOCK_N, BLOCK_D_V],
            V.dtype.element_ty,
        )
    else:
        mask_n = offs_n < seq_len_kv
        q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
        do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    M_off, Delta_off = backward_common_preprocess(
        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
    )
    if off_h < num_softmax_heads:
        low = 0
        if HAS_CAUSAL:
            low = max(low, start_n + uih_len_q - seq_len_kv)
        high = seq_len_q
        for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block_1(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                do_ptrs=do_ptrs,
                device_desc_q=device_desc_q,
                device_desc_do=device_desc_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_doh=stride_doh,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                max_q_len=max_q_len,
                M=M,
                Delta=Delta,
                stride_mm=stride_mm,
                num_softmax_heads=num_softmax_heads,
                num_targets=num_targets,
                causal=causal,
                n_targets=n_targets,
                uih_len_q=uih_len_q,
                M_off=M_off,
                Delta_off=Delta_off,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_CAUSAL=HAS_CAUSAL,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                ATOMIC_ADD=ATOMIC_ADD,
                ENABLE_TMA=ENABLE_TMA,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )
    if off_h >= num_softmax_heads:
        low = 0
        if HAS_CAUSAL:
            low = max(low, start_n + uih_len_q - seq_len_kv)
        high = seq_len_q
        for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block_0(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                do_ptrs=do_ptrs,
                device_desc_q=device_desc_q,
                device_desc_do=device_desc_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_doh=stride_doh,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                max_q_len=max_q_len,
                M=M,
                Delta=Delta,
                stride_mm=stride_mm,
                num_softmax_heads=num_softmax_heads,
                num_targets=num_targets,
                causal=causal,
                n_targets=n_targets,
                uih_len_q=uih_len_q,
                M_off=M_off,
                Delta_off=Delta_off,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_CAUSAL=HAS_CAUSAL,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                ATOMIC_ADD=ATOMIC_ADD,
                ENABLE_TMA=ENABLE_TMA,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )
    # write-back
    dk = dk * alpha
    if ENABLE_TMA and not GQA_ATOMIC_ADD:
        tl._experimental_descriptor_store(
            device_desc_dv,
            dv.to(k.dtype),
            [start_n, (off_h_kv * stride_dvh).to(tl.int32)],
        )
        tl._experimental_descriptor_store(
            device_desc_dk,
            dk.to(k.dtype),
            [start_n, (off_h_kv * stride_dkh).to(tl.int32)],
        )
    else:
        if ENABLE_TMA:
            # TMA path with GQA needs to compute pointers with head offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv
            offs_v_d = tl.arange(0, BLOCK_D_V)
            offs_qk_d = tl.arange(0, BLOCK_D_Q)
            dv_ptrs = (
                DV
                + off_h_kv * stride_dvh
                + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
            )
            dk_ptrs = (
                DK
                + off_h_kv * stride_dkh
                + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            )
        else:
            # Non-TMA path: DV/DK pointers are already offset by off_h_kv in the caller
            dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
            dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        if GQA_ATOMIC_ADD:
            # GQA: Multiple Q heads write to the same KV head, need atomic accumulation
            tl.atomic_add(
                dv_ptrs,
                dv.to(k.dtype),
                mask=mask_n[:, None],  # pyre-ignore[61]
                sem="relaxed",
            )
            tl.atomic_add(
                dk_ptrs,
                dk.to(k.dtype),
                mask=mask_n[:, None],  # pyre-ignore[61]
                sem="relaxed",
            )
        else:
            tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    tma_workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    seq_offsets_q,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
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
    attn_scale,
    Z,
    AUTOTUNE_Z,
    H,
    G,
    max_q_len,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    M,
    Delta,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    off_h_kv = (off_h // G).to(tl.int64)
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
    # offset pointers for batch/head
    Q = Q + seq_start_q * stride_qm
    K = K + seq_start_kv * stride_kn
    V = V + seq_start_kv * stride_vn
    DOut = DOut + seq_start_q * stride_dom
    DQ = DQ + seq_start_q * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start_kv * stride_dkn
    DV = DV + seq_start_kv * stride_dvn
    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_do = None
    device_desc_dk = None
    device_desc_dv = None
    H_kv = H // G
    if ENABLE_TMA:
        workspace_base = tma_workspace_ptr + TMA_DESC_SIZE * 6 * (
            tl.program_id(1) + tl.program_id(0) * tl.num_programs(1)
        )
        device_desc_q = workspace_base
        device_desc_k = workspace_base + 1 * TMA_DESC_SIZE
        device_desc_v = workspace_base + 2 * TMA_DESC_SIZE
        device_desc_do = workspace_base + 3 * TMA_DESC_SIZE
        device_desc_dk = workspace_base + 4 * TMA_DESC_SIZE
        device_desc_dv = workspace_base + 5 * TMA_DESC_SIZE

        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_q,
            global_address=Q,
            load_size=[
                BLOCK_M,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_q, H * DimQ],
            element_ty=Q.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_do,
            global_address=DOut,
            load_size=[
                BLOCK_M,
                BLOCK_D_V,
            ],
            global_size=[seq_len_q, H * DimV],
            element_ty=DOut.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_do)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_k,
            global_address=K,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_kv, H_kv * DimQ],
            element_ty=K.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_k)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_dk,
            global_address=DK,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_kv, H_kv * DimQ],
            element_ty=DK.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_dk)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_v,
            global_address=V,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_len_kv, H_kv * DimV],
            element_ty=V.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_v)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_dv,
            global_address=DV,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_len_kv, H_kv * DimV],
            element_ty=DV.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_dv)
    else:
        Q += off_h * stride_qh
        K += off_h_kv * stride_kh
        V += off_h_kv * stride_vh
        DOut += off_h * stride_doh
        DK += off_h_kv * stride_dkh
        DV += off_h_kv * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len_kv:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            device_desc_q=device_desc_q,
            device_desc_k=device_desc_k,
            device_desc_v=device_desc_v,
            device_desc_do=device_desc_do,
            device_desc_dk=device_desc_dk,
            device_desc_dv=device_desc_dv,
            LOCK=LOCK,
            off_h=off_h,
            off_h_kv=off_h_kv,
            off_z=off_z,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_doh=stride_doh,
            stride_dkh=stride_dkh,
            stride_dvh=stride_dvh,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            attn_scale=attn_scale,
            max_q_len=max_q_len,
            seq_start_q=seq_start_q,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_softmax_heads=num_softmax_heads,
            num_targets=num_targets,
            causal=causal,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            HAS_CAUSAL=HAS_CAUSAL,
            ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
            GQA_ATOMIC_ADD=G > 1,
            ENABLE_TMA=ENABLE_TMA,
        )
    else:
        for start_n in range(0, seq_len_kv, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                device_desc_q=device_desc_q,
                device_desc_k=device_desc_k,
                device_desc_v=device_desc_v,
                device_desc_do=device_desc_do,
                device_desc_dk=device_desc_dk,
                device_desc_dv=device_desc_dv,
                LOCK=LOCK,
                off_h=off_h,
                off_h_kv=off_h_kv,
                off_z=off_z,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                stride_doh=stride_doh,
                stride_dkh=stride_dkh,
                stride_dvh=stride_dvh,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                attn_scale=attn_scale,
                max_q_len=max_q_len,
                seq_start_q=seq_start_q,
                M=M,
                Delta=Delta,
                stride_mm=stride_mm,
                num_softmax_heads=num_softmax_heads,
                num_targets=num_targets,
                causal=causal,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_CAUSAL=HAS_CAUSAL,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
                GQA_ATOMIC_ADD=G > 1,
                ENABLE_TMA=ENABLE_TMA,
            )


@maybe_register_custom_op("hammer::triton_hstu_cross_attn_v2_fwd", mutates_args=())
def triton_hstu_cross_attn_v2_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    M: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> torch.Tensor:
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        return out
    TMA_DESC_SIZE = 128
    workspace = None
    if enable_tma:
        MIN_BLOCK_M = 16
        workspace = torch.empty(
            4 * TMA_DESC_SIZE * (triton.cdiv(max_q_len, MIN_BLOCK_M) * Z * H),
            dtype=torch.uint8,
            device="cuda",
        )
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]),
        Z * H,
    )
    HAS_NUM_TARGETS = num_targets is not None
    HAS_CAUSAL = causal
    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        workspace_ptr=workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        Out=out,
        alpha=alpha,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        attn_scale=attn_scale,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        M=M,
        stride_mm=stride_mm,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_CAUSAL=HAS_CAUSAL,
        ATTN_SCALE_TYPE=attn_scale_type,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        G=G,
    )
    return out


@triton_autotune(
    configs=_get_bw_two_phases_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_bwd_two_phases(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    seq_offsets_q,
    DOut,
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
    attn_scale,
    Z,
    AUTOTUNE_Z,
    H,
    G,
    max_q_len,
    AUTOTUNE_MAX_SEQ_LEN,
    DimQ,
    DimV,
    M,
    Delta,
    stride_mm,
    num_softmax_heads: tl.constexpr,
    num_targets,
    causal: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KV: tl.constexpr,  # Triton TR001: Phase-2 KV inner-loop block
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
):
    # -- setup --
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    off_h_kv = (off_h // G).to(tl.int64)
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
    # offset pointers for batch/head
    Q += seq_start_q * stride_qm + off_h * stride_qh
    K += seq_start_kv * stride_kn + off_h_kv * stride_kh
    V += seq_start_kv * stride_vn + off_h_kv * stride_vh
    DOut += seq_start_q * stride_dom + off_h * stride_doh
    DQ += seq_start_q * stride_dqm + off_h * stride_dqh
    DK += seq_start_kv * stride_dkn + off_h_kv * stride_dkh
    DV += seq_start_kv * stride_dvn + off_h_kv * stride_dvh
    # target/causal preprocessing
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    M_off, Delta_off = backward_common_preprocess(
        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
    )
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)

    # ============== PHASE 1: dK, dV (column-parallel) ==============
    start_n = tl.program_id(1) * BLOCK_N
    if start_n < seq_len_kv:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len_kv
        # load K, V for this column block
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_qk_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_v_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
        # compute causal lower bound for Q row loop
        low = 0
        if HAS_CAUSAL:
            low = max(low, start_n + uih_len_q - seq_len_kv)
        offs_m_base = tl.arange(0, BLOCK_M)
        # -- Phase 1 softmax heads --
        if off_h < num_softmax_heads:
            for start_m in tl.range(low, seq_len_q, BLOCK_M):
                offs_m = start_m + offs_m_base
                mask_m = offs_m < seq_len_q
                # load Q transposed
                q_trans = tl.load(
                    Q + offs_m[None, :] * stride_qm + offs_qk_d[:, None],
                    mask=mask_m[None, :],
                    other=0.0,
                )
                # Triton TR001: load Q non-transposed [BLOCK_M, BLOCK_D_Q]
                # from global to avoid the in-loop tl.trans(q_trans) LDS shuffle
                # in dk. q_trans (transposed) is still loaded above for k @ q_trans.
                q_nt = tl.load(
                    Q + offs_m[:, None] * stride_qm + offs_qk_d[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
                valid_mask_trans = backward_valid_mask(
                    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                )
                qk_trans, act_qk_trans, pT = backward_softmax_activation(
                    qk_trans,
                    alpha,
                    valid_mask_trans,
                    M_off,
                    offs_m,
                    stride_mm,
                    mask_m,
                    k,
                )
                # load DOut
                do = tl.load(
                    DOut + offs_m[:, None] * stride_dom + offs_v_d[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                # Triton TR001: load DOut pre-transposed [BLOCK_D_V, BLOCK_M]
                # from global to avoid the in-loop tl.trans(do) LDS shuffle in
                # dact_qk_trans. do (non-transposed) is still used above for
                # dv += act_qk @ do.
                do_trans = tl.load(
                    DOut + offs_v_d[:, None] + offs_m[None, :] * stride_dom,
                    mask=mask_m[None, :],
                    other=0.0,
                )
                dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                # Triton TR001: dact uses pre-loaded do_trans, not tl.trans(do)
                dact_qk_trans = tl.dot(v, do_trans, allow_tf32=ALLOW_TF32)
                dqk_trans = backward_d_softmax_activation(
                    dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                )
                dqk_trans = dqk_trans.to(k.dtype)
                # Triton TR001: dk uses pre-loaded q_nt instead of tl.trans(q_trans)
                dk += tl.dot(dqk_trans, q_nt, allow_tf32=ALLOW_TF32)
        # -- Phase 1 SiLU heads --
        if off_h >= num_softmax_heads:
            for start_m in tl.range(low, seq_len_q, BLOCK_M):
                offs_m = start_m + offs_m_base
                mask_m = offs_m < seq_len_q
                if ATTN_SCALE_TYPE == "scalar":
                    scale = tl.load(attn_scale).to(tl.float32)
                else:
                    tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
                    scale = tl.load(attn_scale + offs_m, mask=mask_m).to(tl.float32)
                q_trans = tl.load(
                    Q + offs_m[None, :] * stride_qm + offs_qk_d[:, None],
                    mask=mask_m[None, :],
                    other=0.0,
                )
                # Triton TR001: also load Q non-transposed [BLOCK_M, BLOCK_D_Q]
                # from global to avoid the in-loop tl.trans(q_trans) LDS shuffle
                # in dk. q_trans (transposed) is still loaded above for k @ q_trans.
                q_nt = tl.load(
                    Q + offs_m[:, None] * stride_qm + offs_qk_d[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
                valid_mask_trans = backward_valid_mask(
                    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                )
                qk_trans, act_qk_trans, pT = backward_silu_activation(
                    qk_trans, alpha, valid_mask_trans, k.dtype, scale
                )
                do = tl.load(
                    DOut + offs_m[:, None] * stride_dom + offs_v_d[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                # Triton TR001: also load DOut pre-transposed [BLOCK_D_V, BLOCK_M]
                # from global to avoid the in-loop tl.trans(do) LDS shuffle in
                # dact_qk_trans. do (non-transposed) is still used above for
                # dv += act_qk @ do.
                do_trans = tl.load(
                    DOut + offs_v_d[:, None] + offs_m[None, :] * stride_dom,
                    mask=mask_m[None, :],
                    other=0.0,
                )
                dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                # Triton TR001: dact uses pre-loaded do_trans, not tl.trans(do)
                dact_qk_trans = tl.dot(v, do_trans, allow_tf32=ALLOW_TF32)
                dqk_trans = backward_d_silu_activation(
                    dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                )
                dqk_trans = dqk_trans.to(k.dtype)
                # Triton TR001: dk uses pre-loaded q_nt instead of tl.trans(q_trans)
                dk += tl.dot(dqk_trans, q_nt, allow_tf32=ALLOW_TF32)
        # write-back dk, dv
        dk = dk * alpha
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        if GQA_ATOMIC_ADD:
            tl.atomic_add(
                dv_ptrs,
                dv.to(k.dtype),
                mask=mask_n[:, None],
                sem="relaxed",
            )
            tl.atomic_add(
                dk_ptrs,
                dk.to(k.dtype),
                mask=mask_n[:, None],
                sem="relaxed",
            )
        else:
            tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])

    # ============== PHASE 2: dQ (row-parallel) ==============
    start_m2 = tl.program_id(1) * BLOCK_N
    if start_m2 < seq_len_q:
        offs_m2 = start_m2 + tl.arange(0, BLOCK_N)
        mask_m2 = offs_m2 < seq_len_q
        # load Q and DOut for this row block
        q2 = tl.load(
            Q + offs_m2[:, None] * stride_qm + offs_qk_d[None, :],
            mask=mask_m2[:, None],
            other=0.0,
        )
        do2 = tl.load(
            DOut + offs_m2[:, None] * stride_dom + offs_v_d[None, :],
            mask=mask_m2[:, None],
            other=0.0,
        )
        dq = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        # Triton TR001: Phase-2 KV inner block uses a dedicated BLOCK_KV
        # constexpr (decoupled from Phase-1's BLOCK_M) so the seq_kv reduction
        # runs fewer, larger iterations instead of ~seq_kv/BLOCK_M tiny ones.
        offs_n2_base = tl.arange(0, BLOCK_KV)
        # -- Phase 2 softmax heads --
        if off_h < num_softmax_heads:
            m_vals = tl.load(M_off + offs_m2 * stride_mm, mask=mask_m2)
            delta_vals = tl.load(Delta_off + offs_m2 * stride_mm, mask=mask_m2)
            # Triton TR001: step Phase-2 KV reduction by BLOCK_KV (not BLOCK_M)
            for start_n2 in tl.range(0, seq_len_kv, BLOCK_KV):
                offs_n2 = start_n2 + offs_n2_base
                mask_n2 = offs_n2 < seq_len_kv
                k2 = tl.load(
                    K + offs_n2[:, None] * stride_kn + offs_qk_d[None, :],
                    mask=mask_n2[:, None],
                    other=0.0,
                )
                # Triton TR001: load K/V pre-transposed from global to avoid
                # the in-loop tl.trans LDS shuffle + s_barrier stalls.
                # k2_trans is [BLOCK_D_Q, BLOCK_M], v2_trans is
                # [BLOCK_D_V, BLOCK_M]. k2 (non-transposed) is still loaded
                # above for dq += dqk2 @ k2.
                k2_trans = tl.load(
                    K + offs_qk_d[:, None] + offs_n2[None, :] * stride_kn,
                    mask=mask_n2[None, :],
                    other=0.0,
                )
                v2_trans = tl.load(
                    V + offs_v_d[:, None] + offs_n2[None, :] * stride_vn,
                    mask=mask_n2[None, :],
                    other=0.0,
                )
                # QK non-transposed: [BLOCK_N, BLOCK_M]
                qk2 = tl.dot(q2, k2_trans, allow_tf32=ALLOW_TF32)
                # valid mask non-transposed
                valid_mask2 = (offs_m2[:, None] < seq_len_q) & (
                    offs_n2[None, :] < seq_len_kv
                )
                if HAS_CAUSAL:
                    offs_m2_shifted = offs_m2 + seq_len_kv - uih_len_q
                    valid_mask2 = valid_mask2 & (
                        offs_m2_shifted[:, None] >= offs_n2[None, :]
                    )
                # softmax activation non-transposed
                qk2 = qk2 * alpha * 1.44269504
                pij = tl.math.exp2(qk2 - m_vals[:, None])
                pij = tl.where(valid_mask2, pij, 0.0)
                # dq contribution
                dact_qk2 = tl.dot(do2, v2_trans, allow_tf32=ALLOW_TF32)
                dqk2 = pij * (dact_qk2 - delta_vals[:, None])
                dqk2 = dqk2.to(k2.dtype)
                dq += tl.dot(dqk2, k2, allow_tf32=ALLOW_TF32)
        # -- Phase 2 SiLU heads --
        if off_h >= num_softmax_heads:
            if ATTN_SCALE_TYPE == "scalar":
                scale2 = tl.load(attn_scale).to(tl.float32)
            else:
                tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
                scale2 = tl.load(attn_scale + offs_m2, mask=mask_m2).to(tl.float32)
            # Triton TR001: step Phase-2 KV reduction by BLOCK_KV (not BLOCK_M)
            for start_n2 in tl.range(0, seq_len_kv, BLOCK_KV):
                offs_n2 = start_n2 + offs_n2_base
                mask_n2 = offs_n2 < seq_len_kv
                k2 = tl.load(
                    K + offs_n2[:, None] * stride_kn + offs_qk_d[None, :],
                    mask=mask_n2[:, None],
                    other=0.0,
                )
                # Triton TR001: load K/V pre-transposed from global to avoid
                # the in-loop tl.trans LDS shuffle + s_barrier stalls.
                # k2_trans is [BLOCK_D_Q, BLOCK_M], v2_trans is
                # [BLOCK_D_V, BLOCK_M]. k2 (non-transposed) is still loaded
                # above for dq += dqk2 @ k2.
                k2_trans = tl.load(
                    K + offs_qk_d[:, None] + offs_n2[None, :] * stride_kn,
                    mask=mask_n2[None, :],
                    other=0.0,
                )
                v2_trans = tl.load(
                    V + offs_v_d[:, None] + offs_n2[None, :] * stride_vn,
                    mask=mask_n2[None, :],
                    other=0.0,
                )
                qk2 = tl.dot(q2, k2_trans, allow_tf32=ALLOW_TF32)
                valid_mask2 = (offs_m2[:, None] < seq_len_q) & (
                    offs_n2[None, :] < seq_len_kv
                )
                if HAS_CAUSAL:
                    offs_m2_shifted = offs_m2 + seq_len_kv - uih_len_q
                    valid_mask2 = valid_mask2 & (
                        offs_m2_shifted[:, None] >= offs_n2[None, :]
                    )
                # SiLU activation non-transposed
                masked_alpha = tl.where(valid_mask2, alpha, 0.0)
                qk2 = qk2 * masked_alpha
                sig = fast_silu(qk2, MULT_BY_X=False)
                # dq contribution
                dact_qk2 = tl.dot(do2, v2_trans, allow_tf32=ALLOW_TF32)
                if ATTN_SCALE_TYPE == "scalar":
                    dqk2 = dact_qk2 * sig * (1 + qk2 * (1 - sig)) * scale2
                else:
                    dqk2 = dact_qk2 * sig * (1 + qk2 * (1 - sig)) * scale2[:, None]
                dqk2 = tl.where(valid_mask2, dqk2, 0)
                dqk2 = dqk2.to(k2.dtype)
                dq += tl.dot(dqk2, k2, allow_tf32=ALLOW_TF32)
        # store dq directly
        dq = dq * alpha
        dq_ptrs = DQ + offs_m2[:, None] * stride_dqm + offs_qk_d[None, :]
        tl.store(dq_ptrs, dq, mask=mask_m2[:, None])


@torch.library.custom_op("hammer::triton_hstu_cross_attn_v2_bwd", mutates_args=())
def triton_hstu_cross_attn_v2_bwd(
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
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    M: torch.Tensor,
    Delta: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(max_seq_len, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(max_q_len, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    TMA_DESC_SIZE = 128
    tma_workspace = None
    if enable_tma:
        MIN_BLOCK_N = 16
        tma_workspace = torch.empty(
            6 * TMA_DESC_SIZE * (triton.cdiv(max_seq_len, MIN_BLOCK_N) * Z * H),
            dtype=torch.uint8,
            device="cuda",
        )
    HAS_NUM_TARGETS = num_targets is not None
    HAS_CAUSAL = causal
    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        tma_workspace_ptr=tma_workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
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
        attn_scale=attn_scale,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        G=G,
        max_q_len=max_q_len,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        M=M,
        Delta=Delta,
        stride_mm=stride_mm,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_CAUSAL=HAS_CAUSAL,
        ATTN_SCALE_TYPE=attn_scale_type,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )

    return dq.to(q.dtype), dk, dv


@torch.library.custom_op(
    "hammer::triton_hstu_cross_attn_v2_bwd_two_phases", mutates_args=()
)
def triton_hstu_cross_attn_v2_bwd_two_phases(
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
    sort_by_length_indices: Optional[torch.Tensor],
    M: torch.Tensor,
    Delta: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    grid = lambda meta: (  # noqa E731
        Z * H,
        max(
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
            triton.cdiv(max_q_len, meta["BLOCK_N"]),
        ),
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    HAS_NUM_TARGETS = num_targets is not None
    HAS_CAUSAL = causal
    _hstu_attn_bwd_two_phases[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DOut=dout,
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
        attn_scale=attn_scale,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        G=G,
        max_q_len=max_q_len,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        M=M,
        Delta=Delta,
        stride_mm=stride_mm,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_CAUSAL=HAS_CAUSAL,
        ATTN_SCALE_TYPE=attn_scale_type,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        GQA_ATOMIC_ADD=G > 1,
    )

    return dq.to(q.dtype), dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        max_q_len: Optional[int],
        seq_offsets_q: Optional[torch.Tensor],
        sort_by_length: bool,
        enable_tma: bool,
        num_softmax_heads: int,
        num_targets: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        _, H, _ = q.shape
        _, H_kv, _ = k.shape
        assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
        G = H // H_kv  # GQA group size (number of Q heads per KV head)
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets, attn_scale]
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        if seq_offsets_q is not None:
            saved_tensors.append(seq_offsets_q)
        ctx.num_softmax_heads = num_softmax_heads
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.causal = causal
        M, stride_mm = forward_custom_vars(q, num_softmax_heads, saved_tensors)
        ctx.alpha = alpha
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        ctx.G = G
        out = triton_hstu_cross_attn_v2_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            max_q_len=max_q_len,
            seq_offsets_q=seq_offsets_q,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
            M=M,
            stride_mm=stride_mm,
            num_softmax_heads=num_softmax_heads,
            num_targets=num_targets,
            causal=causal,
            G=G,
        )
        saved_tensors.append(out)
        ctx.save_for_backward(*saved_tensors)
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
    ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale = saved_tensors[:5]
        idx = 5
        if ctx.sort_by_length:
            sort_by_length_indices = saved_tensors[idx]
            idx += 1
        else:
            sort_by_length_indices = None
        if ctx.max_q_len is not None:
            seq_offsets_q = saved_tensors[idx]
            idx += 1
        else:
            seq_offsets_q = None
        num_softmax_heads = ctx.num_softmax_heads
        if ctx.has_num_targets:
            num_targets = saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        causal = ctx.causal
        M, Delta, stride_mm = backward_custom_vars(
            dout, num_softmax_heads, idx, saved_tensors
        )
        dq, dk, dv = torch.ops.hammer.triton_hstu_cross_attn_v2_bwd(
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
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=ctx.enable_tma,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_softmax_heads=num_softmax_heads,
            num_targets=num_targets,
            causal=causal,
            G=ctx.G,
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
        )


@torch.jit.unused
@torch.fx.wrap
def triton_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        attn_scale,
        max_q_len,
        seq_offsets_q,
        sort_by_length,
        enable_tma,
        num_softmax_heads,
        num_targets,
        causal,
    )


def triton_hstu_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    return triton_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
    )


@triton_hstu_cross_attn_v2_fwd.register_fake  # pyre-ignore[16]
@triton_hstu_cross_attn_v2_fwd.register_kernel("cpu")  # pyre-ignore[16]
def _triton_hstu_cross_attn_v2_fwd_fake(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    M: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> torch.Tensor:
    # Output has query's batch dim (N_q) and heads (H), but value's feature dim (DimV).
    return torch.empty(
        q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype
    )


@triton_hstu_cross_attn_v2_bwd.register_fake
def _triton_hstu_cross_attn_v2_bwd_fake(
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
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    M: torch.Tensor,
    Delta: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
    )


@triton_hstu_cross_attn_v2_bwd_two_phases.register_fake
def _triton_hstu_cross_attn_v2_bwd_two_phases_fake(
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
    sort_by_length_indices: Optional[torch.Tensor],
    M: torch.Tensor,
    Delta: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
    )
