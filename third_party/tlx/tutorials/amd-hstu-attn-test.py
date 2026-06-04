from typing import List, Optional, Tuple

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

try:
    from triton.language.extra.libdevice import (
        fast_dividef,
        fast_expf,
    )  # @manual=//triton:triton
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.hip.libdevice import fast_dividef, fast_expf
    except ImportError:
        # pyre-ignore[21]
        from triton.language.math import (
            fast_dividef,
            fast_expf,
        )  # @manual=//triton:triton


def prev_power_of_2(x: int) -> int:
    out = triton.next_power_of_2(x)
    return out // 2 if out > x else out

def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if FULL_TUNING:
        for BLOCK_M in [32, 64, 128]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": 0,
                                        "kpack": 2 if get_arch() == "gfx942" else 1,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                )
                            )
    else:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 0,
                    "kpack": 1,
                },
                num_stages=1,
                num_warps=4,
            ),
        ]

    return configs

@triton.jit
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    q,
    K_base,
    V_base,
    stride_kn,
    stride_vn,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    max_attn_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_D_Q)
    offs_d_v = tl.arange(0, BLOCK_D_V)
    k_ptrs = K_base + offs_d_q[:, None] + offs_n[None, :] * stride_kn
    v_ptrs = V_base + offs_n[:, None] * stride_vn + offs_d_v[None, :]

    # -- compute qk ----
    mask_n = offs_n < seq_len
    k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
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
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    if not CAUSAL:
        offs_m_minus_n = tl.where(offs_m_minus_n > 0, offs_m_minus_n, -offs_m_minus_n)
    invalid_mask = invalid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    num_targets,
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
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        K_base = K + off_h * stride_kh + seq_start * stride_kn
        V_base = V + off_h * stride_vh + seq_start * stride_vn
        offs_d_q = tl.arange(0, BLOCK_D_Q)
        mask_m = offs_m < seq_len
        if IS_DELTA_Q:
            Q_base = Q + off_h * stride_qh + off_z * DeltaSize * stride_qm
            q_ptrs = Q_base + (start_m_delta + tl.arange(0, BLOCK_M))[:, None] * stride_qm + offs_d_q[None, :]
            q = tl.load(q_ptrs, mask=((start_m_delta + tl.arange(0, BLOCK_M)) < DeltaSize)[:, None], other=0.0)
        else:
            Q_base = Q + off_h * stride_qh + seq_start * stride_qm
            q_ptrs = Q_base + offs_m[:, None] * stride_qm + offs_d_q[None, :]
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if CAUSAL:
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len
            if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len
            else:
                low = 0
                high = start_m + BLOCK_M
                if HAS_MAX_ATTN_LEN:
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
        else:
            low = 0
            high = seq_len

        for start_n in range(low, high, BLOCK_N):
            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                q=q,
                K_base=K_base,
                V_base=V_base,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )

        if HAS_MULTIPLE_TARGETS and CAUSAL:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_delta,
                        q=q,
                        K_base=K_base,
                        V_base=V_base,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        CAUSAL=CAUSAL,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                    )

        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])



# @triton.autotune(
#     configs=_get_fw_configs(),
#     key=[
#         "AUTOTUNE_Z",
#         "H",
#         "AUTOTUNE_MAX_SEQ_LEN",
#         "DimQ",
#         "DimV",
#         "BUCKET_FN",
#         "ATTN_BIAS_TYPE",
#         "DeltaSize",
#         "IS_DELTA_Q",
#     ],
# )
@triton.jit
def _ragged_hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
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
    H,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
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
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        CAUSAL=CAUSAL,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


