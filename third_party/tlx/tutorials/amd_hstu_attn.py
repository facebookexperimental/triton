from typing import List, Optional, Tuple

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import torch
import torch.nn.functional as F

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


str_to_torch_dtype = {
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


#----------------------------------------------------------------------
# ref implementation
# create attention mask
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(valid_attn_mask, row_col_dist <= max_attn_len)
    if contextual_seq_len > 0:
        valid_attn_mask = torch.logical_or(valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids))
    return valid_attn_mask


# convert sequence input from jagged format to padded dense format
def jagged_to_padded_dense(q: torch.Tensor, offsets: torch.Tensor, max_seq_len: int, padding_value):
    assert len(q.shape) == 2, "q needs to be 2-dim tensor"
    L, D = q.shape
    B = offsets.shape[0] - 1
    padded_shape = (B, max_seq_len, D)
    padded_q = torch.full(padded_shape, padding_value, dtype=q.dtype, device=q.device)
    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        padded_q[i][0:e - s] = q[s:e]

    return padded_q


# pad sequence according to max sequence len
def pad_sequence(q: torch.Tensor, seq_offsets: torch.Tensor, N: int, padding_value):
    L, D = q.shape
    padded_q = jagged_to_padded_dense(q.reshape(L, D), offsets=seq_offsets, max_seq_len=N, padding_value=0.0)

    return padded_q


def qkv_to_padded_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    padded_q = (pad_sequence(q.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2))
    padded_k = (pad_sequence(k.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2))
    padded_v = (pad_sequence(v.reshape(L, H * D), seq_offsets, N, 0.0).view(-1, N, H, D).transpose(1, 2))

    return padded_q, padded_k, padded_v


# convert sequences from dense format to jagged format
def dense_to_jagged(seq: torch.Tensor, offsets: torch.Tensor, L: int):
    B, N, HV = seq.shape
    assert L == offsets[-1], f"jagged dim mismatch {offsets[-1]} != {L}!"
    out = torch.empty((L, HV), dtype=seq.dtype, device=seq.device)

    for i in range(B):
        s = offsets[i]
        e = offsets[i + 1]
        out[s:e] = seq[i][0:e - s]

    return out


# torch hstu reference implementation
def torch_hstu_attention(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = qkv_to_padded_dense(q, k, v, seq_offsets, max_seq_len)  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )
    # raise NotImplementedError(valid_attn_mask[0, :, :].to(torch.int32))
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        seq_offsets,
        L,
    ).view(L, H, V)


#----------------------------------------------------------------------


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

    return pid


FULL_TUNING = False


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
                                ))
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
    lds_k,
    lds_v,
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

    k_ptrs = K_base + offs_d_q[None, :] + offs_n[:, None] * stride_kn
    v_ptrs = V_base + offs_n[:, None] * stride_vn + offs_d_v[None, :]
    k_local = tlx.local_view(lds_k, 0)
    v_local = tlx.local_view(lds_v, 0)

    # -- compute qk ----
    mask_n = offs_n < seq_len
    tok_k = tlx.async_load(k_ptrs, k_local, mask=mask_n[:, None])
    tlx.async_load_commit_group([tok_k])
    tlx.async_load_wait_group(0)
    kt_local = tlx.local_trans(k_local)
    k_dot_op = tlx.local_load(kt_local)

    qk = tl.dot(q, k_dot_op, allow_tf32=ALLOW_TF32) * alpha
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
        invalid_mask = invalid_mask or (offs_m[:, None] == 0 and offs_n[None, :] < max_ids)
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + fast_expf(-1.0 * qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)

    tok_v = tlx.async_load(v_ptrs, v_local, mask=mask_n[:, None])
    tlx.async_load_commit_group([tok_v])
    tlx.async_load_wait_group(0)
    v_dot_op = tlx.local_load(v_local)

    silu = silu.to(V_base.dtype.element_ty)
    return tl.dot(silu, v_dot_op, allow_tf32=ALLOW_TF32)


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

        # allocate lds for KV load
        lds_k = tlx.local_alloc((BLOCK_N, BLOCK_D_Q), tlx.dtype_of(K), 1)
        lds_v = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), 1)

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
                lds_k=lds_k,
                lds_v=lds_v,
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
                for start_delta in tl.range(low_delta, high_delta, BLOCK_N, num_stages=0):
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
                        lds_k=lds_k,
                        lds_v=lds_v,
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
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
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
    Z,
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
    tpid = tl.program_id(0)

    num_tiles = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    tpid = remap_xcd(tpid, num_tiles * Z * H, 8)

    off_h = tpid % H
    off_nz = tpid // H
    pid = off_nz % num_tiles
    off_z = off_nz // num_tiles

    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)

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


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length: bool = False,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    HSTU attention forward pass with SiLU activation: Y = silu(alpha * (Q @ K^T)) @ V.
    Works with jagged tensors (variable-length sequences concatenated along batch dimension).

    Args:
        N (int): Maximum sequence length across all sequences.
        alpha (float): Scale factor applied to Q @ K^T before SiLU activation.
        q (torch.Tensor): Query jagged tensor with shape (total_tokens, num_heads, head_dim).
        k (torch.Tensor): Key jagged tensor with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Value jagged tensor with shape (total_tokens, num_heads, head_dim).
        seq_offsets (torch.Tensor): Sequence boundaries with shape (batch_size + 1,).
            Element i contains cumulative token count up to sequence i.
        causal (bool): Apply causal masking.
        num_targets (Optional[torch.Tensor]): Number of target tokens per sequence for masking.
        max_attn_len (int): Maximum attention span limit. 0 disables limit.
        contextual_seq_len (int): Contextual prefix length. 0 disables contextual masking.
        sort_by_length_indices (Optional[torch.Tensor]): Indices to process sequences in descending length order.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N).

    Returns:
        torch.Tensor: Output jagged tensor with shape (total_tokens, num_heads, head_dim).
    """
    Z = seq_offsets.numel() - 1
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0

    if sort_by_length:
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        _, sort_by_length_indices = torch.sort(seq_lengths, descending=True, stable=False)

    has_sort_by_length_indices = sort_by_length
    if L == 0:
        return out

    DeltaSize = 0
    IS_DELTA_Q = False

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]) * Z * H, )

    _ragged_hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        H=H,
        MAX_SEQ_LEN=N,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        # **config,
    )

    return out


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# generate inputs
def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(1)  # for reproducibility

    if sparsity == 0.0:
        return torch.zeros(size=(size, ), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size, ), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        max_seq_len: int = max_seq_len
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


def apply_SL(
    lengths: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    threshold = int(max_seq_len**(alpha / 2.0))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    lengths = torch.where(users_to_sample, threshold, lengths)
    return lengths


def sanity_check_attention(
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    dropout_pr: float,
    seq2_offsets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
) -> None:
    _, H, _ = q.shape
    torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
    torch._assert(q.dim() == 3, "q must be 3-D")
    torch._assert(k.shape == q.shape, "k must be the same shape as q")
    torch._assert(v.dim() == 3, "v must be 3-D")
    torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
    torch._assert(v.shape[1] == H, "wrong v shape[1]")
    if attn_bias is not None:
        assert seq2_offsets is not None
        torch._assert(attn_bias.dim() == 1, "attn_bias must be 1-D")
        torch._assert(
            seq2_offsets is not None,
            "must have seq2_offsets when using attn_bias",
        )
        torch._assert(seq2_offsets.dim() == 1, "seq2_offsets must be 1-D")
    if max_attn_len is not None:
        torch._assert(max_attn_len > 0, "max_attn_len must be larger than 0")
    if invalid_attn_mask_type != "lower_triangular":
        torch._assert(
            contextual_seq_len == 0,
            "user context mask not supported on non-lower triangular mask",
        )
    torch._assert(q.is_cuda, "q must be CUDA tensor")
    torch._assert(k.is_cuda, "k must be CUDA tensor")
    torch._assert(v.is_cuda, "v must be CUDA tensor")
    torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
    if attn_bias is not None:
        torch._assert(attn_bias.is_cuda, "attn_bias must be CUDA tensor")
        assert seq2_offsets is not None
        torch._assert(seq2_offsets.is_cuda, "seq2_offsets must be CUDA tensor")
    torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")


# calculate flops of the hstu attention
# lower trigualar mask, so no need to multiple by 2
# for flops calculation
def get_flops(seq_offsets: torch.Tensor, heads: int, attn_dim: int, hidden_dim: int):
    total_flops = 0.0
    seq_num = seq_offsets.shape[0] - 1
    for i in range(seq_num):
        len = seq_offsets[i + 1] - seq_offsets[i]
        flops = len * len * (attn_dim + hidden_dim) * heads
        total_flops += flops

    return total_flops


def get_bytes(
    seq_offsets: torch.Tensor,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    elem_size: int,
):

    seq_num = seq_offsets.shape[0] - 1
    total_bytes = 0
    for i in range(seq_num):
        len = seq_offsets[i + 1] - seq_offsets[i]
        bytes = len * (attn_dim + len + hidden_dim) * heads * elem_size
        total_bytes += bytes

    return total_bytes


#shape info
def get_inputs():
    input_info = [
        (32, 1024, 0.5, 4, 128, 128),
        (32, 2048, 0.5, 4, 128, 128),
        (32, 4096, 0.5, 4, 128, 128),
        (32, 8192, 0.5, 4, 128, 128),
        (32, 16384, 0.5, 4, 128, 128),
        (512, 512, 0.97, 4, 128, 128),
        (512, 3072, 0.366, 4, 128, 128),
        (1024, 1024, 0.768, 4, 128, 128),
    ]

    return input_info
