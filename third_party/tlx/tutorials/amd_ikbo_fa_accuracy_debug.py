"""Self-contained reproducer for masked TLX async-load accuracy failures.

Run from the Triton repository after configuring its normal development
environment:

    python third_party/tlx/tutorials/amd_ikbo_fa_accuracy_debug.py

The script defaults to AMD buffer ops disabled and enables Triton IR dumping.
Caller-provided environment variables take precedence. For example:

    HIP_VISIBLE_DEVICES=7 \
      TRITON_DUMP_DIR=$HOME/kernel_ir_dump/ikbo_fa_oss \
      python third_party/tlx/tutorials/amd_ikbo_fa_accuracy_debug.py

Set JFA_KV_SEQLEN_TXT to a file containing comma-separated K/V sequence
lengths to override the captured jagged distribution below.
"""

import math
import os
from datetime import datetime
from pathlib import Path


# Configure compilation and dumping before importing torch or Triton. Values
# explicitly supplied by the caller take precedence over these debug defaults.
_default_dump_dir = (
    Path.home()
    / "kernel_ir_dump"
    / f"{datetime.now().strftime('%m%d_%H%M')}_JFA_OSS"
)
os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
os.environ.setdefault("TRITON_DUMP_DIR", str(_default_dump_dir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.language.extra.tlx as tlx  # noqa: E402


DEVICE = triton.runtime.driver.active.get_active_torch_device()

_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "matrix_instr_nonkdim": 16,
            "NUM_BUFFERS_KV": 2,
        },
        num_stages=2,
        num_warps=2,
    ),
]


@triton.autotune(
    configs=_AMD_CONFIGS,
    key=["q_seq_len", "kv_seq_len", "d_head"],
)
@triton.jit
def _attn_fwd_jagged_tlx(
    query,
    q_offsets,
    key,
    k_offsets,
    value,
    out,
    ad_to_request_offset,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    qk_scale,
    d_head,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    off_z = tl.program_id(axis=1)
    off_h = tl.program_id(axis=2)
    q_offset = off_h.to(tl.int64) * stride_qh
    kv_offset = off_h.to(tl.int64) * stride_kh

    begin_q = tl.load(q_offsets + off_z)
    end_q = tl.load(q_offsets + off_z + 1)
    q_seq_len = end_q - begin_q

    off_zkv = tl.load(ad_to_request_offset + off_z)
    begin_k = tl.load(k_offsets + off_zkv)
    end_k = tl.load(k_offsets + off_zkv + 1)
    kv_seq_len = end_k - begin_k

    if start_m * BLOCK_M >= q_seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query
        + q_offset
        + (begin_q + offs_m[:, None]) * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len, other=0.0)

    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, NUM_BUFFERS_KV)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, NUM_BUFFERS_KV)

    # Make the masked-load failure deterministic. Valid lanes are overwritten
    # by async_load, while masked lanes remain NaN unless async_load zero-fills
    # them. Poison every double-buffered LDS slot before starting the pipeline.
    nan_tile = tl.full((BLOCK_N, BLOCK_D), float("nan"), tl.float16)
    for buffer_idx in tl.static_range(NUM_BUFFERS_KV):
        tlx.local_store(tlx.local_view(k_buf, buffer_idx), nan_tile)
        tlx.local_store(tlx.local_view(v_buf, buffer_idx), nan_tile)
    tl.debug_barrier()

    k_ptrs = (
        key
        + kv_offset
        + (begin_k + offs_n[:, None]) * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        value
        + kv_offset
        + (begin_k + offs_n[:, None]) * stride_vn
        + offs_d[None, :] * stride_vd
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    buffer_id = 0

    k_buf_cur = tlx.local_view(k_buf, 0)
    k0_token = tlx.async_load(k_ptrs, k_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    v_buf_cur = tlx.local_view(v_buf, 0)
    v0_token = tlx.async_load(v_ptrs, v_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    tlx.async_load_commit_group([k0_token, v0_token])

    n_iter = tl.cdiv(kv_seq_len, BLOCK_N)
    n_main = tl.maximum(0, n_iter - 1)
    for i_iter in tl.range(0, n_main, num_stages=0):
        next_off = (i_iter + 1) * BLOCK_N
        next_off = tl.multiple_of(next_off, BLOCK_N)
        buffer_id_next = buffer_id ^ 1

        tlx.async_load_wait_group(tl.constexpr(0))
        k_buf_cur = tlx.local_view(k_buf, buffer_id)
        kt_view = tlx.local_trans(k_buf_cur)
        kt_cur = tlx.local_load(kt_view)
        v_buf_cur = tlx.local_view(v_buf, buffer_id)
        v_cur = tlx.local_load(v_buf_cur)

        next_mask = (next_off + offs_n[:, None]) < kv_seq_len
        k_buf_next = tlx.local_view(k_buf, buffer_id_next)
        k_token = tlx.async_load(
            k_ptrs + next_off * stride_kn, k_buf_next, mask=next_mask
        )
        v_buf_next = tlx.local_view(v_buf, buffer_id_next)
        v_token = tlx.async_load(
            v_ptrs + next_off * stride_vn, v_buf_next, mask=next_mask
        )
        tlx.async_load_commit_group([k_token, v_token])

        qk = tl.dot(q, kt_cur)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
        buffer_id = buffer_id_next

    tlx.async_load_wait_group(tl.constexpr(0))
    k_buf_cur = tlx.local_view(k_buf, buffer_id)
    kt_view = tlx.local_trans(k_buf_cur)
    kt_cur = tlx.local_load(kt_view)
    v_buf_cur = tlx.local_view(v_buf, buffer_id)
    v_cur = tlx.local_load(v_buf_cur)

    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < kv_seq_len, qk, -1.0e10)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    inv_li = 1.0 / l_i[:, None]
    acc *= inv_li
    acc = tl.where(kv_seq_len > 0, acc, 0.0)

    o_ptrs = (
        out
        + off_h.to(tl.int64) * stride_oh
        + (begin_q + offs_m[:, None]) * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        acc.to(out.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_head),
    )


def tlx_jagged_flash_attn_ikbo(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    max_seq_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    d_head = query.shape[-1]
    block_d = triton.next_power_of_2(d_head)
    sm_scale = scale if scale is not None else 1.0 / math.sqrt(d_head)
    qk_scale = sm_scale / math.log(2.0)
    output = torch.empty_like(query)
    num_heads = query.shape[1]
    batch_size = query_offset.size(0) - 1

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            batch_size,
            num_heads,
        )

    _attn_fwd_jagged_tlx[grid](
        query,
        query_offset,
        key,
        key_offset,
        value,
        output,
        ad_to_request_mapping,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        qk_scale,
        d_head=d_head,
        BLOCK_D=block_d,
    )
    return output


# Jagged K/V distribution captured by the original IKBO accuracy driver.
DEFAULT_KV_SEQ_LENGTHS = [
    330,
    1000,
    158,
    1000,
    380,
    344,
    244,
    902,
    270,
    147,
    529,
    97,
    1,
    29,
    519,
    152,
    1000,
    33,
    412,
    1000,
    130,
    96,
    884,
    168,
    1000,
    252,
    236,
    1000,
    35,
    90,
    869,
    282,
]


def _read_kv_seq_lengths(path: str) -> list[int]:
    """Read comma-separated sequence lengths, ignoring '#' metadata lines."""
    values: list[int] = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.extend(int(value) for value in line.split(",") if value)
    if not values:
        raise ValueError(f"No K/V sequence lengths found in {path}")
    if any(value <= 0 for value in values):
        raise ValueError("All K/V sequence lengths must be positive")
    return values


def _get_kv_seq_lengths() -> list[int]:
    path = os.environ.get("JFA_KV_SEQLEN_TXT")
    if path:
        return _read_kv_seq_lengths(path)
    return DEFAULT_KV_SEQ_LENGTHS.copy()


def _build_inputs(
    kv_seq_lengths: list[int],
    ads_per_user: int,
    q_seq_len: int,
    num_heads: int,
    d_head: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    dtype = torch.float16
    num_users = len(kv_seq_lengths)
    batch_size = num_users * ads_per_user

    ad_to_user_mapping = torch.arange(
        num_users, device=DEVICE, dtype=torch.int64
    ).repeat_interleave(ads_per_user)
    query_offset = (
        torch.arange(batch_size + 1, device=DEVICE, dtype=torch.int64) * q_seq_len
    )
    key_offset = torch.zeros(num_users + 1, device=DEVICE, dtype=torch.int64)
    key_offset[1:] = torch.tensor(
        kv_seq_lengths, device=DEVICE, dtype=torch.int64
    ).cumsum(0)

    query = torch.randn(
        (batch_size * q_seq_len, num_heads, d_head),
        device=DEVICE,
        dtype=dtype,
    )
    key = torch.cat(
        [
            torch.randn((length, num_heads, d_head), device=DEVICE, dtype=dtype)
            for length in kv_seq_lengths
        ],
        dim=0,
    )
    value = torch.cat(
        [
            torch.randn((length, num_heads, d_head), device=DEVICE, dtype=dtype)
            for length in kv_seq_lengths
        ],
        dim=0,
    )
    return query, key, value, query_offset, key_offset, ad_to_user_mapping


def _pytorch_jagged_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_seq_lengths: list[int],
    ads_per_user: int,
    q_seq_len: int,
) -> torch.Tensor:
    """Compute the reference in one SDPA call per user."""
    num_heads = query.shape[1]
    d_head = query.shape[2]
    output = torch.empty_like(query)
    key_start = 0
    queries_per_user = ads_per_user * q_seq_len

    for user, kv_seq_len in enumerate(kv_seq_lengths):
        query_start = user * queries_per_user
        query_end = query_start + queries_per_user
        key_end = key_start + kv_seq_len

        q = (
            query[query_start:query_end]
            .reshape(ads_per_user, q_seq_len, num_heads, d_head)
            .permute(0, 2, 1, 3)
        )
        k = (
            key[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(ads_per_user, -1, -1, -1)
        )
        v = (
            value[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(ads_per_user, -1, -1, -1)
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        output[query_start:query_end] = (
            expected.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(queries_per_user, num_heads, d_head)
        )
        key_start = key_end

    return output


def main() -> None:
    kv_seq_lengths = _get_kv_seq_lengths()
    ads_per_user = int(os.environ.get("JFA_ADS_PER_USER", "32"))
    q_seq_len = int(os.environ.get("JFA_Q_SEQ_LEN", "32"))
    num_heads = int(os.environ.get("JFA_NUM_HEADS", "2"))
    d_head = int(os.environ.get("JFA_D_HEAD", "128"))
    seed = int(os.environ.get("JFA_SEED", "2"))
    rtol = float(os.environ.get("JFA_RTOL", "1e-4"))
    atol = float(os.environ.get("JFA_ATOL", "1e-3"))
    batch_size = len(kv_seq_lengths) * ads_per_user

    dump_dir = Path(os.environ["TRITON_DUMP_DIR"])
    dump_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[amd_ikbo_fa_accuracy_debug] "
        f"B={batch_size} Bu={len(kv_seq_lengths)} "
        f"ads_per_user={ads_per_user} q_seq_len={q_seq_len} "
        f"H={num_heads} D={d_head} "
        f"min_kv={min(kv_seq_lengths)} max_kv={max(kv_seq_lengths)}",
        flush=True,
    )
    print(
        "[amd_ikbo_fa_accuracy_debug] "
        f"AMDGCN_USE_BUFFER_OPS={os.environ['AMDGCN_USE_BUFFER_OPS']} "
        f"TRITON_DUMP_DIR={dump_dir}",
        flush=True,
    )
    print(
        f"[amd_ikbo_fa_accuracy_debug] kv_seq_lengths={kv_seq_lengths}",
        flush=True,
    )

    query, key, value, query_offset, key_offset, ad_to_user_mapping = _build_inputs(
        kv_seq_lengths,
        ads_per_user,
        q_seq_len,
        num_heads,
        d_head,
        seed,
    )
    expected = _pytorch_jagged_reference(
        query,
        key,
        value,
        kv_seq_lengths,
        ads_per_user,
        q_seq_len,
    )
    actual = tlx_jagged_flash_attn_ikbo(
        query,
        key,
        value,
        query_offset,
        key_offset,
        ad_to_user_mapping,
        q_seq_len,
    )
    torch.cuda.synchronize()

    finite = torch.isfinite(actual)
    if not bool(finite.all()):
        num_nonfinite = int((~finite).sum().item())
        raise AssertionError(f"TLX IKBO output contains {num_nonfinite} NaN/Inf values")

    max_abs_error = float((actual - expected).abs().max().item())
    print(
        f"[amd_ikbo_fa_accuracy_debug] max_abs_error={max_abs_error}",
        flush=True,
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print("[amd_ikbo_fa_accuracy_debug] accuracy check passed", flush=True)


if __name__ == "__main__":
    main()
