"""TLX linear weight-gradient GEMM for AMD gfx950.

Computes ``dw = dout.T @ act`` from row-major ``dout`` and ``act`` without
materializing either transpose. The kernel transposes the dout tile in LDS and
uses split-K when the output tile grid cannot fill the device.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


NUM_CU = 256
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_BUFFERS = 2
NUM_WARPS = 8
REDUCE_BLOCK = 32
MIN_REDUCTION_PER_SPLIT = 4096


# Triton TR001: this gfx950 path uses a fixed, benchmarked tile configuration.
@triton.jit
def _wgrad_kernel(  # noqa: TR001
    dout_ptr,
    act_ptr,
    out_ptr,
    workspace_ptr,
    reduction_k,
    out_m,
    out_n,
    stride_dout_k,
    stride_dout_m,
    stride_act_k,
    stride_act_n,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid // GRID_MN
    pid = pid % GRID_MN

    num_pid_m = tl.cdiv(out_m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % out_m
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % out_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    split_size = tl.cdiv(reduction_k, SPLIT_K)
    split_begin = split_id * split_size
    split_end = min(split_begin + split_size, reduction_k)
    k_iters = tl.cdiv(split_end - split_begin, BLOCK_SIZE_K)

    dout_smem = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_M),
        tlx.dtype_of(dout_ptr),
        NUM_SMEM_BUFFERS,
    )
    act_smem = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N),
        tlx.dtype_of(act_ptr),
        NUM_SMEM_BUFFERS,
    )

    for i in tl.range(
        0,
        NUM_SMEM_BUFFERS,
        loop_unroll_factor=NUM_SMEM_BUFFERS,
    ):
        k_start = split_begin + i * BLOCK_SIZE_K
        k_mask = offs_k < split_end - k_start
        dout_offsets = (
            (k_start + offs_k[:, None]) * stride_dout_k
            + offs_m[None, :] * stride_dout_m
        )
        act_offsets = (
            (k_start + offs_k[:, None]) * stride_act_k
            + offs_n[None, :] * stride_act_n
        )
        dout_token = tlx.async_load(
            dout_ptr + dout_offsets,
            tlx.local_view(dout_smem, i),
            mask=k_mask[:, None],
        )
        tlx.async_load_commit_group([dout_token])
        act_token = tlx.async_load(
            act_ptr + act_offsets,
            tlx.local_view(act_smem, i),
            mask=k_mask[:, None],
        )
        tlx.async_load_commit_group([act_token])

    tlx.async_load_wait_group((NUM_SMEM_BUFFERS - 1) * 2)
    dout_tile = tlx.local_load(tlx.local_trans(tlx.local_view(dout_smem, 0)))
    act_tile = tlx.local_load(tlx.local_view(act_smem, 0))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for tile_id in tl.range(0, k_iters - NUM_SMEM_BUFFERS):
        prefetch_buf = tile_id % NUM_SMEM_BUFFERS
        next_buf = (tile_id + 1) % NUM_SMEM_BUFFERS
        k_start = split_begin + (tile_id + NUM_SMEM_BUFFERS) * BLOCK_SIZE_K

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(dout_tile, act_tile, acc, allow_tf32=False)

        tlx.async_load_wait_group((NUM_SMEM_BUFFERS - 2) * 2)
        with tlx.warp_pipeline_stage("mem", priority=1):
            k_mask = offs_k < split_end - k_start
            dout_offsets = (
                (k_start + offs_k[:, None]) * stride_dout_k
                + offs_m[None, :] * stride_dout_m
            )
            act_offsets = (
                (k_start + offs_k[:, None]) * stride_act_k
                + offs_n[None, :] * stride_act_n
            )
            dout_token = tlx.async_load(
                dout_ptr + dout_offsets,
                tlx.local_view(dout_smem, prefetch_buf),
                mask=k_mask[:, None],
            )
            tlx.async_load_commit_group([dout_token])
            act_token = tlx.async_load(
                act_ptr + act_offsets,
                tlx.local_view(act_smem, prefetch_buf),
                mask=k_mask[:, None],
            )
            tlx.async_load_commit_group([act_token])
            dout_tile = tlx.local_load(
                tlx.local_trans(tlx.local_view(dout_smem, next_buf))
            )
            act_tile = tlx.local_load(tlx.local_view(act_smem, next_buf))

    acc = tl.dot(dout_tile, act_tile, acc, allow_tf32=False)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, min(k_iters - 1, NUM_SMEM_BUFFERS - 1)):
        buf = (k_iters - (NUM_SMEM_BUFFERS - 1) + i) % NUM_SMEM_BUFFERS
        dout_tile = tlx.local_load(tlx.local_trans(tlx.local_view(dout_smem, buf)))
        act_tile = tlx.local_load(tlx.local_view(act_smem, buf))
        acc = tl.dot(dout_tile, act_tile, acc, allow_tf32=False)

    out_rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_mask = (out_rows[:, None] < out_m) & (out_cols[None, :] < out_n)
    out_offsets = out_rows[:, None] * out_n + out_cols[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + out_offsets, acc.to(tlx.dtype_of(out_ptr)), mask=out_mask)
    else:
        tl.store(
            workspace_ptr + split_id * out_m * out_n + out_offsets,
            acc,
            mask=out_mask,
        )


# Triton TR001: the reduction tile is fixed by the tuned split-K configuration.
@triton.jit
def _reduce_split_k_kernel(  # noqa: TR001
    workspace_ptr,
    out_ptr,
    out_m,
    out_n,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < out_m) & (offs_n[None, :] < out_n)
    offsets = offs_m[:, None] * out_n + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        acc += tl.load(
            workspace_ptr + split_id * out_m * out_n + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(out_ptr + offsets, acc.to(tlx.dtype_of(out_ptr)), mask=mask)


def _choose_split_k(
    reduction_k: int,
    out_m: int,
    out_n: int,
) -> int:
    if out_m == 512 and out_n == 512:
        desired_split_k = 48
    else:
        grid_mn = triton.cdiv(out_m, BLOCK_M) * triton.cdiv(out_n, BLOCK_N)
        desired_split_k = max(1, NUM_CU // grid_mn)
    useful_split_k = max(1, reduction_k // MIN_REDUCTION_PER_SPLIT)
    return min(desired_split_k, useful_split_k)


def wgrad(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    split_k: int | None = None,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
    group_size_m: int = 8,
    num_warps: int = NUM_WARPS,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """Return ``grad_output.T @ hidden`` on gfx950."""
    if grad_output.ndim != 2 or hidden.ndim != 2:
        raise ValueError("grad_output and hidden must be two-dimensional")
    if grad_output.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"batch mismatch: grad_output={tuple(grad_output.shape)}, "
            f"hidden={tuple(hidden.shape)}"
        )
    if grad_output.device != hidden.device or grad_output.dtype != hidden.dtype:
        raise ValueError("grad_output and hidden must have matching device and dtype")
    if not grad_output.is_contiguous() or not hidden.is_contiguous():
        raise ValueError("grad_output and hidden must be contiguous")

    reduction_k, out_m = grad_output.shape
    _, out_n = hidden.shape
    split_k = _choose_split_k(reduction_k, out_m, out_n) if split_k is None else split_k
    if split_k < 1:
        raise ValueError(f"split_k must be positive, got {split_k}")

    out = torch.empty((out_m, out_n), device=hidden.device, dtype=hidden.dtype)
    grid_mn = triton.cdiv(out_m, block_m) * triton.cdiv(out_n, block_n)
    workspace = (
        torch.empty(
            (split_k, out_m, out_n),
            device=hidden.device,
            dtype=torch.float32,
        )
        if split_k > 1
        else out
    )
    _wgrad_kernel[(grid_mn * split_k,)](
        grad_output,
        hidden,
        out,
        workspace,
        reduction_k,
        out_m,
        out_n,
        grad_output.stride(0),
        grad_output.stride(1),
        hidden.stride(0),
        hidden.stride(1),
        GRID_MN=grid_mn,
        SPLIT_K=split_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_size_m,
        NUM_SMEM_BUFFERS=NUM_BUFFERS,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        enable_sched_group_barrier_scheduler=True,
    )
    if split_k > 1:
        _reduce_split_k_kernel[
            (triton.cdiv(out_m, REDUCE_BLOCK), triton.cdiv(out_n, REDUCE_BLOCK))
        ](
            workspace,
            out,
            out_m,
            out_n,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=REDUCE_BLOCK,
            BLOCK_SIZE_N=REDUCE_BLOCK,
            num_warps=4,
        )
    return out
