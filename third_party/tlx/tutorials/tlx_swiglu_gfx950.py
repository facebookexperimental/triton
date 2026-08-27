"""TLX fused SwiGLU forward for gfx950.

Computes ``out = silu(a @ gate_b) * (a @ up_b)`` with one shared A stream and
two B streams. The shared-gate variant computes ``out = silu(a @ gate_b) *
(a @ gate_b)`` with a single B stream. The kernels are adapted from
``amd_addmm_glu``'s optimized TLX direct-to-LDS GEMM pipeline, but keep the
GEMM outputs in registers and perform the SwiGLU epilogue before the single
output store.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


NUM_XCDS = 8

BEST_CONFIG = {
    256: dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=4,
        XCD_CHUNK=4,
        num_warps=4,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
    512: dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        XCD_CHUNK=4,
        num_warps=4,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
    1024: dict(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        XCD_CHUNK=4,
        num_warps=8,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
}

SHARED_BEST_CONFIG = {
    256: dict(
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        num_warps=8,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
    512: dict(
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        num_warps=8,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
    1024: dict(
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        num_warps=8,
        matrix_instr_nonkdim=16,
        waves_per_eu=0,
    ),
}


@triton.jit
def _chiplet_transform_chunked(
    pid,
    num_workgroups,
    num_xcds: tl.constexpr,
    chunk_size: tl.constexpr,
):
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (
        num_xcds * chunk_size
    )
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return (
        (local_pid // chunk_size) * num_xcds * chunk_size
        + xcd * chunk_size
        + (local_pid % chunk_size)
    )


# Triton TR001: launched through BEST_CONFIG keyed by the production K dimension.
@triton.jit
def _tlx_swiglu_kernel(  # noqa: TR001
    a_ptr,
    gate_b_ptr,
    up_b_ptr,
    out_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sgb0,
    sgb1,
    sub0,
    sub1,
    so0,
    so1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sgb0 > 0)
    tl.assume(sgb1 > 0)
    tl.assume(sub0 > 0)
    tl.assume(sub1 > 0)
    tl.assume(so0 > 0)
    tl.assume(so1 > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n
    pid = _chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = offs_cm % M
    offs_n = offs_cn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_base_off = offs_m[:, None] * sa0
    b_base_off = offs_n[None, :]
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    NUM_BUFFERS: tl.constexpr = 3
    smem_a = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(a_ptr),
        NUM_BUFFERS,
    )
    smem_gate_b = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N),
        tlx.dtype_of(gate_b_ptr),
        NUM_BUFFERS,
    )
    smem_up_b = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N),
        tlx.dtype_of(up_b_ptr),
        NUM_BUFFERS,
    )

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = i * BLOCK_SIZE_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        gate_b_offs = (k_start + offs_k[:, None]) * sgb0 + b_base_off * sgb1
        up_b_offs = (k_start + offs_k[:, None]) * sub0 + b_base_off * sub1

        tok_a = tlx.async_load(
            a_ptr + a_offs,
            tlx.local_view(smem_a, i),
            mask=offs_k[None, :] < K - k_start,
        )
        tok_gate_b = tlx.async_load(
            gate_b_ptr + gate_b_offs,
            tlx.local_view(smem_gate_b, i),
            mask=offs_k[:, None] < K - k_start,
        )
        tok_up_b = tlx.async_load(
            up_b_ptr + up_b_offs,
            tlx.local_view(smem_up_b, i),
            mask=offs_k[:, None] < K - k_start,
        )
        tlx.async_load_commit_group([tok_a, tok_gate_b, tok_up_b])

    tlx.async_load_wait_group(1)
    a_tile = tlx.local_load(tlx.local_view(smem_a, 0))
    gate_b_tile = tlx.local_load(tlx.local_view(smem_gate_b, 0))
    up_b_tile = tlx.local_load(tlx.local_view(smem_up_b, 0))

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
        prefetch_buf = i % NUM_BUFFERS
        next_buf = (i + 1) % NUM_BUFFERS
        k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_gate = tl.dot(a_tile, gate_b_tile, acc_gate, allow_tf32=False)
            acc_up = tl.dot(a_tile, up_b_tile, acc_up, allow_tf32=False)

        with tlx.warp_pipeline_stage("mem", priority=1):
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
            gate_b_offs = (
                (k_prefetch + offs_k[:, None]) * sgb0 + b_base_off * sgb1
            )
            up_b_offs = (
                (k_prefetch + offs_k[:, None]) * sub0 + b_base_off * sub1
            )

            tok_a = tlx.async_load(
                a_ptr + a_offs,
                tlx.local_view(smem_a, prefetch_buf),
                mask=offs_k[None, :] < K - k_prefetch,
            )
            tok_gate_b = tlx.async_load(
                gate_b_ptr + gate_b_offs,
                tlx.local_view(smem_gate_b, prefetch_buf),
                mask=offs_k[:, None] < K - k_prefetch,
            )
            tok_up_b = tlx.async_load(
                up_b_ptr + up_b_offs,
                tlx.local_view(smem_up_b, prefetch_buf),
                mask=offs_k[:, None] < K - k_prefetch,
            )
            tlx.async_load_commit_group([tok_a, tok_gate_b, tok_up_b])

            a_tile = tlx.local_load(tlx.local_view(smem_a, next_buf))
            gate_b_tile = tlx.local_load(tlx.local_view(smem_gate_b, next_buf))
            up_b_tile = tlx.local_load(tlx.local_view(smem_up_b, next_buf))

        tlx.async_load_wait_group(1)

    acc_gate = tl.dot(a_tile, gate_b_tile, acc_gate, allow_tf32=False)
    acc_up = tl.dot(a_tile, up_b_tile, acc_up, allow_tf32=False)

    tlx.async_load_wait_group(0)
    for i in tl.static_range(0, NUM_BUFFERS - 1):
        buf = (k_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        gate_b_tile = tlx.local_load(tlx.local_view(smem_gate_b, buf))
        up_b_tile = tlx.local_load(tlx.local_view(smem_up_b, buf))
        acc_gate = tl.dot(a_tile, gate_b_tile, acc_gate, allow_tf32=False)
        acc_up = tl.dot(a_tile, up_b_tile, acc_up, allow_tf32=False)

    out = (acc_gate * tl.sigmoid(acc_gate) * acc_up).to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + offs_cm[:, None] * so0 + offs_cn[None, :] * so1
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask, cache_modifier=".cs")


def _run_kernel(a, gate_b, up_b, out, cfg):
    M, K = a.shape
    _, N = gate_b.shape
    grid = (
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )
    _tlx_swiglu_kernel[grid](
        a,
        gate_b,
        up_b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        gate_b.stride(0),
        gate_b.stride(1),
        up_b.stride(0),
        up_b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


# Triton TR001: launched through SHARED_BEST_CONFIG keyed by the production K dimension.
@triton.jit
def _tlx_swiglu_shared_v9_kernel(  # noqa: TR001
    a_ptr,
    gate_b_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    sa0,
    sa1,
    sgb0,
    sgb1,
    so0,
    so1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sgb0 > 0)
    tl.assume(sgb1 > 0)
    tl.assume(so0 > 0)
    tl.assume(so1 > 0)

    HALF_N: tl.constexpr = BLOCK_N // 2

    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(gate_b_ptr), 2)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(gate_b_ptr), 2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_off = offs_am[:, None] * sa0 + offs_k[None, :] * sa1
    bl_off = offs_k[:, None] * sgb0 + offs_bn[None, :] * sgb1
    br_off = bl_off + HALF_N * sgb1
    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    acc_left = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    iter_max: tl.constexpr = K // BLOCK_K

    tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b_left[0], gate_b_ptr, bl_off + b_k)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], gate_b_ptr, br_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * sa1
    b_k += BLOCK_K * sgb0

    tlx.buffer_load_to_local(smem_a[1], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b_left[1], gate_b_ptr, bl_off + b_k)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], gate_b_ptr, br_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * sa1
    b_k += BLOCK_K * sgb0

    tlx.async_load_wait_group(3)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)

    for _ in tl.range(0, iter_max - 2, 2, num_stages=1):
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tl.dot(a, b_left, acc_left, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smem_b_left[0], gate_b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tl.dot(a, b_right, acc_right, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a = tlx.local_load(smem_a[1], relaxed=True)
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], gate_b_ptr, br_off + b_k)
            tlx.async_load_commit_group()
            a_k += BLOCK_K * sa1
            b_k += BLOCK_K * sgb0

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tl.dot(a, b_left, acc_left, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a[1], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smem_b_left[1], gate_b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tl.dot(a, b_right, acc_right, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a = tlx.local_load(smem_a[0], relaxed=True)
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], gate_b_ptr, br_off + b_k)
            tlx.async_load_commit_group()
            a_k += BLOCK_K * sa1
            b_k += BLOCK_K * sgb0

    acc_left = tl.dot(a, b_left, acc_left, allow_tf32=False)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(smem_b_right[0], relaxed=True)

    acc_right = tl.dot(a, b_right, acc_right, allow_tf32=False)
    a = tlx.local_load(smem_a[1], relaxed=True)
    b_left = tlx.local_load(smem_b_left[1], relaxed=True)

    acc_left = tl.dot(a, b_left, acc_left, allow_tf32=False)
    b_right = tlx.local_load(smem_b_right[1], relaxed=True)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    left = (acc_left * tl.sigmoid(acc_left) * acc_left).to(out_ptr.dtype.element_ty)
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    left_ptrs = out_ptr + so0 * offs_cm[:, None] + so1 * offs_cn_left[None, :]
    tl.store(
        left_ptrs,
        left,
        mask=(offs_cm[:, None] < M) & (offs_cn_left[None, :] < N),
    )

    acc_right = tl.dot(a, b_right, acc_right, allow_tf32=False)

    right = (acc_right * tl.sigmoid(acc_right) * acc_right).to(out_ptr.dtype.element_ty)
    offs_cn_right = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)
    right_ptrs = out_ptr + so0 * offs_cm[:, None] + so1 * offs_cn_right[None, :]
    tl.store(
        right_ptrs,
        right,
        mask=(offs_cm[:, None] < M) & (offs_cn_right[None, :] < N),
    )


def _run_shared_v9_kernel(a, gate_b, out, cfg):
    M, K = a.shape
    _, N = gate_b.shape
    grid_mn = triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(
        N, cfg["BLOCK_SIZE_N"]
    )
    _tlx_swiglu_shared_v9_kernel[(grid_mn,)](
        a,
        gate_b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        gate_b.stride(0),
        gate_b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=cfg["BLOCK_SIZE_M"],
        BLOCK_N=cfg["BLOCK_SIZE_N"],
        BLOCK_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=1,
        GRID_MN=grid_mn,
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


def swiglu(a, gate_b, up_b):
    """Return ``silu(a @ gate_b) * (a @ up_b)``.

    ``a`` is ``(M, K)`` and both B operands are ``(K, N)``. The production
    wrapper usually passes ``W.t().contiguous()`` for each B operand.
    """
    if a.ndim != 2 or gate_b.ndim != 2 or up_b.ndim != 2:
        raise ValueError("swiglu expects two-dimensional matrix operands")
    if gate_b.shape != up_b.shape:
        raise ValueError(
            f"Gate/up matrices must have the same shape, got {tuple(gate_b.shape)} "
            f"and {tuple(up_b.shape)}"
        )
    if a.shape[1] != gate_b.shape[0]:
        raise ValueError(
            f"Incompatible matrix dimensions: {tuple(a.shape)} and "
            f"{tuple(gate_b.shape)}"
        )
    if a.dtype != gate_b.dtype or a.dtype != up_b.dtype:
        raise ValueError("Input and weight matrices must have the same dtype")
    if not a.is_cuda or not gate_b.is_cuda or not up_b.is_cuda:
        raise ValueError("swiglu expects CUDA/HIP tensors")
    K = a.shape[1]
    if K not in BEST_CONFIG:
        raise ValueError(f"No tuned gfx950 SwiGLU config for K={K}")
    out = torch.empty((a.shape[0], gate_b.shape[1]), device=a.device, dtype=a.dtype)
    _run_kernel(a, gate_b, up_b, out, BEST_CONFIG[K])
    return out


def swiglu_shared(a, gate_b):
    """Return ``silu(a @ gate_b) * (a @ gate_b)``."""
    if a.ndim != 2 or gate_b.ndim != 2:
        raise ValueError("swiglu_shared expects two-dimensional matrix operands")
    if a.shape[1] != gate_b.shape[0]:
        raise ValueError(
            f"Incompatible matrix dimensions: {tuple(a.shape)} and "
            f"{tuple(gate_b.shape)}"
        )
    if a.dtype != gate_b.dtype:
        raise ValueError("Input and weight matrices must have the same dtype")
    if not a.is_cuda or not gate_b.is_cuda:
        raise ValueError("swiglu_shared expects CUDA/HIP tensors")
    K = a.shape[1]
    if K not in SHARED_BEST_CONFIG:
        raise ValueError(f"No tuned gfx950 shared SwiGLU config for K={K}")
    out = torch.empty((a.shape[0], gate_b.shape[1]), device=a.device, dtype=a.dtype)
    _run_shared_v9_kernel(a, gate_b, out, SHARED_BEST_CONFIG[K])
    return out
