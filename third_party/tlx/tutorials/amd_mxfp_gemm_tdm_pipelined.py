"""
MXFP TDM-pipelined GEMM for AMD gfx1250.

Ports the baseline Gluon ``mxgemm_tdm_pipelined_kernel`` schedule to TLX APIs:
full-tile TDM loads for A/B and pre-shuffled scales
(``tlx.async_amd_descriptor_load``), ``tlx.local_reshape`` + ``tlx.local_trans``
to un-shuffle the scales out of LDS, ``tlx.local_load`` into registers, and
``tl.dot_scaled`` for the scaled WMMA.

Multi-buffer software pipeline: prologue issues ``NUM_BUFFERS - 1`` tile loads,
then the main loop consumes tile k while prefetching tile ``k + NUM_BUFFERS - 1``.
Output stored via pointer-based ``tl.store`` (TDM store requires
``alignTDMDescriptorEncodings`` which is not yet ported to main).
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


def pack_scale(x: torch.Tensor, preshuffle_factor: int = 128) -> torch.Tensor:
    """Pre-shuffle an ``[non_k, k_scale]`` e8m0 scale tensor into the TDM layout."""
    non_k, k_scale = x.shape
    scale_kwidth = 4 if k_scale >= 4 else k_scale
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k_scale // scale_kwidth
    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(non_k // preshuffle_factor, k_scale * preshuffle_factor)


@triton.jit
def _mxgemm_issue_tdm_loads(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_buf,
    b_buf,
    a_scale_buf,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), [0, load_idx * BLOCK_K_PACKED_A], pred=pred)
    if TRANSPOSE_B:
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), [0, load_idx * BLOCK_K_PACKED_B], pred=pred)
    else:
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), [load_idx * BLOCK_K_PACKED_B, 0], pred=pred)
    tlx.async_amd_descriptor_load(a_scale_desc, tlx.local_view(a_scale_buf, slot),
                                  [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED], pred=pred)
    tlx.async_amd_descriptor_load(b_scale_desc, tlx.local_view(b_scale_buf, slot),
                                  [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED], pred=pred)
    return load_idx + 1


@triton.jit
def _mxgemm_issue_tdm_loads_unpredicated(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_buf,
    b_buf,
    a_scale_buf,
    b_scale_buf,
    load_idx,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), [0, load_idx * BLOCK_K_PACKED_A])
    if TRANSPOSE_B:
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), [0, load_idx * BLOCK_K_PACKED_B])
    else:
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), [load_idx * BLOCK_K_PACKED_B, 0])
    tlx.async_amd_descriptor_load(a_scale_desc, tlx.local_view(a_scale_buf, slot),
                                  [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED])
    tlx.async_amd_descriptor_load(b_scale_desc, tlx.local_view(b_scale_buf, slot),
                                  [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED])
    return load_idx + 1


@triton.jit
def _mxgemm_load_operands(
    a_buf,
    b_buf,
    a_scale_buf,
    b_scale_buf,
    wmma_idx,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_M_PRESHUFFLED: tl.constexpr,
    BLOCK_N_PRESHUFFLED: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = wmma_idx % NUM_BUFFERS
    a = tlx.local_load(tlx.local_view(a_buf, slot))
    b_view = tlx.local_view(b_buf, slot)
    b = tlx.local_load(tlx.local_trans(b_view) if TRANSPOSE_B else b_view)

    scale_a_view = tlx.local_reshape(
        tlx.local_view(a_scale_buf, slot),
        [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
    )
    scale_a_view = tlx.local_trans(scale_a_view, (0, 3, 2, 1, 4))
    scale_a_view = tlx.local_reshape(scale_a_view, [BLOCK_M, BLOCK_K_SCALE])

    scale_b_view = tlx.local_reshape(
        tlx.local_view(b_scale_buf, slot),
        [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
    )
    scale_b_view = tlx.local_trans(scale_b_view, (0, 3, 2, 1, 4))
    scale_b_view = tlx.local_reshape(scale_b_view, [BLOCK_N, BLOCK_K_SCALE])

    scale_a = tlx.local_load(scale_a_view)
    scale_b = tlx.local_load(scale_b_view)

    return a, b, scale_a, scale_b


@triton.jit
def mxgemm_tdm_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale,
    DTYPE_A: tl.constexpr,
    DTYPE_B: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """C = (A * a_scale) @ (B * b_scale) with TDM async loads and a software pipeline."""
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_PACKED_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_BLOCK
    SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
    BLOCK_M_PRESHUFFLED: tl.constexpr = BLOCK_M // 128
    BLOCK_N_PRESHUFFLED: tl.constexpr = BLOCK_N // 128
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr = BLOCK_K_SCALE * 128
    NUM_LOADS_IN_BATCH: tl.constexpr = 4

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_desc = tl.make_tensor_descriptor(
        a_ptr + pid_m * BLOCK_M * stride_am,
        shape=[M, K // DIV_FACTOR_A],
        strides=[stride_am, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_K_PACKED_A],
    )
    if TRANSPOSE_B:
        b_desc = tl.make_tensor_descriptor(
            b_ptr + pid_n * BLOCK_N * stride_bn,
            shape=[N, K // DIV_FACTOR_B],
            strides=[stride_bn, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K_PACKED_B],
        )
        b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K_PACKED_B), tlx.dtype_of(b_ptr), NUM_BUFFERS)
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr + pid_n * BLOCK_N * stride_bn,
            shape=[K // DIV_FACTOR_B, N],
            strides=[stride_bk, tl.constexpr(1)],
            block_shape=[BLOCK_K_PACKED_B, BLOCK_N],
        )
        b_buf = tlx.local_alloc((BLOCK_K_PACKED_B, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    a_scale_desc = tl.make_tensor_descriptor(
        a_scale + pid_m * BLOCK_M_PRESHUFFLED * stride_scale,
        shape=[M // 128, K // SCALE_BLOCK * 128],
        strides=[stride_scale, tl.constexpr(1)],
        block_shape=[BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
    )
    b_scale_desc = tl.make_tensor_descriptor(
        b_scale + pid_n * BLOCK_N_PRESHUFFLED * stride_scale,
        shape=[N // 128, K // SCALE_BLOCK * 128],
        strides=[stride_scale, tl.constexpr(1)],
        block_shape=[BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
    )

    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    a_scale_buf = tlx.local_alloc((BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED), tlx.dtype_of(a_scale), NUM_BUFFERS)
    b_scale_buf = tlx.local_alloc((BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED), tlx.dtype_of(b_scale), NUM_BUFFERS)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    tl.assume(K_ITERS >= NUM_BUFFERS)
    epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)
    load_idx = 0
    wmma_idx = 0

    for _ in tl.static_range(NUM_BUFFERS - 1):
        load_idx = _mxgemm_issue_tdm_loads_unpredicated(
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_buf,
            b_buf,
            a_scale_buf,
            b_scale_buf,
            load_idx,
            BLOCK_K_PACKED_A,
            BLOCK_K_PACKED_B,
            BLOCK_K_SCALE_PRESHUFFLED,
            NUM_BUFFERS,
            TRANSPOSE_B,
        )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.assume(K_ITERS > 0)
    for i in tl.range(0, K_ITERS):
        pred = i - epilogue_lb
        pred = (pred >> 31) & 1
        load_idx = _mxgemm_issue_tdm_loads(
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_buf,
            b_buf,
            a_scale_buf,
            b_scale_buf,
            load_idx,
            pred,
            BLOCK_K_PACKED_A,
            BLOCK_K_PACKED_B,
            BLOCK_K_SCALE_PRESHUFFLED,
            NUM_BUFFERS,
            TRANSPOSE_B,
        )
        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
        a, b, scale_a, scale_b = _mxgemm_load_operands(
            a_buf,
            b_buf,
            a_scale_buf,
            b_scale_buf,
            wmma_idx,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_SCALE,
            BLOCK_M_PRESHUFFLED,
            BLOCK_N_PRESHUFFLED,
            BLOCK_K_SCALE_PRESHUFFLED,
            SCALE_KWIDTH,
            NUM_BUFFERS,
            TRANSPOSE_B,
        )
        wmma_idx += 1
        acc = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, acc)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_BUFFERS": 2,
    "DTYPE_A": "e5m2",
    "DTYPE_B": "e5m2",
    "SCALE_BLOCK": 32,
    "TRANSPOSE_B": False,
    "num_warps": 4,
    "waves_per_eu": 1,
}


def matmul(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, config=None) -> torch.Tensor:
    """C = (A * a_scale) @ (B * b_scale) using a TDM-pipelined MXFP kernel on AMD gfx1250.

    ``a_scale`` / ``b_scale`` must already be pre-shuffled with :func:`pack_scale`.
    When ``config["TRANSPOSE_B"]`` is set, ``b`` is the ``[N, K]`` transposed layout.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    TRANSPOSE_B = cfg["TRANSPOSE_B"]

    M, K = a.shape
    if TRANSPOSE_B:
        N, Kb = b.shape
    else:
        Kb, N = b.shape
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    mxgemm_tdm_pipelined_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        DTYPE_A=cfg["DTYPE_A"],
        DTYPE_B=cfg["DTYPE_B"],
        SCALE_BLOCK=cfg["SCALE_BLOCK"],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        TRANSPOSE_B=TRANSPOSE_B,
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return c
