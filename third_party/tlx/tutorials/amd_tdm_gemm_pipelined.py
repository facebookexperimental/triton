"""
TDM-pipelined GEMM kernels for AMD gfx1250.

The baseline kernel demonstrates descriptor loads, L2 prefetches, and a TDM
store of the result tile.  The single-warp-per-SIMD variant follows the tuned
gfx1250 schedule: each fused TDM instruction assigns the A and B transfers to
different wave groups, while four 32-wide LDS subtiles feed the WMMA sequence.

Both kernels rely on TLX layout propagation to select descriptor-compatible
padded LDS layouts for their local allocations.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _single_warp_per_simd_load_subtile(
    a_buf,
    b_buf,
    consumer,
    start: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_LEN: tl.constexpr,
):
    slot = consumer % NUM_BUFFERS
    a_view = tlx.local_slice(tlx.local_view(a_buf, slot), [0, start], [BLOCK_M, SUBTILE_LEN])
    a = tlx.local_load(a_view)

    if not TRANSPOSE_B:
        b_view = tlx.local_slice(tlx.local_view(b_buf, slot), [start, 0], [SUBTILE_LEN, BLOCK_N])
        b = tlx.local_load(b_view)
    else:
        b_view = tlx.local_slice(tlx.local_view(b_buf, slot), [0, start], [BLOCK_N, SUBTILE_LEN])
        # Transpose the LDS view before loading so dot operand lowering can use
        # a memdesc transpose instead of materializing a register transpose.
        b = tlx.local_load(tlx.local_trans(b_view))

    return a, b


@triton.jit
def _single_warp_per_simd_issue_loads(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    pred,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = producer % NUM_BUFFERS
    if not TRANSPOSE_B:
        a_load_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], pred=pred,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        b_load_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[producer * BLOCK_K, off_n], pred=pred,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        tlx.async_amd_descriptor_load_fused([
            (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
            (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
        ])
    else:
        a_load_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], pred=pred,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        b_load_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, producer * BLOCK_K], pred=pred,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        tlx.async_amd_descriptor_load_fused([
            (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
            (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
        ])
    return producer + 1


@triton.jit
def _single_warp_per_simd_issue_loads_unpredicated(
    a_desc,
    b_desc,
    a_buf,
    b_buf,
    producer,
    off_m,
    off_n,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    slot = producer % NUM_BUFFERS
    if not TRANSPOSE_B:
        a_load_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], pred=True,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        b_load_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[producer * BLOCK_K, off_n], pred=True,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        tlx.async_amd_descriptor_load_fused([
            (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
            (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
        ])
    else:
        a_load_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, producer * BLOCK_K], pred=True,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        b_load_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, producer * BLOCK_K], pred=True,
                                                   clamp_bounds=True, _fused_tdm_explicit_offset=True)
        tlx.async_amd_descriptor_load_fused([
            (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
            (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
        ])
    return producer + 1


@triton.jit
def _single_warp_per_simd_prefetch_unpredicated(
    a_desc,
    b_desc,
    prefetch_iter,
    off_m,
    off_n,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_iter * BLOCK_K])
    if not TRANSPOSE_B:
        tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_iter * BLOCK_K, off_n])
    else:
        tlx.amd_descriptor_prefetch_tensor(b_desc, [off_n, prefetch_iter * BLOCK_K])


@triton.jit
def _single_warp_per_simd_prefetch(
    a_desc,
    b_desc,
    prefetch_iter,
    off_m,
    off_n,
    pred,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_iter * BLOCK_K], pred=pred)
    if not TRANSPOSE_B:
        tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_iter * BLOCK_K, off_n], pred=pred)
    else:
        tlx.amd_descriptor_prefetch_tensor(b_desc, [off_n, prefetch_iter * BLOCK_K], pred=pred)


@triton.jit
def matmul_tdm_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B with TDM loads and a two-buffer software pipeline."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    NUM_BUFFERS: tl.constexpr = 2
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)
    c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, 0), [off_m, 0])
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, 0), [0, off_n])
    prefetch_pred = BLOCK_K < K
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, BLOCK_K], pred=prefetch_pred)
    tlx.amd_descriptor_prefetch_tensor(b_desc, [BLOCK_K, off_n], pred=prefetch_pred)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K_ITERS - 1):
        next_k = k + 1
        next_slot = next_k % NUM_BUFFERS
        tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, next_slot), [off_m, next_k * BLOCK_K])
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, next_slot), [next_k * BLOCK_K, off_n])

        prefetch_k = next_k + 1
        prefetch_pred = prefetch_k < K_ITERS
        tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_k * BLOCK_K], pred=prefetch_pred)
        tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_k * BLOCK_K, off_n], pred=prefetch_pred)

        tlx.async_amd_descriptor_wait(2)

        cur_slot = k % NUM_BUFFERS
        a_reg = tlx.local_load(tlx.local_view(a_buf, cur_slot))
        b_reg = tlx.local_load(tlx.local_view(b_buf, cur_slot))
        acc = tl.dot(a_reg, b_reg, acc)

    tlx.async_amd_descriptor_wait(0)
    last_slot = (K_ITERS - 1) % NUM_BUFFERS
    a_reg = tlx.local_load(tlx.local_view(a_buf, last_slot))
    b_reg = tlx.local_load(tlx.local_view(b_buf, last_slot))
    acc = tl.dot(a_reg, b_reg, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    c_view = tlx.local_view(c_buf, 0)
    tlx.local_store(c_view, c)
    tlx.async_amd_descriptor_store(c_desc, c_view, [off_m, off_n])
    tlx.async_amd_descriptor_wait(0)


@triton.jit
def matmul_tdm_pipelined_single_warp_per_simd_schedule_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    L2_PREFETCH_DISTANCE: tl.constexpr,
):
    """Single-warp-per-SIMD TDM schedule tuned for gfx1250 WMMA."""
    tl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    NUM_SUBTILES: tl.constexpr = 4
    SUBTILE_LEN: tl.constexpr = BLOCK_K // NUM_SUBTILES
    tl.static_assert(SUBTILE_LEN == 32, "Subtile length must match the kdim of the WMMA instruction")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr + off_m * stride_am,
        shape=[M, K],
        strides=[stride_am, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    if not TRANSPOSE_B:
        b_desc = tl.make_tensor_descriptor(
            b_ptr + off_n * stride_bn,
            shape=[K, N],
            strides=[stride_bk, tl.constexpr(1)],
            block_shape=[BLOCK_K, BLOCK_N],
        )
        b_buf = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr + off_n * stride_bn,
            shape=[N, K],
            strides=[stride_bn, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1, reuse=a_buf)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    tl.assume(K_ITERS >= NUM_BUFFERS)
    producer = 0
    consumer = 0

    if L2_PREFETCH_DISTANCE > NUM_BUFFERS:
        for prefetch_offset in tl.static_range(NUM_BUFFERS, L2_PREFETCH_DISTANCE):
            prefetch_pred = prefetch_offset < K_ITERS
            _single_warp_per_simd_prefetch(
                a_desc,
                b_desc,
                prefetch_offset,
                0,
                0,
                prefetch_pred,
                BLOCK_K,
                TRANSPOSE_B,
            )

    for _ in tl.static_range(NUM_BUFFERS - 1):
        producer = _single_warp_per_simd_issue_loads_unpredicated(
            a_desc,
            b_desc,
            a_buf,
            b_buf,
            producer,
            0,
            0,
            BLOCK_K,
            NUM_BUFFERS,
            TRANSPOSE_B,
        )

    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 2)
    a0, b0 = _single_warp_per_simd_load_subtile(
        a_buf,
        b_buf,
        consumer,
        0,
        NUM_BUFFERS,
        TRANSPOSE_B,
        BLOCK_M,
        BLOCK_N,
        SUBTILE_LEN,
    )

    producer = _single_warp_per_simd_issue_loads_unpredicated(
        a_desc,
        b_desc,
        a_buf,
        b_buf,
        producer,
        0,
        0,
        BLOCK_K,
        NUM_BUFFERS,
        TRANSPOSE_B,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)

    tl.assume(K_ITERS > 0)
    for i in tl.range(0, K_ITERS):
        a1, b1 = _single_warp_per_simd_load_subtile(
            a_buf,
            b_buf,
            consumer,
            SUBTILE_LEN,
            NUM_BUFFERS,
            TRANSPOSE_B,
            BLOCK_M,
            BLOCK_N,
            SUBTILE_LEN,
        )
        acc = tl.dot(a0, b0, acc)

        if L2_PREFETCH_DISTANCE > 0:
            prefetch_iter = producer + L2_PREFETCH_DISTANCE - 1
            _single_warp_per_simd_prefetch_unpredicated(
                a_desc,
                b_desc,
                prefetch_iter,
                0,
                0,
                BLOCK_K,
                TRANSPOSE_B,
            )

        a2, b2 = _single_warp_per_simd_load_subtile(
            a_buf,
            b_buf,
            consumer,
            2 * SUBTILE_LEN,
            NUM_BUFFERS,
            TRANSPOSE_B,
            BLOCK_M,
            BLOCK_N,
            SUBTILE_LEN,
        )
        acc = tl.dot(a1, b1, acc)

        a3, b3 = _single_warp_per_simd_load_subtile(
            a_buf,
            b_buf,
            consumer,
            3 * SUBTILE_LEN,
            NUM_BUFFERS,
            TRANSPOSE_B,
            BLOCK_M,
            BLOCK_N,
            SUBTILE_LEN,
        )
        acc = tl.dot(a2, b2, acc)

        consumer += 1
        tlx.async_amd_descriptor_wait(NUM_BUFFERS - 2)
        pred = (i + 1) - epilogue_lb
        pred = (pred >> 31) & 1
        producer = _single_warp_per_simd_issue_loads(
            a_desc,
            b_desc,
            a_buf,
            b_buf,
            producer,
            0,
            0,
            pred,
            BLOCK_K,
            NUM_BUFFERS,
            TRANSPOSE_B,
        )
        a0, b0 = _single_warp_per_simd_load_subtile(
            a_buf,
            b_buf,
            consumer,
            0,
            NUM_BUFFERS,
            TRANSPOSE_B,
            BLOCK_M,
            BLOCK_N,
            SUBTILE_LEN,
        )
        acc = tl.dot(a3, b3, acc)

    c_view = tlx.local_view(c_buf, 0)
    tlx.local_store(c_view, acc.to(tlx.dtype_of(c_ptr)))
    tlx.async_amd_descriptor_store(c_desc, c_view, [off_m, off_n])
    tlx.async_amd_descriptor_wait(0)


def matmul(a: torch.Tensor, b: torch.Tensor, config=None) -> torch.Tensor:
    """C = A @ B using the baseline TDM-pipelined gfx1250 kernel."""
    cfg = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}
    if config is not None:
        cfg.update(config)

    assert a.is_contiguous() and b.is_contiguous(), "A and B must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, \
        "M, N, K must be multiples of their block sizes"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_tdm_pipelined_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def matmul_tdm_pipelined_single_warp_per_simd_schedule(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    NUM_BUFFERS: int = 2,
    TRANSPOSE_B: bool = False,
    L2_PREFETCH_DISTANCE: int = 2,
) -> torch.Tensor:
    """C = A @ B using the grouped single-warp-per-SIMD gfx1250 schedule."""
    assert a.is_contiguous() and b.is_contiguous(), "A and B must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    M, K = a.shape
    if TRANSPOSE_B:
        N, Kb = b.shape
    else:
        Kb, N = b.shape
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    BLOCK_K = 128
    assert NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2"
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, \
        "M, N, K must be multiples of their block sizes"
    assert K // BLOCK_K >= NUM_BUFFERS, "K must contain at least NUM_BUFFERS full tiles"
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_tdm_pipelined_single_warp_per_simd_schedule_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        TRANSPOSE_B=TRANSPOSE_B,
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
        num_warps=4,
        waves_per_eu=1,
    )
    return c
