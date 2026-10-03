"""v7_slice — N-sliced B with manual 4-region pipeline.

Exact translation of Gluon v7: split B into left/right halves,
manual prologue/loop/epilogue, num_stages=1.

v7 is the last 4-wave step: one wave both loads and computes. Three details
match the Gluon kernel and matter in that regime:

  * Bank-conflict-free LDS layout. The padded layout the compiler infers for
    direct-to-LDS buffers keeps plain row-major order: 4 bank conflicts per LDS
    instruction (SQ_LDS_BANK_CONFLICT 1.7e7 per 4096x4096x8192 launch). The
    explicit bit permutations below are the Gluon tutorial's and measure 0.
  * AGPR accumulator pins. `tlx.amd_dot(..., cd_regclass="a")` pins every MFMA
    tile's accumulator input and result to AGPRs. A 256x256 FP32 accumulator on
    4 waves is 256 registers per lane and cannot stay in VGPRs; without pins
    the loop carries ~140 `v_accvgpr` copies, with pins none.
  * Scalar K offsets. The K offset advances the scalar base pointer instead of
    being added to every element of the i32 offset tensors, which removes the
    vector adds from the hot loop.

These make the loop schedulable inside a wave. With the LLIR scheduler pass
plugin (../../plugins/llir_scheduler) loaded, the MFMAs are interleaved with the
memory work and the hot loop is instruction-for-instruction the scheduled Gluon
v7 loop. On their own they are not a consistent win, because LLVM's default
schedule still decides the outcome. See the README for numbers.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# Padded LDS layouts as explicit offset bases: K stays contiguous (so the
# direct-to-LDS writes stay coalesced), and the bits of the other dimension are
# permuted so the `ds_read_b128`s of one wave fall into distinct banks.
# A is [M, K] = [256, 64]: K bits, then rows 16/32/64, rows 1/2/4/8, row 128.
# The position of the row-128 bit matters: placed right after row 64 it costs
# 2 conflicts per LDS instruction on a 256-row tile.
_A_LDS_BASES = tl.constexpr([[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0],
                             [2, 0], [4, 0], [8, 0], [128, 0]])
# B is [K, N/2] = [64, 128]: K bits, then columns 16/32/64, columns 1/2/4/8.
_B_LDS_BASES = tl.constexpr([[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1],
                             [0, 2], [0, 4], [0, 8]])


@triton.jit
def v7_slice(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
             stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    HALF_N: tl.constexpr = BLOCK_N // 2

    a_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _A_LDS_BASES, [BLOCK_M, BLOCK_K])
    b_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], _B_LDS_BASES, [BLOCK_K, HALF_N])
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_shared)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_off = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    bl_off = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    br_off = bl_off + HALF_N * stride_bn
    # Running K offsets, applied to the scalar base pointers.
    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    acc_left = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    iterMax: tl.constexpr = K // BLOCK_K

    # ── Prologue ──
    # Buffer 0: A + B_left (group 0), B_right (group 1)
    tlx.buffer_load_to_local(smem_a[0], a_ptr + a_k, a_off)
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr + b_k, bl_off)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[0], b_ptr + b_k, br_off)
    tlx.async_load_commit_group()

    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    # Buffer 1: A + B_left (group 2), B_right (group 3)
    tlx.buffer_load_to_local(smem_a[1], a_ptr + a_k, a_off)
    tlx.buffer_load_to_local(smem_b_left[1], b_ptr + b_k, bl_off)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[1], b_ptr + b_k, br_off)
    tlx.async_load_commit_group()

    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    # Wait for group 0 (A + B_left in buffer 0)
    tlx.async_load_wait_group(3)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)

    # ── Main loop: step 2, processes 2 K iterations per body ──
    for k in tl.range(0, iterMax - 2, 2, num_stages=1):
        # ──── Region 0 (g_idx=0, l_idx=1) ────
        acc_left = tlx.amd_dot(a, b_left, acc_left, cd_regclass="a")

        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(smem_b_right[0], relaxed=True)

        tlx.buffer_load_to_local(smem_a[0], a_ptr + a_k, a_off)
        tlx.buffer_load_to_local(smem_b_left[0], b_ptr + b_k, bl_off)
        tlx.async_load_commit_group()

        # ──── Region 1 ────
        acc_right = tlx.amd_dot(a, b_right, acc_right, cd_regclass="a")

        tlx.async_load_wait_group(2)
        a = tlx.local_load(smem_a[1], relaxed=True)
        b_left = tlx.local_load(smem_b_left[1], relaxed=True)

        tlx.buffer_load_to_local(smem_b_right[0], b_ptr + b_k, br_off)
        tlx.async_load_commit_group()

        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

        # ──── Region 2 (g_idx=1, l_idx=0) ────
        acc_left = tlx.amd_dot(a, b_left, acc_left, cd_regclass="a")

        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(smem_b_right[1], relaxed=True)

        tlx.buffer_load_to_local(smem_a[1], a_ptr + a_k, a_off)
        tlx.buffer_load_to_local(smem_b_left[1], b_ptr + b_k, bl_off)
        tlx.async_load_commit_group()

        # ──── Region 3 ────
        acc_right = tlx.amd_dot(a, b_right, acc_right, cd_regclass="a")

        tlx.async_load_wait_group(2)
        a = tlx.local_load(smem_a[0], relaxed=True)
        b_left = tlx.local_load(smem_b_left[0], relaxed=True)

        tlx.buffer_load_to_local(smem_b_right[1], b_ptr + b_k, br_off)
        tlx.async_load_commit_group()

        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

    # ── Epilogue: iterMax - 2 ──
    # Region 0
    acc_left = tlx.amd_dot(a, b_left, acc_left, cd_regclass="a")
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(smem_b_right[0], relaxed=True)

    # Region 1
    acc_right = tlx.amd_dot(a, b_right, acc_right, cd_regclass="a")
    a = tlx.local_load(smem_a[1], relaxed=True)
    b_left = tlx.local_load(smem_b_left[1], relaxed=True)

    # ── Epilogue: iterMax - 1 ──
    # Region 2
    acc_left = tlx.amd_dot(a, b_left, acc_left, cd_regclass="a")
    b_right = tlx.local_load(smem_b_right[1], relaxed=True)

    # Store left
    c_left = acc_left.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    c_left_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn_left[None, :]
    c_left_mask = (offs_cm[:, None] < M) & (offs_cn_left[None, :] < N)
    tl.store(c_left_ptrs, c_left, mask=c_left_mask)

    # Region 3
    acc_right = tlx.amd_dot(a, b_right, acc_right, cd_regclass="a")

    # Store right
    c_right = acc_right.to(tlx.dtype_of(c_ptr))
    offs_cn_right = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)
    c_right_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn_right[None, :]
    c_right_mask = (offs_cm[:, None] < M) & (offs_cn_right[None, :] < N)
    tl.store(c_right_ptrs, c_right, mask=c_right_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    v7_slice[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=1,
                   matrix_instr_nonkdim=16)
    return c
