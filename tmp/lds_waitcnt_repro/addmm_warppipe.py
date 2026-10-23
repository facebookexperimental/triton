"""Standalone gfx950 warp-pipelined addmm that exposes the SIInsertWaitcnts loop soft-wait bug.

out = bias + x @ w.t(), BF16 [2033, 2560] x [2577, 2560]^T, BLOCK 64x64x64, 3 LDS buffers, 8 warps.
Extracted from the TorchTLX gfx950 warp-pipe addmm template; needs facebookexperimental/triton and MI350X.

  python addmm_warppipe.py                  # as compiled by this Triton build
  python addmm_warppipe.py --strip-ds-wait  # drop the explicit lgkmcnt(0) waits from the LLIR
"""

import argparse
import re
import sys

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton import knobs

M, N, K = 2033, 2577, 2560
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

# llvm.amdgcn.s.waitcnt(0xC07F) is lgkmcnt(0) alone on gfx9: what the Triton workaround emits.
DS_WAIT = re.compile(r"^\s*tail call void @llvm\.amdgcn\.s\.waitcnt\(i32 49279\).*\n", re.MULTILINE)


@triton.jit
def warppipe_core(a_ptr, b_ptr, smem_a, smem_b, a_base, b_base, offs_k, n_iters, k_tail, tail_width, stride_ak,
                  stride_bk, ACC_TYPE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr):
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = i * BLOCK_K
        a_offs = a_base + (k_start + offs_k[None, :]) * stride_ak
        b_offs = b_base[:, None] + (k_start + offs_k[None, :]) * stride_bk
        tok_a = tlx.async_load(a_ptr + a_offs.to(tl.int32), tlx.local_view(smem_a, i))
        tok_b = tlx.async_load(b_ptr + b_offs.to(tl.int32), tlx.local_view(smem_b, i))
        tlx.async_load_commit_group([tok_a, tok_b])

    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    a_tile = tlx.local_load(tlx.local_view(smem_a, 0))
    b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, 0)))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for tile_id in tl.range(0, n_iters - NUM_BUFFERS):
        prefetch_buf = (tile_id % NUM_BUFFERS).to(tl.int32)
        next_buf = ((tile_id + 1) % NUM_BUFFERS).to(tl.int32)
        k_prefetch = (tile_id + NUM_BUFFERS) * BLOCK_K

        tlx.async_load_wait_group(NUM_BUFFERS - 2)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False, out_dtype=ACC_TYPE)

        # This stage refills slot tile_id % 3 while other waves may still read it: the WAR the
        # LOCAL barrier at the loop head must order, which needs lgkmcnt(0) before s_barrier.
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_offs = a_base + (k_prefetch + offs_k[None, :]) * stride_ak
            b_offs = b_base[:, None] + (k_prefetch + offs_k[None, :]) * stride_bk
            tok_a = tlx.async_load(a_ptr + a_offs.to(tl.int32), tlx.local_view(smem_a, prefetch_buf))
            tok_b = tlx.async_load(b_ptr + b_offs.to(tl.int32), tlx.local_view(smem_b, prefetch_buf))
            tlx.async_load_commit_group([tok_a, tok_b])
            a_tile = tlx.local_load(tlx.local_view(smem_a, next_buf))
            b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, next_buf)))

    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False, out_dtype=ACC_TYPE)
    tlx.async_load_wait_group(0)

    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        buf = ((n_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS).to(tl.int32)
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, buf)))
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False, out_dtype=ACC_TYPE)

    a_offs = a_base + (k_tail + offs_k[None, :]) * stride_ak
    a_tail = tl.load(a_ptr + a_offs.to(tl.int32), mask=offs_k[None, :] < tail_width, other=0.0)
    b_offs = (k_tail + offs_k[:, None]) * stride_bk + b_base[None, :]
    b_tail = tl.load(b_ptr + b_offs.to(tl.int32), mask=offs_k[:, None] < tail_width, other=0.0)
    acc = tl.dot(a_tail, b_tail, acc, allow_tf32=False, out_dtype=ACC_TYPE)
    return acc


@triton.jit
def addmm_warppipe_kernel(bias_ptr, A, B, out_ptr):
    """out = bias + A @ B, with A [M, K] row-major and B [K, N] column-major."""
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8
    NUM_BUFFERS: tl.constexpr = 3
    NUM_XCDS: tl.constexpr = 8
    M = 2033
    N = 2577
    K = 2560
    stride_am = 2560
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2560

    pid = tl.program_id(0).to(tl.int32)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    XCD_CHUNK: tl.constexpr = 4
    grid_mn = grid_m * grid_n
    aligned = (grid_mn // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
    if pid < aligned:
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = (local_pid // XCD_CHUNK) * (NUM_XCDS * XCD_CHUNK) + xcd * XCD_CHUNK + (local_pid % XCD_CHUNK)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    a_base = offs_m[:, None] * stride_am
    b_base = offs_n * stride_bn
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    k_iters = K // BLOCK_K
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(A), NUM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(B), NUM_BUFFERS)
    k_tail = k_iters * BLOCK_K
    acc = warppipe_core(A, B, smem_a, smem_b, a_base, b_base, offs_k, k_iters, k_tail, K - k_tail, stride_ak, stride_bk,
                        ACC_TYPE, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS)

    bias = tl.load(bias_ptr + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask).to(tl.float32)
    tl.store(out_ptr + tl.broadcast_to(idx_n + N * idx_m, [BLOCK_M, BLOCK_N]), acc + bias, mask)


def install_strip_hook(stripped):

    def hook(*args):
        if not args:
            return "strip-ds-wait", "1"
        _, stages, _, _, _ = args
        make_llir = stages["llir"]

        def llir_without_ds_wait(src, metadata):
            llir, n = DS_WAIT.subn("", make_llir(src, metadata))
            stripped.append(n)
            return llir

        stages["llir"] = llir_without_ds_wait

    knobs.runtime.add_stages_inspection_hook = hook


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strip-ds-wait", action="store_true", help="remove the Triton lgkmcnt(0) workaround")
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stripped = []
    if args.strip_ds_wait:
        install_strip_hook(stripped)

    torch.manual_seed(args.seed)
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((N, ), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    expected = torch.addmm(bias, x, w.t()).float()

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    fails = []
    for it in range(args.repeats):
        out.zero_()
        addmm_warppipe_kernel[grid](bias, x, w, out, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0)
        bad = ~torch.isclose(out.float(), expected, atol=3e-2, rtol=3e-2)
        if bad.any():
            fails.append((it, int(bad.sum()), (out.float() - expected).abs().max().item()))
    torch.cuda.synchronize()

    if args.strip_ds_wait and stripped:
        print(f"stripped {sum(stripped)} lgkmcnt(0) waits from the LLIR")
    elif args.strip_ds_wait:
        print("loaded the stripped kernel from the Triton cache")
    print(f"failing launches: {len(fails)} / {args.repeats}")
    for it, nbad, diff in fails[:5]:
        print(f"  launch {it}: {nbad} mismatched, max abs diff {diff:.3g}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
