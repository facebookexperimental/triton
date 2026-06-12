"""Empirical test: 3-WG warp-specialized case6 (TMA load WG / TMA store WG /
compute default) vs the 1-WG software-pipelined generated kernel. Both use the
TMA store, so this isolates the partition question: does splitting load/store/
compute into separate warp groups beat one software-pipelined warp group?

Hypothesis (to verify, not assume): load+store share the one TMA engine, so 3-WG
can't beat 1-WG. Measured below.
"""

from __future__ import annotations

import sys

import generated  # the 1-WG SWP kernel
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def layernorm_3wg(X, W, B, Y, M, eps, N: tl.constexpr, BLOCK_M: tl.constexpr):
    NB: tl.constexpr = 2
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ndiv = tl.cdiv(M, BLOCK_M)
    x_desc = tl.make_tensor_descriptor(X, [M, N], [N, 1], [BLOCK_M, N])
    y_desc = tl.make_tensor_descriptor(Y, [M, N], [N, 1], [BLOCK_M, N])
    x_smem = tlx.local_alloc((BLOCK_M, N), tlx.dtype_of(X), NB)
    y_smem = tlx.local_alloc((BLOCK_M, N), tlx.dtype_of(Y), NB)
    x_full = tlx.alloc_barriers(num_barriers=NB, arrive_count=1)
    x_empty = tlx.alloc_barriers(num_barriers=NB, arrive_count=1)
    y_full = tlx.alloc_barriers(num_barriers=NB, arrive_count=1)
    y_empty = tlx.alloc_barriers(num_barriers=NB, arrive_count=1)
    cols = tl.arange(0, N)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    eb: tl.constexpr = BLOCK_M * N * tlx.size_of(tlx.dtype_of(X))

    with tlx.async_tasks():
        # compute (default, 4 warps)
        with tlx.async_task("default"):
            for tile_id in range(pid, ndiv, nprog):
                _it = (tile_id - pid) // nprog
                slot = _it % NB
                ph = (_it // NB) & 1
                tlx.barrier_wait(x_full[slot], ph)
                x = tlx.local_load(x_smem[slot]).to(tl.float32)
                tlx.barrier_arrive(x_empty[slot], 1)
                sx = tl.sum(x, axis=1, keep_dims=True)
                sxx = tl.sum(x * x, axis=1, keep_dims=True)
                mean = sx / N
                var = sxx / N - mean * mean
                rstd = 1.0 / tl.sqrt(var + eps)
                y = (x - mean) * rstd * w[None, :] + b[None, :]
                tlx.barrier_wait(y_empty[slot], ph ^ 1)
                tlx.local_store(y_smem[slot], y.to(tlx.dtype_of(Y)))
                tlx.fence_async_shared()
                tlx.barrier_arrive(y_full[slot], 1)
        # TMA load (1 warp)
        with tlx.async_task(num_warps=1, num_regs=24):
            for tile_id in range(pid, ndiv, nprog):
                _it = (tile_id - pid) // nprog
                slot = _it % NB
                ph = (_it // NB) & 1
                tlx.barrier_wait(x_empty[slot], ph ^ 1)
                tlx.barrier_expect_bytes(x_full[slot], eb)
                tlx.async_descriptor_load(
                    x_desc, x_smem[slot], [tile_id * BLOCK_M, 0], x_full[slot]
                )
        # TMA store (1 warp)
        with tlx.async_task(num_warps=1, num_regs=24):
            for tile_id in range(pid, ndiv, nprog):
                _it = (tile_id - pid) // nprog
                slot = _it % NB
                ph = (_it // NB) & 1
                tlx.barrier_wait(y_full[slot], ph)
                tlx.async_descriptor_store(y_desc, y_smem[slot], [tile_id * BLOCK_M, 0])
                tlx.async_descriptor_store_wait(0)
                tlx.barrier_arrive(y_empty[slot], 1)


def _time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    N, BLOCK_M, eps = 512, 8, 1e-5
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"{'M':>8} | {'1WG-SWP GB/s':>12} | {'3WG-WS GB/s':>11} | {'3wg/1wg':>7}")
    print("-" * 52)
    for M in [16384, 65536, 262144]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        y1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        y3 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        bytes_moved = M * N * 2 * 2

        def run1():
            generated.layernorm_fwd_nows[(NUM_SMS,)](x, w, b, y1, M, eps, num_warps=4)

        def run3():
            layernorm_3wg[(NUM_SMS,)](x, w, b, y3, M, eps, N=N, BLOCK_M=BLOCK_M)

        run1()
        run3()
        torch.cuda.synchronize()
        ref = F.layer_norm(x.float(), (N,), w.float(), b.float(), eps)
        for nm, y in (("1wg", y1), ("3wg", y3)):
            rel = (y.float() - ref).abs().max().item() / max(
                ref.abs().max().item(), 1e-9
            )
            if rel > 1e-2:
                print(f"  WARN {nm} M={M} rel={rel:.2e}")
        g1 = bytes_moved / (_time(run1) * 1e-3) / 1e9
        g3 = bytes_moved / (_time(run3) * 1e-3) / 1e9
        print(f"{M:>8} | {g1:>12.1f} | {g3:>11.1f} | {g3/g1:>6.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
