"""Perf: case9 blockwise fp8 scaled_mm — sched2tlx generated vs hand-written WS.

Benches the emitter-generated kernel (fixed modulo-derived partition, num_warps=8)
against the CUTLASS-style hand-written warp-specialized reference (autotuned) from
D111153835, both computing blockwise D=(A@B^T)*scales. do_bench medians on B200;
correctness vs the torch blockwise reference is checked first.
"""

from __future__ import annotations

import importlib
import sys

import torch
import triton

try:
    import generated
except ModuleNotFoundError:
    generated = importlib.import_module(
        (__package__ + ".generated") if __package__ else "generated"
    )
try:
    import handwritten
except ModuleNotFoundError:
    handwritten = importlib.import_module(
        (__package__ + ".handwritten") if __package__ else "handwritten"
    )

NUM_SMS = 148


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ref(a, b, scale_a, scale_b):
    M, K = a.shape
    N = b.shape[0]
    af, bf = a.to(torch.float32), b.to(torch.float32)
    out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    for g in range(K // 128):
        partial = af[:, g * 128 : (g + 1) * 128] @ bf[:, g * 128 : (g + 1) * 128].t()
        sa = scale_a[:, g][:, None]
        sb = scale_b[:, g].repeat_interleave(128)[None, :]
        out += partial * sa * sb
    return out


def _gen(a, b, c, scale_a, scale_b, M, N, K):
    generated._scaled_mm_blockwise[(NUM_SMS,)](
        a, b, c, scale_a, scale_b, M, N, K,
        a.stride(0), b.stride(0), c.stride(0), scale_a.stride(1), scale_b.stride(0),
        num_warps=8, num_ctas=1, num_stages=2,
    )
    return c


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
              (8192, 8192, 8192)]
    print(f"{'shape':<20}{'gen rel':<10}{'hw rel':<10}"
          f"{'gen TF':<10}{'hw TF':<10}{'gen/hw':<8}")
    print("-" * 70)
    failed = 0
    for M, N, K in shapes:
        a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        scale_a = (torch.rand(M, K // 128, device="cuda", dtype=torch.float32)
                   .t().contiguous().t())
        scale_b = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32)
        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        ref = _ref(a, b, scale_a, scale_b)
        cg = _gen(a, b, c, scale_a, scale_b, M, N, K)
        ch = handwritten.blackwell_scaled_mm_ws(a, b, scale_a, scale_b,
                                                scale_mode="blockwise")
        rel_g = (cg.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        rel_h = (ch.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        ok = rel_g < 5e-2 and rel_h < 5e-2
        failed += 0 if ok else 1

        gen_ms = triton.testing.do_bench(
            lambda: _gen(a, b, c, scale_a, scale_b, M, N, K), warmup=25, rep=100)
        hw_ms = triton.testing.do_bench(
            lambda: handwritten.blackwell_scaled_mm_ws(a, b, scale_a, scale_b,
                                                       scale_mode="blockwise"),
            warmup=25, rep=100)
        flop = 2 * M * N * K
        gtf = flop / (gen_ms * 1e-3) / 1e12
        htf = flop / (hw_ms * 1e-3) / 1e12
        print(f"{f'{M}x{N}x{K}':<20}{rel_g:<10.2e}{rel_h:<10.2e}"
              f"{gtf:<10.0f}{htf:<10.0f}{gtf / htf:<8.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
