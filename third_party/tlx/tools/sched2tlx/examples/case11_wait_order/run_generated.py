"""Run emitter-generated case11 wait_order kernel on GPU and verify vs torch."""

from __future__ import annotations

import sys

import generated  # local file produced by `python -m sched2tlx`
import torch
import triton


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, KD, ON = 128, 64, 64, 128
    failed = 0
    for G, T in [(2, 4), (8, 16), (148, 32)]:
        x = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
        y = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
        w1 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
        w2 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
        v = torch.randn(BLOCK_N, ON, device="cuda", dtype=torch.float16) * 0.25
        out = torch.full((G * BLOCK_M, ON), float("nan"), device="cuda",
                         dtype=torch.float16)
        scale = 1.0
        generated.wait_order_kernel_nows[(G,)](x, y, w1, w2, v, out, scale, T,
                                               num_warps=4, num_ctas=1,
                                               num_stages=2)
        xf = x.float().view(G, T, BLOCK_M, KD)
        yf = y.float().view(G, T, BLOCK_M, KD)
        ref = torch.zeros(G, BLOCK_M, ON, device="cuda")
        for i in range(T):
            s1 = xf[:, i] @ w1.float().T
            s2 = yf[:, i] @ w2.float().T
            ref += (torch.exp2(s1 * scale) - s2).half().float() @ v.float()
        nan = int(torch.isnan(out).sum().item())
        rel = (out.float().view(G, BLOCK_M, ON) - ref).abs().max().item() / \
            ref.abs().max().item()
        ok = nan == 0 and rel < 2e-2
        print(f"[{'PASS' if ok else 'FAIL'}] G={G} T={T} nan={nan} rel={rel:.3e}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
