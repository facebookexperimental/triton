"""Run case6 hand-written TLX TMA LayerNorm fwd vs torch."""

from __future__ import annotations

import sys

import handwritten
import torch
import torch.nn.functional as F
import triton


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)

    N = 384
    eps = 1e-5
    BLOCK_M = 8
    BLOCK_N = triton.next_power_of_2(N)  # 512, TMA tile (OOB zero-padded)
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    NUM_PERSIST = NUM_SMS * 4

    failed = 0
    for M in [4096, 16384, 65536]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        handwritten.layernorm_fwd_tma[(NUM_PERSIST,)](
            x, w, b, y, M, eps,
            row_stride=x.stride(0),
            N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_PERSIST=NUM_PERSIST,
            num_warps=4,
        )

        ref = F.layer_norm(x.float(), (N,), w.float(), b.float(), eps)
        err = (y.float() - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-9)
        nan = int(torch.isnan(y).sum().item())
        ok = rel < 1e-2 and nan == 0
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] M={M} N={N}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
