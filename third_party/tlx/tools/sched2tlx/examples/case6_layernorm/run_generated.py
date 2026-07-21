"""Run the sched2tlx-generated case6 LayerNorm fwd kernel vs torch.

The generated kernel bakes N=512 and BLOCK_M=8 in as literals (from the schedule
graph); its signature is (X, W, B, Y, M, eps). Single warp group (no WS).
"""

from __future__ import annotations

import sys

import generated
import torch
import torch.nn.functional as F
import triton


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)

    N = 512
    eps = 1e-5
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

    # Launch hints emitted by the modulo pass for memory-bound kernels
    # (absent → legacy launch: 1x-SMS grid, register file auto-fill).
    grid = NUM_SMS * getattr(generated, "RECOMMENDED_GRID_MULTIPLIER", 1)
    launch_kw = {"num_warps": 4}
    if (mreg := getattr(generated, "RECOMMENDED_MAXNREG", None)) is not None:
        launch_kw["maxnreg"] = mreg

    failed = 0
    for M in [4096, 16384, 65536]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        generated.layernorm_fwd_nows[(grid, )](x, w, b, y, M, eps, **launch_kw)

        ref = F.layer_norm(x.float(), (N, ), w.float(), b.float(), eps)
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
