"""Case 6 — LayerNorm forward non-WS baseline: correctness vs torch.

Also the IR-dump source: run with
  MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=$PWD/dump.mlir \
  TRITON_USE_MODULO_SCHEDULE=1 python3 run_layernorm_fwd_nows.py
to capture the pre-modulo TTGIR for the scheduling study.
"""

from __future__ import annotations

import sys

import layernorm_fwd_nows
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
    BLOCK_M = 8
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

    failed = 0
    for M in [4096, 16384, 65536]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        grid = (NUM_SMS,)
        layernorm_fwd_nows.layernorm_fwd_nows[grid](
            x, w, b, y, M, eps, N=N, BLOCK_M=BLOCK_M, num_warps=4
        )

        ref = F.layer_norm(x.float(), (N,), w.float(), b.float(), eps)
        err = (y.float() - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-9)
        ok = rel < 1e-2
        marker = "PASS" if ok else "FAIL"
        nan = int(torch.isnan(y).sum().item())
        print(f"[{marker}] M={M} N={N}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
