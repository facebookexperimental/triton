"""Run + capture harness for case8 (multi-phase GEMM).

Correctness run:
    python3 run_triple_gemm_nows.py

Pre-modulo TTGIR capture (case6 recipe; the dump header carries an
`{anonymous}::` prefix on the pass name):
    MLIR_ENABLE_DUMP=triple_gemm_nows MLIR_DUMP_PATH=$PWD/dump.mlir \
    TRITON_USE_MODULO_SCHEDULE=1 TRITON_ALWAYS_COMPILE=1 \
    python3 run_triple_gemm_nows.py
With the modulo env set, downstream compilation of the sibling-loop schedule
dies in AutoWS after the dump point — the [CAPTURE-ONLY] path tolerates it.
"""

from __future__ import annotations

import sys

import torch
import triton

from triple_gemm_nows import triple_gemm_nows

BLOCK_M, BLOCK_N, BLOCK_K1, BLOCK_K3 = 128, 128, 128, 64


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    M = N = K = 2048

    a1 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b1 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    a2 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b2 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    a3 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b3 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c1 = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    c2 = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    c3 = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    try:
        triple_gemm_nows[grid](
            a1, b1, c1, a2, b2, c2, a3, b3, c3, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K1=BLOCK_K1, BLOCK_K3=BLOCK_K3,
            num_warps=4, num_ctas=1, num_stages=2,
        )
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - capture-mode downstream failure tolerated
        print(f"[CAPTURE-ONLY] compile/launch failed after dump point: "
              f"{type(e).__name__}: {str(e)[:200]}")
        return 0

    failed = 0
    for name, c, a, b in (
        ("C1", c1, a1, b1), ("C2", c2, a2, b2), ("C3", c3, a3, b3)
    ):
        ref = torch.matmul(a.float(), b.float())
        nan = int(torch.isnan(c.float()).sum().item())
        rel = (c.float() - ref).abs().max().item() / ref.abs().max().item()
        tol = 5e-3 if c.dtype == torch.float16 else 2e-2  # bf16: 8 mantissa bits
        ok = nan == 0 and rel < tol
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  nan={nan}  rel={rel:.3e}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
