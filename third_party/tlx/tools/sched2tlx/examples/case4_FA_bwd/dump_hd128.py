"""Dump the HD=128 pre-modulo TTGIR for case4 (paper config: HEAD_DIM=128).

The kernel source (handwritten_nows.fa_bwd_dkdv_5mma) is parametric in
HEAD_DIM; this runner compiles it at HEAD_DIM=128, BLOCK_M=64, BLOCK_N=128
with WARP_SPEC=True — the modulo pre-dump form.  The launch itself may fail
after the modulo pass (in-compiler WS lowering) or at runtime; that is fine,
only the MLIR_ENABLE_DUMP capture matters:

    TRITON_ALWAYS_COMPILE=1 TRITON_USE_MODULO_SCHEDULE=1 \
    MLIR_ENABLE_DUMP=fa_bwd_dkdv_5mma MLIR_DUMP_PATH=$PWD/dump_hd128.mlir \
    env -u LD_LIBRARY_PATH <venv-python> dump_hd128.py
"""

import math

import torch
import triton
from handwritten_nows import fa_bwd_dkdv_5mma


def main(BH=4, N_CTX=2048, HEAD_DIM=128, BLOCK_M=64, BLOCK_N=128):
    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, device="cuda",
                                                dtype=torch.int8))
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(HEAD_DIM)
    q = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda",
                    dtype=torch.float16) * sm
    k, v, do = (torch.randn(BH, N_CTX, HEAD_DIM, device="cuda",
                            dtype=torch.float16) for _ in range(3))
    M = torch.zeros(BH, N_CTX, device="cuda", dtype=torch.float32)
    D = torch.zeros(BH, N_CTX, device="cuda", dtype=torch.float32)
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grid = (N_CTX // BLOCK_N, BH)
    try:
        fa_bwd_dkdv_5mma[grid](q, k, v, do, dq, dk, dv, M, D, HEAD_DIM,
                               HEAD_DIM, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
                               WARP_SPEC=True, num_warps=4, num_ctas=1,
                               num_stages=1)
        torch.cuda.synchronize()
        print("compile+launch OK")
    except Exception as e:
        print(f"launch failed (dump may still be valid): {type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]}")


if __name__ == "__main__":
    main()
