"""Generate real compiled-TTGIR fixtures for the MLIR-native reduction-order tests.

Run once on a GPU node (needs a built triton + ptxas) to (re)write
``bitequiv/tests/ttgir/*.ttgir``. The unit tests then PARSE those committed files
(CPU-only) — no GPU needed at test time. Regenerate if the compiler output format
changes.

    cd <m1 worktree>
    CC=/usr/bin/gcc PYTHONPATH=<m1>/python python bitequiv/tests/gen_ttgir_fixtures.py
"""
import os

import torch

import triton
import triton.language as tl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttgir")
# Strip this checkout's absolute path from the compiled IR's loc(...) debug strings so the
# committed fixtures stay relative/portable. The reduction-order analysis ignores loc(...),
# so this is purely cosmetic (it never changes a descriptor).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
U = tl.ReductionOrdering.UNORDERED
I = tl.ReductionOrdering.INNER_TREE


def _relativize(ttgir):
    """Make absolute source paths in loc(...) debug strings relative to the repo root."""
    return ttgir.replace(_REPO_ROOT + "/", "")


@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0, reduction_ordering=ORD))


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def mul_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=1.0)
    tl.store(dst, tl.reduce(x, 0, _mul, reduction_ordering=ORD))


@triton.jit
def argmin_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=float("inf"))
    tl.store(dst, tl.argmin(x, axis=0, reduction_ordering=ORD))


@triton.jit
def gemm_kernel(a, b, c, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
    offm = tl.arange(0, BM)
    offn = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offk = k0 + tl.arange(0, BK)
        at = tl.load(a + offm[:, None] * sa0 + offk[None, :] * sa1)
        bt = tl.load(b + offk[:, None] * sb0 + offn[None, :] * sb1)
        acc = tl.dot(at, bt, acc, input_precision=PREC)
    tl.store(c + offm[:, None] * sc0 + offn[None, :] * sc1, acc)


def _sum(block, ordv, nw):
    src = torch.randn(block, device="cuda")
    dst = torch.empty(1, device="cuda")
    return sum_kernel.warmup(src, dst, block, BLOCK=block, ORD=ordv, num_warps=nw, grid=(1, )).asm["ttgir"]


def _mulred(ordv, nw):
    src = torch.randn(8192, device="cuda")
    dst = torch.empty(1, device="cuda")
    return mul_kernel.warmup(src, dst, 8192, BLOCK=8192, ORD=ordv, num_warps=nw, grid=(1, )).asm["ttgir"]


def _argmin(nw):
    src = torch.randn(8192, device="cuda")
    dst = torch.empty(1, device="cuda", dtype=torch.int32)
    return argmin_kernel.warmup(src, dst, 8192, BLOCK=8192, ORD=U, num_warps=nw, grid=(1, )).asm["ttgir"]


def _gemm(prec):
    M = N = K = 128
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda")
    return gemm_kernel.warmup(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                              c.stride(1), PREC=prec, BM=M, BN=N, BK=K, num_warps=4, grid=(1, )).asm["ttgir"]


def main():
    os.makedirs(OUT, exist_ok=True)
    fixtures = {
        "sum_uno_nw2": _sum(8192, U, 2),
        "sum_uno_nw4": _sum(8192, U, 4),
        "sum_uno_nw4b": _sum(8192, U, 4),
        "sum_uno_nw8": _sum(8192, U, 8),
        "sum_inner_nw2": _sum(8192, I, 2),
        "sum_inner_nw8": _sum(8192, I, 8),
        "sum_uno_block2048_nw4": _sum(2048, U, 4),
        "sum_uno_block4096_nw4": _sum(4096, U, 4),
        "mul_uno_nw4": _mulred(U, 4),
        "argmin_nw2": _argmin(2),
        "argmin_nw4": _argmin(4),
        "gemm_ieee": _gemm("ieee"),
        "gemm_tf32": _gemm("tf32"),
    }
    for name, ttgir in fixtures.items():
        ttgir = _relativize(ttgir)
        with open(os.path.join(OUT, name + ".ttgir"), "w") as f:
            f.write(ttgir)
        print(f"wrote {name}.ttgir ({len(ttgir)} bytes)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Requires a CUDA GPU to compile fixtures.")
    main()
