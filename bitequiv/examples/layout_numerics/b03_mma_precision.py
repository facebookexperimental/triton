"""b03 — MMA selection & input_precision change the dot-product bits.

Source: ``lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`` rewrites
``tt.dot`` to a tensor-core ``#mma`` op (Blackwell: MMAv5/tcgen05 with a TMEM
accumulator). ``input_precision`` then picks the rounding:

  * ``"ieee"`` — full fp32; stays on the FMA path (no tensor cores). Accurate.
  * ``"tf32"`` — inputs rounded to TF32 (~10 mantissa bits) and run on the
    tensor cores (MMAv5 + TMEM). Faster, lossy.

BIT-CHANGING: the two paths use different accumulation hardware/order AND tf32
truncates the mantissa, so the results differ in bits (here by ~1e-2). This is the
on-ramp to M3, where the goal is to make the MMA path bitwise-equivalent to a
reference (cuBLAS). Requires datacenter Blackwell; self-skips elsewhere.

Run:  python bitequiv/examples/layout_numerics/b03_mma_precision.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_blackwell, is_cuda, show


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def _run(a, b, M, N, K, BM, BN, BK, prec):
    c = torch.empty(M, N, device="cuda")
    matmul_kernel[(1, )](a, b, c, M, N, K, BM, BN, BK, prec)
    torch.cuda.synchronize()
    return c


def main():
    M = N = K = 128
    BM = BN = 128
    BK = 64
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    ck_ieee = compile_only(matmul_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K, BM, BN, BK, "ieee",
                           grid=(1, ))
    ck_tf32 = compile_only(matmul_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K, BM, BN, BK, "tf32",
                           grid=(1, ))

    banner("b03 — input_precision selects the dot lowering path")
    print(f"    ieee:  tt.dot ops={count(ck_ieee, 'ttgir', 'tt.dot')}  TMEM ops={count(ck_ieee, 'ttgir', 'tmem')}"
          "   (FMA path, no tensor cores)")
    print(f"    tf32:  tt.dot ops={count(ck_tf32, 'ttgir', 'tt.dot')}  TMEM ops={count(ck_tf32, 'ttgir', 'tmem')}"
          "   (MMAv5 + TMEM accumulator)")
    show(ck_tf32, "ttgir", grep="tmem", limit=3, label="\nsample tf32 TMEM ops (tensor-core accumulator):")

    banner("b03 — the two paths differ in bits; tf32 is lossy")
    ref = (a.double() @ b.double())
    ci = _run(a, b, M, N, K, BM, BN, BK, "ieee")
    ct = _run(a, b, M, N, K, BM, BN, BK, "tf32")
    print(f"    ieee vs tf32 bitwise-equal: {torch.equal(ci, ct)}")
    print(f"    max|ieee - tf32|         : {(ci - ct).abs().max().item():.4e}")
    print(f"    ieee  max err vs fp64    : {(ci.double() - ref).abs().max().item():.4e}")
    print(f"    tf32  max err vs fp64    : {(ct.double() - ref).abs().max().item():.4e}  (TF32 truncates the mantissa)")

    assert not torch.equal(ci, ct), "ieee and tf32 dot must differ in bits"
    assert (ci.double() - ref).abs().max().item() < 1e-3, "ieee must be accurate"
    assert (ct.double() - ref).abs().max().item() < 0.5, "tf32 is lossy but in the right ballpark"
    print("\n[OK] MMA path/precision changed the dot-product bits (project M3 target: "
          "make the tensor-core path bitwise-equivalent to a reference).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    if not is_blackwell():
        print("[b03] skipped — MMAv5/TMEM precision example requires datacenter Blackwell.")
        raise SystemExit(0)
    main()
