"""b04 — F32DotTC: emulating fp32 dots on tensor cores (3x TF32) changes bits.

Source: ``lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp``. With
``input_precision="tf32x3"`` a single fp32 ``tt.dot`` is rewritten into THREE
TF32 tensor-core dots whose partial products recover ~fp32 accuracy — much more
accurate than a single ``tf32`` pass (b03), while still using the tensor cores.

The point for this project: ``tf32x3`` reaches essentially the same *accuracy* as
the ``ieee`` FMA path, but its accumulation order is different, so the results are
**NOT bitwise-equal**. "Same accuracy" is not "same bits" — bitwise equivalence is
a stricter, order-level property.

BIT-CHANGING. The 3-pass emulation shows up as extra tensor-core / TMEM work in
TTGIR. Requires datacenter Blackwell; self-skips elsewhere.

Run:  python bitequiv/examples/layout_numerics/b04_f32_dot_tc.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_blackwell, is_cuda


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
    ref = a.double() @ b.double()

    banner("b04 — F32DotTC: tf32x3 does 3x the tensor-core work of tf32")
    for prec in ("ieee", "tf32", "tf32x3"):
        ck = compile_only(matmul_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K, BM, BN, BK, prec, grid=(1, ))
        print(f"    {prec:7s}:  tmem_alloc ops = {count(ck, 'ttgir', 'tmem_alloc')}   "
              f"total TMEM ops = {count(ck, 'ttgir', 'tmem')}")

    banner("b04 — tf32x3 matches ieee ACCURACY but not its BITS")
    ci = _run(a, b, M, N, K, BM, BN, BK, "ieee")
    ct = _run(a, b, M, N, K, BM, BN, BK, "tf32")
    c3 = _run(a, b, M, N, K, BM, BN, BK, "tf32x3")
    for name, c in (("ieee  ", ci), ("tf32  ", ct), ("tf32x3", c3)):
        print(f"    {name}:  max err vs fp64 = {(c.double() - ref).abs().max().item():.3e}")
    print(f"\n    ieee vs tf32x3 bitwise-equal: {torch.equal(ci, c3)}   "
          f"max|ieee - tf32x3| = {(ci - c3).abs().max().item():.3e}")

    e_ieee = (ci.double() - ref).abs().max().item()
    e_tf32x3 = (c3.double() - ref).abs().max().item()
    assert not torch.equal(ci, c3), "tf32x3 and ieee must differ in bits"
    assert e_tf32x3 < 1e-3 and e_ieee < 1e-3, "both ieee and tf32x3 are fp32-accurate"
    assert e_tf32x3 < (ct.double() - ref).abs().max().item(), "tf32x3 must be far more accurate than single-pass tf32"
    print("\n[OK] tf32x3 recovers fp32-level accuracy on tensor cores yet differs from the "
          "FMA path in bits — accuracy != bitwise equivalence.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    if not is_blackwell():
        print("[b04] skipped — F32DotTC tensor-core example requires datacenter Blackwell.")
        raise SystemExit(0)
    main()
