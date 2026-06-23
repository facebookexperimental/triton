"""10 — MMA input precision: a POSITIVE numerics example on GEMM.

Pipeline: for an fp32 ``tl.dot`` the **input precision** picks the lowering.
``input_precision="tf32"`` rounds each operand to TF32 (10-bit mantissa) and uses
one tensor-core pass (PTX ``wgmma...f32.tf32.tf32``); ``input_precision="ieee"``
keeps full fp32 (no tensor-core wgmma — an FMA / 3xTF32 path). Same ``tl.dot``,
different ``tritongpu-accelerate-matmul`` / ``tritongpu-F32DotTC`` outcome.

Unlike 09 (instruction choice is bit-neutral), this knob is BIT-CHANGING: TF32
truncates the inputs, so the products differ from the IEEE result. We compile both,
show the PTX instruction differs, and assert the runtime results differ in bits
while both stay close to a high-precision reference.

Run:  python python/tutorials/compilation-pipeline/10_mma_precision_numerics.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, diff, is_cuda


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def _run(a, b, M, N, K, prec):
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    dot_kernel[(1, )](a, b, out, M, N, K, prec)
    torch.cuda.synchronize()
    return out


def main():
    M = N = K = 128
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    tf32 = compile_only(dot_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K, "tf32", grid=(1, ))
    ieee = compile_only(dot_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K, "ieee", grid=(1, ))

    banner("10 — same tl.dot, the input-precision knob picks a different PTX instruction")
    print(f"    tf32:  PTX wgmma(...tf32) ops = {count(tf32, 'ptx', 'wgmma.mma_async')}")
    print(f"    ieee:  PTX wgmma         ops = {count(ieee, 'ptx', 'wgmma')}")
    # The instruction kind (note `f32.tf32.tf32`) shows up in the warp_group_dot's
    # inputPrecision attribute — diff the two TTGIRs to see only that op differ.
    diff(tf32, ieee, "ttgir", grep="warp_group_dot", label_a="tf32", label_b="ieee", limit=8)

    banner("10 — the precision knob changes the BITS (TF32 truncates the inputs)")
    c_tf32 = _run(a, b, M, N, K, "tf32")
    c_ieee = _run(a, b, M, N, K, "ieee")
    ref = (a.double() @ b.double())  # high-precision reference

    assert not torch.equal(c_tf32, c_ieee), "tf32 and ieee must differ in bits"
    err_tf32 = (c_tf32.double() - ref).abs().max().item()
    err_ieee = (c_ieee.double() - ref).abs().max().item()
    print(f"    max|tf32 - ref| = {err_tf32:.3e}")
    print(f"    max|ieee - ref| = {err_ieee:.3e}   (ieee is closer — it keeps full fp32 inputs)")
    assert err_ieee < err_tf32, "ieee should track the high-precision reference more closely"
    print("\n[OK] tf32 and ieee dots are NOT bitwise-equal — input precision is a real numerics"
          " knob (the equivalence checker must treat these configs as inequivalent).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    if torch.cuda.get_device_capability()[0] < 8:
        print("[10] skipped — TF32 tensor-core path requires Ampere or newer (sm_80+).")
        raise SystemExit(0)
    main()
