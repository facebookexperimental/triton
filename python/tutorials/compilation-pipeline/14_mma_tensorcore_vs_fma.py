"""14 — Tensor-core MMA vs the FMA fallback: when does tl.dot use the tensor core?

Pipeline: ``tritongpu-accelerate-matmul`` only rewrites a ``tt.dot`` into a
tensor-core MMA when the operand types/precision map onto one. For an fp32
``tl.dot`` the ``input_precision`` knob decides:

  * ``input_precision="tf32"`` -> operands round to TF32, the dot becomes a
    tensor-core MMA: a ``#ttg.nvidia_mma`` layout + ``warp_group_dot`` ->
    PTX ``wgmma.mma_async``.
  * ``input_precision="ieee"`` -> full fp32 is kept, there is **no** tensor-core
    instruction for it, so the dot stays on the generic register layout and
    lowers to a scalar **FMA** loop (PTX ``fma.rn.f32``) — no ``nvidia_mma``
    layout at all.

This is the same knob 10 studies, but the lens here is the **lowering structure**
(which ops/layout get emitted), not the numerics. Knowing whether a dot is on the
tensor core or on FMA matters for layouts and correctness: the two paths have
different accumulation structure, so they are generally *not* bitwise-equal (10
shows that side).

Both arches (Hopper sm_90, Blackwell sm_100) are cross-compiled so the contrast
is visible on any host; correctness runs on the host's own arch.

Run:  python python/tutorials/compilation-pipeline/14_mma_tensorcore_vs_fma.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, cc_name, compile_for_target, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


_SIG = {
    "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr", "PREC":
    "constexpr"
}


def _compile(cc, prec, M, N, K):
    return compile_for_target(dot_kernel, _SIG, {"M": M, "N": N, "K": K, "PREC": prec}, cc=cc, num_warps=4)


def main():
    M = N = 128
    K = 64

    banner("14 — tf32 takes the tensor core; ieee falls back to a scalar FMA loop")
    for cc in (90, 100):
        tc = _compile(cc, "tf32", M, N, K)
        fma = _compile(cc, "ieee", M, N, K)
        print(f"    {cc_name(cc)}")
        print(f"      tf32 (TC) :  #nvidia_mma={count(tc, 'ttgir', 'nvidia_mma')}"
              f"  wgmma={count(tc, 'ptx', 'wgmma.mma_async')}  tcgen05={count(tc, 'ptx', 'tcgen05.mma')}"
              f"  fma.rn={count(tc, 'ptx', 'fma.rn'):>5}")
        print(f"      ieee (FMA):  #nvidia_mma={count(fma, 'ttgir', 'nvidia_mma')}"
              f"  wgmma={count(fma, 'ptx', 'wgmma.mma_async')}  tcgen05={count(fma, 'ptx', 'tcgen05.mma')}"
              f"  fma.rn={count(fma, 'ptx', 'fma.rn'):>5}")
    print("    ^ ieee has NO tensor-core op and NO #nvidia_mma layout — it is a plain FMA dot.")

    hop_tc = _compile(90, "tf32", M, N, K)
    hop_fma = _compile(90, "ieee", M, N, K)
    show(hop_tc, "ttgir", grep=["nvidia_mma =", "warp_group_dot"], limit=2, label="\n  tf32 -> tensor-core op:")
    show(hop_fma, "ttgir", grep=["tt.dot", "arith.mulf", "arith.addf"], limit=2, label="  ieee -> generic dot/FMA:")

    # The pass responsible: `tritongpu-accelerate-matmul` (the tf32 case rewrites
    # tt.dot into warp_group_dot; run on the host arch so the marker prints).
    banner("14 — the pass responsible: tritongpu-accelerate-matmul (tf32, host arch)")
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    dumps = dump_passes(dot_kernel, a, b, c, M, N, K, "tf32", num_warps=4, grid=(1, ))
    pass_diff(dumps, "accelerate-matmul", grep=["tt.dot", "warp_group_dot", "nvidia_mma"], limit=10)

    # Correctness on the host arch: both paths track the reference (TF32 is looser).
    ref = (a.double() @ b.double())
    for prec in ("tf32", "ieee"):
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        dot_kernel[(1, )](a, b, out, M, N, K, prec, num_warps=4)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.double(), ref, atol=2e-2, rtol=2e-2)
    print("\n[OK] tf32 lowers to a tensor-core MMA, ieee to an FMA dot; both match the"
          " reference (which path you get controls layout — and bitwise behavior, see 10).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    if torch.cuda.get_device_capability()[0] < 8:
        print("[14] skipped — TF32 tensor-core path requires Ampere or newer (sm_80+).")
        raise SystemExit(0)
    main()
