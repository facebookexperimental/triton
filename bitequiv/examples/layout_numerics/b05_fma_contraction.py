"""b05 — FMA contraction: mul+add fused into one rounding (decided BELOW TTGIR).

Source: ``lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp`` emits
``LLVM::FMulAddOp`` for float multiply-accumulate; for elementwise ``a*b+c`` the
LLVM/PTX backend likewise contracts the multiply and add into a single fused
``fma.rn.f32`` — ONE rounding instead of two (round(a*b) then round(+c)).

Two things to take away:

1. **It changes bits.** A fused ``a*b+c`` (Triton) differs from a separately
   rounded ``(a*b)+c`` (torch eager) — and is actually closer to the fp64 result.

2. **It is invisible at TTGIR.** The TTGIR still shows separate ``arith.mulf`` +
   ``arith.addf``; the fusion only appears in the PTX (``fma.rn.f32``, no
   ``mul.f32``). So a TTGIR-level equivalence checker cannot see it — this is
   exactly why the project needs a PTX / ``ptxas --fmad`` backstop (see
   ``equivalence-check-level-ttgir-vs-ptx.md`` and the M1 ``equivalence.py`` note).

BIT-CHANGING. Pure CUDA (no Blackwell requirement).

Run:  python bitequiv/examples/layout_numerics/b05_fma_contraction.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda, show


@triton.jit
def madd_kernel(a_ptr, b_ptr, c_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m)
    b = tl.load(b_ptr + i, mask=m)
    c = tl.load(c_ptr + i, mask=m)
    tl.store(o_ptr + i, a * b + c, mask=m)  # compiler contracts -> single fma.rn.f32


def main():
    n = BLOCK = 8192
    torch.manual_seed(0)
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.randn(n, device="cuda")

    ck = compile_only(madd_kernel, a, b, c, torch.empty(n, device="cuda"), n, BLOCK=BLOCK, grid=(1, ))
    banner("b05 — TTGIR keeps mul+add separate, but PTX fuses them")
    print(f"    TTGIR:  arith.mulf={count(ck, 'ttgir', 'arith.mulf')}  arith.addf={count(ck, 'ttgir', 'arith.addf')}"
          "   (still two ops)")
    print(f"    PTX  :  fma.rn.f32={count(ck, 'ptx', 'fma.rn.f32')}  mul.f32={count(ck, 'ptx', 'mul.f32')}"
          "   (fused: one rounding, no separate multiply)")
    show(ck, "ptx", grep="fma.rn.f32", limit=2, label="\nsample fused PTX:")

    banner("b05 — fused (Triton) vs two-rounding (torch) differ in bits")
    o = torch.empty(n, device="cuda")
    madd_kernel[(1, )](a, b, c, o, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    sep = (a * b) + c  # torch eager: round(a*b), then round(+c) -> two roundings
    ref = a.double() * b.double() + c.double()
    print(f"    triton(fma) vs torch(sep) bitwise-equal: {torch.equal(o, sep)}")
    print(f"    differing elements                     : {int((o != sep).sum().item())} / {n}")
    print(f"    triton(fma) max err vs fp64            : {(o.double() - ref).abs().max().item():.3e}")
    print(f"    torch (sep) max err vs fp64            : {(sep.double() - ref).abs().max().item():.3e}")

    assert count(ck, "ptx", "fma.rn.f32") > 0 and count(ck, "ptx", "mul.f32") == 0, "expected contraction in PTX"
    assert not torch.equal(o, sep), "fused fma must differ from separate mul+add in bits"
    assert (o.double() - ref).abs().max() <= (sep.double() - ref).abs().max(), "fused fma is at least as accurate"
    print("\n[OK] mul+add contracted to one fma (a PTX-level decision): different bits, "
          "more accurate. A TTGIR checker can't see this -> PTX backstop needed.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
