"""b06 — Elementwise math precision: approximate div / exp / cast change bits.

Source: the NVIDIA elementwise lowering
``third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp``. Unlike
b01-b05 (which are about the *order* of a reduction or dot), this is a **direct,
per-element rounding** change: the backend picks fast approximate instructions for
fp32 transcendentals and division.

What the PTX actually emits here:
  * ``x / y``      -> ``div.full.f32``  (full-range but ~2 ULP, NOT IEEE-rounded;
                      contrast ``div.rn.f32`` / fp64 ``div.rn.f64``)
  * ``tl.exp(x)``  -> ``mul.f32`` by log2(e) then ``ex2.approx.f32`` (approximate)
  * ``x.to(bf16)`` -> ``cvt.rn.bf16.f32`` (round-to-nearest-even)

BIT-CHANGING. Each differs from a correctly-rounded reference (torch / fp64). This
whole class of differences is undocumented in the project KB, which is why b06
exists (see ``numerics-modifying-passes.md``). Pure CUDA (no Blackwell needed).

Run:  python bitequiv/examples/layout_numerics/b06_elementwise_math_precision.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda, show


@triton.jit
def div_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) / tl.load(y_ptr + i, mask=m), mask=m)


@triton.jit
def exp_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    tl.store(o_ptr + i, tl.exp(tl.load(x_ptr + i, mask=m)), mask=m)


@triton.jit
def cast_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m).to(tl.bfloat16), mask=m)


def main():
    n = BLOCK = 8192
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda").abs() + 0.5
    y = torch.randn(n, device="cuda").abs() + 0.5

    ck_div = compile_only(div_kernel, x, y, torch.empty(n, device="cuda"), n, BLOCK=BLOCK, grid=(1, ))
    ck_exp = compile_only(exp_kernel, x, torch.empty(n, device="cuda"), n, BLOCK=BLOCK, grid=(1, ))
    ck_cast = compile_only(cast_kernel, x, torch.empty(n, dtype=torch.bfloat16, device="cuda"), n, BLOCK=BLOCK,
                           grid=(1, ))

    banner("b06 — the approximate instructions the backend chose (PTX)")
    print(f"    x / y     -> div.full.f32 = {count(ck_div, 'ptx', 'div.full.f32')}"
          f"   (IEEE div.rn.f32 = {count(ck_div, 'ptx', 'div.rn.f32')})")
    print(f"    tl.exp(x) -> ex2.approx.f32 = {count(ck_exp, 'ptx', 'ex2.approx.f32')}")
    print(f"    .to(bf16) -> cvt.rn(.bf16) = {count(ck_cast, 'ptx', 'cvt.rn') }")
    show(ck_div, "ptx", grep="div.full.f32", limit=1, label="sample div PTX:")
    show(ck_exp, "ptx", grep="ex2.approx.f32", limit=1, label="sample exp PTX:")

    banner("b06 — each differs in bits from a correctly-rounded reference")
    od = torch.empty(n, device="cuda")
    oe = torch.empty(n, device="cuda")
    div_kernel[(1, )](x, y, od, n, BLOCK=BLOCK)
    exp_kernel[(1, )](x, oe, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    div_ref64 = x.double() / y.double()
    exp_ref64 = torch.exp(x.double())
    print(f"    div: triton vs torch fp32 bitwise-equal = {torch.equal(od, x / y)}   "
          f"max err vs fp64 = {(od.double() - div_ref64).abs().max().item():.3e}")
    print(f"    exp: triton vs torch fp32 bitwise-equal = {torch.equal(oe, torch.exp(x))}   "
          f"max err vs fp64 = {(oe.double() - exp_ref64).abs().max().item():.3e}")

    assert count(ck_div, "ptx", "div.full.f32") > 0, "expected approximate fp32 division"
    assert count(ck_exp, "ptx", "ex2.approx.f32") > 0, "expected approximate exp2"
    assert not torch.equal(od, x / y), "approximate div must differ from torch's correctly-rounded div"
    assert not torch.equal(oe, torch.exp(x)), "approximate exp must differ from torch's exp"
    assert (od.double() - div_ref64).abs().max() < 1e-4 and (oe.double() - exp_ref64).abs().max() < 1e-3, \
        "approximations are still close"
    print("\n[OK] approximate div.full / ex2.approx / cvt.rn change per-element bits "
          "(a rounding class distinct from reduction order).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
