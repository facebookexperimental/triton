"""01 — Reading the first IR: TTIR (the hardware-agnostic tensor IR).

Pipeline stage: **TTIR** (Triton IR), produced by the frontend before any GPU
mapping. Source: ``python/triton/compiler/`` (``make_ttir`` in the NVIDIA backend
``third_party/nvidia/backend/compiler.py``).

What to notice: TTIR is *layout-free*. Tensor types are just ``tensor<256xf32>``
with no ``#blocked`` / thread-mapping encoding — that mapping is added in the next
stage (TTGIR). The ops are the high-level tensor ops you wrote: ``tt.make_range``,
``tt.load``, ``arith.addf``, ``tt.store``.

Bit-neutral mechanic: this stage only records *what* to compute, not *how* it maps
onto threads, so it does not by itself decide FP bit results.

Run:  python python/tutorials/compilation-pipeline/01_read_ttir.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, is_cuda, show


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


def main():
    n, BLOCK = 1024, 256
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty_like(x)

    # Compile only (no launch) to read the IR.
    ck = compile_only(add_kernel, x, y, o, n, BLOCK=BLOCK, grid=(1, ))

    banner("01 — TTIR: the hardware-agnostic tensor IR (no layouts yet)")
    show(ck, "ttir", grep=["tt.", "arith."], limit=40)
    print("\nNote: tensor types are plain `tensor<256xf32>` — NO #blocked encoding."
          "\n      The thread/warp mapping is added in the next stage (see 02).")

    # Correctness check (real launch): vector add is exact in fp32.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), )
    add_kernel[grid](x, y, o, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    assert torch.equal(o, x + y), "vector add must match torch exactly"
    print("\n[OK] runtime result is bitwise-equal to torch x + y")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
