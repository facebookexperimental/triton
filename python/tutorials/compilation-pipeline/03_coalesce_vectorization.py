"""03 — Coalescing & vectorization: how memory accesses become vector ld/st in PTX.

Pipeline: **TTGIR** ``Coalesce`` pass (``lib/Dialect/TritonGPU/Transforms/Coalesce.cpp``)
+ ``lib/Analysis/AxisInfo.cpp`` (alignment/contiguity) decide how many elements each
thread loads contiguously (``sizePerThread`` on the blocked layout). That width then
lowers to **PTX** vector memory ops ``ld.global.v{2,4}`` / ``st.global.v{2,4}``.

Knob: with a fixed ``BLOCK``, more ``num_warps`` => fewer elements per thread =>
narrower (or scalar) vectorization. We compile the same copy kernel two ways and
diff the PTX.

Bit-neutral mechanic: vectorization changes the *shape* of the memory traffic, not
the values copied — both versions are bitwise-identical. (It only becomes a numerics
knob indirectly, when a wider ``sizePerThread`` changes an intra-thread reduction
fold.)

Run:  python python/tutorials/compilation-pipeline/03_coalesce_vectorization.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda, show


@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def main():
    n, BLOCK = 1 << 16, 1024
    src = torch.randn(n, device="cuda")
    dst = torch.empty_like(src)

    # Same BLOCK; num_warps controls elements/thread => vectorization width.
    wide = compile_only(copy_kernel, src, dst, n, BLOCK=BLOCK, num_warps=4, grid=(1, ))  # sizePerThread=4
    narrow = compile_only(copy_kernel, src, dst, n, BLOCK=BLOCK, num_warps=32, grid=(1, ))  # sizePerThread=1

    banner("03 — vectorization width on the blocked layout (sizePerThread)")
    show(wide, "ttgir", grep="#blocked =", limit=2, label="num_warps=4  TTGIR layout:")
    show(narrow, "ttgir", grep="#blocked =", limit=2, label="num_warps=32 TTGIR layout:")

    banner("03 — the same width in PTX (vector vs scalar global memory ops)")
    for label, ck in (("num_warps=4 ", wide), ("num_warps=32", narrow)):
        v4 = count(ck, "ptx", "ld.global.v4") + count(ck, "ptx", "st.global.v4")
        v2 = count(ck, "ptx", "ld.global.v2") + count(ck, "ptx", "st.global.v2")
        scalar = count(ck, "ptx", "ld.global.b32") + count(ck, "ptx", "st.global.b32")
        print(f"    {label}:  v4 ops={v4}  v2 ops={v2}  scalar(b32) ops={scalar}")
    print("\n    (sample PTX global-memory ops, num_warps=4)")
    show(wide, "ptx", grep=["ld.global", "st.global"], limit=6)

    # Bit-neutral: vectorization does not change the copied values.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), )
    d4 = torch.empty_like(src)
    d32 = torch.empty_like(src)
    copy_kernel[grid](src, d4, n, BLOCK=BLOCK, num_warps=4)
    copy_kernel[grid](src, d32, n, BLOCK=BLOCK, num_warps=32)
    torch.cuda.synchronize()
    assert torch.equal(d4, src) and torch.equal(d32, src), "copy must be exact"
    assert torch.equal(d4, d32), "vectorization width is bit-neutral for a copy"
    print("\n[OK] wide (v4) and scalar copies are bitwise-identical (vectorization is bit-neutral).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
