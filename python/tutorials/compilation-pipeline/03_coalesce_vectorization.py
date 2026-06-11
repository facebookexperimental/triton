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

We also compile a looped ``num_stages=2`` variant to contrast a synchronous vector
load (``ld.global.v4``) with an asynchronous pipelined load (``cp.async``).

Run:  python python/tutorials/compilation-pipeline/03_coalesce_vectorization.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


@triton.jit
def copy_kernel_pipelined(src, dst, n, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    # Same copy, but over a loop with num_stages=2. The software pipeliner then
    # multi-buffers the load and issues it ASYNCHRONOUSLY (cp.async) so the next
    # iteration's data is fetched while this one stores — the synchronous vector
    # `ld.global` becomes `cp.async`.
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
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

    # The pass that picks sizePerThread: `tritongpu-coalesce`. pass_diff shows it
    # introducing the coalesced (wider sizePerThread) #blocked layout.
    banner("03 — the pass responsible: tritongpu-coalesce")
    dumps = dump_passes(copy_kernel, src, dst, n, BLOCK=BLOCK, num_warps=4, grid=(1, ))
    pass_diff(dumps, "tritongpu-coalesce", grep=["#blocked", "tt.load", "tt.store"], limit=20)

    # num_stages=2 in a for loop: synchronous vector load becomes async cp.async.
    STEPS = 8
    pipelined = compile_only(copy_kernel_pipelined, src, dst, n, BLOCK=BLOCK, STEPS=STEPS, num_warps=4, grid=(1, ))
    banner("03 — num_stages=2 loop: vector ld.global vs asynchronous cp.async")
    for label, ck in (("flat (1 stage) ", wide), ("looped (2 stages)", pipelined)):
        ld_v = count(ck, "ptx", "ld.global.v4") + count(ck, "ptx", "ld.global.v2")
        cp = count(ck, "ptx", "cp.async")
        print(f"    {label}:  vector ld.global ops={ld_v:>2}   cp.async ops={cp:>2}")
    print("\n    (the loop's load is now async — sample TTGIR)")
    show(pipelined, "ttgir", grep="async_copy", limit=3)

    # Bit-neutral: neither vectorization width nor pipelining changes the copy.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), )
    d4 = torch.empty_like(src)
    d32 = torch.empty_like(src)
    copy_kernel[grid](src, d4, n, BLOCK=BLOCK, num_warps=4)
    copy_kernel[grid](src, d32, n, BLOCK=BLOCK, num_warps=32)
    dp = torch.empty_like(src)
    copy_kernel_pipelined[(triton.cdiv(n, BLOCK * STEPS), )](src, dp, n, BLOCK=BLOCK, STEPS=STEPS, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(d4, src) and torch.equal(d32, src), "copy must be exact"
    assert torch.equal(d4, d32), "vectorization width is bit-neutral for a copy"
    assert torch.equal(dp, src), "pipelined (cp.async) copy must be exact too"
    print("\n[OK] vector (v4), scalar, and async (cp.async) copies are all bitwise-identical"
          " (vectorization and pipelining are bit-neutral).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
