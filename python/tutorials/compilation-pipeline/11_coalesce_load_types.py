"""11 — Coalescing across load types: how the access pattern picks the layout.

Pipeline: the **TTGIR** ``Coalesce`` pass
(``lib/Dialect/TritonGPU/Transforms/Coalesce.cpp``) reads per-dimension
contiguity/alignment from ``lib/Analysis/AxisInfo.cpp`` and chooses the blocked
layout's ``order`` (which axis is the fast one) and ``sizePerThread`` (how many
contiguous elements per thread). That choice then decides the **PTX** memory
width: a contiguous run becomes ``ld.global.v4``; a strided run collapses to
scalar ``ld.global.b32`` even though the layout *shape* is unchanged.

Where 03 fixed the access pattern and varied ``num_warps`` to move the
vectorization *width*, this one fixes ``num_warps`` and varies the **access
pattern** to show the pass reacting to each load type:

  * contiguous 1-D            -> order=[0],   v4
  * row-major 2-D (inner contiguous)  -> order=[1,0], v4
  * column-major 2-D (outer contiguous) -> order=[0,1], v4  (axis flipped!)
  * strided (stride-2) gather -> same layout shape, but PTX falls to scalar b32

Then a looped copy contrasts ``num_stages=1`` (synchronous ``ld.global``) with
``num_stages=2`` (the software pipeliner issues an asynchronous ``cp.async``) —
the user-visible "different lowering for different num_stages".

Bit-neutral mechanic: coalescing only reshapes the memory *traffic*; every
variant copies the exact same bytes. We assert all of them are bit-identical.

Run:  python python/tutorials/compilation-pipeline/11_coalesce_load_types.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def copy_1d(src, dst, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs))


@triton.jit
def copy_2d_rowmajor(src, dst, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    idx = rm[:, None] * N + rn[None, :]  # inner axis (n) is contiguous
    tl.store(dst + idx, tl.load(src + idx))


@triton.jit
def copy_2d_colmajor(src, dst, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    idx = rm[:, None] + rn[None, :] * M  # outer axis (m) is contiguous
    tl.store(dst + idx, tl.load(src + idx))


@triton.jit
def gather_strided(src, dst, N: tl.constexpr, S: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs * S))  # stride S => non-contiguous


@triton.jit
def copy_looped(src, dst, n, BLOCK: tl.constexpr, STEPS: tl.constexpr, NS: tl.constexpr):
    base = tl.program_id(0) * BLOCK * STEPS
    for i in tl.range(0, STEPS, num_stages=NS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def _layout(ck):
    return next((ln.split("= ", 1)[-1].strip() for ln in ck.asm["ttgir"].splitlines() if "#blocked =" in ln), "")


def _widths(ck):
    return (count(ck, "ptx", "ld.global.v4"), count(ck, "ptx", "ld.global.v2"), count(ck, "ptx", "ld.global.b32"))


def main():
    M = N = 128
    a2 = torch.randn(M, N, device="cuda")
    b2 = torch.empty(M, N, device="cuda")

    row = compile_only(copy_2d_rowmajor, a2, b2, M, N, num_warps=4, grid=(1, ))
    col = compile_only(copy_2d_colmajor, a2, b2, M, N, num_warps=4, grid=(1, ))

    banner("11 — Coalesce picks the layout ORDER from the contiguous axis")
    print(f"    row-major (inner contiguous):  {_layout(row)}")
    print(f"    col-major (outer contiguous):  {_layout(col)}")
    print("    ^ same kernel shape, but `order` flips to follow the contiguous axis.")

    Nv = 4096
    a1 = torch.randn(Nv * 2, device="cuda")
    b1 = torch.empty(Nv, device="cuda")
    contig = compile_only(copy_1d, a1, b1, Nv, num_warps=4, grid=(1, ))
    strided = compile_only(gather_strided, a1, b1, Nv, 2, num_warps=4, grid=(1, ))

    banner("11 — same layout shape, but contiguity decides the PTX width")
    for label, ck in (("contiguous 1-D", contig), ("row-major 2-D", row), ("col-major 2-D", col), ("strided  (S=2)",
                                                                                                   strided)):
        v4, v2, b32 = _widths(ck)
        print(f"    {label}:  v4={v4:>2}  v2={v2:>2}  scalar(b32)={b32:>2}")
    print("    ^ the stride-2 gather can't coalesce -> it falls back to scalar b32.")

    # The pass responsible: `tritongpu-coalesce`. pass_diff shows it rewriting the
    # tt.load operands onto a coalesced #blocked layout.
    banner("11 — the pass responsible: tritongpu-coalesce")
    dumps = dump_passes(copy_2d_colmajor, a2, b2, M, N, num_warps=4, grid=(1, ))
    pass_diff(dumps, "tritongpu-coalesce", grep=["#blocked", "tt.load", "tt.store"], limit=20)

    # num_stages: a looped load goes synchronous (ld.global) -> asynchronous (cp.async).
    BLOCK, STEPS = 1024, 8
    n = BLOCK * STEPS
    s1 = compile_only(copy_looped, a1, b1, n, BLOCK=BLOCK, STEPS=STEPS, NS=1, num_warps=4, grid=(1, ))
    s2 = compile_only(copy_looped, a1, b1, n, BLOCK=BLOCK, STEPS=STEPS, NS=2, num_warps=4, grid=(1, ))
    banner("11 — num_stages=1 vs 2: synchronous ld.global becomes asynchronous cp.async")
    for label, ck in (("num_stages=1", s1), ("num_stages=2", s2)):
        ld = count(ck, "ptx", "ld.global.v4") + count(ck, "ptx", "ld.global.v2")
        cp = count(ck, "ptx", "cp.async")
        print(f"    {label}:  vector ld.global={ld:>2}   cp.async={cp:>2}")
    show(s2, "ttgir", grep="async_copy", limit=3, label="\n    (num_stages=2 TTGIR uses async_copy)")

    # Bit-neutral: every coalescing/pipelining choice copies the same bytes.
    def run(k, *args, **kw):
        out = torch.empty_like(b2 if k in (copy_2d_rowmajor, copy_2d_colmajor) else b1)
        k[(1, )](args[0], out, *args[1:], **kw)
        return out

    torch.cuda.synchronize()
    r_row = run(copy_2d_rowmajor, a2, M, N, num_warps=4)
    r_col = run(copy_2d_colmajor, a2, M, N, num_warps=4)
    r_contig = run(copy_1d, a1, Nv, num_warps=4)
    r_strided = run(gather_strided, a1, Nv, 2, num_warps=4)
    dp = torch.empty(n, device="cuda")
    copy_looped[(1, )](a1, dp, n, BLOCK=BLOCK, STEPS=STEPS, NS=2, num_warps=4)
    torch.cuda.synchronize()

    assert torch.equal(r_row, a2), "row-major copy must be exact"
    assert torch.equal(r_col, a2), "col-major copy must be exact (different order, same bytes)"
    assert torch.equal(r_contig, a1[:Nv]), "contiguous copy must be exact"
    assert torch.equal(r_strided, a1[:Nv * 2:2]), "strided gather must be exact"
    assert torch.equal(dp, a1[:n]), "pipelined (cp.async) copy must be exact"
    print("\n[OK] every load type — contiguous, row/col-major, strided, and pipelined —"
          " copies the identical bytes (coalescing and pipelining are bit-neutral).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
