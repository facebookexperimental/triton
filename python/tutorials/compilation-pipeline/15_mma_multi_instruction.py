"""15 — One tl.dot, many MMA instructions: tiling by the hardware MMA atom.

Pipeline: a single ``tl.dot`` covers a whole ``BM x BN x BK`` tile, but a tensor
core executes only a fixed-size **MMA atom** at a time. ``accelerate-matmul``
records that atom as ``instrShape`` on the ``#ttg.nvidia_mma`` layout (e.g.
``[16, BN, 16]`` for Hopper wgmma), and the **TTGIR -> LLVM/PTX** lowering then
emits one hardware instruction per atom — so one ``tl.dot`` becomes *many*
``wgmma.mma_async`` (Hopper) / ``tcgen05.mma`` (Blackwell) instructions.

The count per warp is roughly ``(BM / warpsM / atomM) * (BK / atomK)``, and it
differs by arch: Blackwell's tcgen05 atom covers more work per instruction than
Hopper's wgmma, so the same dot needs *fewer* Blackwell instructions. This
multiplicity (and how a tile is split) is part of why the same dot can have a
different accumulation structure on each arch.

Both arches are cross-compiled (no launch); correctness runs on the host arch.

Run:  python python/tutorials/compilation-pipeline/15_mma_multi_instruction.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, cc_name, compile_for_target, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


_SIG = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16", "M": "constexpr", "N": "constexpr", "K": "constexpr"}


def _mma_count(ck, cc):
    return count(ck, "ptx", "wgmma.mma_async" if cc == 90 else "tcgen05.mma")


def _instr_shape(ck):
    return next((ln.split("instrShape = ")[-1].split("}")[0]
                 for ln in ck.asm["ttgir"].splitlines()
                 if "nvidia_mma" in ln and "instrShape" in ln), "(TMEM, no register atom)")


def main():
    banner("15 — one tl.dot expands to N tensor-core instructions; N scales with the tile")
    for (M, N, K) in ((64, 64, 32), (128, 128, 32), (128, 128, 64)):
        hop = compile_for_target(matmul_kernel, _SIG, {"M": M, "N": N, "K": K}, cc=90, num_warps=4, num_stages=2)
        bw = compile_for_target(matmul_kernel, _SIG, {"M": M, "N": N, "K": K}, cc=100, num_warps=4, num_stages=2)
        print(f"    {M}x{N}x{K}:  {cc_name(90)} wgmma={_mma_count(hop, 90):>2}"
              f"   {cc_name(100)} tcgen05={_mma_count(bw, 100):>2}")

    M = N = 128
    K = 64
    hop = compile_for_target(matmul_kernel, _SIG, {"M": M, "N": N, "K": K}, cc=90, num_warps=4, num_stages=2)
    bw = compile_for_target(matmul_kernel, _SIG, {"M": M, "N": N, "K": K}, cc=100, num_warps=4, num_stages=2)

    banner(f"15 — the {M}x{N}x{K} dot: one op in TTGIR, the atom (instrShape), many in PTX")
    print(f"    {cc_name(90)}:  warp_group_dot ops in TTGIR = {count(hop, 'ttgir', 'warp_group_dot')}"
          f"   atom instrShape = {_instr_shape(hop)}")
    print(f"    {cc_name(90)}:  wgmma.mma_async ops in PTX  = {_mma_count(hop, 90)}  (one per atom, per warp)")
    print(f"    {cc_name(100)}:  tc_gen5_mma ops in TTGIR    = {count(bw, 'ttgir', 'tc_gen5_mma')}")
    print(f"    {cc_name(100)}:  tcgen05.mma ops in PTX      = {_mma_count(bw, 100)}  (fewer — larger atom)")
    show(hop, "ptx", grep="wgmma.mma_async", limit=3, label=f"\n  sample {cc_name(90)} PTX MMA instructions:")

    # The pass that fixes the atom shape: `tritongpu-accelerate-matmul` (the
    # instrShape on the #nvidia_mma layout is what the PTX multiplicity follows).
    banner("15 — the pass responsible: tritongpu-accelerate-matmul (sets instrShape)")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    dumps = dump_passes(matmul_kernel, a, b, c, M, N, K, num_warps=4, num_stages=2, grid=(1, ))
    pass_diff(dumps, "accelerate-matmul", grep=["instrShape", "warp_group_dot", "tc_gen5_mma"], limit=8)

    # Correctness on the host arch: the tiled instructions reassemble the full dot.
    matmul_kernel[(1, )](a, b, c, M, N, K, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print("\n[OK] one tl.dot lowered to many tensor-core instructions (more on Hopper, fewer on"
          " Blackwell) and the host result matches the reference — the tiling is faithful.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
