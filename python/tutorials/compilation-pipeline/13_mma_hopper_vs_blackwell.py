"""13 — Same tl.dot, two architectures: Hopper wgmma vs Blackwell tcgen05.

Pipeline: ``tritongpu-accelerate-matmul``
(``lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp``) rewrites ``tt.dot``
into a hardware MMA, but the op and the result layout it picks depend on the
target:

  * Hopper (sm_90):    ``ttng.warp_group_dot`` + a ``#ttg.nvidia_mma`` register
                       layout  -> PTX ``wgmma.mma_async``.
  * Blackwell (sm_100):``ttng.tc_gen5_mma`` writing a **TMEM** accumulator (no
                       ``nvidia_mma`` register layout) -> PTX ``tcgen05.mma``.

These different layouts are exactly why a kernel that is bit-stable on one arch
may not be on the other — the accumulator lives in different storage with a
different reduction structure.

To see *both* on one machine we don't launch anything: ``compile_for_target``
runs the full compile pipeline for an explicit ``GPUTarget`` (sm_90 and sm_100),
so the IR is available regardless of the host GPU. The runtime correctness check
still needs real hardware, so it runs only on the host's own arch.

Run:  python python/tutorials/compilation-pipeline/13_mma_hopper_vs_blackwell.py
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


def main():
    M = N = 128
    K = 64
    consts = {"M": M, "N": N, "K": K}

    hop = compile_for_target(matmul_kernel, _SIG, consts, cc=90, num_warps=4, num_stages=3)
    bw = compile_for_target(matmul_kernel, _SIG, consts, cc=100, num_warps=4, num_stages=3)

    banner("13 — same tl.dot, accelerate-matmul picks a different op + layout per arch")
    for cc, ck in ((90, hop), (100, bw)):
        print(f"    {cc_name(cc):16}:  warp_group_dot={count(ck, 'ttgir', 'warp_group_dot')}"
              f"  tc_gen5_mma={count(ck, 'ttgir', 'tc_gen5_mma')}"
              f"  #nvidia_mma={count(ck, 'ttgir', 'nvidia_mma')}"
              f"  tmem={count(ck, 'ttgir', 'tensor_memory')}")
    show(hop, "ttgir", grep=["nvidia_mma =", "warp_group_dot"], limit=2, label=f"\n  {cc_name(90)} MMA op:")
    show(bw, "ttgir", grep=["tc_gen5_mma", "tensor_memory ="], limit=2, label=f"  {cc_name(100)} MMA op:")

    banner("13 — same tl.dot lowers to a different PTX tensor-core instruction")
    print(f"    {cc_name(90):16}:  PTX wgmma.mma_async={count(hop, 'ptx', 'wgmma.mma_async')}")
    print(f"    {cc_name(100):16}:  PTX tcgen05.mma   ={count(bw, 'ptx', 'tcgen05.mma')}")

    # The pass responsible: `tritongpu-accelerate-matmul`. Run it on the HOST arch
    # (via dump_passes/warmup) so the per-pass marker prints on any GPU.
    banner("13 — the pass responsible: tritongpu-accelerate-matmul (host arch)")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    dumps = dump_passes(matmul_kernel, a, b, c, M, N, K, num_warps=4, num_stages=3, grid=(1, ))
    pass_diff(dumps, "accelerate-matmul", grep=["tt.dot", "warp_group_dot", "tc_gen5_mma", "nvidia_mma"], limit=12)

    # Runtime correctness on the host's own arch (cross-compiled IR can't launch).
    matmul_kernel[(1, )](a, b, c, M, N, K, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print(f"\n[OK] both arches lower the same tl.dot to their native tensor core; the host"
          f" ({cc_name(torch.cuda.get_device_capability()[0] * 10)}) result matches the reference.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
