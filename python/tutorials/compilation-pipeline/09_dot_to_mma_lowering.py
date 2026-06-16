"""09 — GEMM lowering: tl.dot becomes a tensor-core MMA.

Pipeline: **TTGIR** ``tritongpu-accelerate-matmul``
(``lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp``) rewrites a generic
``tt.dot`` into a hardware MMA: it assigns an ``#ttg.nvidia_mma`` layout to the
result, puts the operands in shared memory, and emits ``ttng.warp_group_dot``
(Hopper wgmma) / ``ttng.tc_gen5_mma`` (Blackwell). That lowers to PTX
``wgmma.mma_async`` (or ``mma.sync``).

What to notice: before the pass the body is ``tt.dot`` on ``#blocked`` operands;
after it, a ``warp_group_dot`` on ``!ttg.memdesc`` (shared) operands producing an
``#mma`` result, with a single ``convert_layout`` back to the blocked epilogue.

Bit-neutral mechanic: choosing the MMA *instruction* does not by itself change the
math (given the same input precision) — the accumulation is the tensor core's. The
numeric knob is the input precision, which 10 explores.

Run:  python python/tutorials/compilation-pipeline/09_dot_to_mma_lowering.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # fp16 inputs => natural tensor-core path
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def main():
    # One full tile (BM=M, BN=N) so the kernel computes the whole output and we
    # can check it against a torch reference directly.
    M = N = 128
    K = 64
    BM = BN = 128
    BK = 32
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    args = (a, b, c, M, N, K, BM, BN, BK)

    ck = compile_only(matmul_kernel, *args, num_stages=3, grid=(1, ))

    banner("09 — tl.dot lowered to an MMA: layout, op, and PTX instruction")
    print(f"    #mma layout in TTGIR        = {count(ck, 'ttgir', 'nvidia_mma')}")
    print(f"    ttng.warp_group_dot ops     = {count(ck, 'ttgir', 'warp_group_dot')}")
    print(f"    PTX wgmma.mma_async ops     = {count(ck, 'ptx', 'wgmma.mma_async')}")
    show(ck, "ttgir", grep=["nvidia_mma =", "warp_group_dot"], limit=3, label="\nthe MMA in TTGIR:")

    # The pass that does the rewrite: `tritongpu-accelerate-matmul`.
    banner("09 — the pass responsible: tritongpu-accelerate-matmul")
    dumps = dump_passes(matmul_kernel, *args, num_stages=3, grid=(1, ))
    pass_diff(dumps, "accelerate-matmul", grep=["tt.dot", "warp_group_dot", "nvidia_mma"], limit=14)

    # Correctness (allclose: tensor-core fp16 accumulation differs from a plain
    # fp32 reference by rounding, but is numerically equivalent).
    matmul_kernel[(1, )](*args, num_stages=3)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print("\n[OK] the wgmma matmul matches the fp32 reference within tolerance"
          " (the MMA instruction choice is numerically faithful).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
