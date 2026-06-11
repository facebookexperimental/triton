"""06 — Warp specialization (AutoWS): split warps into producer/consumer roles.

Pipeline: **TTGIR** automatic warp specialization
(``lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`` +
``third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/``). Enabled by
``TRITON_USE_META_WS`` (``knobs.nvidia.use_meta_ws``) on a loop marked
``tl.range(..., warp_specialize=True)``, the pass partitions a warp group into
**producer** warps (TMA loads) and **consumer** warps (MMA), wrapping them in a
``ttg.warp_specialize`` region and tagging ops with ``async_task_id``.

What to notice: with AutoWS on, the persistent matmul's TTGIR gains
``ttg.warp_specialize`` and ``async_task_id`` annotations that are absent otherwise.

Bit-neutral mechanic: WS is a *scheduling* transform (who does what), not a change
to the MMA math — the numeric result is the same. Requires datacenter Blackwell
(MMAv5/tcgen05 + TMA); self-skips elsewhere.

Run:  python python/tutorials/compilation-pipeline/06_warp_specialization.py
"""
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from _ir_utils import banner, compile_only, count, dump_passes, is_blackwell, is_cuda, pass_diff, show


@triton.jit
def _matmul_persistent_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, FLATTEN: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=True,
                            disallow_acc_multi_buffer=True, separate_epilogue_store=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))


@triton.jit
def matmul(a_desc, b_desc, c_desc, M, N, K, FLATTEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    _matmul_persistent_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_SMS, FLATTEN)


def main():
    M, N, K = 512, 512, 256
    BM, BN, BK, GROUP = 128, 128, 64, 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    a_desc = TensorDescriptor(a, [M, K], [K, 1], [BM, BK])
    b_desc = TensorDescriptor(b, [N, K], [K, 1], [BN, BK])
    c_desc = TensorDescriptor(c, [M, N], [N, 1], [BM, BN])
    grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)), )

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True  # == TRITON_USE_META_WS=1
        # FLATTEN=False is the warp-specializing config.
        ck = compile_only(matmul, a_desc, b_desc, c_desc, M, N, K, False, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                          GROUP_M=GROUP, NUM_SMS=NUM_SMS, grid=grid)
        banner("06 — AutoWS partitions warps (producer TMA loads / consumer MMA)")
        print(f"    ttg.warp_specialize ops = {count(ck, 'ttgir', 'warp_specialize')}")
        print(f"    async_task_id tags      = {count(ck, 'ttgir', 'async_task_id')}")
        show(ck, "ttgir", grep="warp_specialize", limit=3, label="\nwarp_specialize region(s):")

        # The pass responsible: `nvgpu-warp-specialization`. pass_diff shows it
        # wrapping the loop body in a ttg.warp_specialize region and tagging ops
        # with async_task_id (producer vs consumer).
        banner("06 — the pass responsible: nvgpu-warp-specialization")
        dumps = dump_passes(matmul, a_desc, b_desc, c_desc, M, N, K, False, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                            GROUP_M=GROUP, NUM_SMS=NUM_SMS, grid=grid)
        pass_diff(dumps, "warp-specialization", grep=["warp_specialize", "async_task_id"], limit=20)

        matmul[grid](a_desc, b_desc, c_desc, M, N, K, False, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP,
                     NUM_SMS=NUM_SMS)
    torch.cuda.synchronize()

    ref = (a.float() @ b.float().T).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print("\n[OK] warp-specialized matmul matches the torch reference"
          " (WS is a scheduling transform, not a math change).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    if not is_blackwell():
        print("[06] skipped — AutoWS example requires datacenter Blackwell (MMAv5 + TMA).")
        raise SystemExit(0)
    main()
