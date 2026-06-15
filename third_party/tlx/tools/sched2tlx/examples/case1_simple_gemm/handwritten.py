# Hand-written TLX target for case1 (simple GEMM).
#
# Mirrors `users/wl/wlei/autows/modulo_schedule/case1_gemm/run_gemm_e2e.py`:
#   - BM=128, BN=128, BK=64, fp16 in / fp16 out, fp32 acc
#   - 1 CTA, no Split-K, no MMA splitting, no epilogue subtile, non-persistent
#   - One inner K-loop with TMA loads + tc_gen5_mma
#
# Modulo schedule (from case1_gemm/anno_02_post_modulo.ttgir):
#   II = 1308 cycles
#   SMEM: 2x buffers_A (128x64 fp16) + 2x buffers_B (64x128 fp16)
#   TMEM: 1x acc (128x128 fp32)
#   Partitions: MEM (TMA loads) + TC (mma) + default (epilogue store)
#
# This file is the *spec* for the emitter — the emitter must produce
# something structurally identical from schedule_graph.json.
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def gemm_kernel(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                NUM_SMEM_BUFFERS: tl.constexpr,  # = 2 from modulo
                NUM_TMEM_BUFFERS: tl.constexpr,  # = 1 from modulo
                ):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ── Allocs (one per modulo BufferInfo, count = N) ──
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)
    # SMEM staging for epilogue TMA store (depth=1 — single store per tile).
    c_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_desc), 1)

    # ── Mbarriers (one full+empty pair per channel) ──
    A_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    A_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    B_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    B_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    k_tiles = tl.cdiv(K, BLOCK_K)

    with tlx.async_tasks():
        # ── Partition: default (epilogue) ──
        with tlx.async_task("default"):
            tlx.barrier_wait(tmem_full[0], 0)
            acc_full = tlx.local_load(acc_tmem[0])
            tlx.barrier_arrive(tmem_empty[0], 1)
            c = acc_full.to(tlx.dtype_of(c_desc))
            tlx.local_store(c_smem[0], c)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_smem[0], [pid_m * BLOCK_M, pid_n * BLOCK_N])
            tlx.async_descriptor_store_wait(0)

        # ── Partition: TC (MMA consumer) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            # Wait for epilogue to release TMEM (initial state = released).
            tlx.barrier_wait(tmem_empty[0], 1)
            for k_tile in range(k_tiles):
                buf = k_tile % NUM_SMEM_BUFFERS
                phase = (k_tile // NUM_SMEM_BUFFERS) & 1
                tlx.barrier_wait(A_full[buf], phase)
                tlx.barrier_wait(B_full[buf], phase)
                use_acc = k_tile > 0
                tlx.async_dot(
                    buffers_A[buf],
                    buffers_B[buf],
                    acc_tmem[0],
                    use_acc=use_acc,
                )
                tlx.barrier_arrive(A_empty[buf], 1)
                tlx.barrier_arrive(B_empty[buf], 1)
            # Signal epilogue that TMEM is filled.
            tlx.barrier_arrive(tmem_full[0], 1)

        # ── Partition: MEM (TMA producer) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            for k_tile in range(k_tiles):
                buf = k_tile % NUM_SMEM_BUFFERS
                phase = (k_tile // NUM_SMEM_BUFFERS) & 1
                # Wait for prior consumer to release this slot.
                tlx.barrier_wait(A_empty[buf], phase ^ 1)
                tlx.barrier_expect_bytes(A_full[buf], BLOCK_M * BLOCK_K * 2)
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[buf],
                    [pid_m * BLOCK_M, k_tile * BLOCK_K],
                    A_full[buf],
                )
                tlx.barrier_wait(B_empty[buf], phase ^ 1)
                tlx.barrier_expect_bytes(B_full[buf], BLOCK_K * BLOCK_N * 2)
                tlx.async_descriptor_load(
                    b_desc,
                    buffers_B[buf],
                    [k_tile * BLOCK_K, pid_n * BLOCK_N],
                    B_full[buf],
                )


def gemm(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMEM_BUFFERS=2,
        NUM_TMEM_BUFFERS=1,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = gemm(a, b)
    ref = torch.matmul(a, b)
    err = (c.float() - ref.float()).abs().max().item()
    print(f"max abs err = {err:.3e}")
    assert err < 1e-1, "correctness failed"
    print("OK")
