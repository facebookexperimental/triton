"""V1 - The Starting Point: register-staged software pipeline.

Reference kernel for the AMD blog "Optimizing GEMM + Activation on CDNA4 with TLX".
It computes the fused addmm + GLU in a single kernel:

    X   = A @ B + bias          (the addmm / projection)
    out = X + X * Y             (the gate / activation)

V1 is the correct-but-slow baseline. Every K-tile takes the register-staging
detour on its way to the matrix cores: HBM -> registers (`tl.load`) -> LDS
(`tlx.local_store`) -> registers (`tlx.local_load`) -> MFMA. Loads, LDS staging,
and MFMA run essentially in sequence, so the matrix unit spends most of its time
waiting on memory. See the blog's "Version 1" section.

DO NOT MERGE - reference code accompanying the blog, not intended for upstreaming.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

M, N = 1024, 21568

# Fixed configs per K (autotuning is introduced in V2).
BEST_CONFIG = {
    256:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1,
              NUM_STAGES=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
    512:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1,
              NUM_STAGES=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
    1024: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1,
              NUM_STAGES=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
}


@triton.jit
def addmm_glu_v1(
    a_ptr, b_ptr, bias_ptr, y_ptr, c_ptr,
    M, N, K,
    sa0, sa1, sb0, sb1, sy0, sy1, sc0, sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.assume(sa0 > 0); tl.assume(sa1 > 0)
    tl.assume(sb0 > 0); tl.assume(sb1 > 0)
    tl.assume(sy0 > 0); tl.assume(sy1 > 0)
    tl.assume(sc0 > 0); tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    # LDS buffers for the (NUM_STAGES - 1)-deep pipeline.
    buffers_a = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES - 1)
    buffers_b = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES - 1)

    # Prologue: prefetch the first stages HBM -> registers -> LDS.
    for stage in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_smem = tlx.local_view(buffers_a, stage)
        b_smem = tlx.local_view(buffers_b, stage)
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - stage * BLOCK_SIZE_K, other=0.0)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - stage * BLOCK_SIZE_K, other=0.0)
        tlx.local_store(a_smem, a_reg)  # the register-staging detour
        tlx.local_store(b_smem, b_reg)
        a_ptrs += BLOCK_SIZE_K * sa1
        b_ptrs += BLOCK_SIZE_K * sb0

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(NUM_STAGES - 1, k_iters, num_stages=0):
        a_smem = tlx.local_view(buffers_a, k % (NUM_STAGES - 1))
        b_smem = tlx.local_view(buffers_b, k % (NUM_STAGES - 1))
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        prev = (k - NUM_STAGES - 1) % (NUM_STAGES - 1)
        a_prev = tlx.local_load(tlx.local_view(buffers_a, prev))
        b_prev = tlx.local_load(tlx.local_view(buffers_b, prev))
        acc = tl.dot(a_prev, b_prev, acc)

        tlx.local_store(a_smem, a_reg)
        tlx.local_store(b_smem, b_reg)
        a_ptrs += BLOCK_SIZE_K * sa1
        b_ptrs += BLOCK_SIZE_K * sb0

    for k in tl.range(k_iters - (NUM_STAGES - 1), k_iters, loop_unroll_factor=NUM_STAGES - 1):
        buf = k % (NUM_STAGES - 1)
        a_prev = tlx.local_load(tlx.local_view(buffers_a, buf))
        b_prev = tlx.local_load(tlx.local_view(buffers_b, buf))
        acc = tl.dot(a_prev, b_prev, acc)

    # Epilogue: addmm bias + GLU gate (out = X + X*Y), each done as a separate step.
    ocm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ocn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cmask = (ocm[:, None] < M) & (ocn[None, :] < N)

    bias = tl.load(bias_ptr + ocn, mask=ocn < N, other=0.0).to(tl.float32)
    x = acc + bias[None, :]

    y_ptrs = y_ptr + ocm[:, None] * sy0 + ocn[None, :] * sy1
    y = tl.load(y_ptrs, mask=cmask, other=0.0).to(tl.float32)
    out = x + x * y

    c_ptrs = c_ptr + ocm[:, None] * sc0 + ocn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=cmask)


def run(a, b, bias, y):
    """Launch the V1 kernel. Returns out = (A@B + bias) + (A@B + bias) * Y."""
    Mx, K = a.shape
    _, Nx = b.shape
    cfg = BEST_CONFIG[K]
    out = torch.empty((Mx, Nx), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(Mx, cfg["BLOCK_SIZE_M"]) * triton.cdiv(Nx, cfg["BLOCK_SIZE_N"]),)
    addmm_glu_v1[grid](
        a, b, bias, y, out, Mx, Nx, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        y.stride(0), y.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_STAGES=cfg["NUM_STAGES"], num_warps=cfg["num_warps"], num_stages=1,
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"], waves_per_eu=cfg["waves_per_eu"],
        kpack=cfg["kpack"],
    )
    return out


def _reference(bias, a, b, y):
    x = torch.matmul(a, b).to(torch.float32) + bias.to(torch.float32)[None, :]
    return (x + x * y.to(torch.float32)).to(torch.float16)


if __name__ == "__main__":
    for K in (256, 512, 1024):
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.randn(M, N, device="cuda", dtype=torch.float16)
        out = run(a, b, bias, y)
        torch.testing.assert_close(out.float(), _reference(bias, a, b, y).float(),
                                   atol=2e-1, rtol=2e-2)
        print(f"V1 K={K}: correctness OK")
