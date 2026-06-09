import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import triton.testing

M, N = 1024, 21568
K_VALUES = [256, 512, 1024]


# ---- TLX fused addmm + GLU kernel ----
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_STAGES": 2,
                "kpack": 1,
                "matrix_instr_nonkdim": 16,
                "waves_per_eu": 0,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_STAGES": 2,
                "kpack": 1,
                "matrix_instr_nonkdim": 16,
                "waves_per_eu": 2,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_STAGES": 2,
                "kpack": 1,
                "matrix_instr_nonkdim": 16,
                "waves_per_eu": 2,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_STAGES": 2,
                "kpack": 1,
                "matrix_instr_nonkdim": 16,
                "waves_per_eu": 0,
            },
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def tlx_addmm_glu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    buffers_a = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES - 1
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES - 1
    )

    # TLX prologue: prefetch the first stages from HBM into LDS.
    for stage in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_smem = tlx.local_view(buffers_a, stage)
        b_smem = tlx.local_view(buffers_b, stage)
        a_reg = tl.load(
            a_ptrs, mask=offs_k[None, :] < K - stage * BLOCK_SIZE_K, other=0.0
        )
        b_reg = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - stage * BLOCK_SIZE_K, other=0.0
        )
        tlx.local_store(a_smem, a_reg)
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

    for k in tl.range(
        k_iters - (NUM_STAGES - 1), k_iters, loop_unroll_factor=NUM_STAGES - 1
    ):
        buf = k % (NUM_STAGES - 1)
        a_prev = tlx.local_load(tlx.local_view(buffers_a, buf))
        b_prev = tlx.local_load(tlx.local_view(buffers_b, buf))
        acc = tl.dot(a_prev, b_prev, acc)

    # === Fused epilogue: addmm + GLU ===
    # Add bias (broadcast over M), then GLU: X = X + X*Y
    ocm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ocn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cmask = (ocm[:, None] < M) & (ocn[None, :] < N)

    # Bias of shape (N,) broadcast over M
    bias = tl.load(bias_ptr + ocn, mask=ocn < N, other=0.0).to(tl.float32)
    x = acc + bias[None, :]

    # Load Y (the gate) and apply GLU: x = x + x*y
    y_ptrs = y_ptr + ocm[:, None] * sy0 + ocn[None, :] * sy1
    y = tl.load(y_ptrs, mask=cmask, other=0.0).to(tl.float32)
    out = x + x * y

    c_ptrs = c_ptr + ocm[:, None] * sc0 + ocn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=cmask)


def tlx_fused_addmm_glu(bias, a, b, y):
    M, K = a.shape
    K2, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    tlx_addmm_glu_kernel[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


# ---- Bench harness ----
def pytorch_baseline(bias, a, b, y):
    """PyTorch: addmm + GLU (two ops)."""
    x = torch.addmm(bias, a, b).to(torch.float32)
    out = (x + x * y.to(torch.float32)).to(torch.float16)
    return out


for K in K_VALUES:
    print(f"\n=== Shape M={M}, N={N}, K={K} fp16 ===")
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(M, N, device="cuda", dtype=torch.float16)

    print(a.stride())
    print(b.stride())

    # Correctness check
    ref = pytorch_baseline(bias, a, b, y)
    ours = tlx_fused_addmm_glu(bias, a, b, y)
    torch.testing.assert_close(ours, ref, atol=2e-2, rtol=2e-2)
    print("  correctness: OK")

    # Bench PyTorch baseline (2 ops)
    base_ms = triton.testing.do_bench(
        lambda: pytorch_baseline(bias, a, b, y),
        warmup=25,
        rep=200,
    )
    # Bench TLX-fused (1 kernel)
    tlx_ms = triton.testing.do_bench(
        lambda: tlx_fused_addmm_glu(bias, a, b, y),
        warmup=25,
        rep=200,
    )

    flops = 2 * M * N * K
    print(
        f"  Baseline (PyTorch addmm + GLU)   : {base_ms*1000:.2f} us  ({flops/(base_ms*1e-3)/1e12:.1f} TFLOPS)"
    )
    print(
        f"  TLX fused (1 kernel)             : {tlx_ms*1000:.2f} us  ({flops/(tlx_ms*1e-3)/1e12:.1f} TFLOPS)"
    )
    print(f"  Speedup (baseline / TLX)         : {base_ms/tlx_ms:.3f}x")
