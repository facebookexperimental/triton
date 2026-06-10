import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

M, N = 1024, 21568

# gfx950 has 8 XCDs per chip
NUM_XCDS = 8
XCD_CHUNK = 4

# Autotuning winning configurations
BEST_CONFIG = {
      256:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, num_warps=4, matrix_instr_nonkdim=16,
  waves_per_eu=0),
      512:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, matrix_instr_nonkdim=16,
  waves_per_eu=0),
      1024: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_warps=8, matrix_instr_nonkdim=16,
  waves_per_eu=0),
  }

# L2 swizzling
@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


# Warp-pipelined async direct-to-LDS fused addmm + GLU kernel
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
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)

    # Num pids along M dim
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # Num pids along N dim
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # How many pids we have in total grid
    grid_mn = num_pid_m * num_pid_n

    # Remap pids via L2 swizzle
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)

    # Num pids in the group (of shared rows along M dim pids)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Which group we are in
    group_id = pid // num_pid_in_group

    # Index of first pid in this group
    first_pid_m = group_id * GROUP_SIZE_M

    # Either GROUP_SIZE_M or smaller due to last group cutting off early
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Which pid along m dim
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    # Which pid along n dim
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Get offsets this pid is responsible for in m dim
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    # Get offsets this pid is responsible for in n dim
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # Get offsets this pid is responsible for in k dim
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers for A along m dim
    a_base_off = offs_m[:, None] * sa0

    # Base pointers for B along n dim
    b_base_off = offs_n[None, :] * sb1

    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    # Create multibuffered shared memory arrays
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), 3)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), 3)

    # Prologue: async direct-to-LDS prefetch of NUM_BUFFERS tiles
    # Use static range
    # for i in tl.static_range(0, NUM_BUFFERS):
    #     k_start = i * BLOCK_SIZE_K

    #     # Get A tile for the iteration k that we are on
    #     a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1

    #     # Get B tile for the iteration k that we are on
    #     b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off

    #     # Fetch tiles asynchronously
    #     tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, i), mask=offs_k[None, :] < K - k_start)
    #     tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, i), mask=offs_k[:, None] < K - k_start)
    #     tlx.async_load_commit_group([tok_a, tok_b])

    # GR 0
    k_start = 0
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
    
    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 0), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 0), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # GR 1
    k_start = BLOCK_SIZE_K
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
    
    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 1), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 1), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # GR 2
    k_start = BLOCK_SIZE_K * 2
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
    
    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 2), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 2), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # Make GR0 and GR1 finish
    tlx.async_load_wait_group(1)

    # Transfer to registers
    a_tile = tlx.local_load(tlx.local_view(smemA, 0))
    b_tile = tlx.local_load(tlx.local_view(smemB, 0))

    # Create accumulator array
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Hot loop: warp-pipelined MFMA / async-prefetch
    for tile_id in tl.range(0, k_iters - 3, loop_unroll_factor=0):
        # Index within the multibuffered circular SMEM where we are storing the global prefetch
        prefetch_buf = tile_id % 3

        # Index of local prefetch buffer that we need to transfer into registers
        next_buf = (tile_id + 1) % 3

        # Which tile we need to prefetch globally along the k dim
        k_prefetch = (tile_id + 3) * BLOCK_SIZE_K

        # Execute MFMA with the data we already have loaded into registers
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        with tlx.warp_pipeline_stage("mem", priority=1):
            # Perform global prefetching
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
            b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off

            tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf),
                                   mask=offs_k[None, :] < K - k_prefetch)
            tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf),
                                   mask=offs_k[:, None] < K - k_prefetch)
            
            tlx.async_load_commit_group([tok_a, tok_b])

            # Perform local prefetching
            a_tile = tlx.local_load(tlx.local_view(smemA, next_buf))
            b_tile = tlx.local_load(tlx.local_view(smemB, next_buf))

        # Most recently committed buffers can be in flight
        tlx.async_load_wait_group(1)

    # Async load the y and bias

    # Epilogue: drain the remaining in-flight tiles
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Wait for all buffers to be loaded before draining in epilogue loop
    tlx.async_load_wait_group(0)

    # Use static range here
    for i in tl.static_range(0, 2):
        buf = (k_iters - 2 + i) % 3

        a_tile = tlx.local_load(tlx.local_view(smemA, buf))
        b_tile = tlx.local_load(tlx.local_view(smemB, buf))

        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Fused epilogue: addmm + GLU ===
    # Add bias (broadcast over M), then GLU: X = X + X*Y
    # Streaming cache hints (.cs) on Y and C: both are touched once and never
    # reused, so we avoid polluting L2 with this 44MB+44MB epilogue traffic

    # y and bias load into registers here

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    x = acc + bias[None, :]

    y_ptrs = y_ptr + offs_cm[:, None] * sy0 + offs_cn[None, :] * sy1
    y = tl.load(y_ptrs, mask=c_mask, other=0.0, cache_modifier=".cs").to(tl.float32)
    out = x + x * y

    c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


def run_kernel(a, b, bias, y, out, cfg):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    return tlx_addmm_glu_kernel[grid](
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
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=XCD_CHUNK,
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )

def pytorch_baseline(bias, a, b, y):
    # Reference: addmm (x = bias + a@b) then GLU (out = x + x*y)
    x = torch.addmm(bias, a, b).to(torch.float32)
    return (x + x * y.to(torch.float32)).to(torch.float16)

# Old Addmm kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_STAGES": 2, "kpack": 1, "matrix_instr_nonkdim": 16, "waves_per_eu": 0}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_STAGES": 2, "kpack": 1, "matrix_instr_nonkdim": 16, "waves_per_eu": 2}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_STAGES": 2, "kpack": 1, "matrix_instr_nonkdim": 16, "waves_per_eu": 2}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_STAGES": 2, "kpack": 1, "matrix_instr_nonkdim": 16, "waves_per_eu": 0}, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def tlx_addmm_glu_kernel_old(
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


def tlx_fused_addmm_glu_old(bias, a, b, y):
    M, K = a.shape
    K2, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    tlx_addmm_glu_kernel_old[grid](
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


def main():
    tflops = lambda ms, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    K_VALUES = [256, 512, 1024]
    results = []

    for K in K_VALUES:
        cfg = BEST_CONFIG[K]
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.randn(M, N, device="cuda", dtype=torch.float16)
        out = torch.empty((M, N), device="cuda", dtype=torch.float16)

        # Correctness check against PyTorch
        run_kernel(a, b, bias, y, out, cfg)
        torch.cuda.synchronize()
        ref = pytorch_baseline(bias, a, b, y)
        max_err = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

        ms_rocblas = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=200)
        ms_old = triton.testing.do_bench(lambda: tlx_fused_addmm_glu_old(bias, a, b, y), warmup=25, rep=200)
        ms_opt = triton.testing.do_bench(lambda: run_kernel(a, b, bias, y, out, cfg), warmup=25, rep=200)
        results.append((K, max_err, ms_rocblas, ms_old, ms_opt))

    print(f"\nM={M} N={N}  fp16   (TFLOPS = 2*M*N*K / time)\n")
    hdr = (f"{'K':>6} {'ROCBLAS':>9} {'original':>9} {'new':>9} "
           f"{'speedup':>8} {'max_err':>9}")
    print(hdr)
    print("-" * len(hdr))
    for K, err, ms_r, ms_o, ms_p in results:
        print(f"{K:>6} {tflops(ms_r, K):>9.1f} {tflops(ms_o, K):>9.1f} {tflops(ms_p, K):>9.1f} "
              f"{ms_o/ms_p:>7.2f}x {err:>9.2e}")
    print("\nROCBLAS = torch.matmul A@B only; original/new = fused addmm+GLU. speedup = original/new.")


if __name__ == "__main__":
    main()
