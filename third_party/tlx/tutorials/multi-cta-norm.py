# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


DEVICE = triton.runtime.driver.active.get_active_torch_device()

sum_only = True
kernel_configs = [triton.Config({"BLOCK_SIZE_M": m}, num_warps=nw) for m in [1, 2] for nw in [1, 2, 4, 8, 16, 32]]
kernel_configs_multi_cta = [triton.Config({"BLOCK_SIZE_M": m, "num_reduction_ctas" : ctas}, num_warps=nw) for m in [1, 2] for nw in [1, 2, 4, 8, 16, 32] for ctas in [2, 4, 8]]
torch.manual_seed(42)

@triton.autotune(
    configs = kernel_configs,
    key=['M', 'N'],
)
@triton.jit
def kernel_norm(
    X,
    Y,
    row_stride,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # pdb.set_trace()
    row_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask_row = row_offsets < M
    mask_col = col_offsets < N
    read_write_offsets = (row_offsets[:, None] * row_stride) + col_offsets[None, :]

    X_ptr = X + read_write_offsets
    Y_ptr = Y + read_write_offsets
    
    read_write_mask = mask_row[:, None] & mask_col[None, :]
    x = tl.load(X_ptr, mask=read_write_mask, other=0.0)
    sum = tl.sum(x, axis=1, keep_dims=True)
    norm = x / sum
    tl.store(Y_ptr, norm, mask=read_write_mask)

@triton.autotune(
    configs = kernel_configs_multi_cta,
    key=['M', 'N'],
)
@triton.heuristics({"BLOCK_SIZE_N": lambda args: triton.next_power_of_2(args["N"] // args["num_reduction_ctas"])})
@triton.jit
def kernel_norm_multi_cta(
    X,
    Y,
    row_stride,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_reduction_ctas: tl.constexpr,
):
    tlx.set_num_reduction_ctas(num_reduction_ctas)
    row_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Partition reduction axes over multiple CTAs
    col_offsets = (tl.program_id(1) % num_reduction_ctas) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_row = row_offsets < M
    mask_col = col_offsets < N
    read_write_offsets = (row_offsets[:, None] * row_stride) + col_offsets[None, :]
    read_write_mask = mask_row[:, None] & mask_col[None, :]

    X_ptr = X + read_write_offsets
    Y_ptr = Y + read_write_offsets

    x = tl.load(X_ptr, mask=read_write_mask, other=0.0)
    local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
    # local memory used in tl.sum warp reduction can be reused by the allocator
    # for the subsequent local_alloc. Since the remote CTA writes to the buffer 
    # alloced by local_alloc. We need to protect agains the WAR hazard of the
    # remote write from, say Cluster1, overwriting the shmem contents before warp 
    # reduction is complete in , say Cluster0. 
    tlx.cluster_barrier()

    local_buff = tlx.local_alloc((BLOCK_SIZE_M, 1), tlx.dtype_of(X_ptr), num_reduction_ctas - 1)
    cta_rank = tlx.cluster_cta_rank()
    # send reduction result to other cluster
    for i in range(num_reduction_ctas):
        if i != cta_rank:
            remote_local_buff_view = tlx.local_view(local_buff, cta_rank)
            tlx.remote_shmem_store(dst=remote_local_buff_view, src=local_partial_sum, remote_cta_rank=i)
    tlx.cluster_barrier()

    final_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tlx.dtype_of(X_ptr))

    for i in range(num_reduction_ctas):
        if i == cta_rank:
            final_sum += local_partial_sum
        else:
            remote_local_buff_view = tlx.local_view(local_buff, i)
            final_sum += tlx.local_load(remote_local_buff_view)
    norm = x / final_sum
    tl.store(Y_ptr, norm, mask=read_write_mask)


def do_norm(x, y, M, N, dtype, single_cta=True):
    def grid_1d(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    
    def grid_2d(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    if not single_cta:
        kernel_norm_multi_cta[grid_2d](
            x,
            y,
            x.stride(0),
            M,
            N,
        )
    else:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        kernel_norm[grid_1d](
            x,
            y,
            x.stride(0),
            M,
            N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    return y


quantiles = [0.5, 0.2, 0.8]
shapes = [(4, 16*1024), (16, 16*1024), (4, 32*1024), (16, 32*1024), (4, 131072), (16, 131072)]
# shapes = [(4, 16*1024)]

impls = ["tlx-norm", "triton-1-cta-norm", "torch-norm"]

benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=[f"{s[0]},{s[1]}" for s in shapes],
        args={"dtype": torch.float32},
        line_arg="provider",
        # line_vals=["triton-1-cta"],
        # line_names=["triton-1-cta"],
        line_vals=impls,
        line_names=impls,
        plot_name="norm",
    )
]

def test_norm(M, N, dtype, device=DEVICE):
    x = torch.randn((M, N), dtype=dtype, device=device)
    y = torch.randn((M, N), dtype=dtype, device=device)
    ref_sum = torch.sum(x, dim=1, keepdim=True) 
    ref_norm = x / ref_sum
    norm = do_norm(x, y, M, N, dtype, single_cta=False)
    print("Verifying norm with torch")
    print(x.shape, dtype)
    if torch.allclose(norm, ref_norm, atol=1e-2, rtol=0):
        print("PASS")
    else:
        print("FAIL")
        print(ref_norm)
        print(norm)


@triton.testing.perf_report(benchmark_configs)
def benchmark(shape:str, provider, dtype):
    M, N = shape.split(",")
    M = int(M)
    N = int(N)
    x = torch.randn((M, N), dtype=dtype, device=DEVICE)
    if provider == "tlx-norm":    
        y = torch.randn((M, N), dtype=dtype, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_norm(x, y, M, N, dtype, single_cta=False), quantiles=quantiles
        )
    elif provider == "torch-norm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x / torch.sum(x, dim=1, keepdim=True), quantiles=quantiles
        )
    elif provider == "triton-1-cta-norm":
        y = torch.zeros((M, N), dtype=dtype, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_norm(x, y, M, N, dtype, single_cta=True), quantiles=quantiles
        )
    perf = lambda ms: M * N * 1e-12 / (ms * 1e-3)
    
    print(shape, provider, perf(ms), perf(max_ms), perf(min_ms))
    return perf(ms), perf(max_ms), perf(min_ms)


for s in shapes:
    test_norm(s[0], s[1], torch.float32, device=DEVICE)
benchmark.run(save_path=".", print_data=True)
