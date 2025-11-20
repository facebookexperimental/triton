# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


DEVICE = triton.runtime.driver.active.get_active_torch_device()

sum_only = True
kernel_configs = [triton.Config({"BLOCK_SIZE_M": m}, num_warps=nw) for m in [1, 2, 4] for nw in [1, 2, 4, 8, 16, 32]]
torch.manual_seed(42)

@triton.autotune(
    configs = kernel_configs,
    key=['M', ''],
)
@triton.jit
def kernel_sum_keep_dim(
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

    store_col_offsets = tl.arange(0, 1)
    store_offsets = (row_offsets[:, None] * 1) + store_col_offsets[None, :]
    store_col_mask = store_col_offsets < 1
    store_mask = mask_row[:, None] & store_col_mask[None, :]

    X_ptr = X + (row_offsets[:, None] * row_stride) + col_offsets[None, :]
    Y_ptr = Y + store_offsets

    x = tl.load(X_ptr, mask=mask_row[:, None] & mask_col[None, :], other=0.0)
    sum = tl.sum(x, axis=1, keep_dims=True)
    tl.store(Y_ptr, sum, mask=store_mask)

@triton.autotune(
    configs = kernel_configs,
    key=['M', 'N'],
)
@triton.jit
def kernel_sum_keep_dim_multi_cta(
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

    store_col_offsets = tl.arange(0, 1)
    store_offsets = (row_offsets[:, None] * 1) + store_col_offsets[None, :]
    store_col_mask = store_col_offsets < 1
    store_mask = mask_row[:, None] & store_col_mask[None, :]

    X_ptr = X + (row_offsets[:, None] * row_stride) + col_offsets[None, :]
    Y_ptr = Y + store_offsets

    x = tl.load(X_ptr, mask=mask_row[:, None] & mask_col[None, :], other=0.0)
    local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
    # local memory used in tl.sum warp reduction can be reused by the allocator
    # for the subsequent local_alloc. Since the remote CTA writes to the buffer 
    # alloced by local_alloc. We need to protect agains the WAR hazard of the
    # remote write from, say Cluster1, overwriting the shmem contents before warp 
    # reduction is complete in , say Cluster0. 
    tlx.cluster_barrier()

    local_buff = tlx.local_alloc((BLOCK_SIZE_M, 1), tlx.dtype_of(X_ptr), 1)
    local_buff_view = tlx.local_view(local_buff, 0)
    # send reduction result from cluster1->cluster0 
    if tlx.cluster_cta_rank() != 0:
        tlx.remote_shmem_store(dst=local_buff_view, src=local_partial_sum, remote_cta_rank=0)   
    tlx.cluster_barrier()

    # compute final sum in cluster0
    if tlx.cluster_cta_rank() == 0:
        remote_partial_sum = tlx.local_load(local_buff_view)
        final_sum = local_partial_sum + remote_partial_sum
        tl.store(Y_ptr, final_sum, mask=store_mask)


def do_sum(x, y, M, N, dtype, single_cta=True):
    def grid_1d(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    
    def grid_2d(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    if not single_cta:
        num_reduction_ctas = 2
        BLOCK_SIZE_N = triton.next_power_of_2(N // num_reduction_ctas)
        kernel_sum_keep_dim_multi_cta[grid_2d](
            x,
            y,
            x.stride(0),
            M,
            N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_reduction_ctas=num_reduction_ctas,
        )
    else:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        kernel_sum_keep_dim[grid_1d](
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

impls = ["tlx-sum", "triton-1-cta", "torch.sum"]
# impls = ["tlx-sum"]
# impls = ["tlx-sum", "triton-1-cta"]


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
        plot_name="sum (X, Y) -> (X, 1)",
    )
]

def test_sum(M, N, dtype, device=DEVICE):
    x = torch.randn((M, N), dtype=dtype, device=device)
    y = torch.randn((M), dtype=dtype, device=device)
    y = torch.unsqueeze(y, 1)
    y_ref = torch.sum(x, dim=1, keepdim=True)
    y = do_sum(x, y, M, N, dtype, single_cta=False)
    print("Verifying tlx 2cta sum with torch.sum")
    print(x.shape, dtype)
    if torch.allclose(y, y_ref, atol=1e-2, rtol=0):
        print("PASS")
    else:
        print("FAIL")
        print(y_ref)
        print(y)


@triton.testing.perf_report(benchmark_configs)
def benchmark(shape:str, provider, dtype):
    M, N = shape.split(",")
    M = int(M)
    N = int(N)
    x = torch.randn((M, N), dtype=dtype, device=DEVICE)
    if provider == "tlx-sum":    
        y = torch.randn((M, N), dtype=dtype, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_sum(x, y, M, N, dtype, single_cta=False), quantiles=quantiles
        )
    elif provider == "torch.sum":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x, dim=1, keepdim=True), quantiles=quantiles
        )
    elif provider == "triton-1-cta":
        y = torch.zeros((M), dtype=dtype, device=DEVICE)
        y = torch.unsqueeze(y, 1)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_sum(x, y, M, N, dtype, single_cta=True), quantiles=quantiles
        )
    perf = lambda ms: M * N * 1e-12 / (ms * 1e-3)
    
    print(shape, provider, perf(ms), perf(max_ms), perf(min_ms))
    return perf(ms), perf(max_ms), perf(min_ms)


for s in shapes:
    test_sum(s[0], s[1], torch.float32, device=DEVICE)
benchmark.run(save_path=".", print_data=True)
