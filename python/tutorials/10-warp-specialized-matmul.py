import os
import sys

import torch
import triton
import triton.language as tl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
    )


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=0, # disable warp specialization
            num_buffers_warp_spec=3,
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def matmul_persistent_tma_ws_cooperative_annotated_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.assume(tl.cdiv(K, BLOCK_SIZE_K)>0)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            with tl.async_task([0]):
                a = tl._experimental_descriptor_load(
                    a_ptr,
                    [offs_am, offs_k],
                    [BLOCK_SIZE_M, BLOCK_SIZE_K],
                    tl.float16,
                )
                b = tl._experimental_descriptor_load(
                    b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16
                )

            accumulator += tl.dot(a, b)
            offs_k += BLOCK_SIZE_K

        c = accumulator.to(tl.float16)

        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            tl._experimental_descriptor_store(c_ptr, c, [offs_am, offs_bn])


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=0, # disable warp specialization
            num_buffers_warp_spec=3,
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def matmul_persistent_tma_ws_cooperative_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_CONSUMER_GROUPS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.assume(tl.cdiv(K, BLOCK_SIZE_K)>0)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl._experimental_descriptor_load(
                a_ptr,
                [offs_am, offs_k],
                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                tl.float16,
            )
            b = tl._experimental_descriptor_load(
                b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16
            )

            accumulator += tl.dot(a, b)
            offs_k += BLOCK_SIZE_K

        c = accumulator.to(tl.float16)
        tl._experimental_descriptor_store(c_ptr, c, [offs_am, offs_bn])


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul_persistent_tma_ws_cooperative(a, b, use_annotation=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            K,
            N,
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_N"],
            b.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    if use_annotation:
        matmul_persistent_tma_ws_cooperative_annotated_kernel[grid](
            desc_a,
            desc_b,
            desc_c,  #
            M,
            N,
            K,  #
        )
    else:
        matmul_persistent_tma_ws_cooperative_kernel[grid](
            desc_a,
            desc_b,
            desc_c,  #
            M,
            N,
            K,  #
        )
    return c


def aten_matmul(a, b):
    return a.mm(b)

def matmul_ws_annotated(a, b):
    return matmul_persistent_tma_ws_cooperative(a, b, use_annotation=True)

def matmul_ws_automatic(a, b):
    return matmul_persistent_tma_ws_cooperative(a, b, use_annotation=False)


test_impls = [
    aten_matmul,
    matmul_ws_annotated,
    matmul_ws_automatic,
]


impl_map = {fn.__name__: fn for fn in test_impls}


def test():
    torch.manual_seed(0)
    m = 4 * 11 * 64
    n = 12 * 256
    k = 64 * 4
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    torch_output = torch.matmul(a, b)
    rtol = 0
    for fn in test_impls:
        triton_output = fn(a, b)
        torch.cuda.synchronize()
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
            print(f" Torch matches {fn.__name__}")
        else:
            print(f" Torch DOES NOT match {fn.__name__}")
            print("torch output:")
            print(torch_output)
            print("triton output:")
            print(triton_output)


x_vals = [(8192, 8192, i) for i in range(256, 8192 + 1, 256)]
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=x_vals,
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[fn.__name__ for fn in test_impls],
        line_names=[fn.__name__ for fn in test_impls],
        # styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-"
        + (
            "fp16" 
        ),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    fn = impl_map[provider]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: fn(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


test()
benchmark.run(show_plots=True, print_data=True)
