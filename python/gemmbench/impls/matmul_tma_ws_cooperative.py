import torch
import triton
import triton.language as tl

# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
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


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_tma_ws_cooperative_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am1 = pid_m * BLOCK_SIZE_M
    offs_am2 = pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M // 2
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    accumulator1 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        with tl.async_task([0]):
            a1 = tl._experimental_descriptor_load(
                a_ptr, [offs_am1, offs_k], [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], tl.float16
            )
            b = tl._experimental_descriptor_load(
                b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16
            )
            a2 = tl._experimental_descriptor_load(
                a_ptr, [offs_am2, offs_k], [BLOCK_SIZE_M // 2, BLOCK_SIZE_K], tl.float16
            )

        accumulator1 += tl.dot(a1, b)
        accumulator2 += tl.dot(a2, b)
        offs_k += BLOCK_SIZE_K

    c1 = accumulator1.to(tl.float16)
    c2 = accumulator2.to(tl.float16)

    with tl.async_task([1]):
        tl._experimental_descriptor_store(c_ptr, c1, [offs_am1, offs_bn])
    with tl.async_task([2]):
        tl._experimental_descriptor_store(c_ptr, c2, [offs_am2, offs_bn])


def matmul_tma_ws_cooperative(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
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
            META["BLOCK_SIZE_M"] // 2,
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
            META["BLOCK_SIZE_M"] // 2,
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
    matmul_tma_ws_cooperative_kernel[grid](
        desc_a,
        desc_b,
        desc_c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
    )
    return c