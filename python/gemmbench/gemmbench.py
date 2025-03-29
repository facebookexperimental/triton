import os
import sys

import torch
import triton
import triton.language as tl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import impls


if torch.cuda.is_available():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


def cublas_matmul(a, b):
    # Check constraints.
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    c = torch.matmul(a, b.T)
    return c


import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.cache_size_limit = 1000000
# torch._inductor.config.cuda.cutlass_op_allowlist_regex = "(64x256x16.*128x256x64.*cooperative|64x128x16.*128x128x64.*pingpong).*tma"
# torch._inductor.config.cuda.cutlass_op_denylist_regex = "stream_k"


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def inductor_matmul(a, b):
    return a @ b


def aten_matmul(a, b):
    return a.mm(b)


test_impls = [
    #aten_matmul,
    # cublas_matmul,
    # inductor_matmul,
    impls.matmul,
    impls.matmul_tma_ws, # this works
    impls.matmul_tma_ws_cooperative,
    #impls.matmul_ws_cooperative,
    #impls.matmul_persistent,
    #impls.matmul_persistent_cooperative,
    #impls.matmul_persistent_tma,
    #impls.matmul_persistent_tma_ws,
    impls.matmul_persistent_tma_ws_cooperative,
    #impls.matmul_persistent_tma_ws_cooperative_manual,
    #impls.matmul_persistent_ws,
    #impls.matmul_persistent_ws_cooperative,
    #impls.matmul_persistent_ws_cooperative_manual,
    #impls.matmul_1d_persistent_swp_tma,
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
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else 0
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
            #torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=rtol)


TORCH_HAS_FP8 = False  # hasattr(torch, "float8_e5m2")

GEMM_SHAPES = os.environ.get("GEMM_SHAPES", "")
if GEMM_SHAPES == "sweepk":
    x_vals = [(8192, 8192, i) for i in range(256, 8192 + 1, 256)]
elif GEMM_SHAPES == "square":
    x_vals = [(i, i, i) for i in range(256, 4096 + 1, 128)]
elif GEMM_SHAPES == "llama":
    KN = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
        (16384, 2304),
        (2048, 16384),
        (16384, 13312),
        (6656, 16384),
    ]
    x_vals = [(m, n, k) for m in [128, 256, 384, 512] for (k, n) in KN]
else:
    # Simple shape with 4 waves over 132 SMs
    x_vals = [(4 * 11 * 128, 12 * 256, 4096)]


configs = []
for fp8_inputs in [False]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
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
                "fp16" if not fp8_inputs else "fp8"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    fn = impl_map[provider]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: fn(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


test()
benchmark.run(show_plots=True, print_data=True, save_path=f"./{GEMM_SHAPES}")
