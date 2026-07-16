import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_scaled_mm_ws import (
    blackwell_scaled_mm_ws as _tlx_scaled_mm, )

from triton._internal_testing import is_blackwell

# vLLM cutlass is an optional reference baseline; it may not be installed.
try:
    from vllm import _custom_ops as vllm_ops
    HAS_VLLM_CUTLASS = True
except ImportError:
    HAS_VLLM_CUTLASS = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# FP8 scaling recipes (see blackwell_scaled_mm_ws for the scale layouts).
SCALE_MODES = ["blockwise", "rowwise", "tensorwise"]

# Representative (M, N, K) shapes, N and K multiples of 128: squares plus igctr
# production non-square (tall = large N/small K, wide = small N/large K).
SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 4608, 6144),
    (4096, 6144, 4608),
    (4096, 16896, 3840),
    (4096, 10496, 9216),
]
"""
This script is used for benchmarking the performance of TLX tutorial kernels.
It's recommended to run with `third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_scaled_mm_perf.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def _make_scales(M, N, K, mode):
    # Canonical dequant scales per recipe (values are arbitrary for perf; only
    # the shapes/strides matter). The TLX wrapper reshapes rowwise/tensorwise
    # internally; torch._scaled_mm and vLLM take these shapes directly.
    if mode == "blockwise":
        # DeepSeek: scale_a M-major [M, K//128], scale_b row-major [N//128, K//128].
        scale_a = torch.rand(M, K // 128, device=DEVICE, dtype=torch.float32).t().contiguous().t()
        scale_b = torch.rand(N // 128, K // 128, device=DEVICE, dtype=torch.float32)
    elif mode == "rowwise":
        scale_a = torch.rand(M, 1, device=DEVICE, dtype=torch.float32)
        scale_b = torch.rand(1, N, device=DEVICE, dtype=torch.float32)
    else:  # tensorwise: one scalar per operand
        scale_a = torch.rand((), device=DEVICE, dtype=torch.float32)
        scale_b = torch.rand((), device=DEVICE, dtype=torch.float32)
    return scale_a, scale_b


def _providers(modes):
    # One "<backend>-<mode>" line per supported (backend, recipe): our TLX kernel
    # for all recipes; torch._scaled_mm for the K-independent recipes; and vLLM
    # cutlass (all recipes) when available.
    provs = [f"tlx-{m}" for m in modes]
    provs += [f"torch-{m}" for m in modes if m in ("tensorwise", "rowwise")]
    if HAS_VLLM_CUTLASS:
        provs += [f"vllm-{m}" for m in modes]
    return provs


def create_benchmark(modes):
    providers = _providers(modes)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=SHAPES,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOPS",
            plot_name="scaled-mm-performance-fp8",
            args={},
        ))
    def benchmark(M, N, K, provider):
        backend, mode = provider.split("-")
        # TLX blockwise stages per-K-group scales in SMEM (~12*K bytes); it OOMs at large K.
        if backend == "tlx" and mode == "blockwise" and K >= 12288:
            return float("nan"), float("nan"), float("nan")
        a = (torch.randn((M, K), device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn((N, K), device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
        scale_a, scale_b = _make_scales(M, N, K, mode)
        out_dtype = torch.bfloat16
        quantiles = [0.5, 0.2, 0.8]
        # All backends compute (scale_a * a) @ (scale_b * b^T); a=[M,K], b=[N,K].
        if backend == "tlx":
            fn = lambda: _tlx_scaled_mm(a, b, scale_a, scale_b, scale_mode=mode)
        elif backend == "torch":
            fn = lambda: torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
        else:  # vLLM cutlass; blockwise wants scale_b as [K//128, N//128], so transpose.
            if mode == "blockwise":
                major, minor = torch.cuda.get_device_capability(DEVICE)
                if not vllm_ops.cutlass_scaled_mm_supports_block_fp8(major * 10 + minor):
                    return float("nan"), float("nan"), float("nan")
            sb = scale_b.t() if mode == "blockwise" else scale_b
            fn = lambda: vllm_ops.cutlass_scaled_mm(a, b.t(), scale_a, sb, out_dtype=out_dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=2000, rep=2000)

        def tflops(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX Blackwell scaled_mm vs torch._scaled_mm / vLLM cutlass")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=SCALE_MODES,
        help=f"Run only the specified scaling recipe(s). Choices: {SCALE_MODES}",
    )
    args = parser.parse_args()

    if is_blackwell():
        modes = args.version if args.version else SCALE_MODES
        print(f"Running benchmarks for: {modes}")
        benchmark = create_benchmark(modes)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
