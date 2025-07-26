"""
Vector Addition
===============

Performs two independent elementwise additions in parallel:

out1 = x + y
out2 = a + b

Each addition is applied across corresponding elements of input vectors, producing
two output vectors of the same shape.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add2_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output1 = x + y
    output2 = a + b
    tl.store(z_ptr + offsets, output1, mask=mask)
    tl.store(c_ptr + offsets, output2, mask=mask)


def add2(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    assert x.device == DEVICE and y.device == DEVICE and output1.device == DEVICE
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE=1024)
    return output1, output2


@triton.jit
def add2_warp_specialized_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)
        with tlx.async_task(num_warps=4):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            output = a + b
            tl.store(c_ptr + offsets, output, mask=mask)


def add2_warp_specialized(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    assert x.device == DEVICE and y.device == DEVICE and output1.device == DEVICE
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_warp_specialized_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE=1024)
    return output1, output2


def dual_add(x, y, a, b):
    return x + y, a + b


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
output_torch_1, output_torch_2 = dual_add(x, y, a, b)
output_triton_1, output_triton_2 = add2(x, y, a, b)
output_triton_ws_1, output_triton_ws_2 = add2_warp_specialized(x, y, a, b)

print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(output_torch_1 - output_triton_1))}")
print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(output_torch_2 - output_triton_2))}")
print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(output_torch_1 - output_triton_ws_1))}")
print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(output_torch_2 - output_triton_ws_2))}")
# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "triton_ws", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Triton_WS", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    a = torch.rand(size, device=DEVICE, dtype=torch.float32)
    b = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dual_add(x, y, a, b), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2(x, y, a, b), quantiles=quantiles)
    if provider == "triton_ws":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add2_warp_specialized(x, y, a, b), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
