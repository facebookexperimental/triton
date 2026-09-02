"""Perf harness for the TLX AMD GDPA forward kernel (gfx950 / MI350, CDNA4).

Run:
    python third_party/tlx/tutorials/testing/test_gfx950_gdpa_perf.py
    python third_party/tlx/tutorials/testing/test_gfx950_gdpa_perf.py --version tlx --shape prod

Reference points on the production shape (OmniFM V3 pFFN:
B=2048, max_M=500, D=256, H=4, head_dim=64, dff=256, sparsity=0.68, bf16),
measured on MI350X with the stock Triton GDPA kernel:

    forward, no AMD optimizations       1.4714 ms /  58.4 TFLOPS
    + gelu fast math (sigmoid)          0.7694 ms / 111.7 TFLOPS
    + PID swizzle (size=8)              0.8015 ms / 107.2 TFLOPS
    + matrix_instr_nonkdim=16           0.8441 ms / 101.8 TFLOPS
    + waves_per_eu (all opts)           0.5927 ms / 144.9 TFLOPS   <- target

TFLOPS here uses the same formula as that report
(`fwd_flops = 4 * H * d * total_Q * kv_len`); at total_Q = 327,680 and
0.5927 ms it reproduces 144.9 TFLOPS exactly, so the numbers are directly
comparable.

That total_Q corresponds to a *uniform* sequence length of 160
(= (1 - 0.68) * 500) across all 2048 batches, which is the default
`seq_len_mode="uniform"` here. The Blackwell GDPA generator reads sparsity
differently -- randint over [(2*sp - 1)*max_M, max_M), average ~337 -- which is
2.1x the work. That mode is available as `--seq-len-mode random` and is the
harder scheduling case, but its latency is *not* comparable to the numbers
above. total_Q is always measured from the generated offsets, never assumed.
"""

import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.gfx950_gdpa import (
    SEQ_LEN_MODES,
    generate_gdpa_data,
    gdpa_ref,
    gdpa_tflops,
    fast_gelu_ref,
    get_kernel,
)

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# name -> (B, max_M, D, H, dff, sparsity)
SHAPES = {
    # The production target.
    "prod": (2048, 500, 256, 4, 256, 0.68),
    # Same geometry, fully dense Q -- isolates the jagged scheduling cost.
    "prod_dense": (2048, 500, 256, 4, 256, 1.0),
    # Smaller batch, same per-sequence work -- occupancy / tail behaviour.
    "small_batch": (256, 500, 256, 4, 256, 0.68),
    # Wider KV -- shifts the balance toward the second GEMM.
    "wide_kv": (2048, 500, 256, 4, 512, 0.68),
}
DEFAULT_SHAPES = ["prod"]

PROVIDERS = ["tlx", "torch"]
DEFAULT_PROVIDERS = ["tlx"]


def _torch_eager(data):
    """Eager torch baseline: the same math via the reference path.

    This is a sanity floor, not a competitive baseline -- `gdpa_ref` loops over
    batches in Python, so at B=2048 it is dominated by launch overhead. The
    meaningful comparison is against the 144.9 TFLOPS Triton number recorded in
    the module docstring.
    """
    return lambda: gdpa_ref(
        data["q"],
        data["k"],
        data["v"],
        data["q_offsets"],
        data["dff"],
        qk_scale=1.0,
        activation=fast_gelu_ref,
    )


def create_benchmark(shapes, providers, kernel_name="pipelined", seq_len_mode="uniform"):
    x_vals = [SHAPES[s] for s in shapes]
    x_ids = list(shapes)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B", "max_M", "D", "H", "dff", "sparsity"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOPS",
            plot_name=f"gfx950-gdpa-forward-bf16-{seq_len_mode}",
            args={"kernel_name": kernel_name, "seq_len_mode": seq_len_mode},
        ))
    def benchmark(B, max_M, D, H, dff, sparsity, provider, kernel_name, seq_len_mode):
        data = generate_gdpa_data(B, max_M, D, H, dff, sparsity=sparsity, dtype=torch.bfloat16, device=DEVICE,
                                  seq_len_mode=seq_len_mode)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            fn = _torch_eager(data)
        elif provider == "tlx":
            gdpa = get_kernel(kernel_name)
            fn = lambda: gdpa(data["q"], data["k"], data["v"], data["q_offsets"], data["dff"], qk_scale=1.0)
        else:
            raise ValueError(f"unknown provider {provider!r}")

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=100, rep=200)

        perf = lambda t: gdpa_tflops(t, data["total_q"], H, data["d"], dff)
        return perf(ms), perf(max_ms), perf(min_ms)

    return benchmark, x_ids


def _print_shape_summary(shapes, seq_len_mode):
    print(f"Shapes under test (seq_len_mode={seq_len_mode}, total_q measured from generated offsets):")
    for name in shapes:
        B, max_M, D, H, dff, sparsity = SHAPES[name]
        data = generate_gdpa_data(B, max_M, D, H, dff, sparsity=sparsity, dtype=torch.bfloat16, device=DEVICE,
                                  seq_len_mode=seq_len_mode)
        note = ""
        if name == "prod" and data["total_q"] == 327680:
            note = "  <- matches the 144.9 TFLOPS reference total_q"
        print(f"  {name:<12} B={B} max_M={max_M} D={D} H={H} d={data['d']} dff={dff} "
              f"sp={sparsity} total_q={data['total_q']} (avg seq {data['total_q'] / B:.1f}){note}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the TLX AMD GDPA forward kernel")
    parser.add_argument("--shape", type=str, nargs="+", choices=list(SHAPES), default=None,
                        help=f"Shapes to run. Choices: {list(SHAPES)}")
    parser.add_argument("--version", type=str, nargs="+", choices=PROVIDERS, default=None,
                        help=f"Providers to run. Choices: {PROVIDERS}")
    parser.add_argument("--kernel", type=str, default="pipelined", help="Kernel variant from KERNEL_REGISTRY")
    parser.add_argument("--seq-len-mode", type=str, choices=list(SEQ_LEN_MODES), default="uniform",
                        help="Sparsity convention; 'uniform' reproduces the reference total_q")
    args = parser.parse_args()

    if not is_hip():
        print("Skipping benchmarks, no AMD GPU found.")
        raise SystemExit(0)

    shapes = args.shape if args.shape else DEFAULT_SHAPES
    providers = args.version if args.version else DEFAULT_PROVIDERS
    print(f"Running GDPA benchmarks: shapes={shapes} providers={providers} "
          f"kernel={args.kernel} seq_len_mode={args.seq_len_mode}")
    _print_shape_summary(shapes, args.seq_len_mode)
    benchmark, _ = create_benchmark(shapes, providers, kernel_name=args.kernel, seq_len_mode=args.seq_len_mode)
    benchmark.run(print_data=True)
