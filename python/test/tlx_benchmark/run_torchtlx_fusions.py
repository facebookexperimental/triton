# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Validate the TorchInductor TLX local-buffer-retention prototype on MI350X."""

import argparse
import statistics

import torch
import torch.nn.functional as F
import triton  # @manual=//triton:triton

M = 1024
N = 6144
EPS = 1.0e-5


def double_layernorm_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    z = residual + F.layer_norm(x, (N, ), weight1, bias1, EPS)
    return z * torch.sigmoid(F.layer_norm(z, (N, ), weight2, bias2, EPS))


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    residual = torch.randn((M, N), device="cuda", dtype=torch.float16)
    parameters = tuple(torch.randn((N, ), device="cuda", dtype=torch.float16) for _ in range(4))
    return x, residual, *parameters


def compile_variant(mode: str | None, inputs: tuple[torch.Tensor, ...]):
    torch._dynamo.reset()
    overrides: dict[str, object] = {"triton.tlx_mode": mode}
    if mode is not None:
        # The retained kernel is offered as a MultiKernel choice and
        # triton.multi_kernel defaults to 0, so without this the candidate
        # compiles to exactly the baseline and the A/B measures nothing.
        overrides["triton.multi_kernel"] = 1
    with torch._inductor.config.patch(overrides):
        compiled = torch.compile(double_layernorm_silu, fullgraph=True)
        output = compiled(*inputs)
    torch.cuda.synchronize()
    return compiled, output


def bench_us(fn, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep)) * 1000.0


def main() -> None:
    global N
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    N = args.n

    torch.compiler.config.force_disable_caches = True
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape=({M}, {N}); dtype=fp16")

    inputs = make_inputs()
    eager = double_layernorm_silu(*inputs)
    baseline, baseline_out = compile_variant(None, inputs)
    candidate, candidate_out = compile_variant("allow", inputs)

    torch.testing.assert_close(baseline_out, eager, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_out, baseline_out, atol=2e-2, rtol=2e-2)
    diff = (candidate_out.float() - baseline_out.float()).abs()
    print("correctness_vs_baseline "
          f"max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f}")

    variants = {
        "baseline": lambda: baseline(*inputs),
        "local_buffer_retention": lambda: candidate(*inputs),
    }
    orders = (
        ("baseline", "local_buffer_retention"),
        ("local_buffer_retention", "baseline"),
    )
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for sample in range(args.samples):
        for name in orders[sample % len(orders)]:
            samples[name].append(bench_us(variants[name], args.warmup, args.rep))
        print(f"sample={sample + 1} "
              f"baseline={samples['baseline'][-1]:.3f}us "
              f"local_buffer_retention={samples['local_buffer_retention'][-1]:.3f}us")

    baseline_median = statistics.median(samples["baseline"])
    candidate_median = statistics.median(samples["local_buffer_retention"])
    print(f"FINAL baseline={baseline_median:.3f}us "
          f"local_buffer_retention={candidate_median:.3f}us "
          f"speedup={baseline_median / candidate_median:.3f}x")


if __name__ == "__main__":
    main()
