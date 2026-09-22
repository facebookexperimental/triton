# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Validate the TorchInductor TLX local-buffer-retention prototype on MI350X."""

import argparse
import dataclasses
import statistics
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
import triton  # @manual=//triton:triton

M = 1024
N = 6144
EPS = 1.0e-5


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    name: str
    problem: Callable[[], str]
    model: Callable[..., torch.Tensor]
    make_inputs: Callable[[], tuple[torch.Tensor, ...]]
    candidate_name: str
    candidate_config: Mapping[str, object]
    atol: float = 2.0e-2
    rtol: float = 2.0e-2


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


DOUBLE_LAYERNORM = BenchmarkCase(
    name="gfx950_01_double_layernorm",
    problem=lambda: f"M={M} N={N} dtype=fp16",
    model=double_layernorm_silu,
    make_inputs=make_inputs,
    candidate_name="local_buffer_retention",
    # Retention is offered as a MultiKernel choice; the default value 0 would
    # compile the candidate into the baseline and make the A/B meaningless.
    candidate_config={"triton.tlx_mode": "allow", "triton.multi_kernel": 1},
)

CASES = {case.name: case for case in (DOUBLE_LAYERNORM, )}


def compile_variant(
    case: BenchmarkCase,
    config: Mapping[str, object],
    inputs: tuple[torch.Tensor, ...],
):
    torch._dynamo.reset()
    with torch._inductor.config.patch(config):
        compiled = torch.compile(case.model, fullgraph=True)
        output = compiled(*inputs)
    torch.cuda.synchronize()
    return compiled, output


def bench_us(fn, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep)) * 1000.0


def main() -> None:
    global N
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default=DOUBLE_LAYERNORM.name)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    N = args.n
    if args.list:
        for case in CASES.values():
            print(f"{case.name}: {case.problem()}")
        return
    case = CASES[args.case]

    torch.compiler.config.force_disable_caches = True
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"case={case.name}; {case.problem()}")

    inputs = case.make_inputs()
    eager = case.model(*inputs)
    baseline, baseline_out = compile_variant(case, {"triton.tlx_mode": None}, inputs)
    candidate, candidate_out = compile_variant(case, case.candidate_config, inputs)

    torch.testing.assert_close(baseline_out, eager, atol=case.atol, rtol=case.rtol)
    torch.testing.assert_close(candidate_out, baseline_out, atol=case.atol, rtol=case.rtol)
    diff = (candidate_out.float() - baseline_out.float()).abs()
    print("correctness_vs_baseline "
          f"max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f}")

    variants = {
        "baseline": lambda: baseline(*inputs),
        case.candidate_name: lambda: candidate(*inputs),
    }
    orders = (
        ("baseline", case.candidate_name),
        (case.candidate_name, "baseline"),
    )
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for sample in range(args.samples):
        for name in orders[sample % len(orders)]:
            samples[name].append(bench_us(variants[name], args.warmup, args.rep))
        print(f"sample={sample + 1} "
              f"baseline={samples['baseline'][-1]:.3f}us "
              f"{case.candidate_name}={samples[case.candidate_name][-1]:.3f}us")

    baseline_median = statistics.median(samples["baseline"])
    candidate_median = statistics.median(samples[case.candidate_name])
    print(f"FINAL baseline={baseline_median:.3f}us "
          f"{case.candidate_name}={candidate_median:.3f}us "
          f"speedup={baseline_median / candidate_median:.3f}x")


if __name__ == "__main__":
    main()
