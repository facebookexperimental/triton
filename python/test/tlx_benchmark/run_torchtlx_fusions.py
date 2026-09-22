# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Validate the TorchInductor TLX local-buffer-retention prototype on MI350X."""

import argparse
import dataclasses
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
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
    baseline_config: Mapping[str, object]
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
    baseline_config={"triton.tlx_mode": None},
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default=DOUBLE_LAYERNORM.name)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmup <= 0 or args.rep <= 0:
        parser.error("--warmup and --rep must be positive")
    return args


def run_variant(case: BenchmarkCase, variant: str, args) -> dict[str, object]:
    config = case.baseline_config if variant == "baseline" else case.candidate_config
    label = "baseline" if variant == "baseline" else case.candidate_name

    torch.compiler.config.force_disable_caches = True
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False

    print(f"variant={label}; torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"case={case.name}; {case.problem()}")

    inputs = case.make_inputs()
    eager = case.model(*inputs)
    compiled, output = compile_variant(case, config, inputs)
    torch.testing.assert_close(output, eager, atol=case.atol, rtol=case.rtol)
    diff = (output.float() - eager.float()).abs()
    print(f"correctness_vs_eager max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f}")

    samples = []
    for sample in range(args.samples):
        latency = bench_us(lambda: compiled(*inputs), args.warmup, args.rep)
        samples.append(latency)
        if not args.result_json:
            print(f"sample={sample + 1} {label}={latency:.3f}us")
    return {"label": label, "samples_us": samples, "median_us": statistics.median(samples)}


def run_comparison(case: BenchmarkCase, args) -> None:
    results = {}
    with tempfile.TemporaryDirectory(prefix="torchtlx_fusion_") as tmp:
        for variant in ("baseline", "candidate"):
            result_path = pathlib.Path(tmp) / f"{variant}.json"
            command = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--case",
                case.name,
                "--n",
                str(args.n),
                "--warmup",
                str(args.warmup),
                "--rep",
                str(args.rep),
                "--samples",
                str(args.samples),
                "--variant",
                variant,
                "--result-json",
                str(result_path),
            ]
            completed = subprocess.run(command)
            if completed.returncode:
                raise SystemExit(f"{variant} process failed with exit code {completed.returncode}")
            results[variant] = json.loads(result_path.read_text())

    baseline = results["baseline"]
    candidate = results["candidate"]
    for sample, (baseline_us, candidate_us) in enumerate(
            zip(baseline["samples_us"], candidate["samples_us"]),
            start=1,
    ):
        print(f"sample={sample} baseline={baseline_us:.3f}us "
              f"{candidate['label']}={candidate_us:.3f}us")
    print(f"FINAL baseline={baseline['median_us']:.3f}us "
          f"{candidate['label']}={candidate['median_us']:.3f}us "
          f"speedup={baseline['median_us'] / candidate['median_us']:.3f}x")


def main() -> None:
    global N
    args = parse_args()
    N = args.n
    if args.list:
        for case in CASES.values():
            print(f"{case.name}: {case.problem()}")
        return
    case = CASES[args.case]
    if args.variant:
        result = run_variant(case, args.variant, args)
        if args.result_json:
            pathlib.Path(args.result_json).write_text(json.dumps(result))
        return
    run_comparison(case, args)


if __name__ == "__main__":
    main()
