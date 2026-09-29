# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Run TorchTLX fusion benchmarks."""

import argparse
import importlib
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def load_runtime(case) -> None:
    # Keep case discovery and --list usable without loading GPU extension
    # modules. This also lets users inspect available cases before activating
    # the PyTorch environment used to run them.
    global torch, triton, run_and_get_code

    import torch as torch_module
    import triton as triton_module  # @manual=//triton:triton
    from torch._inductor.utils import run_and_get_code as torch_run_and_get_code

    import triton.language.extra.tlx.inductor.registry  # noqa: F401

    torch = torch_module
    triton = triton_module
    run_and_get_code = torch_run_and_get_code
    case.torch = torch_module


def discover_cases() -> dict[str, object]:
    cases = {}
    for path in sorted(_HERE.glob("*_[0-9][0-9]_*.py")):
        case = importlib.import_module(f"torchtlx_benchmark.{path.stem}")
        if case.NAME != path.stem:
            raise ValueError(f"case name {case.NAME!r} does not match {path.name}")
        if case.NAME in cases:
            raise ValueError(f"duplicate case name: {case.NAME}")
        cases[case.NAME] = case
    return cases


def parse_args(cases):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default=next(iter(cases)))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)

    selected, _ = parser.parse_known_args()
    cases[selected.case].add_arguments(parser)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmup <= 0 or args.rep <= 0:
        parser.error("--warmup and --rep must be positive")
    return args


def validate_candidate_code(case, source_files) -> None:
    generated_code = "\n".join(source_files)
    markers = getattr(case, "CANDIDATE_CODE_MARKERS", ())
    marker_groups = getattr(case, "CANDIDATE_CODE_MARKER_GROUPS", ())
    missing = [marker for marker in markers if marker not in generated_code]
    if missing:
        raise RuntimeError("candidate implementation was not generated; missing code markers: " + ", ".join(missing))
    missing_groups = [group for group in marker_groups if not any(marker in generated_code for marker in group)]
    if missing_groups:
        formatted = [" or ".join(group) for group in missing_groups]
        raise RuntimeError("candidate implementation was not generated; missing one of: " + ", ".join(formatted))


def compile_variant(case, config, inputs, variant):
    torch._dynamo.reset()
    with torch._inductor.config.patch(config):
        compiled = torch.compile(case.model, fullgraph=True)
        has_code_markers = any(
            getattr(case, name, ()) for name in (
                "CANDIDATE_CODE_MARKERS",
                "CANDIDATE_CODE_MARKER_GROUPS",
            ))
        if variant == "candidate" and has_code_markers:
            output, source_files = run_and_get_code(compiled, *inputs)
            validate_candidate_code(case, source_files)
        else:
            output = compiled(*inputs)
    torch.cuda.synchronize()
    return compiled, output


def bench_us(fn, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )) * 1000.0


def run_variant(case, variant: str, args) -> dict[str, object]:
    config = case.BASELINE_CONFIG if variant == "baseline" else case.CANDIDATE_CONFIG
    label = "baseline" if variant == "baseline" else case.CANDIDATE_NAME

    torch.compiler.config.force_disable_caches = True
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False

    print(f"variant={label}; torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"case={case.NAME}; {case.problem()}")

    inputs = case.make_inputs()
    eager = case.model(*inputs)
    compiled, output = compile_variant(case, config, inputs, variant)
    torch.testing.assert_close(output, eager, atol=case.ATOL, rtol=case.RTOL)
    diff = (output.float() - eager.float()).abs()
    print(f"correctness_vs_eager max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f}")

    samples = []
    for sample in range(args.samples):
        latency = bench_us(lambda: compiled(*inputs), args.warmup, args.rep)
        samples.append(latency)
        if not args.result_json:
            print(f"sample={sample + 1} {label}={latency:.3f}us")
    return {"label": label, "samples_us": samples, "median_us": statistics.median(samples)}


def run_comparison(case) -> None:
    results = {}
    with tempfile.TemporaryDirectory(prefix="torchtlx_fusion_") as tmp:
        for variant in ("baseline", "candidate"):
            result_path = pathlib.Path(tmp) / f"{variant}.json"
            command = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                *sys.argv[1:],
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
    cases = discover_cases()
    args = parse_args(cases)
    case = cases[args.case]
    case.configure(args)
    if args.list:
        for listed_case in cases.values():
            print(f"{listed_case.NAME}: {listed_case.problem()}")
        return
    load_runtime(case)
    if args.variant:
        result = run_variant(case, args.variant, args)
        if args.result_json:
            pathlib.Path(args.result_json).write_text(json.dumps(result))
        return
    run_comparison(case)


if __name__ == "__main__":
    main()
