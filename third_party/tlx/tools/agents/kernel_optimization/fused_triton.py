from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from . import cli


_HERE = Path(__file__).resolve().parent
_HARNESS = _HERE / "fused_triton_harness.py"

_FUSED_TRITON_GUIDANCE = """The candidate is a generated Inductor output_code_fused.py wrapper.
Preserve call(args), argument order, output structure, shapes, strides, and dtypes. Optimize
the complete wrapper, including avoidable launch boundaries and global intermediates. The
TLX source is optional design evidence and must never be imported by the candidate."""

_FUSED_AUTOWS_GUIDANCE = """
This run enables {implementation} Triton AutoWS only for the fused leg. Annotate the
recurring load/MMA loop with tl.range(..., warp_specialize=True); do not add or modify
process-global environment settings in candidate.py. Treat final-TTGIR proof of
materialized ttg.warp_specialize partitions as mandatory before attributing results to
AutoWS. Do not assume upstream and Meta-WS have equivalent performance; benchmark them
as distinct lowering modes when either could be deployed."""


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("fused_triton_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import PostFuser analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case_payload(case_dir: Path) -> dict[str, object]:
    payload = json.loads((case_dir / "manifest.json").read_text())
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("case manifest must contain exactly one case")
    case = cases[0]
    if not isinstance(case, dict) or not case.get("geo_kernel"):
        raise ValueError("case manifest has no geo_kernel")
    return case


def resolve_registry_reference(case_dir: Path, fbsource_root: Path) -> Path:
    postfuser_root = case_dir.parent.parent
    analyzer_path = postfuser_root / "analyze_geo_unfused_corpus.py"
    analyzer = _load_module(analyzer_path)
    case = _case_payload(case_dir)
    sources = analyzer.load_geo_sources(fbsource_root, case_dir / "manifest.json")
    source = sources.get(str(case["geo_kernel"]))
    if source is None:
        raise ValueError(f"registry source not found for {case['geo_kernel']}")
    if source.implementation is None:
        raise ValueError(
            f"shape-specific registry implementation is {source.implementation_status}"
        )
    return Path(source.implementation)


def _resolve_triton_dir(path: Path | None) -> Path:
    if path is None:
        default = Path.cwd().resolve()
        try:
            answer = input(f"Triton checkout to use [{default}]: ").strip()
        except EOFError as error:
            raise SystemExit(
                "cannot prompt for a Triton checkout; pass --triton-dir PATH"
            ) from error
        path = Path(answer).expanduser() if answer else default
    resolved = path.expanduser().resolve()
    package = resolved / "python/triton/__init__.py"
    native_library = resolved / "python/triton/_C/libtriton.so"
    if not package.is_file():
        raise SystemExit(f"not a Triton checkout (missing {package})")
    if not native_library.is_file():
        raise SystemExit(
            f"Triton is not built (missing {native_library}); run `make` in {resolved}"
        )
    return resolved


def _validate_local_runtime(triton_dir: Path, python_executable: Path) -> None:
    environment = dict(os.environ)
    python_path = str(triton_dir / "python")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path
    probe = subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "from pathlib import Path; import torch, triton; "
                "import triton._C.libtriton; "
                "print(Path(triton.__file__).resolve().parent)"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    expected = (triton_dir / "python/triton").resolve()
    observed = probe.stdout.strip()
    if probe.returncode != 0:
        diagnostics = probe.stderr.strip() or probe.stdout.strip()
        raise SystemExit(
            f"cannot use {python_executable} with Triton checkout {triton_dir}: "
            f"{diagnostics[-2000:]}"
        )
    if Path(observed).resolve() != expected:
        raise SystemExit(
            f"selected {triton_dir}, but {python_executable} imports Triton from "
            f"{observed}"
        )


_SUMMARY_BEGIN = "# BEGIN FUSED TRITON AGENT SUMMARY"
_SUMMARY_END = "# END FUSED TRITON AGENT SUMMARY"


def _median_from_case(case: object) -> float | None:
    if not isinstance(case, dict):
        return None
    timing = case.get("timing")
    if not isinstance(timing, dict):
        return None
    samples = timing.get("samples_us")
    if not isinstance(samples, list) or not samples:
        return None
    try:
        return statistics.median(float(sample) for sample in samples)
    except (TypeError, ValueError):
        return None


def _winner_change(result: dict[str, object]) -> str:
    winner_id = result.get("winner_experiment_id")
    experiments = result.get("experiments")
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            if experiment.get("experiment_id") != winner_id:
                continue
            for key in ("mutation_summary", "hypothesis"):
                value = experiment.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    title = result.get("winner_commit_title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "Retained the correctness-validated agent winner."


def _summary_header(
    result: dict[str, object], *, reference_is_tuned: bool
) -> str:
    baseline = result.get("baseline")
    final = result.get("final")
    baseline_cases = baseline.get("cases") if isinstance(baseline, dict) else None
    final_cases = final.get("cases") if isinstance(final, dict) else None
    baseline_case = (
        baseline_cases[0]
        if isinstance(baseline_cases, list) and baseline_cases
        else None
    )
    final_case = (
        final_cases[0]
        if isinstance(final_cases, list) and final_cases
        else None
    )
    baseline_us = _median_from_case(baseline_case)
    winner_us = _median_from_case(final_case)
    verification = (
        final_case.get("verification") if isinstance(final_case, dict) else None
    )
    metrics = verification.get("metrics") if isinstance(verification, dict) else None
    tlx_us = metrics.get("tlx_median_us") if isinstance(metrics, dict) else None
    try:
        tlx_us = float(tlx_us) if tlx_us is not None else None
    except (TypeError, ValueError):
        tlx_us = None

    lines = [
        _SUMMARY_BEGIN,
        "# Optimized, correctness-validated fused Triton wrapper.",
        "# Applied:",
    ]
    lines.extend(f"#   {line}" for line in _winner_change(result).splitlines())
    lines.append("# Performance (complete-wrapper CUDA-event timing):")
    if baseline_us is not None and winner_us is not None:
        lines.append(
            f"#   generated baseline {baseline_us:.3f} us -> optimized {winner_us:.3f} us "
            f"({baseline_us / winner_us:.3f}x)."
        )
    elif winner_us is not None:
        lines.append(f"#   optimized {winner_us:.3f} us.")
    if tlx_us is not None and winner_us is not None:
        label = "tuned TLX" if reference_is_tuned else "TLX reference"
        lines.append(
            f"#   {label} {tlx_us:.3f} us; optimized/TLX speedup "
            f"{tlx_us / winner_us:.3f}x."
        )
    precision = (
        metrics.get("tlx_precision_comparison")
        if isinstance(metrics, dict)
        else None
    )
    precision_outputs = (
        precision.get("outputs") if isinstance(precision, dict) else None
    )
    if isinstance(precision_outputs, list):
        for output in precision_outputs:
            if not isinstance(output, dict):
                continue
            try:
                lines.append(
                    "#   TLX precision "
                    f"{output['path']}: exact={float(output['exact_fraction']):.6f}, "
                    f"max_abs={float(output['max_abs_error']):.8g}, "
                    f"relative_l2={float(output['relative_l2']):.8g}."
                )
            except (KeyError, TypeError, ValueError):
                continue
    autows_proof = metrics.get("autows_proof") if isinstance(metrics, dict) else None
    if isinstance(autows_proof, dict) and autows_proof.get("requested") is True:
        if autows_proof.get("materialized") is True:
            lines.append(
                "#   AutoWS proof: final TTGIR contains physical warp-specialized "
                "partitions."
            )
        else:
            lines.append(
                "#   AutoWS proof: annotation requested but no physical final-TTGIR "
                "partitions materialized; do not attribute performance to AutoWS."
            )
    lines.extend(
        (
            "# Timings use the run's configured warmup, sampling, and cache policy.",
            _SUMMARY_END,
            "",
        )
    )
    return "\n".join(lines)


def _strip_summary(source: str) -> str:
    if not source.startswith(_SUMMARY_BEGIN):
        return source
    end = source.find(_SUMMARY_END)
    if end < 0:
        return source
    return source[end + len(_SUMMARY_END) :].lstrip("\r\n")


def _persist_completed_kernel(
    output_dir: Path,
    destination: Path,
    *,
    reference_is_tuned: bool = False,
) -> None:
    """Atomically install a summarized, correctness-validated session winner."""
    result_path = output_dir / "result.json"
    best_path = output_dir / "best_kernel.py"
    if not result_path.is_file() or not best_path.is_file():
        raise RuntimeError(
            f"agent session did not produce result.json and best_kernel.py in {output_dir}"
        )

    result = json.loads(result_path.read_text())
    final = result.get("final")
    cases = final.get("cases") if isinstance(final, dict) else None
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("agent result has no final correctness evaluation")
    if any(
        not isinstance(case, dict)
        or not isinstance(case.get("verification"), dict)
        or case["verification"].get("passed") is not True
        for case in cases
    ):
        raise RuntimeError(
            "refusing to persist a kernel that failed final verification"
        )

    best_source = best_path.read_text()
    if result.get("best_kernel") != best_source:
        raise RuntimeError("best_kernel.py does not match the completed agent result")
    persisted_source = _summary_header(
        result, reference_is_tuned=reference_is_tuned
    ) + _strip_summary(best_source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(persisted_source)
            temporary_path = Path(temporary.name)
        if destination.exists():
            temporary_path.chmod(stat.S_IMODE(destination.stat().st_mode))
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune one generated PostFuser case with the kernel optimization agent."
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--kernel",
        type=Path,
        default=None,
        help=(
            "Candidate wrapper to optimize; defaults to CASE_DIR/output_code_fused.py. "
            "Pass a prior run's best_kernel.py to continue hill climbing."
        ),
    )
    parser.add_argument("--fbsource-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--triton-dir",
        type=Path,
        default=None,
        help="Triton checkout/build to use; prompts at session start when omitted.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help=(
            "Python with torch installed; defaults to TRITON_DIR/.venv/bin/python "
            "when present, otherwise the launcher's Python."
        ),
    )
    parser.add_argument("--reference-kernel", type=Path, default=None)
    parser.add_argument(
        "--reference-config-json",
        type=Path,
        default=None,
        help=(
            "Optional fixed configuration passed to the registry reference kernel; "
            "use this when its default heuristic is invalid or would cold-autotune."
        ),
    )
    parser.add_argument(
        "--reference-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Environment override applied only to the isolated reference process; "
            "repeat for multiple values."
        ),
    )
    parser.add_argument(
        "--registry-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Provide the shape-specific TLX registry source to the candidate agent "
            "when it can be resolved (default: enabled)."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--benchmark-warmup-ms", type=int, default=500)
    parser.add_argument("--benchmark-duration-ms", type=int, default=2000)
    parser.add_argument(
        "--autows",
        action="store_true",
        help=(
            "Enable Meta-Triton autoWS for the fused leg and benchmark the TLX "
            "reference in a separate process. The candidate must annotate its loop."
        ),
    )
    parser.add_argument(
        "--autows-implementation",
        choices=("meta", "upstream"),
        default="meta",
        help=(
            "AutoWS lowering selected by --autows (default: meta). Use upstream "
            "to exercise Triton's non-Meta warp-specialization pass."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--max-candidate-seconds", type=float, default=900.0)
    parser.add_argument("--max-total-seconds", type=float, default=7200.0)
    parser.add_argument("--min-speedup", type=float, default=1.01)
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prior-run", type=Path, default=None)
    parser.add_argument("--provider", choices=("codex", "mock"), default="codex")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    triton_dir = _resolve_triton_dir(args.triton_dir)
    default_python = triton_dir / ".venv/bin/python"
    python_executable = (
        args.python_executable.expanduser().resolve()
        if args.python_executable is not None
        else (default_python if default_python.is_file() else Path(sys.executable))
    )
    if not python_executable.is_file():
        raise SystemExit(f"Python executable does not exist: {python_executable}")
    _validate_local_runtime(triton_dir, python_executable)
    print(f"Using Triton checkout: {triton_dir}", file=sys.stderr)
    print(f"Using Python runtime: {python_executable}", file=sys.stderr)
    case_dir = args.case_dir.resolve()
    fbsource_root = args.fbsource_root.resolve()
    output_dir = args.output_dir.resolve()
    kernel = (
        args.kernel.resolve()
        if args.kernel is not None
        else case_dir / "output_code_fused.py"
    )
    for filename in (
        "output_code.py",
        "output_code_fused.py",
        "manifest.json",
        "bench_vs_geo.py",
    ):
        if not (case_dir / filename).is_file():
            raise SystemExit(f"case directory is missing {filename}: {case_dir}")
    if not kernel.is_file():
        raise SystemExit(f"candidate kernel does not exist: {kernel}")
    reference = args.reference_kernel.resolve() if args.reference_kernel else None
    if reference is None and args.registry_reference:
        try:
            reference = resolve_registry_reference(case_dir, fbsource_root)
        except (OSError, RuntimeError, ValueError) as error:
            print(
                f"warning: TLX registry reference unavailable ({error}); "
                "continuing with benchmark and profile evidence",
                file=sys.stderr,
            )
    if reference is not None and not reference.is_file():
        raise SystemExit(f"TLX reference kernel does not exist: {reference}")
    reference_config = (
        args.reference_config_json.expanduser().resolve()
        if args.reference_config_json is not None
        else None
    )
    if reference_config is not None:
        if not reference_config.is_file():
            raise SystemExit(
                f"reference configuration does not exist: {reference_config}"
            )
        try:
            reference_config_payload = json.loads(reference_config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot read reference configuration {reference_config}: {error}"
            ) from error
        if not isinstance(reference_config_payload, dict):
            raise SystemExit("reference configuration must be a JSON object")
    reference_environment: dict[str, str] = {}
    for assignment in args.reference_env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise SystemExit(
                f"invalid --reference-env {assignment!r}; expected NAME=VALUE"
            )
        reference_environment[name] = value

    if (output_dir / "result.json").exists() or (output_dir / "experiments").exists():
        raise SystemExit(
            f"output directory already contains an agent run: {output_dir}; "
            "use --prior-run with a fresh --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "config"
    config_dir.mkdir(exist_ok=True)
    case = _case_payload(case_dir)
    cases_path = config_dir / "cases.json"
    target_path = config_dir / "target.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": str(case["geo_kernel"]),
                    "parameters": {
                        "warmup_ms": args.benchmark_warmup_ms,
                        "benchmark_ms": args.benchmark_duration_ms,
                    },
                    "weight": 1.0,
                    "protected": True,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    target_environment = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "FUSED_TRITON_CASE_DIR": str(case_dir),
        "FUSED_TRITON_FBSOURCE_ROOT": str(fbsource_root),
        "FUSED_TRITON_DIR": str(triton_dir),
        "FUSED_TRITON_PYTHON": str(python_executable),
        # Shape-specific TLX references expose a heuristic path that
        # bypasses their otherwise very large cold autotune search.
        "TLX_GEMM_USE_HEURISTIC": "1",
    }
    if reference_config is not None:
        target_environment["FUSED_TRITON_REFERENCE_CONFIG"] = str(reference_config)
    if reference_environment:
        target_environment["FUSED_TRITON_REFERENCE_ENV"] = json.dumps(
            reference_environment, sort_keys=True
        )
    if args.autows:
        target_environment.update(
            {
                "FUSED_TRITON_AUTOWS": "1",
                "FUSED_TRITON_AUTOWS_IMPLEMENTATION": args.autows_implementation,
            }
        )
        if args.autows_implementation == "meta":
            target_environment.update(
                {
                    "TRITON_USE_META_WS": "1",
                    "TRITON_DISABLE_WSBARRIER_REORDER": "1",
                }
            )
    target_path.write_text(
        json.dumps(
            {
                "backend": "cuda",
                "architecture": "GB200",
                "device": "cuda:0",
                "environment": target_environment,
                "optimization_guidance": _FUSED_TRITON_GUIDANCE
                + (
                    _FUSED_AUTOWS_GUIDANCE.format(
                        implementation=args.autows_implementation
                    )
                    if args.autows
                    else ""
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    cli_arguments = [
        "--kernel",
        str(kernel),
        "--harness",
        str(_HARNESS),
        "--cases",
        str(cases_path),
        "--target",
        str(target_path),
        "--output-dir",
        str(output_dir),
        "--skip-host-validation",
        "--no-commit-winner",
        "--provider",
        args.provider,
        "--max-rounds",
        str(args.max_rounds),
        "--candidates-per-round",
        str(args.candidates_per_round),
        "--max-candidate-seconds",
        str(args.max_candidate_seconds),
        "--max-total-seconds",
        str(args.max_total_seconds),
        "--min-speedup",
        str(args.min_speedup),
        "--max-cv",
        str(args.max_cv),
        "--benchmark-repetitions",
        "1",
    ]
    if reference is not None:
        cli_arguments.extend(("--reference-kernel", str(reference)))
    if args.model:
        cli_arguments.extend(("--model", args.model))
    if args.prior_run:
        cli_arguments.extend(("--prior-run", str(args.prior_run.resolve())))
    exit_code = cli.main(cli_arguments)
    # Keep the complete audit trail in output_dir and save the final revalidated
    # winner beside, without replacing, the generated fused baseline.
    if exit_code in (0, 2):
        destination = case_dir / "output_code_fused_opt.py"
        _persist_completed_kernel(
            output_dir,
            destination,
            reference_is_tuned=reference_config is not None,
        )
        print(f"Saved validated optimized fused kernel: {destination}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
