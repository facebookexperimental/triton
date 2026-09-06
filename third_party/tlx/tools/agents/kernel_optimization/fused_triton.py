from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from . import cli


_HERE = Path(__file__).resolve().parent
_HARNESS = _HERE / "fused_triton_harness.py"

_FUSED_TRITON_GUIDANCE = """The candidate is a generated Inductor output_code_fused.py wrapper.
Preserve call(args), argument order, output structure, shapes, strides, and dtypes. Optimize
the complete wrapper, including avoidable launch boundaries and global intermediates. The
TLX source is optional design evidence and must never be imported by the candidate."""


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
        "--registry-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Provide the shape-specific TLX registry source to the candidate agent "
            "when it can be resolved (default: enabled)."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--benchmark-warmup-ms", type=int, default=100)
    parser.add_argument("--benchmark-duration-ms", type=int, default=500)
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
    target_path.write_text(
        json.dumps(
            {
                "backend": "cuda",
                "architecture": "GB200",
                "device": "cuda:0",
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
                    "FUSED_TRITON_CASE_DIR": str(case_dir),
                    "FUSED_TRITON_FBSOURCE_ROOT": str(fbsource_root),
                    "FUSED_TRITON_DIR": str(triton_dir),
                    "FUSED_TRITON_PYTHON": str(python_executable),
                },
                "optimization_guidance": _FUSED_TRITON_GUIDANCE,
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
    return cli.main(cli_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
