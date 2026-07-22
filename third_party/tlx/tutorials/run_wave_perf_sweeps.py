#!/usr/bin/env python3
"""Run the f16, MXFP, GLU, and FA performance sweeps for LLVM and Wave."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_COMPILE_WORKERS = max(1, min(8, os.cpu_count() or 1))
ALL_SWEEPS = ("f16", "mxfp", "glu", "fa")
F16_TIMING_MODES = ("triton", "batched")
MXFP_TIMING_MODES = ("batched", "triton")
DEFAULT_MXFP_WARMUP_LAUNCHES = 25
DEFAULT_MXFP_TIMED_LAUNCHES = 1000
DEFAULT_MXFP_TIMING_REPEATS = 7
F16_V10_BASELINE_SHAPE = "8192x8192x8192"
F16_V11_BASELINE_SHAPE = "8192x8192x8192"
F16_INTER_WAVE_BASELINE_SHAPE = (8192, 8192, 8192)


def _load_f16_inputs():
    path = SCRIPT_DIR / "gfx9_gemm" / "f16_inputs.py"
    spec = importlib.util.spec_from_file_location("_tlx_gfx9_sweep_f16_inputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import f16 input helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F16_INPUT_MODES = _load_f16_inputs().INPUT_MODES


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    backend: str
    command: tuple[str, ...]
    cache_dir: Path


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all TLX LLVM-versus-Wave performance sweeps sequentially.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=ALL_SWEEPS,
        default=list(ALL_SWEEPS),
        help="sweeps to run; default: f16 mxfp glu fa",
    )
    parser.add_argument(
        "--device",
        help="ROCR device index or UUID for child processes; defaults to the current environment",
    )
    parser.add_argument(
        "--compile-workers",
        type=nonnegative_int,
        default=DEFAULT_COMPILE_WORKERS,
        help=("parallel compilation workers within each sweep; 0 disables precompilation; "
              f"default: {DEFAULT_COMPILE_WORKERS}"),
    )
    parser.add_argument(
        "--rep",
        type=positive_int,
        default=500,
        help="timed duration in milliseconds for do_bench-based sweeps and legacy MXFP timing; default: 500",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=50,
        help="warmup duration in milliseconds for do_bench-based sweeps; default: 50",
    )
    parser.add_argument(
        "--f16-input-mode",
        choices=F16_INPUT_MODES,
        default="normal",
        help="f16 input distribution; default: normal",
    )
    parser.add_argument(
        "--f16-input-seed",
        type=nonnegative_int,
        default=0,
        help="deterministic f16 input seed; default: 0",
    )
    parser.add_argument(
        "--f16-timing-mode",
        choices=F16_TIMING_MODES,
        default="triton",
        help="f16 timing methodology; default: triton",
    )
    parser.add_argument(
        "--f16-warmup-launches",
        type=nonnegative_int,
        default=25,
        help="warmups before each f16 batched timing repeat; default: 25",
    )
    parser.add_argument(
        "--f16-timed-launches",
        type=positive_int,
        default=1000,
        help="launches per f16 batched event span; default: 1000",
    )
    parser.add_argument(
        "--f16-timing-repeats",
        type=positive_int,
        default=7,
        help="f16 batched timing samples; default: 7",
    )
    parser.add_argument(
        "--mxfp-timing-mode",
        choices=MXFP_TIMING_MODES,
        default="batched",
        help="MXFP timing methodology; default: batched",
    )
    parser.add_argument(
        "--mxfp-warmup-launches",
        type=nonnegative_int,
        default=DEFAULT_MXFP_WARMUP_LAUNCHES,
        help=("warmups before each MXFP batched timing repeat; "
              f"default: {DEFAULT_MXFP_WARMUP_LAUNCHES}"),
    )
    parser.add_argument(
        "--mxfp-timed-launches",
        type=positive_int,
        default=DEFAULT_MXFP_TIMED_LAUNCHES,
        help=("launches per MXFP batched event span; "
              f"default: {DEFAULT_MXFP_TIMED_LAUNCHES}"),
    )
    parser.add_argument(
        "--mxfp-timing-repeats",
        type=positive_int,
        default=DEFAULT_MXFP_TIMING_REPEATS,
        help=("MXFP batched timing samples; "
              f"default: {DEFAULT_MXFP_TIMING_REPEATS}"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="cache and log root; defaults to a fresh timestamped directory under the system temp directory",
    )
    parser.add_argument(
        "--wave-split-barriers",
        action="store_true",
        help="enable split barriers for Wave runs only",
    )
    parser.add_argument("--wave-opt", type=Path, help="wave-opt binary to use for all Wave compilations")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed sweep")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def default_cache_root():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.gettempdir()) / f"tlx-wave-perf-{timestamp}-{os.getpid()}"


def build_run_specs(args, cache_root):
    python = sys.executable
    timing = ("--rep", str(args.rep), "--warmup", str(args.warmup))
    workers = ("--compile-workers", str(args.compile_workers))
    specs = []

    if "f16" in args.sweeps:
        cache_dir = cache_root / "f16"
        command = (
            python,
            str(SCRIPT_DIR / "gfx9_gemm/a16w16/bench.py"),
            "--version",
            "9",
            "--providers",
            "tlx",
            "wave",
            *timing,
            *workers,
            "--input-mode",
            args.f16_input_mode,
            "--seed",
            str(args.f16_input_seed),
            "--timing-mode",
            args.f16_timing_mode,
            "--warmup-launches",
            str(args.f16_warmup_launches),
            "--timed-launches",
            str(args.f16_timed_launches),
            "--timing-repeats",
            str(args.f16_timing_repeats),
            "--cache-dir",
            str(cache_dir),
        )
        if args.wave_split_barriers:
            command += ("--wave-split-barriers", )
        specs.append(RunSpec("f16", "f16 v9: LLVM vs Wave", "both", command, cache_dir))

        # Keep the Wave-derived eight-wave kernel as a stable 8K baseline while
        # applying the same selected input distribution as the other f16 runs.
        cache_dir = cache_root / "f16-v10"
        command = (
            python,
            str(SCRIPT_DIR / "gfx9_gemm/a16w16/bench.py"),
            "--version",
            "10",
            "--providers",
            "tlx",
            "wave",
            "--shape",
            F16_V10_BASELINE_SHAPE,
            *timing,
            *workers,
            "--input-mode",
            args.f16_input_mode,
            "--seed",
            str(args.f16_input_seed),
            "--timing-mode",
            "batched",
            "--warmup-launches",
            str(args.f16_warmup_launches),
            "--timed-launches",
            str(args.f16_timed_launches),
            "--timing-repeats",
            str(args.f16_timing_repeats),
            "--cache-dir",
            str(cache_dir),
        )
        if args.wave_split_barriers:
            command += ("--wave-split-barriers", )
        specs.append(
            RunSpec(
                "f16-v10",
                "f16 v10 8192x8192x8192 baseline: LLVM vs Wave",
                "both",
                command,
                cache_dir,
            )
        )

        # The four-wave variant is the specialized counterpart of the v10
        # eight-wave baseline.  Keep the shape, selected inputs, timing, and
        # provider pair identical so its result is directly comparable.
        cache_dir = cache_root / "f16-v11"
        command = (
            python,
            str(SCRIPT_DIR / "gfx9_gemm/a16w16/bench.py"),
            "--version",
            "11",
            "--providers",
            "tlx",
            "wave",
            "--shape",
            F16_V11_BASELINE_SHAPE,
            *timing,
            *workers,
            "--input-mode",
            args.f16_input_mode,
            "--seed",
            str(args.f16_input_seed),
            "--timing-mode",
            "batched",
            "--warmup-launches",
            str(args.f16_warmup_launches),
            "--timed-launches",
            str(args.f16_timed_launches),
            "--timing-repeats",
            str(args.f16_timing_repeats),
            "--cache-dir",
            str(cache_dir),
        )
        if args.wave_split_barriers:
            command += ("--wave-split-barriers", )
        specs.append(
            RunSpec(
                "f16-v11",
                "f16 v11 specialized 4-wave 8192x8192x8192: LLVM vs Wave",
                "both",
                command,
                cache_dir,
            )
        )

        script = str(SCRIPT_DIR / "gfx9_gemm/inter_wave/a16w16/bench.py")
        for backend in ("llvm", "wave"):
            cache_dir = cache_root / f"f16-inter-wave-{backend}"
            command = (
                python,
                script,
                "--shape",
                *(str(dim) for dim in F16_INTER_WAVE_BASELINE_SHAPE),
                *timing,
                "--input-mode",
                args.f16_input_mode,
                "--seed",
                str(args.f16_input_seed),
            )
            specs.append(
                RunSpec(
                    f"f16-inter-wave-{backend}",
                    f"f16 inter-wave 8192x8192x8192: {backend.upper()}",
                    backend,
                    command,
                    cache_dir,
                )
            )

    if "mxfp" in args.sweeps:
        script = str(SCRIPT_DIR / "gfx9_gemm/a4w4/bench.py")
        for backend in ("llvm", "wave"):
            cache_dir = cache_root / f"mxfp-{backend}"
            command = (
                python,
                script,
                *timing,
                *workers,
                "--timing-mode",
                args.mxfp_timing_mode,
                "--warmup-launches",
                str(args.mxfp_warmup_launches),
                "--timed-launches",
                str(args.mxfp_timed_launches),
                "--timing-repeats",
                str(args.mxfp_timing_repeats),
                "--cache-dir",
                str(cache_dir),
            )
            if backend == "wave" and args.wave_split_barriers:
                command += ("--wave-split-barriers", )
            specs.append(RunSpec(f"mxfp-{backend}", f"MXFP: {backend.upper()}", backend, command, cache_dir))

    if "glu" in args.sweeps:
        script = str(SCRIPT_DIR / "amd-addmm-glu-opt_test.py")
        for backend in ("llvm", "wave"):
            cache_dir = cache_root / f"glu-{backend}"
            command = (python, script, *timing, *workers)
            specs.append(RunSpec(f"glu-{backend}", f"GLU: {backend.upper()}", backend, command, cache_dir))

    if "fa" in args.sweeps:
        script = str(SCRIPT_DIR / "amd-fa-pipelined_test.py")
        for backend in ("llvm", "wave"):
            cache_dir = cache_root / f"fa-{backend}"
            # Keep the canonical FA matrix, baselines, and tolerance in the FA driver.
            command = (python, script, "--mode", "perf_test", *workers)
            specs.append(RunSpec(f"fa-{backend}", f"FA: {backend.upper()}", backend, command, cache_dir))

    return specs


def child_environment(args, spec):
    env = os.environ.copy()
    if args.device is not None:
        env["ROCR_VISIBLE_DEVICES"] = args.device
        # Avoid applying a second, differently indexed visibility mask.
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)

    env["TRITON_CACHE_DIR"] = str(spec.cache_dir)
    if args.wave_opt is not None:
        env["TRITON_WAVE_OPT"] = str(args.wave_opt.expanduser().resolve())

    if spec.backend == "wave":
        env["TRITON_DEFAULT_BACKEND"] = "tlx_wave"
        env["TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS"] = "1" if args.wave_split_barriers else "0"
    else:
        env.pop("TRITON_DEFAULT_BACKEND", None)
        env.pop("TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS", None)
    return env


def print_run(args, spec):
    settings = [f"TRITON_CACHE_DIR={shlex.quote(str(spec.cache_dir))}"]
    if args.device is not None:
        settings.insert(0, f"ROCR_VISIBLE_DEVICES={shlex.quote(args.device)}")
    if spec.backend == "wave":
        settings.append("TRITON_DEFAULT_BACKEND=tlx_wave")
        settings.append(f"TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS={int(args.wave_split_barriers)}")
    print(f"\n{'=' * 80}\n{spec.label}\n{'=' * 80}", flush=True)
    print(" ".join((*settings, shlex.join(spec.command))), flush=True)


def run_spec(args, spec, log_dir):
    print_run(args, spec)
    if args.dry_run:
        return True

    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{spec.name}.stdout.log"
    stderr_path = log_dir / f"{spec.name}.stderr.log"
    failures = []

    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open("w", encoding="utf-8") as stderr_log:
        process = subprocess.Popen(
            spec.command,
            cwd=REPO_ROOT,
            env=child_environment(args, spec),
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stdout_log.write(line)
                if "-> SKIPPED" in line or "FAIL" in line:
                    failures.append(line.strip())
            returncode = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            process.wait()
            raise

    if returncode != 0:
        failures.append(f"process exited with status {returncode}")
    if failures:
        print(f"FAILED: {spec.label}", file=sys.stderr)
        for failure in failures[:10]:
            print(f"  {failure}", file=sys.stderr)
        print(f"  logs: {stdout_path} and {stderr_path}", file=sys.stderr)
        return False

    print(f"Completed {spec.label}; logs: {stdout_path} and {stderr_path}", flush=True)
    return True


def main():
    args = parse_args()
    cache_root = (args.cache_dir.expanduser().resolve() if args.cache_dir is not None else default_cache_root())
    specs = build_run_specs(args, cache_root)
    print(f"Cache/log root: {cache_root}")
    print(f"Compilation workers: {args.compile_workers}")
    print(f"Wave split barriers: {'enabled' if args.wave_split_barriers else 'disabled'}")
    if "mxfp" in args.sweeps:
        if args.mxfp_timing_mode == "batched":
            print(
                f"MXFP timing: batched median, {args.mxfp_timing_repeats}x"
                f"{args.mxfp_timed_launches} timed launches, "
                f"{args.mxfp_warmup_launches} warmups/repeat"
            )
        else:
            print(f"MXFP timing: triton median, {args.warmup}ms warmup/{args.rep}ms timed")

    if args.dry_run:
        for spec in specs:
            print_run(args, spec)
        print("\nDry run complete.")
        return 0

    failed = []
    for spec in specs:
        if not run_spec(args, spec, cache_root / "logs"):
            failed.append(spec.label)
            if args.fail_fast:
                break

    if failed:
        print("\nFailed sweeps: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("\nAll requested LLVM-versus-Wave performance sweeps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
