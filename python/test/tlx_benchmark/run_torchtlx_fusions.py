"""Measure TorchInductor graphs before and after TLX fusion."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inductor_fusions import cases_for_arch, catalog  # noqa: E402
from inductor_fusions.common import FusionCase  # noqa: E402
from inductor_fusions.compat import install_torchtlx  # noqa: E402
from inductor_fusions.measure import (  # noqa: E402
    MAX_PAIRED_SPEEDUP_SPREAD, PairedSample, PairedSummary, summarize_pairs,
)

COMPARISONS = ("forced", "autotuned")
_WORKER_RESULT_FD = "TORCHTLX_FUSION_RESULT_FD"
_WORKER_TIMEOUT_S = 900
_MAX_IDLE_MEMORY_MIB = 1024
_AUTO_ATTEMPTS = 2


@contextmanager
def _observe_multi_kernel_choice():
    """Capture initial MultiKernel selection without affecting timed calls."""
    from torch._inductor.codegen.multi_kernel import MultiKernelCall

    choices = []
    original_run = MultiKernelCall.run

    def observed_run(worker, *args, **kwargs):
        result = original_run(worker, *args, **kwargs)
        if not choices and worker.picked_kernel is not None:
            kernel = worker.kernels[worker.picked_kernel]
            metadata = kernel.inductor_meta
            choices.append({
                "index": worker.picked_kernel,
                "kernel": metadata.get("kernel_name"),
                "candidate": "tlx_local_buffer_retention" in metadata,
            })
        return result

    MultiKernelCall.run = observed_run
    try:
        yield choices
    finally:
        MultiKernelCall.run = original_run


def _compile(
    torch: Any,
    case: FusionCase,
    inputs: tuple[Any, ...],
    comparison: str,
    side: str,
    capture_code: bool,
) -> tuple[Callable[[], Any], Any, str]:
    from torch._inductor import config

    mode, variant_overrides = case.variant(comparison, side)
    torch._dynamo.reset()
    overrides = {"triton.tlx_mode": mode}
    overrides.update(case.config_overrides)
    overrides.update(variant_overrides)
    with config.patch(overrides):
        compiled = torch.compile(case.model, fullgraph=True)
        if capture_code:
            from torch._inductor.utils import run_and_get_code

            output, code = run_and_get_code(compiled, *inputs)
        else:
            output = compiled(*inputs)
            code = ()
    torch.cuda.synchronize()
    return lambda: compiled(*inputs), output, "\n".join(code)


def _find_case(arch: str, name: str) -> FusionCase:
    matches = [case for case in cases_for_arch(arch) if case.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} case for {arch}, found {len(matches)}")
    return matches[0]


def _worker_main(args: argparse.Namespace) -> int:
    result_fd = int(os.environ[_WORKER_RESULT_FD])
    with os.fdopen(result_fd, "w", buffering=1) as result_stream:

        def send(message: dict[str, object]) -> None:
            result_stream.write(json.dumps(message, sort_keys=True) + "\n")

        try:
            case = _find_case(args.worker_arch, args.cases[0])
            torch, triton = install_torchtlx(require_custom_op_autotuning=case.requires_custom_op_autotuning)
            torch.compiler.config.force_disable_caches = True
            torch._inductor.config.force_disable_caches = True
            torch._inductor.config.fx_graph_cache = False
            torch._inductor.config.fx_graph_remote_cache = False
            inputs = case.make_inputs()
            observe_choice = (args.worker_comparison == "autotuned" and args.worker_side == "after"
                              and not args.worker_validate)
            observation = _observe_multi_kernel_choice() if observe_choice else nullcontext([])
            with observation as multi_kernel_choices:
                expected = case.model(*inputs)
                fn, output, code = _compile(
                    torch,
                    case,
                    inputs,
                    args.worker_comparison,
                    args.worker_side,
                    args.worker_validate,
                )
                torch.testing.assert_close(output, expected, atol=case.atol, rtol=case.rtol)

            if args.worker_validate and args.worker_side == "before":
                case.code.validate_before(code)
                generated = "baseline"
            elif args.worker_validate and args.worker_comparison == "forced":
                case.code.validate_after(code)
                generated = "candidate"
            elif args.worker_validate:
                generated = case.code.classify_autotuned(code)
            elif args.worker_side == "before":
                generated = "baseline"
            elif args.worker_comparison == "forced":
                generated = "candidate"
            elif multi_kernel_choices:
                generated = ("candidate_selected" if multi_kernel_choices[-1]["candidate"] else "fallback_selected")
            else:
                raise RuntimeError("autotuned MultiKernel did not expose its selected subkernel")

            latencies = [
                float(triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)) * 1000.0
                for _ in range(args.worker_samples)
            ]
            properties = torch.cuda.get_device_properties(0)
            send({
                "status": "complete",
                "generated": generated,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "device": properties.name,
                "multi_kernel_choice": multi_kernel_choices[-1] if multi_kernel_choices else None,
                "latencies_us": latencies,
            })
            return 0
        except BaseException as error:
            send({
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            return 1


def _run_worker_batch(
    case: FusionCase,
    *,
    arch: str,
    comparison: str,
    side: str,
    warmup: int,
    rep: int,
    samples: int,
    validate: bool = False,
) -> dict[str, object]:
    result_read_fd, result_write_fd = os.pipe()
    env = os.environ.copy()
    env[_WORKER_RESULT_FD] = str(result_write_fd)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--worker-arch",
        arch,
        "--worker-comparison",
        comparison,
        "--worker-side",
        side,
        "--worker-samples",
        str(samples),
        "--case",
        case.name,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
    ]
    if validate:
        command.append("--worker-validate")
    process = subprocess.Popen(command, env=env, text=True, pass_fds=(result_write_fd, ))
    os.close(result_write_fd)
    with os.fdopen(result_read_fd) as results:
        readable, _, _ = select.select([results], [], [], _WORKER_TIMEOUT_S)
        if not readable:
            process.kill()
            process.wait()
            raise TimeoutError("fusion benchmark worker timed out")
        line = results.readline()
    try:
        returncode = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError("fusion benchmark worker did not exit")
    if not line:
        raise RuntimeError(f"fusion benchmark {comparison}/{side} worker exited unexpectedly ({returncode})")
    message = json.loads(line)
    if message.get("status") == "error":
        raise RuntimeError(f"{message['error']}\n{message['traceback']}")
    if message.get("status") != "complete" or returncode != 0:
        raise RuntimeError(f"fusion benchmark {comparison}/{side} worker returned "
                           f"{message.get('status')!r} ({returncode})")
    return message


def _assert_device_idle(device_index: int) -> None:
    from _harness.denoise import list_devices

    current = next((device for device in list_devices() if device.index == device_index), None)
    if current is None:
        raise RuntimeError(f"GPU {device_index} disappeared while preparing a measurement block")
    if current.memory_used_mib > _MAX_IDLE_MEMORY_MIB:
        raise RuntimeError(f"GPU {device_index} is no longer idle: {current.memory_used_mib:.0f} MiB allocated; "
                           "discarding the run")


def _pair_abba_blocks(
    before_ab: Sequence[float],
    after_ab: Sequence[float],
    after_ba: Sequence[float],
    before_ba: Sequence[float],
) -> tuple[PairedSample, ...]:
    lengths = {len(before_ab), len(after_ab), len(after_ba), len(before_ba)}
    if len(lengths) != 1:
        raise ValueError(f"ABBA worker batches have different lengths: {sorted(lengths)}")
    paired = []
    for before, after, reverse_before, reverse_after in zip(before_ab, after_ab, before_ba, after_ba):
        paired.append(PairedSample(float(before), float(after), "AB"))
        paired.append(PairedSample(float(reverse_before), float(reverse_after), "BA"))
    return tuple(paired)


def _pair_reference_blocks(
    before: Sequence[float],
    after: Sequence[float],
) -> tuple[PairedSample, ...]:
    if len(before) != len(after):
        raise ValueError(f"reference worker batches have different lengths: {len(before)} and {len(after)}")
    return tuple(PairedSample(float(a), float(b), "AB") for a, b in zip(before, after))


def _run_reference_comparison(
    case: FusionCase,
    *,
    arch: str,
    warmup: int,
    rep: int,
    samples: int,
    max_spread: float,
    device_index: int,
) -> tuple[PairedSummary, dict[str, object]]:
    common = {
        "case": case,
        "arch": arch,
        "comparison": "forced",
        "warmup": warmup,
        "rep": rep,
        "samples": samples,
    }
    _assert_device_idle(device_index)
    before = _run_worker_batch(side="before", **common)
    _assert_device_idle(device_index)
    after = _run_worker_batch(side="after", **common)
    # Code capture perturbs AMD timing substantially, so validate the candidate
    # in one disposable compile-only process after both timing workers exit.
    _run_worker_batch(side="after", validate=True, **{**common, "samples": 0})
    paired = _pair_reference_blocks(before["latencies_us"], after["latencies_us"])
    metadata = {
        "before_generated": before["generated"],
        "after_generated": after["generated"],
        "torch": after["torch"],
        "triton": after["triton"],
        "device": after["device"],
        "multi_kernel_choices": [before["multi_kernel_choice"], after["multi_kernel_choice"]],
    }
    return summarize_pairs(paired, max_spread=max_spread), metadata


def _run_autotuned_selection(
    case: FusionCase,
    *,
    arch: str,
    warmup: int,
    rep: int,
    device_index: int,
) -> dict[str, object]:
    _assert_device_idle(device_index)
    result = _run_worker_batch(
        case,
        arch=arch,
        comparison="autotuned",
        side="after",
        warmup=warmup,
        rep=rep,
        samples=0,
    )
    if result["multi_kernel_choice"] is None:
        raise RuntimeError("production autotuning did not report a selected subkernel")
    return result


def _run_rigorous_comparison(
    case: FusionCase,
    *,
    arch: str,
    comparison: str,
    warmup: int,
    rep: int,
    samples: int,
    max_spread: float,
    device_index: int,
) -> tuple[PairedSummary, dict[str, object]]:
    batch_samples = samples // 2
    common = {
        "case": case,
        "arch": arch,
        "comparison": comparison,
        "warmup": warmup,
        "rep": rep,
        "samples": batch_samples,
    }

    def run(side: str, *, validate: bool = False, samples: int = batch_samples):
        _assert_device_idle(device_index)
        return _run_worker_batch(side=side, validate=validate, **{**common, "samples": samples})

    before_ab = run("before")
    after_ab = run("after")
    after_ba = run("after")
    before_ba = run("before")
    # Validate generated code only after timing. This keeps run_and_get_code
    # instrumentation out of every measured process and avoids unnecessary
    # ROCm context churn before the four ABBA timing blocks.
    run("before", validate=True, samples=0)
    run("after", validate=True, samples=0)
    paired = _pair_abba_blocks(
        before_ab["latencies_us"],
        after_ab["latencies_us"],
        after_ba["latencies_us"],
        before_ba["latencies_us"],
    )
    generated = {str(after_ab["generated"]), str(after_ba["generated"])}
    metadata = {
        "before_generated": before_ab["generated"],
        "after_generated": next(iter(generated)) if len(generated) == 1 else "mixed",
        "torch": after_ab["torch"],
        "triton": after_ab["triton"],
        "device": after_ab["device"],
        "multi_kernel_choices": [after_ab["multi_kernel_choice"], after_ba["multi_kernel_choice"]],
    }
    return summarize_pairs(paired, max_spread=max_spread), metadata


def _render(
    case: FusionCase,
    comparison: str,
    summary: PairedSummary,
    metadata: dict[str, object],
) -> str:
    lines = [f"\n{case.name} [{comparison}]: {case.problem}"]
    for index, sample in enumerate(summary.samples, start=1):
        lines.append(f"  sample={index} order={sample.order} "
                     f"before={sample.before_us:.3f}us after={sample.after_us:.3f}us "
                     f"speedup={sample.speedup:.3f}x")
    status = "noisy" if summary.noisy else "stable"
    lines.append(f"  FINAL before={summary.before_us:.3f}us after={summary.after_us:.3f}us "
                 f"speedup={summary.speedup:.3f}x paired_spread={summary.speedup_spread:.1%} "
                 f"status={status} generated={metadata['after_generated']}")
    return "\n".join(lines)


def _render_selection(case: FusionCase, metadata: dict[str, object]) -> str:
    choice = metadata["multi_kernel_choice"]
    assert isinstance(choice, dict)
    return (f"\n{case.name} [autotuned]: {case.problem}\n"
            f"  FINAL generated={metadata['generated']} kernel={choice['kernel']} index={choice['index']}")


def _list_cases() -> None:
    for arch, cases in catalog().items():
        print(f"{arch}:")
        if not cases:
            print("  (none)")
        for case in cases:
            print(f"  {case.name}: {case.problem}")


def _select_cases(arch: str, names: Sequence[str] | None) -> list[FusionCase]:
    available = list(cases_for_arch(arch))
    if not names:
        return available
    wanted = set(names)
    selected = [case for case in available if case.name in wanted]
    missing = wanted - {case.name for case in selected}
    if missing:
        raise ValueError(f"case(s) not available for {arch}: {', '.join(sorted(missing))}")
    return selected


def _select_run_device(spec: str, excluded: set[int]):
    from _harness.denoise import list_devices, select_device

    if spec != "auto":
        device = select_device(spec)
        if device is not None and device.memory_used_mib > _MAX_IDLE_MEMORY_MIB:
            raise RuntimeError(f"GPU {device.index} has {device.memory_used_mib:.0f} MiB allocated; "
                               "wait for an idle GPU")
        return device
    available = [
        device for device in list_devices()
        if device.index not in excluded and device.memory_used_mib <= _MAX_IDLE_MEMORY_MIB
    ]
    if not available:
        return None
    least_memory = min(device.memory_used_mib for device in available)
    # Idle cards differ by a few KiB of driver bookkeeping. Prefer the highest
    # equivalent index because unrelated jobs most often claim GPU 0.
    equivalent = [device for device in available if device.memory_used_mib <= least_memory + 64]
    return max(equivalent, key=lambda device: device.index)


def _write_json(
    path: str,
    *,
    arch: str,
    device: Any,
    governor: Any,
    results: list[tuple[FusionCase, str, PairedSummary, dict[str, object]]],
    selections: list[tuple[FusionCase, dict[str, object]]],
    measurement: dict[str, object],
) -> None:
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = results[0][3] if results else selections[0][1] if selections else {}
    document = {
        "schema_version":
        3,
        "environment": {
            "arch": arch,
            "device": runtime.get("device", device.name),
            "device_index": device.index,
            "torch": runtime.get("torch"),
            "triton": runtime.get("triton"),
            "governor": governor.to_dict(),
        },
        "measurement":
        measurement,
        "results": [{
            "name": case.name,
            "problem": case.problem,
            "comparison": comparison,
            "generated": metadata["after_generated"],
            "multi_kernel_choices": metadata["multi_kernel_choices"],
            **summary.to_dict(),
        } for case, comparison, summary, metadata in results] +
        [{
            "name": case.name,
            "problem": case.problem,
            "comparison": "autotuned",
            "selection_only": True,
            "generated": metadata["generated"],
            "multi_kernel_choices": [metadata["multi_kernel_choice"]],
        } for case, metadata in selections],
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases", help="case name; repeat to select multiple")
    parser.add_argument("--list", action="store_true", help="list the catalog without loading the GPU runtime")
    parser.add_argument("--device", default="auto", help="physical GPU index, or auto for the least-used GPU")
    parser.add_argument("--warmup", type=int, default=100, help="warmup duration per timing block in milliseconds")
    parser.add_argument("--rep", type=int, default=500, help="measurement duration per timing block in milliseconds")
    parser.add_argument("--samples", type=int, default=5, help="number of timing samples per variant")
    parser.add_argument(
        "--selection",
        choices=(*COMPARISONS, "both"),
        default="forced",
        help="forced candidate (default), production autotuning, or both",
    )
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="use four-process ABBA timing plus separate code-validation workers",
    )
    parser.add_argument(
        "--max-paired-spread",
        type=float,
        default=MAX_PAIRED_SPEEDUP_SPREAD,
        help="maximum relative interdecile range of paired speedups",
    )
    parser.add_argument("--json", help="optional machine-readable result path")
    parser.add_argument(
        "--allow-noisy",
        action="store_true",
        help="allow a rigorous run to succeed when paired speedups exceed the noise limit",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-arch", help=argparse.SUPPRESS)
    parser.add_argument("--worker-comparison", choices=COMPARISONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-side", choices=("before", "after"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-samples", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-validate", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args)
    if args.list:
        _list_cases()
        return 0
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.rigorous and (args.samples < 4 or args.samples % 2):
        parser.error("--rigorous requires an even --samples value of at least 4")
    if args.warmup <= 0 or args.rep <= 0:
        parser.error("--warmup and --rep must be positive")
    if args.max_paired_spread < 0:
        parser.error("--max-paired-spread must be nonnegative")

    from _harness.denoise import AMD, Governor

    comparisons = COMPARISONS if args.selection == "both" else (args.selection, )
    excluded: set[int] = set()
    max_attempts = _AUTO_ATTEMPTS if args.device == "auto" else 1
    for attempt in range(max_attempts):
        device = _select_run_device(args.device, excluded)
        if device is None:
            raise RuntimeError("no idle supported GPU found")
        os.environ[device.visibility_env] = str(device.index)
        arch = device.arch
        if arch is None:
            raise RuntimeError(f"cannot map {device.name} to a fusion catalog architecture")
        try:
            selected = _select_cases(arch, args.cases)
        except ValueError as error:
            parser.error(str(error))

        with Governor(device, govern_device=device.vendor != AMD) as governor:
            print(f"selected=gpu{device.index} {device.name} ({device.memory_used_mib:.0f} MiB in use)")
            for step in governor.applied:
                print(f"denoise={step}")
            for step in governor.skipped:
                print(f"denoise=SKIPPED {step}")

            results = []
            selections = []
            failures = []
            for case in selected:
                for comparison in comparisons:
                    try:
                        if args.rigorous:
                            summary, metadata = _run_rigorous_comparison(
                                case,
                                arch=arch,
                                comparison=comparison,
                                warmup=args.warmup,
                                rep=args.rep,
                                samples=args.samples,
                                max_spread=args.max_paired_spread,
                                device_index=device.index,
                            )
                            results.append((case, comparison, summary, metadata))
                            print(_render(case, comparison, summary, metadata))
                        elif comparison == "forced":
                            summary, metadata = _run_reference_comparison(
                                case,
                                arch=arch,
                                warmup=args.warmup,
                                rep=args.rep,
                                samples=args.samples,
                                max_spread=args.max_paired_spread,
                                device_index=device.index,
                            )
                            results.append((case, comparison, summary, metadata))
                            print(_render(case, comparison, summary, metadata))
                        else:
                            metadata = _run_autotuned_selection(
                                case,
                                arch=arch,
                                warmup=args.warmup,
                                rep=args.rep,
                                device_index=device.index,
                            )
                            selections.append((case, metadata))
                            print(_render_selection(case, metadata))
                    except Exception as error:
                        failures.append((case, comparison, error))
                        print(
                            f"\n{case.name} [{comparison}]: ERROR {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )

        noisy = any(summary.noisy for _, _, summary, _ in results)
        retryable_failure = any("no longer idle" in str(error) or "exited unexpectedly (-11)" in str(error)
                                for _, _, error in failures)
        retry_noisy = args.rigorous and noisy and not args.allow_noisy
        if attempt + 1 < max_attempts and (retryable_failure or retry_noisy):
            excluded.add(device.index)
            reason = "contention" if retryable_failure else "noisy timing"
            print(f"retry={reason}; discarding gpu{device.index} results")
            continue

        if args.json:
            _write_json(
                args.json,
                arch=arch,
                device=device,
                governor=governor,
                results=results,
                selections=selections,
                measurement={
                    "mode":
                    "rigorous" if args.rigorous else "fast",
                    "isolation": ("non-overlapping worker process per ABBA block"
                                  if args.rigorous else "one worker process per variant"),
                    "order": ("A^(samples/2) B^(samples/2) B^(samples/2) A^(samples/2)"
                              if args.rigorous else "A^samples B^samples; autotuned selection only"),
                    "precondition":
                    "each do_bench sample includes its own warmup",
                    "code_validation": ("separate workers for both variants"
                                        if args.rigorous else "one candidate-only worker after timing"),
                    "samples":
                    args.samples,
                    "warmup_ms":
                    args.warmup,
                    "rep_ms":
                    args.rep,
                    "max_paired_spread":
                    args.max_paired_spread,
                    "selection":
                    list(comparisons),
                },
            )
        if failures:
            return 1
        if args.rigorous and noisy and not args.allow_noisy:
            return 2
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
