from __future__ import annotations

import argparse
import functools
import importlib
import os
import sys

import torch

from triton.tlx.ops.kernels.mm._shapes import flops, label, operand

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _harness import (DEFAULT_REPLICATES, Case, Status, capture_env, cold_compile,  # noqa: E402
                      host_overhead_us, measure, stable)
from _harness.denoise import Governor, list_devices, select_device  # noqa: E402
from _harness import report as report_mod  # noqa: E402
from _harness import verdict  # noqa: E402

OP = "mm"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@functools.lru_cache(maxsize=1)
def arch() -> str | None:
    devices = list_devices()
    return devices[0].arch if devices else None


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.mm.{arch()}").PERF_SHAPES)


def default_json() -> str:
    return f"/tmp/tlx_benchmark/{OP}.{arch()}.json"


def cases(head: int | None = None, synthetic: bool = False) -> list[Case]:
    out = [
        # dtype is a Case field, so it is dropped from `shape` -- carrying it in
        # both duplicates it in the key and in the report.
        Case(op=OP, arch=arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic)
    ]
    return out[:head] if head else out


def _operands(case: Case):
    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    return operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)


#: Relative tolerance for the accuracy check, per dtype. Same values the L1
#: correctness suite uses, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def _accuracy(out, ref_out, dtype: str) -> tuple[bool, str]:
    precision = REL_PRECISION[dtype]
    try:
        torch.testing.assert_close(out, ref_out, atol=precision * ref_out.abs().max().item(), rtol=precision)
    except AssertionError as mismatch:
        return False, f"output does not match the reference: {str(mismatch).splitlines()[0]}"
    return True, ""


def run_case(case: Case, *, space: str):
    from triton.tlx.ops import mm as tlx_mm

    a, b = _operands(case)
    tlx_fn = lambda: tlx_mm(a, b, arch=arch(), space=space)  # noqa: E731
    ref_fn = lambda: torch.matmul(a, b)  # noqa: E731

    compile_stat = cold_compile(tlx_fn)

    out = tlx_fn()  # tune and compile outside the measured window
    ref_out = ref_fn()
    torch.cuda.synchronize()
    correct, accuracy_note = _accuracy(out, ref_out, case.dtype)
    del out, ref_out

    # The FLOP count is what makes the returned Stats throughputs: `measure`
    # converts every timed iteration, so both providers come back in TFLOP/s
    # with the dispersion measured on that quantity rather than on latency.
    M, N, K = case.shape[0], case.shape[1], case.shape[2]
    flop_count = flops(M, N, K)
    tlx = measure(tlx_fn, flop_count=flop_count, replicates=DEFAULT_REPLICATES)
    ref = measure(ref_fn, flop_count=flop_count, replicates=DEFAULT_REPLICATES)
    host_us = host_overhead_us(tlx_fn)

    result = verdict.judge(case, tlx, ref, tlx_host_us=host_us, compile_stat=compile_stat, correct=correct,
                           accuracy_note=accuracy_note)
    result.flop_count = flop_count

    # The operands are freed when this frame exits; empty_cache then returns the
    # blocks to the driver so the next case (up to 2 GB at 1000000x512) can be
    # allocated. Do not `del` them -- the closures above still name them.
    torch.cuda.empty_cache()
    return result


_FATAL_CUDA_ERRORS = (
    "device-side assert",
    "illegal memory access",
    "launch failure",
    "launch timeout",
    "misaligned address",
)


def _is_fatal_cuda_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA_ERRORS)


def _run_cases(case_list, run_one):
    results = []
    for case in case_list:
        try:
            results.append(run_one(case))
        except Exception as exc:
            results.append(_errored(case, exc))
            # Validation failures are isolated to one case. A fatal CUDA
            # error poisons the process context, so every later error would
            # be a cascade rather than an independent result.
            if _is_fatal_cuda_error(exc):
                results[-1].notes.append("stopping: the CUDA context may be unusable after this error")
                break
    return results


def run(*, space="heuristic", head=None, synthetic=False, governor=None):
    env = capture_env()
    if governor is not None:
        env["governed"] = governor.to_dict()
    with stable() as info:
        results = _run_cases(cases(head, synthetic), lambda case: run_case(case, space=space))
    # The autotune space is part of what a number means: a heuristic-space
    # latency and a full-space latency for the same shape differ by 4x, so two
    # artifacts are only comparable when this matches.
    env["space"] = space
    env["replicates"] = DEFAULT_REPLICATES
    if head:
        env["head"] = head
    env["shapes"] = "synthetic" if synthetic else "focus"
    env["run"] = {k: info[k] for k in ("problems", "clock_trace", "elapsed_s") if k in info}
    return results, env


def _errored(case: Case, exc: Exception):
    from _harness import Result

    result = Result(case=case, status=Status.ERROR)
    result.notes.append(f"{type(exc).__name__}: {exc}")
    return result


def supported() -> bool:
    return arch() is not None


# --------------------------------------------------------------------------
# CLI entry point -- the deterministic command
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="GPU index, or 'auto' (default) for the least-used one")
    parser.add_argument(
        "--space", choices=("heuristic", "full", "smoke"), default="heuristic",
        help="autotune search space; 'heuristic' is what tlx.ops.mm uses by default, and "
        "measuring anything else measures a path users do not take")
    parser.add_argument("--head", type=int, default=None, metavar="N", help="only the first N cases, for a quick look")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run the correctness shapes instead of this arch's focus list; they are "
        "mostly too small to time, so this is for looking, not for gating")
    parser.add_argument("--json", default=default_json(), help=f"machine-readable artifact (default {default_json()})")
    args = parser.parse_args(argv)

    # Pick and pin the GPU before torch touches CUDA. Selection has to happen
    # here rather than in a wrapper script so that the suite is one command,
    # and it has to happen before the first CUDA call because the visibility
    # variable is read once at context creation.
    device = select_device(args.device)
    if device is not None:
        os.environ[device.visibility_env] = str(device.index)
        print(f"device: gpu{device.index} {device.name} "
              f"({'least used' if args.device == 'auto' else 'requested'}, "
              f"{device.memory_used_mib:.0f} MiB in use)")

    # Governing is unconditional: a number taken on an ungoverned machine is not
    # comparable to anything, so there is no switch to take one.
    with Governor(device) as governor:
        for step in governor.applied:
            print(f"  denoise: {step}")
        for step in governor.skipped:
            print(f"  denoise: SKIPPED {step}")
        results, env = run(space=args.space, head=args.head, synthetic=args.synthetic, governor=governor)
    if not results:
        # An empty focus list is legitimate -- an arch may have no capture yet --
        # but a silent zero-row table reads like a pass. Say which list was empty.
        print(f"no {'synthetic' if args.synthetic else 'focus'} shapes for {arch()}; nothing measured")
        return 0
    print(report_mod.render(results, env, args.json))

    return 1 if report_mod.failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
