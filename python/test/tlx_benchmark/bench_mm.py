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
    """Catalog key for the GPU under test, or None if ``mm`` has no entry.

    Read off the part name via smi rather than via torch, because this is called
    before the device is pinned and a CUDA context created here would ignore
    ``--device``. Takes device 0: a box with two different parts in it is not a
    machine anyone should be gating perf on.
    """
    devices = list_devices()
    return devices[0].arch if devices else None


def shapes() -> list[list]:
    """Synthetic shapes plus this arch's focus list.

    L2 measures the synthetic shapes as well as validating them, and leaves out
    only *other* arches' focus lists. See ``kernels/mm/_shapes.py``.
    """
    from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC

    return list(SYNTHETIC) + list(importlib.import_module(f"triton.tlx.ops.kernels.mm.{arch()}").PERF_SHAPES)


def default_json() -> str:
    """Where the machine-readable artifact goes when nothing is asked for.

    Written unconditionally: the JSON is the interface a review agent reads, and
    making it opt-in meant the common invocation produced nothing to read. Not
    in the repo -- it is a per-run output, not a checked-in one.
    """
    return f"/tmp/tlx_benchmark/{OP}.{arch()}.json"


#: What one row of the report describes, printed in the legend. An mm case is a
#: product shape plus the memory layout of each operand -- the layout is not
#: decoration, it selects a different kernel path and moves the TMA alignment
#: constraint to a different dimension.
INPUT_SPEC = ("(M x K) @ (K x N) with each operand's recorded strides; [[a0, a1], [b0, b1]]. "
              "A leading stride wider than the row is a slice of a padded buffer, and 0 is a broadcast.")


def cases(head: int | None = None) -> list[Case]:
    """Every case, or the first ``head`` of them.

    One case per entry, not a dtype cross-product: each shape carries the dtype
    it was captured in, so running it in the other precision would measure a
    workload nobody has.
    """
    out = [
        # dtype is a Case field, so it is dropped from `shape` -- carrying it in
        # both duplicates it in the key and in the report.
        Case(op=OP, arch=arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes()
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
    """Whether TLX's output matches the reference, and why not if it does not.

    Timing a kernel without checking it produces a number that looks like signal
    and is not -- and the autotuner ranks configs by speed without ever looking
    at their results, so a wrong-answer config can win. Cheap next to the
    measurement window, and it runs once per case rather than per iteration.
    """
    precision = REL_PRECISION[dtype]
    try:
        torch.testing.assert_close(out, ref_out, atol=precision * ref_out.abs().max().item(), rtol=precision)
    except AssertionError as mismatch:
        return False, f"output does not match the reference: {str(mismatch).splitlines()[0]}"
    return True, ""


def run_case(case: Case, *, space: str):
    """Measure one case and judge it.

    Latency and cold-compile are two passes over two different cache states:
    ``t_cold`` needs an empty Triton cache and steady-state latency needs a
    warm one, so neither can be derived from the other's call.
    """
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


def run(*, space="heuristic", head=None, governor=None):
    env = capture_env()
    if governor is not None:
        env["governed"] = governor.to_dict()
    results = []
    with stable() as info:
        for case in cases(head):
            try:
                results.append(run_case(case, space=space))
            except Exception as exc:  # a broken case must not hide the others
                results.append(_errored(case, exc))
    # The autotune space is part of what a number means: a heuristic-space
    # latency and a full-space latency for the same shape differ by 4x, so two
    # artifacts are only comparable when this matches.
    env["space"] = space
    env["replicates"] = DEFAULT_REPLICATES
    env["input_spec"] = INPUT_SPEC
    if head:
        env["head"] = head
    env["run"] = {k: info[k] for k in ("problems", "clock_trace", "elapsed_s") if k in info}
    return results, env


def _errored(case: Case, exc: Exception):
    from _harness import Result

    result = Result(case=case, status=Status.ERROR)
    result.notes.append(f"{type(exc).__name__}: {exc}")
    return result


def supported() -> bool:
    """Whether this op can run on the current device at all."""
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
        results, env = run(space=args.space, head=args.head, governor=governor)
    print(report_mod.render(results, env, args.json))

    return 1 if report_mod.failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
