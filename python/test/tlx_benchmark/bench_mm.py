"""Perf and compile-time guardrail for `tlx.ops.mm`, against `torch.matmul`."""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

from triton.tlx.ops.kernels.mm._shapes import flops, label, operand

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "mm"
REF_NAME = "torch.matmul"
#: mm has a shape heuristic, so its default is a single analytically chosen
#: config and a first call stays under a second.
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

#: Relative tolerance for the accuracy check, per dtype. Same values the L1
#: correctness suite uses, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.mm._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.mm.{driver.arch()}").PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    # dtype is a Case field, so it is dropped from `shape` -- carrying it in
    # both duplicates it in the key and in the report.
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             label=label(*entry)) for entry in shapes(synthetic)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import mm as tlx_mm

    M, N, K, a_strides, b_strides = case.shape
    dtype = getattr(torch, case.dtype)
    a, b = operand(M, K, a_strides, dtype), operand(K, N, b_strides, dtype)

    tlx_fn = lambda: tlx_mm(a, b, arch=driver.arch(), space=space)  # noqa: E731
    ref_fn = lambda: torch.matmul(a, b)  # noqa: E731
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=ref_fn,
        flop_count=flops(M, N, K),
        check=lambda: close_enough(tlx_fn(), ref_fn(), REL_PRECISION[case.dtype]),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

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
