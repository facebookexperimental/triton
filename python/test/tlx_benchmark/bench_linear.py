"""Benchmark the five frozen gfx942 ``tlx.ops.linear`` shapes against aten.

Both providers consume the same contiguous BF16 ``input[M, K]`` and
``weight[N, K]`` tensors and write to preallocated outputs. Four shapes
include a vector bias; one is bias-free. Run on MI300X with:

    python python/test/tlx_benchmark/bench_linear.py
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import torch

from triton.tlx.ops.kernels.linear.gfx942 import LINEAR_CONFIGS

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _harness import DEFAULT_REPLICATES, Case, Status, capture_env, cold_compile, host_overhead_us, measure, stable  # noqa: E402
from _harness import report as report_mod  # noqa: E402
from _harness import verdict  # noqa: E402
from _harness.denoise import Governor, list_devices, select_device  # noqa: E402

OP = "linear"
DTYPE = torch.bfloat16
REL_PRECISION = 0.05


@functools.lru_cache(maxsize=1)
def arch() -> str | None:
    devices = list_devices()
    return devices[0].arch if devices else None


def default_json() -> str:
    return f"/tmp/tlx_benchmark/{OP}.{arch()}.json"


def cases(head: int | None = None) -> list[Case]:
    out = []
    for (M, N, K), config in LINEAR_CONFIGS.items():
        bias = config["bias"]
        label = f"{M}x{N}x{K} weight:NK bias:{'yes' if bias else 'no'}"
        out.append(Case(op=OP, arch="gfx942", dtype="bfloat16", shape=(M, N, K, bias), label=label))
    return out[:head] if head else out


def _operands(case: Case):
    M, N, K, has_bias = case.shape
    a = torch.randn((M, K), device="cuda", dtype=DTYPE)
    weight = torch.randn((N, K), device="cuda", dtype=DTYPE)
    bias = torch.randn((N, ), device="cuda", dtype=DTYPE) if has_bias else None
    tlx_out = torch.empty((M, N), device="cuda", dtype=DTYPE)
    ref_out = torch.empty_like(tlx_out)
    return a, weight, bias, tlx_out, ref_out


def _accuracy(out, ref_out) -> tuple[bool, str]:
    try:
        torch.testing.assert_close(out, ref_out, atol=REL_PRECISION, rtol=REL_PRECISION)
    except AssertionError as mismatch:
        return False, f"output does not match the reference: {str(mismatch).splitlines()[0]}"
    return True, ""


def run_case(case: Case):
    from triton.tlx.ops import linear as tlx_linear

    M, N, K, has_bias = case.shape
    a, weight, bias, tlx_out, ref_out = _operands(case)
    b = weight.T
    tlx_fn = lambda: tlx_linear(a, weight, bias, out=tlx_out, arch="gfx942")  # noqa: E731
    if has_bias:
        ref_fn = lambda: torch.addmm(bias, a, b, out=ref_out)  # noqa: E731
    else:
        ref_fn = lambda: torch.mm(a, b, out=ref_out)  # noqa: E731

    compile_stat = cold_compile(tlx_fn)
    out = tlx_fn()
    reference = ref_fn()
    torch.cuda.synchronize()
    correct, accuracy_note = _accuracy(out, reference)

    flop_count = 2 * M * N * K
    tlx = measure(tlx_fn, flop_count=flop_count, replicates=DEFAULT_REPLICATES)
    ref = measure(ref_fn, flop_count=flop_count, replicates=DEFAULT_REPLICATES)
    result = verdict.judge(case, tlx, ref, tlx_host_us=host_overhead_us(tlx_fn), compile_stat=compile_stat,
                           correct=correct, accuracy_note=accuracy_note)
    result.flop_count = flop_count

    return result


def _errored(case: Case, exc: Exception):
    from _harness import Result

    result = Result(case=case, status=Status.ERROR)
    result.notes.append(f"{type(exc).__name__}: {exc}")
    return result


def supported() -> bool:
    return arch() == "gfx942"


def run(*, space=None, head=None, synthetic=False, governor=None):
    del space, synthetic  # The five configurations are frozen; there is no search space.
    env = capture_env()
    if governor is not None:
        env["governed"] = governor.to_dict()
    results = []
    with stable() as info:
        for case in cases(head):
            try:
                results.append(run_case(case))
            except Exception as exc:
                results.append(_errored(case, exc))
            torch.cuda.empty_cache()
    env["space"] = "frozen"
    env["replicates"] = DEFAULT_REPLICATES
    if head:
        env["head"] = head
    env["shapes"] = "gfx942-linear-focus"
    env["run"] = {key: info[key] for key in ("problems", "clock_trace", "elapsed_s") if key in info}
    return results, env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="GPU index, or 'auto' for the least-used one")
    parser.add_argument("--head", type=int, default=None, metavar="N", help="only the first N cases")
    parser.add_argument("--json", default=default_json(), help=f"machine-readable artifact (default {default_json()})")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    if device is None or device.arch != "gfx942":
        print("tlx.ops.linear requires a gfx942 GPU")
        return 1
    os.environ[device.visibility_env] = str(device.index)
    print(f"device: gpu{device.index} {device.name} "
          f"({'least used' if args.device == 'auto' else 'requested'}, {device.memory_used_mib:.0f} MiB in use)")

    with Governor(device) as governor:
        for step in governor.applied:
            print(f"  denoise: {step}")
        for step in governor.skipped:
            print(f"  denoise: SKIPPED {step}")
        results, env = run(head=args.head, governor=governor)
    print(report_mod.render(results, env, args.json))
    return 1 if report_mod.failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
