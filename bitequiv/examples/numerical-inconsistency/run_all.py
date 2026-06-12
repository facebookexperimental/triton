"""Run every numerical-inconsistency example in order.

Self-skips cleanly on hosts without a CUDA GPU. Never runs perf benchmarks (none
exist in this folder yet -- the before/after benchmark is deferred future work,
see README). A non-zero exit means an example's divergence assertion failed,
which would itself be a notable finding (the bits stopped diverging).

Run:  python bitequiv/examples/numerical-inconsistency/run_all.py
"""
import importlib
import sys

from _helpers import banner, is_cuda

EXAMPLES = [
    "n01_autotuner_picks_silently",
    "n02_sum_reduction_classes",
    "n03_softmax_row_reduction",
    "n04_layernorm_two_reductions",
    "n05_dot_reduce",
]


def main():
    if not is_cuda():
        print("[skip] no CUDA GPU; numerical-inconsistency examples require a GPU.")
        return 0
    failures = []
    for name in EXAMPLES:
        banner(f"RUN {name}")
        try:
            importlib.import_module(name).main()
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failures.append(name)
        except SystemExit as e:  # an example self-skipped
            print(f"[skip] {name}: {e}")
    if failures:
        print(f"\n{len(failures)} example(s) did not show divergence: {failures}")
        return 1
    print("\nAll examples ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
