"""Time a full-autotune bench_mm run over candidate sm100_ci_selected shapes.

One-off measurement: prints the wall time of every case so the suite can be
sized to the 45-minute per-bench limit. The shapes are listed here rather than
in _shapes.py because the installed wheel carries its own copy of _shapes.py.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import driver  # noqa: E402

import bench_mm  # noqa: E402

# One shape per (regime, layout) group of sm100_1, picked near the group's
# median FLOPs.
CANDIDATES = (
    (512, 544, 589824, (589824, 1), (544, 1), "bf16"),
    (256, 256, 2433024, (1, 256), (256, 1), "bf16"),
    (512, 384, 294912, (1, 512), (1, 294912), "bf16"),
    (442368, 512, 192, (192, 1), (512, 1), "bf16"),
    (2433024, 256, 256, (256, 1), (1, 256), "bf16"),
    (49152, 512, 1152, (1, 49152), (512, 1), "bf16"),
    (2304, 65536, 1024, (1024, 1), (65536, 1), "bf16"),
    (1024, 65536, 1152, (1, 1024), (65536, 1), "bf16"),
    (3072, 294912, 384, (1, 3072), (1, 384), "bf16"),
    (2304, 12800, 1024, (1024, 1), (12800, 1), "bf16"),
    (12800, 1024, 2304, (1, 12800), (1024, 1), "bf16"),
    (512, 4096, 73728, (1, 512), (1, 73728), "bf16"),
    (1152, 4096, 2048, (2048, 1), (4096, 1), "bf16"),
    (2304, 1056, 1056, (1088, 1), (1, 1056), "bf16"),
    (2048, 4096, 1152, (1, 2048), (4096, 1), "bf16"),
)

bench_mm.shapes = lambda synthetic=False, suites=None: list(CANDIDATES)

_run_case = driver.run_case


def _timed_run_case(bench, case, **kwargs):
    start = time.time()
    try:
        return _run_case(bench, case, **kwargs)
    finally:
        print(f"CASE_TIME {case.label} cold={kwargs.get('cold')} {time.time() - start:.1f}s", flush=True)


driver.run_case = _timed_run_case

if __name__ == "__main__":
    sys.argv = ["bench_mm.py"] + sys.argv[1:]
    start = time.time()
    try:
        raise SystemExit(bench_mm.main())
    finally:
        print(f"TOTAL_TIME {time.time() - start:.1f}s", flush=True)
