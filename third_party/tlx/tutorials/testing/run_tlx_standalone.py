"""Run a generated TTGIR-to-TLX standalone benchmark under beta Triton.

The TTGIR-to-TLX converter, when driven with TRITON_DUMP_TLX_BENCHMARK=1, writes
a self-contained `_test_standalone.py` that compiles the generated TLX kernel and
times it (and the source kernel) via `triton.testing.do_bench`. That script needs
a beta-Triton Python environment to run; this thin wrapper provides one via buck
(`:triton-py` dep) and execs the standalone. Point TLX_STANDALONE at the path.
"""

from __future__ import annotations

import os
import runpy


def main() -> None:
    # Required rather than defaulted: a stale standalone from an earlier dump
    # benchmarks the wrong kernel and still exits 0.
    standalone = os.environ.get("TLX_STANDALONE")
    if not standalone:
        raise SystemExit("TLX_STANDALONE is unset; point it at a dumped _test_standalone.py")
    if not os.path.isfile(standalone):
        raise SystemExit(f"TLX_STANDALONE is not a file: {standalone}")

    print(f"[run_tlx_standalone] executing {standalone}", flush=True)
    runpy.run_path(standalone, run_name="__main__")


if __name__ == "__main__":
    main()
