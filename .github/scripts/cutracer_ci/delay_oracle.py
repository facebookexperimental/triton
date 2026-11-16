#!/usr/bin/env python3
"""Oracle for ``cutracer reduce``: does one test still fail under a delay config?

``cutracer reduce`` injects CUTracer with a candidate config and runs this
script, which runs the test and exits as reduce expects:

    0  the test failed, or crashed with a CUDA error (the bug showed)
    1  the test passed
    2  anything else (no record, a death without a CUDA error), which reduce
       treats as an infrastructure error

Usage:
    delay_oracle.py 'python/test/unit/language/test_tlx_dot_sm100.py::test_async_dot_scaled_2cta[m64_per_cta_unit_scales]'
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from random_delay import CUDA_ERROR, PYTEST, read_records, status


def main(argv: list[str] | None = None) -> int:
    (test, ) = argv if argv is not None else sys.argv[1:]
    with tempfile.TemporaryDirectory() as scratch:
        records = Path(scratch) / "records.jsonl"
        proc = subprocess.run(PYTEST + [test], env=dict(os.environ, RECORD_OUTCOMES=str(records)), capture_output=True,
                              text=True, errors="replace", check=False)
        run = read_records(records.read_text().splitlines() if records.exists() else [])
    outcome = status(run.outcomes, test)
    if outcome == "failed" or (run.crashed == test and CUDA_ERROR.search(proc.stdout + proc.stderr)):
        return 0
    return 1 if outcome == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
