"""Traced child process: launched by :mod:`att` under ``rocprofv3 -- ...``.

Runs once per ``att.collect()``, so once per profiled candidate.

Deliberately tiny and side-effect free. Its whole job is to issue a
deterministic number of dispatches so that ``--kernel-iteration-range`` can name
exactly one of them:

    warmup dispatches 1..W, then the traced dispatch W+1.

One dispatch per call holds only while the target resolves to a single autotuner
config (`mm`'s `space="heuristic"`); `Autotuner.run` skips benchmarking unless
`len(self.configs) > 1`. A candidate that widens the search space would issue one
dispatch per config on the first call, landing the iteration range somewhere
arbitrary -- :mod:`att`'s traced-dispatch count is the check on that.

Inputs arrive by environment variable because rocprofv3 owns the command line.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

from inputs import make_inputs  # same directory; sys.path[0] is this script's dir


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tlx_att_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tlx_att_candidate"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch

    kernel_path = Path(os.environ["TLX_ATT_KERNEL"])
    case = json.loads(os.environ["TLX_ATT_CASE"])
    entry_point = os.environ.get("TLX_ATT_ENTRY", "matmul")
    warmup = int(os.environ.get("TLX_ATT_WARMUP", "3"))

    a, b = make_inputs(case)
    call = getattr(_load(kernel_path), entry_point)

    # Absorbs JIT compilation, so the traced dispatch is not a cold first launch.
    for _ in range(warmup):
        call(a, b)
    torch.cuda.synchronize()

    call(a, b)  # <- the traced dispatch
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
