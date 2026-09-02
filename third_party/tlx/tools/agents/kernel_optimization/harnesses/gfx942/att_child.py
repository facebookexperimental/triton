"""Traced child process: launched by :mod:`att` under ``rocprofv3 -- ...``.

Deliberately tiny and side-effect free. Its whole job is to issue a
deterministic number of dispatches so that ``--kernel-iteration-range`` can name
exactly one of them:

    warmup dispatches 1..W, then the traced dispatch W+1.

That determinism only holds while the autotuner is pinned to a single config
(``TLX_AGENT_PIN_CONFIG``); an unpinned run issues one dispatch per config
during the search and the iteration range would land somewhere arbitrary inside
it. :mod:`att` always sets the pin for this reason.

Everything arrives by environment variable rather than argv because rocprofv3
owns the command line.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


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

    parameters = case["parameters"]
    device = parameters.get("device", "cuda")
    dtype = getattr(torch, parameters.get("dtype", "float16"))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(parameters.get("seed", 0)))
    a = torch.randn(
        (int(parameters["m"]), int(parameters["k"])),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    b = torch.randn(
        (int(parameters["k"]), int(parameters["n"])),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    module = _load(kernel_path)
    pin_config = os.environ.get("TLX_AGENT_PIN_CONFIG")
    if pin_config:
        import pinning  # same directory; sys.path[0] is this script's dir

        pinning.pin(module, json.loads(pin_config))
    call = getattr(module, entry_point)

    # Warmup also absorbs JIT compilation, so the traced dispatch measures the
    # steady-state kernel rather than a cold-cache first launch.
    for _ in range(warmup):
        call(a, b)
    torch.cuda.synchronize()

    call(a, b)  # <- the traced dispatch
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
