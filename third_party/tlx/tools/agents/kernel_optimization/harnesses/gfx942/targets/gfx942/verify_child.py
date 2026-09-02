"""Check an agent candidate through the default, unpinned user launch path.

Spawned by ``harness._verify_default_environment()`` on every ``verify`` call, so a
candidate is also checked without the autotuner pin the rest of the harness applies.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> int:
    import torch

    kernel_path = Path(os.environ["TLX_VERIFY_KERNEL"])
    case = json.loads(os.environ["TLX_VERIFY_CASE"])
    entry_point = os.environ.get("TLX_VERIFY_ENTRY", "matmul")

    spec = importlib.util.spec_from_file_location("tlx_verify_candidate", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tlx_verify_candidate"] = module
    spec.loader.exec_module(module)

    parameters = case["parameters"]
    device = parameters.get("device", "cuda")
    dtype = getattr(torch, parameters.get("dtype", "float16"))
    k = int(parameters["k"])
    torch.manual_seed(int(parameters.get("seed", 0)))
    a = (torch.randn((int(parameters["m"]), k), device=device, dtype=dtype) + 1) / k
    b = (torch.randn((k, int(parameters["n"])), device=device, dtype=dtype) + 1) / k

    actual = getattr(module, entry_point)(a, b)
    torch.testing.assert_close(actual, torch.matmul(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
