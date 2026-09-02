"""Check an agent candidate through the default user launch path.

Spawned by ``harness._verify_default_environment()`` on every ``verify`` call, so a
candidate is also checked without the performance environment the gate runs under
(``TRITON_DISABLE_POST_MISCHED`` / ``TRITON_USE_C_DISPATCHER``).

Note this shares the parent's Triton compile cache, so it re-checks the *launch
environment*, not a cold compile.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

# `harnesses/gfx942/`, which holds the shared inputs module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inputs import make_inputs  # noqa: E402


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

    a, b = make_inputs(case)
    torch.testing.assert_close(getattr(module, entry_point)(a, b), torch.matmul(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
