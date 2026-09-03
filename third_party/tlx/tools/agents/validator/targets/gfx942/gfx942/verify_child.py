"""Check a candidate without the gate's performance environment.

Spawned per ``verify`` by ``harness._verify_default_environment()``, with
``TRITON_DISABLE_POST_MISCHED`` / ``TRITON_USE_C_DISPATCHER`` unset. Shares the
parent's Triton compile cache, so this re-checks the launch environment only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from third_party.tlx.tools.agents.validator.targets.gfx942.shared.inputs import (
    make_inputs,
)
from third_party.tlx.tools.agents.validator.targets.gfx942.shared.loader import (
    load_candidate,
)


def main() -> int:
    import torch

    kernel_path = Path(os.environ["TLX_VERIFY_KERNEL"])
    case = json.loads(os.environ["TLX_VERIFY_CASE"])
    entry_point = os.environ.get("TLX_VERIFY_ENTRY", "matmul")

    module = load_candidate(kernel_path, suffix="verify_candidate")
    a, b = make_inputs(case)
    torch.testing.assert_close(getattr(module, entry_point)(a, b), torch.matmul(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
