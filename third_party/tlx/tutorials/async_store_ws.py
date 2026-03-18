"""
Warp-Specialized Store (PlanCTA regression test)
=================================================

Tests tl.store in a warp-specialized context where the store partition
has fewer warps (1) than the default partition, with num_ctas=2 to
ensure PlanCTA actually runs (it skips when num_ctas=1).

This exercises PlanCTA's per-op numWarps lookup: the store's layout
must be planned with 1 warp (the partition's warp count), not the
function-level total. Without the fix (lookupNumWarps(store) instead
of lookupNumWarps(funcOp)), PlanCTA would assign warpsPerCTA=[4]
inside the 1-warp partition, producing an invalid layout.
"""

import torch
import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hopper_or_newer

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def store_ws_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            # Generate data in registers using tl.arange (creates blocked layout)
            _ = tl.arange(0, BLOCK_SIZE)

        with tlx.async_task(num_warps=1):
            # Store from registers to global in a 1-warp partition.
            # PlanCTA must use this partition's numWarps (1), not the
            # function-level numWarps, when computing the store layout.
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            data = offsets.to(tl.float32)
            tl.store(output_ptr + offsets, data)


@pytest.mark.skipif(
    not is_hopper_or_newer(),
    reason="Requires Hopper GPU or above",
)
def test_store_ws():
    BLOCK_SIZE = 256
    n_elements = 1024
    n_blocks = n_elements // BLOCK_SIZE

    output = torch.empty(n_elements, device=DEVICE, dtype=torch.float32)
    # num_ctas=2 ensures PlanCTA runs (it skips when num_ctas=1).
    store_ws_kernel[(n_blocks, )](output, BLOCK_SIZE=BLOCK_SIZE, num_ctas=2)

    expected = torch.arange(n_elements, device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    test_store_ws()
