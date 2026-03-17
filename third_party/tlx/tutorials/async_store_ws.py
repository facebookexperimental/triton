"""
Async Store with Warp Specialization
=====================================

Tests the tlx.async_store operation in a warp-specialized context.
The default partition generates data in registers (via tl.arange) and stores
it to SMEM. The store partition (1 warp) uses async_store to bulk-copy from
SMEM to global memory.
"""

import torch
import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hopper_or_newer

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def async_store_ws_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Allocate a single SMEM buffer
    smem_buf = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)

    # Allocate mbarrier for producer-consumer synchronization
    bar = tlx.alloc_barriers(1, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            # Generate data in registers using tl.arange (creates blocked layout)
            data = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.float32)

            # Store register data to SMEM
            tlx.local_store(smem_buf[0], data)

            # Signal barrier: data is ready in SMEM
            tlx.barrier_arrive(bar[0])

        with tlx.async_task(num_warps=1):
            # Wait for producer to finish writing to SMEM
            tlx.barrier_wait(bar[0], 0)

            # Proxy fence: make generic-proxy SMEM writes visible to async proxy
            tlx.fence("async_shared")

            # Async bulk copy from SMEM to global memory
            byte_size: tl.constexpr = BLOCK_SIZE * tlx.size_of(tl.float32)
            dst = output_ptr + pid * BLOCK_SIZE
            tlx.async_store(dst, smem_buf[0], byte_size)

            # Wait for async store completion
            tlx.async_descriptor_store_wait(0)


@pytest.mark.skipif(
    not is_hopper_or_newer(),
    reason="Requires Hopper GPU or above",
)
def test_async_store_ws():
    BLOCK_SIZE = 256
    n_elements = 1024
    n_blocks = n_elements // BLOCK_SIZE

    output = torch.empty(n_elements, device=DEVICE, dtype=torch.float32)
    async_store_ws_kernel[(n_blocks,)](output, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.arange(n_elements, device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    test_async_store_ws()
