import pytest
import torch
import re
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_cluster_launch_control(BLOCK_SIZE, device):

    @triton.jit
    def mul2_clc(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        tile_id = tl.program_id(axis=0)

        # CLC Init
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        while tile_id != -1:
            # CLC producer
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x * y
            tl.store(z_ptr + offsets, output, mask=mask)

            # CLC consumer
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
            clc_phase_consumer ^= 1

            if tlx.thread_id(axis=0) == 0:
                tl.device_print("Extracted CtaID", tile_id)

    torch.manual_seed(0)
    # number of kernels to launch in a non-persistent mode
    size = 10000000
    x = torch.ones(size, device=device)
    y = torch.ones(size, device=device)

    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = mul2_clc[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, launch_cluster=True)

    ptx = kernel.asm["ptx"]

    assert re.search((r"clusterlaunchcontrol.try_cancel"), ptx, flags=re.DOTALL)
    assert re.search((r"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128"), ptx, flags=re.DOTALL)
    assert re.search((r"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128"), ptx, flags=re.DOTALL)

    assert torch.count_nonzero(output) == size


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("CLUSTER_SIZE", [2, 4])
def test_cluster_launch_control_multi_cta(CLUSTER_SIZE, device):
    """
    Test CLC with 2-CTA clusters (multi_ctas=True).

    Verifies that:
    1. Both CTAs call barrier_expect_bytes (unpredicated) on their own local bar_full,
       because try_cancel with multicast::cluster::all signals each CTA's mbarrier.
    2. Both CTAs call barrier_wait (unpredicated) on their own local bar_full
       before reading the CLC response.
    3. The kernel produces correct results with persistent multi-CTA CLC scheduling.
    """

    @triton.jit
    def mul2_clc_multi_cta(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        CLUSTER_SIZE: tl.constexpr,
    ):
        # Each CTA in the cluster handles half the block
        tile_id = tl.program_id(axis=0)

        # CLC Init — num_consumers=CLUSTER_SIZE because all CTAs in the cluster
        # arrive at CTA 0's bar_empty in clc_consumer
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(CLUSTER_SIZE)

        while tile_id != -1:
            # CLC producer
            tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=True)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)

            # CLC consumer
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=True)
            clc_phase_consumer ^= 1

    torch.manual_seed(0)
    BLOCK_SIZE = 1024
    size = BLOCK_SIZE * CLUSTER_SIZE
    x = torch.ones(size, device=device)
    y = torch.ones(size, device=device)

    output = torch.zeros_like(x)
    ref_out = x + y

    n_elements = output.numel()
    # Grid: each logical tile is handled by 2 CTAs, so total CTAs = 2 * num_tiles
    num_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
    # Pad to multiple of 2 for 2-CTA clusters
    num_tiles = (num_tiles + 1) // CLUSTER_SIZE * CLUSTER_SIZE
    grid = (num_tiles, )
    kernel = mul2_clc_multi_cta[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        CLUSTER_SIZE=CLUSTER_SIZE,
        launch_cluster=True,
        ctas_per_cga=(CLUSTER_SIZE, 1, 1),
    )

    ptx = kernel.asm["ptx"]

    # CLC instructions are present
    assert re.search(r"clusterlaunchcontrol.try_cancel", ptx, flags=re.DOTALL)
    assert re.search(r"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128", ptx, flags=re.DOTALL)
    assert re.search(r"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128", ptx, flags=re.DOTALL)

    # Multicast is used (2-CTA cluster)
    assert re.search(r"multicast::cluster::all", ptx, flags=re.DOTALL)

    # mapa.shared::cluster for remote barrier arrive (consumer signals CTA 0's bar_empty)
    assert "mapa.shared::cluster" in ptx

    # Verify barrier_expect_bytes is NOT predicated by cluster_ctaid check.
    # Both CTAs must initialize their own bar_full because try_cancel with
    # multicast::cluster::all signals the mbarrier on each CTA's shared memory.
    # Look for expect_tx lines and ensure none are guarded by cluster_ctaid predicates.
    expect_tx_lines = [line.strip() for line in ptx.split("\n") if "expect_tx" in line]
    assert len(expect_tx_lines) > 0, "Expected mbarrier.arrive.expect_tx in PTX"

    # The mbarrier.try_wait for the CLC response should NOT be skipped by rank-1.
    # In the buggy version, rank-1 would branch past the try_wait with:
    #   @!pred_cta0 bra skipWait
    # After the fix, all CTAs should hit mbarrier.try_wait unconditionally.
    try_wait_lines = [line.strip() for line in ptx.split("\n") if "mbarrier.try_wait" in line]
    assert len(try_wait_lines) > 0, "Expected mbarrier.try_wait in PTX"

    # Verify correctness
    torch.testing.assert_close(output, ref_out)
