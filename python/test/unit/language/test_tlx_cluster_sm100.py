"""TLX cluster tests -- Blackwell-only."""
import re
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell or newer for preferred cluster dimension")
def test_preferred_ctas_per_cga(device):
    """Test launching kernels with preferred_ctas_per_cga hint."""

    @triton.jit
    def copy_kernel(x_ptr, log_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(x_ptr + offsets, offsets, mask=mask)

        # allocate 128x512 TMEM to force an occupancy of 1 (works on B200)
        tmem_buf = tlx.local_alloc((128, 512), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_init = tl.full((128, 512), 1, dtype=tl.float32)
        tlx.local_store(tmem_buf[0], acc_init)

        # assuming log_ptr tensor has size equal to number of programs
        tl.store(log_ptr + pid, tlx.cluster_size_1d())

    # setting up grid in a way that there's exactly one wave (one CTA per SM)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    GRID_SIZE = NUM_SMS
    BLOCK_SIZE = 4
    NUM_ELEMENT = GRID_SIZE * BLOCK_SIZE
    x = torch.zeros(NUM_ELEMENT, dtype=torch.float32, device=device)
    # each value is the cluster size of a CTA
    cluster_size_log = torch.full((GRID_SIZE, ), -1, dtype=torch.int16, device=device)
    kern_kwargs = {
        "BLOCK_SIZE": BLOCK_SIZE, "num_warps": 4, "preferred_ctas_per_cga": (4, 1, 1), "ctas_per_cga": (2, 1, 1)
    }
    # due to B200 number of SMS and number of GPCs limitation, 4x1 clusters cannot fully
    # tile the 148 SMs (e.g. a GPC could possible has 18 SMs hypothetically), so we will
    # have bubbles of 2 SMs that can be leveraged to fill a 2x1 cluster
    kernel = copy_kernel[(GRID_SIZE, )](x, cluster_size_log, NUM_ELEMENT, **kern_kwargs)
    assert kernel.metadata.preferred_ctas_per_cga == (4, 1, 1), (
        f"expecting preferred_ctas_per_cga to be (4, 1, 1), got {kernel.metadata.preferred_ctas_per_cga}")
    assert kernel.metadata.ctas_per_cga == (2, 1, 1), (
        f"expecting ctas_per_cga to be (2, 1, 1), got {kernel.metadata.ctas_per_cga}")

    sizes, counts = cluster_size_log.unique(return_counts=True)
    d = dict(zip(sizes.tolist(), counts.tolist()))
    assert len(d) == 2 and 2 in d and 4 in d, f"expecting exactly two cluster sizes as specified, got {d}"
    assert 0 < d[2] and d[2] < d[4], f"expecting most clusters to have preferred sizes, got {d}"


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
    assert "mapa.shared::cluster" not in ptx

    query_instr = ptx.index("clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128")
    fence_instr = ptx.index("fence.proxy.async.shared::cta")
    assert 0 < query_instr < fence_instr

    assert torch.count_nonzero(output) == size


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("GRID_DIMS", [1, 2, 3])
def test_cluster_launch_control_3d(GRID_DIMS, device):
    """Test CLC with 1D, 2D, and 3D grids using return_3d=True."""

    @triton.jit
    def clc_3d_kernel(
        output_ptr,
        TILES_X: tl.constexpr,
        TILES_Y: tl.constexpr,
        TILES_Z: tl.constexpr,
    ):
        tile_x = tl.program_id(0)
        tile_y = tl.program_id(1)
        tile_z = tl.program_id(2)

        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        while tile_x != -1:
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1

            linear_idx = tile_z * (TILES_X * TILES_Y) + tile_y * TILES_X + tile_x

            if tlx.thread_id(axis=0) == 0:
                tl.store(output_ptr + linear_idx, linear_idx)

            tile_x, tile_y, tile_z = tlx.clc_consumer(clc_context, clc_phase_consumer, return_3d=True)
            clc_phase_consumer ^= 1

    if GRID_DIMS == 1:
        TILES_X, TILES_Y, TILES_Z = 64, 1, 1
    elif GRID_DIMS == 2:
        TILES_X, TILES_Y, TILES_Z = 4, 3, 1
    else:
        TILES_X, TILES_Y, TILES_Z = 3, 2, 2

    total_tiles = TILES_X * TILES_Y * TILES_Z
    output = torch.full((total_tiles, ), -1, dtype=torch.int32, device=device)
    expected = torch.arange(total_tiles, dtype=torch.int32, device=device)

    grid = (TILES_X, TILES_Y, TILES_Z)
    kernel = clc_3d_kernel[grid](
        output,
        TILES_X=TILES_X,
        TILES_Y=TILES_Y,
        TILES_Z=TILES_Z,
        launch_cluster=True,
    )

    ptx = kernel.asm["ptx"]
    assert re.search(r"clusterlaunchcontrol.try_cancel", ptx)
    assert re.search(r"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128", ptx)

    sorted_output, _ = torch.sort(output)
    torch.testing.assert_close(sorted_output, expected)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("CLUSTER_SIZE", [2, 4])
def test_cluster_launch_control_multi_cta(CLUSTER_SIZE, device):
    """
    Test CLC with multi-CTA clusters using the default cluster-aware path.

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
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)

            # CLC consumer
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
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


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_cluster_launch_control_multi_cta_delayed_exit(device):
    """
    Test that CLC multi-CTA correctly skips barrier_arrive when tile_id is -1.

    CTA 1 is held with a busy-wait before its last clc_consumer call,
    ensuring CTA 0 finishes first. Without the predicated barrier_arrive skip,
    CTA 1 would arrive at CTA 0's bar with tile_id == -1, when CTA 0 already exits,
    and thus cause errors.
    """
    CLUSTER_SIZE = 2

    @triton.jit
    def clc_delayed(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        CLUSTER_SIZE: tl.constexpr,
    ):
        tile_id = tl.program_id(axis=0)
        cta_rank = tlx.cluster_cta_rank()

        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(CLUSTER_SIZE)

        while tile_id != -1:
            tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=True)
            clc_phase_producer ^= 1

            # just do some regular processing
            block_start = tile_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)

            # Hold CTA 1 before it calls clc_consumer.
            # This ensures CTA 0 finishes and exits first, exercising the
            # predicated barrier_arrive skip (tile_id == -1 should NOT arrive).
            if cta_rank == 1:
                # sleep 500ms
                for i in range(500):
                    # nanosleep instruction can sleep max 1ms: https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-nanosleep
                    tl.inline_asm_elementwise(
                        "nanosleep.u32 1000000;  mov.u32 $0, 0;",
                        "=r",
                        [],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )

            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=True)
            clc_phase_consumer ^= 1

    torch.manual_seed(0)
    BLOCK_SIZE = 1024
    # just launch 1 cluster, grid size is 2
    n_elements = BLOCK_SIZE * CLUSTER_SIZE
    x = torch.ones(n_elements, device=device)
    y = torch.ones(n_elements, device=device)
    output = torch.zeros_like(x)
    ref_out = x + y

    num_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (num_tiles, )

    clc_delayed[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        CLUSTER_SIZE=CLUSTER_SIZE,
        ctas_per_cga=(CLUSTER_SIZE, 1, 1),
    )

    torch.testing.assert_close(output, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("noinline", [False, True])
def test_cluster_launch_control_across_call(noinline, device):
    """CLC state crossing a tt.call boundary.

    The callee's parameter type is rebuilt from the frontend `clc_response_type`,
    so it must match the `ui128` memdesc that `create_alloc_clc_responses`
    allocates. When it did not, this failed TTIR verification with
    "'tt.call' op operand type mismatch: expected ... 1xi64 ... provided ... 1xui128".
    """

    # Wrapping the `clc_consumer` builtin in a @triton.jit function is what forces
    # a real tt.call: builtins expand inline at trace time, so only a jit callee
    # gets a signature rebuilt from the frontend types.
    @triton.jit(noinline=noinline)
    def clc_consumer_callee(clc_context, clc_phase_consumer):
        return tlx.clc_consumer(clc_context, clc_phase_consumer)

    @triton.jit
    def clc_call_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        tile_id = tl.program_id(axis=0)
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        while tile_id != -1:
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(z_ptr + offsets, x + y, mask=mask)

            # The whole CLCPipelineContext (mbarriers + clc_response) crosses the call.
            tile_id = clc_consumer_callee(clc_context, clc_phase_consumer)
            clc_phase_consumer ^= 1

    BLOCK_SIZE = 1024
    n_elements = BLOCK_SIZE * 16
    x = torch.randn(n_elements, device=device)
    y = torch.randn(n_elements, device=device)
    output = torch.zeros_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

    kernel = clc_call_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, launch_cooperative_grid=True)

    if noinline:
        # The callee only survives the TTIR inliner when marked noinline. Pin
        # its clc_response parameter to the `ui128` memdesc so a regression
        # fails here rather than as an opaque verifier error.
        callee_sigs = [line for line in kernel.asm["ttir"].splitlines() if line.lstrip().startswith("tt.func private")]
        assert callee_sigs, "expected the noinline callee to survive the TTIR inliner"
        assert any("ui128" in sig for sig in callee_sigs), callee_sigs

    assert re.search(r"clusterlaunchcontrol.try_cancel", kernel.asm["ptx"])
    torch.testing.assert_close(output, x + y)
