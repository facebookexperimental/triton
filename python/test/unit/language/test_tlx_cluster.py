import re
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_custer_cta_rank(device):

    @triton.jit
    def test_cta_0_kernel(
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # without multi-cta cluster launch, this test does not validate much except
        # the fact that the IR lowering flow works
        cta_id = tlx.cluster_cta_rank()
        tl.store(output_ptr + offsets, cta_id, mask=mask)

    tensor_size = 32
    # init with 1, expected to be filled with 0
    output = torch.ones(tensor_size, dtype=torch.int32, device=device)
    kernel = test_cta_0_kernel[(1, )](output, tensor_size, tensor_size, num_warps=1)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvg.cluster_id") == 1

    torch.cuda.synchronize()
    expected_output = torch.zeros(tensor_size, dtype=torch.int32, device=device)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper/Blackwell")
def test_cluster_dims(device):

    @triton.jit
    def test_kernel():
        pid = tl.program_id(axis=0)
        if pid == 0:
            return

    k = kernel = test_kernel[(2, )](ctas_per_cga=(2, 1, 1))
    assert kernel.metadata.ctas_per_cga == (2, 1, 1)
    assert ('"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32'
            in k.asm["ttgir"])


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper/Blackwell for clusters")
def test_cluster_size_1d(device):

    @triton.jit
    def cluster_size_kernel(out_ptr, GRID_SIZE_X: tl.constexpr, GRID_SIZE_Y: tl.constexpr):
        size = tlx.cluster_size_1d()
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_z = tl.program_id(2)
        offset = pid_x + GRID_SIZE_X * (pid_y + GRID_SIZE_Y * pid_z)
        tl.store(out_ptr + offset, size)

    GRID_SIZE = (10, 8, 12)
    out = torch.full(GRID_SIZE, -1, device=device, dtype=torch.int32)
    cluster_size_kernel[GRID_SIZE](out, GRID_SIZE[0], GRID_SIZE[1], ctas_per_cga=(2, 1, 3))
    assert torch.all(out == 6)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper/Blackwell for DSM")
def test_remote_shmem_store(device):

    @triton.jit
    def remote_shmem_store_kernel(
        x,
        y,
    ):
        local_buff = tlx.local_alloc((1, ), tl.float32, 2)
        cluster_cta_rank = tlx.cluster_cta_rank()
        remote_store_view = tlx.local_view(local_buff, cluster_cta_rank ^ 1)
        offset = tl.arange(0, 1) + cluster_cta_rank
        value = tl.load(x + offset) + (cluster_cta_rank + 1) * 100

        tlx.remote_shmem_store(
            dst=remote_store_view,
            src=value,
            remote_cta_rank=cluster_cta_rank ^ 1,
        )
        tlx.cluster_barrier()
        local_load_view = tlx.local_view(local_buff, cluster_cta_rank)
        remote_value = tlx.local_load(local_load_view)
        tl.store(y + offset, remote_value)

    x = torch.empty((2, ), device=device, dtype=torch.float32)
    x[0] = 42.0
    x[1] = 43.0
    y = torch.empty((2, ), device=device, dtype=torch.float32)
    remote_shmem_store_kernel[(2, )](x, y, ctas_per_cga=(2, 1, 1))
    assert y[1] == 142.0 and y[0] == 243.0


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_async_remote_shmem_store(num_ctas, device):
    """Test that remote_shmem_store correctly aggregates 2D data across multiple CTAs."""

    @triton.jit
    def remote_store_sum_kernel(
        input_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NUM_CTAS: tl.constexpr,
    ):
        # Configure the number of CTAs participating in reduction
        BLOCK_N: tl.constexpr = triton.cdiv(N, NUM_CTAS)

        # Allocate NUM_CTAS buffers in shared memory, each with shape (BLOCK_M,)
        # to hold a 1D vector of float32 values
        local_buffs = tlx.local_alloc((BLOCK_M, ), tl.float32, NUM_CTAS)

        # Allocate barriers for synchronization across CTAs
        # Each non-zero CTA will use a barrier to signal when its data is written
        barriers = tlx.alloc_barriers(num_barriers=NUM_CTAS)

        # CTA 0 expects to receive (NUM_CTAS - 1) tiles from other CTAs
        # Each tile is BLOCK_M * sizeof(float32) bytes
        for i in tl.static_range(1, NUM_CTAS):
            tlx.barrier_expect_bytes(barriers[i], BLOCK_M * tlx.size_of(tl.float32))

        # Synchronize all CTAs before starting computation
        tlx.cluster_barrier()

        # Get the rank of this CTA within the cluster
        cta_rank = tlx.cluster_cta_rank()

        # Each CTA processes its portion of the input data (2D tile)
        # Layout: each CTA gets a different BLOCK_N columns
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = cta_rank * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load 2D tile: (BLOCK_M, BLOCK_N)
        offsets = offs_m[:, None] * N + offs_n[None, :]
        data = tl.load(input_ptr + offsets)

        # Compute sum over this tile along N dimension, resulting in shape [BLOCK_M]
        local_sum = tl.sum(data, axis=1)

        # Non-zero CTAs: send their 2D tile to CTA 0's shared memory asynchronously
        if cta_rank != 0:
            tlx.async_remote_shmem_store(dst=local_buffs[cta_rank],  # Destination buffer in CTA 0's shared memory
                                         src=local_sum,  # Source 2D tensor from this CTA
                                         remote_cta_rank=0,  # Target CTA is CTA 0
                                         barrier=barriers[cta_rank],  # Signal barrier when write completes
                                         )

        # CTA 0: aggregate all tiles and write final result
        if cta_rank == 0:
            # Start with CTA 0's own local sum
            final_sum = local_sum

            # Wait for each non-zero CTA to write its data, then accumulate
            for i in tl.static_range(1, NUM_CTAS):
                tlx.barrier_wait(barriers[i], phase=0)  # Wait for CTA i's data
                final_sum += tlx.local_load(local_buffs[i])  # Accumulate CTA i's sum

            # Write the final aggregated sum to output
            offs_m = tl.arange(0, BLOCK_M)
            tl.store(output_ptr + offs_m, final_sum)

    torch.manual_seed(0)
    M = 64
    N = 256
    input_tensor = torch.randn((M, N), dtype=torch.float32, device=device)
    output = torch.zeros(M, dtype=torch.float32, device=device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), META["NUM_CTAS"])

    kernel = remote_store_sum_kernel[grid](input_tensor, output, M=M, N=N, BLOCK_M=64, NUM_CTAS=num_ctas, num_warps=1,
                                           ctas_per_cga=(1, num_ctas, 1))

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_remote_shmem_store") == 1

    expected = torch.sum(input_tensor, dim=1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_remote_shmem_copy(device):
    """Test that async_remote_shmem_copy bulk-copies local SMEM to a remote CTA's SMEM."""

    @triton.jit
    def remote_copy_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
    ):
        # Each CTA allocates: a 1-slot shared memory buffer and 1 mbarrier.
        smem_buf = tlx.local_alloc((N, ), tl.float32, 1)
        barriers = tlx.alloc_barriers(num_barriers=1)

        cta_rank = tlx.cluster_cta_rank()

        # CTA 1 (receiver): initialize barrier to expect N float32 bytes.
        # barrier_expect_bytes also counts as the mbarrier arrive, so no
        # separate arrive is needed.
        if cta_rank == 1:
            tlx.barrier_expect_bytes(barriers[0], N * tlx.size_of(tl.float32))

        # CTA 0 (sender): load from global memory into registers, store to
        # local SMEM, then bulk-copy that SMEM to CTA 1's SMEM and signal
        # CTA 1's mbarrier.
        if cta_rank == 0:
            offs = tl.arange(0, N)
            vals = tl.load(input_ptr + offs)
            tlx.local_store(smem_buf[0], vals)
            tlx.fence("async_shared")
            # Copy local buffer to CTA 1
            tlx.async_remote_shmem_copy(
                dst=smem_buf[0],
                src=smem_buf[0],
                remote_cta_rank=1,
                barrier=barriers[0],
            )

        # CTA 1 (receiver): wait for the copy to complete, read SMEM, store
        # to output.
        if cta_rank == 1:
            tlx.barrier_wait(barriers[0], phase=tl.constexpr(0))
            result = tlx.local_load(smem_buf[0])
            offs = tl.arange(0, N)
            tl.store(output_ptr + offs, result)

    N = 1024
    input_tensor = torch.rand(N, dtype=torch.float32, device=device)
    output = torch.zeros(N, dtype=torch.float32, device=device)

    kernel = remote_copy_kernel[(2, )](input_tensor, output, N=N, num_warps=1, ctas_per_cga=(2, 1, 1))

    ttgir = kernel.asm["ttgir"]
    ptx = kernel.asm["ptx"]
    assert ttgir.count("ttg.async_remote_shmem_copy") == 1
    assert ptx.count("fence.proxy.async.shared::cta") == 1
    assert ptx.count("mapa.shared::cluster") == 2
    assert ptx.count("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes") == 1

    torch.testing.assert_close(output, input_tensor)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster support")
def test_ctas_per_cga(device):
    """Test launching kernels with 2x1x1 ctas_per_cga (CUDA cluster dimensions) in autotune config."""

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE": 64},
                num_warps=4,
            ),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def simple_kernel_clustered(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(x_ptr + offsets, offsets, mask=mask)

    x = torch.zeros(256, dtype=torch.float32, device=device)
    num_blocks = triton.cdiv(256, 64)

    # Launch with autotuned config containing ctas_per_cga=(2,1,1)
    kernel = simple_kernel_clustered[(num_blocks, )](x, 256, ctas_per_cga=(2, 1, 1))

    # verify kernel launch cluster
    assert kernel.metadata.ctas_per_cga == (2, 1, 1), (
        f"expecting ctas_per_cga to be (2, 1, 1), got {kernel.metadata.ctas_per_cga}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}")


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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_atomic_add_cga(device):
    """Test that atomic operations work correctly in CGA (cluster) kernels.

    In a 2-CTA cluster, both CTAs should execute the atomic_add,
    resulting in a counter value of 2 (one increment per CTA).
    """

    @triton.heuristics(values={"ctas_per_cga": lambda args: (2, 1, 1)})
    @triton.jit
    def atomic_add_cga_kernel(counter_ptr, out_ptr, NUM_CTAS: tl.constexpr):
        pid = tl.program_id(0)
        cta_rank = tlx.cluster_cta_rank()

        # Each CTA's thread 0 should atomic_add on the same counter
        val = tl.atomic_add(counter_ptr, 1, sem="relaxed")

        # Store the returned value and CTA rank for verification
        tl.store(out_ptr + pid * 2, val)
        tl.store(out_ptr + pid * 2 + 1, cta_rank)

    grid_size = 2  # 2 CTAs in the cluster
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.full((grid_size * 2, ), -1, dtype=torch.int32, device=device)

    atomic_add_cga_kernel[(grid_size, )](counter, out, NUM_CTAS=grid_size)

    # Check the results
    counter_val = counter.item()

    # Each CTA should have executed the atomic, so counter should be 2
    assert counter_val == grid_size, f"Expected counter={grid_size}, got {counter_val}"

    # Check that both CTAs participated
    atomic_vals = []
    cta_ranks = []
    for i in range(grid_size):
        atomic_val = out[i * 2].item()
        cta_rank = out[i * 2 + 1].item()
        atomic_vals.append(atomic_val)
        cta_ranks.append(cta_rank)

    # The atomic values should be 0 and 1 (in some order)
    # showing that both CTAs executed the atomic
    assert set(atomic_vals) == {0, 1}, f"Expected atomic values {{0, 1}}, got {set(atomic_vals)}"

    # CTA ranks should be 0 and 1
    assert set(cta_ranks) == {0, 1}, f"Expected CTA ranks {{0, 1}}, got {set(cta_ranks)}"


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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster sync")
def test_explicit_cluster_sync_ws(device):
    """Test that explicit cluster_barrier() in WS mode sets the
    tlx.explicit_cluster_sync module attribute and suppresses heuristic
    cluster sync insertion.  The kernel uses two CTAs in a cluster with
    warp specialization: the default task does a remote barrier arrive
    to signal CTA 1, and a partition task waits on the barrier.
    """

    @triton.jit
    def explicit_cluster_sync_ws_kernel(
        x_ptr,
        y_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        # need this fence to make mbar init visible to cluster
        tlx.fence_mbarrier_init_cluster()
        cta_rank = tlx.cluster_cta_rank()

        # Explicit cluster sync placed by user – compiler must not auto-insert
        with tlx.async_tasks():
            with tlx.async_task("default"):
                # This has to be inside default task, because at WS entry there'd be task syncs
                tlx.cluster_barrier()

                # CTA 0 arrives on remote barrier in CTA 1
                if cta_rank == 0:
                    tlx.barrier_arrive(bar=bars[0], remote_cta_rank=1)

            with tlx.async_task(num_warps=2):
                # This has to be in async task because trunk path belongs to default task
                tlx.cluster_barrier()
                offsets = tl.arange(0, BLOCK_SIZE) + cta_rank * BLOCK_SIZE
                data = tl.load(x_ptr + offsets)
                # CTA 1 waits for the remote arrive from CTA 0
                if cta_rank == 1:
                    tlx.barrier_wait(bars[0], phase=0)
                tl.store(y_ptr + offsets, data)
            with tlx.async_task(num_warps=2):
                # idle warps also have to participate in cluster wide sync
                tlx.cluster_barrier()

    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE * 2, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    kernel = explicit_cluster_sync_ws_kernel[(2, )](
        x,
        y,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        ctas_per_cga=(2, 1, 1),
    )

    ttgir = kernel.asm["ttgir"]
    # The Fixup pass should have detected the user cluster_barrier and set this
    assert "tlx.explicit_cluster_sync = true" in ttgir, (
        f"Expected tlx.explicit_cluster_sync module attr in TTGIR:\n{ttgir}")
    # User placed exactly one cluster arrive+wait pair for each task (from cluster_barrier)
    assert ttgir.count("ttng.fence_mbarrier_init_release_cluster") == 1, (
        f"Expected exactly 1 fence_mbarrier_init_release_cluster in TTGIR:\n{ttgir}")
    assert ttgir.count("ttng.cluster_arrive") == 3, (f"Expected exactly 3 cluster_arrive in TTGIR:\n{ttgir}")
    assert ttgir.count("ttng.cluster_wait") == 3, (f"Expected exactly 3 cluster_wait in TTGIR:\n{ttgir}")

    ptx = kernel.asm["ptx"]
    assert ptx.count("fence.mbarrier_init.release.cluster") == 1, (
        f"Expected exactly 1 fence.mbarrier_init.release.cluster in PTX:\n{ptx}")
    # The user's cluster_barrier should produce exactly one
    # barrier.cluster.arrive.aligned and one barrier.cluster.wait.aligned
    # No extra heuristic ones should be inserted
    assert ptx.count("barrier.cluster.arrive.aligned") == 3, (
        f"Expected exactly 3 barrier.cluster.arrive.aligned in PTX:\n{ptx}")
    assert ptx.count("barrier.cluster.wait.aligned") == 3, (
        f"Expected exactly 3 barrier.cluster.wait.aligned in PTX:\n{ptx}")

    # --- Check correctness ---
    torch.testing.assert_close(y, x)
