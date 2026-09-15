"""TLX cluster tests -- Hopper and newer."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer
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
def test_ctas_per_cga_regroups_grid_without_multiplying(device):
    """`ctas_per_cga` groups CTAs the grid already has; `num_ctas` spawns more.

    A grid of GRID with a 2-CTA physical cluster must still launch GRID thread
    blocks -- GRID // 2 clusters -- where the same grid with `num_ctas=2`
    launches 2 * GRID. Normalizing `ctas_per_cga` to `num_ctas == 1` is what
    keeps the launcher's `gridX * num_ctas` from multiplying the request, so
    this pins the launch geometry rather than only the recorded metadata.
    """

    @triton.jit
    def count_kernel(counter_ptr):
        tl.atomic_add(counter_ptr, 1)

    GRID = 1000

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    physical = count_kernel[(GRID, )](counter, ctas_per_cga=(2, 1, 1))
    torch.cuda.synchronize()
    assert counter.item() == GRID, "ctas_per_cga must regroup the grid, not multiply it"
    assert physical.metadata.num_ctas == 1
    assert tuple(physical.metadata.cluster_dims) == (2, 1, 1)

    # The contrasting model is asserted through metadata only. Under `num_ctas`
    # the cluster's CTAs cooperate on a single program, so a scalar atomic runs
    # once per program rather than once per thread block: the counter reads GRID
    # under both models and cannot see the extra CTAs. It is a valid probe above
    # precisely because `ctas_per_cga` pins `num_ctas == 1`, which makes programs
    # and thread blocks coincide.
    logical = count_kernel[(GRID, )](counter, num_ctas=2)
    assert logical.metadata.num_ctas == 2
    assert tuple(logical.metadata.cluster_dims) == (1, 1, 1)


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

        # Delay one CTA before it initializes its DSM so a missing cluster
        # barrier is observable: an early remote store would be overwritten.
        if cluster_cta_rank == 1:
            tl.inline_asm_elementwise(
                "nanosleep.u32 1000000; mov.u32 $0, 0;",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        local_init_view = tlx.local_view(local_buff, cluster_cta_rank)
        tlx.local_store(local_init_view, tl.full((1, ), -1.0, tl.float32))

        # Ensure every CTA has entered and initialized its DSM before access.
        tlx.cluster_barrier()
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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster sync")
def test_explicit_cluster_sync_ws(device):
    """Test explicit cluster_barrier() behavior in WS mode.

    The kernel uses two CTAs in a cluster with warp specialization: the
    default task does a remote barrier arrive to signal CTA 1, and a
    partition task waits on the barrier. PTX may still contain additional
    compiler-inserted cluster sync from maybeInsertClusterSync.
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

        # User places explicit cluster sync via cluster_barrier().
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
    # User placed exactly one cluster arrive+wait pair for each task (from cluster_barrier)
    assert ttgir.count("ttng.fence_mbarrier_init_release_cluster") == 1, (
        f"Expected exactly 1 fence_mbarrier_init_release_cluster in TTGIR:\n{ttgir}")
    assert ttgir.count("ttng.cluster_arrive") == 3, (f"Expected exactly 3 cluster_arrive in TTGIR:\n{ttgir}")
    assert ttgir.count("ttng.cluster_wait") == 3, (f"Expected exactly 3 cluster_wait in TTGIR:\n{ttgir}")

    ptx = kernel.asm["ptx"]
    # 1 user fence + 1 compiler-inserted fence from maybeInsertClusterSync
    assert ptx.count("fence.mbarrier_init.release.cluster") == 2, (
        f"Expected exactly 2 fence.mbarrier_init.release.cluster in PTX:\n{ptx}")
    # 3 user cluster_barrier arrives (non-relaxed)
    assert ptx.count("barrier.cluster.arrive.aligned") == 3, (
        f"Expected exactly 3 barrier.cluster.arrive.aligned in PTX:\n{ptx}")
    # 1 compiler-inserted entry arrive + 1 WS non-default warp arrive, both relaxed
    assert ptx.count("barrier.cluster.arrive.relaxed.aligned") == 2, (
        f"Expected exactly 2 barrier.cluster.arrive.relaxed.aligned in PTX:\n{ptx}")
    # 3 user cluster_barrier + 1 compiler-inserted
    assert ptx.count("barrier.cluster.wait.aligned") == 4, (
        f"Expected exactly 4 barrier.cluster.wait.aligned in PTX:\n{ptx}")

    # --- Check correctness ---
    torch.testing.assert_close(y, x)
