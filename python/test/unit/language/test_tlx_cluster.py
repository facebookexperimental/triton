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
    assert ttgir.count("nvgpu.cluster_id") == 1

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
    assert kernel.metadata.cluster_dims == (2, 1, 1)
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
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
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
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster_dims to be (2, 1, 1), got {kernel.metadata.cluster_dims}")

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
