"""Explicit cluster-scope barrier tests -- Hopper and newer."""
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer
import triton.language.extra.tlx as tlx


@pytest.mark.parametrize("scope", ["warp", 0, None])
def test_alloc_barriers_invalid_scope(scope):
    semantic = SimpleNamespace(builder=SimpleNamespace(options=SimpleNamespace(backend_name="cuda", arch="sm100")))
    with pytest.raises(AssertionError, match="scope must be 'cta' or 'cluster'"):
        tlx.alloc_barriers(tl.constexpr(1), scope=scope, _semantic=semantic)


@pytest.mark.parametrize("backend, arch", [("cuda", "sm80"), ("hip", "gfx950"), ("cpu", 0)])
def test_alloc_barriers_cluster_scope_unsupported_target(backend, arch):
    semantic = SimpleNamespace(builder=SimpleNamespace(options=SimpleNamespace(backend_name=backend, arch=arch)))
    with pytest.raises(AssertionError, match="cluster barrier scope requires NVIDIA sm90"):
        tlx.alloc_barriers(tl.constexpr(1), scope="cluster", _semantic=semantic)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster scope")
def test_alloc_barriers_nonconstant_scope(device):

    @triton.jit
    def nonconstant_scope_kernel(output_ptr):
        tlx.alloc_barriers(1, scope=tl.load(output_ptr))

    output = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError, match="scope must be 'cta' or 'cluster'"):
        nonconstant_scope_kernel[(1, )](output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster scope")
@pytest.mark.parametrize("scope", ["default", "cta", "cluster"])
@pytest.mark.parametrize("cluster_model", ["single", "logical", "physical"])
@pytest.mark.parametrize("num_barriers", [1, 3])
def test_alloc_barriers_cluster_scope(device, scope, cluster_model, num_barriers):

    @triton.jit
    def scope_kernel(output_ptr, SCOPE: tl.constexpr, NUM_BARRIERS: tl.constexpr):
        if SCOPE == "default":
            barriers = tlx.alloc_barriers(NUM_BARRIERS, arrive_count=3)
        else:
            barriers = tlx.alloc_barriers(NUM_BARRIERS, arrive_count=3, scope=SCOPE)
        for i in tl.static_range(NUM_BARRIERS):
            tlx.barrier_arrive(barriers[i], arrive_count=3)
            tlx.barrier_wait(barriers[i], phase=0)
        if tlx.cluster_cta_rank() == 0:
            tl.store(output_ptr, 1)
        tlx.cluster_barrier()

    options = {"num_warps": 4}
    if cluster_model == "logical":
        options["num_ctas"] = 2
    elif cluster_model == "physical":
        options["ctas_per_cga"] = (2, 1, 1)
    output = torch.zeros(1, dtype=torch.int32, device=device)
    grid = (2, ) if cluster_model == "physical" else (1, )
    kernel = scope_kernel[grid](output, SCOPE=scope, NUM_BARRIERS=num_barriers, **options)
    ttgir = kernel.asm["ttgir"]
    ptx = kernel.asm["ptx"]
    assert ttgir.count("ttng.init_barrier") == num_barriers
    assert ttgir.count("bCluster = true") == (num_barriers if scope == "cluster" else 0)
    fence = "fence.mbarrier_init.release.cluster"
    if scope == "cluster" and cluster_model != "single":
        assert ptx.count(fence) == 1
        assert ptx.index("mbarrier.init") < ptx.index(fence)
        assert ptx.index(fence) < ptx.index("mbarrier.arrive")
    else:
        assert fence not in ptx
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster scope")
@pytest.mark.parametrize("cluster_position", [0, 1, 2])
def test_alloc_barriers_mixed_scope_bootstrap(device, cluster_position):

    @triton.jit
    def mixed_scope_kernel(output_ptr, CLUSTER_POSITION: tl.constexpr):
        bar0 = tlx.alloc_barriers(1, scope="cluster" if CLUSTER_POSITION == 0 else "cta")
        bar1 = tlx.alloc_barriers(1, scope="cluster" if CLUSTER_POSITION == 1 else "cta")
        bar2 = tlx.alloc_barriers(1, scope="cluster" if CLUSTER_POSITION == 2 else "cta")
        tlx.barrier_arrive(bar0[0])
        tlx.barrier_wait(bar0[0], phase=0)
        tlx.barrier_arrive(bar1[0])
        tlx.barrier_wait(bar1[0], phase=0)
        tlx.barrier_arrive(bar2[0])
        tlx.barrier_wait(bar2[0], phase=0)
        tl.store(output_ptr + tlx.cluster_cta_rank(), 1)

    output = torch.zeros(2, dtype=torch.int32, device=device)
    kernel = mixed_scope_kernel[(2, )](output, CLUSTER_POSITION=cluster_position, ctas_per_cga=(2, 1, 1), num_warps=4)
    ttgir = kernel.asm["ttgir"]
    ptx = kernel.asm["ptx"]
    assert ttgir.count("ttng.init_barrier") == 3
    assert ttgir.count("bCluster = true") == 1
    fence = "fence.mbarrier_init.release.cluster"
    arrive = "barrier.cluster.arrive.relaxed.aligned"
    wait = "barrier.cluster.wait.aligned"
    assert ptx.count("mbarrier.init.shared::cta.b64") == 3
    assert ptx.count(fence) == 1
    assert ptx.count(arrive) == 1
    assert ptx.count(wait) == 1
    assert ptx.rindex("mbarrier.init.shared::cta.b64") < ptx.index(fence)
    assert ptx.index(fence) < ptx.index(arrive) < ptx.index(wait)
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_cluster_scope_async_remote_shmem_store(num_ctas, device):
    """Cluster scope supplies initialization visibility for remote completions."""

    @triton.jit
    def remote_store_sum_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                                NUM_CTAS: tl.constexpr):
        BLOCK_N: tl.constexpr = triton.cdiv(N, NUM_CTAS)
        local_buffs = tlx.local_alloc((BLOCK_M, ), tl.float32, NUM_CTAS)
        barriers = tlx.alloc_barriers(num_barriers=NUM_CTAS, scope="cluster")
        for i in tl.static_range(1, NUM_CTAS):
            tlx.barrier_expect_bytes(barriers[i], BLOCK_M * tlx.size_of(tl.float32))
        cta_rank = tlx.cluster_cta_rank()
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = cta_rank * BLOCK_N + tl.arange(0, BLOCK_N)
        data = tl.load(input_ptr + offs_m[:, None] * N + offs_n[None, :])
        local_sum = tl.sum(data, axis=1)
        if cta_rank != 0:
            tlx.async_remote_shmem_store(dst=local_buffs[cta_rank], src=local_sum, remote_cta_rank=0,
                                         barrier=barriers[cta_rank])
        if cta_rank == 0:
            final_sum = local_sum
            for i in tl.static_range(1, NUM_CTAS):
                tlx.barrier_wait(barriers[i], phase=0)
                final_sum += tlx.local_load(local_buffs[i])
            tl.store(output_ptr + offs_m, final_sum)
        # Keep remote shared-memory owners alive until aggregation completes.
        tlx.cluster_barrier()

    torch.manual_seed(0)
    input_tensor = torch.randn((64, 256), dtype=torch.float32, device=device)
    output = torch.zeros(64, dtype=torch.float32, device=device)
    grid = lambda META: (triton.cdiv(64, META["BLOCK_M"]), META["NUM_CTAS"])
    kernel = remote_store_sum_kernel[grid](input_tensor, output, M=64, N=256, BLOCK_M=64, NUM_CTAS=num_ctas,
                                           num_warps=1, ctas_per_cga=(1, num_ctas, 1))
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_remote_shmem_store") == 1
    assert "ttng.fence_mbarrier_init_release_cluster" not in ttgir
    assert kernel.asm["ptx"].count("fence.mbarrier_init.release.cluster") == (1 if num_ctas > 1 else 0)
    torch.testing.assert_close(output, torch.sum(input_tensor, dim=1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_cluster_scope_async_remote_shmem_copy(device):
    """Bulk-copy completion uses compiler-owned barrier initialization."""

    @triton.jit
    def remote_copy_kernel(input_ptr, output_ptr, N: tl.constexpr):
        smem_buf = tlx.local_alloc((N, ), tl.float32, 1)
        barriers = tlx.alloc_barriers(num_barriers=1, scope="cluster")
        cta_rank = tlx.cluster_cta_rank()
        if cta_rank == 1:
            tlx.barrier_expect_bytes(barriers[0], N * tlx.size_of(tl.float32))
        if cta_rank == 0:
            offs = tl.arange(0, N)
            tlx.local_store(smem_buf[0], tl.load(input_ptr + offs))
            tlx.fence("async_shared")
            tlx.async_remote_shmem_copy(dst=smem_buf[0], src=smem_buf[0], remote_cta_rank=1, barrier=barriers[0])
        if cta_rank == 1:
            tlx.barrier_wait(barriers[0], phase=tl.constexpr(0))
            tl.store(output_ptr + tl.arange(0, N), tlx.local_load(smem_buf[0]))
        # Preserve the sender's source until the receiver has finished using it.
        tlx.cluster_barrier()

    input_tensor = torch.rand(1024, dtype=torch.float32, device=device)
    output = torch.zeros_like(input_tensor)
    kernel = remote_copy_kernel[(2, )](input_tensor, output, N=1024, num_warps=1, ctas_per_cga=(2, 1, 1))
    ttgir = kernel.asm["ttgir"]
    ptx = kernel.asm["ptx"]
    assert ttgir.count("ttg.async_remote_shmem_copy") == 1
    assert "ttng.fence_mbarrier_init_release_cluster" not in ttgir
    assert ptx.count("fence.mbarrier_init.release.cluster") == 1
    assert ptx.index("fence.mbarrier_init.release.cluster") < ptx.index("cp.async.bulk")
    assert ptx.count("fence.proxy.async.shared::cta") == 1
    assert ptx.count("mapa.shared::cluster") == 2
    assert ptx.count("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes") == 1
    torch.testing.assert_close(output, input_tensor)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster sync")
def test_cluster_scope_sync_ws(device):
    """Cluster scope bootstraps default and worker warps without manual init sync."""

    @triton.jit
    def cluster_scope_sync_ws_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1, scope="cluster")
        cta_rank = tlx.cluster_cta_rank()
        with tlx.async_tasks():
            with tlx.async_task("default"):
                if cta_rank == 0:
                    tlx.barrier_arrive(bar=bars[0], remote_cta_rank=1)
            with tlx.async_task(num_warps=2):
                offsets = tl.arange(0, BLOCK_SIZE) + cta_rank * BLOCK_SIZE
                data = tl.load(x_ptr + offsets)
                if cta_rank == 1:
                    tlx.barrier_wait(bars[0], phase=0)
                tl.store(y_ptr + offsets, data)

    x = torch.arange(256, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    kernel = cluster_scope_sync_ws_kernel[(2, )](x, y, BLOCK_SIZE=128, num_warps=4, ctas_per_cga=(2, 1, 1))
    ttgir = kernel.asm["ttgir"]
    assert "ttng.fence_mbarrier_init_release_cluster" not in ttgir
    assert "ttng.cluster_arrive" not in ttgir
    assert "ttng.cluster_wait" not in ttgir
    ptx = kernel.asm["ptx"]
    assert ptx.count("fence.mbarrier_init.release.cluster") == 1
    assert "barrier.cluster.arrive.aligned" not in ptx
    assert ptx.count("barrier.cluster.arrive.relaxed.aligned") == 2
    assert ptx.count("barrier.cluster.wait.aligned") == 1
    torch.testing.assert_close(y, x)
