"""Tests for the core Triton CLC (Cluster Launch Control) tile scheduler.

CLC is a Blackwell (SM100+) feature. The scheduler (`tl.clc_tile_scheduler`)
emits a single high-level, barrier-free op (`ttng.clc_advance`) into the initial
TTIR; the TTGIR pipeline then lowers it in stages: split into the async-token
form (`clc_try_cancel_async` + `clc_read`), hoist the issue for overlap, and
materialize the token into completion and clustered-reuse mbarriers.

Two kinds of tests:
- IR-shape tests on the initial TTIR (target-independent, no GPU).
- Execution + materialized-TTGIR tests (require a Blackwell GPU).

See docs/design/triton-clc-tile-scheduler.md.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton._C.libtriton import ir


def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_blackwell = pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")


def initial_ttir(fn, signature, constexprs=None):
    """Return the initial TTIR (frontend output, before any lowering pass)."""
    target = GPUTarget("cuda", 100, 32)  # sm100 (Blackwell); frontend is target-independent
    backend = make_backend(target)
    src = ASTSource(fn, signature, constexprs)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    return str(src.make_ir(target, options, codegen_fns, module_map, context))


# Barrier / low-level CLC ops that must NOT appear in the initial (pre-lowering)
# TTIR -- they are introduced by the TTGIR lowering passes.
_LOWERING_ONLY_OPS = [
    "ttng.init_barrier",
    "ttng.wait_barrier",
    "ttng.barrier_expect",
    "ttng.clc_try_cancel",
    "ttng.clc_load_result",
    "ttng.clc_is_canceled",
    "ttng.clc_get_program_id",
    "ttng.clc_try_cancel_async",
    "ttng.clc_read",
]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _clc_count_kernel(counts_ptr, num_tiles):
    sched = tl.clc_tile_scheduler()
    while sched.is_valid():
        tid = sched.tile_id[0]
        tl.atomic_add(counts_ptr + tid, 1, mask=tid < num_tiles)
        sched = sched.advance()


@triton.jit
def _clc_cluster_count_kernel(counts_ptr, steals_ptr, delay_ptr, grid_x: tl.constexpr, grid_y: tl.constexpr):
    sched = tl.clc_tile_scheduler()
    initial_x = tl.program_id(0)
    initial_y = tl.program_id(1)
    initial_z = tl.program_id(2)
    while sched.is_valid():
        pid_x, pid_y, pid_z = sched.tile_id
        linear_pid = pid_x + grid_x * (pid_y + grid_y * pid_z)
        tl.atomic_add(counts_ptr + linear_pid, 1)
        # clc_try_cancel is hoisted above the body. Keep this cluster resident
        # long enough for the asynchronous request to cancel pending work.
        for _ in tl.static_range(64):
            tl.atomic_add(delay_ptr, 1)
        is_stolen = (pid_x != initial_x) | (pid_y != initial_y) | (pid_z != initial_z)
        tl.atomic_add(steals_ptr, 1, mask=is_stolen)
        sched = sched.advance()


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _clc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                       GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    sched = tl.clc_tile_scheduler()
    while sched.is_valid():
        pid_m, pid_n = _compute_pid(sched.tile_id[0], num_pid_in_group, num_pid_m, GROUP_M)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
        sched = sched.advance()


# ---------------------------------------------------------------------------
# Frontend IR-shape tests (no GPU required)
# ---------------------------------------------------------------------------
def test_clc_scheduler_emits_advance_and_seed():
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    assert "ttng.clc_advance" in ttir, "expected the high-level fetch op in the initial TTIR"
    assert "tt.get_program_id" in ttir, "the initial tile should be seeded from program_id"
    assert "scf.while" in ttir, "expected the scheduler to form an scf.while persistent loop"


def test_clc_scheduler_carries_decoded_tile():
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    assert "(i1, i32, i32, i32) -> (i1, i32, i32, i32)" in ttir, \
        "the scheduler loop must carry the decoded tile (is_valid, x, y, z)"


def test_clc_initial_ttir_has_no_lowering_ops():
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    for op in _LOWERING_ONLY_OPS:
        assert op not in ttir, f"{op} must not appear in the high-level (pre-lowering) TTIR"


# ---------------------------------------------------------------------------
# Execution tests (require a Blackwell GPU)
# ---------------------------------------------------------------------------
@requires_blackwell
@pytest.mark.parametrize("num_tiles", [1, 7, 133, 1000])
def test_clc_scheduler_visits_each_tile_once(num_tiles):
    # Every tile in the grid must be claimed exactly once across all CTAs.
    counts = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)
    _clc_count_kernel[(num_tiles, )](counts, num_tiles)
    counts = counts.cpu()
    assert counts.min().item() == 1, "some tile was never claimed"
    assert counts.max().item() == 1, "some tile was claimed more than once"


@requires_blackwell
@pytest.mark.parametrize(
    "grid,cluster",
    [
        ((8192, 1, 1), (2, 1, 1)),
        ((2048, 2, 2), (2, 2, 2)),
    ],
)
def test_clc_cluster_scheduler_visits_each_tile_once(grid, cluster):
    # The grid is deliberately much larger than the machine so at least one
    # pending physical cluster is canceled and processed by a running cluster.
    num_tiles = grid[0] * grid[1] * grid[2]
    counts = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)
    steals = torch.zeros(1, device="cuda", dtype=torch.int32)
    delay = torch.zeros(1, device="cuda", dtype=torch.int32)
    _clc_cluster_count_kernel[grid](counts, steals, delay, grid[0], grid[1], ctas_per_cga=cluster, launch_cluster=True)
    counts = counts.cpu()
    assert counts.min().item() == 1, "some clustered tile was never claimed"
    assert counts.max().item() == 1, "some clustered tile was claimed more than once"
    assert steals.item() > 0, "test grid did not exercise a successful cluster steal"


@requires_blackwell
@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (512, 512, 256), (1000, 1000, 500)])
def test_clc_gemm(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _clc_matmul_kernel[(num_tiles, )](
        a, b, c, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)


@requires_blackwell
def test_clc_materialized_ttgir():
    # After the TTGIR lowering passes the high-level ops are gone and replaced by
    # the low-level try_cancel / barrier form.
    counts = torch.zeros(8, device="cuda", dtype=torch.int32)
    kernel = _clc_count_kernel[(8, )](counts, 8)
    ttgir = kernel.asm["ttgir"]
    for op in ("ttng.clc_advance", "ttng.clc_try_cancel_async", "ttng.clc_read"):
        assert op not in ttgir, f"{op} should be lowered away by the TTGIR passes"
    for op in ("ttng.clc_try_cancel", "ttng.clc_load_result", "ttng.barrier_expect", "ttng.wait_barrier"):
        assert op in ttgir, f"expected {op} in the materialized TTGIR"


@requires_blackwell
def test_clc_cluster_materializes_multicast_request():
    counts = torch.zeros(1024, device="cuda", dtype=torch.int32)
    steals = torch.zeros(1, device="cuda", dtype=torch.int32)
    delay = torch.zeros(1, device="cuda", dtype=torch.int32)
    kernel = _clc_cluster_count_kernel[(1024, 1, 1)](counts, steals, delay, 1024, 1, ctas_per_cga=(2, 1, 1),
                                                     launch_cluster=True)
    ptx = kernel.asm["ptx"]
    assert "multicast::cluster::all" in ptx
    assert "mapa.shared::cluster" in ptx
    assert "mbarrier.arrive.shared::cluster" in ptx
