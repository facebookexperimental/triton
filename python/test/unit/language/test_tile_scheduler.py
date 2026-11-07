"""Tests for the unified tile-scheduler stdlib (``triton.language.schedule``).

All four schedules -- ``NonPersistentScheduler``, ``StaticPersistent1DScheduler``,
``DynamicPersistent1DScheduler``, ``ClcTileScheduler`` -- expose the same opaque
loop API (``initialize`` / ``is_valid`` / ``tile_id`` / ``advance``), so a single
persistent loop body works for any schedule and the schedule can be selected as a
``tl.constexpr`` (the autotuning axis).

Two kinds of tests:
- IR-shape tests on the initial TTIR (target-independent, no GPU).
- Execution tests: each tile is visited exactly once, and a matmul is correct
  (require a GPU; CLC additionally requires Blackwell).

See docs/design/triton-clc-tile-scheduler.md and the plan in
.claude/plans/inspired-by-the-clc-mutable-eclipse.md.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.core as tl_core
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton._C.libtriton import ir


def _num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# CLC needs Blackwell; the other three run on any GPU.
_PERSISTENT_SCHEDULES = [
    tl.NonPersistentScheduler,
    tl.StaticPersistent1DScheduler,
    tl.DynamicPersistent1DScheduler,
]
_ALL_SCHEDULES = _PERSISTENT_SCHEDULES + [tl.ClcTileScheduler]


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


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _num_tiles_fn(lowering_args):
    M, N, BLOCK_M, BLOCK_N = lowering_args
    return tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _count_kernel(counts_ptr, M, N, tile_counter, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SCHEDULE: tl.constexpr):
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    lowering_args = (M, N, BLOCK_M, BLOCK_N)
    sched = SCHEDULE.initialize(lowering_args, _num_tiles_fn, tile_counter)
    while sched.is_valid():
        tid = sched.tile_id[0]
        tl.atomic_add(counts_ptr + tid, 1, mask=tid < num_tiles)
        sched = sched.advance()


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, tile_counter,  #
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                   GROUP_M: tl.constexpr, SCHEDULE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    lowering_args = (M, N, BLOCK_M, BLOCK_N)
    sched = SCHEDULE.initialize(lowering_args, _num_tiles_fn, tile_counter)
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
# Host-side launch helper (grid + workspace are user-provided per the design)
# ---------------------------------------------------------------------------
def _launch_config(schedule, num_tiles, device):
    """Return (grid, tile_counter) for a schedule. num_programs == grid size."""
    if schedule in (tl.NonPersistentScheduler, tl.ClcTileScheduler):
        grid = num_tiles
    else:
        grid = min(_num_sms(), num_tiles)
    # dynamic seeds programs 0..grid-1 statically, then claims grid, grid+1, ...
    tile_counter = torch.full((1, ), grid, dtype=torch.int32, device=device)
    return (grid, ), tile_counter


# ---------------------------------------------------------------------------
# Frontend IR-shape tests (no GPU required)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schedule", _ALL_SCHEDULES)
def test_scheduler_forms_while_loop(schedule):
    sig = {"counts_ptr": "*i32", "M": "i32", "N": "i32", "tile_counter": "*i32"}
    ttir = initial_ttir(_count_kernel, sig, {"BLOCK_M": 64, "BLOCK_N": 64, "SCHEDULE": schedule})
    assert "scf.while" in ttir, "every schedule drives an scf.while persistent loop"


def test_static_persistent_uses_num_programs_stride():
    sig = {"counts_ptr": "*i32", "M": "i32", "N": "i32", "tile_counter": "*i32"}
    ttir = initial_ttir(_count_kernel, sig, {"BLOCK_M": 64, "BLOCK_N": 64, "SCHEDULE": tl.StaticPersistent1DScheduler})
    assert "tt.get_num_programs" in ttir, "static persistent advances by num_programs"
    assert "ttng.clc_advance" not in ttir


def test_dynamic_persistent_advances_via_atomic():
    sig = {"counts_ptr": "*i32", "M": "i32", "N": "i32", "tile_counter": "*i32"}
    ttir = initial_ttir(_count_kernel, sig, {"BLOCK_M": 64, "BLOCK_N": 64, "SCHEDULE": tl.DynamicPersistent1DScheduler})
    # Two atomic_rmw: the kernel body's count + the scheduler's tile claim.
    assert ttir.count("tt.atomic_rmw") >= 2, "dynamic persistent claims tiles via an atomic counter"
    assert "ttng.clc_advance" not in ttir


def test_clc_scheduler_emits_advance():
    sig = {"counts_ptr": "*i32", "M": "i32", "N": "i32", "tile_counter": "*i32"}
    ttir = initial_ttir(_count_kernel, sig, {"BLOCK_M": 64, "BLOCK_N": 64, "SCHEDULE": tl.ClcTileScheduler})
    assert "ttng.clc_advance" in ttir, "CLC fetches the next tile via the high-level clc_advance op"


@tl_core._aggregate
class _NoIsValidScheduler(tl.TileScheduler):
    """A schedule that (wrongly) doesn't implement is_valid -- must fail to compile."""
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor

    @triton.jit
    def initialize(lowering_args, num_tiles_fn, tile_counter):
        return _NoIsValidScheduler(tl.program_id(0), tl.to_tensor(0), tl.to_tensor(0))

    @triton.jit
    def advance(self):
        return _NoIsValidScheduler(self._x, self._y, self._z)


_NoIsValidScheduler.tile_id = property(lambda self: tl.tuple([self._x, self._y, self._z]))


def test_is_valid_is_required():
    # TileScheduler's base is_valid static_asserts, so a subclass that fails to
    # implement it must not compile.
    sig = {"counts_ptr": "*i32", "M": "i32", "N": "i32", "tile_counter": "*i32"}
    with pytest.raises(Exception) as excinfo:
        initial_ttir(_count_kernel, sig, {"BLOCK_M": 64, "BLOCK_N": 64, "SCHEDULE": _NoIsValidScheduler})
    # The static_assert message is nested in the compilation error chain.
    chain, err = [], excinfo.value
    while err is not None:
        chain.append(str(err))
        err = err.__cause__ or err.__context__
    assert any("must implement is_valid" in m for m in chain), chain


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("schedule", _PERSISTENT_SCHEDULES)
@pytest.mark.parametrize("M, N", [(64, 64), (256, 320), (1000, 500)])
def test_visits_each_tile_once(schedule, M, N):
    BLOCK_M, BLOCK_N = 64, 64
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    counts = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)
    grid, tile_counter = _launch_config(schedule, num_tiles, "cuda")
    _count_kernel[grid](counts, M, N, tile_counter, BLOCK_M, BLOCK_N, schedule)
    counts = counts.cpu()
    assert counts.min().item() == 1, "some tile was never claimed"
    assert counts.max().item() == 1, "some tile was claimed more than once"


@requires_gpu
@pytest.mark.parametrize("schedule", _PERSISTENT_SCHEDULES)
@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (512, 384, 256), (500, 700, 320)])
def test_matmul(schedule, M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid, tile_counter = _launch_config(schedule, num_tiles, "cuda")
    _matmul_kernel[grid](
        a, b, c, M, N, K, tile_counter,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, schedule)

    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")
@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (512, 384, 256)])
def test_matmul_clc(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid, tile_counter = _launch_config(tl.ClcTileScheduler, num_tiles, "cuda")
    _matmul_kernel[grid](
        a, b, c, M, N, K, tile_counter,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, tl.ClcTileScheduler)

    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# CLC partial work-stealing (num_stealing_ctas)
# ---------------------------------------------------------------------------
@triton.jit
def _clc_owner_kernel(counts_ptr, owner_ptr, NUM_STEALING_CTAS: tl.constexpr):
    """Records which CTA (linear program id, x fastest) ran each tile."""
    sched = tl.clc_tile_scheduler(num_stealing_ctas=NUM_STEALING_CTAS)
    nx = tl.num_programs(0)
    while sched.is_valid():
        tile = sched.tile_id[1] * nx + sched.tile_id[0]
        tl.atomic_add(counts_ptr + tile, 1)
        tl.store(owner_ptr + tile, tl.program_id(1) * nx + tl.program_id(0))
        sched = sched.advance()


def test_clc_partial_steal_predicates_advance():
    sig = {"counts_ptr": "*i32", "owner_ptr": "*i32"}
    ttir = initial_ttir(_clc_owner_kernel, sig, {"NUM_STEALING_CTAS": 132})
    assert "ttng.clc_advance %" in ttir, "num_stealing_ctas passes a steal predicate to clc_advance"
    ttir = initial_ttir(_clc_owner_kernel, sig, {"NUM_STEALING_CTAS": None})
    assert "ttng.clc_advance :" in ttir, "without num_stealing_ctas clc_advance has no predicate"


def test_clc_partial_steal_rejects_negative():
    sig = {"counts_ptr": "*i32", "owner_ptr": "*i32"}
    with pytest.raises(Exception) as excinfo:
        initial_ttir(_clc_owner_kernel, sig, {"NUM_STEALING_CTAS": -1})
    chain, err = [], excinfo.value
    while err is not None:
        chain.append(str(err))
        err = err.__cause__ or err.__context__
    assert any("num_stealing_ctas" in m for m in chain), chain


@pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")
@pytest.mark.parametrize("cluster", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("num_stealing", [None, 0, 1, 7, 64, "num_sms"])
@pytest.mark.parametrize("grid", [(2, 2), (1000, 1), (20000, 1), (64, 300)])
def test_clc_partial_steal_ownership(cluster, num_stealing, grid):
    """Every tile runs exactly once, exactly the first num_stealing_ctas CTAs
    (whole clusters, linear cluster order) steal, and every other CTA runs only
    its own tile."""
    if grid[1] % cluster[1]:
        pytest.skip("grid must be divisible by the cluster shape")
    if num_stealing == "num_sms":
        num_stealing = _num_sms()
    gx, gy = grid
    num_tiles = gx * gy
    counts = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)
    owner = torch.full((num_tiles, ), -1, device="cuda", dtype=torch.int32)
    opts = {"ctas_per_cga": (*cluster, 1)} if cluster != (1, 1) else {}
    _clc_owner_kernel[(gx, gy)](counts, owner, num_stealing, **opts)
    counts, owner = counts.cpu(), owner.cpu()
    assert counts.min().item() == 1, "some tile was never run (dropped)"
    assert counts.max().item() == 1, "some tile was run more than once"
    tiles = torch.arange(num_tiles, dtype=torch.int32)

    def cluster_id(linear):
        x, y = linear % gx, linear // gx
        return x // cluster[0] + (gx // cluster[0]) * (y // cluster[1])

    # A cluster is claimed as a unit: all tiles of one cluster are run by the
    # members of a single cluster.
    tile_cluster, owner_cluster = cluster_id(tiles).long(), cluster_id(owner)
    num_clusters = num_tiles // (cluster[0] * cluster[1])
    lo = torch.full((num_clusters, ), 1 << 30, dtype=torch.int32).scatter_reduce(0, tile_cluster, owner_cluster, "amin")
    hi = torch.full((num_clusters, ), -1, dtype=torch.int32).scatter_reduce(0, tile_cluster, owner_cluster, "amax")
    assert torch.equal(lo, hi), "members of one claimed cluster disagree"
    stolen = owner != tiles
    if num_stealing is None:
        return
    stealers = min(num_stealing // (cluster[0] * cluster[1]), num_clusters)
    stealing_clusters = cluster_id(owner[stolen]).unique()
    assert bool((stealing_clusters < stealers).all()), "a non-stealing CTA ran a tile that was not its own"
    if num_tiles >= 100 * _num_sms() and stealers > 0:
        # With far more tiles than SMs every stealer gets to steal; this also
        # guards against a lowering that silently never steals.
        assert stealing_clusters.numel() == stealers, (stealing_clusters.numel(), stealers)


@triton.jit
def _ws_tma_gemm_kernel(a_desc, b_desc, c_desc, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr,
                        NUM_SMS: tl.constexpr, DP: tl.constexpr, SES: tl.constexpr, TWO_CTAS: tl.constexpr,
                        CLC: tl.constexpr, NUM_STEALING_CTAS: tl.constexpr, owner_ptr=None):
    """autoWS persistent TMA GEMM shaped like Inductor's Blackwell template;
    CLC=False is the static persistent reference."""
    grid_m = tl.cdiv(M, BLOCK_M)
    if TWO_CTAS:
        grid_m = (grid_m + 1) // 2 * 2
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * grid_n
    if CLC:
        sched = tl.clc_tile_scheduler(num_stealing_ctas=NUM_STEALING_CTAS)
        while tl.condition(sched.is_valid(), warp_specialize=True, data_partition_factor=DP,
                           separate_epilogue_store=SES):
            pid_m, pid_n = _compute_pid(sched.tile_id[0], num_pid_in_group, grid_m, GROUP_M)
            if owner_ptr is not None:
                tl.store(owner_ptr + sched.tile_id[0], tl.program_id(0))
            _ws_tma_gemm_tile(a_desc, b_desc, c_desc, pid_m, pid_n, k_tiles, BLOCK_M, BLOCK_N, BLOCK_K,
                              EPILOGUE_SUBTILE, TWO_CTAS)
            sched = sched.advance()
    else:
        for tile_id in tl.range(tl.program_id(0), grid_m * grid_n, NUM_SMS, flatten=False, warp_specialize=True,
                                data_partition_factor=DP, separate_epilogue_store=SES):
            pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, grid_m, GROUP_M)
            _ws_tma_gemm_tile(a_desc, b_desc, c_desc, pid_m, pid_n, k_tiles, BLOCK_M, BLOCK_N, BLOCK_K,
                              EPILOGUE_SUBTILE, TWO_CTAS)


@triton.jit
def _ws_tma_gemm_tile(a_desc, b_desc, c_desc, pid_m, pid_n, k_tiles, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr, TWO_CTAS: tl.constexpr):
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(k_tiles):
        a = a_desc.load([offs_am, ki * BLOCK_K])
        b = b_desc.load([offs_bn, ki * BLOCK_K])
        acc = tl.dot(a, b.T, acc, two_ctas=TWO_CTAS)
    acc = acc.to(c_desc.dtype)
    if EPILOGUE_SUBTILE:
        c0, c1 = tl.split(tl.permute(tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1)))
        c_desc.store([offs_am, offs_bn], c0)
        c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1)
    else:
        c_desc.store([offs_am, offs_bn], acc)


def _launch_ws_tma_gemm(a, b, c, cfg, clc, num_stealing, owner=None):
    from triton.tools.tensor_descriptor import TensorDescriptor

    BM, BN, BK, ES, DP, SES, CTAS, stages = cfg
    M, K = a.shape
    N = b.shape[0]
    a_d = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    b_d = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    c_d = TensorDescriptor(c, c.shape, c.stride(), [BM, BN // (2 if ES else 1)])
    num_sms = _num_sms()
    tiles = triton.cdiv(triton.cdiv(M, BM), CTAS) * CTAS * triton.cdiv(N, BN)
    grid = tiles if clc else min(num_sms // CTAS * CTAS, tiles)
    opts = {"num_stages": stages, "num_warps": 4}
    if CTAS == 2:
        opts["ctas_per_cga"] = (2, 1, 1)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        return _ws_tma_gemm_kernel[(grid, )](a_d, b_d, c_d, M, N, K, BM, BN, BK, 8, ES, num_sms // CTAS * CTAS, DP, SES,
                                             CTAS == 2, clc, num_stealing, owner, **opts)


_WS_GEMM_CONFIGS = [
    # BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_SUBTILE, DP, SES, CTAS, num_stages
    (128, 128, 64, False, 1, True, 1, 3),
    (128, 256, 64, True, 1, True, 1, 3),
    (256, 128, 64, True, 2, True, 1, 3),
    (128, 128, 64, False, 1, False, 1, 3),
    (128, 256, 64, True, 1, True, 2, 3),
    (128, 128, 64, False, 1, False, 2, 3),
]


@pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")
@pytest.mark.parametrize("cfg", _WS_GEMM_CONFIGS)
@pytest.mark.parametrize("M, N, K", [(1024, 2048, 256), (1152, 12800, 128), (6202, 2048, 512), (8192, 8192, 1024)])
@pytest.mark.parametrize("num_stealing", [None, 0, 7, "num_sms-16"])
def test_ws_clc_partial_steal_matches_static(cfg, M, N, K, num_stealing):
    """Partial-steal CLC under Meta autoWS is bit-identical to the static
    persistent kernel and leaves no tile unwritten (NaN-poisoned output)."""
    if num_stealing == "num_sms-16":
        num_stealing = _num_sms() - 16
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").bfloat16()
    b = torch.randn(N, K, device="cuda").bfloat16()
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    ref = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
    _launch_ws_tma_gemm(a, b, ref, cfg, False, None)
    assert not torch.isnan(ref).any()
    torch.testing.assert_close(ref, (a.float() @ b.float().T), atol=1e-3, rtol=1e-3)
    for _ in range(3):
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
        k = _launch_ws_tma_gemm(a, b, c, cfg, True, num_stealing)
        assert "ttg.warp_specialize" in k.asm["ttgir"], "the CLC kernel must be warp-specialized"
        assert torch.equal(c, ref), "partial-steal CLC output differs from static persistent"
    # Which CTA ran each tile: only the first num_stealing CTAs (whole
    # clusters) run tiles other than their own, and they do steal.
    BM, BN, SES, CTAS = cfg[0], cfg[1], cfg[5], cfg[6]
    if CTAS == 2 and SES:
        # The extra side store hits an unrelated AutoWS memory-planner failure
        # for 2-CTA CLC with a separate epilogue store ("N-buffer reuse group:
        # producer and consumer orderings are inconsistent"), with or without
        # partial stealing, fixed by #3811; the SES=False 2-CTA config covers
        # ownership until then.
        return
    num_tiles = triton.cdiv(triton.cdiv(M, BM), CTAS) * CTAS * triton.cdiv(N, BN)
    owner = torch.full((num_tiles, ), -1, device="cuda", dtype=torch.int32)
    k = _launch_ws_tma_gemm(a, b, c, cfg, True, num_stealing, owner)
    assert "ttg.warp_specialize" in k.asm["ttgir"]
    assert torch.equal(c, ref)
    owner = owner.cpu()
    tiles = torch.arange(num_tiles, dtype=torch.int32)
    assert bool((owner >= 0).all())
    stolen_by = owner[owner != tiles]
    stealers = num_tiles if num_stealing is None else num_stealing // CTAS * CTAS
    assert bool((stolen_by < stealers).all()), "a non-stealing CTA ran a tile that was not its own"
    if num_tiles >= 4 * _num_sms() and stealers > 0:
        assert stolen_by.numel() > 0, "no CTA stole a tile"
