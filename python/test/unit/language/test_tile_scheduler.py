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
