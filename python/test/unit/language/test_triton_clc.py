"""Tests for the core Triton CLC (Cluster Launch Control) tile scheduler.

CLC is a Blackwell (SM100+) feature. The scheduler (`tl.clc_tile_scheduler`)
emits high-level, barrier-free ops (`clc_init` / `clc_advance` + the pure
`clc_is_canceled` / `clc_get_program_id`) into the initial TTIR; a later compiler
lowering pass materializes the response buffer, mbarrier, phase, seed, and the
issue/wait overlap. Until that lowering lands these tests are IR-only: they check
the shape of the initial TTIR (which is target-independent and needs no GPU).
"""
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton._C.libtriton import ir


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


# Barrier / buffer ops that must NOT appear in the initial TTIR -- they belong to
# the deferred lowering, not the high-level scheduler.
_BARRIER_OPS = [
    "ttng.init_barrier",
    "ttng.wait_barrier",
    "ttng.barrier_expect",
    "ttng.clc_try_cancel",
    "ttng.clc_load_result",
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
# IR-shape tests (no GPU required)
# ---------------------------------------------------------------------------
def test_clc_scheduler_emits_high_level_ops():
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    for op in ("ttng.clc_init", "ttng.clc_advance", "ttng.clc_is_canceled", "ttng.clc_get_program_id"):
        assert op in ttir, f"expected {op} in initial TTIR"
    assert "scf.while" in ttir, "expected the scheduler to form an scf.while persistent loop"


def test_clc_scheduler_carries_only_i128():
    # The persistent loop should carry exactly one value: the 128-bit CLC result.
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    assert "scf.while" in ttir
    assert "(i128) -> i128" in ttir, "the scheduler loop must carry only the i128 CLC result"


def test_clc_initial_ttir_has_no_barriers():
    # Barrier/buffer plumbing must live entirely in the deferred lowering.
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    for op in _BARRIER_OPS:
        assert op not in ttir, f"{op} must not appear in the high-level TTIR"


def test_clc_drops_unused_tile_dims():
    # Only tile_id[0] is read, so exactly one dimension should be decoded.
    ttir = initial_ttir(_clc_count_kernel, {"counts_ptr": "*i32", "num_tiles": "i32"})
    assert ttir.count("ttng.clc_get_program_id") == 1, "unused tile dimensions should not be decoded"


def test_clc_gemm_emits_high_level_ops():
    sig = {
        "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16", "M": "i32", "N": "i32", "K": "i32", "stride_am": "i32",
        "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32"
    }
    constexprs = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}
    ttir = initial_ttir(_clc_matmul_kernel, sig, constexprs)
    assert "ttng.clc_init" in ttir and "ttng.clc_advance" in ttir
    assert "scf.while" in ttir
    for op in _BARRIER_OPS:
        assert op not in ttir, f"{op} must not appear in the high-level TTIR"


# ---------------------------------------------------------------------------
# Execution is deferred until the CLC lowering pass lands (see
# memory: clc-ttir-ops-followup). The GEMM/work-stealing correctness tests will
# be re-enabled then.
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="CLC lowering (ttng.clc_init/clc_advance -> barrier ops) not implemented yet")
def test_clc_gemm_execution():
    pass
