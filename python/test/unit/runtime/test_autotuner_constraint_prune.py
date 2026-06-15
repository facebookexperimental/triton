"""Tests for constraint-aware autotuning: static artifact/IR-PTX-based pruning.

The autotuner exposes a single IR-based pruning hook,
``prune_configs_by={"ir_config_prune": ...}``: it runs *after* each config is
benchmarked, reusing the CompiledKernel the benchmark already produced (no extra
compilation), inspects its TTGIR/PTX, and prunes a rejected config by marking its
timing invalid. Static bitwise-equivalence pruning is layered on this hook in
``bitequiv`` and is unit-tested there (``bitequiv/tests/test_equivalence.py``).

The hook is a **general IR filter** — it prunes by any spec in the compiled IR, not just
bitwise equivalence. This file has two layers:

* **CPU logic tests** (no GPU) drive ``Autotuner.run`` end-to-end with a *mock* JIT
  function so the prune control flow (artifact inspection, keep/drop, compile-error
  handling) is validated deterministically without a GPU.
* **GPU end-to-end tests** (skipped without CUDA) exercise the same hook through real
  Triton compilation, with three general (non-equivalence) showcases:
  - ``test_gpu_vectorization_filter`` — PTX feature selection (wide vector mem ops),
  - ``test_gpu_autows_kloop_warp_specialize`` — TTGIR feature selection (warp
    specialization on a K-loop matmul; Hopper+),
  - ``test_gpu_autows_keeps_warp_specialized`` — same idea on a persistent matmul (Blackwell),
  - ``test_gpu_tmem_load_ir_prune`` — TTGIR op drop (Blackwell).

The equivalence *use* of the same hook is tested in ``bitequiv`` (it lives on
``bitequiv.equivalence``): ``bitequiv/tests/test_constraint_pruning.py`` (end-to-end) and
``bitequiv/tests/test_equivalence.py`` (unit).
"""
import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.autotuner import Autotuner, Config, AutotunerError
from triton.tools.tensor_descriptor import TensorDescriptor

# Disk caching would need a real backend/driver; force it off for the CPU tests.
triton.knobs.autotuning.cache = False


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper():
    # Hopper (sm90, e.g. H100) or newer: has TMA + wgmma and supports warp specialization.
    # CUDA compute capability is (major, minor); major 9 = Hopper, 10/11 = Blackwell.
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    # Datacenter Blackwell (sm100/sm103): adds MMAv5/tcgen05 with a TMEM accumulator (tmem_load).
    return is_cuda() and torch.cuda.get_device_capability()[0] in (10, 11)


# ---------------------------------------------------------------------------
# CPU logic tests (mock JIT function, no GPU required)
# ---------------------------------------------------------------------------
class _FakeKernel:
    """Stand-in for triton.compiler.CompiledKernel: just carries asm + metadata."""

    class _Meta:

        def __init__(self, num_warps):
            self.num_warps = num_warps

    def __init__(self, block_size, num_warps):
        # IR text that varies by config the way real TTGIR encodes tile shapes,
        # so artifact predicates can match on it exactly like real IR.
        self.asm = {
            "ttir": f"module {{ %0 = tt.make_range tensor<{block_size}xi32> }}",
            "ttgir": f"module attributes {{\"ttg.num-warps\" = {num_warps} : i32}} {{ tensor<{block_size}xf32> }}",
            "ptx": f"// block_size={block_size} num_warps={num_warps}\nld.global.f32\n",
        }
        self.metadata = _FakeKernel._Meta(num_warps)


class _FakeJIT:
    """Minimal object that satisfies the bits of JITFunction that Autotuner uses.

    ``run`` always returns a ``_FakeKernel`` (mirroring ``JITFunction.run``, which returns
    the CompiledKernel on a normal launch too) and records the block size so the mock
    ``do_bench`` can derive a deterministic "time" (smaller block == faster). The autotuner
    captures that returned kernel during benchmarking and reads its ``.asm`` for the
    post-bench IR prune.
    """

    def __init__(self, arg_names):
        self.arg_names = arg_names
        self.last_block_size = None

        def _impl():  # base_fn resolution walks .fn until it hits a real function
            return None

        self.fn = _impl

    def run(self, *args, **kwargs):
        block_size = kwargs["BLOCK_SIZE"]
        num_warps = kwargs.get("num_warps", 4)
        self.last_block_size = block_size
        return _FakeKernel(block_size, num_warps)


def _make_tuner(configs, *, prune_configs_by=None):
    fake = _FakeJIT(["dst", "src", "N", "BLOCK_SIZE"])

    # do_bench gets only kernel_call; recover a deterministic "time" from the
    # block size the fake just ran (smaller block == faster), so the fastest
    # config is the smallest BLOCK_SIZE unless it is pruned.
    def do_bench(kernel_call, quantiles=None):
        kernel_call()
        t = float(fake.last_block_size)
        return [t, t, t]

    tuner = Autotuner(fake, fake.arg_names, configs, key=["N"], reset_to_zero=None, restore_value=["dst"],
                      prune_configs_by=prune_configs_by, do_bench=do_bench)
    return tuner, fake


def _bs(tuner):
    return tuner.best_config.kwargs["BLOCK_SIZE"]


def _configs():
    return [Config({"BLOCK_SIZE": bs}) for bs in (256, 512, 1024, 2048)]


def test_ir_prune_filters_on_ttgir():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)

    # Keep only configs whose TTGIR contains the 1024-wide tile (BLOCK_SIZE=1024).
    def ir_prune(config, asm, metadata):
        assert set(asm) >= {"ttir", "ttgir", "ptx"}  # real artifact dict is available
        return "tensor<1024xf32>" in asm["ttgir"]

    tuner, _ = _make_tuner(_configs(), prune_configs_by={"ir_config_prune": ir_prune})
    tuner.run(dst, src, N=N, grid=(1, ))

    dropped = {c.kwargs["BLOCK_SIZE"] for c in tuner.pruned_by_ir}
    assert dropped == {256, 512, 2048}
    assert _bs(tuner) == 1024


def test_ir_prune_on_metadata():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    configs = [Config({"BLOCK_SIZE": 256}, num_warps=nw) for nw in (1, 2, 4, 8)]

    tuner, _ = _make_tuner(configs, prune_configs_by={"ir_config_prune": lambda c, asm, md: md.num_warps <= 2})
    tuner.run(dst, src, N=N, grid=(1, ))

    kept_warps = tuner.best_config.num_warps
    assert kept_warps <= 2


def test_ir_prune_all_pruned_raises():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs(), prune_configs_by={"ir_config_prune": lambda c, asm, md: False})
    with pytest.raises(AutotunerError, match="IR pruning"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_no_hooks_is_unchanged_behavior():
    """When no hook is set, the fastest config wins (baseline behavior)."""
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs())
    tuner.run(dst, src, N=N, grid=(1, ))
    assert _bs(tuner) == 256  # globally fastest, nothing pruned
    assert tuner.pruned_by_ir == {}
    assert tuner.ir_prune_success_rate is None  # hook unused -> never set


def test_ir_prune_success_rate():
    """After IR pruning, the kept fraction is stored on `ir_prune_success_rate`."""
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)

    # Keep only BLOCK_SIZE=1024 -> 1 of 4 configs survive -> 25%.
    tuner, _ = _make_tuner(_configs(),
                           prune_configs_by={"ir_config_prune": lambda c, asm, md: "tensor<1024xf32>" in asm["ttgir"]})
    tuner.run(dst, src, N=N, grid=(1, ))
    assert tuner.ir_prune_success_rate == 0.25

    # Keep everything -> 100%.
    tuner2, _ = _make_tuner(_configs(), prune_configs_by={"ir_config_prune": lambda c, asm, md: True})
    tuner2.run(dst, src, N=N, grid=(1, ))
    assert tuner2.ir_prune_success_rate == 1.0


# ---------------------------------------------------------------------------
# GPU end-to-end tests (real Triton compile)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_ir_prune_on_real_ir(device="cuda"):
    N = 1024
    src = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty(1, device=device, dtype=torch.float32)

    seen = {}

    def ir_prune(config, asm, metadata):
        # Real compiled artifacts must be present and inspectable.
        assert "ttgir" in asm and "ptx" in asm
        assert isinstance(asm["ttgir"], str) and len(asm["ttgir"]) > 0
        seen[config.num_warps] = ("tt." in asm["ttgir"])
        return metadata.num_warps <= 4  # keep only <=4 warp configs

    configs = [triton.Config({"BLOCK_SIZE": 1024}, num_warps=nw) for nw in (1, 2, 4, 8)]

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"ir_config_prune": ir_prune})
    @triton.jit
    def kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    kernel[(1, )](src, out, N)
    assert kernel.best_config.num_warps <= 4
    dropped = {c.num_warps for c in kernel.pruned_by_ir}
    assert 8 in dropped
    assert all(seen.values())  # every inspected TTGIR really contained IR text


# ---------------------------------------------------------------------------
# Showcase 1: vectorization (PTX) — keep only configs that emit wide vector mem ops.
#
# When a thread loads several *contiguous* elements, the backend fuses them into one wide
# vector instruction (e.g. ``ld.global.v4.b32`` = 4 contiguous 32-bit loads in one
# transaction). Width = min(128/dtype_bits, contiguity, mask_alignment, sizePerThread), and
# sizePerThread is bounded by BLOCK_SIZE/(num_warps*32); for fp32/num_warps=4,
# BLOCK_SIZE>=512 -> v4, 256 -> v2, 128 -> scalar. Arch-independent (H100 fine). We record
# the predicate's per-config verdict and assert the prune outcome matches it exactly.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_vectorization_filter(device="cuda"):
    N = 1 << 20
    src = torch.randn(N, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    seen = {}

    def keep_vectorized(config, asm, metadata):
        ptx = asm.get("ptx", "")
        ok = ("ld.global.v4" in ptx) or ("st.global.v4" in ptx) or ("ld.global.v2" in ptx)
        seen[config] = ok
        return ok

    configs = [triton.Config({"BLOCK_SIZE": bs}) for bs in (64, 128, 256, 1024, 4096)]

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"ir_config_prune": keep_vectorized})
    @triton.jit
    def copy_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < N
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    copy_kernel[grid](src, dst, N)

    pruned = set(copy_kernel.pruned_by_ir)
    rejected = {c for c, ok in seen.items() if not ok}
    assert pruned == rejected  # pruned set == configs the predicate rejected
    assert copy_kernel.best_config not in pruned  # winner survived
    assert seen[copy_kernel.best_config] is True  # ...and it is vectorized


# ---------------------------------------------------------------------------
# Showcase 2: AutoWS (TTGIR) — keep only configs that warp-specialize. A TMA matmul whose
# K-loop is tl.range(..., warp_specialize=True) splits warps into a TMA-load producer and an
# MMA consumer that overlap; the TTGIR then has a ``ttg.warp_specialize`` op. Hopper+ (H100
# and newer) via Meta's WS path (knobs.nvidia.use_meta_ws). Kernel mirrors
# test_warp_specialization.py::matmul_tma_ws_kernel. Whether a given config warp-specializes
# can vary by shape/GPU, so we assert the prune outcome matches the predicate's verdict.
# ---------------------------------------------------------------------------
@triton.jit
def _ws_compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _matmul_tma_ws_kernel(a_ptr, b_ptr, c_ptr, a_s0, a_s1, b_s0, b_s1, c_s0, c_s1, M, N, K, KSTAGES: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_s0, a_s1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_s0, b_s1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_s0, c_s1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _ws_compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # warp_specialize=True on the K-loop -> AutoWS partitions warps into TMA-load + MMA roles.
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=KSTAGES):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)
    c_desc.store((off_am, off_bn), accumulator.to(tl.float16))


@pytest.mark.skipif(not is_hopper(), reason="requires Hopper+ (sm90 / H100 or newer)")
def test_gpu_autows_kloop_warp_specialize(device="cuda"):
    M = N = K = 512
    BM = BN = 128
    BK = 64
    GROUP = 8
    KSTAGES = 2
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((N, K), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))

    seen = {}

    def keep_warp_specialized(config, asm, metadata):
        ws = "ttg.warp_specialize" in asm.get("ttgir", "")
        seen[config] = ws
        return ws

    configs = [triton.Config({}, num_warps=nw, num_stages=2) for nw in (4, 8)]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"ir_config_prune": keep_warp_specialized})
    @triton.jit
    def matmul(a_ptr, b_ptr, c_ptr, a_s0, a_s1, b_s0, b_s1, c_s0, c_s1, M, N, K, KSTAGES: tl.constexpr,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               GROUP_SIZE_M: tl.constexpr):
        _matmul_tma_ws_kernel(a_ptr, b_ptr, c_ptr, a_s0, a_s1, b_s0, b_s1, c_s0, c_s1, M, N, K, KSTAGES, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

    grid = lambda meta: (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True  # == TRITON_USE_META_WS=1
        matmul[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), M, N, K,
                     KSTAGES=KSTAGES, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_SIZE_M=GROUP)

    pruned = set(matmul.pruned_by_ir)
    rejected = {cfg for cfg, ws in seen.items() if not ws}
    assert pruned == rejected  # pruned set == configs that did NOT warp-specialize
    assert matmul.best_config not in pruned  # winner survived
    assert seen[matmul.best_config] is True  # ...and it warp-specializes (ttg.warp_specialize in TTGIR)


# ---------------------------------------------------------------------------
# Pre-existing Blackwell AutoWS test (kept): a persistent TMA matmul where FLATTEN=False
# warp-specializes (kept) and FLATTEN=True does not (dropped).
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_tma_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, FLATTEN: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=True,
                            disallow_acc_multi_buffer=True, separate_epilogue_store=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([pid_m * BLOCK_M, offs_k])
            b = b_desc.load([pid_n * BLOCK_N, offs_k])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16))


@pytest.mark.skipif(not is_blackwell(), reason="requires Blackwell (AutoWS + MMAv5)")
def test_gpu_autows_keeps_warp_specialized(device="cuda"):
    M = N = 512
    K = 256
    BM = BN = 128
    BK = 64
    GROUP = 8
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((N, K), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))
    a_desc = TensorDescriptor(a, [M, K], [K, 1], [BM, BK])
    b_desc = TensorDescriptor(b, [N, K], [K, 1], [BN, BK])
    c_desc = TensorDescriptor(c, [M, N], [N, 1], [BM, BN])

    def keep_ws(config, asm, metadata):
        ttgir = asm.get("ttgir", "")
        return ("ttg.warp_specialize" in ttgir) or ("async_task_id" in ttgir)

    # FLATTEN=False warp-specializes (kept); FLATTEN=True does not (dropped).
    configs = [triton.Config({"FLATTEN": flat}, num_warps=4, num_stages=2) for flat in (False, True)]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"ir_config_prune": keep_ws})
    @triton.jit
    def kernel(a_desc, b_desc, c_desc, M, N, K, FLATTEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
        _matmul_tma_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_SMS, FLATTEN)

    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)), )
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True  # == TRITON_USE_META_WS=1
        kernel[grid](a_desc, b_desc, c_desc, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP,
                     NUM_SMS=NUM_SMS)

    assert kernel.best_config.kwargs["FLATTEN"] is False  # the warp-specialized config won
    assert {c.kwargs["FLATTEN"] for c in kernel.pruned_by_ir} == {True}  # non-WS dropped


@pytest.mark.skipif(not is_blackwell(), reason="requires Blackwell (MMAv5/TMEM)")
def test_gpu_tmem_load_ir_prune(device="cuda"):
    M = N = K = 256
    BM = BN = 128
    BK = 64
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)

    def drop_tmem_load(config, asm, metadata):
        return "tmem_load" not in asm.get("ttgir", "")  # keep configs WITHOUT the op

    # PREC="tf32" -> MMAv5 + TMEM accumulator -> emits ttng.tmem_load (dropped);
    # PREC="ieee" -> FMA path -> no tmem_load (kept).
    configs = [triton.Config({"PREC": p}) for p in ("ieee", "tf32")]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"ir_config_prune": drop_tmem_load})
    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr, M, N, K, PREC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            aa = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            bb = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(aa, bb, input_precision=PREC)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    grid = lambda meta: (triton.cdiv(M, BM), triton.cdiv(N, BN))
    kernel[grid](a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)

    assert kernel.best_config.kwargs["PREC"] == "ieee"  # no-tmem_load config won
    assert {c.kwargs["PREC"] for c in kernel.pruned_by_ir} == {"tf32"}  # tmem_load config dropped
