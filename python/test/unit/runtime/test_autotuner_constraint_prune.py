"""Tests for constraint-aware autotuning: correctness checking (T3) and
artifact/IR-PTX-based pruning (T4).

This file has two layers:

* **CPU logic tests** (no GPU) drive ``Autotuner.run`` end-to-end with a *mock*
  JIT function so the new control flow (correctness gating, success-rate
  recording, artifact pruning, compile-error handling) is validated
  deterministically without a GPU.
* **GPU end-to-end tests** (skipped without CUDA) exercise the same hooks through
  real Triton compilation + launch, modeled on ``test_autotuner.py``.
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


def is_blackwell():
    # Datacenter Blackwell (sm100/sm103): has MMAv5/tcgen05 (TMEM accumulator) and AutoWS.
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

    ``run`` emulates a copy kernel: it copies the first ``min(BLOCK_SIZE, N)``
    elements of ``src`` into ``dst``. Configs with ``BLOCK_SIZE < N`` therefore
    produce a WRONG (truncated) result, which is exactly what the correctness
    check should catch. With ``warmup=True`` it returns a ``_FakeKernel`` instead
    of running, mirroring ``run(warmup=True)``.
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
        if kwargs.get("warmup", False):
            return _FakeKernel(block_size, num_warps)
        named = dict(zip(self.arg_names, args))
        named.update({k: v for k, v in kwargs.items() if k in self.arg_names})
        dst, src, n = named["dst"], named["src"], named["N"]
        k = min(block_size, n)
        dst.zero_()
        dst[:k] = src[:k]
        return None


def _make_tuner(configs, *, correctness_fn=None, correctness_prune=True, prune_configs_by=None):
    fake = _FakeJIT(["dst", "src", "N", "BLOCK_SIZE"])

    # do_bench gets only kernel_call; recover a deterministic "time" from the
    # block size the fake just ran (smaller block == faster), so the fastest
    # config is the smallest BLOCK_SIZE unless it is pruned.
    def do_bench(kernel_call, quantiles=None):
        kernel_call()
        t = float(fake.last_block_size)
        return [t, t, t]

    tuner = Autotuner(fake, fake.arg_names, configs, key=["N"], reset_to_zero=None, restore_value=["dst"],
                      prune_configs_by=prune_configs_by, do_bench=do_bench, correctness_fn=correctness_fn,
                      correctness_prune=correctness_prune)
    return tuner, fake


def _bs(tuner):
    return tuner.best_config.kwargs["BLOCK_SIZE"]


def _configs():
    return [Config({"BLOCK_SIZE": bs}) for bs in (256, 512, 1024, 2048)]


def test_correctness_prune_excludes_wrong_configs():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()

    def correctness_fn(named):
        return torch.equal(named["dst"], ref)

    tuner, _ = _make_tuner(_configs(), correctness_fn=correctness_fn, correctness_prune=True)
    tuner.run(dst, src, N=N, grid=(1, ))

    # Only BLOCK_SIZE >= N produce a full (correct) copy.
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in tuner.correctness_results.items()}
    assert results == {256: False, 512: False, 1024: True, 2048: True}
    # Fastest *correct* config wins (1024 < 2048), not the globally-fastest 256.
    assert _bs(tuner) == 1024


def test_correctness_record_only_does_not_prune():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()

    tuner, _ = _make_tuner(_configs(), correctness_fn=lambda named: torch.equal(named["dst"], ref),
                           correctness_prune=False)
    tuner.run(dst, src, N=N, grid=(1, ))

    # Results still recorded...
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in tuner.correctness_results.items()}
    assert results[256] is False and results[1024] is True
    # ...but nothing is pruned, so the globally-fastest (wrong) config wins.
    assert _bs(tuner) == 256


def test_correctness_prune_all_fail_raises():
    N = 4096  # larger than every BLOCK_SIZE -> every config is wrong
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()
    tuner, _ = _make_tuner(_configs(), correctness_fn=lambda named: torch.equal(named["dst"], ref),
                           correctness_prune=True)
    with pytest.raises(AutotunerError, match="correctness check"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_artifact_prune_filters_on_ttgir():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)

    # Keep only configs whose TTGIR contains the 1024-wide tile (BLOCK_SIZE=1024).
    def artifact_prune(config, asm, metadata):
        assert set(asm) >= {"ttir", "ttgir", "ptx"}  # real artifact dict is available
        return "tensor<1024xf32>" in asm["ttgir"]

    tuner, _ = _make_tuner(_configs(), prune_configs_by={"artifact_config_prune": artifact_prune})
    tuner.run(dst, src, N=N, grid=(1, ))

    dropped = {c.kwargs["BLOCK_SIZE"] for c in tuner.pruned_by_artifact}
    assert dropped == {256, 512, 2048}
    assert _bs(tuner) == 1024


def test_artifact_prune_on_metadata():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    configs = [Config({"BLOCK_SIZE": 256}, num_warps=nw) for nw in (1, 2, 4, 8)]

    tuner, _ = _make_tuner(configs, prune_configs_by={"artifact_config_prune": lambda c, asm, md: md.num_warps <= 2})
    tuner.run(dst, src, N=N, grid=(1, ))

    kept_warps = tuner.best_config.num_warps
    assert kept_warps <= 2


def test_artifact_prune_all_pruned_raises():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs(), prune_configs_by={"artifact_config_prune": lambda c, asm, md: False})
    with pytest.raises(AutotunerError, match="artifact pruning"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_no_hooks_is_unchanged_behavior():
    """When neither hook is set, the fastest config wins (baseline behavior)."""
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs())
    tuner.run(dst, src, N=N, grid=(1, ))
    assert _bs(tuner) == 256  # globally fastest, nothing pruned
    assert tuner.correctness_results == {}
    assert tuner.pruned_by_artifact == {}


# ---------------------------------------------------------------------------
# Static equivalence pruning (M1): keep only configs whose compiled artifact has
# the SAME equivalence key as the reference (first) config. No kernel launch.
# (The mock's equivalence key stands in for a real TTGIR reduction-order
# signature; here we key on metadata.num_warps.)
# ---------------------------------------------------------------------------
def test_equivalence_prune_keeps_reference_class():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    # First config (num_warps=4) is the reference order; a second num_warps=4 config is equivalent;
    # num_warps 2 and 8 are different orders and must be pruned.
    configs = [
        Config({"BLOCK_SIZE": 256}, num_warps=4),
        Config({"BLOCK_SIZE": 512}, num_warps=4),
        Config({"BLOCK_SIZE": 256}, num_warps=2),
        Config({"BLOCK_SIZE": 256}, num_warps=8),
    ]
    eq = lambda config, asm, md: md.num_warps  # stand-in equivalence key (real: reduction signature)
    tuner, _ = _make_tuner(configs, prune_configs_by={"equivalence_fn": eq})
    tuner.run(dst, src, N=N, grid=(1, ))

    assert tuner.best_config.num_warps == 4  # winner is in the reference (num_warps=4) class
    assert {c.num_warps for c in tuner.pruned_by_equivalence} == {2, 8}  # non-equivalent dropped
    # equivalence_classes is {level: {key: [Config]}}; the equivalence_fn path uses the "custom" level.
    assert len(tuner.equivalence_classes["custom"]) == 3  # keys seen: 4, 2, 8


def test_equivalence_prune_all_same_key_keeps_all():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    configs = [Config({"BLOCK_SIZE": bs}, num_warps=4) for bs in (256, 512, 1024)]
    tuner, _ = _make_tuner(configs, prune_configs_by={"equivalence_fn": lambda c, asm, md: md.num_warps})
    tuner.run(dst, src, N=N, grid=(1, ))

    assert tuner.pruned_by_equivalence == {}  # all share the reference order -> nothing pruned
    assert len(tuner.equivalence_classes["custom"]) == 1
    assert _bs(tuner) == 256  # fastest of the (entirely-kept) class


# ---------------------------------------------------------------------------
# Level-selectable equivalence: prune_configs_by={"equivalence_level": ...,
# "equivalence_checkers": <registry>}. Choose ttgir / ptx / both (or a list);
# multiple levels run as a two-stage pipeline. (Mock checkers stand in for the
# real per-level signatures.)
# ---------------------------------------------------------------------------
def test_equivalence_level_ttgir_keeps_reference():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    configs = [
        Config({"BLOCK_SIZE": 256}, num_warps=4),
        Config({"BLOCK_SIZE": 512}, num_warps=4),
        Config({"BLOCK_SIZE": 256}, num_warps=2),
        Config({"BLOCK_SIZE": 256}, num_warps=8),
    ]
    registry = {"ttgir": lambda config, asm, md: md.num_warps}
    tuner, _ = _make_tuner(configs, prune_configs_by={"equivalence_level": "ttgir", "equivalence_checkers": registry})
    tuner.run(dst, src, N=N, grid=(1, ))

    assert tuner.best_config.num_warps == 4
    assert {c.num_warps for c in tuner.pruned_by_equivalence} == {2, 8}
    assert len(tuner.equivalence_classes["ttgir"]) == 3


def test_equivalence_level_two_stage_pipeline():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    # Stage l1 keys on num_warps; stage l2 keys on BLOCK_SIZE. Reference = first config (warps=4, bs=256).
    configs = [
        Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),  # reference (l1=4, l2=256)
        Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),  # matches both -> kept
        Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),  # l1 matches, l2 differs -> pruned at l2
        Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),  # l1 differs -> pruned at l1
    ]
    registry = {
        "l1": lambda config, asm, md: md.num_warps,
        "l2": lambda config, asm, md: config.kwargs["BLOCK_SIZE"],
    }
    tuner, _ = _make_tuner(configs,
                           prune_configs_by={"equivalence_level": ["l1", "l2"], "equivalence_checkers": registry})
    tuner.run(dst, src, N=N, grid=(1, ))

    assert tuner.best_config.num_warps == 4 and tuner.best_config.kwargs["BLOCK_SIZE"] == 256
    # l1 pruned the num_warps=2 config; l2 pruned the BLOCK_SIZE=512 one.
    reasons = {(c.num_warps, c.kwargs["BLOCK_SIZE"]): r for c, r in tuner.pruned_by_equivalence.items()}
    assert reasons[(2, 256)].startswith("l1:")
    assert reasons[(4, 512)].startswith("l2:")
    assert set(tuner.equivalence_classes) == {"l1", "l2"}
    assert len(tuner.equivalence_classes["l1"]) == 2  # num_warps 4 vs 2
    assert len(tuner.equivalence_classes["l2"]) == 2  # among l1-survivors: bs 256 vs 512


def test_equivalence_level_missing_in_registry_raises():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    # >1 config so the autotuner actually enters pruning (it skips tuning for a single config).
    configs = [Config({"BLOCK_SIZE": 256}, num_warps=4), Config({"BLOCK_SIZE": 512}, num_warps=4)]
    tuner, _ = _make_tuner(
        configs, prune_configs_by={"equivalence_level": "ptx", "equivalence_checkers": {"ttgir": lambda c, a, m: 0}})
    with pytest.raises(AutotunerError, match="not provided"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_equivalence_level_not_implemented_raises():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)

    def not_built(config, asm, metadata):
        raise NotImplementedError("PTX-level not implemented yet")

    configs = [Config({"BLOCK_SIZE": 256}, num_warps=4), Config({"BLOCK_SIZE": 512}, num_warps=4)]
    tuner, _ = _make_tuner(configs,
                           prune_configs_by={"equivalence_level": "ptx", "equivalence_checkers": {"ptx": not_built}})
    with pytest.raises(AutotunerError, match="not available"):
        tuner.run(dst, src, N=N, grid=(1, ))


# ---------------------------------------------------------------------------
# GPU end-to-end tests (real Triton compile + launch)
# ---------------------------------------------------------------------------
@triton.jit
def _sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
    # Single-block partial sum: only correct when BLOCK_SIZE >= N.
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_correctness_prune_partial_sum(device="cuda"):
    N = 1024
    src = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty(1, device=device, dtype=torch.float32)
    ref = src.sum()

    def correctness_fn(named):
        # named["dst"] holds this config's output after one run.
        return torch.allclose(named["dst"], ref, atol=1e-3, rtol=1e-3)

    configs = [triton.Config({"BLOCK_SIZE": bs}) for bs in (256, 512, 1024, 2048)]

    @triton.autotune(configs=configs, key=["N"], restore_value=["dst"], correctness_fn=correctness_fn,
                     correctness_prune=True)
    @triton.jit
    def kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    kernel[(1, )](src, out, N)
    # Winner must be a config that actually sums all N elements.
    assert kernel.best_config.kwargs["BLOCK_SIZE"] >= N
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in kernel.correctness_results.items()}
    assert results[256] is False and results[1024] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_artifact_prune_on_real_ir(device="cuda"):
    N = 1024
    src = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty(1, device=device, dtype=torch.float32)

    seen = {}

    def artifact_prune(config, asm, metadata):
        # Real compiled artifacts must be present and inspectable.
        assert "ttgir" in asm and "ptx" in asm
        assert isinstance(asm["ttgir"], str) and len(asm["ttgir"]) > 0
        seen[config.num_warps] = ("tt." in asm["ttgir"])
        return metadata.num_warps <= 4  # keep only <=4 warp configs

    configs = [triton.Config({"BLOCK_SIZE": 1024}, num_warps=nw) for nw in (1, 2, 4, 8)]

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"artifact_config_prune": artifact_prune})
    @triton.jit
    def kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    kernel[(1, )](src, out, N)
    assert kernel.best_config.num_warps <= 4
    dropped = {c.num_warps for c in kernel.pruned_by_artifact}
    assert 8 in dropped
    assert all(seen.values())  # every inspected TTGIR really contained IR text


# ---------------------------------------------------------------------------
# Blackwell-only T4 filters: AutoWS (TTGIR feature selection) and TMEM_LOAD
# (TTGIR correctness prune). Both fire a real keep AND drop on one kernel.
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

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"artifact_config_prune": keep_ws})
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
    assert {c.kwargs["FLATTEN"] for c in kernel.pruned_by_artifact} == {True}  # non-WS dropped


@pytest.mark.skipif(not is_blackwell(), reason="requires Blackwell (MMAv5/TMEM)")
def test_gpu_tmem_load_correctness_prune(device="cuda"):
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

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"artifact_config_prune": drop_tmem_load})
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
    assert {c.kwargs["PREC"] for c in kernel.pruned_by_artifact} == {"tf32"}  # tmem_load config dropped
