"""Tests for constraint-aware autotuning: static IR-based config pruning.

These exercise the ``triton.autotune`` IR-pruning hook end-to-end on a real GPU —
all **static** (each config is compiled with ``run(warmup=True)``; no kernel launch
for the prune decision, no reference output):

* **T4 artifact (IR/PTX) pruning** via ``prune_configs_by={"ir_config_prune": ...}``:
  - vectorization (PTX feature selection),
  - AutoWS (TTGIR feature selection, Blackwell),
  - TMEM_LOAD (TTGIR op drop, Blackwell).
* **M1 static bitwise-equivalence pruning** via
  ``bitequiv.equivalence.reduction_equivalence_prune(level)``: keep only configs
  whose compiled IR reduces in the same order as the reference (first) config.

(These were previously runnable scripts under ``bitequiv/examples/``; they are now
assertion-based tests.) Blackwell-only cases self-skip on other hardware.
"""
import os
import sys

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
    _IMPORT_OK = True
except Exception:  # pragma: no cover - torch/triton not importable
    _IMPORT_OK = False

# Make `bitequiv` importable when run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if _IMPORT_OK:
    from bitequiv.equivalence import reduction_equivalence_prune


def is_cuda():
    return _IMPORT_OK and torch.cuda.is_available() and \
        triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    # Datacenter Blackwell (sm100/sm103): MMAv5/tcgen05 (TMEM accumulator) + AutoWS.
    return is_cuda() and torch.cuda.get_device_capability()[0] in (10, 11)


requires_cuda = pytest.mark.skipif(not is_cuda(), reason="requires a CUDA GPU")
requires_blackwell = pytest.mark.skipif(not is_blackwell(), reason="requires Blackwell (MMAv5 / AutoWS)")


# ---------------------------------------------------------------------------
# T4-A: vectorization (PTX) — keep only configs that emit wide vector mem ops.
# We record the predicate's verdict per config and assert the prune outcome
# matches it exactly (mechanism test, independent of which BLOCK_SIZEs vectorize).
# ---------------------------------------------------------------------------
@requires_cuda
def test_vectorization_filter():
    N = 1 << 20
    src = torch.randn(N, device="cuda", dtype=torch.float32)
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
    # The pruned set is exactly the configs the predicate rejected.
    assert pruned == rejected
    # The winner survived the filter (i.e. it is vectorized).
    assert copy_kernel.best_config not in pruned
    assert seen[copy_kernel.best_config] is True


# ---------------------------------------------------------------------------
# T4-B: AutoWS (TTGIR) — keep only warp-specialized configs. Persistent TMA matmul
# whose loop is tl.range(..., warp_specialize=True): FLATTEN=False warp-specializes
# (kept), FLATTEN=True does not (dropped).
# ---------------------------------------------------------------------------
if _IMPORT_OK:

    @triton.jit
    def _matmul_tma_persistent_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
                                  FLATTEN: tl.constexpr):
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
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                acc = tl.dot(a, b.T, acc)
            c_desc.store([offs_am, offs_bn], acc.to(tl.float16))


@requires_blackwell
def test_autows_filter():
    M, N, K = 512, 512, 256
    BM, BN, BK, GROUP = 128, 128, 64, 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    a_desc = TensorDescriptor(a, [M, K], [K, 1], [BM, BK])
    b_desc = TensorDescriptor(b, [N, K], [K, 1], [BN, BK])
    c_desc = TensorDescriptor(c, [M, N], [N, 1], [BM, BN])

    def keep_warp_specialized(config, asm, metadata):
        ttgir = asm.get("ttgir", "")
        return ("ttg.warp_specialize" in ttgir) or ("async_task_id" in ttgir)

    # FLATTEN=False -> warp-specializes (kept); FLATTEN=True -> does not (dropped).
    configs = [triton.Config({"FLATTEN": flat}, num_warps=4, num_stages=2) for flat in (False, True)]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"ir_config_prune": keep_warp_specialized})
    @triton.jit
    def matmul(a_desc, b_desc, c_desc, M, N, K, FLATTEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
        _matmul_tma_persistent_ws(a_desc, b_desc, c_desc, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_SMS, FLATTEN)

    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)), )
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True  # == TRITON_USE_META_WS=1
        matmul[grid](a_desc, b_desc, c_desc, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP,
                     NUM_SMS=NUM_SMS)

    assert matmul.best_config.kwargs["FLATTEN"] is False  # warp-specialized config won
    assert {c.kwargs["FLATTEN"] for c in matmul.pruned_by_ir} == {True}  # non-WS dropped


# ---------------------------------------------------------------------------
# T4-C: TMEM_LOAD (TTGIR) — drop configs that lower to a ttng.tmem_load op.
# fp32 matmul: input_precision="tf32" -> MMAv5 + TMEM accumulator -> emits
# ttng.tmem_load (dropped); "ieee" -> FMA path -> no tmem_load (kept).
# ---------------------------------------------------------------------------
@requires_blackwell
def test_tmem_load_filter():
    M = N = K = 256
    BM, BN, BK = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    def drop_tmem_load(config, asm, metadata):
        return "tmem_load" not in asm.get("ttgir", "")  # keep configs WITHOUT the op

    configs = [triton.Config({"PREC": p}) for p in ("ieee", "tf32")]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"ir_config_prune": drop_tmem_load})
    @triton.jit
    def matmul(a_ptr, b_ptr, c_ptr, M, N, K, PREC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    matmul[grid](a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)

    assert matmul.best_config.kwargs["PREC"] == "ieee"  # no-tmem_load config won
    assert {c.kwargs["PREC"] for c in matmul.pruned_by_ir} == {"tf32"}  # tmem_load config dropped


# ---------------------------------------------------------------------------
# M1: STATIC bitwise-equivalence pruning at TTGIR level. No launch, no reference
# output: compile each config, read the reduce op's data layout, keep only configs
# whose reduction order matches the reference (first) config. num_warps changes
# warpsPerCTA along the reduce axis -> different cross-warp tree -> different bits.
# ---------------------------------------------------------------------------
@requires_cuda
def test_static_reduction_equivalence():
    N = 4096
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)

    # First config (num_warps=4) defines the reference order; the second (num_warps=4)
    # shares it and is kept; num_warps 2 and 8 reduce differently and are pruned.
    configs = [
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ]

    prune = reduction_equivalence_prune("ttgir")

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"ir_config_prune": prune})
    @triton.jit
    def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    sum_kernel[(1, )](src, out, N)

    # Reference (num_warps=4) class kept; num_warps 2 and 8 pruned as non-equivalent.
    assert sum_kernel.best_config.num_warps == 4
    assert {c.num_warps for c in prune.pruned} == {2, 8}
    assert len(prune.classes) == 3  # equivalence classes seen: num_warps 4, 2, 8
