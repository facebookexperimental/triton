"""Examples for constraint-aware autotuning: static IR-based pruning.

These demonstrate the ``triton.autotune`` IR-pruning capability added in this
branch — all **static** (compile only, no kernel launch, no reference output):

* **T4 — artifact (IR/PTX) pruning** via
  ``prune_configs_by={"ir_config_prune": ...}``: each config is compiled
  (``run(warmup=True)`` — no launch) and its TTGIR/PTX is inspected, so configs
  can be kept/dropped by a feature present in the generated code.

* **M1 — static bitwise-equivalence pruning**: layered on the same hook via
  ``bitequiv.equivalence.reduction_equivalence_prune(level)`` — keep only configs
  whose compiled IR reduces in the same order as the reference (first) config.

The T4 filters cover the three required targets at the two IR levels:

* T4-A vectorization  — **PTX** feature selection (keep wide vector mem ops)
* T4-B AutoWS         — **TTGIR** feature selection (keep warp-specialized configs)
* T4-C TMEM_LOAD      — **TTGIR** correctness prune (drop configs that emit the op)

T4-B and T4-C require Blackwell (MMAv5 / AutoWS) and are skipped elsewhere.

Run (needs a CUDA GPU):

    python bitequiv/examples/constraint_pruning_examples.py

Each example prints what was pruned and which config won.
"""
import os
import sys

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Make `bitequiv` importable whether this file is run as a script or imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.equivalence import reduction_equivalence_prune  # noqa: E402


def _is_blackwell():
    """Datacenter Blackwell (sm100/sm103, cc major 10-11) — has MMAv5/tcgen05 with
    a TMEM accumulator and is where Triton's AutoWS pipeline applies."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (10, 11)


# ---------------------------------------------------------------------------
# T4 filter A: vectorization (PTX) — keep only configs that emit wide vector
#              global memory ops (feature selection / specialization).
# ---------------------------------------------------------------------------
def example_vectorization_filter():
    N = 1 << 20
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    def keep_vectorized(config, asm, metadata):
        ptx = asm.get("ptx", "")
        # Wide vectorized loads/stores show up as ld/st.global.v4 / v2 in PTX.
        return ("ld.global.v4" in ptx) or ("st.global.v4" in ptx) or ("ld.global.v2" in ptx)

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
    print("[T4-A] vectorization (PTX ld/st.global.v*) feature selection")
    print(f"     dropped (non-vectorized): "
          f"{sorted(c.kwargs['BLOCK_SIZE'] for c in copy_kernel.pruned_by_ir)}")
    print(f"     winner BLOCK_SIZE={copy_kernel.best_config.kwargs['BLOCK_SIZE']}\n")


# ---------------------------------------------------------------------------
# T4 filter B: AutoWS (TTGIR) — keep only warp-specialized configs.
#
# Real keep/drop on ONE kernel: a persistent TMA matmul whose loop is
# ``tl.range(..., warp_specialize=True)``. With AutoWS enabled
# (``TRITON_USE_META_WS`` / ``knobs.nvidia.use_meta_ws``), the ``FLATTEN=False``
# config warp-specializes (TTGIR gets ``ttg.warp_specialize`` / ``async_task_id``)
# while ``FLATTEN=True`` does NOT — so the filter keeps the former and drops the
# latter. (FLATTEN suppressing WS is the documented behaviour in
# python/test/unit/language/test_autows_addmm.py.)
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_tma_persistent_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FLATTEN: tl.constexpr,
):
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
            a = a_desc.load([offs_am, offs_k])  # [BLOCK_M, BLOCK_K]
            b = b_desc.load([offs_bn, offs_k])  # [BLOCK_N, BLOCK_K]
            acc = tl.dot(a, b.T, acc)
        c_desc.store([offs_am, offs_bn], acc.to(tl.float16))


def example_autows_filter():
    if not _is_blackwell():
        print("[T4-B] AutoWS example skipped — requires Blackwell (MMAv5 + AutoWS)\n")
        return

    M, N, K = 512, 512, 256
    BM, BN, BK, GROUP = 128, 128, 64, 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
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
    kept = matmul.best_config
    print("[T4-B] AutoWS (TTGIR warp_specialize) feature selection")
    print(f"     dropped (non-WS): {[c.kwargs['FLATTEN'] for c in matmul.pruned_by_ir]} (FLATTEN values)")
    print(f"     winner (warp-specialized): FLATTEN={kept.kwargs['FLATTEN']}\n")


# ---------------------------------------------------------------------------
# T4 filter C: TMEM_LOAD (TTGIR) — correctness prune: drop configs that lower to
#              a ``ttng.tmem_load`` op.
#
# Real keep/drop on ONE fp32 matmul: ``input_precision="tf32"`` lowers to the
# Blackwell MMAv5 (tcgen05) path with a TMEM accumulator -> TTGIR has
# ``ttng.tmem_load`` (dropped); ``input_precision="ieee"`` stays on the FMA path
# -> no tmem_load (kept). This exercises the detect-and-prune mechanism on a REAL
# op. NOTE: the actual FA TMEM_LOAD *accuracy bug* repro is not in OSS source, so
# here ``tmem_load`` stands in for "the op a correctness filter would drop" — swap
# in the real repro when available (see report / open questions).
# ---------------------------------------------------------------------------
def tmem_load_correctness_filter(config, asm, metadata):
    """Return False (drop) for any config whose TTGIR contains a TMEM load op."""
    ttgir = asm.get("ttgir", "")
    return "tmem_load" not in ttgir  # keep configs WITHOUT the (hypothetically buggy) op


def example_tmem_load_filter():
    if not _is_blackwell():
        print("[T4-C] TMEM_LOAD example skipped — requires Blackwell (MMAv5/TMEM)\n")
        return

    M = N = K = 256
    BM, BN, BK = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    # PREC="tf32" -> MMAv5 + TMEM accumulator -> emits ttng.tmem_load (dropped);
    # PREC="ieee" -> FMA path -> no tmem_load (kept).
    configs = [triton.Config({"PREC": p}) for p in ("ieee", "tf32")]

    @triton.autotune(configs=configs, key=["M", "N", "K"],
                     prune_configs_by={"ir_config_prune": tmem_load_correctness_filter})
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
    print("[T4-C] TMEM_LOAD (TTGIR ttng.tmem_load) correctness prune")
    print(f"     dropped (emits tmem_load): {[c.kwargs['PREC'] for c in matmul.pruned_by_ir]}")
    print(f"     winner (no tmem_load): PREC={matmul.best_config.kwargs['PREC']}\n")


# ---------------------------------------------------------------------------
# M1 core: STATIC bitwise-equivalence pruning at TTGIR level (the T3 redesign).
# No kernel launch, no reference output: compile each config, read the reduce
# op's data layout from the TTGIR, and keep only configs whose reduction order
# matches the reference (first) config. `num_warps` changes warpsPerCTA along the
# reduce axis -> a different cross-warp tree -> a different bitwise result, which
# the static signature detects.
# ---------------------------------------------------------------------------
def example_static_reduction_equivalence():
    N = 4096
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)

    # First config (num_warps=4) defines the reference reduction order. The second (also num_warps=4)
    # shares it and is kept; num_warps 2 and 8 reduce in a different order and are pruned.
    configs = [
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ]

    # Choose the IR level via `reduction_equivalence_prune("ttgir" | "ptx" | "both")`; it returns an
    # `ir_config_prune` predicate (compile-only, no launch). "ptx" raises until that engine is
    # built — TTGIR works today. Keep a handle to read `.classes` / `.pruned` afterwards.
    prune = reduction_equivalence_prune("ttgir")

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"ir_config_prune": prune})
    @triton.jit
    def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    sum_kernel[(1, )](src, out, N)
    print("[M1] static reduction-order equivalence prune (level=ttgir, no launch, no reference output)")
    print(f"     equivalence classes seen: {len(prune.classes)}")
    print(f"     pruned (different reduction order): "
          f"{sorted(c.num_warps for c in prune.pruned)} (num_warps)")
    print(f"     winner (matches reference order): num_warps={sum_kernel.best_config.num_warps}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("These examples require a CUDA GPU.")
    example_static_reduction_equivalence()
    example_vectorization_filter()
    example_autows_filter()
    example_tmem_load_filter()
