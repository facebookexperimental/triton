"""Blackwell (sm100) skinny GEMM for `tlx.ops.mm` with M <= 64 and a K-major B (the `x @ w.T` linear layer).

At this M the GEMM only streams B, and streaming it at the HBM rate needs ~128 KB of TMA loads in flight per SM.
The WS GEMM's M-tiled schedule cannot provide that, so each CTA instead owns one (N-block, K-slice) of W = B.T
with a deep TMA ring, and the CLUSTER K-slices of an N-block run as one thread-block cluster:

- producer (1 warp): TMA-loads the W tile [BN, BK] and the A tile [BMP, BK] of each K step. A rows past M are
  zero-filled; BMP >= 8 is the tcgen05 MMA minimum N.
- MMA (1 warp): acc[BN, BMP] += W_tile @ A_tile^T in TMEM; each MMA's completion frees its SMEM slot.
- epilogue (default, 4 warps): peers (cluster rank > 0) st.async their FP32 accumulator into the leader's slot
  (DSMEM, counted in bytes on the leader's mbarrier); the leader adds them and stores C^T's tile. There is no
  workspace, counter or second launch.

The kernel is launched with programmatic dependent launch: its prologue overlaps the previous kernel when that
kernel triggers early launch. The producer waits on earlier kernels before its first load and the epilogue before
it writes C, since A, W and C may all be in use by them.
"""

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

MAX_M = 64
MAX_M_UNSPLIT = 32
BLOCK_N = 64
BLOCK_K = 128
_SMEM_BUDGET = 200 * 1024
_MAX_STAGES = 8
_MIN_K_STEPS = 4


@triton.jit
def skinny_mm_kernel(a_desc, w_desc, c_ptr, M, N, stride_cm, BN: tl.constexpr, BK: tl.constexpr, BMP: tl.constexpr,
                     CLUSTER: tl.constexpr, K_ITERS: tl.constexpr, NUM_STAGES: tl.constexpr):
    # Not 2-CTA MMA, but with CLUSTER > 1 the peers arrive on rank 0's red_bar remotely, so entry-block barrier inits
    # must precede the first TMEM alloc so a cluster sync fits between them (the compiler hoists them and errors if
    # it cannot). Not required for CLUSTER == 1. Allocate all barriers first.
    full_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    empty_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    acc_bar = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    if CLUSTER > 1:
        red_bar = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    w_bufs = tlx.local_alloc((BN, BK), tlx.dtype_of(w_desc), NUM_STAGES)
    a_bufs = tlx.local_alloc((BMP, BK), tlx.dtype_of(a_desc), NUM_STAGES)
    acc_tmem = tlx.local_alloc((BN, BMP), tl.float32, 1, tlx.storage_kind.tmem)
    STAGE_BYTES: tl.constexpr = (BN * BK + BMP * BK) * 2
    if CLUSTER > 1:
        redbuf = tlx.local_alloc((BN, BMP), tl.float32, CLUSTER)
        rank = tlx.cluster_cta_rank()
    else:
        rank = 0
    n0 = (tl.program_id(0) // CLUSTER) * BN
    k_base = rank * (K_ITERS * BK)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue
            if CLUSTER > 1:
                if rank == 0:
                    tlx.barrier_expect_bytes(red_bar[0], (CLUSTER - 1) * BN * BMP * 4)
            tlx.barrier_wait(acc_bar[0], 0)
            tl.extra.cuda.gdc_wait()  # C may still be in use by an earlier kernel
            acc = tlx.local_load(acc_tmem[0])
            offs_n = n0 + tl.arange(0, BN)
            offs_m = tl.arange(0, BMP)
            mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
            c_ptrs = c_ptr + offs_m[None, :] * stride_cm + offs_n[:, None]
            if CLUSTER == 1:
                tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
            else:
                if rank == 0:
                    tlx.barrier_wait(red_bar[0], 0)
                    for r in tl.static_range(1, CLUSTER):
                        acc += tlx.local_load(redbuf[r])
                    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
                else:
                    tlx.async_remote_shmem_store(dst=redbuf[rank], src=acc, remote_cta_rank=0, barrier=red_bar[0])

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA
            tlx.barrier_wait(full_bars[0], 0)
            tlx.async_dot(w_bufs[0], tlx.local_trans(a_bufs[0]), acc_tmem[0], use_acc=False, mBarriers=[empty_bars[0]],
                          out_dtype=tl.float32)
            for i in range(1, K_ITERS):
                s = i % NUM_STAGES
                tlx.barrier_wait(full_bars[s], (i // NUM_STAGES) & 1)
                tlx.async_dot(w_bufs[s], tlx.local_trans(a_bufs[s]), acc_tmem[0], use_acc=True,
                              mBarriers=[empty_bars[s]], out_dtype=tl.float32)
            tlx.tcgen05_commit(acc_bar[0])

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA loads
            tl.extra.cuda.gdc_wait()  # A and W may come from earlier kernels
            for i in range(0, K_ITERS):
                s = i % NUM_STAGES
                tlx.barrier_wait(empty_bars[s], ((i // NUM_STAGES) & 1) ^ 1)
                tlx.barrier_expect_bytes(full_bars[s], STAGE_BYTES)
                tlx.async_descriptor_load(w_desc, w_bufs[s], [n0, k_base + i * BK], full_bars[s],
                                          eviction_policy="evict_first")
                tlx.async_descriptor_load(a_desc, a_bufs[s], [0, k_base + i * BK], full_bars[s])
            tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _occupancy_probe(X):
    pass


@functools.lru_cache(maxsize=None)
def _max_active_clusters(cluster):
    """How many clusters of `cluster` CTAs run at once at this kernel's one CTA per SM. A cluster has to fit in one
    GPC, so for large clusters this is below #SMs / cluster (15 clusters of 8 on a 152-SM GB300, not 19)."""
    probe = _occupancy_probe.warmup(torch.empty(1, device="cuda"), grid=(1, ), num_warps=4)
    probe._init_handles()
    return triton.runtime.driver.active.utils.cuOccupancyMaxActiveClusters(probe.function, _SMEM_BUDGET, cluster)


def eligible(a, b, out=None):
    """Whether `a @ b` takes the skinny path: M <= 64, row-major A, and B a column-major view of a contiguous W.

    Above `MAX_M_UNSPLIT` rows the WS GEMM keeps the problems whose N-blocks alone fill the SMs (cluster 1, no K
    split): it is as fast there.
    """
    M, K = a.shape
    N = b.shape[1]
    w = b.T
    if not (M <= MAX_M and a.stride(1) == 1 and (a.stride(0) * a.element_size()) % 16 == 0 and w.is_contiguous()
            and N % BLOCK_N == 0 and K % BLOCK_K == 0 and K // BLOCK_K >= _MIN_K_STEPS and
            (out is None or out.is_contiguous())):
        return False
    return M <= MAX_M_UNSPLIT or heuristic_config(M, N, K)["CLUSTER"] > 1


def _stages(bmp, cluster):
    stage_bytes = (BLOCK_N + bmp) * BLOCK_K * 2
    redbuf_bytes = BLOCK_N * bmp * 4 * cluster if cluster > 1 else 0
    return max(2, min(_MAX_STAGES, (_SMEM_BUDGET - redbuf_bytes) // stage_bytes))


def heuristic_config(M, N, K):
    """Largest power-of-two cluster whose clusters all run at once while keeping >= 4 K steps per CTA."""
    bmp = max(8, triton.next_power_of_2(M))
    n_blocks = N // BLOCK_N
    cluster = 1
    while (cluster * 2 <= 8 and n_blocks <= _max_active_clusters(cluster * 2) and K % (BLOCK_K * cluster * 2) == 0
           and K // (BLOCK_K * cluster * 2) >= _MIN_K_STEPS):
        cluster *= 2
    return dict(BMP=bmp, CLUSTER=cluster, NUM_STAGES=_stages(bmp, cluster))


def _space(M, N, K):
    """The heuristic config plus its cluster neighbours, for `space="full"`."""
    base = heuristic_config(M, N, K)
    configs = [base]
    for cluster in (base["CLUSTER"] // 2, base["CLUSTER"] * 2):
        if 1 <= cluster <= 8 and K % (BLOCK_K * cluster) == 0 and K // (BLOCK_K * cluster) >= _MIN_K_STEPS:
            configs.append(dict(base, CLUSTER=cluster, NUM_STAGES=_stages(base["BMP"], cluster)))
    return configs


def _launch(a, w, c, cfg):
    M, K = a.shape
    N = w.shape[0]
    bmp, cluster = cfg["BMP"], cfg["CLUSTER"]
    a_desc = TensorDescriptor(a, [M, K], [a.stride(0), 1], [bmp, BLOCK_K])
    w_desc = TensorDescriptor(w, [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    kwargs = dict(ctas_per_cga=(cluster, 1, 1)) if cluster > 1 else {}
    skinny_mm_kernel[((N // BLOCK_N) * cluster, )](a_desc, w_desc, c, M, N, c.stride(0), BN=BLOCK_N, BK=BLOCK_K,
                                                   BMP=bmp, CLUSTER=cluster, K_ITERS=K // (BLOCK_K * cluster),
                                                   NUM_STAGES=cfg["NUM_STAGES"], num_warps=4, launch_pdl=True, **kwargs)


@functools.lru_cache(maxsize=None)
def _best_config(M, N, K, dtype, device):
    """Pick the fastest of `_space` once per shape, timed in a CUDA graph like the WS GEMM's autotuner."""
    a = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.empty(M, N, device=device, dtype=dtype)
    configs = _space(M, N, K)
    if len(configs) == 1:
        return configs[0]
    times = [triton.testing.do_bench_cudagraph(lambda cfg=cfg: _launch(a, w, c, cfg)) for cfg in configs]
    return configs[times.index(min(times))]


def mm(a, b, c, *, space):
    """`c = a @ b` for an `eligible` problem; `space` is "heuristic", "smoke" or "full"."""
    M, K = a.shape
    N = b.shape[1]
    if space == "full":
        cfg = _best_config(M, N, K, a.dtype, a.device)
    else:
        cfg = heuristic_config(M, N, K)
    _launch(a, b.T, c, cfg)
    return c
