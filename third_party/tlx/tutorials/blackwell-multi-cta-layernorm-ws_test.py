"""
Multi-CTA Layer Normalization — Persistent + Warp-Specialized (2-WG) variant
============================================================================

This is the **narrow-N fast path** for the multi-CTA LayerNorm in
``blackwell-multi-cta-layernorm_test.py``. The plain (non-WS) kernel there
launches one cluster per row-block and does load → reduce → normalize → store
inline in a single warp group. That is optimal when the problem is large and
bandwidth-bound (wide N), but for *narrow* N it is **latency/overhead-bound**:
the cross-CTA reduction (remote shared-memory exchange + mbarrier round trip)
stalls the single warp group, and per-row-block cluster launch/teardown is pure
overhead.

This file fixes both with two classic techniques, and **only** these:

1. **Persistence.** We launch a small, fixed number of clusters (≈ SMs /
   reduction-CTAs) and each cluster loops over many row-blocks. CTA/cluster
   setup is paid once, not once per row-block.

2. **2 warp groups (load / compute).** A dedicated 1-warp LOAD task prefetches
   the next X tile into a depth-2 SMEM ring *while* the default COMPUTE task is
   busy in the high-latency cross-CTA reduction of the current tile. The load
   latency is hidden behind the reduction latency.

Measured on B200 (fp16), median-of-3, vs the non-WS baseline. The win at
narrow N scales with M (i.e. with the number of row-blocks each persistent
cluster loops over — more iterations means more load-prefetch/reduction overlap
and more amortized cluster setup):

    N=16384, M=1024..2048 : ~1.03-1.04x  (≈ parity; loop too short to overlap)
    N=16384, M=4096       : ~1.18x  WIN
    N=16384, M=8192       : ~1.26x  WIN
    N=65536               : bandwidth-bound; persistence cuts concurrency -> no win

Net: routing narrow N here is a clear win at large M and harmless (parity) at
small M, so the dispatcher in the baseline file routes on N alone. (Noise floor
is ~3%, measured from a same-kernel control.)

Why 2 warp groups and NOT 3 (separate store)?
---------------------------------------------
A third "store" warp group was prototyped and **measured 0.37-0.85x vs this
2-WG version** — i.e. consistently worse. The store here is an ordinary pointer
``tl.store``, and isolating it into its own warp group costs:
  * a *second* full-tile SMEM channel (compute → store), which doubles SMEM and
    forces a shallow ring → serializes compute against store;
  * a register round-trip (the store WG must ``local_load`` Y back from SMEM
    before ``tl.store``, since a pointer store reads from registers);
  * with no async drain to overlap (plain global stores issue cheaply inline).
A dedicated store warp group pays off only for *async* (TMA ``descriptor_store``)
stores, where the store reads directly from SMEM and drains in the background.
For a pointer ``tl.store``, keep the store in the compute warp group (2-WG).

Key TLX features demonstrated:
- Persistent grid with a per-cluster row-block loop
- Warp specialization via ``tlx.async_tasks`` / ``tlx.async_task``
- Producer/consumer SMEM ring with ``full``/``empty`` mbarrier handshake
- Cross-CTA reduction via ``tlx.async_remote_shmem_store`` + ``barrier_expect_bytes``
- The "no register-tensor capture across the WS boundary" rule: each task
  recomputes its own offsets/masks; only X crosses (via the SMEM ring).
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor.runtime.triton_compat import libdevice
from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = (
    torch.cuda.get_device_properties(0).multi_processor_count
    if torch.cuda.is_available()
    else 0
)


@triton.jit
def _cross_cta_sum(
    x,
    cta_rank,
    redbuf,
    barrier,
    phase,
    BLOCK_SIZE_M: tl.constexpr,
    num_reduction_ctas: tl.constexpr,
):
    """Sum ``x`` along N across all reduction CTAs in the cluster.

    Each CTA reduces its own N-slice locally, publishes the partial to every
    peer's ``redbuf`` slot via an async remote shared-memory store, waits on the
    mbarrier (the caller pre-armed it with ``barrier_expect_bytes``), then sums
    all per-CTA partials back in rank order. ``redbuf`` is indexed by the
    *writer's* rank so the final accumulation order is deterministic without a
    branch."""
    dtype_x = tlx.dtype_of(x)
    local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
    tlx.local_store(redbuf[cta_rank], local_partial_sum)
    for i in tl.static_range(num_reduction_ctas):
        if cta_rank != i:
            tlx.async_remote_shmem_store(
                dst=redbuf[cta_rank],
                src=local_partial_sum,
                remote_cta_rank=i,
                barrier=barrier,
            )
    tlx.barrier_wait(barrier, phase=phase)
    final_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=dtype_x)
    for i in tl.static_range(num_reduction_ctas):
        final_sum += tlx.local_load(tlx.local_view(redbuf, i))
    return final_sum


# --- Autotuning -------------------------------------------------------------
# Same shape-derived pruning as the non-WS baseline: BLOCK_SIZE_N, masking flags
# and cp.async legality are computed in the prune hook (we can't use
# @triton.heuristics in triton_pytest targets — Buck bytecode precompilation
# breaks inspect.getsourcelines()).
_ws_configs = [
    triton.Config(
        {
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": 8192,  # placeholder; overwritten in prune
            "num_reduction_ctas": ctas,
            "SHOULD_MASK_ROW": False,
            "SHOULD_MASK_COL": False,
        },
        num_warps=nw,
        ctas_per_cga=(1, ctas, 1),
    )
    # Curated set: on B200 (fp16) the narrow-N sweet spot is consistently
    # num_reduction_ctas=2 with 8 warps; only BLOCK_SIZE_M needs tuning per
    # shape. Keeping this list tight makes the first-call autotune cheap (a
    # broad sweep recompiles the cluster+WS kernel dozens of times).
    for m in [1, 2, 4]
    for nw in [8]
    for ctas in [2]
]


def _prune_ws_configs(configs, named_args, **kwargs):
    N = kwargs["N"]
    M = kwargs["M"]
    pruned = []
    for conf in configs:
        num_ctas = conf.kwargs["num_reduction_ctas"]
        block_size_m = conf.kwargs["BLOCK_SIZE_M"]
        blocksize_n = triton.next_power_of_2(N // num_ctas)
        # Drop configs where rounding leaves a tail CTA with no work.
        if triton.cdiv(N, blocksize_n) != num_ctas:
            continue
        # cp.async requires >= 4 bytes per thread.
        bytes_per_thread = (block_size_m * blocksize_n * 2) // (conf.num_warps * 32)
        if bytes_per_thread < 4:
            continue
        conf.kwargs["BLOCK_SIZE_N"] = blocksize_n
        conf.kwargs["SHOULD_MASK_ROW"] = M % block_size_m != 0
        conf.kwargs["SHOULD_MASK_COL"] = N % blocksize_n != 0
        pruned.append(conf)
    return pruned


@triton.autotune(
    configs=_ws_configs,
    prune_configs_by={"early_config_prune": _prune_ws_configs},
    key=["M", "N"],
)
@triton.jit
def kernel_layernorm_multi_cta_ws(
    X,  # pointer to the input  [M, N]
    Y,  # pointer to the output [M, N]
    W,  # pointer to the weights [N]
    B,  # pointer to the biases  [N]
    Mean_out,  # pointer to the mean [M]
    Rstd_out,  # pointer to the 1/std [M]
    row_stride,  # input row stride
    M,  # number of rows
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # per-CTA N slice = next_pow2(N / num_reduction_ctas)
    num_reduction_ctas: tl.constexpr,
    SHOULD_MASK_ROW: tl.constexpr,
    SHOULD_MASK_COL: tl.constexpr,
):
    rank = tlx.cluster_cta_rank()
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)  # number of persistent clusters along the row axis
    n_row_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    COMPUTE_DTYPE = tl.float32
    NB: tl.constexpr = 2  # X prefetch ring depth (depth-2 => one tile in flight)
    SHOULD_MASK: tl.constexpr = SHOULD_MASK_ROW or SHOULD_MASK_COL

    # --- Shared state, allocated once and reused across the row-block loop ---
    # X ring: the LOAD task fills slot s, the COMPUTE task drains it.
    x_buffer = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), X.dtype.element_ty, NB)
    # Cross-CTA reduction scratch: one row-partial slot per reduction CTA. The
    # mean and variance reductions get SEPARATE buffers: with a single shared
    # buffer, a peer that finishes the mean reduction first could issue its
    # variance remote store into our slot before we finish reading it as a mean
    # partial (the two reductions are not cross-CTA-fenced from each other).
    redbuf_mean = tlx.local_alloc((BLOCK_SIZE_M, 1), COMPUTE_DTYPE, num_reduction_ctas)
    redbuf_var = tlx.local_alloc((BLOCK_SIZE_M, 1), COMPUTE_DTYPE, num_reduction_ctas)
    # red[0] for the mean reduction, red[1] for the variance reduction; both are
    # reused every iteration with a per-iteration phase bit.
    red = tlx.alloc_barriers(num_barriers=2)
    # Ring handshake: x_full[s] producer->consumer, x_empty[s] consumer->producer.
    x_full = tlx.alloc_barriers(num_barriers=NB)
    x_empty = tlx.alloc_barriers(num_barriers=NB)
    # Bytes each CTA receives from its (num_reduction_ctas - 1) peers per reduction.
    expect_bytes: tl.constexpr = (
        BLOCK_SIZE_M * tlx.size_of(COMPUTE_DTYPE) * (num_reduction_ctas - 1)
    )

    with tlx.async_tasks():
        # ============================ LOAD task =============================
        # One warp. Prefetches X tiles into the depth-2 ring so the next tile's
        # global->SMEM transfer overlaps the current tile's cross-CTA reduction.
        # NOTE: register tensors cannot be captured across the warp-spec
        # boundary, so this task recomputes its own column offsets/masks.
        with tlx.async_task(num_warps=1, num_regs=24):
            col = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_col = col < N if SHOULD_MASK_COL else None
            for it in range(pid, n_row_blocks, nprog):
                _it = (it - pid) // nprog  # 0,1,2,... local iteration index
                slot = _it % NB
                phase = (_it // NB) & 1
                row = it * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                x_ptrs = X + (row[:, None] * row_stride) + col[None, :]
                rw_mask = None
                if SHOULD_MASK:
                    mr = (
                        row < M
                        if SHOULD_MASK_ROW
                        else tl.full([BLOCK_SIZE_M], True, tl.int1)
                    )
                    mc = (
                        mask_col
                        if mask_col is not None
                        else tl.full([BLOCK_SIZE_N], True, tl.int1)
                    )
                    rw_mask = mr[:, None] & mc[None, :]
                other = 0.0 if SHOULD_MASK else None
                # Wait until the consumer has drained this slot (skip on the very
                # first use of each slot via the inverted phase).
                tlx.barrier_wait(x_empty[slot], phase ^ 1)
                tok = tlx.async_load(
                    x_ptrs, tlx.local_view(x_buffer, slot), mask=rw_mask, other=other
                )
                tlx.async_load_commit_group([tok])
                tlx.async_load_wait_group(0)
                tlx.fence_async_shared()  # make the cp.async write visible to the consumer
                tlx.barrier_arrive(x_full[slot], 1)

        # ========================== COMPUTE task ===========================
        # Default warp group. Drains the ring, runs the two cross-CTA reductions
        # (mean, then variance), normalizes, and stores Y/mean/rstd inline.
        # The store stays here (2-WG, not 3-WG) — see the module docstring.
        with tlx.async_task("default"):
            col = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_col = col < N if SHOULD_MASK_COL else None
            # W/B are reused for every row-block of this cluster — hoist them.
            w = tl.load(W + col, mask=mask_col).to(COMPUTE_DTYPE)
            b = tl.load(B + col, mask=mask_col).to(COMPUTE_DTYPE)
            for it in range(pid, n_row_blocks, nprog):
                _it = (it - pid) // nprog
                slot = _it % NB
                phase = (_it // NB) & 1
                phase_red = _it & 1  # red[] barriers are depth-1, reused each iter
                row = it * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                y_ptrs = Y + (row[:, None] * row_stride) + col[None, :]
                mask_row = None
                if SHOULD_MASK_ROW:
                    mask_row = row < M
                elif SHOULD_MASK_COL:
                    mask_row = tl.full([BLOCK_SIZE_M], True, dtype=tl.int1)
                rw_mask = None
                if SHOULD_MASK:
                    mr = (
                        mask_row
                        if mask_row is not None
                        else tl.full([BLOCK_SIZE_M], True, tl.int1)
                    )
                    mc = (
                        mask_col
                        if mask_col is not None
                        else tl.full([BLOCK_SIZE_N], True, tl.int1)
                    )
                    rw_mask = mr[:, None] & mc[None, :]

                # Arm both reduction mbarriers before consuming X so peers' remote
                # stores (which may already be arriving) are accounted for.
                tlx.barrier_expect_bytes(red[0], size=expect_bytes)
                tlx.barrier_expect_bytes(red[1], size=expect_bytes)

                tlx.barrier_wait(x_full[slot], phase)
                x = tlx.local_load(tlx.local_view(x_buffer, slot)).to(COMPUTE_DTYPE)
                tlx.barrier_arrive(
                    x_empty[slot], 1
                )  # release the slot to the LOAD task

                # mean = sum(x) / N  (across all reduction CTAs)
                s = _cross_cta_sum(
                    x,
                    rank,
                    redbuf_mean,
                    red[0],
                    phase_red,
                    BLOCK_SIZE_M,
                    num_reduction_ctas,
                )
                mean = s / N
                if SHOULD_MASK:
                    x_minus_mean = tl.where(rw_mask, x - mean, 0.0)
                else:
                    x_minus_mean = x - mean
                # var = sum((x-mean)^2) / N  (across all reduction CTAs); uses a
                # separate buffer from the mean reduction (see redbuf alloc).
                s2 = _cross_cta_sum(
                    x_minus_mean * x_minus_mean,
                    rank,
                    redbuf_var,
                    red[1],
                    phase_red,
                    BLOCK_SIZE_M,
                    num_reduction_ctas,
                )
                rstd = libdevice.rsqrt(s2 / N + eps)

                tl.store(
                    Mean_out + row, tl.reshape(mean, (BLOCK_SIZE_M,)), mask=mask_row
                )
                tl.store(
                    Rstd_out + row, tl.reshape(rstd, (BLOCK_SIZE_M,)), mask=mask_row
                )
                y = ((x - mean) * rstd) * w + b
                tl.store(y_ptrs, tl.cast(y, y_ptrs.dtype.element_ty), mask=rw_mask)


def multi_cta_layernorm_ws(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Persistent + warp-specialized (2-WG) multi-CTA LayerNorm forward.

    Same signature/returns as ``multi_cta_layernorm`` in the baseline file; this
    is the narrow-N fast path it dispatches to. Config (BLOCK_SIZE_M,
    num_reduction_ctas, num_warps) is chosen by autotune.
    """
    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    # Persistent grid: a small fixed number of clusters, each looping over
    # row-blocks (never more clusters than row-blocks). Use a closure capturing
    # `m` so concurrent callers don't race on shared state.
    def grid(meta):
        nrb = triton.cdiv(m, meta["BLOCK_SIZE_M"])
        num_clusters = min(nrb, max(1, NUM_SMS // meta["num_reduction_ctas"]))
        return (num_clusters, meta["num_reduction_ctas"])

    kernel_layernorm_multi_cta_ws[grid](
        X=x,
        Y=out,
        W=weight,
        B=bias,
        Mean_out=mean,
        Rstd_out=rstd,
        row_stride=x.stride(0),
        M=m,
        N=n,
        eps=eps,
    )

    out = out.view(original_shape)
    return out, mean, rstd


def _torch_layernorm_impl(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


torch_layernorm = torch.compile(_torch_layernorm_impl)


@pytest.mark.skipif(
    not is_blackwell(), reason="Requires Blackwell GPU for multi-CTA support"
)
@pytest.mark.parametrize(
    "M,N", [(4, 16384), (1152, 16384), (1024, 16384), (1024, 32768)]
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_op(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    weight = torch.randn(N, device=DEVICE, dtype=dtype)
    bias = torch.randn(N, device=DEVICE, dtype=dtype)
    eps = 1e-5

    output_torch = torch_layernorm(x, weight, bias, eps)
    output_triton, _, _ = multi_cta_layernorm_ws(x, weight, bias, eps)

    rtol = atol = 1e-2 if dtype == torch.float16 else 1e-3
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    print(f"[M={M}, N={N}, dtype={dtype}] Max difference: {max_diff}")
    assert torch.allclose(
        output_torch, output_triton, rtol=rtol, atol=atol
    ), f"Output mismatch: max diff = {max_diff}"


if __name__ == "__main__":
    if is_blackwell():
        test_op(M=1024, N=16384, dtype=torch.float16)
        print("OK")
    else:
        print("Skipping: requires Blackwell GPU")
