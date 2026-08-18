"""TLX GEMM tutorial for AMD MI300X (gfx942 / CDNA3).

Every other AMD GEMM tutorial here targets gfx950 (CDNA4) and stages operands
with *direct-to-LDS* copies. On CDNA3 the right choice is the register-staged
path instead:

    global --tl.load--> VGPR --tlx.local_store--> LDS --tlx.local_load--> MFMA

What is left to tune, and what this kernel does:

* **Size the ring to the 64 KB CDNA3 LDS budget**, not CDNA4's 160 KB. The gfx950
  kernels' 256x256 tile with 64-deep K and two buffers wants ~128 KB and cannot
  be made to fit; even 256x128x64 needs 96 KB. ``lds_bytes`` computes the
  footprint and ``_prune_configs`` drops anything over budget before it is
  compiled.
* **Remap program ids across the 8 XCDs.** MI300X dispatches consecutive
  workgroups round-robin over the chiplets, which scatters tiles that should be
  sharing B columns in one L2. Measured, this is **neutral** here -- 0.86x aten
  with the remap vs 0.87x without at 4096^3 and 0.79x vs 0.79x at 8192^3, i.e.
  inside the noise -- because the GROUP_M swizzle already captures that reuse.
  It is kept as the standard MI300X grid transform; set ``NUM_XCDS=1`` in
  ``_configs()`` to A/B it.
* **Autotune the tile per (M, N, K).** With 304 CUs no single tile serves both
  ends: a 1024^2 output decomposes into only 16 tiles of 256x256 and leaves the
  chip idle (0.35x aten), while the 64x64 tile that wins there collapses to
  0.54x at 4096^3. Picking per shape is the single largest effect in the whole
  kernel, and it is worth ~1.15x vs 0.35x at 1024^3. ``_prune_configs`` drops
  tiles that overflow the LDS budget or outrun K before they are compiled.
* ``matrix_instr_nonkdim=16`` -- gfx942 fp16 MFMA is ``16x16x16`` / ``32x32x8``,
  half the K of CDNA4's ``16x16x32`` / ``32x32x16``.

Exposes ``matmul`` for the correctness suite (``testing/test_correctness.py``) and
the perf script (``testing/test_amd_gemm_gfx942_perf.py``, which compares against
aten).
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# MI300X: 8 XCDs, 304 CUs. Consecutive program ids are dispatched round-robin
# across the XCDs, so the remap below undoes that to restore tile locality.
NUM_XCDS = 8

# Per-workgroup LDS on CDNA3. The launcher checks configs against this so an
# oversized tile fails with a clear message instead of an out-of-resources error.
CDNA3_LDS_BYTES = 64 * 1024


@triton.jit
def _xcd_remap(pid, grid_mn, num_xcds: tl.constexpr):
    """Undo the hardware's round-robin XCD dispatch of consecutive program ids.

    Workgroup ``pid`` runs on XCD ``pid % num_xcds``. Left alone, tiles that
    should share B columns land on different chiplets with different L2s. This
    maps each XCD's slice of the grid back to a contiguous range of tile ids.
    Handles a grid that is not a multiple of ``num_xcds``: the first
    ``grid_mn % num_xcds`` XCDs get one extra tile.
    """
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    tall_xcds = grid_mn % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if xcd < tall_xcds:
        return xcd * pids_per_xcd + local_pid
    return tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid


@triton.jit
def matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """C = A @ B, staging global -> VGPR -> LDS (the fastest operand path on CDNA3)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = _xcd_remap(pid, num_pid_m * num_pid_n, NUM_XCDS)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Wrap the row/column offsets so an edge tile re-reads valid memory; the
    # epilogue store is masked, so the duplicated work is discarded. This keeps
    # the hot loop's loads unmasked in M/N -- only K needs a mask.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    K_ITERS = tl.cdiv(K, BLOCK_K)

    # The bank-conflict-avoiding padded shared layout is inferred by the compiler
    # from how these buffers feed tl.dot -- see gfx9_gemm/a16w16 v3 vs v4 for the
    # explicit form and why the inferred one is identical.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: fill the whole LDS ring.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.local_store(tlx.local_view(smem_a, i), a_reg)
        tlx.local_store(tlx.local_view(smem_b, i), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop. Iteration k multiplies K tile ``k - NUM_BUFFERS``, which lives in
    # buffer ``k % NUM_BUFFERS`` (since ``(k - NUM_BUFFERS) % NUM_BUFFERS ==
    # k % NUM_BUFFERS``), and refills that same buffer with tile k. Both global
    # loads are issued first so their latency overlaps the MFMA burst below; the
    # local_store then lands on a buffer the dot has already consumed.
    #
    # num_stages=1 disables the automatic software pipeliner: the ring and the
    # prefetch distance are managed by hand.
    for k in tl.range(NUM_BUFFERS, K_ITERS, num_stages=1):
        buf = k % NUM_BUFFERS
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K)

        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

        tlx.local_store(tlx.local_view(smem_a, buf), a_reg)
        tlx.local_store(tlx.local_view(smem_b, buf), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Epilogue: drain the NUM_BUFFERS tiles still sitting in the ring. Tile
    # ``K_ITERS - NUM_BUFFERS + i`` is in buffer ``(K_ITERS + i) % NUM_BUFFERS``.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        buf = (K_ITERS + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def lds_bytes(block_m, block_n, block_k, num_buffers, elem_bytes=2):
    """LDS footprint of a tile, ignoring shared-layout padding."""
    return (block_m * block_k + block_k * block_n) * elem_bytes * num_buffers


def _configs():
    """Tiles worth trying on MI300X.

    Deliberately small: MI300X has 304 CUs, so the useful range runs from a
    64x64 tile (enough workgroups to fill the chip on a 1024^2 output, which
    only decomposes into 16 tiles of 256x256) up to 256x256 (which needs a
    4096^2 output before it saturates). BLOCK_K is mostly 32 because the 64 KB
    CDNA3 LDS budget will not hold a deep K tile at the wide end -- 256x256x64
    would want 128 KB. `_prune_configs` drops whatever does not fit.
    """
    tiles = [
        (64, 64, 64, 4),
        (128, 128, 32, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (256, 128, 32, 8),
        (128, 256, 32, 8),
        (256, 256, 32, 8),
    ]
    return [
        triton.Config(
            {
                "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm, "NUM_BUFFERS": nb, "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": 0
            },
            num_warps=warps,
            # The manual LDS ring does the pipelining, so the automatic software
            # pipeliner must bail out -- which it does at num_stages=1.
            num_stages=1,
        ) for (bm, bn, bk, warps) in tiles for gm in (4, 8) for nb in (2, 3)
    ]


def _prune_configs(configs, named_args, **kwargs):
    """Drop tiles that cannot run on this shape before anything is compiled."""
    K = named_args["K"]
    elem_bytes = named_args["a_ptr"].element_size()
    kept = []
    for config in configs:
        bm = config.kwargs["BLOCK_M"]
        bn = config.kwargs["BLOCK_N"]
        bk = config.kwargs["BLOCK_K"]
        nb = config.kwargs["NUM_BUFFERS"]
        # The ring reads a buffer before refilling it, so it needs >= 2 buffers,
        # and K must supply at least that many tiles.
        if nb < 2 or triton.cdiv(K, bk) < nb:
            continue
        if lds_bytes(bm, bn, bk, nb, elem_bytes) > CDNA3_LDS_BYTES:
            continue
        kept.append(config)
    if not kept:
        raise RuntimeError(f"No config fits K={K} within the {CDNA3_LDS_BYTES} B gfx942 LDS budget")
    return kept


matmul_kernel_gfx942 = triton.autotune(
    configs=_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)(matmul_kernel_gfx942)


def matmul(a: torch.Tensor, b: torch.Tensor, config=None) -> torch.Tensor:
    """C = A @ B on AMD MI300X (gfx942).

    ``config`` is accepted for a uniform launcher signature across the tutorials
    and is intentionally unused -- the tile is autotuned per (M, N, K). Set
    ``TRITON_PRINT_AUTOTUNING=1`` to see which one wins.
    """
    assert a.shape[1] == b.shape[0], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    matmul_kernel_gfx942[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        # 16x16x16 MFMA on gfx942 fp16.
        matrix_instr_nonkdim=16,
    )
    return c
