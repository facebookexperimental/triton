"""TLX GEMM tutorial for AMD MI300X (gfx942 / CDNA3).

Every other AMD GEMM tutorial here targets gfx950 (CDNA4) and stages operands
with *direct-to-LDS* copies. On CDNA3 the right choice is the register-staged
path instead:

    global --tl.load--> VGPR --tlx.local_store--> LDS --tlx.local_load--> MFMA

What is left to tune, and what this kernel does:

* **Size the ring to the 64 KB CDNA3 LDS budget**, not CDNA4's 160 KB. The gfx950
  kernels' 256x256 tile with 64-deep K and two buffers wants ~128 KB and cannot
  be made to fit; even 256x128x64 needs 96 KB. ``lds_bytes`` computes the
  footprint and the launcher rejects anything over budget up front.
* **Remap program ids across the 8 XCDs.** MI300X dispatches consecutive
  workgroups round-robin over the chiplets, which scatters tiles that should be
  sharing B columns in one L2. Measured, this is **neutral** on square shapes --
  0.86x aten with the remap vs 0.87x without at 4096^3, 0.79x vs 0.79x at
  8192^3, i.e. inside the noise. It is kept because it is the standard MI300X
  grid transform and the A/B is one flag away (``NUM_XCDS=1``, exposed as the
  ``gfx942_noxcd`` provider in the perf script), but do not expect it to carry
  this kernel -- the GROUP_M swizzle is already capturing the L2 reuse here.
* **Choose the tile from the output shape** (``pick_config``). With 304 CUs, one
  fixed tile cannot serve both ends: pinning the 256x256 tile costs 0.35x aten
  at 1024^3 where the shape-aware pick gets 1.15x, and 0.51x vs 0.88x at
  2048^3. This is the single largest effect in the whole kernel.
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


# MI300X has 304 CUs, which is a lot of machine to fill: one 256x256 tile per
# workgroup means a 4096^2 output is only 256 workgroups and a 1024^2 output is
# 16. So there is no single best tile -- the default is chosen per shape.
NUM_CUS = 304

# LDS = (256*32 + 32*256) * 2 B * 2 buffers = exactly 64 KB, the whole CDNA3
# budget. This is the arch squeeze in one line: the gfx950 kernels run the same
# 256x256 tile at BLOCK_K=64 with room to spare, but that would want 128 KB
# here, so K depth is what CDNA3 gives up.
CONFIG_LARGE = {
    "BLOCK_M": 256,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 4,
    "NUM_BUFFERS": 2,
    "num_warps": 8,
    "waves_per_eu": 0,
}

# 32 KB. The all-rounder: BLOCK_K can go back to 64 once the tile is smaller.
CONFIG_MEDIUM = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 4,
    "NUM_BUFFERS": 2,
    "num_warps": 8,
    "waves_per_eu": 0,
}

# 16 KB, 4 warps. For outputs too small to fill the chip any other way.
CONFIG_SMALL = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "GROUP_M": 4,
    "NUM_BUFFERS": 2,
    "num_warps": 4,
    "waves_per_eu": 0,
}

# Kept as the name the perf script pins when it wants one fixed tile.
DEFAULT_CONFIG = CONFIG_LARGE


def pick_config(M, N):
    """Choose a tile for an MxN output, keyed on how much of the chip it fills.

    The threshold is the count of 256x256 tiles the output decomposes into,
    against the 304 CUs. Measured on MI300X fp16 (TFLOPS as a fraction of aten;
    the winner per row is what this function returns):

        M     N     K   256-tiles  256x256x32/W8  128x128x64/W8  64x64x64/W4
      1024  1024  1024         16         0.35x         0.86x        1.20x
      1536  1536  1536         36         0.38x         0.94x        0.90x
      2048  2048  2048         64         0.48x         0.86x        0.68x
      3072  3072  3072        144         0.70x         0.80x        0.64x
      4096  4096  4096        256         0.82x         0.60x        0.54x
      8192  8192  8192       1024         0.79x           --           --

    Known gap: on strongly rectangular shapes (1024x8192x8192, 8192x1024x8192,
    both 128 tiles) this picks the medium tile for ~0.65x, where a 128x128x32
    tile at 4 warps measures ~0.74x. Rather than special-case the aspect ratio,
    pass an explicit config -- or run
    ``test_amd_gemm_gfx942_perf.py --sweep --sweep-shape M N K``.
    """
    tiles_256 = triton.cdiv(M, 256) * triton.cdiv(N, 256)
    if tiles_256 >= NUM_CUS - 48:  # ~a full wave of 256x256 workgroups
        return CONFIG_LARGE
    if tiles_256 <= 16:
        return CONFIG_SMALL
    return CONFIG_MEDIUM


def lds_bytes(config, elem_bytes=2):
    """LDS footprint of a config, ignoring shared-layout padding."""
    per_buffer = (config["BLOCK_M"] * config["BLOCK_K"] + config["BLOCK_K"] * config["BLOCK_N"]) * elem_bytes
    return per_buffer * config["NUM_BUFFERS"]


def matmul(a: torch.Tensor, b: torch.Tensor, config=None) -> torch.Tensor:
    """C = A @ B on AMD MI300X (gfx942). ``config=None`` picks a tile by shape."""
    assert a.shape[1] == b.shape[0], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"

    M, K = a.shape
    _, N = b.shape
    if config is None:
        config = pick_config(M, N)

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    GROUP_M = config.get("GROUP_M", 8)
    NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
    num_warps = config.get("num_warps", 8)
    waves_per_eu = config.get("waves_per_eu", 0)
    num_xcds = config.get("NUM_XCDS", NUM_XCDS)

    assert NUM_BUFFERS >= 2, "The ring reads a buffer before refilling it, so it needs >= 2 buffers"
    assert triton.cdiv(K, BLOCK_K) >= NUM_BUFFERS, \
        f"K={K} has fewer than NUM_BUFFERS={NUM_BUFFERS} tiles of BLOCK_K={BLOCK_K}"
    used = lds_bytes(config, a.element_size())
    assert used <= CDNA3_LDS_BYTES, \
        f"config needs {used} B of LDS, over the {CDNA3_LDS_BYTES} B gfx942 budget"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        NUM_BUFFERS=NUM_BUFFERS,
        NUM_XCDS=num_xcds,
        num_warps=num_warps,
        # The manual ring does the pipelining, so the automatic software
        # pipeliner must bail out -- which it does at num_stages=1.
        num_stages=1,
        # 16x16x16 MFMA on gfx942 fp16.
        matrix_instr_nonkdim=16,
        waves_per_eu=waves_per_eu,
    )
    return c
