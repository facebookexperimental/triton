"""Shared-A BMM for gfx950, structured after the rocBLAS/Tensile kernel.

Tensile picks `MT256x256x64 MI16x16 WG32_8_1 PGR2 PLR1` for these shapes, and the
dispatch record shows `LDS_Block_Size = 66560` -- exactly one 64KB stage plus a 1KB
pad. So rocBLAS **single-buffers LDS and holds its pipeline depth in registers**
(PGR2 = prefetch-global-read 2). That one choice is what the whole design hangs on:

    LDS depth 1                            -- not 3 buffers
      -> affords a 256x256 macro tile
      -> 2x2 waves => a SQUARE 128x128 per-wave tile
      -> 64 MFMA per wave per K-tile from 16 fragment reads (4.0 MFMA/read)
      -> ~4.7x less ds_read traffic, hence far fewer waits and barriers

Also inherited from the a16w16 work in this directory:
  * the K tail is PEELED, so the main loop's loads are unmasked and can use the
    CDNA4 unaligned-wide-load path (one dwordx4 per lane even when K is odd);
  * AGPRs are left enabled -- the accumulator is 256 VGPR/lane and needs them.

**B is split along N into two independent halves.** This is what took the kernel
from 2108 -> 2025 us. With one monolithic `tl.dot(a, b, acc)` every ds_read is a
true dependence of the single dot op, so no interleaving is legal and codegen
emits `RRRRx24` then `MMMMx86`: the wave stalls on lgkmcnt with no math to overlap.
Splitting B means `dot(a, b_lo)` does not depend on the load of `b_hi`, so that
load slides under the previous MFMA group. Measured: global-load stall 31.5% ->
20.3%, ds_read 13.8% -> 9.8%.
Splitting A as well (2x2 quadrants, see amd_bmm_rbq.py) is WORSE -- it pushes VGPR
to the 512 ceiling and the scheduler re-clumps for lack of free registers.

**The ds_writes ARE covered by MFMAs, but only by the LLIR scheduler** -- this
kernel wants `TRITON_ENABLE_LLIR_SCHED=1`. MFMAs preceding each of the 8 ds_writes
in the final ISA loop body:

    LLIR off   [64, 0, 0, 0, 0, 0, 0, 0]   2189/2268 us  -- one clump, zero overlap
    LLIR on    [34, 10, 1, 1, 1, 1, 1, 1]  2007/2018 us  -- 6 of 8 shadowed, ~9%

The writes land in scheduler region 1 (`region 1 BEFORE: 32M 8LW`), which the pass
schedules to a perfect `1M 1LW` alternation in LLVM IR and pins with mask-0
sched.barriers; the ISA above is what survives register allocation.

DEEPENING THAT COVER IS WORTH NOTHING -- measured, do not retry.
The cover is thin: the byte model prices 8 narrow writes at ldsBudget=6, so each
falls back to the 1-MFMA floor while the unspent surplus (24 MFMAs) goes to the
head/tail split. `LLIRSCHED_MFMA_PER_LW=N` (added for this) reserves a uniform floor
of N from that surplus before the split. It works mechanically -- the ISA shows
exactly `3M 1LW` x8 -- and it does not move the clock:

    PER_LW   ISA MFMA-before-each-write     GPU0    GPU4
      0      [34, 10, 1, 1, 1, 1, 1, 1]     2040    2087
      3      [34,  2, 3, 3, 3, 3, 3, 3]     2042    2091
      4      [32,  1, 2, 4, 4, 4, 4, 4]     surplus exhausted, early writes
      6      [32,  0, 0, 0, 1, 6, 6, 6]     re-clump at the head

So the 10.5% ATT stall attributed to ds_write is NOT on the critical path -- the
drain already retires inside the existing 1-MFMA gap plus the barrier wait. What the
LLIR pass buys (~9%) is having the writes interleaved AT ALL versus one clump; how
much math sits between them is second-order.

MEASUREMENT WARNING: an earlier 1.9% "win" here was an artifact of a 3-module
round-robin A/B harness -- the FIRST-LOADED module posted the best time regardless
of which config it held. Use the single-module driver, one config per process.
This box also emits 3-30x outliers on a contended GPU; check `rocm-smi --showpids`
and prefer a quiet device.

Source-level attempts to reposition the writes all failed -- do not retry:
  * stores hoisted above both dots      2179us  (all 3 reads must then precede the
                                                 barrier, killing the read interleave)
  * MFMAs split across the barrier      2143us  (dot_lo covers the b_hi read, dot_hi
                                                 covers the stores -- still worse)
  * scheduler knobs LLIRSCHED_LW_BEFORE / MFMA_PER_LW_BARRIER: inert. Verified
    byte-identical scheduler output and ISA; the pass has already placed what it
    can, and these knobs only reposition cover, they cannot deepen it.
"""
import os

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_M = 256
BLOCK_N = 256
# 32 is the measured best. 64 matches Tensile's DepthU and halves LDS bank
# conflicts (586M -> 315M) because the row pitch becomes 64 fp16 = 128 B = one
# full 32-bank row, but it needs RB_PGR=1 to avoid a 49-VGPR spill, and the
# latency cover PGR1 gives up costs more than the conflicts it saves (2147 vs
# 2066). See the LDS section below.
BLOCK_K = int(os.environ.get('RB_BK', 32))
# ---- LDS bank conflicts: 50.68% for us vs 0.68% for rocBLAS (measured) --------
# SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT), same shape:
#
#     ours     idx 1,741,946,880  conflict 585,891,840   50.68%
#     rocBLAS  idx 1,172,275,200  conflict   7,864,320    0.68%
#
# Our CLEAN cycles (1,156,088,320) match rocBLAS's total, so the access pattern is
# right and we pay a ~50% surcharge purely in conflicts.
#
# It is the READS, not the writes. rocBLAS's own ds_write_b128 runs at 22.1 cy/hit
# vs our 24.7 -- within 12%. But rocBLAS issues ONLY ds_read_b128 (8.3 cy/hit),
# while we issue 48 ds_read_b64_tr_b16, the LDS transpose read: half-width
# (8 B/lane vs 16) AND 2.6x over ideal. Total read cost is 1.86x rocBLAS.
# We need the transpose because B sits in LDS N-contiguous while MFMA wants 8
# halves along K per lane -- a 256 B stride, i.e. every lane on one bank.
#
# Everything tried against it FAILED -- do not retry without new evidence:
#   RB_PAD row padding, 6 values: 32 B pad is a BIT-EXACT no-op (it reaches the
#     IR -- `padded_shared<[32:+16]>` -- and moves the counter by zero); any pad
#     that is not a multiple of 32 B de-vectorizes instead (8 B pad: 5.3x LDS
#     cycles). At BK=64 the clean cycles are invariant across every pad.
#   RB_BORDER=1 (store B K-contiguous, order [0,1]): kills the transpose read
#     exactly as intended (ds_read_b64_tr_b16 48 -> 0) but moves the transpose to
#     the store side (ds_write_b16 16 -> 96). 5929 us.
#   RB_BK=64 + RB_PGR=1: conflicts halved, LDS cycles -15%, and 4% SLOWER.
#     Conflicts are not the binding constraint -- buffer_load is 18.3% of stall
#     against ~9.4% for all LDS reads combined.
#   RB_NW=8 to fit BK=64 with PGR2: spill goes UP (122), worse.
LDS_PAD = int(os.environ.get('RB_PAD', 0))
# 1 = store B K-contiguous in LDS (order [0,1]) to kill the transpose read.
B_ORDER = int(os.environ.get('RB_BORDER', 0))
# global-prefetch depth: 2 = Tensile PGR2 (default), 1 = shallow ring for BK=64.
PGR_DEPTH = int(os.environ.get('RB_PGR', 2))
NUM_WARPS = int(os.environ.get('RB_NW', 4))  # 2x2 -> square 128x128 per-wave tile; nw=8 gives [2,4] and halves reuse
NUM_XCDS = 8


@triton.jit
def _chip(pid, nt, nx: tl.constexpr, cs: tl.constexpr):
    """L2 XCD-chunk remap: keep a batch's MN-tiles on one XCD."""
    al = (nt // (nx * cs)) * (nx * cs)
    if pid >= al:
        return pid
    x = pid % nx
    lp = pid // nx
    return (lp // cs) * nx * cs + x * cs + (lp % cs)


@triton.jit
def _bmm_rbs(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, NUM_XCDS: tl.constexpr, GMN: tl.constexpr, NT: tl.constexpr,
             PAD: tl.constexpr, BORDER: tl.constexpr, PGR: tl.constexpr):
    HN: tl.constexpr = BN // 2  # B is halved along N into two independent operands

    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN
    pid = pidf % GMN
    pm = pid // npn
    pn = pid % npn

    # ---- LDS: ONE buffer each. Pipeline depth lives in registers, not here. ----
    # PAD>0 inserts PAD fp16 elements at the end of every row, shifting each
    # row's starting bank. Unpadded, A's row is BK*2 = 64 B (half the 32-bank
    # x 4 B width, so rows alias in pairs) and B's row is HN*2 = 256 B (exactly
    # 2x the width, so EVERY row starts at bank 0). Measured 50.68% conflict.
    if BORDER > 0:
        # Store B K-CONTIGUOUS (order [0,1]) instead of N-contiguous.
        #
        # v_mfma_f32_16x16x32_f16 wants each lane holding 8 halves along K. With
        # the default N-contiguous B tile those 8 halves are HN*2 = 256 B apart --
        # every one lands on the same bank -- so codegen falls back to
        # ds_read_b64_tr_b16, the LDS transpose read. That is half-width (8 B/lane
        # vs 16) AND conflicted (10.4 cy against a 4-cycle ideal), which is where
        # our LDS read cost goes: 1.86x rocBLAS, which issues only ds_read_b128 at
        # a near-ideal 8.3 cy. K-contiguous should make our read a plain b128 too.
        b_sh: tl.constexpr = tlx.swizzled_shared_layout_encoding(vectorSize=1, perPhase=1, maxPhase=1, order=[0, 1],
                                                                 numCTAs=1, numCTAsPerCGA=[1, 1], numCTASplit=[1, 1],
                                                                 numCTAOrder=[1, 0])
        lds_a = tlx.local_view(tlx.local_alloc((BM, BK), tl.float16, 1), 0)
        lds_b_lo = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1, layout=b_sh), 0)
        lds_b_hi = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1, layout=b_sh), 0)
    elif PAD > 0:
        a_sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(BK, PAD)], [BM, BK])
        b_sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(HN, PAD)], [BK, HN])
        lds_a = tlx.local_view(tlx.local_alloc((BM, BK), tl.float16, 1, layout=a_sh), 0)
        lds_b_lo = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1, layout=b_sh), 0)
        lds_b_hi = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1, layout=b_sh), 0)
    else:
        lds_a = tlx.local_view(tlx.local_alloc((BM, BK), tl.float16, 1), 0)
        lds_b_lo = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1), 0)
        lds_b_hi = tlx.local_view(tlx.local_alloc((BK, HN), tl.float16, 1), 0)

    offs_m = (pm * BM + tl.arange(0, BM)) % M
    offs_n_lo = (pn * BN + tl.arange(0, HN)) % N
    offs_n_hi = (pn * BN + HN + tl.arange(0, HN)) % N
    offs_k = tl.arange(0, BK)
    a_ptr = a_ptr + bid.to(tl.int64) * sab
    b_ptr = b_ptr + bid.to(tl.int64) * sbb
    a_row = offs_m[:, None] * sam
    b_col_lo = offs_n_lo[None, :] * sbn
    b_col_hi = offs_n_hi[None, :] * sbn
    n_ktiles = K // BK  # whole K-tiles; the host guarantees >= 2

    # ------------------------------------------------------------------
    # Three K-tiles are in flight at once, each in a different place:
    #
    #   LDS       (lds_*)   holds tile k      <- the MFMAs consume this
    #   registers (next_*)  hold  tile k+1    <- fetched, waiting for an LDS slot
    #   in-flight (pref_*)  fetching tile k+2 <- issued this iteration
    #
    # Holding k+1 in VGPRs instead of a second LDS buffer is Tensile's PGR2, and
    # is what leaves LDS room for the 256x256 tile.
    # ------------------------------------------------------------------
    # prologue: tile 0 -> LDS, tile 1 -> registers
    tile0_a = tl.load(a_ptr + a_row + offs_k[None, :] * sak)
    tile0_b_lo = tl.load(b_ptr + offs_k[:, None] * sbk + b_col_lo)
    tile0_b_hi = tl.load(b_ptr + offs_k[:, None] * sbk + b_col_hi)
    tlx.local_store(lds_a, tile0_a)
    tlx.local_store(lds_b_lo, tile0_b_lo)
    tlx.local_store(lds_b_hi, tile0_b_hi)
    tl.debug_barrier()

    acc_lo = tl.zeros((BM, HN), dtype=tl.float32)
    acc_hi = tl.zeros((BM, HN), dtype=tl.float32)

    if PGR == 1:
        # ---- 1-deep ring (Tensile PGR1) ------------------------------------
        # Only tile k+1 is in flight, so three of the six operand tiles vanish
        # from the register file. That is what makes BK=64 fit: at BK=64 the
        # 2-deep ring spills 49 VGPRs, and a spilling inner loop costs ~4x.
        # BK=64 matters because it puts the LDS row pitch at 64 fp16 = 128 B =
        # exactly one 32-bank x 4 B row, which is the pitch rocBLAS uses and the
        # reason its reads are conflict-free; at BK=32 the row is 64 B, half a
        # bank row, and rows alias in pairs.
        # The cost is less latency cover: the global fetch has only this
        # iteration's two dots to hide under, not two iterations'.
        for k in tl.range(0, n_ktiles - 1):
            k_next = (k + 1) * BK
            cur_a = tlx.local_load(lds_a)
            cur_b_lo = tlx.local_load(lds_b_lo)
            nxt_a = tl.load(a_ptr + a_row + (k_next + offs_k[None, :]) * sak)
            nxt_b_lo = tl.load(b_ptr + (k_next + offs_k[:, None]) * sbk + b_col_lo)
            acc_lo = tl.dot(cur_a, cur_b_lo, acc_lo)
            cur_b_hi = tlx.local_load(lds_b_hi)
            nxt_b_hi = tl.load(b_ptr + (k_next + offs_k[:, None]) * sbk + b_col_hi)
            acc_hi = tl.dot(cur_a, cur_b_hi, acc_hi)
            tl.debug_barrier()
            tlx.local_store(lds_a, nxt_a)
            tlx.local_store(lds_b_lo, nxt_b_lo)
            tlx.local_store(lds_b_hi, nxt_b_hi)
            tl.debug_barrier()
        # drain: LDS holds the last whole tile
        cur_a = tlx.local_load(lds_a)
        acc_lo = tl.dot(cur_a, tlx.local_load(lds_b_lo), acc_lo)
        acc_hi = tl.dot(cur_a, tlx.local_load(lds_b_hi), acc_hi)
    else:
        # ---- 2-deep ring (Tensile PGR2), the default -------------------------
        next_a = tl.load(a_ptr + a_row + (BK + offs_k[None, :]) * sak)
        next_b_lo = tl.load(b_ptr + (BK + offs_k[:, None]) * sbk + b_col_lo)
        next_b_hi = tl.load(b_ptr + (BK + offs_k[:, None]) * sbk + b_col_hi)
        for k in tl.range(0, n_ktiles - 2):
            k_prefetch = (k + 2) * BK  # byte-index of the tile we start fetching now

            # -- stage 1: pull tile k out of LDS, and start the tile k+2 global fetch.
            # The global loads are issued BEFORE the math on purpose: their latency is
            # hundreds of cycles and the MFMAs below (plus the next iteration) cover it.
            cur_a = tlx.local_load(lds_a)
            cur_b_lo = tlx.local_load(lds_b_lo)
            pref_a = tl.load(a_ptr + a_row + (k_prefetch + offs_k[None, :]) * sak)
            pref_b_lo = tl.load(b_ptr + (k_prefetch + offs_k[:, None]) * sbk + b_col_lo)

            # -- stage 2: first half of the math. Depends only on cur_a and cur_b_lo.
            acc_lo = tl.dot(cur_a, cur_b_lo, acc_lo)

            # -- stage 3: the second half. Its operand load is deliberately placed AFTER
            # the first dot: nothing above depends on lds_b_hi, so this ds_read (and the
            # global load beside it) execute *underneath* the MFMAs just issued. This is
            # the whole point of splitting B -- with one dot it would be a hard
            # dependence and the hardware would drain every read before any MFMA.
            cur_b_hi = tlx.local_load(lds_b_hi)
            pref_b_hi = tl.load(b_ptr + (k_prefetch + offs_k[:, None]) * sbk + b_col_hi)
            acc_hi = tl.dot(cur_a, cur_b_hi, acc_hi)

            # -- stage 4: hand tile k+1 from registers into LDS. Two barriers because
            # LDS is single-buffered: the first guarantees every wave has finished
            # reading tile k before we overwrite it, the second that tile k+1 is
            # visible before anyone reads it. ~2 barriers per 128 MFMAs.
            tl.debug_barrier()
            tlx.local_store(lds_a, next_a)
            tlx.local_store(lds_b_lo, next_b_lo)
            tlx.local_store(lds_b_hi, next_b_hi)
            tl.debug_barrier()

            # -- stage 5: rotate the register ring; tile k+2 becomes "next".
            next_a = pref_a
            next_b_lo = pref_b_lo
            next_b_hi = pref_b_hi

        # drain: LDS holds tile n-2 and the registers hold tile n-1
        cur_a = tlx.local_load(lds_a)
        acc_lo = tl.dot(cur_a, tlx.local_load(lds_b_lo), acc_lo)
        acc_hi = tl.dot(cur_a, tlx.local_load(lds_b_hi), acc_hi)
        tl.debug_barrier()
        tlx.local_store(lds_a, next_a)
        tlx.local_store(lds_b_lo, next_b_lo)
        tlx.local_store(lds_b_hi, next_b_hi)
        tl.debug_barrier()
        cur_a = tlx.local_load(lds_a)
        acc_lo = tl.dot(cur_a, tlx.local_load(lds_b_lo), acc_lo)
        acc_hi = tl.dot(cur_a, tlx.local_load(lds_b_hi), acc_hi)

    # masked scalar tail for a partial final K-tile: runs 0 or 1 times, no LDS.
    # Keeping it out of the loop above is what lets those loads stay unmasked and
    # therefore vectorize to wide (unaligned) dwordx4.
    for k_tail in tl.range(n_ktiles * BK, K, BK):
        k_mask = (k_tail + offs_k) < K
        tail_a = tl.load(a_ptr + a_row + (k_tail + offs_k[None, :]) * sak, mask=k_mask[None, :], other=0.0)
        tail_b_lo = tl.load(b_ptr + (k_tail + offs_k[:, None]) * sbk + b_col_lo, mask=k_mask[:, None], other=0.0)
        tail_b_hi = tl.load(b_ptr + (k_tail + offs_k[:, None]) * sbk + b_col_hi, mask=k_mask[:, None], other=0.0)
        acc_lo = tl.dot(tail_a, tail_b_lo, acc_lo)
        acc_hi = tl.dot(tail_a, tail_b_hi, acc_hi)

    # epilogue: the two accumulators are adjacent N-halves of the output tile
    et = c_ptr.dtype.element_ty
    cb = c_ptr + bid.to(tl.int64) * scb
    out_m = pm * BM + tl.arange(0, BM)
    out_n_lo = pn * BN + tl.arange(0, HN)
    out_n_hi = out_n_lo + HN
    tl.store(cb + scm * out_m[:, None] + scn * out_n_lo[None, :], acc_lo.to(et),
             mask=(out_m[:, None] < M) & (out_n_lo[None, :] < N))
    tl.store(cb + scm * out_m[:, None] + scn * out_n_hi[None, :], acc_hi.to(et),
             mask=(out_m[:, None] < M) & (out_n_hi[None, :] < N))


def bmm(a, b):
    """C = A @ B, shared-A, ROW-major B. 256x256 tile, 4 waves, MI16x16, split-B."""
    Bs, M, K = a.shape
    N = b.shape[-1]
    assert K // BLOCK_K >= 2, f"need >= 2 whole K-tiles of {BLOCK_K}, got K={K}"
    GMN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    NT = Bs * GMN
    c = torch.empty((Bs, M, N), device=a.device, dtype=a.dtype)
    st = (a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1),
          c.stride(2))
    _bmm_rbs[(NT, )](a, b, c, M, N, K, *st, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K, NUM_XCDS=NUM_XCDS, GMN=GMN, NT=NT,
                     PAD=LDS_PAD, BORDER=B_ORDER, PGR=PGR_DEPTH, num_warps=NUM_WARPS, num_stages=1, matrix_instr_nonkdim=16)
    return c
