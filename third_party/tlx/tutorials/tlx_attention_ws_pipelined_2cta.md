# 2-CTA Backward Kernel — TLX vs FA4 Comparison

## Overview

The TLX 2-CTA backward kernel uses `two_ctas=True` on `async_dot` so that
each CTA holds half the B operand and the MMA hardware combines them.
Each CTA in a cluster processes a **different N-block** (consecutive rows of
K/V), matching the 1-CTA behavior but sharing Q/dO loads via multicast.

## SMEM Analysis: TLX Does NOT Achieve FA4's Savings

### Problem Summary

The current TLX implementation doesn't get the SMEM savings that FA4's
CUTLASS version gets:

1. **k_tiles / v_tiles: [BLOCK_N, HEAD_DIM] — full size, not halved.**
   These are A operands in dots 1,2 but B operands in dots 4,5. In FA4,
   all B operands are halved by the 2-CTA hardware split.

2. **ds_tiles: [BLOCK_N * 2, BLOCK_M] — doubled.** Stores the full dS from
   both CTAs (own + peer). FA4 avoids this by keeping sdS at the same size
   and using a small `sdS_xchg = (tile_n, tile_m/2)` staging buffer for the
   DSMEM exchange.

3. **kt_tiles, qt_tiles, dot_tiles: allocated but unused — dead code.**

4. **q_tiles / do_tiles: halved — this is correct.**

**Net result:** the halving of Q and dO is eaten up (and then some) by the
doubled dS. K and V aren't halved at all. No net SMEM savings, possibly
a regression.

### FA4's Approach (for reference)

Using tile_n=128, tile_m=128, hdim=128, fp16 (2 bytes per element).

**Key design choices:**
- B operands (Q, Qt, dO, dOt, Kt) are halved by the 2-CTA hardware split
- A operands (K, V) stay the same size — each CTA owns its tile_n rows
- sdS stays the same total size (M halved by A-operand split, K doubled
  to span both CTAs → cancels out)
- Only a small `sdS_xchg = (tile_n, tile_m/2)` staging buffer is added
  for the DSMEM cross-CTA dS exchange
- Q_stage drops from 2 (1-CTA) to 1 (2-CTA)

#### 1-CTA SMEM layout (Q_stage=2)

| Buffer | Symbolic shape | Concrete shape | Stages | Size | Notes |
|--------|---------------|----------------|--------|------|-------|
| sQ | (tile_m, hdim) | (128, 128) | 2 | 64 KB | B-operand; sQt aliases via recast |
| sK | (tile_n, hdim) | (128, 128) | 1 | 32 KB | A-operand |
| sV | (tile_n, hdim) | (128, 128) | 1 | 32 KB | A-operand |
| sdO | (tile_m, hdim) | (128, 128) | 1 | 32 KB | B-operand; sdOt aliases via recast |
| sdS | (tile_n, tile_m) | (128, 128) | 1 | 32 KB | sdSt aliases via recast |
| sdQaccum | (tile_m×dQ_reduce_ncol, stages) | (128×32, 2) f32 | — | 32 KB | |
| sLSE | (tile_m, Q_stage) | (128, 2) f32 | — | 1 KB | |
| sdPsum | (tile_m, dO_stage) | (128, 1) f32 | — | 0.5 KB | |
| **Total** | | | | **~225 KB** | |

#### 2-CTA SMEM layout (Q_stage=1, hdim≤128)

| Buffer | Symbolic shape | Concrete shape | Stages | Size | Δ vs 1-CTA |
|--------|---------------|----------------|--------|------|------------|
| sQ | (tile_m/2, hdim) | (64, 128) | 1 | 16 KB | **−48 KB** (halved B + 1 stage) |
| sK | (tile_n, hdim) | (128, 128) | 1 | 32 KB | same |
| sV | (tile_n, hdim) | (128, 128) | 1 | 32 KB | same |
| sdO | (tile_m/2, hdim) | (64, 128) | 1 | 16 KB | **−16 KB** (halved B) |
| sQt | (hdim/2, tile_m) | (64, 128) | 1 | 16 KB | **+16 KB** (new, separate layout for dK GEMM) |
| sdOt | (hdim/2, tile_m) | (64, 128) | 1 | 16 KB | **+16 KB** (new, separate layout for dP GEMM) |
| sKt | (tile_n, hdim/2) | (128, 64) | 1 | 16 KB | **+16 KB** (new, B-operand for dQ GEMM) |
| sdS | (tile_n, tile_m) | (128, 128) | 1 | 32 KB | same (M halved, K doubled → cancels) |
| sdS_xchg | (tile_n, tile_m/2) | (128, 64) | — | 16 KB | **+16 KB** (new, DSMEM staging buffer) |
| sdQaccum | (tile_m×dQ_reduce_ncol, stages) | (128×8, 4) f32 | — | 16 KB | **−16 KB** |
| sLSE | (tile_m, Q_stage) | (128, 1) f32 | — | 0.5 KB | −0.5 KB |
| sdPsum | (tile_m, dO_stage) | (128, 1) f32 | — | 0.5 KB | same |
| Extra mbarriers | | | | ~0.1 KB | +0.1 KB (Qt, Kt, dS_cluster, dQaccum_empty) |
| **Total** | | | | **~209 KB** | **~−16 KB** |

#### Net savings breakdown

- **Saved**: sQ (−48 KB) + sdO (−16 KB) + sdQaccum (−16 KB) = **−80 KB**
- **Added**: sQt (+16) + sdOt (+16) + sKt (+16) + sdS_xchg (+16) = **+64 KB**
- **Net: ~16 KB savings**

The SMEM savings are modest. The main benefit of 2-CTA is **doubling the
MMA compute throughput** by having two CTAs cooperate on each GEMM
instruction, not SMEM reduction.

## Buffer Layout (current TLX, per CTA)

BLOCK_N1=128, BLOCK_M1=128, HEAD_DIM=128, NUM_CTAS=2, fp16 (2 bytes).

| Buffer | Symbolic shape | Concrete shape | Bufs | Storage | Size | Notes |
|--------|---------------|----------------|------|---------|------|-------|
| `k_tiles` | [BLOCK_N1, HEAD_DIM] | [128, 128] | 1 | SMEM | 32 KB | **Not halved** (A operand) |
| `v_tiles` | [BLOCK_N1, HEAD_DIM] | [128, 128] | 1 | SMEM | 32 KB | **Not halved** (A operand) |
| `q_tiles` | [BLOCK_M1, HEAD_DIM/NUM_CTAS] | [128, 64] | 2 | SMEM | 32 KB | Correctly halved (B operand for dots 3,5) |
| `do_tiles` | [BLOCK_M1, HEAD_DIM/NUM_CTAS] | [128, 64] | 1 | SMEM | 16 KB | Correctly halved (B operand for dot 3) |
| `kt_tiles` | [BLOCK_N1, HEAD_DIM/NUM_CTAS] | [128, 64] | 1 | SMEM | 16 KB | **Dead code** |
| `qt_tiles` | [BLOCK_M1/NUM_CTAS, HEAD_DIM] | [64, 128] | 2 | SMEM | 0 KB | reuse=q_tiles; **Dead code** |
| `dot_tiles` | [BLOCK_M1/NUM_CTAS, HEAD_DIM] | [64, 128] | 1 | SMEM | 0 KB | reuse=do_tiles; **Dead code** |
| `ds_tiles` | [BLOCK_N1, BLOCK_M1] | [256, 128] | 1 | SMEM | 64 KB | **Doubled** (own + peer dS) |
| `ds_peer_tiles` | [BLOCK_N1, BLOCK_M1/2] | [128, 64] | 1 | SMEM | 16 KB | Peer's dS exchange buffer |
| `qk_tiles` / `p_tiles` | [BLOCK_N1, BLOCK_M1] | [128, 128] | — | TMEM | — | S→P reuse (S overwritten by P) |
| `dp_tiles` / `dq_tiles` | [BLOCK_N1, BLOCK_M1] | [128, 128] | — | TMEM | — | dP→dQ reuse (dP overwritten by dQ) |
| `dk_tiles` | [BLOCK_N1, HEAD_DIM] | [128, 128] | — | TMEM | — | dK accumulator |
| `dv_tiles` | [BLOCK_N1, HEAD_DIM] | [128, 128] | — | TMEM | — | dV accumulator |
| **SMEM Total** | | | | | **208 KB** | |

## Dot Products (all use `two_ctas=True`)

| Dot | Operation | A (local) | B (per CTA, combined by HW) | Result |
|-----|-----------|-----------|----------------------------|--------|
| 1 | qkT = K @ Q^T | k_tiles [N, D] | qt_tiles [D, M/2] → [D, M] | qk_tiles [N, M] |
| 2 | dpT = V @ dO^T | v_tiles [N, D] | dot_tiles [D, M/2] → [D, M] | dp_tiles [N, M] |
| 3 | dV += P @ dO | p_tiles [N, M] (tmem) | do_tiles [M, D/2] → [M, D] | dv_tiles [N, D] |
| 4 | dQ = dS^T @ K | dsT [M, N] | kt_tiles [N, D/2] → [N, D] | dq_tiles [M, D] |
| 5 | dK += dS @ Q | ds_tiles [N, M] | q_tiles [M, D/2] → [M, D] | dk_tiles [N, D] |

## Tile Scheduling

- `n_tile_num = cdiv(N_CTX, BLOCK_N1 * NUM_CTAS)` — each cluster handles 2 consecutive N-blocks.
- `start_n = pid * NUM_CTAS + cluster_cta_rank` — CTA 0 gets even N-blocks, CTA 1 gets odd.

## DSMEM Exchange

After the compute warpgroup produces `dsT [BLOCK_N, BLOCK_M]`, each CTA
sends half of it to the peer via `async_remote_shmem_store`:
- CTA 0 sends `dsT[:, 0:M/2]`, CTA 1 sends `dsT[:, M/2:M]`
- Peer receives it in `ds_peer_tiles`
- Then overwrites the peer's half in `ds_tiles` with received data

This doubles `ds_tiles` to `[BLOCK_N * 2, BLOCK_M]` (own + peer), unlike
FA4 which keeps sdS at the same size and uses a small staging buffer.

### FA4's DSMEM Exchange (for comparison)

FA4 avoids doubling the sdS buffer by using a small staging buffer
(`sdS_xchg`) and writing the peer's data directly into sdS via DSMEM.

**Why the exchange is needed:** The dQ GEMM (`dQ = dS @ K`) has a
reduction dimension spanning both CTAs (`tile_n * 2`). Each CTA only
computes its own dS in TMEM, but the dQ GEMM needs dS from both CTAs
as the A-operand in SMEM. The dK GEMM (`dK = dS^T @ Q`) reads dS from
TMEM (local only, no exchange needed).

**How it works:** The dS tile `(tile_n, tile_m)` is processed in 2
sub-tile stages (tile_m split into two halves). One stage is the
"exchange stage" (sent to peer), the other stays local.

```
exchange_stage = cta_rank_in_cluster ^ 1
  CTA0 (rank=0): exchange_stage = 1  (sends its stage 1)
  CTA1 (rank=1): exchange_stage = 0  (sends its stage 0)
```

**Step-by-step with a 4×4 example** (tile_n=4, tile_m=4, stage 0 = cols 0-1,
stage 1 = cols 2-3):

CTA0 computes dS₀:
```
  stage 0        stage 1
  a  b            c  d
  e  f            g  h
  i  j            k  l
  m  n            o  p
```

CTA1 computes dS₁:
```
  stage 0        stage 1
  A  B            C  D
  E  F            G  H
  I  J            K  L
  M  N            O  P
```

**During dS computation (registers → SMEM):**
- Non-exchange stage → written directly to `sdS`
- Exchange stage → held in registers, then written to `sdS_xchg`

**DSMEM copy:**
- CTA0 sends `sdS_xchg` (stage 1: `c d / g h / k l / o p`)
  → peer CTA1's `sdS[stage 0]`
- CTA1 sends `sdS_xchg` (stage 0: `A B / E F / I J / M N`)
  → peer CTA0's `sdS[stage 1]`

**After exchange — each CTA's sdS (tile_n × tile_m = 4×4):**

CTA0's sdS:
```
  stage 0 (own)   stage 1 (from CTA1)
  a  b              A  B
  e  f              E  F
  i  j              I  J
  m  n              M  N
```

CTA1's sdS:
```
  stage 0 (from CTA0)  stage 1 (own)
  c  d                   C  D
  g  h                   G  H
  k  l                   K  L
  o  p                   O  P
```

Each CTA now has its own dS half plus the peer's half in `sdS` — ready
for the dQ GEMM's reduction over `tile_n * 2`. The `sdS_xchg` buffer
is dead after this point.

**Key advantage over TLX:** FA4's `sdS` stays at `(tile_n, tile_m)` =
32 KB (same as 1-CTA), plus a small `sdS_xchg = (tile_n, tile_m/2)` =
16 KB staging buffer. Total = 48 KB. TLX doubles `ds_tiles` to
`(tile_n*2, tile_m)` = 64 KB plus `ds_peer_tiles` = 16 KB. Total = 80 KB.

## TMEM Layout (FA4 bwd, hdim=128)

TMEM has **512 columns** (each column = 128 rows × 32 bits). The M-dimension
of each MMA output maps to TMEM columns. The backward kernel fully packs
all 512 columns with overlapping buffers.

### 1-CTA TMEM layout

```
Column:  0                 128        256                384       512
         |-----------------|----------|------------------|---------|
         |      S/P        |    dV    |   dP/dS/dQ       |   dK    |
         |     (128)       |  (128)   |     (128)        |  (128)  |
         |_________________|__________|__________________|_________|
```

| Buffer | Offset | Columns | Overlaps with |
|--------|--------|---------|---------------|
| S      | 0      | 128     | P (same offset) |
| P      | 0      | 128     | S (same offset) |
| dV     | 128    | 128     | — |
| dP     | 256    | 128     | dS, dQ (all same offset) |
| dS     | 256    | 128     | dP, dQ (all same offset) |
| dQ     | 256    | 128     | dP, dS (all same offset) |
| dK     | 384    | 128     | — |

In 1-CTA, dQ uses **128 columns** (full tile_m) and aliases dP/dS — all
three are used at different phases so the overlap is safe.

### 2-CTA TMEM layout

```
Column:  0          64    128        256        384       512
         |----------|------|----------|----------|---------|
         |    S/P   | dQ   |    dV    |  dP/dS   |   dK    |
         |  (128)   | (64) |  (128)   |  (128)   |  (128)  |
         |__________|______|__________|__________|_________|
```

| Buffer | Offset | Columns | Overlaps with |
|--------|--------|---------|---------------|
| S      | 0      | 128     | P (same offset) |
| P      | 0      | 128     | S (same offset) |
| dQ     | 64     | 64      | overlaps S/P cols 64-127 |
| dV     | 128    | 128     | — |
| dP     | 256    | 128     | dS (same offset) |
| dS     | 256    | 128     | dP (same offset) |
| dK     | 384    | 128     | — |

In 2-CTA, dQ uses only **64 columns** (tile_m/2) because the M-dimension
is split across CTAs. This lets dQ fit in the S/P region (cols 64-127)
instead of needing a separate 128-column slot. dQ no longer aliases dP/dS.

### Why overlaps are safe (both modes)

- **S → P**: S is computed, then overwritten with P = exp(S − LSE)
- **dP → dS**: dP is computed, then dS = P × (dP − D) overwrites it
- **dQ overlaps S/P** (2-CTA) or **dP/dS** (1-CTA): dQ is computed
  after the overlapped buffers are fully consumed

## Constraints

- **Non-causal only** (STAGE=1): Causal gives different `num_steps` per CTA.
- **BLOCK_M1=128**: Required by pair-CTA MMA.
- **REUSE_DP_FOR_DQ=True** when BLOCK_M1=128 and HEAD_DIM=128.

## TMA Descriptors

| Descriptor | Block Shape | Used For |
|------------|------------|----------|
| `desc_k` | [BLOCK_N1, HEAD_DIM] | Loading K (full — not halved) |
| `desc_v` | [BLOCK_N1, HEAD_DIM] | Loading V (full — not halved) |
| `desc_q` | [BLOCK_M1, HEAD_DIM // NUM_CTAS] | Loading Q (D-split) |
| `desc_do` | [BLOCK_M1, HEAD_DIM // NUM_CTAS] | Loading dO (D-split) |
| `desc_kt` | [BLOCK_N1, HEAD_DIM // NUM_CTAS] | Loading K (D-split for dot 4) |
| `desc_qt` | [BLOCK_M1 // NUM_CTAS, HEAD_DIM] | Loading Q (M-split for dots 1,2) |
| `desc_dot` | [BLOCK_M1 // NUM_CTAS, HEAD_DIM] | Loading dO (M-split for dots 1,2) |
| `desc_dq` | [BLOCK_M1, HEAD_DIM // EPILOGUE_SUBTILE] | Atomic-add dQ |
| `desc_dk` | [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE] | Store dK |
| `desc_dv` | [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE] | Store dV |
