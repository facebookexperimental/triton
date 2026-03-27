# 2-CTA Backward Kernel — TLX vs FA4 Comparison

## Overview

The TLX 2-CTA backward kernel uses `two_ctas=True` on `async_dot` so that
each CTA holds half the B operand and the MMA hardware combines them.
Each CTA in a cluster processes a **different N-block** (consecutive rows of
K/V), matching the 1-CTA behavior but sharing Q/dO loads via multicast.

## Worked example: 2-CTA backward with real numbers

`tile_m=2, tile_n=2, hdim=2, cta_group_size=2`.
One `n_block_cta_group` covers `tile_n * 2 = 4` rows of K/V.

### Input tensors

```
Q = [2  1]     dO = [1  3]
    [4  3]          [2  1]

K0 = [1  2]  (CTA 0, rows 0-1)     V0 = [3  1]
     [3  4]                              [2  4]

K1 = [5  6]  (CTA 1, rows 2-3)     V1 = [1  5]
     [7  8]                              [4  2]

LSE = [2, 3]     D = [1, 1]
```

### Dot 1: S.T = K @ Q.T (`two_ctas=True`)

```
Inputs:
  A (per-CTA, each CTA has its own tile_n rows):
    CTA 0: sK = [1  2]     CTA 1: sK = [5  6]
                [3  4]                  [7  8]
  B (multicast, split along tile_m — each CTA holds one column of Q.T):
    Full Q.T = [2  4]
               [1  3]
    CTA 0 holds: [2]     CTA 1 holds: [4]
                 [1]                   [3]
    HW combines → full Q.T:
      [2  4]
      [1  3]

CTA 0: S0 = Q @ K0.T = [ 4  10]
                        [10  24]

CTA 1: S1 = Q @ K1.T = [16  22]
                        [38  52]
```

### Compute: P (assigned directly for clarity)

In practice, `P = exp(S - LSE)` with `LSE = logsumexp(S)` so rows sum to 1.
For this example we assign P directly:

```
CTA 0: P0 = [0.1  0.2]     CTA 1: P1 = [0.3  0.1]
            [0.4  0.3]                  [0.2  0.5]
```

### Dot 2: dP.T = V @ dO.T (`two_ctas=True`)

```
Inputs:
  A (per-CTA, each CTA has its own tile_n rows):
    CTA 0: sV = [3  1]     CTA 1: sV = [1  5]
                [2  4]                  [4  2]
  B (multicast, split along tile_m — each CTA holds one column of dO.T):
    Full dO.T = [1  2]
                [3  1]
    CTA 0 holds: [1]     CTA 1 holds: [2]
                 [3]                   [1]
    HW combines → full dO.T:
      [1  2]
      [3  1]

CTA 0: dP0.T = V0 @ dO.T = [ 5  10]     →  dP0 = [ 5  10]
                            [10  10]               [10  10]

CTA 1: dP1.T = V1 @ dO.T = [11   8]     →  dP1 = [11   8]
                            [ 8  14]               [ 8  14]
```

### Compute: dS = P * (dP - D), with D = [1, 1]

```
Inputs:
  CTA 0: P0 = [0.1  0.2]   dP0 = [ 5  10]
              [0.4  0.3]         [10  10]
  CTA 1: P1 = [0.3  0.1]   dP1 = [11   8]
              [0.2  0.5]         [ 8  14]

CTA 0: dP0 - D = [ 4   9]
                  [ 9   9]
  dS0 = [0.4   1.8]
        [3.6   2.7]

CTA 1: dP1 - D = [10   7]
                  [ 7  13]
  dS1 = [3.0   0.7]
        [1.4   6.5]
```

### Dot 3: dV += P.T @ dO (`two_ctas=True`)

```
Inputs:
  A (TMEM, per-CTA):
    CTA 0: P0.T = [0.1  0.4]     CTA 1: P1.T = [0.3  0.2]
                  [0.2  0.3]                    [0.1  0.5]
  B (multicast, split along hdim — each CTA holds one column of dO):
    Full dO = [1  3]
              [2  1]
    CTA 0 holds: [1]     CTA 1 holds: [3]
                 [2]                   [1]
    HW combines → full dO:
      [1  3]
      [2  1]

CTA 0: dV0 = P0.T @ dO = [0.9  0.7]
                          [0.8  0.9]

CTA 1: dV1 = P1.T @ dO = [0.7  1.1]
                          [1.1  0.8]
```

### Dot 5: dK += dS.T @ Q (`two_ctas=True`)

```
Inputs:
  A (TMEM, per-CTA):
    CTA 0: dS0.T = [0.4  3.6]     CTA 1: dS1.T = [3.0  1.4]
                   [1.8  2.7]                     [0.7  6.5]
  B (multicast, split along hdim — each CTA holds one column of Q):
    Full Q = [2  1]
             [4  3]
    CTA 0 holds: [2]     CTA 1 holds: [1]
                 [4]                   [3]
    HW combines → full Q:
      [2  1]
      [4  3]

CTA 0: dK0 = dS0.T @ Q = [15.2  11.2]
                          [14.4   9.9]

CTA 1: dK1 = dS1.T @ Q = [11.6   7.2]
                          [27.4  20.2]
```

### Dot 4: dQ = dS @ K (`two_ctas=True`, reduction spans both N-blocks)

```
Inputs:
  A (SMEM, MMA reads from both CTAs — each CTA contributes tile_n rows):
    CTA 0's sdS: dS0.T = [0.4  3.6]
                          [1.8  2.7]
    CTA 1's sdS: dS1.T = [3.0  1.4]
                          [0.7  6.5]
    HW reads both → combined A (2 rows, 4 cols):
      [0.4  1.8  3.0  0.7]
      [3.6  2.7  1.4  6.5]

  B (SMEM, per-CTA — each CTA loads ALL 4 K rows, one hdim column):
    Full K = [1  2]
             [3  4]
             [5  6]
             [7  8]
    CTA 0's sKt: [1]     CTA 1's sKt: [2]
                 [3]                   [4]
                 [5]                   [6]
                 [7]                   [8]
    HW combines → full K (4 rows, 2 cols):
      [1  2]
      [3  4]
      [5  6]
      [7  8]
```

The 2-CTA MMA computes one combined GEMM: `dQ = A @ B`:

```
dQ = [0.4  1.8  3.0  0.7] @ [1  2] = [25.7  31.6]
     [3.6  2.7  1.4  6.5]   [3  4]   [64.2  78.4]
                             [5  6]
                             [7  8]
```

**Both CTAs compute the full dQ — output is redundant:**

Both CTAs have the same A (sdS after exchange) and the hardware combines
B (sKt) into the same full K. So both CTAs compute the identical full dQ
`[[25.7, 31.6], [64.2, 78.4]]` in TMEM. This is redundant computation.

Each CTA only **reads out and writes** half the hdim to global:
```
CTA 0 reads dQ[:,0] from TMEM → atomic-adds [25.7, 64.2] to global
CTA 1 reads dQ[:,1] from TMEM → atomic-adds [31.6, 78.4] to global
```

The other half sits in TMEM unused. The atomic-add is needed to accumulate
across N-block groups (outer loop iterations), not between CTAs — within
one N-block group there is no overlap between CTA 0 and CTA 1's writes.

### Key observations

1. **Dots 1,2,3,5:** K/V are A operands (per-CTA, different N-block rows).
   Q/dO are B operands (multicast, shared). Each CTA gets different results
   because A differs.

2. **Dot 4:** Both A (sdS, after exchange) and combined B (sKt → full K)
   are the same for both CTAs. The computation is **redundant** — both CTAs
   produce the identical full dQ. Each CTA only writes a different hdim
   slice to global. This is a tradeoff: redundant compute for a single
   large GEMM (tile_n*2 reduction) with better MMA utilization.

3. **sK vs sKt — same data, different slicing:**
   ```
   sK  (A, dot 1):  own tile_n rows, full hdim
   sKt (B, dot 4):  ALL tile_n*2 rows, half hdim
   ```

4. **DSMEM exchange** sends half of each CTA's dS to the peer so that
   the 2-CTA MMA's A operand spans both N-blocks' dS.

5. **Atomic-add for dQ:** needed to accumulate across outer loop iterations
   (different N-block groups), not between CTAs. Within one N-block group,
   CTA 0 and CTA 1 write to non-overlapping hdim slices.

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
| sKt | (tile_n*2, hdim/2) | (256, 64) | 1 | 32 KB | **+32 KB** (new, B-operand for dQ GEMM, all rows both N-blocks) |
| sdS | (tile_n, tile_m) | (128, 128) | 1 | 32 KB | same (M halved, K doubled → cancels) |
| sdS_xchg | (tile_n, tile_m/2) | (128, 64) | — | 16 KB | **+16 KB** (new, DSMEM staging buffer) |
| sdQaccum | (tile_m×dQ_reduce_ncol, stages) | (128×8, 4) f32 | — | 16 KB | **−16 KB** |
| sLSE | (tile_m, Q_stage) | (128, 1) f32 | — | 0.5 KB | −0.5 KB |
| sdPsum | (tile_m, dO_stage) | (128, 1) f32 | — | 0.5 KB | same |
| Extra mbarriers | | | | ~0.1 KB | +0.1 KB (Qt, Kt, dS_cluster, dQaccum_empty) |
| **Total** | | | | **~225 KB** | **~0 KB** |

#### Net savings breakdown

- **Saved**: sQ (−48 KB) + sdO (−16 KB) + sdQaccum (−16 KB) = **−80 KB**
- **Added**: sQt (+16) + sdOt (+16) + sKt (+32) + sdS_xchg (+16) = **+80 KB**
- **Net: ~0 KB savings**

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

## How K is used differently in dot 1 vs dot 4 (2-CTA)

In 2-CTA mode, both CTAs in a cluster share the same `n_block_cta_group`,
which covers `tile_n * 2` rows of K. But K plays different roles in dot 1
(A operand) vs dot 4 (B operand), so it's sliced differently.

**Example:** `tile_n=2, hdim=4, cta_group_size=2`. One `n_block_cta_group`
covers 4 rows of K:

```
K (global) = [ a b c d ]  row 0 ─┐ CTA 0's n_block
             [ e f g h ]  row 1 ─┘
             [ i j k l ]  row 2 ─┐ CTA 1's n_block
             [ m n o p ]  row 3 ─┘
```

### Dot 1: S = K @ Q^T — K is A operand

A is per-CTA (not combined by hardware). Each CTA loads its own `tile_n`
rows, full hdim:

```
CTA 0's sK = [ a b c d ]    shape [2, 4]  — rows 0-1, full D
             [ e f g h ]

CTA 1's sK = [ i j k l ]    shape [2, 4]  — rows 2-3, full D
             [ m n o p ]
```

### Dot 4: dQ = dS @ K — K is B operand (sKt)

B is combined by hardware across CTAs. The full B has shape
`[K_dim, N_dim] = [tile_n*2, hdim] = [4, 4]`. Split along N (hdim),
each CTA loads **all 4 rows**, half the columns:

```
CTA 0's sKt = [ a b ]    shape [4, 2]  — ALL rows, cols 0-1
              [ e f ]
              [ i j ]
              [ m n ]

CTA 1's sKt = [ c d ]    shape [4, 2]  — ALL rows, cols 2-3
              [ g h ]
              [ k l ]
              [ o p ]
```

Hardware combines along hdim → full `K [4, 4]`.

### Summary

```
sK  (A, dot 1):  [tile_n, hdim]       — own rows, full D
sKt (B, dot 4):  [tile_n*2, hdim/2]   — ALL rows, half D
```

Same K data, sliced differently. In 2-CTA UMMA: A is per-CTA (not combined),
B is combined across CTAs (split along the output dimension). The "t" in `sKt`
refers to the SMEM layout (different swizzle for the B operand), not a
mathematical transpose — the math is still `dQ = dS @ K`.

In FA4 code: `sKt` in 1-CTA is just a recast of `sK` (same bytes, line 1297).
In 2-CTA, `sKt` is a separate buffer loaded via `tma_atom_Kt` (line 676).

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
