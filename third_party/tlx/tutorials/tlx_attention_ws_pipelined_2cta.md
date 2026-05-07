# TLX 2-CTA Backward Kernel Design

## Overview

Each CTA in a cluster processes a **different N-block** (consecutive rows
of K/V). Q/dO are shared via multicast (same M-block for both CTAs). All
5 dot products use `two_ctas=True`.

In 2-CTA MMA: A is per-CTA, B is split across CTAs and combined by
hardware. The output is split across CTAs along the M dimension.

## Worked example with real numbers

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

Dot 4 needs dS from **both** CTAs as the A operand. Before dot 4, each
CTA's sdS only has its own dS. The DSMEM exchange rearranges M-columns
so each CTA's sdS has the data needed for its query rows.

**Before exchange** — each CTA has its own dS in transposed form:
```
CTA 0's sdS: dS0.T = [0.4  3.6]     CTA 1's sdS: dS1.T = [3.0  1.4]
                      [1.8  2.7]                           [0.7  6.5]
```

**After DSMEM exchange** — each CTA's sdS has the dS values for its
query rows, spanning both N-blocks:
```
CTA 0's A row: [0.4  1.8  3.0  0.7]   (query row 0: dS0[0,:] then dS1[0,:])
CTA 1's A row: [3.6  2.7  1.4  6.5]   (query row 1: dS0[1,:] then dS1[1,:])
```

```
Inputs:
  A (SMEM, after DSMEM exchange):
    CTA 0's row: [0.4  1.8  3.0  0.7]
    CTA 1's row: [3.6  2.7  1.4  6.5]

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

**Output split along M (query rows):**

Each CTA computes its own query row:
```
CTA 0 computes dQ[0,:] = [25.7, 31.6] → TMA reduce-add to global
CTA 1 computes dQ[1,:] = [64.2, 78.4] → TMA reduce-add to global
```

The TMA reduce-add accumulates across outer loop iterations (different
N-block groups contributing partial dQ for the same M-block).

### Key observations

1. **Dots 1,2,3,5:** K/V are A operands (per-CTA, different N-block rows).
   Q/dO are B operands (multicast, shared). The 2-CTA MMA splits the
   output along M — each CTA gets its own tile_n rows (matching its
   N-block). No redundancy.

2. **Dot 4:** Same M-split mechanism. Each CTA produces tile_m/2 query
   rows of dQ with full hdim. The reduction dimension is tile_n*2
   (spanning both N-blocks). Requires DSMEM exchange of dS and a
   separate sKt buffer with all tile_n*2 K rows.

3. **sK vs sKt — same data, different slicing:**
   ```
   sK  (A, dot 1):  own tile_n rows, full hdim
   sKt (B, dot 4):  ALL tile_n*2 rows, half hdim
   ```

4. **DSMEM exchange** sends half of each CTA's dS to the peer so that
   the 2-CTA MMA's A operand spans both N-blocks' dS.

5. **TMA reduce-add for dQ:** accumulates partial dQ across outer loop
   iterations (different N-block groups). Within one N-block group,
   CTA 0 and CTA 1 write non-overlapping query row ranges.

## How K is used differently in dot 1 vs dot 4

Both CTAs share the same `n_block_cta_group`, which covers `tile_n * 2`
rows of K. K plays different roles in dot 1 (A) vs dot 4 (B), so it's
sliced differently.

**Example:** `tile_n=2, hdim=4, cta_group_size=2`:

```
K (global) = [ a b c d ]  row 0 ─┐ CTA 0's n_block
             [ e f g h ]  row 1 ─┘
             [ i j k l ]  row 2 ─┐ CTA 1's n_block
             [ m n o p ]  row 3 ─┘
```

**Dot 1 (A operand):** each CTA loads own tile_n rows, full hdim:
```
CTA 0: sK  = [ a b c d ]     CTA 1: sK  = [ i j k l ]
             [ e f g h ]                   [ m n o p ]
```

**Dot 4 (B operand):** each CTA loads ALL tile_n*2 rows, half hdim:
```
CTA 0: sKt = [ a b ]         CTA 1: sKt = [ c d ]
             [ e f ]                       [ g h ]
             [ i j ]                       [ k l ]
             [ m n ]                       [ o p ]
```

## TLX design: tile scheduling

In TLX with CLC, each CTA in a cluster gets its own `program_id` and
tile_id. Two CTAs naturally get consecutive tiles (pid 0, pid 1).
No special tile scheduling is needed — `start_n = pid` works as-is.
Grid size and `n_tile_num` do NOT change between 1-CTA and 2-CTA.

## TLX design: loads

**Per-CTA (no multicast):** K, V — each CTA loads from its own
`start_block_n`. Kt — each CTA loads ALL `tile_n*2` K rows, half hdim.

**Multicast (`two_ctas=True`):** Q, Qt, dO, dOt — both CTAs share the
same M-block.

## TLX design: DSMEM exchange

After compute produces dS, exchange M-column halves with peer via
`async_remote_shmem_store`. This makes the 2-CTA MMA's A operand span
both N-blocks' dS for dot 4.

## TLX design: dQ write-out

Each CTA writes its tile_m/2 query rows to global via TMA reduce-add.
No double-counting — CTAs write non-overlapping row ranges. The
reduce-add accumulates across N-block groups.

## TLX design: SMEM budget

Following FA4's 2-CTA approach: `Q_stage=1` (single-buffered Q) with
separate `q_tiles` and `qt_tiles` buffers.

| Buffer | Shape | Bufs | Size | Notes |
|--------|-------|------|------|-------|
| `k_tiles` | [N, D] | 1 | 32 KB | A for dot 1, per-CTA |
| `v_tiles` | [N, D] | 1 | 32 KB | A for dot 2, per-CTA |
| `q_tiles` | [M, D/2] | 1 | 16 KB | B for dots 3,5 (multicast, D-halved) |
| `qt_tiles` | [M/2, D] | 1 | 16 KB | B for dots 1,2 (multicast, M-halved) |
| `do_tiles` | [M, D/2] | 1 | 16 KB | B for dot 3 (multicast, D-halved) |
| `dot_tiles` | [M/2, D] | 1 | 16 KB | B for dots 1,2 (multicast, M-halved) |
| `kt_tiles` | [N*2, D/2] | 1 | 32 KB | B for dot 4, all K rows |
| `ds_tiles` | [N*2, M/2] | 1 | 32 KB | A for dot 4, after exchange |
| `ds_xchg_tiles` | [N, M/2] | 1 | 16 KB | DSMEM staging |
| epilogue staging | | | ~8 KB | Hidden SMEM for TMA stores |
| **Total** | | | **~216 KB** | Within 228 KB limit |

With single-buffered Q, the MMA loop order must free `q_empties`
(via dk dot) before dot1 loads new Q. This requires dk/dq before dot1
in the loop body. On the first tile, skip dk/dq (no previous data).

FA4 2-CTA loop order (hdim<=128): S → dK → dP → dQ → dV.

## TLX design: TMEM layout

TLX uses `tlx.local_alloc(..., tlx.storage_kind.tmem)` for TMEM buffers.
In 2-CTA mode, the output M dimension is split across CTAs. For dots 1,2,3,5
where M = `cta_group_size * tile_n` = 256, each CTA gets 128 TMEM columns.
For dot 4 where M = `tile_m` = 128, each CTA gets 64 TMEM columns.

| Buffer | Per-CTA shape | dtype | TMEM cols | Overlaps with | Notes |
|--------|--------------|-------|-----------|---------------|-------|
| `qk_tiles` (S) | (tile_n, tile_m) = (128, 128) | f32 | tile_n = 128 | `p_tiles` (shared), `dq_tiles` (distinct) | Dot 1 output |
| `p_tiles` (P) | (tile_n, tile_m) = (128, 128) | f16 | tile_n/2 = 64 | `qk_tiles` (shared), `dq_tiles` (distinct) | P = softmax(S) overwrites S |
| `dq_tiles` (dQ) | (tile_m/2, tile_hdim) = (64, 128) | f32 | tile_m/2 = 64 | `qk_tiles`/`p_tiles` cols 64-127 | Dot 4 output. Distinct from P |
| `dp_tiles` (dP) | (tile_n, tile_m) = (128, 128) | f32 | tile_n = 128 | `dsT_tmem_tiles`, `dsT_xchg_tiles` (shared) | Dot 2 output |
| `dsT_tmem_tiles` | (tile_n, tile_m) = (128, 128) | f16 | tile_n/2 = 64 | `dp_tiles`, `dsT_xchg_tiles` (shared) | dS in TMEM for Dot 5 (dk) |
| `dsT_xchg_tiles` | (tile_n, tile_m) = (128, 128) | f16 | tile_n/2 = 64 | `dp_tiles`, `dsT_tmem_tiles` (shared) | dS copy for DSMEM exchange (2-CTA only) |
| `dv_tiles` (dV) | (tile_n, tile_hdim) = (128, 128) | f32 | tile_n = 128 | — | Dot 3 accumulator |
| `dk_tiles` (dK) | (tile_n, tile_hdim) = (128, 128) | f32 | tile_n = 128 | — | Dot 5 accumulator |

Overlaps:
- **S → P**: P = softmax(S) overwrites S in-place (full overlap)
- **S/P ↔ dQ**: dQ (64 cols) partially overlaps S/P (128 cols) at columns
  64-127. In FA4, dQ is placed at `tmem_S_offset + tile_hdim//2 = 64`.
  This is safe because each CTA only reads P columns 0-63 (from the
  M-split), and dQ only writes to columns 64-127.
- **dP ↔ dsT_tmem ↔ dsT_xchg**: all three share `dp_dq_storage_alias`
  with `shared` overlap (same TMEM offset). dP (fp32, 128 cols) and
  dsT_tmem/dsT_xchg (fp16, 64 cols each) have sequential lifetime.

```
Column:  0          64    128        256                384       512
         |----------|------|----------|------------------|---------|
         |    S/P   | dQ   |    dV    |     dP/dS        |   dK    |
         |  (128)   | (64) |  (128)   |     (128)        |  (128)  |
         |__________|______|__________|__________________|_________|

qk_p_storage_alias (128 cols):  S/P shared at 0, dQ distinct at 64
dp_dq_storage_alias (128 cols): dP/dsT_tmem/dsT_xchg all shared at 0
```

dQ partially overlaps S/P at columns 64-127. In FA4, this is safe with
the loop order (dQ before dV) because FA4 controls the TMEM offset
(`tmem_dQ_offset = 64`), placing dQ at columns 64-127 while each CTA
reads P from columns 0-63. TLX does not have TMEM offset control —
`reuse=qk_tiles` places dQ at offset 0, fully overlapping P. Therefore,
TLX uses a different loop order: **dV before dQ** (S → dK → dP → dV → dQ),
ensuring P is consumed before dQ overwrites it.

Total: 512 TMEM columns (128 + 64 + 128 + 128 + 128 = 576, but dQ
overlaps 64 columns with S/P, so net = 512).

## TLX design: data flow (per N-block group)

```
Load (per-CTA): K → k_tiles, V → v_tiles, Kt → kt_tiles
Load (multicast): LSE, dPsum

Prolog (first M-block):
  Load (multicast): Q → q_tiles/qt_tiles, dO → do_tiles/dot_tiles

  MMA dot 1: S.T = K @ Q.T         → TMEM S (per-CTA, different)
  Compute:   P = exp(S - LSE)       → TMEM P
  MMA dot 2: dP.T = V @ dO.T       → TMEM dP (per-CTA, different)
  MMA dot 3: dV += P.T @ dO         → TMEM dV (per-CTA, different)
  Compute:   dS = P * (dP - D)      → TMEM dS, then copy to sdS
  DSMEM exchange: swap M-halves of sdS with peer

Per M-block (main loop, from 2nd block onward):
  Load (multicast): Q → q_tiles/qt_tiles, dO → do_tiles/dot_tiles

  MMA dot 1: S.T = K @ Q.T         → TMEM S
  Compute:   P = exp(S - LSE)       → TMEM P
  MMA dot 5: dK += dS.T[prev] @ Q[prev]  → TMEM dK (from prev iteration's dS)
  MMA dot 2: dP.T = V @ dO.T       → TMEM dP
  MMA dot 4: dQ = dS[prev] @ K            → TMEM dQ (frees q_empties[prev])
  MMA dot 3: dV += P.T @ dO         → TMEM dV
  Compute:   dS = P * (dP - D)      → TMEM dS, then copy to sdS
  DSMEM exchange: swap M-halves of sdS with peer
  Reduce:    dQ TMEM → TMA reduce-add to global

Epilog (last M-block's dk/dq):
  MMA dot 5: dK += dS.T[last] @ Q[last]
  MMA dot 4: dQ = dS[last] @ K

Epilogue: store dK, dV via TMA (each CTA to its own N-block)
```

---

## Appendix: FA4 reference

### FA4 code pointers for dot 4 (dQ = dS @ K)

| What | Line | Code |
|------|------|------|
| MMA tiler | 107 | `mma_tiler_dsk = (tile_m, tile_hdim, tile_n * cta_group_size)` |
| TiledMma | 310-317 | `tiled_mma_dQ = make_trivial_tiled_mma(..., cta_group)` |
| sKt layout | 392-398 | `sKt_layout = make_smem_layout_b(tiled_mma_dQ, mma_tiler_dsk, ...)` |
| TMA atom (B) | 673-683 | `tma_atom_Kt = make_tiled_tma_atom_B(...)` — 2-CTA only |
| Kt load | 1880-1892 | `gKt = local_tile(mKt_cur, ..., (0, n_block_cta_group))` |
| sKt in 1-CTA | 1297 | `sKt = recast_ptr(sK.iterator, sKt_layout)` — recast of sK |
| sKt in 2-CTA | 1295 | `sKt = storage.sKt.get_tensor(...)` — separate buffer |
| A fragment | 2255 | `tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)` |
| B fragment | 2256 | `tdQrK = tiled_mma_dQ.make_fragment_B(sKt)` |
| GEMM call | 2298-2306 | `gemm_w_idx(..., num_unroll_groups=2)` — different from other dots |
| TMEM alloc | 196-199 | `tmem_dQ_offset = tmem_S_offset + tile_hdim // 2` = 64 cols |
| dQ reduce | 3549 | `tile_hdim // cta_group_size` hdim per CTA in reduce |
| stage_offset | 3451 | `expected_reduce_stages * cta_rank_in_cluster` |

### FA4 1-CTA SMEM layout

tile_n=128, tile_m=128, hdim=128, Q_stage=2, fp16.

| Buffer | Shape | Stages | Size | Notes |
|--------|-------|--------|------|-------|
| sQ | (tile_m, hdim) | 2 | 64 KB | B-operand; sQt aliases via recast |
| sK | (tile_n, hdim) | 1 | 32 KB | A-operand; sKt aliases via recast |
| sV | (tile_n, hdim) | 1 | 32 KB | A-operand |
| sdO | (tile_m, hdim) | 1 | 32 KB | B-operand; sdOt aliases via recast |
| sdS | (tile_n, tile_m) | 1 | 32 KB | A for dot 4; sdSt aliases via recast |
| sdQaccum | (tile_m*32, 2) f32 | — | 32 KB | |
| sLSE | (tile_m, 2) f32 | — | 1 KB | |
| sdPsum | (tile_m, 1) f32 | — | 0.5 KB | |
| **Total** | | | **~225 KB** | |

### FA4 1-CTA TMEM layout

```
Column:  0                 128        256                384       512
         |-----------------|----------|------------------|---------|
         |      S/P        |    dV    |   dP/dS/dQ       |   dK    |
         |     (128)       |  (128)   |     (128)        |  (128)  |
         |_________________|__________|__________________|_________|
```

### FA4 2-CTA SMEM layout

| Buffer | Shape per CTA | Concrete | Stages | Size | Notes |
|--------|--------------|----------|--------|------|-------|
| sQ | (tile_m, hdim/2) | (128, 64) | 1 | 16 KB | B for dots 1,5 (multicast, D-halved) |
| sK | (tile_n, hdim) | (128, 128) | 1 | 32 KB | A for dot 1 (per-CTA, own N-block) |
| sV | (tile_n, hdim) | (128, 128) | 1 | 32 KB | A for dot 2 (per-CTA, own N-block) |
| sdO | (tile_m, hdim/2) | (128, 64) | 1 | 16 KB | B for dots 2,3 (multicast, D-halved) |
| sQt | (tile_m/2, hdim) | (64, 128) | 1 | 16 KB | B for dots 1,2 (multicast, M-halved) |
| sdOt | (tile_m/2, hdim) | (64, 128) | 1 | 16 KB | B for dots 1,2 (multicast, M-halved) |
| sKt | (tile_n*2, hdim/2) | (256, 64) | 1 | 32 KB | B for dot 4 (all K rows, D-halved) |
| sdS | (tile_n, tile_m) | (128, 128) | 1 | 32 KB | A for dot 4 (MMA reads from both CTAs) |
| sdS_xchg | (tile_n, tile_m/2) | (128, 64) | 1 | 16 KB | DSMEM staging buffer |
| sdQaccum | (tile_m*8, 4) | — | — | 16 KB | dQ reduce-add staging (f32) |
| sLSE | (tile_m, 1) | (128, 1) | — | 0.5 KB | |
| sdPsum | (tile_m, 1) | (128, 1) | — | 0.5 KB | |
| **Total** | | | | **~225 KB** | |

### FA4 2-CTA TMEM layout

Each CTA allocates 512 TMEM columns. Each column holds 128 elements × 32 bits.
The M dimension of each MMA output maps to TMEM columns. In 2-CTA, the M
dimension is split across CTAs (each CTA gets M/cta_group_size columns).

For dots 1,2,3,5: M = `cta_group_size * tile_n` = 256, per-CTA = 128 columns.
For dot 4: M = `tile_m` = 128, per-CTA = 64 columns.

```
Column:  0          64    128        256        384       512
         |----------|------|----------|----------|---------|
         |    S/P   | dQ   |    dV    |  dP/dS   |   dK    |
         |  (128)   | (64) |  (128)   |  (128)   |  (128)  |
         |__________|______|__________|__________|_________|
```

| Buffer | Offset | Columns | Per-CTA shape | Notes |
|--------|--------|---------|---------------|-------|
| S/P | 0 | 128 | (tile_n, tile_m) | M/2 = tile_n per CTA |
| dQ | 64 | 64 | (tile_m/2, tile_hdim) | M/2 = tile_m/2 per CTA. Overlaps S/P cols 64-127 |
| dV | 128 | 128 | (tile_n, tile_hdimv) | M/2 = tile_n per CTA |
| dP/dS | 256 | 128 | (tile_n, tile_m) | M/2 = tile_n per CTA |
| dK | 384 | 128 | (tile_n, tile_hdim) | M/2 = tile_n per CTA |

### FA4 dot 4: different PTX instruction

Dot 4 uses `gemm_w_idx` with `num_unroll_groups=2`, while dots 1,2,3,5
use `gemm_ptx_w_idx` with `cta_group=cta_group_size`. This may affect
how the output is distributed across CTAs.
