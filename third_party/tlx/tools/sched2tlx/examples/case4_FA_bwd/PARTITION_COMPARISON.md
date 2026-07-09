# case4 FA-backward: generated vs hand-written warp-group partition

This compares the warp-specialized partition the modulo scheduler produces (and
`sched2tlx` lowers into `generated.py`) against the hand-written TLX ground truth
(`handwritten.py`), for the single-CTA 5-MMA FA backward kernel.

Per-iteration the kernel computes, over a fixed K/V block and a loop of Q tiles:

```
qkT = K @ Qᵀ                         (MMA1)
pT  = exp2(qkT*log2e - m)            (softmax)
dpT = V @ dOᵀ                        (MMA2)
dV += pT @ dO                        (MMA3)
dsT = pT * (dpT - D)                 (dS)
dQ  = trans(dsT) @ K                 (MMA4)   → atomic-add to global
dK += dsT @ Q                        (MMA5)
```

## Hand-written (`handwritten.py`) — clean role split, software-pipelined

Four groups, each with a single role; **all 5 MMAs live in one `mma` group**,
which is **skewed** (software-pipelined):

```
load     (nw1) : K,V resident + Q,dO per-iter (Q double-buffered, NUM_BUFFERS_Q=2)
mma      (nw1) : prolog blk=0  -> qkT(blk), dpT(blk), dV(blk)
                 main   blk>=1 -> qkT(blk), dK(blk-1), dQ(blk-1), dpT(blk), dV(blk)
                 epilog        -> dK(last), dQ(last)
                 TMEM aliasing: qk/p share, dp/dsT/dq share
compute  (nw4) : softmax pT + dS  (then dK/dV epilogue store)
reduction(nw4) : dQ atomic-add
```

The skew is the key: in one `mma` loop body the group **produces** `qkT(blk)`
for `compute` and **consumes** `dsT(blk-1)` that `compute` already produced the
previous iteration. So the `mma` group never waits on `compute`'s
*same-iteration* output — the `qkT -> softmax -> dsT -> dK/dQ` cycle is broken by
the one-iteration lag.

## Generated (modulo scheduler) — scrambled groups, single-stage flat loop

Four groups too, but roles are **mixed** and the loop is **flat** (no skew). The
5 MMAs are split **3 / 2** across two tensor-core groups, and one of them also
absorbs the TMA loads and the softmax/dS CUDA work:

```
wg0  (TC,   nw1) : qkT  +  dQ  +  dK              (3 MMAs)
wg1  (nw4)       : loads(Q,dO) + softmax + dS + dpT + dV   (2 MMAs + loads + CUDA)
wg2  (CUDA, nw4) : dQ reduce (atomic-add)
default          : dK, dV epilogue stores (post-loop)
```

## Per-op assignment + iteration

| op                       | hand-written group / iter | generated wg / iter (cyc) |
|--------------------------|---------------------------|---------------------------|
| load K,V                 | `load` (resident)         | `wg1` blk (resident)      |
| load Q, dO               | `load` blk (Q ×2-buffer)  | `wg1` blk (cyc 0-90)      |
| MMA1 qkT = K@Qᵀ          | `mma` **blk**             | `wg0` blk (508)           |
| softmax pT               | `compute` blk             | `wg1` blk (1067-1467)     |
| MMA2 dpT = V@dOᵀ         | `mma` **blk**             | `wg1` blk (538)           |
| dS / dsT                 | `compute` blk             | `wg1` blk (2037-2138)     |
| MMA3 dV += pT@dO         | `mma` **blk**             | `wg1` blk (2190)          |
| MMA4 dQ = dsTᵀ@K         | `mma` **blk-1** (skew)    | `wg0` blk (2158)          |
| MMA5 dK += dsT@Q         | `mma` **blk-1** (skew)    | `wg0` blk (2220)          |
| dQ atomic-add            | `reduction` blk           | `wg2` blk (3058-3243, stage 1) |

(Generated schedule: II=3136, single-stage — 34 nodes at stage 0, only the
3-op reduce tail at stage 1.)

## Summary of the difference

| aspect        | hand-written            | generated                         |
|---------------|-------------------------|-----------------------------------|
| 5 MMAs        | one `mma` group         | split wg0 (3) / wg1 (2)           |
| loads         | own `load` group        | folded into wg1                   |
| softmax/dS    | own `compute` group     | folded into wg1                   |
| pipeline      | skewed (prolog/main/epi)| flat, single-stage                |
| Q buffering   | double                  | single                            |
| TMEM reuse    | explicit aliasing       | none (HD=64 fits; aliasing gated) |

The dependency chain is identical, but the generated partition places `wg0` on
**both ends** of it within one iteration (produces qkT, then needs wg1's
same-iteration dsT, then produces dK/dQ) while `wg1` needs wg0's qkT. That
bidirectional, within-iteration cross-WG coupling is what made FA-bwd the first
case to require multi-consumer barrier handling in the emitter: the dO tile feeds
both dpT and dV in wg1, and the dsT tile feeds both dK and dQ in wg0 — a buffer
read by N MMAs, whose EMPTY barrier needs `arrive_count = N`.

Functionally the two partitions are equivalent; on the measured shapes the
generated (flat) kernel is marginally faster than the hand-written (skewed) one,
because the flat schedule keeps all four groups busy each iteration. The
hand-written remains the cleaner, more readable structure.
