# case11_wait_order — schedule-draw order-sensitivity fixture

A minimal synthetic kernel that isolates one scheduling degree of freedom:
the position of a deferrable async-result read inside a warp group's
in-order instruction stream. Per loop iteration:

```
s1 = x_i @ w1^T          # MMA_A -> TMEM; its read feeds the exp2 chain
s2 = y_i @ w2^T          # MMA_B -> TMEM; its read feeds ONLY the combine
p  = exp2(s1 * scale)    # long SFU chain
z  = (p - s2) -> fp16    # combine + trunc
acc += z @ v             # MMA_C, loop-carried accumulator
```

The dependences admit two order families for `read(s2)`: coalesced with
`read(s1)` before the chain, or deferred until just before the combine.
Both families are model-equivalent (same II, same objective value, same
partition feasibility), yet they emit different in-order streams. On
case4 FA-bwd the same axis measured a 14% spread (292.8 vs 269.6 vs
255.8 TF at identical II/partition/objective, 2026-07-21 draw
calibration); this case distills that mechanism to 6 loop ops so draw
experiments can be run under control.

## Contents

| file | role |
|---|---|
| `wait_order_nows.py` | source kernel + torch reference; dump command in its docstring |
| `wait_order_pre_modulo.ttgir` | modulo-pass input extracted from the MLIR dump |
| `ddg.json` / `schedule_graph.json` | pass dumps (`TRITON_MODULO_DUMP_DDG` / `_SCHEDULE`) |
| `generated.py` | sched2tlx emit of the committed schedule graph |
| `run_generated.py` | correctness vs torch (three shapes) |
| `bench_spec.py` | perf-harness spec (no handwritten reference) |

Verified on B200: source kernel rel=3.0e-4; generated kernel PASS at
(G,T) = (2,4), (8,16), (148,32) with rel<=3.1e-4; 410 TFLOPS at
(592, 64).

## Sizing constraints (why the tiles look like this)

- `s1`/`s2` are [128, 64] fp32 on purpose: the async-reader legality
  forces the TMEM readers out of the producers' warp groups, so the two
  reader->producer recycle channels are always cut. At [128, 128] each
  cut is charged 65552 bytes of channel SMEM and the partition solve goes
  structurally infeasible against the 227KB budget; at [128, 64] the
  charge halves and everything fits (case4 uses the same accumulator
  footprint).
- `tl.trans(w1)` / `tl.trans(w2)` must stay hoisted OUTSIDE the loop.
  Written inside the loop, the resident tiles' loop-invariant
  `local_alloc`/`memdesc_trans` nodes stay in the scheduled loop body and
  the emitted kernel deadlocks (reproduced on both the auto 7-WG split
  and a hand-merged 6-WG partition; hoisting removes the hang).
