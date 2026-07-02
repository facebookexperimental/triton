# AutoWS HSTU cross-attention `reduce_dq` — dq accuracy triage

Status: **root cause localized, not yet fixed.** autoWS produces wrong `dq` for the
HSTU cross-attention backward `reduce_dq` kernel. `dk`/`dv` are correct. The
non-WS `redq` kernel and the hand-written TLX `attn_bwd_ws` kernel are both
correct. Related task: T277922819.

## Symptom / the law

Comparing autoWS `dq` against `redq` (and a torch-float autograd reference), the
number of wrong `dq` Q-blocks is:

```
bad_Q_blocks = min(KV_blocks - 1, num_stages)   # always at the TAIL of the Q range
```

KV_blocks = 1 is always correct; the corruption grows with #KV blocks and
saturates at `num_stages`; independent of Q length. Deterministic.

## Repro

Environment: GB200 (sm100), `~/.conda/envs/metamain2/bin/python`, triton at
`~/MetaMain2/triton`, kernels at
`third_party/tlx/tutorials/hstu_cross_attn/`.

Accuracy (redq / autows / tlx vs torch-float ref; dq bad-blocks vs redq):
```bash
cd third_party/tlx/tutorials/hstu_cross_attn
~/.conda/envs/metamain2/bin/python bench_bwd.py --acc
# autows: KV=2 -> dq relL2 ~0.5, Qblk[3];  KV=3 -> ~0.7, Qblk[2,3].  redq/tlx clean.
```

KV/num_stages sweep that established the law: `/tmp/l10b.py` `go(Lq,Lkv,ns)`
(compares autoWS vs redq, prints bad Q-block indices).

Fast oracle (compute-sanitizer, tiny KV=2 shapes, `/tmp/rc.py`):
```bash
CS=/usr/local/cuda-12.8/bin/compute-sanitizer
V=autows $CS --tool synccheck --target-processes application-only python /tmp/rc.py
#   -> "Barrier error detected. Missing wait." at triton_bw_cross_attention.py:3272 (ABORTS)
V=tlx    $CS --tool synccheck ... python /tmp/rc.py    # -> clean
V=autows $CS --tool racecheck ... python /tmp/rc.py    # -> 0 hazards (both variants)
```

## Root cause (current best understanding)

The `dqᵀ` accumulator is a **single-copy (`buffer.copy=1`), cross-partition,
non-loop-carried** TMEM buffer (`buffer.id 5`, shared via reuse with `dP`):
produced by the **gemm** partition (`tc_gen5_mma`, `use_acc=false`) and consumed
by the **reduction** partition (`tmem_load` -> `async_tma_reduce add` to global
DQ) **every inner iteration**. `DQ[q]` accumulates across KV blocks in global
memory.

The producer and consumer are peeled by **different passes**, so their
`dqᵀ` FULL/EMPTY handshake phases do not stay paired across the KV boundary:

| side | partition | transform | prologue-wait phase |
|---|---|---|---|
| producer | gemm (task 1) | **software pipeliner (ExpandLoops)** SWP peel; multi-stage schedule | pipeliner accumCnt |
| consumer | reduction (task 0) | **ExpandLoops** first-iteration peel (dynamic-loop, guarded by `seq_len_q>0`; reduction has 1 op at `loop.stage=1` -> maxStage=1) | outer counter `%argN & 1` (peeled first Q-block) / inner counter (loop) |

`compute-sanitizer synccheck` confirms the runtime effect: a **present-but-mis-phased
wait** — `Barrier error: Missing wait` at the reduction's read of `dqᵀ`
(`triton_bw_cross_attention.py:3272`, `dq_trans * alpha`). It is invisible to
`racecheck` (no SMEM/global data race) and to static arrive/wait count-balancing
(all balanced); only the barrier-protocol model (`synccheck`) catches it. TLX
avoids it by hand-writing a single continuous `accum_cnt_q` and an explicit
epilogue drain, so both sides share one phase progression.

### Hypotheses tested and REFUTED (empirically)

1. **Loop-carriedness** — dq is non-loop-carried in *both* FA and HSTU (both use
   per-iter `async_tma_reduce`); only dk/dv are loop-carried.
2. **Nesting per se** — FA bwd *persistent* is also nested (tile x q) and correct.
3. **Escaping-counter freeze** (`arith.select` in the pipeliner drain) — removing
   both freezes changed the IR (9->7 selects) but dq was bit-identical (parity
   preserved for even `num_stages`).
4. **`self_latency` / prologue depth** — forcing `self_latency=0` on dv/dk shrank
   the gemm prologue 7->2 (1-deep, matches TLX) but dq was unchanged, same depth-2.
5. **Dynamic bound / missing peeled epilogue** — TLX *and* FA bwd both have
   data-dependent (runtime) loop bounds too (TLX `seq_len_q` from offsets; FA
   `num_steps = N_CTX/128`, N_CTX a runtime arg), so both take the pipeliner's
   `!peelEpilogue` (predicated drain) path, yet both are correct.

The live root cause is the **producer/consumer ExpandLoops-peel
phase mismatch on the single-slot cross-partition `dqᵀ` handshake**, not any of
the above.

## How the triage was done (methodology)

1. **Faithful OSS port + benchmark.** Ported redq/TLX/autows to OSS; built
   `bench_bwd.py` (accuracy vs torch-float ref + dq bad-blocks vs redq, and
   per-variant perf). This gives a trusted 3-way oracle.
2. **Characterize before theorizing.** Swept KV blocks / `num_stages` / Q length
   (`l10b.py`) to derive the exact law `min(KV-1, num_stages)` tail blocks — this
   constrains every hypothesis.
3. **Static barrier/phase analysis.** Used the `barrier-visualization` skill
   (mbarrier phase model, Section 3 index/phase) on the post-pipeline TTGIR to
   check the `dqᵀ` (buffer.id 5) FULL/EMPTY arrives/waits — found all balanced by
   count and parity (so NOT a missing/imbalanced barrier statically).
4. **Empirical hypothesis elimination.** Each candidate fix was implemented,
   rebuilt, and measured — refuting five hypotheses (above). Empirical over
   reasoning throughout: IR change without accuracy change = refuted.
5. **Runtime barrier oracle.** `compute-sanitizer racecheck` (clean) vs
   `synccheck` (autoWS: Missing wait at line 3272; TLX: clean) pinpointed a
   barrier-*protocol* violation invisible to data-race and count analysis.
6. **IR localization.** Dumped the final TTGIR (`TRITON_KERNEL_DUMP`), mapped the
   synccheck PC/source line to the reduction's `dqᵀ` read, and read the
   `wait_barrier` phase/predicate operands directly (prologue predicated by
   `seq_len_q>0`, phase from outer counter; inner loop unconditional). Confirmed via per-pass `MLIR_ENABLE_DUMP`: the reduction peel appears ONLY after ExpandLoops (the reduction loop has one op at `loop.stage=1` -> maxStage=1); FuseNestedLoops/WS/LowerLoops leave it a plain nested loop.

## Tools used

- `bench_bwd.py` — 3-way accuracy + perf benchmark (in-repo).
- `l10b.py` — KV/num_stages/Q sweep vs redq.
- `triton-opt --nvgpu-warp-specialization=... --tritongpu-pipeline="... dump-intermediate-steps=true"` — per-pass IR (schedule-loops, LowerLoops, ExpandLoops).
- `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=... TRITON_ALWAYS_COMPILE=1` — final compiled TTGIR/PTX.
- `compute-sanitizer --tool {racecheck,synccheck}` (CUDA 12.8) — runtime data-race and barrier-protocol checks. **synccheck is the fast oracle** (autoWS fails / TLX clean at KV=2).
- Skills: `barrier-visualization` (phase model), `ir-debugging`, `ir-override-ablation` (`TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=...` / `triton.Config(ir_override=...)` to A/B-edit the TTGIR).
- Lit test: `test/Hopper/WarpSpecialization/metaws_reduce_dq_expander_prologue.mlir` (pins the expander prologue behavior).

## FuseNestedLoops rotation — deep dive

`lib/Dialect/TritonGPU/Transforms/FuseNestedLoops.cpp` runs before the software
pipeliner and rewrites a one-level loop nest

```
for i:
  prologue0(i); for j0: body0; epilogue1(i); for j1: body1; ... epilogue(i)
```

into a single flat loop over `total_iters = len_i * inner_len`, with
`inner_len = max(1,len_j0)+...+max(1,len_jN) - N`, using a sub-counter `T`
(0..inner_len-1) and `scf.if` guards: `T==0` runs the prologue and inits `j0`;
`T in [start_k, start_k+len_jk)` runs body_k; `T==...` runs the epilogue. The
last body-k iteration overlaps epilogue_k and the first body-(k+1) iteration
(the `-N`). See the big comment block at `FuseNestedLoops.cpp:377-465`.

Key mechanics relevant to this bug:

- **Speculation / `max(1, len_jk)` (`ttg.must-execute`).** Each inner loop is
  treated as running **at least once**; the actual empty case is handled by the
  body guard `T < start+len_jk`. This is the `seq_len_q > 0` predicate
  (`%dv_65 = cmpi sgt, %seq_len_q, 0`) seen guarding the reduction's first-Q-block
  prologue, plus `select(%dv_65, accum_cnt, prev_cnt)` to thread the counter for
  the speculated-vs-empty case.
- **Bounds must be outer-loop-invariant; data-dependent bounds are NOT fully
  fused** (`FuseNestedLoops.cpp:445-453`): "We could fuse loops with ...
  data-dependent bounds, but this will require generating `scf.while`...". HSTU's
  inner Q bound (`seq_len_q`, from `seq_offsets`) is runtime/data-dependent, so
  the nest is **not fully fused** — instead it is **rotated/speculated**: the
  first inner iteration is peeled into the outer body (guarded by `seq_len_q>0`),
  the inner loop bound becomes `seq_len_q - BLOCK_M`, and the induction/counter is
  threaded via the `select`.
- **Induction variable threading** (`FuseNestedLoops.cpp:502-503`, 462-489):
  captures become loop-carried; the counter is incremented **inside the
  prologue** to avoid an epilogue dependency ("helps the scheduler behave").

**IR-DUMP CORRECTION (per-pass `MLIR_ENABLE_DUMP`): FuseNestedLoops does NOT peel this kernel.** The dumps show redq stays a plain nested `for KV { for Q }` after FuseNestedLoops (data-dependent inner bound `seq_len_q` -> not fused, not rotated), after WarpSpecialization, and after LowerLoops. The first-Q-block peel (predicate `%130 = cmpi sgt seq_len_q,0`, `select`-threaded counter, inner bound `seq_len_q-64`) appears ONLY **after ExpandLoops**. So BOTH the reduction and gemm inner loops are peeled by **ExpandLoops** (the pipeliner's dynamic-loop first-iteration peel + SWP), each threading the single-slot dqT phase differently -> the boundary mismatch. The FuseNestedLoops section below is retained as background on the pass, but it is NOT the source of the peel here.

Consequence for the `dqᵀ` handshake: the **consumer (reduction)** loop is
transformed by FuseNestedLoops rotation (peeled first Q-block, `select`-threaded
counter, `seq_len_q>0` predicate) and is then **not** SWP-peeled (single-stage).
The **producer (gemm)** loop is *also* rotated by FuseNestedLoops but is then
**SWP-peeled by the pipeliner** (multi-stage schedule). The two transforms thread
the shared single-slot `dqᵀ` phase differently, so the producer's per-Q-block
arrive and the consumer's rotated prologue wait diverge in parity at the KV
boundary — the `synccheck` Missing wait.

## Suggested fix directions (untested)

- Make the consumer's rotated-prologue `dqᵀ` wait use the **same continuous
  counter/phase** as the producer's SWP peel (align FuseNestedLoops rotation
  phase with the pipeliner peel for cross-partition single-slot channels).
- Or exempt the single-copy cross-partition `dqᵀ` channel from independent
  per-partition peeling (peel producer and consumer coherently).
- Validate any fix with `synccheck` (should go clean) + `bench_bwd.py --acc`
  (0 bad blocks, KV>=2) + FA bwd `test_op` unregressed.
- Fallback: use `redq` (non-WS) or TLX `attn_bwd_ws` for `reduce_dq`; both are
  correct and synccheck-clean.

## A/B loop via IR override

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dd python /tmp/rc.py   # dump
# edit dd/<hash>/_hstu_attn_bwd_redq.ttgir : the reduction dqT wait phase/predicate
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=dd TRITON_ALWAYS_COMPILE=1 \
  compute-sanitizer --tool synccheck python /tmp/rc.py     # oracle: clean?
```
