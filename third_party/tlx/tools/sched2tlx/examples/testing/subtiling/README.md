# Steady-state sub-tiling (parallel duplication) through the modulo scheduler

Companion to Pass A.7's epilogue subtiling (sequential N-slicing of the
outer-loop store chain): this directory exercises SUB-TILING OF THE
STEADY-STATE LOOP — the whole FA computation (QK MMA, softmax chain, PV
MMA, accumulators, iter_args) duplicated into two independent 128-row
M-instances that the partitioner maps onto separate warp groups.

- `fa_subtiled_experiment.py` — the source-level vehicle: FA fwd with
  BLOCK_M=256 written as two 128-row sub-tiles sharing each iteration's
  K/V tiles, no warp specialization in source. Runs correctness on the
  default pipeline; with the modulo env set it feeds the dump below.
- `run_subtiled_generated.py <generated.py> [--small] [--bench]` —
  correctness (out AND M_lse) + bench for an emitted kernel.

Reproduce end to end (B200):

```bash
# 1. schedule + partition (Rau + exhaustive scoreCandidate — no solver):
TRITON_ALWAYS_COMPILE=1 TRITON_USE_MODULO_SCHEDULE=1 \
TRITON_MODULO_DISABLE_MMA_GUARD=1 \
TRITON_MODULO_DUMP_SCHEDULE=/tmp/subtiled.json \
python fa_subtiled_experiment.py
# 2. emit:
python -m sched2tlx /tmp/subtiled.json -o /tmp/subtiled_generated.py
# 3. validate + bench:
python run_subtiled_generated.py /tmp/subtiled_generated.py --bench
```

Measured (2026-07-06): II=2487, 6-WG partition (2 TMA + 2 TC + 2
softmax), correctness PASS including M_lse, **687.9 TFLOPS** at
(1,32,8192) vs 665.5 for the regenerated single-tile case3 and 651 for
the previous committed kernel. The guard bypass is required: with the
serial-MMA guard on, the split graph prices at ResMII 2918 and never
reaches this design point (the guard exists to protect SINGLE-tile FA's
softmax cohesion; auto-scoping it for multi-instance graphs is the
follow-up).

Full history of how this design point was found (plateau analysis,
clock64 replays, the register-spill discovery behind the emitter's
load-at-use rule): PR #1917's SubTilingDesign.md.
