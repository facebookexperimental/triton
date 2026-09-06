# NVIDIA AutoWS And TLX Matching

Use this guidance when a Triton candidate is being compared with a hand-written
TLX warp-specialized kernel. Treat AutoWS as a compiler transformation that must
be proven in final IR, not as a source flag.

## Enable And Isolate AutoWS

Annotate the loop that contains the recurring load/MMA pipeline with
`tl.range(..., warp_specialize=True)`. Enable both `TRITON_USE_META_WS=1` and
`TRITON_DISABLE_WSBARRIER_REORDER=1` before JIT compilation. Pin a viable tile,
warp count, and stage count while iterating; `num_stages >= 2` is normally
required for stage/order scheduling.

Meta-WS is process-global and must not be applied to the hand-written TLX
reference. Compile and benchmark the AutoWS candidate and TLX reference in
separate processes. Keep their input shapes, correctness oracle, timing method,
and GPU conditions identical.

Before treating TLX latency as the target, autotune it once on the target GPU.
Use the same L2-cache policy as the candidate, reject resource failures, and
check every measured configuration against the numerical oracle. Persist the
complete winner: kernel constexpr kwargs plus `num_warps`, `num_stages`, and
cluster launch metadata. Native autotuner timings alone are insufficient when
their cache policy differs from the outer harness.

Time direct, pre-bound kernel launches for kernel-to-kernel comparisons. CUDA
events around a Python registry wrapper can include idle-stream time between the
start event and kernel submission, making a fast TLX kernel look slower because
of routing or descriptor-setup work. Report wrapper latency separately when it
is itself a deployment target; never compare direct latency on one leg with
wrapper latency on the other.

## Prove The Transformation Fired

After every AutoWS source or compiler change, compile from a fresh cache and
inspect the exact final TTGIR used by the benchmark. It must contain both
`ttg.warp_specialize` and materialized `partitionN(` regions. Record partition
roles and warp counts. If either is absent, report that AutoWS did not fire and
do not attribute correctness or performance to AutoWS.

## Align Data Movement And Persistence Before AutoWS

When a semantically matching TLX implementation exists, use this staged order:

1. Confirm the candidate and TLX kernel implement the same input layouts,
   operation, epilogue, output dtype, and numerical tolerance.
2. In plain Triton, align the important data-movement structure: descriptor/TMA
   loads and stores where legal, tile shapes, epilogue subtiling, and persistent
   grid traversal. Preserve a pointer-based fallback when descriptor alignment,
   shape, or boundary requirements are not satisfied.
3. Validate correctness and benchmark this non-AutoWS structural baseline. This
   separates gains from removing launches, adopting TMA, or becoming persistent
   from gains due to warp specialization.
4. Only then annotate the persistent loop and enable AutoWS in its isolated
   process. Prove materialized partitions in final TTGIR before comparing it
   with either the plain-Triton baseline or TLX.

Do not enable AutoWS first and simultaneously rewrite pointer I/O to TMA: if the
candidate fails or changes speed, the cause is ambiguous. Keep each step as one
correctness-gated, measured hypothesis.

## Match Structure Before Micro-Tuning

After the plain-Triton TMA/persistent baseline is established, compare the final
AutoWS and TLX TTGIRs in this order: loop/persistence, compiled tile, operation
categories per partition, software-pipeline order, TMEM/SMEM reuse and staging
depth, partition warp counts and CTA topology, then register budgets. Compare
load, GEMM, compute, and epilogue roles by operations rather than numeric
partition IDs. Use the TLX implementation as design evidence, not code to import
or call from the candidate.

Change one structural axis at a time. Re-run the final-TTGIR gate, correctness,
and isolated timing after each change. A correct but slower first AutoWS result
is expected; promotion still requires stable end-to-end improvement.
