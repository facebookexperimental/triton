# RLLayer fused-kernel tuning map

Last updated: 2026-09-25

This document maps the current RLLayer tuning backlog to reusable TLX patterns
already present in the GEO BF16 fused-kernel library. It is a tuning aid, not a
kernel registry or a statement that any existing router is a semantic match.

Sources:

- `~/fbsource/fbcode/triton/tools/post-fuser/docs/rllayer_fused_kernel_tuning_tasks.md`
- `~/fbsource/fbcode/gem/next_gen/geo/kernel_library/registry.json`
- `~/fbsource/fbcode/gem/next_gen/geo/kernel_library/benchmark_shapes.json`
- the current router and shape implementations named below

GEMM shapes use `(M, N, K)`. Batched and compound operators spell out their
logical tensor shapes. All task inputs and materialized activation boundaries
below are BF16 unless stated otherwise; GEMM accumulators and normalization
statistics are FP32.

## RLLayer tuning workloads

### Forward

| Task | Core shape | Operator and fusion boundary | Required outputs / constraints |
|---|---|---|---|
| `T289775012` | two `(5120, 4096, 8192)` GEMMs; output viewed as `[81920,256]` | two down GEMMs -> residual add -> weighted RMSNorm -> direct stores | Preserve post-add BF16 state, FP32 rstd, normalized BF16 output, and destination partitioning. The overlapping partial boundary already has a small TLX win. |
| `T289757910` | `(5120, 512, 8192)` | GEMM -> weighted RMSNorm D=512 -> SiLU | Integrated ABI also saves raw GEMM output, normalized/SiLU output, and rstd. |
| `T289757930` | two `(5120, 1024, 2048)` GEMMs | GEMM -> weighted RMSNorm D=1024 -> SiLU | Preserve both raw GEMM outputs, both activated outputs, and rstd. |
| `T289757870` | two `(5120, 16384, 4096)` packed gate/up GEMMs | two GEMMs -> split gate/up -> `SiLU(gate) * up` | Training ABI saves packed GEMM output; a direct-hidden-only route is a different contract. |
| `T289914992` | two `(5120, 4096, 8192)` down GEMMs | materialized SwiGLU -> down GEMM -> residual add -> weighted RMSNorm -> direct stores | Full fusion must preserve hidden, post-add, rstd, and destination outputs. The materialized-SwiGLU hybrid is the current winning boundary. |
| `T290058050` | two `(2097152, 256, 512)` GEMMs | `concat(SiLU(U), LayerNorm(X) * U)` -> GEMM + residual | Mean/rstd are already produced. Candidate removes each 2-GiB reconstructed activation while preserving its BF16 boundary. |
| `T289757859` | two `(1310720, 16, 64)` GEMMs | skinny GEMM -> `[5120,16,256]` grouped layout | Write the destination layout directly. |
| `T289757878` | `(1310720, 64, 112)` | small-K GEMM -> three layouts | Emit `[5120,256,64,1]`, `[5120,64,256]`, and `[5120,256,64]` without a materialized common output. |
| `T289757895` | `(1310720, 80, 64)` | skinny GEMM -> `[5120,80,256]` grouped layout | Write the grouped layout directly. |
| `T289757919` | BMM `[5120,32,256] @ [5120,256,64]` | BMM -> concat/copy into compact and padded layouts | Copy existing feature slices and write both layouts without materializing the BMM output. |

### Backward

| Task | Core shape | Operator and fusion boundary | Required outputs / constraints |
|---|---|---|---|
| `T289901422` | `(4096, 8192, 5120)` | reconstruct `BF16(SiLU(gate) * up)` in the `dW_down` GEMM prologue | Preserve the BF16 activation boundary, FP32 GEMM accumulation, and BF16 output. |
| `T290084482` | two `(2097152, 256, 1024)` GEMMs | GEMM -> affine LayerNorm backward -> residual add | Emit dx and FP32 dweight/dbias partials; a second kernel finalizes dweight/dbias. |
| `T289840347` | two `(5120,1024,2048)` and one `(5120,512,2048)` | GEMM -> RMSNorm/SiLU backward | Preserve BF16 GEMM output for the sibling dweight reduction. D=512 now has a standalone TLX win. |
| `T289881412` | rows=5120, D=8472 | weighted RMSNorm backward, tiled multipass | No GEMM. Emit dx plus FP32 dweight partials and finalize dweight. This already wins substantially. |
| `T289845731` | reduction width 40; D in `{2048,1024,2048,1024,2048,512}` | six FP32 partial-tensor -> BF16 N:1 reductions | Only D=512 currently wins; keep FP32 accumulation through the final cast. |
| `T289866753` | two `(5120, 2048, 1024)` addmm regions | addmm -> two-pass normalization/SiLU consumer -> deferred aliasing store | GEMM output must materialize as BF16 between passes; defer the destination write until all aliased reads complete. |
| `T290048886` | projection `(2097152,768,256)` and dW `(256,1024,2097152)` | one saved affine-LayerNorm activation feeds projection and transposed dW GEMMs | Preserve BF16 normalized activation semantics while avoiding two independent reconstructions where possible. |
| `T290079238` | two `(512,256,2097152)` dW GEMMs | backward outputs plus reconstructed `concat(SiLU(U), LayerNorm(X) * U)` in a split-K GEMM prologue | Keep dx, du, dgamma, dbeta, BF16 reconstructed-Y boundary, FP32 MMA/split reduction, and BF16 dW. |

The `T289914992`, `T289775012`, and `T289757870` boundaries overlap. Do not
sum their latency opportunities. Two occurrences with the same semantics and
ABI are one GEO owner with one shape route, not two public kernels.

## Relevant GEO TLX schedule donors

These entries are implementation references. A task needs a new semantic owner
or shape route unless its complete graph, ABI, numerical boundaries, effects,
rank, and layouts match an existing contract.

| GEO entry | Registered production shapes | Reusable mechanism | Most relevant tasks |
|---|---|---|---|
| `mm` (baseline) | includes `(262144,512,1024)`, `(3836160,256,256)`, `(626688,256,256)`, `(2304,4096,106560)` | Blackwell persistent TLX GEMM, TMA producer, TMEM accumulator, CLC scheduling, Split-K variants | Base schedule for every GEMM-heavy task; none of the large RLLayer shapes is an exact registered route. |
| `addmm_split_sigmoid_mul` | `(4096,384,256)`, `(65536,384,256)`, `(294912,384,256)`, `(1179648,384,256)` | Gate/up GEMM with split, sigmoid, and multiply in the epilogue | `T289757870`; pointwise/epilogue scheduling for `T289914992`. |
| `mul_mul_sigmoid_grad_cat_matmul` | `(4096,256,384)`, `(294912,256,384)` | Four-task warp specialization: TMA loader, computed BF16 prologue, MMA, epilogue; concatenation is never materialized | Closest prologue-fusion donor for `T289901422` and `T290079238`. |
| `reshape_mul_mul_sigmoid_grad_cat_matmul__returns_grad_projection` | `(1152,256,256,384)` | Rank-three version of computed pointwise/concat prologue feeding an FP32 projection GEMM | `T289901422`, `T290079238`, and ABI design for saved outputs. |
| `layernorm_addmm__affine__returns_layernorm_mean_layernorm_rstd` | `(192,512,512)`, `(4096,512,384)`, `(1024,384,256)`, `(1024,384,384)`, `(128,128,64)`, `(3159809,384,384)` | Affine LayerNorm -> BF16 boundary -> GEMM. Default route uses a single-X-read warp-specialized full-K A ring. | `T290058050`, projection half of `T290048886`; useful contrast for full-K versus streaming prologues. |
| `layernorm_addmm__affine__returns_xnorm_layernorm_mean_layernorm_rstd` | `(1024,384,256)`, `(4096,512,384)`, `(3159809,384,384)`, `(128,128,64)` | Same prologue while returning the normalized BF16 activation | Saved-activation ABI for `T290048886`. |
| `mm_rmsnorm` | `(1024,384,384)`, `(1024,512,256)`, `(1179648,1024,512)`, `(4096,1024,512)` | GEMM followed by full-row RMSNorm using a two-pass epilogue and cross-CTA DSMEM reduction | `T289775012`, `T289757910`, `T289757930`, `T289914992`. |
| `mm_layernorm_grad__affine__returns_layernorm_weight_grad_layernorm_bias_grad` | `(4096,256,512)`, `(1024,128,256)`, `(1024,384,384)`, `(1179648,256,512)` | GEMM + affine LayerNorm backward; two DSMEM row reductions and TMEM staging | Primary semantic/schedule donor for `T290084482`. |
| `mm_add_layernorm_grad_add__affine` | `(4096,384,384)`, `(65536,384,384)`, `(294912,384,384)`, `(1179648,384,384)` | GEMM, incoming-gradient add, affine LayerNorm backward, and residual add in one TLX owner | Residual/direct-store extension donor for `T290084482`; its extra pre-normalization add means it is not an exact contract match. |
| `mm_rmsnorm_grad` | `(4096,256,512)`, `(1024,128,256)`, `(1179648,256,512)` | GEMM + RMSNorm backward with producer TMA, TMEM staging, and cross-CTA reduction | Primary donor for `T289840347`. |
| `matmul_add_rmsnorm_grad__weighted__returns_rmsnorm_weight_grad` | `(1152,32768,12800)` | Multi-stage TLX matmul/add, row-stat finalization, input/weight-gradient pass, and final cast | `T290084482`, `T289840347`, and multipass ideas for `T289881412`. |
| `transpose_mm_sum` | `(8192,256,256)`, `(262144,512,1024)`, `(294912,512,384)`, `(1152,49152,512)`, plus smaller routes | Transposed wgrad GEMM and column-sum reduction sharing the K traversal; Split-K finalization where needed | `T289845731`, `T290079238`, and the dW side of `T290048886`. |
| `transpose_mm_mm` | `(4096,4096,512)`, `(1152,49152,512)`, `(1152,1024,24576)`, `(294912,512,384)`, `(3159809,384,384)` | Two GEMMs share one staged input; includes a small-K transposed/reused-A path | Dual-consumer structure for `T290048886`. |
| `matmul_add_add_reshape_cat_rmsnorm__weighted__returns_rmsnorm_rstd` | `(1152,256,128,256)` | GEMM, two adds, reshape/cat, weighted RMSNorm, and returned rstd with explicit BF16 boundaries | Closest semantic seed for `T289775012`; downstream half of `T289914992`. |
| `linear_reshape_matmul_add_add_reshape_cat_rmsnorm__weighted__returns_rmsnorm_rstd` | `(1152,12800,32768,256)` | Two GEMM stages followed by residual assembly and weighted RMSNorm | Full-chain scheduling ideas for `T289914992`. |
| `reshape_mm_reshape_permute_contiguous_add` | `768x256x8x512x384` | Direct reshaped/transposed output from a TLX GEMM plus repeated bias | Direct-layout stores for `T289757859`, `T289757878`, and `T289757895`. |
| `mm_reshape_permute_contiguous_reshape` | `768x512x384x2048` | Backward TLX GEMM writing its final permutation directly | Same three layout tasks. |
| `matmul_einsum_bij_bik_jk_sum_transpose_slice` | `768x256x64x2600` | Shared gradient staging, direct unpadded/transposed dgrad, and split wgrad/bias finalization | `T289757919` and multi-output reduction/layout designs. |

## Current `T290079238` TLX prototype

The current OSS prototype is outside GEO at
`third_party/tlx/tutorials/bwd_layernorm_mul_dropout_buf157_tlx.py` in the
MetaMain Triton checkout. It starts from the tuned FP32-workspace Blackwell TLX
GEMM and replaces the A-operand TMA load with a software producer that writes
reconstructed BF16 Y tiles directly to the MMA SMEM ring.

Production shape and schedule:

```text
GEMM:       (M,N,K) = (512,256,2097152)
tile:       BM256 / BN256 / BK64
cluster:    2 CTA
pipeline:   4 SMEM buffers / 1 TMEM buffer
split-K:    76 (one resident cluster per two GB200 SMs)
epilogue:   32 output subtiles, FP32 split workspace + final reduction
producer:   16 warps, 128 registers, PROLOGUE_K=64
```

The producer computes either `SiLU(U)` or
`(LayerNorm(X; mean,rstd,gamma,beta) * U)` in FP32, rounds once to BF16 in
shared memory, and feeds a two-CTA FP32 MMA. It does not fuse the preceding
backward kernel and its split-K finalization is a second launch.

GB200 standalone results, `warmup=5`, `samples=20`, `repeat=3`:

| Region | Best median | Relative result |
|---|---:|---:|
| materialize Y + PyTorch/cuBLAS | `2.076 ms` | `1.000x` |
| fused TLX, FP32 split workspace | `2.547 ms` | `0.815x` |
| fused TLX, FP16 split workspace | `2.538 ms` | `0.819x` |
| original fused Triton seed | `3.510 ms` | `0.592x` |

FP32-workspace dW relative-L2 error for seeds 0/1/2 is
`3.97e-4 / 4.19e-4 / 4.40e-4`. FP16 workspace is marginally faster but gives
`8.53e-4 / 8.86e-4 / 8.27e-4`, close to the `1e-3` acceptance limit. Keep FP32
as the default tuning baseline.

The main performance lesson is that the 2-CTA GEMM duplicates the computed A
prologue in both CTAs, whereas a normal A tensor is loaded once and multicast.
The complete fused region also runs the preceding backward kernel, which has
already evaluated much of the LayerNorm/SiLU math. A useful next design should
either:

1. fuse dx/du/dgamma/dbeta production into the TLX producer so the elementwise
   work is shared; or
2. publish each reconstructed A tile once for multiple consumers/CTAs/output-N
   tiles rather than recomputing it.

## Current `T289901422` TLX prototype

The OSS benchmark and TLX kernel are in
`third_party/tlx/tutorials/bwd_swiglu_gemm.py` and
`bwd_swiglu_gemm_tlx.py`. They reproduce the `(4096,8192,5120)` transposed-dW
GEMM and preserve the FP32 SwiGLU computation, BF16 pre-dot boundary, FP32 MMA
accumulation, and BF16 output.

The best exact schedule uses `BM128/BN128/BK64`, four M accumulator groups, one
A slot, three computed-B slots, sixteen producer warps, and a four-warp
eight-subtile epilogue. Reusing each reconstructed B tile across four MMAs is
the largest valid improvement found. Locked GB200 best-of-five medians are
`272.320 us` unfused, `798.368 us` for the fused Triton seed, and `813.760 us`
for TLX. All three verification seeds happen to be bit exact; exact equality is
not a requirement because GEMM reduction reassociation is allowed.

The negative result is structural. Plain `tlx.ops.mm` is close to cuBLAS at
this shape (`254.714` versus `230.817 us`), but exact sigmoid reconstruction is
repeated for every output-M group. NCU reports 40 registers/thread, 119.11 KiB
dynamic SMEM, 37.5% theoretical occupancy, and only 27.6% compute throughput.
The optional GEO-style packed-FP32 `tanh.approx` sigmoid, tuned to
`BM128/BN128/BK128` with two computed-B slots, lowers fused TLX to `495.360 us`.
It preserves the BF16 pre-dot boundary, FP32 accumulation, and BF16 output;
relative-L2 is `4.54e-4` to `4.61e-4` and max abs is `0.015625` across seeds
0/1/2. The corresponding 2-CTA form is slower at `539.904 us`. NCU reports 70
registers/thread, 201.01 KiB dynamic SMEM, one resident block, 37.5% occupancy,
48.3% compute throughput, and 17.3% DRAM throughput. A correct exact 2-CTA
route measures `884.064 us`; an attempted four-CTA DSMEM broadcast did not
complete and is not exposed by the driver.

## Current `T290084482` TLX prototype

The OSS prototype is in
`third_party/tlx/tutorials/bwd_weighted_layernorm_gemm_tlx.py`, with its
standalone reference in `bwd_weighted_layernorm_gemm.py` and detailed results
in `bwd_weighted_layernorm_gemm.md`.

The winning schedule uses `BM64/BN128/BK64`, two CTAs split across N, six SMEM
stages, two TMEM buffers, eight epilogue warps, and one static persistent wave
(152 CTAs on GB200). The CTAs exchange the two LayerNorm row reductions through
double-buffered DSMEM. Dweight/dbias remain FP32 in per-CTA register
accumulators and are atomically finalized once per persistent CTA. The GEMM and
dx BF16 boundaries are preserved exactly.

On a locked GB200, the tuned TLX kernel measures `2.462 ms`, versus `7.715 ms`
for the original fused seed and `2.332 ms` for the optimized OSS unfused
composition. It is a 3.13x improvement over the seed, but is still 5.6% slower
than unfused. Final relative-L2 is at most `5.36e-6` across seeds 0/1/2;
dweight/dbias match after their BF16 output cast.

The rejected pair-CTA schedule avoids duplicated A traffic and gives a strong
bare GEMM (~`1.04 ms`), but leaves each CTA responsible for all 256 LayerNorm
columns and measures `3.60 ms` or worse after fusion. A multicast on the
winning N-split schedule is correct but neutral/slightly slower because its
cross-CTA readiness handshake offsets the saved traffic. Static grid-stride
scheduling is the latest win: CLC measures `2.53-2.59 ms` on this uniform grid.

NCU reports 168 registers/thread, about 154 KB dynamic SMEM, no local-memory
traffic, and one resident CTA per SM. Lower max-register limits regress and
cannot raise occupancy because SMEM is also limiting. Static persistence lowers
NCU instrumented duration from `3.93` to `3.70 ms`, raises tensor-pipe activity
from `40.1%` to `42.6%`, and reduces long-scoreboard stalls from `22.0%` to
`20.7%`; barrier stalls remain about `23.7%`.

## Current `T290048886` TLX prototype

The OSS reproducer is in `third_party/tlx/tutorials/bwd_saved_layernorm_gemm.py`;
the TLX projection and transposed-dW routes are in
`bwd_saved_layernorm_gemm_tlx.py`. Both consume saved mean/rstd and reconstruct
the affine-LayerNorm BF16 boundary in their GEMM prologues.

The projection winner uses `BM64/BN256/BK128`, two A slots, two B slots, two
TMEM accumulators, and one static persistent CTA per GB200 SM. One normalized A
tile is reused across all three output-N tiles. The dW winner uses
`BM256/BN256/BK64`, one CTA, three SMEM stages, two 128-row MMA groups, and
split-K 38 with an FP32 workspace.

Locked GB200 best-of-five medians (`warmup=5`, `samples=20`):

| Region | Unfused | Original fused seed | Tuned TLX |
|---|---:|---:|---:|
| projection `(2097152,768,256)` | `1.125 ms` | `2.221 ms` | `1.295 ms` |
| dW `(256,1024,2097152)` | `1.182 ms` | `2.168 ms` | `1.791 ms` |
| complete fan-out | `1.900 ms` | `4.198 ms` | `3.048 ms` |

Projection is bit-exact to the fused reference for seeds 0/1/2. dW
relative-L2 against materialized-BF16 PyTorch/cuBLAS is
`5.84e-4 / 5.91e-4 / 5.94e-4`, within the `1e-3` task limit. TLX improves the
two fused seeds by 1.72x and 1.21x, respectively, but the complete fused route
is still 60% slower than unfused because it reconstructs the common activation
independently for the two incompatible GEMM orientations.

NCU reports 128 registers/thread and 205.13 KiB dynamic SMEM for projection;
dW uses 60 registers/thread and 205.32 KiB. Both are limited to one resident
block by registers and shared memory. A weight-stationary projection schedule
that retains the two weight tiles measured `1.333 ms` at BM64 and `1.892 ms` at
BM128, so reusing A across the three N tiles remains preferable. For dW,
split-K 38 improved on split-K 76 (`~1.80` versus `~1.82 ms`); split-K 19 and
152 measured `3.37` and `1.89 ms`. A BN128 route measured `3.33 ms`; the
tested two-CTA route was numerically invalid.

This result is a useful negative boundary: optimizing the two prologue-fused
GEMMs independently cannot beat the unfused fan-out, which computes and stores
the common 1-GiB BF16 activation once. A next design needs one shared
producer/materialization serving both GEMM orientations, not further local
hill-climbing of either standalone route.

A follow-up monolithic kernel distributes dW as eight `(128,256)` tiles and
folds six `(64,128)` projection tiles into the same 152 persistent CTAs. It is
bit-exact but measures `4.66 ms`; physical eight-CTA clustering measures
`7.90 ms`, and its dW-only ablation is `2.94 ms`. The competitive dW schedule
needs two concurrent M128xN256 accumulator groups and consumes all 512 TMEM
columns, so projection cannot be added without falling back to this weaker
half-width dW schedule.

## Current `T289840347` D=512 TLX prototype

The OSS benchmark and TLX kernel are in
`third_party/tlx/tutorials/bwd_rmsnorm_silu_gemm.py` and
`bwd_rmsnorm_silu_gemm_tlx.py`. The winning schedule uses `BM64/BK64`, four A
stages, two B stages, two N=256 TMEM accumulators, and two parallel eight-warp
RMSNorm/SiLU epilogue tasks. Each epilogue task also emits its FP32 dweight
partial, removing that standalone launch; the existing final reduction remains.

Locked GB200 best-of-five medians are `147.344 us` unfused, `99.136 us` for
the compact Triton fused seed, and `123.104 us` for TLX. TLX is 16.5% faster
than unfused; the seed is 32.7% faster. Projection, dx, and dweight are bit-exact
for seeds 0/1/2. NCU reports 128 registers/thread, 181.11 KiB dynamic SMEM, and
one resident block; instrumented tensor-pipe activity is 30.0%. Four A stages
improve the TLX path by about 3 us over three, while a third B stage regresses
to about 212 us and exceeds the useful SMEM pipeline depth.

## GEO Kernel Optimizer

GEO has a general TLX-capable Kernel Optimizer under
`fbcode/gem/next_gen/geo/agents/kernel_optimizer`; it is not a TLX-only agent.
It freezes a GEO-style accuracy and benchmark contract, uses NCU for iteration,
and separates setup/optimization execution from independent supervision. Its
`non-geo-kernel-launcher-skill.md` can adapt an external or custom kernel only
after the source is available inside the selected fbsource checkout, so the OSS
prototype here was tuned manually without modifying GEO.

The most useful GEO implementation patterns for this task were:

- static persistent grid-stride scheduling for uniform shapes, reserving CLC
  for ragged or variable work;
- a broad joint search over BM/BN/BK, SMEM stages, TMEM buffers, warp count, and
  epilogue subtiles;
- loading non-GEMM epilogue inputs after the K-loop in the TMA producer, when
  the added SMEM and barrier cost is justified;
- keeping parameter gradients in FP32 across persistent iterations and paying
  one atomic flush per resident CTA; and
- using explicit low-register producer/MMA tasks and checking NCU local-memory
  traffic before treating register count itself as a problem.

For this shape, only static persistence improved the result. Extra epilogue TMA
staging, cached gamma, more/fewer pipeline buffers, alternate tile widths,
subtiling, and lower register caps all regressed. BF16 dweight math was also
rejected because relative-L2 rose above the `2e-3` contract.

## Current `T290058050` TLX prototype

The OSS prototype is in
`third_party/tlx/tutorials/fwd_layernorm_mul_gemm_tlx.py`, with its standalone
reference in `fwd_layernorm_mul_gemm.py` and detailed results in
`fwd_layernorm_mul_gemm.md`.

The winning static-persistent schedule uses `BM64/BN256/BK128`, separate
two-entry A/B shared-memory rings, one TMEM accumulator, and an eight-warp
computed-prologue producer. The LayerNorm statistics pass batches 16 rows per
program. The 2 GiB concatenated BF16 activation is never written to global
memory.

On a locked GB200, the tuned TLX path measures `1.446 ms`, versus `5.011 ms`
for the OSS fused Triton seed and `1.754 ms` for activation materialization plus
PyTorch/cuBLAS. It is 3.47x faster than the seed and 17.6% faster than unfused.
Projection relative-L2 against the FP32-accumulating BF16-boundary reference is
at most `6.01e-6` over seeds 0/1/2 and at most `1.78e-5` against the fused seed.

NCU reports 128 registers/thread, 172.31 KiB dynamic SMEM, 25% occupancy, and
small but nonzero local traffic (5.28 MB loads, 1.12 MB stores over the whole
launch). Lower/higher register caps and a de-unrolled producer all regress.
The GEO single-X-read full-K A-ring pattern is correct here but slower at
`~1.78 ms`: with only one output-N tile, its extra normalizer task is not
amortized by A reuse.

## Tuning guidance by fusion family

### Computed GEMM prologues

- Start from `mul_mul_sigmoid_grad_cat_matmul`, not from a plain GEMM copy. It
  already separates TMA loading, computed BF16 prologue, MMA, and epilogue.
- Maximize reuse of a reconstructed tile across output-N tiles. A prologue that
  is recomputed per N tile can lose even when it removes a multi-GiB tensor.
- Audit 2-CTA asymmetry: B is partitioned, but A is commonly replicated. A
  scalar-heavy A producer can become the critical path even when bare GEMM is
  near cuBLAS.
- Preserve the task's BF16 materialization boundary before MMA. Do not compare
  an FP32-reconstructed input against a reference that explicitly rounded Y.

### Normalization epilogues

- Prefer RMSNorm's algebraic freedom when applicable: accumulate the row
  square sum while MMA runs, then apply one row scalar to the completed
  accumulator. The extra A-buffer-depth mechanism from `D120426461` is a
  relevant latency-hiding tool because A feeds both reduction and MMA.
- LayerNorm needs both mean and variance reductions. Centering cannot generally
  be deferred as one row scaling operation. Affine bias and elementwise `* U`
  introduce additional correction terms, so use `mm_layernorm*` as the donor
  rather than assuming the RMSNorm schedule transfers directly.
- For output widths larger than one CTA tile, use DSMEM only when it avoids
  duplicate full-row work and its rendezvous cost is amortized.

### Layout and multi-output epilogues

- Reuse the direct-layout TMA-store patterns rather than materializing a dense
  GEMM output followed by transpose/reshape/concat kernels.
- Keep every public saved tensor and cast boundary in the benchmark reference;
  otherwise the baseline and candidate measure different contracts.
- For aliasing destinations, delay the store until every old-value read has
  completed. Do not trade the two-pass dependency for an unsafe single pass.

### Reductions and split-K

- Use FP32 workspace by default. A lower-precision split workspace rounds every
  split partial before final accumulation and is therefore an intermediate
  precision change, not merely a storage optimization.
- Specialize N:1 reductions by width. The current evidence supports D=512 but
  not treating all six reduction widths as one winning schedule.
- Count the split finalizer in end-to-end timing and in the fused semantic
  contract. A fast partial kernel with a slow finalizer is not a win.

## Suggested order of work

1. Promote the existing narrow wins: `T289881412`, the materialized-SwiGLU
   boundary of `T289775012`, D=512 from `T289845731`, and the D=512 route from
   `T289840347`.
2. Tune `T290084482` from the affine `mm_layernorm_grad` family rather than
   extending its current pointer-load correctness seed.
3. Revisit `T290079238` only with producer sharing across CTAs/output-N tiles
   or fusion with its backward outputs; retain FP32 workspace as the baseline.
4. Treat `T290048886` as a cross-GEMM reuse problem: its independently tuned
   projection and dW routes remain slower than materializing the shared BF16
   activation once. Resume only with a design that shares that producer across
   both GEMM orientations.
5. Treat the skinny/layout and batched-concat tasks as direct-store schedule
   work; do not begin by adding more pointwise work to a weak GEMM core.
6. Attempt the full `T289914992` boundary only after reconstructed-hidden reuse
   across N tiles is explicit.

## GEO admission checklist

Before turning any row above into a registered GEO entry:

- audit the complete public tensor/scalar ABI and normalized ATen graph;
- choose a semantic operator name with no task, buffer, or model provenance;
- create a self-contained owner with router, `contract.py`, and numeric shape
  directories; do not import another fused owner;
- add every production variant to `benchmark_shapes.json`, the accuracy test,
  benchmark, and registry;
- validate the exact BF16/FP32 materialization and reduction boundaries;
- run the structure checker, contract checker and independent matching tests,
  lint, build, full accuracy, then the dedicated-GPU benchmark; and
- require both a standalone win and a matching full-artifact win before
  promotion.
