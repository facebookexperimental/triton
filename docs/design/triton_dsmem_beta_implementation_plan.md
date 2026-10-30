# Triton-beta DSMEM collective implementation plan

**Status:** implemented beta K=1 collective

## Scope

This document describes the compiler support available before the optional
online-softmax extension. The frontend marker combines one CTA-local f32 scalar
or one statically shaped, nonempty f32 tensor across a one-dimensional physical
cluster. Supported combiners are add, max, and min.

The compiler does not partition input work or assign output ownership. Kernels
must explicitly select rank-strided shards and provide one CTA-local partial to
the marked reduction. Results are replicated in every participating CTA.

Identity-safe variadic online-softmax state is a separate extension. Until that
extension is present, a marked `(m, l)` reduction is rejected during formation.

## Frontend contract

`tl.reduce`, `tl.sum`, `tl.max`, and `tl.min` accept `multi_cta=True`. The
frontend places `tt.multi_cta` on the emitted `tt.reduce`; ordinary `tl.range`
loops have no multi-CTA option.

An active kernel must:

- launch a physical cluster through `ctas_per_cga`;
- use one non-unit cluster dimension with P equal to 2 or 4;
- keep logical `ttg.num-ctas` equal to one;
- derive local rank through `program_id(cluster_axis) % P` or
  `cluster_cta_rank()`;
- use an unordered f32 add, max, or min reduction; and
- satisfy the convergent-placement and memory-safety contracts below.

P=1 removes the marker and leaves the CTA-local reduction unchanged. Active
collectives are supported on NVIDIA compute capabilities 90, 100, and 103.
SM90 and SM100 have full-pipeline hardware coverage; SM103 currently has
lit-level coverage because it requires a PTX 8.8-capable toolchain.

## Semantic operation

Formation removes `tt.multi_cta` and creates compiler-owned
`ttng.cluster_reduce` immediately after the CTA-local reduction. The operation
carries `cluster_axis`, `group_size`, and an isolated combine region. It has
shared-memory effects and returns the same result in every physical CTA.

For tensor results, version one accepts exactly one statically shaped,
nonempty f32 tensor and a combine region containing one direct binary add,
max, or min operation. General variadic tensor state is not supported.

## Formation legality

The formation pass runs immediately after TTIR-to-TTGIR conversion. It rejects:

- unsupported targets, cluster shapes, group sizes, types, or combiners;
- explicitly ordered marked reductions;
- calls and warp-specialized functions;
- conditional, nested, or rank-divergent collective placement;
- nonconstant combine-region captures;
- unsupported or unproved global memory effects; and
- writes whose cross-CTA disjointness cannot be established by the supported
  ownership patterns.

A collective may be in the function entry block or directly inside one
`scf.for` whose bounds and step are independent of cluster rank. This permits
uniform persistent kernels while preserving convergence.

Supported writes include canonical rank-strided pointer stores, descriptor
stores with one rank-dependent tile dimension, canonical rank-zero side
stores, and intrinsic CTA-local scratch. Atomics, reducing writes, scatter,
unmodeled memory effects, and unsupported address projections fail closed.

The analysis rejects provable same-root aliasing, but runtime aliasing between
distinct kernel arguments is a caller obligation. Output/statistics arguments
and row/base slices must be disjoint as documented by the frontend API.

## TTGIR lowering

An early NVIDIA pass lowers each verified `ttng.cluster_reduce` to ordinary
allocator-visible TTG/TTNG primitives:

1. Allocate one dense local `[P, S]` f32 buffer and one mbarrier.
2. Initialize the barrier and expect exactly the bytes sent by the other
   P-1 CTAs.
3. Execute an unconditional cluster arrive/wait so every peer observes the
   initialized DSMEM and barrier state.
4. Store the local partial and asynchronously publish it to every other CTA.
5. Wait for remote completion, load rank slots in ascending order, and apply
   the verified combiner.
6. Execute a CTA-local barrier before allocator-managed scratch can be reused.

For a uniform persistent loop, every iteration reinitializes the barrier and
executes the initialization rendezvous. This prevents one CTA from reusing
scratch while peers are still consuming the previous iteration.

The lowering emits no dedicated raw PTX/NVVM cluster-reduce conversion and no
leader-only gather/broadcast protocol. Normal allocation and barrier lowering
own the concrete resources.

## Implementation map

- `python/triton/language/core.py`: reduction marker and caller contract.
- `python/triton/language/semantic.py`: `tt.multi_cta` emission.
- `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`:
  `ttng.cluster_reduce` and its terminator.
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`: semantic verification.
- `third_party/nvidia/hopper/lib/Transforms/FormMultiCTACollectives.cpp`:
  formation, convergence, ownership, and memory-safety validation.
- `third_party/nvidia/hopper/lib/Transforms/LowerClusterReduceToTTGIR.cpp`:
  allocator-visible DSMEM expansion.
- `third_party/nvidia/backend/compiler.py`: early pipeline scheduling.
- `test/Hopper/cluster_reduce_ttgir.mlir` and
  `test/TritonNvidiaGPU/invalid.mlir`: formation/lowering and invalid-IR tests.
- `python/test/unit/language/test_multi_cta_reduction.py`: frontend and
  full-pipeline consumer coverage.

## Consumer coverage at this boundary

The LayerNorm consumers explicitly partition columns by physical CTA rank and
mark independent mean and variance reductions. They cover scalar and tensor
add reductions, P=1/2/4, ragged rows, tails, replicated statistics, and
rank-strided output ownership.

The online-softmax tutorial and TritonBench provider are intentionally not part
of this boundary because they require the separate K=2 combiner-recognition
extension.

## Non-goals

- compiler-owned work partitioning;
- P-CTA tensor encodings or changes to global program-ID semantics;
- cluster sizes other than P=2 or P=4;
- clusters with more than one non-unit dimension;
- f16/bf16 collective state, integer/product/arg reductions, or K>1 state;
- conditional or rank-divergent collectives;
- AutoWS composition;
- AMD or a generic target interface; and
- profitability modeling or automatic cluster selection.
