# NVWS-MetaAWS passes

[← Back to README](README.md)

## Contents

- [Paths and scope](#paths-and-scope)
- [Pass order](#pass-order)
- [Contracts between passes](#contracts-between-passes)
- [Code map](#code-map)

## Paths and scope

On Blackwell, NVWS automatic warp specialization currently runs only for a
single-CTA cluster. Both routes use canonical data partitioning. They use
different planning paths and converge on NVWS semaphore codegen:

| Step | Default NVWS | MetaAWS–NVWS bridge (`TRITON_NVWS_USE_META=1`) |
|---|---|---|
| Data partition | Canonical `WSDataPartition` | Canonical `WSDataPartition` |
| Ordinary pre-schedule | Standard latency schedule | Meta-aware latency schedule |
| Ownership and storage planning | Heuristic TritonGPU partition scheduling for NVWS planning | MetaAWS planning; see the [bridge](meta-aws-nvws-bridge.md) |
| NVWS representation boundary | Already in NVWS form | `MetaToNVWSConvert` |
| Codegen | NVWS semaphores | NVWS semaphores |

On Blackwell, `TRITON_NVWS_USE_META=1` takes precedence over
`TRITON_USE_META_WS=1` when both are enabled. With the NVWS flag disabled,
`TRITON_USE_META_WS=1` selects the complete MetaAWS backend; it is not an NVWS
route. A 2-CTA cluster skips NVWS AWS. To use the supported multi-CTA MetaAWS
path, leave `TRITON_NVWS_USE_META` disabled.

[↑ Back to contents](#contents)

## Pass order

```text
LLM or modulo schedule                 [optional; mutually selected]
-> canonical data partition
-> assign latencies and schedule loops [only without a custom scheduler]

default NVWS:
  -> AutomaticWarpSpecialization
       TritonGPU partition scheduling
       hoist loop-invariant TMEM stores
       InsertAllocas

MetaAWS–NVWS bridge:
  -> TMA-store lowering and SinkBroadcast
  -> AutomaticWarpSpecialization
       canonical MetaAWS planning prefix
       NVWSPackEpilogueSlices
       MetaToNVWSConvert
         promote Meta scheduling scope to the NVWS physical WS root
         translate task ownership and the completed memory plan
       NVWSOrderBufferGroups
     (see meta-aws-nvws-bridge.md for both allocation variants)

common NVWS suffix:
  -> InsertSemas
  -> LowerSemaphore
       combine eligible SMEM semaphore pairs
       widen eligible TMA-load-fed SMEM backings
       AssignStagePhase
       lower semaphore IR to barriers
  -> partition loops
  -> lower warp groups
  -> terminal schedule loops
  -> multi-buffer TMA descriptors
  -> clear internal WS metadata
-> Triton software pipeliner
```

[↑ Back to contents](#contents)

## Contracts between passes

```text
default partition schedule
  fixed ttg.partition / partition.outputs / WS tags and loop schedule

InsertAllocas (default route)
  explicit producer writes and consumer reads over mutable SMEM/TMEM buffers

MetaAWS–NVWS bridge
  Meta's function-wide task scope normalized to an NVWS loop-level WS root
  completed MetaAWS ownership and memory-plan annotations translated to NVWS

InsertSemas
  semaphore operations, exact token threading, optional stage offsets,
  and synchronization schedules compatible with fixed `loop.stage` and
  `loop.cluster` assignments from the Triton software pipeliner

LowerSemaphore
  optional semaphore combining and backing widening
  -> final buffer stages and phases
  -> hardware mbarrier operations and concrete buffer views
```

Storage planning and synchronization remain separate. The default route
materializes logical communication storage in InsertAllocas and lets
InsertSemas/LowerSemaphore choose any later automatic depth. The bridge route
uses canonical MetaAWS MemoryPlanner decisions and translates their annotations
without replacing planned allocations. InsertSemas interprets the translated
buffer annotations and materializes required backings and views.

The bridge's two NVWS-specific scheduling policies are documented separately:
[merged-epilogue slice packing](pack-epilogue-slices.md) targets shorter
register live ranges in the generated pattern before MemoryPlanner, while
[buffer-group ordering](order-buffer-groups.md) selects the discovery order
used by InsertSemas after conversion.

[↑ Back to contents](#contents)

## Code map

- Backend selection and pre-AWS scheduling:
  [`compiler.py`](../python/triton/backends/nvidia/compiler.py), `make_ttgir`.
- AWS orchestration:
  [`AutomaticWarpSpecialization.cpp`](../lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp).
- Bridge behavior:
  [MetaAWS–NVWS bridge](meta-aws-nvws-bridge.md).
- Conversion:
  [`MetaToNVWSConvert.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/MetaToNVWSConvert.cpp).
- Semaphore lowering:
  [`AssignStagePhase.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp) and
  [`LowerSempahores.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSempahores.cpp).
- Pass definitions:
  [`Passes.td`](../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td).

[↑ Back to contents](#contents)
