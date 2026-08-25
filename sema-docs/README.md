# NVWS automatic warp-specialization documentation

This directory describes two NVWS automatic warp-specialization planning
paths: heuristic NVWS planning and the MetaAWS–NVWS bridge. Both paths use
NVWS semaphore codegen from `NVWSInsertSemas` through hardware barrier
lowering. It is written for a reader familiar with MetaAWS.

The documents describe algorithms and contracts. Source is authoritative.

## Reading order

1. [NVWS-MetaAWS passes](nvws-aws-overview.md)
2. [MetaAWS–NVWS bridge](meta-aws-nvws-bridge.md)
   - [InsertAllocas](insert-allocas.md)
   - [Packing merged-epilogue slices](pack-epilogue-slices.md)
   - [Ordering NVWS buffer groups](order-buffer-groups.md)
3. [Semaphore Insertions](insert-semas/overview.md)
   - [ACCESS-DAG](insert-semas/access-dag.md)
   - [SYNC-DAG](insert-semas/sync-dag.md)
   - [EMIT-IR](insert-semas/emit-ir.md)
4. [AssignStagePhase and LowerSemaphore](assign-stage-phase-and-lower-semaphores.md)

## Flags

### MetaAWS

- `TRITON_USE_META_WS=1` uses the complete MetaAWS backend when
  `TRITON_NVWS_USE_META` is not enabled.

### NVWS

Both default NVWS and the MetaAWS–NVWS bridge currently run only for a
single-CTA cluster. For supported 2-CTA warp specialization, select the
complete MetaAWS backend instead.

- `TRITON_NVWS_USE_META=1` selects the single-CTA
  [MetaAWS–NVWS bridge](meta-aws-nvws-bridge.md): canonical MetaAWS planning
  followed by NVWS synchronization and lowering. On Blackwell, this takes
  precedence over `TRITON_USE_META_WS` when both are enabled.
- `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1`, together with
  `TRITON_NVWS_USE_META=1`, selects the bridge's NVWS allocation variant. It
  converts ownership early, runs `NVWSInsertAllocas`, lets the canonical MetaAWS
  memory planner consume the resulting NVWS descriptor producers, and performs
  the final conversion after planning. This variable is not currently part of
  the compilation-cache key; clear the cache or force recompilation when
  toggling it.
- `NVWS_USE_SSA_TMEM=1` allows eligible cross-partition SSA communication to
  use TMEM. It applies on the default NVWS path and, on the bridge route,
  only when `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` is enabled.
- `NVWS_INSERT_SEMA_DUMP_DAG=1` dumps the completed InsertSemas SYNC-DAG,
  including its access and ownership annotations.
- `TRITON_DUMP_WS_GRAPHS=/path` dumps canonical MetaAWS memory-planner
  channel/allocation graphs and therefore applies to the bridge route, not
  default NVWS.

### Other

- `TRITON_USE_LLM_SCHEDULE=1` uses the LLM scheduler. This variable is not
  currently part of the compilation-cache key; clear the cache or force
  recompilation when toggling it.
- `TRITON_USE_MODULO_SCHEDULE=<1|sms|exhaustive|random>` uses modulo scheduling
  instead of ordinary latency scheduling.
- `TRITON_FP8_PROMOTE_TO_TMEM=0|1` controls FP8 LHS promotion into TMEM.
- `TRITON_ALWAYS_COMPILE=1` forces recompilation instead of using cached
  compilation.
- `MLIR_ENABLE_DIAGNOSTICS=warnings` enables compiler warnings.
- `MLIR_ENABLE_DUMP=1` dumps IR around compiler passes.
