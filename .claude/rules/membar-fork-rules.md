---
globs:
  - "lib/Analysis/Membar.cpp"
  - "include/triton/Analysis/Membar.h"
  - "lib/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.cpp"
  - "include/triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
  - "lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeMBarrierArrivals.cpp"
  - "lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp"
  - "test/Analysis/test-membar-ttng.mlir"
  - "test/TritonNvidiaGPU/tmem_barrier_insertion.mlir"
---

# Membar: fork rules that upstream picks must keep

Upstream's Membar series (#11481, #11693, #11573 and related, picked in
ce0aa8059) adds barriers that make the TLX FA sm100 kernels about 2.5% slower.
The fork keeps the series and adds the rules below on top (PR #5259). When an
upstream pick conflicts in these files, resolve it so every rule survives.

| Rule | Where | Lit cases that fail if it is dropped |
|---|---|---|
| Waits, fences, arrivals, expects and commits carry no payload: track them as `syncIssuers`, not `effectIssuers` | `carriesNoPayload`, `getThreadEffects`, `requiresThreadSyncFor` | `@elected_issue_after_waits` |
| An elected MMA, commit or TMA copy in a single-warp region, and a named barrier that counts every thread, wait for the whole region | `waitsForWholeRegion` | `@elected_issue_after_waits`, `@partial_named_barrier_after_wait` |
| MMAs and commits share the warp-zero leader issuer | `getThreadSyncInfo` | `@elected_issue_after_waits` |
| Weak shared and tensor memory stores and allocs do not notify a peer | `mayNotifyPeer` | `@notifications_after_waits`, `@wait_before_divergent_notification` |
| Prints and asserts need no rendezvous | `getThreadSyncInfo` | `@mbarrier_wait_before_print`, `@completion_before_divergent_print` |
| Synchronize before a divergent branch, never inside it | `DivergentControlFlow`, `requiresSyncBeforeBranch`, TMEM `publishesUnderBranch` | `@wait_before_divergent_notification`, `@tmem_load_then_divergent_named_barrier` |
| A pending barrier init or inval is synchronized before an MMA completion or a per-thread arrival | `missesBarrierLifetime` | `@init_before_mma_completion`, `@sync_before_divergent_branches` |
| A predicated per-thread arrival does not arrive for every thread | `arrivesFromEveryThread` | `@predicated_perthread_arrive`, `@tmem_load_then_predicated_perthread_arrive` |
| A named barrier is a rendezvous only if its count covers the region | `countsEveryThread` | `@partial_named_barrier_after_wait`, `@tmem_load_then_partial_named_barrier` |
| Two executions of one store into a view fixed for the region need no barrier, when each element lives in one thread | `isRepeatedLocalStore`, `storesEachAddressFromOneThread` | `@repeated_local_store`, `@repeated_local_store_copies_in_warps`, `@repeated_local_store_copies_in_lanes` |
| Per-thread arrivals from every thread become per-warp arrivals, and a converted arrival is a per-warp publication | `prepareMBarrierArrivals`, `getThreadSyncInfo` | `@tmem_load_then_perthread_arrive` |
| TMEM barrier insertion: a per-thread arrival from every thread, or a named barrier that counts every thread, completes each thread's own TMEM accesses | `getTMemBoundary` | `@tmem_load_then_perthread_arrive_on_argument`, `@tmem_load_then_perthread_arrive` |
| `clc_query_cancel` only reads its response | `TritonNvidiaGPUOps.td` | `@notifications_after_waits` |

Each rule except the last was switched off on its own, and the listed cases
failed every time. TMEM barrier insertion also keeps thread granularity
(`TMemThreadOwnership`, from #5056).

## Resolving a conflict
- Never edit the CHECK lines of these cases to make a pick pass. A failure
  there means a rule was lost: restore the rule.
- If upstream now does the same thing, keep upstream's version, drop ours, and
  re-measure as below.
- Known conflicts after train frontier 2cd5519881:
  - #12661 "Track elected NVIDIA publication issuers", in `syncIfNeeded`: keep
    `canSkip`, `isRepeatedLocalStore` and the barrier-lifetime hazard, and add
    upstream's elected-issuer logic. Upstream may now cover part of
    `waitsForWholeRegion`.
  - #12648 "Infer async_wait completion scope", in `getThreadEffects`: keep the
    `carriesNoPayload` branch and use upstream's `getCompletionSync` in the
    other branch.

## After a pick that touches these files
Re-measure on a B200 with ncu `gpc__cycles_elapsed.max` at locked clocks:
- FA4 forward, B4 H48 S8192 D128 bf16, config BLOCK_M 256, BLOCK_N 128,
  NUM_BUFFERS_KV 3, NUM_CTAS 2: about 6.89M cycles (7.07M without these rules).
- FA backward autotune winner on the same shape: about 17.36M cycles (17.82M
  without these rules).
