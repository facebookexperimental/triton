# NVIDIA Target Profiling With NCU

Use this guide only when `target.json.backend` is CUDA/NVIDIA. It supplements
the generic TLX optimization workflow with NVIDIA NCU requirements. Proton
wrapper/launch attribution and diagnostic instrumentation live in
`references/proton-profiling.md`.

## Metric Discovery

Do not assume one hard-coded metric list works across Hopper, Blackwell, NCU
versions, and kernel launch modes. Before the run, query the active profiler and
GPU for supported metrics using `ncu --query-metrics` or the equivalent command
supported by that installation. Resolve the semantic groups below to available
metric names and persist the resolved mapping in the run artifacts.

If a metric is unsupported, encode the normalized field as JSON `null` and add a
diagnostic explaining that the metric was unavailable on this profiler/GPU. Never
encode a missing metric as zero.

Resolve metrics for these groups when supported:

- barrier and async-wait stalls;
- scoreboard and dependency stalls;
- registers, occupancy, and occupancy limiters;
- spill and local-memory traffic;
- L1/shared-memory behavior;
- L2 bytes, sectors, and hit behavior;
- DRAM bytes, sectors, and throughput;
- tensor-core or MMA activity.

Persist the raw NCU report, CSV exports, metric mapping, and exact commands as
absolute artifact paths.

## Summary Profile

Collect this inexpensive group for the baseline and every correct candidate:

- profiled kernel duration;
- SM throughput percentage;
- DRAM throughput percentage;
- exact profiled kernel name and launch scope.

Use summary data for fast rejection and for deciding whether deep profiling is
needed. It cannot by itself prove that a kernel is compute-bound or memory-bound.
Pair it with Proton attribution so endpoint benchmark time, wrapper time, and
main-kernel time are not conflated.

## Deep Profile

Collect deep profiling for:

- the baseline before candidate generation;
- a candidate within one percentage point of the promotion threshold;
- disagreement between benchmark timing, Proton attribution, and NCU summary;
- repeated rounds with no promoted candidate;
- schedule or warp-specialization hypotheses that depend on stall, occupancy,
  spill, or memory-hierarchy evidence;
- the finalist before promotion or commit.

Do not spend deep-profile time on incorrect or clearly slow candidates.

### Barrier And Async Waits

Resolve counters for:

- barrier stall cycles or warp issue stall due to barriers;
- memory-barrier or async-wait stalls when exposed by the architecture;
- active/eligible warps needed to distinguish waits from insufficient occupancy.

A barrier increase with higher duration directs investigation to wait placement,
phase arithmetic, producer-consumer imbalance, and overly shallow/deep buffering.
Do not change barriers without proving ownership and phase transitions.

### Scoreboards And Dependencies

Resolve counters for:

- long-scoreboard stalls;
- short-scoreboard stalls;
- MIO/throttle or instruction-dependency stalls when available.

Higher long-scoreboard stalls usually justify investigating global/TMA latency,
load-to-use distance, and overlap. Higher short-scoreboard or MIO stalls point
toward shared-memory, special-function, or instruction-pipeline pressure. Use
the architecture documentation before mapping a stall name to a source change.

### Registers, Occupancy, And Spills

Collect or derive:

- registers per thread or task where the tool exposes them;
- achieved occupancy and occupancy limiters;
- theoretical occupancy when available;
- local-memory load/store sectors or bytes;
- compiler-reported spill loads/stores when available.

A register-budget reduction is invalidated when local-memory/spill traffic rises
or duration regresses. Lower register allocation is not itself an optimization;
it must improve occupancy/scheduling without introducing spills.

### L1, Shared Memory, L2, And DRAM

Resolve counters for:

- L1/TEX traffic and hit behavior when relevant;
- shared-memory traffic or bank-conflict signals when available;
- L2 read/write bytes or sectors and hit rate;
- DRAM read/write bytes or sectors, not only throughput percentage.

Lower DRAM throughput with unchanged bytes and worse duration means the pipeline
feeds memory less effectively. Lower L2/DRAM bytes with unchanged useful work can
support a locality or traffic-reduction hypothesis.

### Tensor And Issue Activity

Resolve counters for:

- tensor/MMA activity;
- issue-slot utilization;
- eligible/active warps;
- instruction mix when needed to explain a close candidate.

Use these metrics to distinguish tensor-core starvation from arithmetic or
instruction-issue pressure. Do not optimize a utilization percentage in
isolation; kernel duration and useful work remain primary.

## Normalized Profile Schema

Return target profiling under `profile()["ncu"]`. Keep normalized JSON compact
and spill raw reports to artifacts:

```json
{
  "ncu": {
    "level": "summary or deep",
    "scope": {
      "kernel": "exact kernel name",
      "launches": 1
    },
    "summary": {
      "duration_ns": 0,
      "sm_throughput_pct": 0.0,
      "dram_throughput_pct": 0.0
    },
    "stalls": {
      "barrier_pct": null,
      "async_wait_pct": null,
      "long_scoreboard_pct": null,
      "short_scoreboard_pct": null,
      "mio_throttle_pct": null,
      "dependency_pct": null
    },
    "registers": {
      "registers_per_thread": null,
      "achieved_occupancy_pct": null,
      "theoretical_occupancy_pct": null,
      "occupancy_limiters": null,
      "local_load_bytes": null,
      "local_store_bytes": null,
      "spill_loads": null,
      "spill_stores": null
    },
    "memory": {
      "l1_bytes": null,
      "l1_hit_rate_pct": null,
      "shared_bank_conflicts": null,
      "l2_read_bytes": null,
      "l2_write_bytes": null,
      "l2_hit_rate_pct": null,
      "dram_read_bytes": null,
      "dram_write_bytes": null
    },
    "compute": {
      "tensor_activity_pct": null,
      "issue_active_pct": null,
      "eligible_warps_per_cycle": null,
      "active_warps_per_cycle": null
    },
    "artifacts": {
      "ncu_report": "/absolute/path/to/profile.ncu-rep",
      "csv": "/absolute/path/to/profile.csv",
      "metric_mapping": "/absolute/path/to/ncu_metric_mapping.json",
      "commands": "/absolute/path/to/ncu.commands.txt"
    },
    "raw_metrics": {},
    "diagnostics": []
  }
}
```

Omit unsupported normalized fields only when the consumer accepts sparse output;
otherwise use JSON `null` with a diagnostic. Include units in `raw_metrics` when
values are strings; normalized numeric fields must use the units implied by their
names.

## Candidate Decisions

- Veto material kernel-duration regressions even when endpoint timing improves
  inside the measurement noise floor.
- Veto lower register budgets that increase spill/local-memory traffic.
- Treat higher duration plus lower SM/DRAM utilization as scheduling or
  synchronization loss.
- Require a barrier or scoreboard hypothesis to cite the corresponding deep
  counter and compare it with baseline.
- Require memory-traffic claims to cite bytes and cache behavior, not throughput
  percentage alone.
- Require tensor-core or issue-pressure claims to cite tensor activity,
  eligible/active warps, or instruction mix as appropriate.
- Feed rejected metric signatures into later candidate prompts so the agent does
  not repeat the same configuration under different wording.
