# gfx950 GEMM Profiling And Artifact Analysis

Use wall-clock measurements for promotion and lower-level evidence to explain
results or reject nonconforming candidates. No single counter, trace statistic,
or static instruction count establishes a performance win.

## Match Evidence To The Question

| Question | Primary evidence | Supporting evidence |
|---|---|---|
| Is the candidate faster? | Same-contract wall-clock distribution | Stable sentinels and independent processes |
| Did dispatch or launch change? | Resolved plan and launch manifest | Runtime logs and grid dimensions |
| Did resource pressure change? | Final VGPR/AGPR/LDS/scratch metadata | Occupancy counters and static code |
| Is memory latency exposed? | Final waits and instruction trace | VMEM counters and cache-hit evidence |
| Is MFMA issue sparse? | Per-wave issue/stall timeline | Normalized MFMA counts and dependency analysis |
| Did a lowering mechanism survive? | Final TTGIR/LLVM/AMDGCN artifact | Compiler diagnostics |

Use counters and traces to rank or falsify mechanisms, then confirm with the
production timing harness.

## Normalize Before Comparing

Different plans can launch different CTA counts, execute different K splits, or
perform different amounts of padded work. Report static and dynamic quantities
per useful output tile, per CTA, per wave, or per useful FLOP as appropriate.
Retain absolute totals as well; normalization can hide extra launch or reduction
work.

Check that compared traces capture the same semantic region and loop behavior.
A shorter kernel may fit entirely in a trace while a longer one is truncated.
One trace may include prologue only, another several steady-state iterations.
Do not compare raw instruction totals until capture completeness is understood.

## Use Hardware Counters Diagnostically

Collect a small hypothesis-driven set rather than every available counter. Map
each counter to a predicted artifact change. Useful classes include:

- wave and workgroup residency;
- VALU/MFMA issue and active cycles;
- VMEM, LDS, and cache transactions;
- wait, barrier, and stall categories; and
- scratch or spill traffic.

Counter names and aggregation semantics vary by profiler version. Record the
tool and runtime revision, event definitions, launch filtering, and whether the
value is summed, averaged, or normalized across compute units. Treat multiplexed
or unstable counters as qualitative evidence.

A higher MFMA issue rate is not automatically a win if the candidate performs
more padded MFMAs. More cache hits are not automatically useful if traffic also
increased. Relate counters to useful work and wall clock.

## Read ATT At Wave And Loop Granularity

Instruction traces can identify exposed gaps that aggregate counters blur.
Inspect multiple representative waves and distinguish prologue, steady state,
tail, and epilogue. For each wave, examine:

```text
VMEM issue groups and the wait that consumes them
LDS publish/read sequences and lgkm waits
MFMA issue gaps and accumulator dependency chains
barriers, nops, and scalar/address setup
stage transitions and final drains
```

Summarize distributions, not only one attractive wave. A workgroup can contain
producer and consumer waves with intentionally different instruction mixes.
Inter-wave pipelines in particular require tracing both sides of the handoff.

Interpret cycles per useful MFMA together with MFMA issue rate. A low issue
rate can result from VMEM latency, LDS dependencies, barriers, accumulator
dependencies, or simply fewer useful MFMAs in that phase. Identify the
instructions immediately preceding the largest gaps before choosing a fix.

ATT captures may truncate, sample an unrepresentative dispatch, or perturb
timing. Use them to form and test scheduling hypotheses, never as the promotion
latency.

## Compare AMDGCN By Instruction Class And Order

Normalize symbols, addresses, and unrelated register numbering before a static
diff. Compare:

- MFMA opcode, count, K width, and accumulator dependency order;
- buffer/global load width, count, and cache modifiers;
- direct-to-LDS versus VGPR payload movement;
- LDS read/write width and count;
- scalar and vector address setup;
- `s_waitcnt` fields, barriers, and nops;
- scratch loads/stores; and
- epilogue conversion and output-store width.

Instruction count is not enough. Preserve the ordered class stream to see
whether memory operations are actually interleaved with independent MFMAs and
whether waits retain useful runway.

As a provisional ranking signal, count adjacent uncovered memory operations or
other repeated memory pairs that have no intervening independent compute. A
high count can identify poor overlap, but a low count can be misleading when
loads are moved immediately before a draining wait or when the compute between
them is dependent. Always verify waits, live ranges, and timing.

## Inspect Waits And Spills Closely

Repeated `vmcnt(0)` or `lgkmcnt(0)` in the steady state may indicate that the
pipeline frequently drains, but some full waits are required at ownership or
reuse boundaries. Classify waits by purpose before moving them.

Scratch traffic is a strong rejection signal for GEMM hot loops. Confirm whether
it comes from explicit local state, register spilling, or a layout conversion.
Register allocation can change after seemingly harmless source scheduling, so
recheck scratch and occupancy for every serious candidate.

Do not delete barriers or weaken waits based only on a trace. Prove the memory
and buffer ownership relation in source and lowering, then retain focused
correctness tests around multiple loop trips and tails.

## Verify The Final Artifact Checklist

For the incumbent and every promoted candidate retain:

```text
resolved public dispatch and complete compile-time plan
grid, workgroup size, waves, auxiliary launches, and workspace
MFMA and load/store instruction forms and normalized counts
LDSDMA/direct-to-LDS presence when intended
wait/barrier/nop placement and steady-state class order
VGPR, AGPR, LDS, scratch, and occupancy
timing samples, regime, sentinels, and correctness coverage
```

When comparing with an external library, use its artifact only to discover
missing mechanisms such as a different work distribution, tile orientation,
register lifetime, or issue pattern. Do not copy opaque instruction sequences
or assume its reported matrix convention matches the public API. Recreate the
mechanism through legal source and compiler abstractions, then validate the
result under the same workload contract.

## Avoid Common Profiling Traps

- A trace of the wrong public dispatch proves nothing about the candidate.
- A compiler dump from a stale cache can hide an option or layout change.
- A raw library benchmark can exclude output allocation or workspace included
  by the public path.
- A median can hide thermal or cache-state modes; retain raw samples.
- Aggregate speedup can hide one severe production regression.
- Static similarity to a fast kernel does not establish equal scheduling after
  register allocation.
- A measured improvement below the noise floor is not a promotion result.
