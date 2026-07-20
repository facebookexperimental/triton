# TLX-to-Wave Async Memory Protocol

Status: implemented compatibility contract

## Compatibility Contract

This protocol exists to make existing TLX programs work unchanged on the Wave
backend. It is not a new source-level synchronization model.

The following requirements are non-negotiable:

1. Existing TLX kernels that are valid on the AMD LLVM backend remain valid on
   the Wave backend without source edits.
2. <code>tl.debug_barrier()</code> is never required to make a TLX async copy,
   wait, local load, local store, or buffer reuse correct. The bridge derives
   the required compiler barriers and token edges.
3. The external TLX API, TTGIR dialect, LLVM lowering, and source kernels are
   unchanged. Generic scheduling markers may bound Wave scheduling regions,
   but correctness never depends on those markers.
4. Async-DMA completion is authorized only by an explicit
   <code>async_load_wait_group</code>. A compiler-generated barrier may publish
   a completed group, but it may never substitute for the wait.
5. A partial wait preserves all requested overlap. Retained groups contribute
   issue order, never completion.
6. The bridge does not infer DMA completion or DMA issue order from a
   destination alias, pending-access table, allocation history, or future
   access.
7. Emission is mechanical. All queue selection, readiness, reuse, barrier mode,
   and control-flow carries are represented in verified target IR first.

If the bridge cannot preserve these requirements, conversion must fail with a
diagnostic. Asking the kernel author to add <code>tl.debug_barrier()</code> is
not an acceptable fallback.

## Existing Semantic Baseline

The protocol must preserve the behavior of the existing TLX-to-AMD compilation
pipeline, not reinterpret one TTGIR operation in isolation.

The generic <code>ttg.async_wait num = K</code> operation selects DMA groups
for completion and does not, by itself, define CTA rendezvous. On AMD,
however, the operation has the memory-wait trait. The existing membar analysis
automatically places a local workgroup barrier after the wait before the next
memory effect when no later synchronization point already provides it. The
wait then lowers to the architecture's async wait instruction, while the
compiler-created local barrier lowers separately.

Therefore the effective readiness point used by existing TLX kernels is:

~~~text
explicit async wait completion
        +
compiler-provided LDS publication/rendezvous
        =
workgroup-ready local-memory epoch
~~~

This is why existing kernels can correctly call
<code>async_load_wait_group</code> and then read LDS without spelling
<code>tl.debug_barrier()</code>. The Wave bridge must build the equivalent
combined event structurally.

The local implementation references for this compatibility baseline are:

- [the generic async-wait contract](../../../include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td);
- [the compiler membar insertion](../../../lib/Analysis/Membar.cpp);
- [the AMD async-wait lowering](../../amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp);
- [the AMD wait-to-local-load annotation](../../amd/lib/TritonAMDGPUToLLVM/AsyncUtility.cpp);
- [the AMD redundant-barrier filter](../../amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp); and
- [the upstream Wave DMA/reuse pipeline](../../wave/python/mlir/dialects/wave_matmul.py).

The distinction remains important:

- the explicit wait decides which DMA groups complete;
- the compiler-generated publication step makes those completed writes safe
  for the required LDS consumers;
- a dependency-free compiler barrier may also align workgroup issue epochs
  without completing any DMA request;
- an ordinary barrier with no explicit wait dependency does not complete DMA;
- an explicit source <code>tl.debug_barrier()</code>, if present, is preserved
  as an ordinary source barrier but is not part of the required async API
  protocol.

## Scope

This document applies to the structural converter in
<code>third_party/tlx_wave/backend/converter</code> and the Wave/WaveAMD IR it
emits.

### Goals

- Preserve the observable semantics of the current AMD LLVM lowering.
- Reproduce the dependency graph of the high-performance upstream Wave
  pipelines.
- Keep f16 GEMM, MXFP GEMM, GLU, and flash-attention sources pristine.
- Preserve <code>wait_group(K)</code> without draining the newest K committed
  groups.
- Order LDS consumption, LDS release, and direct-to-LDS issue without
  destination-based DMA hazards.
- Carry only the live async queue and LDS frontier through structured control
  flow.
- Make every generated workgroup barrier verifier-visible with semantic
  provenance.

### Non-goals

- No LLVM-lowering change.
- No public TLX or TTGIR API change.
- No source-kernel barrier insertion.
- No kernel-specific Wave scheduler change or correctness dependence on
  scheduling metadata.
- No kernel recognizer, shape-specific protocol, or GEMM-only lowering.
- No conversion of retained DMA completion into a conservative full drain.
- No WaveAMD fragment value across an operation, memory, token, or
  control-flow boundary.

## Core Invariants

### 1. Explicit waits are the only DMA-completion authority

Every direct-to-LDS request produces a DMA-completion event. Commit closes a
group of such events. Only an explicit source wait may select committed groups
for completion.

Neither a local load, local store, generated workgroup barrier, explicit source
barrier, alias relation, nor allocation lifetime event may complete a DMA
group.

### 2. Wait readiness includes compiler-owned publication

A wait result is not exposed to a local consumer as a raw per-wave completion
event. The converter constructs a ready point from:

- completion events for exactly the groups selected by the wait;
- issue-only events for retained groups;
- the live LDS-release frontier for the current pipeline epoch; and
- the required compiler-owned rendezvous mode.

For cross-wave LDS consumption the ready point emits a workgroup
<code>wave.barrier</code>. For a proven wave-private access it may emit a
wave-local <code>wave.after</code> or <code>wave.join</code>. Conservatively
choosing a workgroup barrier is correct; requiring a source debug barrier is
not.

The publication barrier is semantically associated with the wait even though
the generic TTGIR wait and the barrier remain distinct events.

The backend runs the existing compiler membar analysis before importing TTGIR.
When that analysis places a local CTA barrier immediately after a wait, the
target wait records the exact source barrier index and coalesces it with the
workgroup-ready <code>wave.barrier</code>. The verifier requires same-region
adjacency and workgroup publication; a wave-local wait cannot erase the source
barrier. This is structural coalescing, not an alias query in the bridge.

Other compiler barriers remain independent target barriers. In particular, a
dependency-free barrier between direct-to-LDS warm-up epochs is an issue
rendezvous across the workgroup. It has no DMA-completion input, emits no
wait-count drain by itself, and cannot replace any explicit wait.

### 3. Retained groups contribute issue order only

For <code>wait_group(K)</code>, the newest K committed groups remain live. The
ready point must know that these groups were issued before the wait, but it
must not wait for their DMA completion.

The retained side is represented through <code>wave.issue_token</code> or an
equivalent typed target event. That projection preserves the prerequisites of
issuing the DMA request while removing the DMA-completion event itself.

### 4. Wait-dominated LDS operations form a release frontier

Local loads and stores associated with a wait-ready point consume its token and
produce LDS-completion tokens. Those tokens form the release/reuse frontier of
that epoch.

A later direct-to-LDS issue that is structurally sequenced after the epoch
consumes the current release frontier as its <code>after</code> dependency. For
cross-wave storage, the per-wave frontier is first published as an LDS-release
event with a compiler-owned workgroup barrier. The DMA consumes that published
event, not the raw per-wave DS tokens.
This is a source-protocol edge:

~~~text
explicit wait -> dominated LDS access -> later DMA issue
~~~

It is not inferred by asking whether the later DMA destination aliases an
earlier access. All independent packets in one issue batch consume the same
frontier; they are not serialized with one another.

This edge orders DMA issue, not DMA completion. Completion still requires the
next explicit wait. A later wait-ready barrier cannot replace release
publication: placing the only barrier after the refill issue permits a fast
wave's DMA response to overwrite LDS while a slower wave is still reading it.

### 5. Source debug barriers are optional, not protocol repairs

An explicit <code>tl.debug_barrier()</code> is lowered with its normal source
semantics when present. It may be coalesced with an equivalent generated
barrier only when equivalence is proven.

The converter must never:

- diagnose a missing debug barrier after a valid wait;
- ask a kernel author to add one for publication or slot reuse;
- use one as the only link between a wait and its LDS consumers; or
- treat one as completing an async group.

### 6. Control-flow state is explicit

Async queue entries, wait-ready tokens, and LDS-release frontiers that cross a
loop or branch are carried through target structured-control-flow operands and
yields.

When only one branch produces an event, the other branch yields a neutral
token. The merge result, rather than hidden emitter state, is used by later
operations.

## Reference Pipeline

The debug-barrier-free double-buffered pipeline is the canonical compatibility
case:

~~~text
G0 = commit(DMA buffer 0)
G1 = commit(DMA buffer 1)

R0 = wait_ready(
       complete = G0,
       retained_issue = issue(G1))
L0, U0 = LDS reads of buffer 0 after R0

loop:
P0 = publish_release(U0)
G2 = commit(DMA refill after P0)
  compute with register-resident L0

  R1 = wait_ready(
         complete = G1,
         retained_issue = issue(G2),
         lds_release = P0)
  L1, U1 = LDS reads of buffer 1 after R1

  carry G2 and U1
~~~

The corresponding Wave shape is:

~~~text
%reuse0 = join(%lds_read_tokens0)
%reuse0_issued = issue_token(%reuse0)
%released0 = barrier(%reuse0_issued)
%g2 = dma_issue(after = %released0)
%g2_issued = issue_token(%g2)
%ready1 = barrier(%g1_completed, %g2_issued, %released0)
%values1, %reuse1 = lds_reads(after = %ready1)
~~~

There is no source <code>tl.debug_barrier()</code>.

The important transitive relationships are:

- the release barrier publishes completion of the previous LDS reads;
- the refill cannot issue before that collective release;
- the ready barrier waits for the older completed group;
- the ready barrier sees that the newer group was issued;
- the ready barrier does not wait for the newer group's completion; and
- the next LDS reads cannot move before the ready point.

This is the protocol shape that must be compared with the upstream Wave kernel.

## Event Domains

All protocol values may share the same physical Wave token type, but target IR
and verification distinguish their semantic domains.

### DMA completion

Produced by a direct-to-LDS request and grouped by commit. It is consumed for
completion only by an explicit wait.

Provenance: source DMA operation, packet, and commit-group ID.

### DMA issue

An issue-only projection of a DMA or committed group. It contains the request's
issue event and issue prerequisites, but no DMA-completion event.

Wave mapping: <code>wave.issue_token</code>.

Provenance: retained side of a specific explicit wait.

### Wait completion

The completion of exactly the groups selected by an explicit wait, before
publication is applied.

Provenance: source wait, source <code>num</code>, completed group IDs, and
retained group IDs.

### LDS access completion

Produced by local loads, stores, gather/scatter, transpose accesses, and other
structural LDS operations when a later protocol operation needs it. The data
result remains an ordinary SIMD/tensor value.

Provenance: source local-memory operation and readiness epoch.

### LDS issue

An issue-only projection of an LDS-access completion frontier. It proves that
the DS operations were issued before a following workgroup collective without
asking Wave to insert a per-wave <code>lgkmcnt(0)</code> drain first. The
collective itself supplies the required workgroup completion and visibility.

This projection is legal only as the input to compiler-owned cross-wave LDS
release publication. It cannot authorize local consumption, replace a wait,
or carry DMA-completion authority.

Wave mapping: <code>wave.issue_token</code>.

Provenance: the exact LDS frontier being published for async-DMA reuse.

### LDS released

The workgroup-published form of an LDS-access frontier. A cross-wave
direct-to-LDS overwrite consumes this event as an issue prerequisite. It
contains no DMA-completion authority.

Wave mapping: <code>wave.barrier</code> for workgroup-owned storage or
<code>wave.after</code> for proven wave-private storage.

Provenance: the tracked LDS frontier and the later source-sequenced DMA issue.

### Workgroup readiness

The published ready point used by cross-wave LDS consumers. It consumes an
explicit wait-completion event and may also consume issue-only and LDS-release
events.

Wave mapping: <code>wave.barrier</code>.

Provenance: compiler membar after a specific source wait, plus its ownership
proof or conservative reason.

### Wave-local readiness

The relaxed form of workgroup readiness, legal only when every relevant byte
is produced and consumed by the same wave and convergence is proven.

Wave mapping: <code>wave.after</code> or <code>wave.join</code>.

### Empty event

The identity for a missing branch event or an empty frontier.

Wave mapping: <code>wave.token</code>.

## Protocol Construction

### Async-group queue

Token analysis maintains an ordered committed-group queue on every structured
control-flow path.

For a wait with <code>num = K</code>:

- completed groups are all groups older than the newest K;
- retained groups are the newest K;
- explicit token operands restrict the source groups according to existing
  TTGIR semantics;
- an implicit wait uses the path-sensitive queue; and
- a queue that cannot be represented exactly is diagnosed rather than drained.

Commit-group members share their input dependency and are joined as members of
one group. Packet or scalar fallback loops must not create a serial token chain.

### Associating a wait with LDS consumers

The converter recognizes an async-ready LDS consumer through existing source
information:

- an explicit wait token operand on the local operation;
- the imported <code>syncedViaAsyncWait</code> annotation and the dominating
  wait; or
- an equivalent structured token carry already present in TTGIR.

No new source API is required. The converter threads the resolved ready token
as an explicit target operand. Textual order in emitted Wave IR is not a
readiness proof.

If an LDS read of DMA-written storage has no dominating explicit wait, the
program is missing DMA completion and conversion fails. A debug barrier cannot
repair that error.

### Building the ready point

The target wait has three independent operand segments:

1. completed-group tokens;
2. retained-group issue-only tokens; and
3. the LDS-release frontier carried into this synchronization point.

It records a publication mode:

- <code>workgroup</code> for a compiler-created Wave barrier; or
- <code>wave_local</code> when structural ownership proves rendezvous is
  unnecessary.

The result is the sole readiness token consumed by associated LDS operations.
Canonicalization may move independent arithmetic around it but may not merge
the operand domains or move a local consumer before it.

### Building the LDS-release frontier

Every tracked local access produces a completion token only when it has a later
protocol user. The frontier is the minimal join of live local-access tokens
since the previous ready point.

When a direct-to-LDS issue follows that frontier:

- cross-wave storage first produces an explicit published-release target
  event; its raw completion frontier is projected to LDS issue order before
  the workgroup barrier, while wave-private storage may use a wave-local
  release carrying completion directly;
- the issue receives the published release as an explicit target operand;
- every packet in the batch receives the same dependency;
- the frontier is not derived from the DMA destination;
- the DMA result remains a completion event in its new commit group; and
- the frontier remains available to the next ready point when a workgroup
  rendezvous still needs it.

An optional optimization may discard an access from the frontier only from an
already-proven semantic independence relation. It may not query mutable
emitter alias state or infer safety from a kernel name, shape, allocation
offset coincidence, or scheduler behavior.

Release publication and wait readiness are distinct rendezvous points in a
buffer-reuse pipeline. The release must precede an overwrite; the ready point
must follow completion of the different buffer selected by the explicit wait.
They may be coalesced only when the target graph proves those two orderings are
the same event. Emission must not coalesce them by inspecting textual order or
DMA destinations.

### Compiler-generated barrier selection

The compatibility default after a wait that feeds local memory is
<code>workgroup</code>, matching the effective AMD membar behavior.

The converter may select <code>wave_local</code> only if:

1. the wait and all associated consumers are structurally convergent;
2. every relevant producer/consumer byte has the same wave owner;
3. the proof composes the distributed and shared-memory layout maps;
4. there are no relevant atomics, volatile accesses, fences, unknown memory
   effects, or allocation-lifetime changes; and
5. the proof is serialized into target IR and checked by the verifier.

Failure to prove relaxation keeps the generated workgroup barrier. It does not
request a source debug barrier.

### Structured loops

Pre-conversion liveness determines the hidden loop arguments and yields. The
steady-state double-buffered pipeline normally carries:

- the retained committed-group token;
- the current wait-ready token only when a source/target consumer crosses the
  backedge; and
- the LDS-release frontier needed by the next DMA issue or ready point.

Consumed frontiers are killed. The converter must not carry an
allocation-wide history of all local operations.

### Structured branches

Each branch yields the same protocol slots. A branch with no real event yields
an empty token. The containing <code>scf.if</code> merges those values, and all
later uses consume the merge result.

Async queue occupancy remains path-sensitive: an empty event does not count as
a committed group. If a partial wait after a branch cannot identify completed
and retained groups exactly, conversion fails instead of becoming a full wait.

## Target IR Contract

The target schema must make the following information explicit and
verifier-visible.

### Direct-to-LDS DMA

- ordinary address, offset, mask, and destination operands;
- a distinct source-protocol issue-dependency segment;
- a distinct published LDS-release dependency segment;
- packet and commit-group provenance; and
- one DMA-completion result.

There is no implicit destination-access operand and no emission-time hazard
lookup.

### LDS-release publication

- one or more LDS-issue operands for workgroup publication, or
  LDS-completion/frontier operands for wave-local publication;
- workgroup or wave-local publication mode;
- compiler provenance identifying DMA reuse; and
- one LDS-released result consumed by every packet in the issue batch.

### Async wait-ready operation

- completed-group operand segment;
- retained issue-only operand segment;
- LDS-release operand segment;
- source <code>num</code>;
- exact completed and retained group IDs;
- publication mode and proof provenance; and
- one ready-token result.

### Tracked local access

- existing address/memdesc/data operands;
- explicit ready or ordinary-memory dependency operands;
- ordinary data result, where applicable; and
- a synthetic LDS-completion result only when live.

MMA fragments are not target representations. Packing occurs immediately
before MMA emission and unpacking immediately after it.

### Explicit source barrier

An explicit source barrier, when present, remains a separate target operation
with source provenance. It is not required to create the wait-ready operation.

### Joins and projections

Completion joins, LDS-frontier joins, issue-only projections, and empty tokens
are explicit operations or equally explicit typed target records. Emission
must not rediscover event meaning by walking producers.

## Wave Emission

After verification, emission is a direct mapping:

| Target event | Wave operation |
| --- | --- |
| Empty event | <code>wave.token</code> |
| Completion or LDS join | <code>wave.join</code> |
| Retained DMA or LDS-release issue projection | <code>wave.issue_token</code> |
| Compiler workgroup issue rendezvous | dependency-free <code>wave.barrier</code> |
| Workgroup LDS release | <code>wave.barrier</code> |
| Wave-local LDS release | <code>wave.after</code> |
| Workgroup wait-ready point | <code>wave.barrier</code> |
| Wave-local wait-ready point | <code>wave.after</code> or <code>wave.join</code> |
| Tracked LDS access | structural Wave load/store/gather/scatter and token result |
| Direct-to-LDS issue | <code>waveamd.dma_load_lds(..., after=frontier)</code> |

Emission must not:

- keep a mutable pending-LDS-access protocol;
- inspect DMA destinations for hazards;
- infer dependencies at MMA boundaries;
- turn all wait operands into completion events;
- choose a barrier from the number of waves; or
- compensate with scheduler changes.

Scheduling markers remain scheduling markers and never prove readiness,
release, convergence, or DMA completion.

## Partial-Wait Example

For committed groups G0 and G1 followed by
<code>wait_group(1)</code>:

~~~text
completed = completion(G0)
retained = issue_token(G1)
ready = barrier(completed, retained, lds_release)
~~~

The retained token preserves G1's issue prerequisites but contains no G1
completion event. Therefore consuming <code>ready</code> may produce a
workgroup barrier and still leave G1 in flight.

The following are forbidden:

~~~text
barrier(completion(G0), completion(G1))  # drains the retained group
barrier(join(all pending DMA))           # destination/history inference
after(completion(G1), local_load)        # implicit completion without wait
~~~

Only a source <code>wait_group(0)</code> intentionally drains the whole queue.

## Verification

The verifier rejects target IR unless all of these hold:

1. Every consumption of DMA completion as a ready event originates at an
   explicit wait.
2. Every DMA-backed LDS consumer has a dominating ready token for the correct
   completed group.
3. A wait-ready token includes compiler publication when required; no source
   debug barrier is required.
4. Retained groups contribute issue order and never completion.
5. Every direct-to-LDS issue dependency originates at the explicit
   wait-to-LDS release protocol or another explicit source dependency, never a
   destination query.
6. Independent packets share an issue frontier and are not serialized.
7. Every tracked LDS completion token is consumed, carried, or proven dead.
8. Every generated barrier records wait-ready or LDS-release provenance and a
   publication mode.
9. Every loop/branch event dominates its use or is carried through structured
   arguments and yields.
10. Empty branch events do not change queue length.
11. No fragment type appears in target IR or across a control-flow/memory
    boundary.
12. No unsupported ambiguity is repaired with a full wait, extra debug
    barrier, hidden emitter dependency, or scheduler constraint.

Diagnostics identify the source operation, event domain, commit group, region,
and failed proof.

## Validation Plan

### Semantic oracle checks

Document and test the effective AMD path for representative kernels:

~~~text
ttg.async_wait
  -> architecture async wait
compiler local membar
  -> workgroup barrier before LDS consumption
~~~

The Wave target graph must preserve the same completion and publication
semantics without requiring a source debug barrier.

### Unit tests

Add structural tests for:

- debug-barrier-free wait followed by cross-wave LDS reads;
- debug-barrier-free wait followed by wave-private LDS reads;
- one completed and one retained group under <code>wait_group(1)</code>;
- retained issue projection preserving issue prerequisites;
- LDS-read frontier ordering a later DMA issue;
- cross-wave LDS release being published before that issue;
- workgroup release using an LDS-issue projection without a redundant
  pre-barrier LDS completion drain;
- multiple DMA packets sharing one reuse frontier;
- no destination-derived DMA dependency;
- an explicit source debug barrier remaining optional and separately
  represented;
- loop-carried group and LDS-release frontiers;
- a branch merging a real event with an empty event; and
- rejection of an LDS consumer with no explicit wait completion.

Each structural rule with natural layout or stage-count variation should have
at least two variants.

### Wave IR checks

The f16 8-wave kernel must show:

- source-pristine TTGIR with no added debug barriers;
- one retained group and one LDS-release frontier in the steady state;
- prior LDS-read tokens reaching a release barrier that orders the refill DMA
  issue;
- <code>wave.issue_token</code> for the retained group;
- one workgroup-ready point for the older completed group;
- next LDS reads consuming that ready point;
- no bridge-created full drain;
- no fragment across the loop boundary; and
- no emitter-inferred DMA hazard.

### Assembly checks

Use invariant checks rather than a complete assembly golden:

- the hot-loop wait retains the requested newer group;
- no hot-loop <code>vmcnt(0)</code> or equivalent full drain unless requested;
- the refill DMA remains ordered after the required LDS reads;
- each cross-wave buffer reuse has a release barrier before the refill and a
  wait-ready barrier before the next LDS consumers;
- no extra barrier appears at MFMA boundaries; and
- the tail drain matches source <code>wait_group(0)</code>.

### Runtime and performance

Correctness is required first. Then run the existing Wave backend tests and the
complete TLX Wave performance sweep against the recorded LLVM and Wave
baselines for:

- f16 GEMM, including the 8-wave kernel;
- MXFP GEMM;
- GLU; and
- flash attention, including the 8K case.

The protocol is not complete if an existing kernel needs a source edit, gains a
full wait or unexplained steady-state barrier, loses DMA overlap, or materially
regresses performance. The 8-wave f16 target remains correctness plus the
previously demonstrated 1.3+ PFLOP/s class of performance, not merely successful
compilation.

## Rollout

Implementation was gated on acceptance of this compatibility contract and
target event model. The accepted rollout is:

1. Add protocol dumps and verifier records without changing emission.
2. Compare existing LLVM behavior, upstream Wave IR, and the proposed target
   graph for the same debug-barrier-free kernels.
3. Add explicit group, issue, ready, LDS-completion, LDS-released, and neutral
   event domains.
4. Add loop/branch carries and verifier checks.
5. Switch one operation family at a time to mechanical emission.
6. Remove equivalent mutable emitter state only after structural and runtime
   parity.
7. Run correctness, assembly invariants, and the full performance sweep at
   every step.

At no stage may rollout require a source <code>tl.debug_barrier()</code>, an
external TLX API change, an LLVM-lowering change, or a Wave scheduler
workaround.
