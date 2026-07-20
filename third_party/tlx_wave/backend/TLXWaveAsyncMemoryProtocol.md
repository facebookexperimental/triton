# TLX-to-Wave Async Memory Protocol

Status: implemented compatibility contract

## Compatibility Contract

This protocol makes existing TLX programs work unchanged on the Wave backend.
It is not a new source synchronization model.

The following requirements are non-negotiable:

1. Existing TLX kernels that are valid on the AMD LLVM backend remain valid on
   the Wave backend without source edits.
2. `tl.debug_barrier()` is never required for correctness. Compiler-generated
   barriers provide the same LDS publication and reuse ordering used by LLVM.
3. The external TLX API, TTGIR dialect, LLVM lowering, and source kernels remain
   unchanged.
4. Async-DMA completion is authorized only by an explicit
   `async_load_wait_group`. A barrier never completes a DMA group.
5. `wait_group(K)` leaves the newest K groups live. Retained groups contribute
   issue order, never completion.
6. Direct-to-LDS DMA issue never acquires a dependency from an LDS alias,
   destination, pending access, allocation history, or bridge-created release
   frontier.
7. The bridge preserves explicit compiler barriers as synchronization
   operations. It does not reconstruct membar analysis or insert another
   workgroup barrier before DMA.
8. Emission is mechanical. Queue selection, readiness, barrier coalescing, and
   structured-control-flow carries are represented in verified target IR.

If the bridge cannot preserve these requirements, conversion must fail with a
diagnostic. Asking the kernel author to add a debug barrier or conservatively
draining all DMA groups is not an acceptable fallback.

## Existing Semantic Baseline

The Wave compiler runs the same warp-pipeline conversion and AMD membar
analysis used by the LLVM path before importing TTGIR. The imported program
therefore already contains the compiler-owned barriers needed for local-memory
publication and storage reuse.

The responsibilities are distinct:

- `ttg.async_wait num = K` selects exactly which committed DMA groups complete;
- the compiler barrier immediately after a cross-wave wait publishes those
  completed writes before LDS consumption;
- other compiler barriers separate local reads/writes from later storage reuse;
- an ordinary barrier does not complete an in-flight DMA; and
- direct DMA issue consumes only source async-protocol issue dependencies and
  completion-free ordering from an explicit full-memory source barrier.

The bridge may coalesce an adjacent compiler barrier with a workgroup
wait-ready operation because they represent the same physical rendezvous. It
must preserve every other compiler barrier independently.

The implementation references for this compatibility baseline are:

- [the generic async-wait contract](../../../include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td);
- [the compiler membar insertion](../../../lib/Analysis/Membar.cpp);
- [the AMD async-wait lowering](../../amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp);
- [the AMD wait-to-local-load annotation](../../amd/lib/TritonAMDGPUToLLVM/AsyncUtility.cpp); and
- [the AMD redundant-barrier filter](../../amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp).

## Scope

This document applies to the structural converter in
`third_party/tlx_wave/backend/converter` and the Wave/WaveAMD IR it emits.

### Goals

- Preserve the observable semantics of the AMD LLVM lowering.
- Keep f16 GEMM, MXFP GEMM, GLU, and flash-attention sources pristine.
- Preserve partial-wait overlap without converting a hot-loop wait into a full
  `vmcnt(0)`/`lgkmcnt(0)` drain.
- Represent wait-to-DS readiness and structured-control-flow token carries in
  target SSA.
- Preserve compiler-created synchronization without duplicating it.
- Keep direct DMA issue independent of inferred LDS state.

### Non-goals

- No LLVM-lowering change.
- No public TLX or TTGIR API change.
- No source-kernel barrier insertion.
- No kernel recognizer, shape-specific exception, or GEMM-only rule.
- No Wave scheduler workaround for missing bridge dependencies.
- No WaveAMD fragment value across an operation, memory, token, or
  control-flow boundary.

## Core Invariants

### 1. Explicit waits are the only DMA-completion authority

Every direct-to-LDS request produces a DMA-completion event. Commit closes a
group of such events. Only an explicit source wait may select committed groups
for completion.

Neither a local operation, compiler barrier, explicit debug barrier, alias
relation, nor allocation-lifetime event may complete a DMA group.

### 2. Wait readiness is an explicit dependency of LDS consumers

A cross-wave wait-ready point combines:

- completion events for exactly the groups selected by the wait;
- issue-only events for retained groups; and
- the adjacent compiler-owned publication barrier, when AMD membar emitted
  one.

The result is a workgroup-ready token consumed by every associated LDS
operation. Wave-local ownership may use a wave-local ready token when the
structural proof is available.

The bridge threads this dependency through loop arguments, branch results, and
yields. Textual proximity is not a readiness proof.

### 3. Compiler barriers own LDS reuse epochs

AMD membar runs before conversion and places a barrier wherever cross-wave LDS
accesses must complete before later storage reuse. The bridge translates that
barrier at the same structural point.

A barrier may consume the preceding LDS-completion frontier so local accesses
cannot move across it. A following direct DMA does **not** consume the raw
barrier result or any LDS/DMA-completion token. It may consume the
completion-free issue-order projection of an explicit full-memory barrier.

`wave.barrier` lowers to a hardware barrier, but it is not a WaveAMDMachine
scheduling-region delimiter. A full-memory CTA barrier (`addrSpace = 31`)
forbids memory issue from crossing the barrier even when no value edge exists.
The bridge therefore builds a sparse, region-local issue frontier:

```text
%pre_issue = wave.issue_token %pre_memory_tokens
%barrier = wave.barrier %pre_issue
%post_issue = wave.issue_token %barrier

wave.load  ... after %post_issue
wave.store ... after %post_issue
```

The first projection orders the barrier after preceding memory issue without
carrying load/store/DMA completion. The second strips any completion events
carried by the raw barrier result. Every following memory issuer consumes the
same `%post_issue` epoch until the next full barrier, so sibling operations are
not serialized with each other. Pure arithmetic, MFMA, reductions, and
`wave.redistribute` receive no epoch operand and remain movable across the
barrier when ordinary SSA dependencies allow it.

Local-memory barriers and adjacent publication barriers coalesced into a
workgroup wait-ready barrier do not create a full-memory issue epoch. Their
affected operations are ordered by the verified readiness/local-memory token
graph, while independent DMA issue retains the overlap permitted by the async
protocol.

This rule is deliberately allocation- and kernel-independent:

```text
wait_ready -> LDS access -> compiler barrier -> direct DMA issue
```

It replaces the invalid global rule:

```text
wait_ready -> every dominated LDS access -> synthesized barrier -> next DMA
```

The latter serializes unrelated buffers, duplicates membar barriers, and
shortens DMA prefetch distance. If the imported program lacks a barrier needed
for safe storage reuse, the defect belongs to the source protocol or the
pre-bridge compiler pipeline; the bridge must not repair it by inspecting DMA
destinations.

### 4. Retained groups contribute issue order only

For `wait_group(K)`, the newest K committed groups remain live. The ready point
may depend on an issue-only projection of those groups so it cannot overtake
their issue, but it must not consume their completion.

The projection is represented by `wave.issue_token` with DMA-issue provenance.
There is no LDS-issue projection.

### 5. Independent DMA packets remain independent

Packets or scalar chunks in one async group are group members, not a serial
chain. They may share the same explicit source issue prerequisites, but no
packet receives the previous packet or an alias-derived LDS access as a new
`after` dependency. Packets after an explicit full-memory barrier may all share
the same completion-free barrier-issue epoch.

### 6. Control-flow state is explicit

Async queue entries and wait-ready/local-order tokens crossing a loop or branch
are carried through target operands, block arguments, results, and yields.

When only one branch produces an event, the other branch yields a neutral
token. Hidden emitter state must not substitute for an SSA join.

## Reference Pipeline

A debug-barrier-free double-buffered pipeline lowers as follows:

```text
G0 = commit(DMA buffer 0)
G1 = commit(DMA buffer 1)

R0 = wait_ready(
       complete = G0,
       retained_issue = issue(G1))
L0 = LDS reads of buffer 0 after R0
compute(L0)

B0 = compiler_barrier(L0)
G2 = commit(DMA refill; source_issue_after + barrier_issue_after)

R1 = wait_ready(
       complete = G1,
       retained_issue = issue(G2))
L1 = LDS reads of buffer 1 after R1
```

The corresponding Wave shape is:

```text
%g1_issued = wave.issue_token %g1
%ready0 = wave.barrier %g0, %g1_issued
%values0, %reads0 = wave.gather ... after %ready0

%reads0_issued = wave.issue_token %reads0
%reuse0 = wave.barrier %reads0, %reads0_issued
%reuse0_issued = wave.issue_token %reuse0
%g2 = waveamd.dma_load_lds ... after %source_issue_dependency, %reuse0_issued
```

The DMA does not use raw `%reuse0` as its `after` operand. It uses only
`%reuse0_issued`, which preserves issue order without carrying completion. This
keeps refill issue free from an implicit DMA wait while preserving the same
workgroup epochs as LLVM.

## Event Domains

All protocol values lower to `!wave.mem.token`, but target IR distinguishes
their semantic domains.

### DMA completion and group

Direct DMA produces DMA completion; commit joins group members into a DMA-group
event. Only an explicit wait consumes these as completion.

### DMA issue

An issue-only projection of a retained DMA group. It preserves issue
prerequisites without carrying DMA completion.

Wave mapping: `wave.issue_token`.

### Memory completion and issue

A real global, buffer, local, or direct-DMA memory issuer exposes its raw token
only when a later full barrier needs an ordering proof. The bridge joins those
tokens with one `wave.issue_token` projection immediately before that barrier.
The resulting memory-issue event carries issue prerequisites but no completion.

### Full barrier and barrier issue

A full-memory barrier exposes its raw result only when a later memory issuer or
full barrier needs it. A second `wave.issue_token` projection produces the
barrier-issue epoch consumed by following memory issuers. The raw full-barrier
result is never a direct-DMA `after` operand.

### LDS access completion and frontier

Tracked local operations may produce completion tokens. Their minimal live join
is carried only to an explicit compiler barrier, a later source synchronization
point, or through structured control flow. It is not a DMA issue dependency.

### LDS released

The result of an explicit compiler/source barrier that consumes a local-access
frontier. It may maintain local ordering through later synchronization, but it
is never accepted directly as a DMA issue operand. Only the full-barrier
completion-free projection is accepted in the distinct barrier-order segment.

### Workgroup and wave-local readiness

The published result of an explicit async wait. Workgroup readiness maps to
`wave.barrier`; wave-local readiness maps to `wave.after` or `wave.join`.

### Empty event

The identity used for missing branch events or empty control-flow carries.

Wave mapping: `wave.token`.

## Protocol Construction

### Async-group queue

Token analysis maintains the ordered committed-group queue on every structured
control-flow path. For `wait_group(K)`:

- completed groups are all groups older than the newest K;
- retained groups are the newest K;
- explicit operands restrict the source groups according to TTGIR semantics;
- an implicit wait uses the path-sensitive queue; and
- an ambiguous queue is diagnosed rather than fully drained.

### Associating waits with LDS consumers

The converter recognizes a ready LDS consumer from existing source information:

- an explicit wait-token operand;
- the imported `syncedViaAsyncWait` annotation and dominating wait; or
- an equivalent structured token carry already present in TTGIR.

The resolved ready token becomes an explicit target operand. If a DMA-backed
LDS read has no explicit wait proof, conversion must reject the protocol rather
than infer completion from the destination.

### Preserving compiler barriers

The backend runs AMD membar before source import. An adjacent local CTA barrier
after a workgroup wait may be coalesced into the wait-ready `wave.barrier` when
the verifier confirms same-region adjacency and provenance.

Every other compiler/source barrier remains a target `barrier` at its original
structural point. Preceding tracked LDS tokens may be operands of that barrier;
the bridge does not propagate completion from its result into a later direct
DMA. A target-IR pass gives a full-memory barrier (`address_space = 31`) the
completion-free predecessor/successor issue projections described above. This
decision depends only on source barrier scope and target-region order, not on a
kernel, allocation, destination, or alias query.

### Structured loops and branches

Pre-conversion analysis determines live async group and readiness carries.
Local-order frontiers are carried only when a later barrier or synchronization
uses them. Consumed frontiers are killed; allocation-wide history is not a
source of DMA dependencies.

## Target IR Contract

### Direct-to-LDS DMA

- ordinary address, offset, mask, and destination operands;
- a source-protocol issue-dependency segment;
- an optional, distinct full-barrier issue-order segment;
- packet and commit-group provenance; and
- one DMA-completion result.

The verifier requires `source_issue_dependency_count == issue_dependency_count`
for the source segment and accepts only DMA-issue, empty, or explicit source
token domains there. The optional barrier-order segment accepts only a
completion-free barrier-issue token with structurally verified full-barrier
provenance. There is no LDS-release segment.

### Async wait-ready operation

- completed-group operands;
- retained DMA-issue operands;
- any local-order frontier needed by the synchronization point;
- source `num` and exact completed/retained group IDs;
- publication mode and adjacent-barrier provenance; and
- one ready-token result.

### Tracked local access

- ordinary address/memdesc/data operands;
- explicit ready or ordinary-memory dependencies;
- ordinary data results; and
- a synthetic LDS-completion result only when a later explicit synchronization
  operation or structured carry needs it.

### Compiler/source barrier

The target barrier remains a side-effecting operation with source provenance.
It may consume a local completion frontier and publish a local-order result.
It never completes DMA. The full-memory form may consume a completion-free
predecessor issue token and publish a raw result that is immediately projected
to a completion-free successor epoch. Neither projection is a scheduler-region
cut.

## Wave Emission

After verification, emission maps target events mechanically:

| Target event | Wave operation |
| --- | --- |
| Empty event | `wave.token` |
| Token/frontier join | `wave.join` |
| Retained DMA issue projection | `wave.issue_token` |
| Pre-barrier memory issue projection | `wave.issue_token` |
| Post-barrier issue epoch | `wave.issue_token` |
| Local compiler/source barrier | `wave.barrier` |
| Full-memory compiler/source barrier (`address_space = 31`) | `wave.barrier` with explicit issue-token edges |
| Workgroup wait-ready point | `wave.barrier` |
| Wave-local wait-ready point | `wave.after` or `wave.join` |
| Tracked LDS access | structural load/store/gather/scatter plus token result |
| Direct-to-LDS issue | `waveamd.dma_load_lds(..., after=source_issue/barrier_issue)` |

Emission must not:

- query DMA destinations for hazards;
- append pending LDS access state to a direct DMA dependency;
- insert a CTA barrier before DMA;
- infer dependencies at MMA boundaries;
- turn retained groups into completion dependencies; or
- manufacture barrier ordering from an alias, allocation, or destination
  relation instead of an explicit full-memory source/compiler barrier.

## Partial-Wait Example

For committed groups G0 and G1 followed by `wait_group(1)`:

```text
completed = completion(G0)
retained = issue_token(G1)
ready = barrier(completed, retained)
```

The following are forbidden:

```text
barrier(completion(G0), completion(G1))  # drains the retained group
barrier(join(all pending DMA))           # destination/history inference
dma_issue(after = lds_read_or_barrier)   # inferred LDS reuse dependency
after(completion(G1), local_load)         # completion without explicit wait
```

Only a source `wait_group(0)` intentionally drains the whole queue.

## Verification

The verifier rejects target IR unless all of these hold:

1. DMA completion reaches readiness only through an explicit source wait.
2. Every DMA-backed LDS consumer has a dominating ready token.
3. Retained groups contribute issue order and never completion.
4. Every direct-DMA issue operand comes from the explicit source async
   protocol; LDS completion/released domains are rejected.
5. Compiler barriers remain explicit except for verified adjacent wait
   publication coalescing.
6. Independent DMA packets are not serialized.
7. Loop and branch tokens dominate their uses or are carried through structured
   arguments and yields.
8. No unsupported ambiguity is repaired with a full wait, extra barrier,
   destination query, or inferred scheduling constraint.

## Validation

### Unit and structural tests

Coverage must include:

- cross-wave and wave-local wait-to-LDS readiness;
- partial waits with retained issue-only groups;
- compiler reuse barriers before direct DMA;
- absence of bridge-created `lds_release` operations and DMA barrier operands;
- multiple circular-buffer depths and independent dynamic indices;
- independent packets sharing only explicit source prerequisites;
- loop/branch readiness carries; and
- rejection of malformed DMA, memory, and full-barrier issue projections.

### Wave IR and assembly checks

For representative pipelines:

- every required source/compiler barrier remains visible;
- every full-memory source/compiler barrier has verified completion-free issue
  edges to surrounding memory operations, while pure/layout operations have no
  such edge;
- no bridge-created `wave.sched_barrier` accompanies a full-memory barrier;
- no additional release barrier appears immediately before DMA;
- no raw barrier result, LDS token, or DMA-completion token is a DMA `after`
  operand;
- hot-loop waits retain the requested newer groups;
- no full drain appears unless requested; and
- DMA retains useful issue lead before the following wait.

### Runtime and performance

Correctness is required first. Then run the existing Wave backend tests and the
complete LLVM-versus-Wave performance sweep for:

- f16 GEMM, including v9, v10, and the 8-wave inter-wave kernel;
- MXFP GEMM;
- GLU; and
- flash attention, including the 8K case.

The protocol is not complete if an existing kernel needs a source edit, gains
an unexplained hot-loop barrier or full wait, loses DMA overlap, or materially
regresses the established sweep.
