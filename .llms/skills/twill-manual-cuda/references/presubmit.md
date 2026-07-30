# Presubmit and Review Gate

Complete this checklist before requesting review or making correctness,
performance, or provenance claims.

## Artifact integrity

- [ ] The input schema is exactly `twill-pipelined-warp-ir-v1`.
- [ ] The pipelined IR hash matches the handoff manifest and every manual
      manifest.
- [ ] `manual_cuda_authoring.json` contains
      `"authoring_mode": "agent-assisted-manual-cuda"`.
- [ ] Mapping, memory, and synchronization manifests are present, serialized by
      a real serializer, hashed, and referenced from the authoring record.
- [ ] Source and test evidence identify exact revisions and B200 toolchain.
- [ ] Generated or derived evidence has a reproducible command and owner.

## Clean-room review

- [ ] FA4/FA3 use is recorded as `black-box-baselines-only`.
- [ ] No FA4/FA3 source, PTX, SASS, profiler timeline, warp roles, MMA order,
      buffer protocol, or barrier protocol informed the implementation.
- [ ] The implementation was not produced by editing an FA4/FA3 kernel.
- [ ] Every reused TMA/TCGEN05/mbarrier wrapper is a reviewed leaf primitive,
      carries no scheduling policy, and is listed with symbol/revision/reviewer.
- [ ] Any contamination or non-clean-room experiment is disclosed and excluded
      from faithful-lowering claims.

## Manifest reviews

- [ ] Mapping coverage is 100% for IR nodes and dependencies.
- [ ] Physical group widths, lanes, per-group issue order, and all pipeline
      regions match the Twill artifact.
- [ ] Every operand has a memory-plan allocation and reviewed layout.
- [ ] Lifetimes include async completion, ring wrap, alias transitions, and
      epilogue drain.
- [ ] Compiler-reported registers, SMEM, TMEM, and occupancy are reconciled
      with the memory plan.
- [ ] Every cross-warp, async-completion, reuse, proxy-ordering, and store-drain
      obligation appears in the sync manifest.
- [ ] Phase traces cover initialization, two or more slot wraps, predicates,
      zero/partial work, and epilogue.
- [ ] Mapping, memory, and sync reviewers signed the exact manifest hashes.
- [ ] `deviations` is empty for a faithful-lowering claim.

## Validation

- [ ] The exact build command succeeds without relevant warnings.
- [ ] Disassembly contains the reviewed B200 primitive families, shapes, and
      scopes at the mapped anchors.
- [ ] The full correctness matrix passes with recorded tolerances and raw
      evidence.
- [ ] Memory/race/synchronization tooling passes or every limitation is
      documented.
- [ ] Scale and timeout tests exercise repeated barrier phases and large grids.
- [ ] Performance tests were run only if explicitly requested, using the
      repository performance workflow.
- [ ] FA4/FA3 comparisons used identical black-box inputs and methodology.
- [ ] Debug synchronization, fallback paths, and instrumentation are disabled
      in the measured binary.

Do not invent test commands or target names. Record commands verified for the
owning target. C++/CUDA changes require the repository-prescribed rebuild,
format, lint, and target tests; use the applicable project skills and rules.

## Required diff disclosure

Include a plainly visible statement equivalent to:

> `authoring_mode=agent-assisted-manual-cuda`. This is a human-reviewed manual
> B200 CUDA C++ lowering of the hashed `twill-pipelined-warp-ir-v1` artifact,
> authored with agent assistance. FA4/FA3 were used only as black-box
> correctness/performance baselines. No FA4/FA3 warp roles, MMA order, or
> barrier protocol were reused. Reviewed schedule-neutral leaf wrappers are
> listed in `manual_cuda_authoring.json`.

Also state:

- exact input and manifest hashes;
- which humans reviewed mapping, memory, synchronization, and leaf wrappers;
- correctness/sanitizer results and limitations;
- performance methodology and raw evidence, if performance was requested;
- all deviations and whether the result remains a faithful lowering.

Do not call the implementation Twill-generated, automatically lowered, or
paper-generated.

## Change control

After any source change:

1. Update the mapping, memory, or synchronization manifest first if its
   reviewed decision changed.
2. Recompute all affected hashes.
3. Repeat the corresponding review and validation slice.
4. Repeat the entire correctness matrix for instruction, layout, aliasing,
   phase, predicate, or launch-geometry changes.
5. Repeat performance only when requested and when the measured binary changed.

An optimization without updated evidence is not presubmit-ready.

## Post-submit evidence

Archive the exact manifests, raw logs, disassembly, binary/source revision, and
environment record used for claims. Monitor failures and performance drift
against those hashes; do not compare a later binary to an earlier manifest.
