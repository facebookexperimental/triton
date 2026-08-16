# Workflow and Evidence Boundary

## Scope

This workflow starts with a validated `paper-joint-pipelined-ir-v3` JSON
artifact and ends with a human-reviewed manual B200 CUDA C++ implementation.
The agent may draft code, manifests, tests, and audits, but a human must approve
the mapping, memory, and synchronization decisions. Do not describe the result
as automatic code generation, a paper-generated kernel, or proof that the paper
automatically lowers to CUDA.

The handoff artifact records scheduling intent. It deliberately leaves memory
allocation, layouts, synchronization placement, and instruction selection to
manual implementation.

## Input gate

Before authoring anything:

1. Require the handoff manifest and the exact pipelined IR file.
2. Verify `schema_version == "paper-joint-pipelined-ir-v3"`.
3. Recompute the IR hash and compare it with the
   `paper-joint-handoff-v2` manifest.
4. Record the solution hash, source-program hash, machine-model provenance,
   the source kernel name and argument list, `ii`, `length`, `copies`,
   `horizon`, and region bounds.
5. Reject legacy or partial artifacts. Do not infer missing cycles, warps,
   issue traces, dependencies, or region boundaries.
6. Treat the source operation table as semantic input and the instruction
   table plus dependencies as schedule input. Preserve both.

Stop if a hash is stale, a node or edge is missing, a scheduled operation has
no supported B200 lowering, or the requested kernel geometry cannot satisfy a
documented hardware constraint. Ask for a new solver artifact instead of
silently relocating, clamping, dropping, or replacing scheduled work.

## Clean-room gate

FA4 and FA3 may be used only as black-box baselines:

- Allowed: invoke their public benchmark/correctness interface, compare
  outputs, and record aggregate latency or throughput under the same inputs.
- Forbidden: inspect or copy their kernel source, PTX, SASS, profiler timeline,
  warp specialization, MMA issue order, buffer indexing, stage/phase state, or
  barrier protocol as an implementation aid.
- Forbidden: start from an FA4/FA3 kernel and edit it until it matches the
  paper's schedule.
- Forbidden: describe a result that inherited those choices as a clean-room
  manual lowering of the paper.

The narrow exception is a reviewed leaf wrapper for a single TMA, TCGEN05, or
mbarrier operation. A reusable wrapper must expose the primitive's operands and
scope, contain no loop schedule, warp-role policy, buffer rotation, phase
state, or composite synchronization protocol, and have an independent review
record. Record its file, symbol, revision, reviewer, primitive, and reason for
reuse. A helper that chooses when or where to issue a primitive is not a leaf
wrapper.

If an author or agent has already used FA4/FA3 implementation details to make
a mapping, memory, or synchronization decision, mark the affected work
contaminated and do not claim clean-room status. Restart those decisions from
the paper artifact and public B200 specifications with a reviewer who can
verify the boundary.

## Required artifact set

Keep these artifacts beside the implementation or in its designated evidence
directory:

- `manual_cuda_authoring.json`: provenance, clean-room declaration, reviewers,
  input/output hashes, and the exact field
  `"authoring_mode": "agent-assisted-manual-cuda"`.
- `mapping_manifest.json`: complete node-to-source and node-to-instruction
  coverage; see `mapping-contract.md`.
- `memory_plan.json`: layouts, allocations, lifetimes, alias proofs, and
  resource budgets; see `memory-plan.md`.
- `sync_manifest.json`: dependency channels, completion witnesses, phases,
  participants, and drain proofs; see `sync-phase-audit.md`.
- Correctness, sanitizer, build, and requested performance evidence with exact
  commands, revisions, hardware, and raw output locations.

Every manifest must include the pipelined IR SHA-256 and
`authoring_mode=agent-assisted-manual-cuda`. Hash the reviewed manifests and
record those hashes in `manual_cuda_authoring.json`. Regenerate hashes after
every approved change.

Create the initial fail-closed bundle with both validated inputs:

```bash
python -m skc scaffold \
  --ir pipelined_ir.json \
  --handoff handoff.json \
  --out-dir manual_cuda
```

The emitted `kernel.cu` contains `#error` and is deliberately non-executable.
It is not CUDA code generation. After manual completion and human review, run
`audit-mapping`, `audit-memory`, and `audit-sync`, then use `audit-bundle` as
the final provenance, hash, cross-reference, and clean-room gate. An
individual structural audit is not a faithful-lowering claim by itself.

A minimal authoring record contains:

```json
{
  "schema_version": "paper-joint-agent-assisted-manual-cuda-v2",
  "authoring_mode": "agent-assisted-manual-cuda",
  "target": {"gpu": "B200", "cuda_arch": "sm_100a"},
  "status": "manual_completion_required",
  "pipelined_ir": {
    "schema_version": "paper-joint-pipelined-ir-v3",
    "sha256": "<sha256>"
  },
  "handoff": {
    "schema_version": "paper-joint-handoff-v2",
    "sha256": "<sha256>"
  },
  "solution_sha256": "<sha256>",
  "source_program_sha256": "<sha256>",
  "solver_provenance": {},
  "clean_room": {
    "fa4_fa3_use": "black-box-baselines-only",
    "prohibited_reuse": [
      "warp_roles",
      "mma_order",
      "barrier_protocol"
    ],
    "reviewed_leaf_wrappers": [],
    "declaration_reviewed_by": null
  },
  "manifests": {
    "mapping": {"path": "mapping_manifest.json", "sha256": "<sha256>"},
    "memory": {"path": "memory_plan.json", "sha256": "<sha256>"},
    "synchronization": {"path": "sync_manifest.json", "sha256": "<sha256>"}
  },
  "human_reviews": []
}
```

Use a real serializer when creating or updating machine-readable artifacts.
Do not interpolate dynamic values into hand-built JSON or YAML.

## Authoring sequence and review gates

1. **Freeze inputs.** Validate and hash the handoff files. Create the authoring
   record and clean-room declaration.
2. **Map the schedule.** Draft `mapping_manifest.json` for every instruction
   and dependency. A human approves physical warp placement, instruction
   selection, predicates, and prologue/steady-state/epilogue realization.
3. **Plan memory.** Draft `memory_plan.json` from values and lifetimes, not from
   a baseline kernel. A human approves layouts, ring depths, aliasing, and
   aggregate resource budgets.
4. **Design synchronization.** Draft `sync_manifest.json` from dependency and
   resource-reuse obligations. A human approves every completion witness,
   participant count, scope, initial phase, recurrence, and drain path.
5. **Write the CUDA C++ skeleton.** Derive warp dispatch and region structure
   from the approved manifests. Add stable source anchors used by the mapping
   manifest. Do not hide scheduled work inside opaque composite helpers.
6. **Lower primitives.** Select documented B200 operations and approved leaf
   wrappers. Update the mapping before changing instruction selection.
7. **Compile and audit.** Inspect diagnostics, resources, and disassembly for
   the intended operations. Compiler output is evidence about the manual
   implementation, not permission to mutate the paper's schedule silently.
8. **Prove correctness.** Run reference comparisons, edge cases, race/memory
   tools, and scale/hang tests before any performance claim.
9. **Measure performance only when explicitly requested.** Use the repository's
   B200 benchmark workflow, disclose methodology, and keep FA4/FA3 black-box.
10. **Presubmit.** Rehash artifacts, obtain mapping/memory/sync reviews, and use
    the exact provenance language in `presubmit.md`.

If implementation pressure requires a schedule, mapping, memory, or sync
change, update and review the corresponding manifest first. If the change
contradicts a cycle, issue trace, warp assignment, or dependency from the
paper, obtain a new paper artifact or label the result as a non-faithful
experiment.

## Claim vocabulary

Use:

- "agent-assisted manual B200 CUDA C++ lowering from
  `paper-joint-pipelined-ir-v3`"
- "manual implementation preserving the reviewed mapping manifest"
- "FA4/FA3 black-box comparison"

Do not use:

- "paper-generated CUDA"
- "automatic paper codegen"
- "paper-generated kernel"
- "paper performance reproduced" unless the full paper protocol and all
  required evidence have independently been satisfied

Correctness or performance alone does not prove schedule fidelity. Schedule
fidelity comes from the reviewed manifests and their agreement with source and
machine code.
