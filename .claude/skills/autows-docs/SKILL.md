---
name: autows-docs
description: >
  Consult and maintain AutoWS documentation. Use BEFORE exploring AutoWS source
  code — when investigating, planning, or modifying files under
  WarpSpecialization/, partition scheduling, warp_specialize ops, WSCodePartition,
  WSDataPartition, WSTaskPartition, WSMemoryPlanner, or related passes. Also use
  AFTER making non-trivial changes to AutoWS code to keep docs in sync.
---

# AutoWS Documentation

AutoWS has comprehensive design docs that live alongside the source code at:

```
third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/
```

## CRITICAL: Read docs BEFORE reading source

When investigating or planning changes to AutoWS code, **always read the
relevant docs first** before exploring the source files. The docs explain the
design intent, invariants, and relationships between passes — information that
is difficult to reconstruct from code alone. Reading docs first will:

- Give you the correct mental model before diving into implementation details
- Identify which files are relevant so you search less
- Surface invariants and edge cases that aren't obvious from code

### How to find the right doc

Use the file map below to match your task to the relevant doc(s):

| If you're working on... | Read this doc first |
|---|---|
| Overall pipeline, pass ordering | `docs/Overview.md` |
| Task ID assignment (Hopper) | `docs/TaskPartitionAndPropagation.md` |
| Splitting ops across warp groups | `docs/DataPartition.md` |
| Channel insertion, async copies, barriers | `docs/CodePartition.md` |
| Code specialization / cloning into regions | `docs/CodeSpecialization.md` |
| SMEM/TMEM allocation, multi-buffering | `docs/BufferAllocation.md`, `docs/SmemAllocationDesign.md` |
| Memory planner liveness analysis | `docs/MemoryPlannerVisualization.md` |
| Memory lowering (global/shared/tensor) | `docs/MemoryLowering.md` |
| Token/barrier lowering to hardware | `docs/TokenBarrierLowering.md` |
| Ping-pong scheduling | `docs/PingPongScheduling.md` |
| Barrier fusion/merging | `docs/BarrierFusion.md` |
| Operand D / accumulator handling | `docs/OperandDHandling.md` |
| Reuse groups for buffer sharing | `docs/ReuseGroups.md` |
| TMEM allocation heuristics | `docs/TMEMAllocationHeuristics.md` |
| Utility functions | `docs/Utilities.md` |

### Workflow

1. **Read** the matching doc(s) from the table above.
2. **Then** explore source files, guided by what the docs describe.
3. If no doc matches your task, read `docs/Overview.md` for the pipeline
   context and file map, then proceed to source.

## CRITICAL: Update docs AFTER non-trivial code changes

When you make changes to AutoWS code that go beyond a simple bug fix, you
**must** update the corresponding documentation. Specifically, update docs when:

- **Adding a new pass or file**: Add an entry to `docs/Overview.md` (file map
  and pipeline diagram) and create a new doc if the pass is substantial.
- **Changing pass behavior or invariants**: Update the doc that describes that
  pass to reflect the new behavior.
- **Adding or changing data structures**: Update the doc that references those
  structures.
- **Changing the pipeline order**: Update `docs/Overview.md`.
- **Adding new concepts or terminology**: Document them in the relevant doc or
  create a new one if no existing doc fits.

Do NOT update docs for:
- Pure bug fixes that don't change documented behavior
- Code style / refactoring that preserves semantics

### Doc conventions

- Docs live in `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/`
- Each doc covers one logical area (one pass or closely related group of passes)
- Docs should explain **why**, not just **what** — design rationale matters
- Include the file(s) the doc covers at the top
- Use code snippets or IR examples to illustrate transformations
