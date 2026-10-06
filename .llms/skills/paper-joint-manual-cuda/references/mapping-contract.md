# Instruction and Warp Mapping Contract

Create and review `mapping_manifest.json` before CUDA C++ implementation. The
manifest is the auditable contract between `paper-joint-pipelined-ir-v3` and the
manual source; comments alone are not sufficient.

## Source fields that must remain visible

The mapping consumes:

- `pipeline.{ii,length,copies,horizon,regions}`;
- each top-level `warps[]` entry, including `id` and `issue_trace`; a warp has
  no width, and a multi-warp instruction appears in every member warp's trace;
- every `instructions[]` entry: `id`, `op_ref`, `op_kind`, `pipeline`, `cycle`,
  `stage`, `offset`, and `warps` (the sorted physical warp IDs that issue it);
- every `dependencies[]` edge: `index`, `src`, `dst`, `distance`, `latency`,
  `producer_warps`, `consumer_warps`, `ws_semantics`, and `spill_cost`;
- `pipelined_program.instances`, `pipelined_program.instance_dependencies`, and
  `pipelined_program.steady_state.slots`, which the manifest replays verbatim;
- `cross_warp_dependencies[]` and source-operation semantics.

Do not infer or repair missing fields. Do not collapse multiple nodes merely
because they lower to the same instruction family.

## Required manifest structure

A mapping manifest must include at least:

```json
{
  "schema_version": "paper-joint-manual-cuda-mapping-v2",
  "authoring_mode": "agent-assisted-manual-cuda",
  "target": {"gpu": "B200", "cuda_arch": "sm_100a"},
  "pipelined_ir": {
    "schema_version": "paper-joint-pipelined-ir-v3",
    "sha256": "<sha256>"
  },
  "status": "manual_completion_required",
  "cuda_source": {"path": "kernel.cu", "sha256": null},
  "build_evidence": {
    "status": "pending",
    "command": null,
    "toolchain": null,
    "cuda_arch": "sm_100a",
    "source_sha256": null,
    "binary": {"path": null, "sha256": null},
    "disassembly": {"path": null, "sha256": null},
    "reviewer": null
  },
  "physical_warps": [
    {
      "warp": 0,
      "thread_mapping": null,
      "dispatch_anchor": null,
      "entry_predicate": null,
      "regions": null,
      "reviewer": null
    }
  ],
  "regions": [],
  "nodes": [],
  "dependencies": [],
  "coverage": {
    "ir_node_count": 0,
    "mapped_node_count": 0,
    "ir_edge_count": 0,
    "mapped_edge_count": 0,
    "ir_instance_count": 0,
    "mapped_instance_count": 0,
    "ir_instance_dependency_count": 0,
    "mapped_instance_dependency_count": 0,
    "ir_steady_state_slot_count": 0,
    "mapped_steady_state_slot_count": 0
  },
  "deviations": []
}
```

Use a real serializer. The empty arrays above describe required sections, not
an acceptable completed manifest. The scaffold uses
`manual_completion_required`; the mapping audit accepts only `approved`, with
a named reviewer, verified source and disassembly status for every node, a
real CUDA source hash, complete coverage counts, and no deviations.
Build evidence must also name the exact `command`, `toolchain`, and `reviewer`.
The final bundle also binds the exact source, 64-bit NVIDIA cubin, and
disassembly hashes. Source anchors are unique per node. A node with an
exclusive machine instruction needs a unique disassembly anchor; nodes sharing
one instruction must share a reviewed equivalence proof. A `NONE`-pipeline
zero-cost alias has no disassembly anchor and must carry its own proof.

## Physical warp mapping

The IR names physical warps directly, so no logical group is left for the
author to bind. Record one `physical_warps[]` entry per warp in `warps[]`:

- `warp`: the physical warp ID exactly as the IR names it;
- `thread_mapping`: how that warp's threads map to CUDA thread IDs and CTA rank;
- `dispatch_anchor`: a source anchor occurring exactly once in the CUDA source;
- `entry_predicate`: the predicate that admits threads to the warp's work;
- `regions`: participation in prologue, steady state, and epilogue;
- `reviewer`: the human who approved the entry;
- register allocation target, if it is controlled.

Every scheduled warp must be realized exactly once, and no unscheduled warp may
appear. Do not assign conventional names such as "TMA warp" or "MMA warp" first
and then fit nodes into them. Derive a descriptive role only after mapping the
warp's solved instruction set. The solved warp assignment is a constraint, not
a tuning suggestion.

## Per-node mapping

Each `nodes[]` entry must record:

- all original identity and schedule fields, replayed verbatim: `op_ref`,
  `op_kind`, `pipeline`, `cycle`, `stage`, `offset`, and `warps`;
- `issue_order`, `instances`, and `steady_state_slot`, replayed verbatim from
  the IR issue traces and `pipelined_program`;
- CUDA source file and stable anchor/label;
- `manual_cuda.physical_warps`, equal to the instruction's `warps`, plus lane
  predicate and CTA scope;
- loop coordinates and iteration expression;
- realized region and stage/slot expression;
- semantic lowering kind and exact primitive or scalar operation;
- `semantic_completion`: the single point at which the node's semantics are
  complete;
- operand/result IDs from `memory_plan.json`;
- synchronization IDs from `sync_manifest.json`;
- predicate and boundary behavior;
- source status and disassembly status;
- machine realization (`instruction`, `shared_instruction`, or a reviewed
  `zero_cost_alias`) and any required equivalence proof;
- review state and reviewer.

If one IR node expands into multiple CUDA/PTX operations, list the ordered
expansion and identify the single semantic completion point. If several IR
nodes share one machine instruction, provide an explicit equivalence proof;
otherwise reject the merge.

## Region realization

Record how the exact cycle intervals for prologue, steady state, and epilogue
map to source control flow. For every scheduled copy, preserve:

- `cycle = stage * ii + offset`;
- per-warp ordering by `(stage, offset, node id)`;
- the solved number of overlapping copies;
- loop-distance interpretation on dependency edges;
- fill and drain operations that exist only outside steady state.

Source loops may be compact, but the manifest must expand them symbolically
enough to show which dynamic instruction instance realizes each scheduled
copy. Do not validate only the steady-state body.

## Dependency mapping

Each dependency entry must identify:

- source and destination node IDs and dynamic iteration relation;
- value, memory region, or ordering token carried by the edge;
- whether program order is sufficient;
- if not, the synchronization-channel ID and completion witness;
- how `distance` and `latency` are preserved;
- prologue availability and epilogue drain behavior.

Every cross-warp dependency must point to an entry in `sync_manifest.json`.
Edges whose `producer_warps` and `consumer_warps` are equal still need an
explicit program-order or async-completion argument.

## Issue-order audit

For every warp:

1. Sort IR nodes by `(stage, offset, id)` using `issue_trace` as the source of
   truth.
2. Expand the source loop for at least one full pipeline period plus fill and
   drain.
3. Compare dynamic source anchors to the expected trace.
4. Check that compiler motion cannot cross required fences or dependencies.
5. Confirm the intended instruction sequence and shapes in disassembly.

Do not require identical cycle timing from CUDA source or SASS; the contract is
that ordering, ownership, dependencies, and pipeline overlap preserve the
reviewed schedule intent. Any deliberate relaxation belongs in `deviations`.

## Deviations and failure policy

An empty `deviations` list is required for a faithful lowering claim. If a node
is relocated, a warp assignment changed, an MMA reordered, or a dependency
replaced:

- stop the faithful-lowering workflow;
- request a new solver artifact, or
- label the implementation as an explicitly non-faithful experiment.

Never use correctness or good performance as retroactive approval for an
unrecorded mapping change.
