# Instruction and Warp Mapping Contract

Create and review `mapping_manifest.json` before CUDA C++ implementation. The
manifest is the auditable contract between `twill-pipelined-warp-ir-v1` and the
manual source; comments alone are not sufficient.

## Source fields that must remain visible

The mapping consumes:

- `pipeline.{ii,length,copies,horizon,regions}`;
- each `warp_groups[]` entry, including `id`, `width`, and `issue_trace`;
- every `instructions[]` entry: `id`, `op_ref`, `op_kind`, `pipeline`, `cycle`,
  `stage`, `offset`, `group`, `group_width`, and `lanes`;
- every `dependencies[]` edge: `index`, `src`, `dst`, `distance`, `latency`,
  `producer_group`, and `consumer_group`;
- `cross_warp_dependencies[]` and source-operation semantics.

Do not infer or repair missing fields. Do not collapse multiple nodes merely
because they lower to the same instruction family.

## Required manifest structure

A mapping manifest must include at least:

```json
{
  "schema_version": "twill-manual-cuda-mapping-v1",
  "authoring_mode": "agent-assisted-manual-cuda",
  "target": {"gpu": "B200", "cuda_arch": "sm_100a"},
  "pipelined_ir": {
    "schema_version": "twill-pipelined-warp-ir-v1",
    "sha256": "<sha256>"
  },
  "status": "manual_completion_required",
  "cuda_source": {"path": "kernel.cu", "sha256": null},
  "build_evidence": {
    "status": "pending",
    "cuda_arch": "sm_100a",
    "source_sha256": null,
    "binary": {"path": null, "sha256": null},
    "disassembly": {"path": null, "sha256": null}
  },
  "physical_groups": [],
  "regions": [],
  "nodes": [],
  "dependencies": [],
  "coverage": {
    "ir_node_count": 0,
    "mapped_node_count": 0,
    "ir_edge_count": 0,
    "mapped_edge_count": 0
  },
  "deviations": []
}
```

Use a real serializer. The empty arrays above describe required sections, not
an acceptable completed manifest. The scaffold uses
`manual_completion_required`; the mapping audit accepts only `approved`, with
a named reviewer, verified source and disassembly status for every node, a
real CUDA source hash, complete coverage counts, and no deviations.
The final bundle also binds the exact source, 64-bit NVIDIA cubin, and
disassembly hashes. Source anchors are unique per node. A node with an
exclusive machine instruction needs a unique disassembly anchor; nodes sharing
one instruction must share a reviewed equivalence proof. A `NONE`-pipeline
zero-cost alias has no disassembly anchor and must carry its own proof.

## Physical group mapping

For every Twill group, record:

- group ID and solved width;
- physical warp IDs and CTA rank;
- active lanes and how a Twill lane list maps to CUDA thread IDs;
- register allocation target if controlled;
- source dispatch anchor and entry predicate;
- region participation in prologue, steady state, and epilogue.

Do not assign conventional names such as "TMA warp" or "MMA warp" first and
then fit nodes into them. Derive a descriptive role only after mapping the
group's solved instruction set. Group widths and lane masks are constraints,
not tuning suggestions.

## Per-node mapping

Each `nodes[]` entry must record:

- all original identity and schedule fields;
- CUDA source file and stable anchor/label;
- physical group, warp, lane predicate, and CTA scope;
- loop coordinates and iteration expression;
- realized region and stage/slot expression;
- semantic lowering kind and exact primitive or scalar operation;
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
- per-group ordering by `(stage, offset, node id)`;
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
Edges within one group still need an explicit program-order or async-completion
argument.

## Issue-order audit

For every group:

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
is relocated, a lane mask widened, an MMA reordered, or a dependency replaced:

- stop the faithful-lowering workflow;
- request a new solver artifact, or
- label the implementation as an explicitly non-faithful experiment.

Never use correctness or good performance as retroactive approval for an
unrecorded mapping change.
