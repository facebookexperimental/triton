# schedule_graph.json — Schema

`schema_version: "0.1"`. Output of modulo Pass A's `dumpScheduleGraphAsJSON`,
input to the TLX emitter. Designed to be self-sufficient — the emitter does
not need the original IR or any out-of-band context.

## Top-level

```json
{
  "schema_version": "0.1",
  "kernel": { ... },
  "ops": { "<id>": { ... }, ... },
  "data_partition_candidates": [ { ... }, ... ],
  "launch_hints": { "memory_bound": true, "total_warps": 12,
                    "maxnreg": 56, "grid_multiplier": 4 },
  "loops": [ { ... }, ... ]
}
```

- `kernel` — function signature.
- `ops` — flat **table** (string id → op record) of every op the emitter may
  need: function-scope (preamble/epilogue) ops AND in-loop scheduled ops.
- `data_partition_candidates` — Pass A.5 candidate surface (may be absent in
  older dumps). One entry per MMA node with at least one legal M-split:

  ```json
  {"loop_id": 0, "node_id": 5, "op_kind": "ttng.tc_gen5_mma", "dim": 0,
   "applied_n": 1, "factors": [{"n": 2, "m_size": 64}]}
  ```

  `factors` lists every legal split factor (BM % n == 0, BM/n >= max(64,
  TMEM blockM)); `applied_n` is the factor actually applied in THIS dump
  (1 = unpartitioned). The emitter ignores this section — it exists so
  external tooling can re-run the modulo pass with
  `TRITON_DATA_PARTITION_N=<n>` per candidate and compare the resulting
  schedules.
- `launch_hints` — OPTIONAL; present only for memory-bound warp-specialized
  kernels (no MMA in the module, ≥1 TMA node, ≥2 warp groups). Without a
  maxnreg cap the WS lowering auto-fills the full register file, pinning
  residency at 1 CTA/SM — right for TC-bound kernels, wrong for memory-bound
  ones. `maxnreg` = regs/thread sized for 3 co-resident CTAs given
  `total_warps` (default WG + non-default WGs padded to 4-warp groups);
  `grid_multiplier` = persistent-grid scale (× NUM_SMS). The emitter forwards
  these as `RECOMMENDED_MAXNREG` / `RECOMMENDED_GRID_MULTIPLIER` module
  constants in the generated kernel; launchers pass `maxnreg=` and scale the
  grid. Measured on B200 case6 LayerNorm: 3161 → 6185 GB/s (parity with the
  hand-written reference) with no kernel-body change.
- `loops` — each scheduled loop's schedule graph (nodes + edges) plus its
  bounds and induction variable.

## `kernel`

```json
{
  "name": "gemm_kernel",
  "args": [
    {"name": "A", "type": "*f16"},
    {"name": "M", "type": "i32"}
  ]
}
```

`type` strings follow Triton signature conventions (e.g. `*f16`, `i32`).

## `ops` table

Every op the emitter may reference appears here. Keys are stable string ids
derived from MLIR pointers (opaque — only used for cross-references).

```json
"ops": {
  "op_181268768": {
    "kind": "tt.get_program_id",
    "scope": "function",
    "operands": [{"const": 0, "type": "i32"}],
    "attributes": {},
    "result_types": ["i32"]
  },
  "op_181362800": {
    "kind": "tt.descriptor_load",
    "scope": "loop:0",
    "operands": [
      {"op": "op_X"},
      {"op": "op_Y"},
      {"op": "op_Z"}
    ],
    "attributes": {},
    "result_types": ["tensor<128x64xf16>"]
  }
}
```

### Op record fields

| field | type | meaning |
|---|---|---|
| `kind` | string | MLIR op name (e.g. `tt.descriptor_load`, `ttg.local_alloc`, `ttng.tc_gen5_mma`) |
| `scope` | `"function"` \| `"loop:<id>"` | where the op lives |
| `operands` | list of operand refs | structured references (see below) |
| `attributes` | object | flat key/value bag of MLIR attrs we care about |
| `result_types` | list of strings | MLIR-printed types |

### Operand refs (the heart of structured operands)

```
{"op":   "<id>"}                 reference to another op's result
{"arg":  "<name>"}               reference to a function argument by name
{"iv":   <loop_id>}              reference to a loop's induction variable
{"const": <value>, "type": "..."} literal constant
{"block_arg": {"region": "...", "idx": N}}  scf.for iter_arg refs (rare)
```

Recursive resolution: the emitter expands these into Python expressions by
recursively emitting the referenced op or substituting the arg/iv/const.

### Buffer-bearing ops

`ttg.local_alloc` / `ttng.tmem_alloc` carry buffer info as attributes:

```json
"attributes": {
  "buffer": {
    "shape": [128, 64],
    "dtype": "f16",
    "kind": "smem",
    "count": 3,
    "buffer_id": 0
  }
}
```

`count` and `buffer_id` come from modulo's Steps 3-4.5 (depth + interval-graph
coloring); the emitter uses these to emit one `tlx.local_alloc` per `buffer_id`.

## `loops[i]`

```json
{
  "id": 0,
  "II": 1038,
  "max_stage": 2,
  "warp_groups": ["TMA", "TC"],
  "induction_var": {"name": "k", "type": "i32"},
  "lower_bound": {"const": 0, "type": "i32"},
  "upper_bound": {"arg": "K"},
  "step": {"const": 64, "type": "i32"},
  "graph": { "nodes": [...], "edges": [...] }
}
```

- `induction_var.name` — recovered from MLIR location (e.g. `arg12`); the
  emitter is free to rename for readability.
- `warp_groups` — Phase 4 partitioning plan in declaration order
  (`"default"` first if present, then own-group pipelines).

### `graph.nodes`

```json
{
  "id": 0,                       // = DDG node idx, stable within this loop
  "op_ref": "op_181362800",      // key into top-level ops table
  "pipeline": "TMA",             // "TMA" | "TC" | "CUDA" | "SFU" | "NONE"
  "partition": 0,                // index into loops[i].warp_groups
  "latency": 518,
  "self_latency": 518,
  "schedule": {
    "cycle": 0,
    "stage": 2,
    "cluster": 0
  }
}
```

- `pipeline` is the HW classification (TMA / TC / CUDA / SFU / NONE).
- `partition` is the warp-group assignment (Phase 4 plan); resolves to
  `loops[i].warp_groups[partition]`. For pipelines merged into the default
  group, this is the default group's index. Used by the emitter to route
  each op into the correct `tlx.async_task` body.
- `latency` is the consumer-visible latency (used for edge latencies).
- `self_latency` is the pipeline occupancy (used for ResMII).
- `cycle = stage * II + (cluster contribution)` — but `cycle` is authoritative;
  `stage` and `cluster` are derived for downstream convenience.

#### Optional epilogue-subtile fields (Pass A.7)

Emitted only when an op was produced by splitting an epilogue chain into S
independent sub-chains along the N dimension. When absent, the op operates
on the full `(BM, BN)` tile.

```json
{
  ...
  "subtile_index": 2,   // 0..S-1, this op's slot
  "subtile_count": 4,   // S
  "n_offset": 128,      // i * (BN/S)
  "n_size": 64          // BN/S
}
```

The emitter groups sibling sub-ops by `subtile_count` and emits a single
`for sub in range(S):` block templated on `n_offset`/`n_size`.

### `graph.edges`

```json
{
  "src": 0,
  "dst": 1,
  "kind": "data",      // "data" | "anti" | "mem" (future)
  "distance": 0,       // 0 = intra-iteration, >=1 = loop-carried
  "latency": 518
}
```

## What the emitter does with this

1. **Preamble** = walk all `scope == "function"` ops in source order, emit each
   as a Python assignment (`<name> = <expr>`) using the recursive expression
   builder over `operands`.
2. **Per-loop allocs** = walk loop nodes whose op has `attributes.buffer`,
   group by `buffer_id`, emit one `tlx.local_alloc` per group.
3. **Mbarriers** = walk `graph.edges` with `kind=="data"` between nodes whose
   pipelines differ, group by destination buffer, emit one full+empty pair
   per channel.
4. **Per-warpgroup task body** = for each pipeline P in `warp_groups`, filter
   nodes with `pipeline == P`, sort by `(cycle, cluster)`, emit a `for IV in ...:`
   loop containing one TLX op per node (recursive operand expansion).
5. **Epilogue** = function-scope ops that come after the loop in source order.

No hand-authored context needed — the dumper provides everything.
