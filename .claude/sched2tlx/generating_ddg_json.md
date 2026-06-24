---
description: How to generate the modulo scheduler's Data Dependence Graph dump (ddg.json, schema ddg-0.1) for each sched2tlx example case from its pre_modulo.ttgir. Use when you need to (re)produce ddg.json alongside the existing schedule_graph.json.
---

# Generating `ddg.json` for sched2tlx Example Cases

Each example case under `examples/<case>/` ships a pre-modulo TTGIR input and a
dumped `schedule_graph.json`. This guide produces the companion `ddg.json` — the
**pre-schedule** Data Dependence Graph (schema `ddg-0.1`): everything the modulo
solver consumes (per-op hardware cost nodes, data/loop-carried edges, the derived
MinII analysis, loop structure, and global budgets) *before* any scheduling
results are produced.

The dumper lives in the modulo-schedule pass (added in D108942541) and is
triggered by the `TRITON_MODULO_DUMP_DDG` env var. It runs at the **end** of the
`-nvgpu-modulo-schedule` pass, at the same point as the ScheduleGraph dump
(`TRITON_MODULO_DUMP_SCHEDULE`), so it captures the solver input that yields the
existing `schedule_graph.json`.

## Prerequisites

The dumper is C++; it must be compiled into `triton-opt`. Build (or rebuild)
`triton-opt` first so the binary contains the dumper:

```bash
buck2 build fbsource//third-party/triton/beta/triton:triton-opt --show-full-output
```

The `--show-full-output` line prints the absolute path of the built binary, e.g.
`.../buck-out/.../__triton-opt__/triton-opt`.

## Command

For one case, run the modulo-schedule pass over its pre-modulo TTGIR with the
dump env var pointed at the case folder. `-allow-unregistered-dialect` is
required (some cases contain unregistered ops; it is part of case5's RUN line).
The transformed IR on stdout is not needed — discard it.

```bash
TRITON_MODULO_DUMP_DDG=<case>/ddg.json \
  <triton-opt-bin> <case>/<pre_modulo>.ttgir \
  -allow-unregistered-dialect -nvgpu-modulo-schedule > /dev/null
```

The pre-modulo input filename differs per case:

| Case                    | Input TTGIR                      |
| ----------------------- | -------------------------------- |
| `case1_simple_gemm`     | `pre_modulo.ttgir`               |
| `case2_persistent_gemm` | `pre_modulo.ttgir`               |
| `case3_FA`              | `fa_fwd_nows_pre_modulo.ttgir`   |
| `case5_addmm_bias`      | `addmm_bias_pre_modulo.ttgir`    |
| `case6_layernorm`       | `layernorm_fwd_pre_modulo.ttgir` |

### Do all cases at once

```bash
BIN="$(buck2 build fbsource//third-party/triton/beta/triton:triton-opt \
        --show-full-output 2>/dev/null | awk '{print $2}')"
EX="$(dirname "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")")"   # examples/

declare -A TTGIR=(
  [case1_simple_gemm]=pre_modulo.ttgir
  [case2_persistent_gemm]=pre_modulo.ttgir
  [case3_FA]=fa_fwd_nows_pre_modulo.ttgir
  [case5_addmm_bias]=addmm_bias_pre_modulo.ttgir
  [case6_layernorm]=layernorm_fwd_pre_modulo.ttgir
)

for c in "${!TTGIR[@]}"; do
  D="$EX/$c"
  TRITON_MODULO_DUMP_DDG="$D/ddg.json" \
    "$BIN" "$D/${TTGIR[$c]}" -allow-unregistered-dialect -nvgpu-modulo-schedule \
    > /dev/null 2>"/tmp/$c.ddg.stderr"
  grep -i "Dumped DDG" "/tmp/$c.ddg.stderr" || echo "[$c] DUMP MISSING — check stderr"
done
```

## Gotcha: non-zero exit on persistent / outer-loop cases

`case2_persistent_gemm` and `case5_addmm_bias` make `triton-opt` exit with code
**1**, printing:

```
error: 'arith.muli' op does not have expected attribute ttg.partition
       which is expected for ops whose parent has partitions
```

This is a **post-dump** IR verifier error on the transformed IR (which would
normally be consumed by downstream warp-specialization passes). It fires *after*
`ddg.json` has been fully written, so the dump is complete and valid. **Judge
success by the stderr line, not the exit code:**

```
[modulo-schedule] Dumped DDG (ddg-0.1) to <path>/ddg.json (N loop(s))
```

## Validation

A correct `ddg.json` should satisfy:

- `schema_version == "ddg-0.1"`
- `kernel.name` matches the sibling `schedule_graph.json`'s `kernel.name`
- every node `op_ref` resolves in the top-level `ops` table
- persistent cases (case2, case5) have **2** loops (inner + outer); the others
  have **1**
- each loop has non-empty `ddg.nodes` and `ddg.edges`, plus `min_ii` /
  `res_mii` / `rec_mii`

```bash
python3 - <<'PY'
import json, os, glob
for dpath in glob.glob(os.path.join(os.path.dirname(__file__), "..", "case*", "ddg.json")):
    d = json.load(open(dpath))
    sch = json.load(open(os.path.join(os.path.dirname(dpath), "schedule_graph.json")))
    ops = d.get("ops", {})
    loops = d.get("loops", [])
    unresolved = sum(
        1 for l in loops for n in l["ddg"]["nodes"]
        if n.get("op_ref") and n["op_ref"] not in ops
    )
    ok = (d.get("schema_version") == "ddg-0.1"
          and d["kernel"]["name"] == sch["kernel"]["name"]
          and unresolved == 0 and loops)
    case = os.path.basename(os.path.dirname(dpath))
    print(f"[{case}] {'OK' if ok else 'CHECK'} "
          f"kernel={d['kernel']['name']} loops={len(loops)} "
          f"unresolved_op_refs={unresolved}")
PY
```

## Notes

- The dump env var is `TRITON_MODULO_DUMP_DDG` (the ScheduleGraph counterpart is
  `TRITON_MODULO_DUMP_SCHEDULE`). The older `../design.md` mentions
  `TRITON_DUMP_MODULO_SCHEDULE` / `TRITON_MODULO_DUMP_AND_EXIT` — those names are
  stale; no early-exit flag is needed because the dump runs as the pass finishes.
- Source: `third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloSchedulePass.cpp`
  (`dumpDDGAsJSON`, `jsonDumpDDGLoop`, `jsonDumpDDGNode`, `jsonDumpDDGEdge`).
