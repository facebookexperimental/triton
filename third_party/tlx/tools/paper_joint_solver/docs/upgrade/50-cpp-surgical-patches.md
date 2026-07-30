# 50 — Main-Repo ModuloScheduling Surgical Patches (P1/P2/P3) Implementation Design

> Status: v1 design draft (2026-07-30). **Design only, not implemented.**
> Positioning: a companion piece to Plan 2 (solver branch as the home base). The three patches each become independent PRs, **all landing on small branches freshly cut from main** (not on `hwu27/paper-joint-solver`), so that both worktrees can inherit them via rebase/merge.
> Discipline: C++ changes require recompiling with `pip install -e . --no-build-isolation` to take effect; run `pre-commit run --all` before committing; **with their switches off, all patches behave bit-identical to the status quo** (zero behavior change by default), and each PR ships with a LIT regression proving this property.

## 0. General Rules

1. New env vars must be registered in the whitelist in `include/triton/Tools/Sys/GetEnv.hpp` (otherwise `tools::getStrEnv`/`getBoolEnv` refuse to read them), and their cache-invalidation attribute must be decided: P1/P2 change the compilation artifact → **cache-invalidating**; P3 only changes dump content, not the artifact → non-invalidating.
2. The three patches introduce no mutual dependencies and can be reviewed in any order; the recommended landing order is in §4.
3. Reader-side contract: `paper_joint_solver/ddg.py` and sched2tlx consume the JSON **tolerantly** without exception (unknown fields ignored, missing fields fall back), guaranteeing cross-version interoperability.

## 1. P1 — NVLatencyModel Calibration Override Entry Point (post-W1)

### 1.1 Motivation

After W1's DiffTune-style fitting produces `machine_calib.json`, the **shared water source both camps drink from** (NVLatencyModel is the sole source of `ddg.json` costs) must be able to consume the calibrated values without code changes; at the same time, the hardcoded values are kept as fallback so calibration experiments can be A/B'd.

### 1.2 Design

- env: `TRITON_MODULO_LATENCY_CALIB=<json path>`. Unset → fully take the existing hardcoded path (bit-identical).
- Load timing: read once at pass construction and cache as `struct LatencyCalib` (per-field `std::optional<int>`); NVLatencyModel gains a member `const LatencyCalib *calib` (may be null). In each method, the function-level `constexpr` constants are rewritten into the form `calib && calib->X ? *calib->X : kDefaultX` — this is the only kind of intrusion point in this patch.
- JSON schema (top level `{"schema_version": "calib-0.1", "nv_latency_model": {...}, "modulo_pass": {...}}`; the key names are exactly this table, serving as the alignment contract with the W1 doc's machine_calib.json):
  - Parsed with `llvm/Support/JSON.h`; `schema_version` mismatch → hard error; unknown keys → warning + ignore; missing keys → fall back to hardcoded values.
  - On successful load, print to `llvm::errs()` (DEBUG_TYPE gated) "the list of overridden coefficients + the source file hash", for experiment-log traceability.

### 1.3 Overridable Coefficient List

| JSON key | Current value | Code location |
|---|---|---|
| `tma.load.base / per_kb / per_inst / inst_cap` | 460 / 6 / 240 / 2 | NVLatencyModel.cpp:157 |
| `tma.store.base / per_kb` | 130 / 52 | :171 |
| `tma.issue` | 30 | :70 |
| `mma.lat_k128 / lat_k64` | 900 / 559 | :181-182 |
| `mma.issue` | 30 | :76 |
| `tc.macs_per_cycle_fp16` | 4096 | :197 |
| `tmem.load_base / store_base` (@16384 elems) | 532 / 96 | :326-333 |
| `tmem.load_selflat / store_selflat` | 256 / 48 | :411-413 |
| `reduce.sum_last / sum_outer / max_last / max_outer` | 1691 / 963 / 461 / 961 | :363 |
| `sfu.exp2 / log2` (@16384) | 1140 / 11268 | :475-477 |
| `barrier.wait / arrive` | 30 / 20 | :313-316 |
| `xwg.mbarrier_issue / named_issue` | 45 / 30 | ModuloSchedulePass.cpp:2107-2120 |
| `xwg.roundtrip_base / smem_per_kb` | 150 / 16 | :2913-2952 |

**Consistency point**: `tma.load.per_kb=6` and `tma.store.per_kb=52` appear in both the latency law and the occupancy formula (:683-688) — the same physical quantity (HBM bandwidth share); the config struct stores a single copy shared by both sites, and overriding them separately is forbidden (otherwise ResMII and RecMII would use different bandwidth assumptions).

### 1.4 Touch Points and Size

NVLatencyModel.cpp (constants made configurable, ~120 lines), two groups of constants in ModuloSchedulePass.cpp (~40 lines), new files `LatencyCalib.h/.cpp` (loader, ~120 lines), GetEnv.hpp registration (1 line). Total diff ≈ 280 lines.

### 1.5 LIT Tests

`test/TritonGPU/modulo-latency-calib.mlir` + `Inputs/calib_test.json` (overriding only `mma.lat_k128: 1234`). Two RUN lines: no env → DDG dump CHECK `latency = 900`; `env TRITON_MODULO_LATENCY_CALIB=%S/Inputs/calib_test.json` → CHECK `latency = 1234` and the log contains "override: mma.lat_k128". An additional RUN with a bad schema_version verifies the hard error.

### 1.6 Risks

Semantic drift between the calibration file and the code (the formula a key points at gets refactored) — mitigated by the `schema_version` hard gate + override log; the review burden lies in the mechanical "constexpr → config member" rewrite, which must be checked item by item for completeness (the coefficient table is the checklist).

## 2. P2 — per-WG Register Budget Knob (post-W3 productionization)

### 2.1 Motivation and Architectural Boundary

The W3-L0 closed loop (ptxas spill → tighten budget → re-solve) **stays outside the compiler**: an external Python driver repeatedly compiles, parses `ptxas -v`, and bisects the budget. The compiler side only needs a per-WG budget knob that the external driver can inject — the current register model is a global bucket (`regsForWarpCount` 24/152/232 @ModuloSchedulePass.cpp:2480-2527) with no per-group tightening entry point. **This patch is done only after the solver branch's Python closed loop has proven its value on the bwd golden case.**

### 2.2 Design

- env: `TRITON_MODULO_WG_REG_BUDGETS=<csv>` (e.g. `152,24,104,152`, indexed by warp group; empty/unset → status quo). Indices beyond the list's length fall back to the `regsForWarpCount` bucket value; extra entries are ignored with a warning.
- Consumption point: the static register model in Phase 4 scoring — the deficit computation changes from "footprint − bucket value × thread count" to "footprint − min(bucket value, budget[wg]) × thread count", still going through the `kDeficitPenalty=0.5` soft penalty (:3659-3666). **No new hard constraint**: keep the soft-penalty semantics to avoid an abrupt semantic shift between scoring and the feasible region; the external driver determines whether the tightening took effect by reading partition_cost/residual from the dump.
- Relation to the Step 4.6 SMEM budget-reduction loop (reduceBufferGroup fixed-point II recomputation @:1484-1767, the budget-reduction LIT semantics): **not reused**. That loop tightens SMEM ring depth, whereas this patch only changes the register penalty baseline; the two naturally coexist in scoring.

### 2.3 Touch Points and Size

ModuloSchedulePass.cpp: env parsing + passing the budgets vector into the scoring context (~30 lines), deficit-computation rewrite (~20 lines), echoing the budgets in the dump (~10 lines); GetEnv.hpp registration. Total ≈ 70 lines.

### 2.4 LIT Tests

`test/TritonGPU/modulo-wg-reg-budgets.mlir`: no env → CHECK the baseline's selected partition signature; `TRITON_MODULO_WG_REG_BUDGETS=24,24,24,24` (extreme tightening) → CHECK the scoring log shows a nonzero residual / a change in partition selection (reusing TRITON_MODULO_DUMP_TOPN's partition_cost output as the CHECK anchor).

### 2.5 Risks

Coupling between the soft-penalty coefficient (0.5) and the tightening magnitude — under extreme budgets the soft penalty may still be insufficient to flip the selection; the external driver must use "selection change" rather than "budget value" as the bisection feedback signal; the docs state this in the driver-side protocol.

## 3. P3 — ddg.json Added Fields (ddg-0.1 → ddg-0.2, pre-W4)

### 3.1 Motivation

W4.1's audit ceiling needs `BW_lb = DRAM bytes per iteration / measured bandwidth` and `Issue_lb`, and the TFLOPS conversion needs MACs per iteration. Currently the byte count can only be back-solved from `occupancy = 6·KB` (fragile, and the store coefficient differs), and MACs are entirely unobtainable.

### 3.2 Design

- Each node gains optional fields: `tensor_bytes` (total bytes for TMA-class nodes; sourced from `getTMATile` @NVLatencyModel.cpp:96-137 — currently file-static, so it must either be promoted to a public method of `NVLatencyModel` or computed at DDG build time and stored into `DDGNode`; the latter is recommended: add virtual methods `getTensorBytes/getMACs` to `LatencyModel`, defaulting to returning 0, with the NV implementation reusing the existing shape extraction) and `macs` (M·N·K for MMA nodes; sourced from the shape extraction in getMMAOccupancyCycles @:224-260).
- Loop-level aggregate field: `flops_per_iter = Σ 2·macs` (FMA = 2 FLOPs), written into the existing loop config block.
- Schema field `"ddg-0.1"` → `"ddg-0.2"`; dump site = the TRITON_MODULO_DUMP_DDG serialization point (ModuloSchedulePass.cpp:6079-6244).
- Backward compatibility: the read side of `paper_joint_solver/ddg.py` handles absences tolerantly (missing `tensor_bytes/macs` → None; when W4 tools encounter None they fall back to the occupancy back-solve and flag it as "low confidence" in the report); ddg.py accepts both the 0.1 and 0.2 versions.

### 3.3 Touch Points and Size

Virtual methods in LatencyModel.h (~15 lines), NVLatencyModel.cpp implementation (~40 lines, reusing existing extractors), DataDependenceGraph node fields and population (~20 lines), dump serialization (~20 lines). Total ≈ 95 lines.

### 3.4 LIT Tests

Extend the existing `modulo-schedule-graph.mlir` family: CHECK the dump contains `"schema": "ddg-0.2"`, TMA node `"tensor_bytes": 32768` (128×128 fp16), MMA node `"macs": 1048576` (128×128×64), and `flops_per_iter` in the loop block.

### 3.5 Risks

Lowest (purely additive fields). The only caveat: store-class ops take the operand fallback path in `getTMATile`, so the tests must cover one load and one store each.

## 4. Landing Order and Dependencies

| Order | Patch | Prerequisite | Trigger condition |
|---|---|---|---|
| 1 | P3 (ddg-0.2) | None | Doable immediately; W4.1's BW_lb/TFLOPS conversion needs it |
| 2 | P1 (calib override) | W1 produces machine_calib.json (the mechanism can land first, using a test json) | After W1's fitting passes the ranking gate |
| 3 | P2 (per-WG budgets) | W3 golden case (bwd zero-manual 2-group→3-group) running end-to-end on the solver branch | When deciding to productionize L0 |

## 5. worktree / rebase / Recompile Notes

- For the solver worktree (`triton-beta-3-paper-solver-wt`, based on main@e9e84e901) to inherit the patches, it needs a rebase or merge of main; **afterwards, editable triton must be reinstalled in that worktree's .venv** (the pitfall from §14 of the replication plan: leftover uv `_C/.tmp*` empty shells hijack triton into a namespace package and must be cleaned up manually).
- The main worktree (`/projects/kzhou6/hwu27/triton-beta`) likewise: **after any C++ change, run `pip install -e . --no-build-isolation`, otherwise the change silently fails to take effect** — historically the most-tripped-over pitfall; all three patch PR descriptions must repeat this reminder.
- LIT run note: `triton-opt` runs only on dgx003 (cluster constraint); local verification outside CI must happen on that node.
