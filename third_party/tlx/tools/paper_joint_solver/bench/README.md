# Paper benchmark bars

`bench_bars.py` evaluates the paper shapes (`fp16`, non-causal, `B=4`,
`H=32`, `D=128`, and sequence lengths 2048 through 16384). Performance runs
require a B200; registry and skip-behavior tests do not require a GPU.

## Bar mapping

| Runner bar | Paper comparison |
| --- | --- |
| `tlx_default` / `tlx_bwd_default` | TLX equivalent of CUDA-Default |
| `tlx_jos_fwd` | Forward system bar from the non-paper emitter-compatible v8 solve |
| `tlx_jos_bwd` | Backward default-budget bar from the non-paper emitter-compatible v8 solve |
| `tlx_jos_bwd_lr` | Backward LR4096 bar from the non-paper emitter-compatible v8 solve |
| `cudnn` / `cudnn_bwd` | cuDNN |
| `fa4` / `fa4_bwd` | FA4, run through the separate FA4 environment |

The JOS builders expect these emitter outputs:

```text
case3_FA_fp16_subtiled/generated_jos_v8.py
case4_FA_bwd_subtiled/generated_jos_v8.py
case4_FA_bwd_subtiled/generated_jos_lr4096_v8.py
```

The paper-fidelity artifacts from `run_main_cases.sh` are evidence artifacts,
not sched2tlx inputs: their physical lane masks may be non-prefix. Do not
canonicalize or relabel them. On WS-D, run the separate solver workflow that
adds both prefix-lane and full-group-lane selection constraints:

```bash
SOLVER_LIB_PATH=<yices-lib>:<cudd-lib> ./run_emitter_cases.sh
```

These are explicit backend-realizability constraints and are not part of the
paper model. The script creates separately named
`fwd_subtiled_emitter_v8.json`, `bwd_emitter_v8.json`, and
`bwd_lr4096_emitter_v8.json`; commands, logs, environment details, and strict
post-validation remain separate from the paper v8 gates.

Materialize those emitter-mode solutions with the append-never workflow:

```bash
env -u LD_LIBRARY_PATH ../../../../.venv/bin/python \
  emit_bench_kernels.py run --solution-dir solutions
```

The workflow refuses paper-mode, non-SAT, or legacy solutions; stale
DDG/baseline/source hashes; unexpected machine budgets; missing
emitter-constrained optimal-search evidence; missing backward TMEM-liveness
evidence; mixed raw/normalized timing; and every pre-existing output. It
stages and validates all three rewritten graphs before publishing:

```text
case3_FA_fp16_subtiled/schedule_graph_jos_v8.json
case4_FA_bwd_subtiled/schedule_graph_jos_v8.json
case4_FA_bwd_subtiled/schedule_graph_jos_lr4096_v8.json
```

Commands, logs, staged artifacts, hashes, and the success/failure manifest are
written to `solutions/bench_emit_emitter_v8/`. The manifest and every JOS
benchmark result label the schedule as `non-paper-emitter-compatibility`. A
failed or completed record is never reused; remove stale targets deliberately
or choose a fresh `--record-dir`.
This workflow only generates kernels. It does not run performance benchmarks
or create benchmark result JSON.

When emission used a custom record directory, pass its manifest to the runner
with `--jos-manifest <record-dir>/manifest.json` or set
`JOS_EMISSION_MANIFEST`. Every JOS bar refuses to run without a successful
manifest whose source, solution, rewritten-graph, and kernel hashes still
match; the selected manifest path and all hashes are copied into each result.

The pre-v8 measurements are retained under `bench/archive/` and are explicitly
non-comparable with the corrected model. Fresh `results_fwd.json` and
`results_bwd.json` must only be created after the corresponding v8 generated
kernels exist; until then the new bars report `SKIP` with a reason.

The paper's Triton-WS variants and SWP-only variants have no exact bar in this
runner. `triton_ws_off`, `triton_ws_on`, and `triton_tiled` are useful local
controls, but they are not labeled as those missing paper analogues.

## Guards and timing

Every in-process bar launches once before timing. That launch both compiles the
kernel and checks its output. A missing generated file, import/compile failure,
or correctness failure is retained in the output JSON with `status: "SKIP"`,
`skip_phase`, and `reason`; it is never silently omitted.

Forward bars report kernel time. TLX backward bars also report only the fused
backward kernel: reference gradients and the `M`/`D` preprocessing inputs are
prepared outside the timed region. cuDNN and FA4 backward bars use their
end-to-end forward-plus-backward measurement minus a separately measured
forward median. These boundaries are intentionally recorded here because the
two backward timing methods are not identical.
