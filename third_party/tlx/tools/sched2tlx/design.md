# sched2tlx — Project Overview

A Python tool that lowers modulo's `schedule_graph.json` to runnable TLX
Python source. The emitter is a mechanical lowering — no optimization, no
pattern matching — that ports the auto-WS pipeline's algorithm to a TLX
output target.

_Last updated: 2026-05-17._

## Why

The auto-WS path (modulo Pass A → AutoWS → SoftwarePipeliner) keeps blowing
up SMEM and fighting modulo's intent. Per-partition cloning, K×N alloc
duplication, and pipeliner re-buffering are architectural mismatches with
modulo's "ONE alloc, N copies, lifetime-merged" model. See
`issues/001_annotation_smem_overflow/STATUS.md`.

Rather than fight the auto-WS lowering, take modulo's schedule and emit
**TLX Python source** directly. TLX is hand-written, fully explicit, and
the reference kernels we tune from (e.g. `blackwell_gemm_ws.py`,
`blackwell_fa_ws.py`) already encode exactly the structure modulo wants to
produce. Mechanical translation, not research.

## Pipeline

```
kernel.py
    │
    │ TRITON_USE_MODULO_SCHEDULE=1
    ▼
[Modulo Pass A]
    │
    │ TRITON_DUMP_MODULO_SCHEDULE=path/to/graph.json
    ▼
schedule_graph.json
    │
    │ python -m sched2tlx <graph.json> -o generated.py
    ▼
generated.py (TLX Python source)
    │
    │ Triton compile (any TLX-supporting Triton)
    ▼
PTX → cubin → run on GPU
```

The emitter is **compiler-agnostic on the output side**: the generated TLX
runs through whatever TLX-supporting Triton you have. The dumper side lives
in modulo Pass A (auto-ws / fbsource beta).

## What's in this directory

| Path | Role |
|---|---|
| `sched2tlx/` | Python package (the emitter itself). |
| `sched2tlx/schedule_graph.py` | JSON → typed dataclasses. |
| `sched2tlx/semaphore_ir.py` | Symbolic semaphore IR (derive → combine → assign stage/phase → lower to mbarrier). Default barrier-emission path. |
| `sched2tlx/emitter.py` | The lowering algorithm. |
| `sched2tlx/cli.py` | `python -m sched2tlx <graph.json> -o out.py` |
| `examples/case1_simple_gemm/` | Single-loop GEMM end-to-end (real modulo dump + generated kernel + GPU runner + handwritten reference). |
| `examples/case2_persistent_gemm/` | Persistent GEMM (nested loops + TMEM hand-off). |
| `examples/case3_FA/` | Flash-Attention forward (non-WS source; emitter produces a 6-WG split). `perf_generated_vs_handwritten.py` benchmarks against the hand-written TLX-WS reference. |
| `examples/case4_FA_bwd/` | Flash-Attention backward — gold-standard reference kernels (`handwritten_nows.py`, `handwritten.py`) + runners. Emitter wiring is the next milestone; pre-modulo TTGIR not yet generated. |
| `examples/case5_addmm_bias/` | Persistent GEMM + 2D bias add (`addmm(a, b) + bias`). Pre-modulo TTGIR + dumped `schedule_graph.json` are in place; non-WS and TLX-WS reference kernels pass on B200. Emitter currently emits a `<tma_load_inline_unsupported>` placeholder for the bias TMA load in the outer-loop epilogue — see emitter_design.md "Open questions". |
| `SCHEMA.md` | The JSON input contract (v0.1). |
| `emitter_design.md` | Algorithm description + implementation status. |
| `design.md` | This file (project overview). |

## Status

- **case1** (single-loop GEMM): generated kernel runs on B200, bit-exact
  vs `torch.matmul` across 5 shapes (1024³, 2048³, 4096³, 1024×1024×960,
  1024×1024×1152).
- **case2** (persistent GEMM with nested K-loop, TC↔default TMEM hand-off):
  generated kernel runs on B200, bit-exact across 6 shapes (256×256×128 →
  8192³ + 1024×1024×16384). Uses modulo's actual TMEM count=2 ring buffer
  for cross-tile pipelining.
- **case3** (Flash-Attention forward, non-WS pre-modulo source): generated
  kernel runs on B200, PASS vs `F.scaled_dot_product_attention` across 6
  shapes ((1,4,512) → (1,32,8192)). Emitter produces a 6-WG split
  ([Q+K MEM] / [V MEM] / [QK MMA] / [rescale] / [PV MMA] / [softmax]) that
  beats the hand-written TLX-WS reference on every shape (1.04× – 1.15×),
  peaking at **617 TFLOPS** on (1,32,8192).
- **case4** (Flash-Attention backward): gold-standard reference kernels in
  place (non-WS and TLX-WS); pre-modulo TTGIR + emitter generation is the
  next milestone.
- **case5** (persistent GEMM + 2D bias add, `addmm(a, b) + bias`):
  pre-modulo TTGIR + dumped `schedule_graph.json` in place; non-WS and
  TLX-WS reference kernels pass all 6 shapes (256³ → 8192³ + 1024×1024×16384).
  Emitter wiring partially works (GEMM body emits cleanly) but emits a
  `<tma_load_inline_unsupported>` placeholder for the bias TMA load in the
  outer-loop epilogue; closing this gap is the next emitter milestone.

See `emitter_design.md` for full implementation status, including which
auto-WS algorithm pieces are ported, what's deferred, and known issues.

## Quick start

```bash
# 1. Get a schedule_graph.json from modulo (using fbsource beta triton-opt):
TRITON_DUMP_MODULO_SCHEDULE=/tmp/graph.json TRITON_MODULO_DUMP_AND_EXIT=1 \
  buck2 run fbsource//third-party/triton/beta/triton:triton-opt -- \
  --nvgpu-modulo-schedule path/to/pre_modulo.ttgir

# 2. Emit TLX:
python -m sched2tlx /tmp/graph.json -o generated.py

# 3. Run it (with whatever TLX-supporting Triton you have installed):
python run_generated.py    # see examples/<case>/ for runner templates
```

## Principles

1. Pure mechanical lowering. No optimization, no heuristics, no pattern
   recognition.
2. Trust modulo's authoritative values. The JSON is the contract.
3. Compiler-agnostic output.
4. One algorithm for all kernels — single GEMM, persistent GEMM, FA — they
   differ only in graph shape.
5. Comments trace each emitted construct back to its `schedule_graph` source.
