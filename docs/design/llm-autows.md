# LLM-Assisted AutoWS: DDG-to-TLX Code Generation

## Motivation

Triton provides a high-level DSL for writing GPU kernels. TLX extends it
with low-level, hardware-near primitives — explicit barrier protocols,
warp specialization, multibuffered TMA pipelines, TMEM management. Today,
going from a Triton kernel to an optimized TLX kernel requires either:

1. **Manual rewriting** by an expert who understands the hardware, or
2. **AutoWS** (the compiler's modulo scheduling pass), which produces
   optimized IR but not human-readable TLX source code.

The problem with (2) is that the output is compiler IR — opaque, not
user-editable, and hard to debug when something goes wrong. When a kernel
hangs, hits an SMEM overflow, or underperforms, the user has no artifact
they can inspect and fix incrementally.

**The goal: use an LLM to produce TLX Python source code from a Triton
kernel.** The output is a `.py` file that the user owns — readable,
editable, and incrementally fixable. If the generated kernel has a bug,
the user (or the LLM) can patch the TLX code directly, without touching
compiler internals.

## Input / Output

### Input: DDG + Latency Table

The Data Dependence Graph (DDG) and hardware latency table, built by the
Triton compiler in C++ using the existing `DataDependenceGraph` and
`LatencyModel` infrastructure. The LLM receives a structured summary —
not raw TTGIR, not Python source:

```
DDG for loop (trip_count=32):
  ResMII=2, RecMII=1005, MinII=1005

Nodes:
  N0: arith.muli        pipe=NONE  lat=0    selfLat=0
  N1: tt.descriptor_load pipe=MEM   lat=1218 selfLat=1
  N2: tt.descriptor_load pipe=MEM   lat=1218 selfLat=1
  N3: ttg.local_alloc    pipe=MEM   lat=700  selfLat=0
  N4: ttg.local_alloc    pipe=MEM   lat=700  selfLat=0
  N5: ttng.tmem_alloc    pipe=NONE  lat=0    selfLat=0
  N6: ttng.tc_gen5_mma   pipe=TC    lat=900  selfLat=1
  N7: ttng.tmem_load     pipe=CUDA  lat=105  selfLat=1

Edges:
  N0 -> N1  lat=0    dist=0
  N1 -> N3  lat=518  dist=0
  N3 -> N6  lat=700  dist=0
  N6 -> N7  lat=900  dist=0
  N7 -> N5  lat=105  dist=1    # loop-carried
  ...
```

### Output

**TLX Python source code** — a complete, runnable `@triton.jit` kernel
with warp specialization, explicit barriers, multibuffered TMA loads,
and hardware-optimal scheduling. The user can:

- Read and understand the pipeline structure
- Run correctness tests directly
- Edit specific sections (e.g., change buffer depth, adjust barrier protocol)
- Incrementally fix issues without regenerating from scratch

This is fundamentally different from compiler-internal IR annotations
(`tt.autows`, `tt.num_stages`) which are invisible to the user and
cannot be edited.

## Why Inside Triton

The LLM codegen is built into Triton's compiler pipeline (not an external
tool) because:

1. **Triton has the DDG and latency model** — the compiler already builds
   the data dependence graph and computes cycle-accurate hardware
   latencies. The LLM receives these as structured input, not raw MLIR.

2. **Triton has the IR** — the compiler's intermediate representation
   captures precise semantics (tensor shapes, memory hierarchy, pipeline
   classification) that would be lost if the LLM only saw Python source.

3. **Triton validates the output** — the generated TLX code compiles
   through the same Triton pipeline, so correctness tests, SMEM budget
   checks, and barrier analysis all apply automatically.

## Architecture

```
Triton compiler (TTIR → TTGIR)
    │
    ▼
DDG + Latency Table (built in C++)
    │
    ├── Nodes: op name, pipeline, latency, selfLatency
    ├── Edges: src → dst, latency, loop-carried distance
    ├── MinII, ResMII, RecMII (pre-computed)
    │
    ▼
LLM (Claude, local CLI)
    │
    ├── Step 1: Schedule graph (II, stages, cycles, buffers)
    ├── Step 2: TLX Python source code
    │
    ▼
TLX kernel (.py) ← user-owned output
    │
    ├── User reviews, edits, fixes
    ├── Run correctness tests
    └── Iterate
```

### Two-step generation

**Step 1: Schedule graph** — the LLM assigns each op to a cycle, stage,
and cluster, determines buffer depths, and pairs barriers. This is the
"thinking" step that requires understanding hardware latencies and
pipeline overlapping.

**Step 2: TLX code emission** — the schedule graph is translated into
TLX Python with `async_tasks`, `local_alloc`, `alloc_barriers`,
`async_descriptor_load`, `async_dot`, barrier wait/arrive protocols,
and the pipeline loop structure.

Step 1 is already working (the `nvgpu-llm-schedule` pass). Step 2 can
be done by the LLM in the same call, or as a separate pass that takes
the schedule graph and emits TLX.

## Current Implementation

### C++ pass: `nvgpu-llm-schedule`

Registered as `nvidia.passes.hopper.add_llm_schedule(pm)`, gated by
`TRITON_USE_LLM_SCHEDULE=1`.

| File | Purpose |
|------|---------|
| `third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/LLMSchedulePass.cpp` | C++ pass: builds DDG, calls Claude CLI, parses schedule, sets IR attrs |
| `third_party/nvidia/backend/compiler.py` | Pipeline integration |
| `python/triton/knobs.py` | `TRITON_USE_LLM_SCHEDULE` env var |
| `third_party/tlx/tools/scheduling_prompt.md` | System prompt with scheduling rules |
| `test/TritonGPU/llm-schedule.mlir` | LIT test |

### How it works

1. Compiler builds DDG using `DataDependenceGraph::build()` and
   `LatencyModel` (same infrastructure as modulo scheduling)
2. Formats the DDG as structured text (nodes + edges + latencies)
3. Calls `claude --bare --system-prompt-file ... -p ...` via `std::system()`
4. Parses the `modulo.schedule` response
5. Sets `tt.autows` on MMA ops and `tt.num_stages` on loops

### Example result

For a standard GEMM kernel, the LLM produced:

```
modulo.schedule @loop0 {
  ii = 1005, max_stage = 2, prologue_latency = 0, trip_count = 32

  modulo.stage @s0 {
    N1  {pipe: MEM, cycle: 0, cluster: 0, latency: 1218, selfLatency: 1}
    N2  {pipe: MEM, cycle: 1, cluster: 1, latency: 1218, selfLatency: 1}
    N3  {pipe: MEM, cycle: 518, cluster: 2, latency: 700, selfLatency: 0}
    N4  {pipe: MEM, cycle: 519, cluster: 3, latency: 700, selfLatency: 0}
  }
  modulo.stage @s1 {
    N6  {pipe: TC, cycle: 1219, cluster: 0, latency: 900, selfLatency: 1}
  }
  modulo.stage @s2 {
    N7  {pipe: CUDA, cycle: 2119, cluster: 0, latency: 105, selfLatency: 1}
  }
}
```

This corresponds to a 3-stage pipeline (II=1005, triple-buffered SMEM)
which is actually better than the modulo scheduler's output (II=1601,
single-buffered) for this kernel — the LLM kept the lower II and used
more buffers, which is the right trade-off when SMEM budget allows it.

## Phased Rollout

### Phase 1: Schedule graph generation (done)

The LLM receives the DDG and produces a `modulo.schedule` graph.
Verified on the GEMM LIT test — correct II, stages, and cycle placement.

### Phase 2: TLX code emission

The LLM receives the DDG + schedule graph and produces complete TLX
Python source code. The system prompt includes:
- TLX API reference (barriers, memory ops, MMA, warp specialization)
- 1-2 tutorial kernel examples matched by pattern (GEMM, FA)
- The schedule graph from Phase 1 as the "plan"

### Phase 3: Complex kernels (FA, persistent, 2-CTA)

Extend to Flash Attention, persistent kernels with CLC, and multi-CTA
patterns. These require:
- Multiple MMAs with different dependency depths
- Pingpong scheduling
- Cross-CTA synchronization via DSMEM
- TMEM buffer reuse (storage_alias_spec)

### Phase 4: Interactive refinement

The user runs the generated TLX kernel, sees results, and asks the LLM
to adjust:
- "Add one more pipeline stage"
- "Move the softmax chain to the consumer task"
- "Switch from 1-CTA to 2-CTA"
- "Fix the barrier deadlock on line 42"

The LLM modifies the TLX source directly. This is where producing
user-owned TLX code (vs compiler IR) pays off — the user and LLM
can iterate on the same readable artifact.

## Success Criteria

1. Generated TLX kernel passes numerical correctness tests
2. Performance within 80% of hand-tuned TLX for GEMM and FA fwd
3. User can read, understand, and modify the generated code
4. End-to-end generation takes < 60 seconds
5. When the kernel has a bug, the user can fix it in the TLX source
   without regenerating from scratch
