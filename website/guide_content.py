"""Curated TLX guide content that supplements the repository README."""

GUIDE_CONTENT = {
    "home":
    r"""
# Overview

This site covers four connected parts of Meta's work: the Triton compiler, the TLX low-level programming model, the TorchTLX path that brings TLX kernels into PyTorch 2, and the tools used to understand and improve generated GPU kernels.

## News

[Aug'26] GPU MODE competition wins:

- Eigh: #5
- Cholesky: #3
- QR decomposition: #7
- Trimul competition: #1

[Jul'26] TritonParse accepted to ASPLOS 2026

## Explore the project

| Area | Focus |
|---|---|
| [Triton](website/triton.html) | Compiler-managed performance portability, including automatic warp specialization and its roadmap |
| [TLX](website/tlx.html) | Explicit warp-group orchestration, asynchronous memory and tensor-core operations, barriers, and hardware-specific kernel development |
| [uTLX](https://pypi.org/project/triton-utlx/) | The same TLX programming model packaged as a standalone plugin for upstream Triton, for users who are not on this fork |
| [TorchTLX](website/torchtlx.html) | TLX-backed Inductor templates, epilogue fusion, and the PyTorch 2 integration path |
| [CI](website/ci.html) | Workflows, runners, nightly failure handling, and per-project test coverage |
| [Tooling](website/tooling.html) | Compilation tracing, profiling, runtime diagnostics, sanitizers, and benchmarking |

## Install

FBTriton installs as the `triton` package, so `import triton` resolves to this fork. Uninstall upstream Triton first: both distributions own the same `triton/` directory, and installing one over the other leaves an inconsistent environment rather than a clean error.

```bash
pip uninstall -y triton
pip install fbtriton
```

Nightly `.dev` wheels are published to a self-managed index, not PyPI:

```bash
pip install --pre fbtriton \
  --index-url https://facebookexperimental.github.io/triton/nightly/simple/
```

Each nightly is built from the newest `main` commit whose GPU and CI checks are all green, and reports a version of the form `3.8.0.dev<YYYYMMDD>+fb.git<hash>`. Nightlies are retained for about 30 days. Binary wheels cover CPython 3.10 through 3.14.

FBTriton is intended as a drop-in replacement for upstream Triton on a best-effort basis: it tracks upstream closely, and existing Triton kernels are expected to work unchanged. That is not a formal guarantee — if something that works upstream breaks here, please file an issue.

### Build from source

```bash
git clone https://github.com/facebookexperimental/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

C++ changes require a rebuild to take effect; Python-only changes do not. Run `pre-commit run --all` before sending a pull request.

### uTLX: standalone TLX for upstream Triton

TLX is also published on its own as [`triton-utlx`](https://pypi.org/project/triton-utlx/), which ships the same `tlx` module as a Triton plugin rather than as a fork of Triton. Reach for it when replacing your `triton` install with FBTriton is not an option — the programming model is the one documented under [TLX](website/tlx.html).

```bash
pip install torch
pip install triton-utlx
```

Plugins load only into a Triton built with `TRITON_EXT_ENABLED`, which exposes the `libtriton` symbols they link against. The `triton` that ships with a PyTorch release has it on by default, so `torch` is enough. To run against a Triton you build yourself instead, turn it on at build time from a checkout of [triton-lang/triton](https://github.com/triton-lang/triton):

```bash
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
```

Either way, point `TRITON_PLUGIN_PATHS` — a colon-separated list of shared libraries — at the `libutlx.so` inside the installed package:

```bash
export TRITON_PLUGIN_PATHS=$(python -c \
  "import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), 'libutlx.so'))")
```

Nothing sets that variable for you, and a Triton built without `TRITON_EXT_ENABLED` warns and skips the plugin rather than failing outright, so an unset or ignored path shows up as unregistered TLX ops rather than a load error.

## Hardware support

| Vendor | Targets |
|---|---|
| NVIDIA | Hopper (`sm90`), Blackwell (`sm100`) |
| AMD | MI300 / CDNA3 (`gfx942`), MI350 / CDNA4 (`gfx950`), RDNA4 (`gfx1250`) |

Support varies per feature. Every operation in the [TLX reference](website/tlx.html) carries a tag naming the targets it runs on, and AutoWS requires `sm90` or newer.

### Check the install

```bash
python -c "import triton; print(triton.__version__)"
pytest third_party/tlx/tutorials/testing/test_correctness.py
```

The tutorial suite is arch-gated, so cases that do not apply to your GPU skip rather than fail.
""",
    "triton":
    r"""
# Triton

Triton is a compiler and language for writing performance-portable AI kernels with a productive blocked programming model. This FBTriton repository is Meta's development branch, where compiler-managed features such as automatic warp specialization are developed alongside lower-level TLX kernels that provide an expert-written performance reference.

The central goal is to keep ordinary Triton kernels productive while allowing the compiler to discover schedules that use modern GPU pipelines efficiently.

## Automatic warp specialization

Warp specialization assigns different groups of warps to different roles inside one kernel. A data partition can issue asynchronous loads, compute partitions can drive tensor cores, and epilogue or correction partitions can run concurrently when dependencies allow.

Specialization helps by:

- Reducing control-flow divergence between unrelated roles.
- Hiding memory and instruction latency through concurrent execution.
- Keeping independent hardware units busy at the same time.
- Allowing different tasks to receive different register and scheduling resources.

Triton's AutoWS pipeline makes these decisions at JIT compilation time rather than requiring every partition and barrier to be handwritten.

## The compiler pipeline today

The current design separates decision-making passes from the transformations that materialize those decisions:

| Stage | Responsibility |
|---|---|
| Data partitioning | Expose independent GEMMs or operation chains that can execute concurrently |
| Loop scheduler | Build a software-pipeline schedule and attach scheduling decisions to the IR |
| Partition scheduler | Assign operations to data, compute, correction, reduction, or epilogue warp partitions |
| Buffer creation | Identify cross-partition values and choose SMEM or TMEM communication buffers |
| Memory planner | Choose buffer depth and reuse based on liveness, dependencies, and hardware limits |
| Code partitioner | Create channels, barriers, buffer rotation, and physically outlined warp-group regions |

### Partition scheduling

The partition scheduler propagates task decisions as operation attributes. Current strategies recognize roles such as TMA data movement, tensor-core computation, softmax correction, reductions, and epilogue stores. Once operations are assigned, cross-partition SSA dependencies become explicit communication channels.

### Software pipelining

The loop scheduler reorders independent chains from adjacent iterations to maximize producer-consumer distance. In attention, for example, a dot product from one iteration can overlap softmax and value accumulation from another. Data partitioning gives the scheduler more independent work to interleave.

### Channels and memory planning

A channel represents both the storage and synchronization needed to move a value between partitions. The memory planner decides:

- Whether the channel belongs in shared memory or Blackwell tensor memory.
- How many cyclic buffer copies are required to cover pipeline distance.
- Which buffers can safely reuse storage when live ranges do not overlap.
- How to stay within register, SMEM, and TMEM capacity.

### Code partitioning

The code partitioner turns abstract channels into executable regions. It derives rotating buffer indices and barrier phases from loop iteration counts, reuses hardware completion barriers where possible, and performs local instruction ordering to reduce live ranges and register pressure.

## TLX and AutoWS

TLX and AutoWS solve related problems at different levels:

| TLX | Triton AutoWS |
|---|---|
| Kernel author explicitly defines warp-group tasks and synchronization | Compiler derives partitions, communication, and synchronization from Triton IR |
| Best for expert control, new hardware mechanisms, and hand-tuned reference kernels | Best for applying warp specialization broadly without rewriting kernels in a lower-level DSL |
| Exposes the intended pipeline directly in Python | Keeps the source close to ordinary Triton and specializes during compilation |

The [TLX documentation](tlx.html) covers the explicit programming model used to validate and refine many of these compiler strategies.

## Roadmap

### Profile-guided partition scheduling

Near-term work uses measured operation and region costs to rank partition choices. Communication overhead between partitions must be weighed against the concurrency gained by separating them.

### Model-based global optimization

Scheduling, partitioning, synchronization, layouts, and memory planning form a joint combinatorial search problem. The roadmap calls for hardware cost models—specified statically or learned from profiling—to guide a global planner that can choose software-pipeline schedules, channel depths, buffer reuse, partitions, and memory placement together.

### Kernel fusion and megakernels

IR-level fusion could combine handwritten and generated Triton kernels at tile granularity. The long-term goal is to optimize dependent and independent kernels together, reducing launch overhead and global-memory traffic while retaining provable correctness for algorithmic transformations.

### Broader kernel and hardware coverage

Current AutoWS focuses on Hopper and Blackwell. Continued work aims to generalize beyond attention-style independent dot chains, reduce target-specific heuristics, and make measured hardware models the main ingredient needed to support new architectures.

## Read the design article

This page is a condensed guide to the published design. See [Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) for the full discussion.
""",
    "tooling":
    r"""
# Tooling

Triton and TLX development use a small set of complementary tools. Start from the question you need to answer rather than collecting every trace at once.

| Tool | Best used for |
|---|---|
| TritonParse | Capturing compilation and launch events, extracting reproductions, comparing IR, and bisecting compiler regressions |
| Triton-MPP | Classifying compute, memory, occupancy, and latency bottlenecks with targeted profiling passes |
| CUTracer | Inspecting SASS-level execution, warp progress, TMA/MMA activity, deadlocks, and race candidates |
| Compute Sanitizer | Runtime checks for invalid memory access, uninitialized values, synchronization errors, and shared-memory hazards |
| TritonBench | Reproducible kernel and workload benchmarking across implementations and shapes |
| Proton | Instrumenting Triton kernels and collecting performance data with low-level execution context |
| triton-lint | Static checks for common Triton correctness and performance anti-patterns |

## Suggested workflow

- Create a stable single-kernel reproduction with fixed shapes and configuration.
- Use TritonParse to preserve compilation inputs and compare good and bad builds.
- Use Triton-MPP or a hardware profiler to classify the performance limit.
- Add CUTracer or Compute Sanitizer only when runtime ordering, races, or memory access are suspect.
- Confirm improvements with TritonBench or the repository's correctness and performance harnesses.

The [debugging guide](debugging.html) provides symptom-driven workflows for performance and numerical issues.
""",
}
