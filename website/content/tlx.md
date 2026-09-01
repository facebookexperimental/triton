TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It offers intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, computation, and asynchronous control flow. TLX is designed for expert users pushing Triton closer to the metal.

Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:

- Hardware-specific intrinsics (e.g., wgmma, async_copy, barrier)
- Shared and local memory allocation
- Instruction-level scheduling and control
- Cross-warpgroup synchronization

While this approach places more responsibility on the user, it reduces the compiler's role as a performance bottleneck. Although it may introduce divergence across hardware platforms, it empowers users to perform deeper, architecture-specific optimizations without relying solely on compiler heuristics.

## Optimize TLX kernels

Use the [TLX Agent](tlx-agent.html) to run a harness-driven optimization loop with isolated candidates, correctness gates, Proton and hardware profiling, final revalidation, and an optional local winner commit. Running it through a coding agent is recommended because the coding agent can prepare and freeze the workload-specific harness before invoking the TLX Agent CLI.

## Hardware tags

Every operation below is tagged with the targets it runs on, using the
architecture ids from `triton._internal_testing` — the same vocabulary the rest
of this repo uses:

| Tag | Hardware |
|-----|----------|
| `sm90` | NVIDIA Hopper |
| `sm100` | NVIDIA Blackwell |
| `sm90+` | Hopper and newer NVIDIA |
| `gfx942` | AMD MI300 (CDNA3) |
| `gfx950` | AMD MI350 (CDNA4) |
| `gfx942+` | MI300 and newer AMD |
| `gfx1250` | AMD RDNA4 |
| `amd` | All AMD targets |

A trailing `?` (e.g. `gfx942+?`) marks availability that has **not been
confirmed yet** and needs verification. An absent vendor means this document
makes no claim for that vendor, not necessarily that the op is unsupported.

Note: async copies (`async_load` and its commit/wait groups) and the AMD buffer
ops require **gfx950** — they are **not** available on gfx942. `barrier_arrive`
on AMD requires `arrive_count == 1`.
