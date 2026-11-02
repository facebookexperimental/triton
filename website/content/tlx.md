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

Tags describe compiler support; per-hardware pages call out API restrictions
and distinguish source implementation from target-specific test coverage.
An omitted vendor makes no support claim, rather than implying unsupported hardware.

AMD buffer operations are available on gfx942, gfx950, and gfx1250, with
operation-specific dtype and copy-width restrictions. The explicit
`buffer_load_to_local` operation targets gfx942/gfx950; gfx1250 uses separate
TDM or pointer-based copy paths. See [global memory access](global-memory.html)
and [async memory access](async-memory.html). `barrier_arrive` on AMD requires
`arrive_count == 1`.

## Per-hardware support

- [TLX on gfx950 (MI350)](tlx-gfx950.html)
- [TLX on gfx942 (MI300)](tlx-gfx942.html)
- [TLX on sm100 (Blackwell)](tlx-sm100.html)
- [TLX on sm90 (Hopper)](tlx-sm90.html)
- [TLX on gfx1250 (RDNA4)](tlx-gfx1250.html)
