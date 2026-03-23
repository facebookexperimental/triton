# NVIDIA GPU Memory Hierarchy

Reference: CUDA Programming Guide, Release 13.2, Sections 1.2.2–1.2.3, 2.2.3,
3.2.2–3.2.6, Tables 30–32.

## Overview

An NVIDIA GPU is organized as a set of **Streaming Multiprocessors (SMs)**
grouped into **Graphics Processing Clusters (GPCs)**. The memory hierarchy
spans two levels: memory private to each SM (intra-SM) and memory shared
across all SMs (across-SM).

### Across-SM Memory

- **Global Memory (HBM/DRAM)**: Device-attached DRAM, accessible by all SMs
  and all CTAs. Highest capacity, highest latency. Capacity and bandwidth
  vary by GPU product. All persistent kernel data lives here.
  **User-managed**: allocated/freed via CUDA APIs (`cudaMalloc`/`cudaFree`),
  read/written explicitly by kernel code.
- **L2 Cache**: Shared across all SMs. Caches global memory accesses. Can
  reserve a portion for persisting accesses (`cudaLimitPersistingL2CacheSize`).
  Coherency point for device-scope and cluster-scope operations.
  **Hardware-managed / transparent**: automatically caches global and local
  memory accesses. Users can influence behavior via access policy hints but
  do not directly allocate or address L2.
- **Constant Memory**: 64 KB read-only region in global memory, cached per-SM
  (8 KB constant cache).
  **User-declared, compiler-assisted**: declared by the user with
  `__constant__` and initialized from host code. The compiler may also
  place kernel parameters here automatically.
- **Local Memory**: Per-thread, but physically resides in global memory.
  The "local" refers to its logical scope, not physical location. Used for
  register spills, large arrays with non-constant indices, and large structs.
  Max 512 KB per thread. Cached in L1/L2. Accessed with coalesced patterns
  (consecutive 32-bit words by consecutive thread IDs).
  **Compiler-managed / transparent**: the compiler decides what spills to
  local memory. Users do not explicitly allocate or address it, though they
  can influence spilling via `--maxrregcount` or `__maxnreg__()`.

### Intra-SM Memory

Each SM contains a **unified data cache** that is carved into L1 cache and
shared memory at runtime. The carveout is configurable per kernel via
`cudaFuncSetAttribute`. See `nvgpu-hardware-spec.md` for capacity options
per compute capability.

- **Registers (RF)**: Per-thread. 64K 32-bit registers per SM, max 255 per
  thread. Fastest access. When a kernel exceeds register capacity, the
  compiler spills to local memory (see above).
  **Compiler-managed / transparent**: register allocation is handled by the
  compiler. Users can cap usage with `--maxrregcount` or `__maxnreg__()`.
- **L1 Cache**: Per-SM, part of the unified data cache.
  **Hardware-managed / transparent**: automatically caches global and local
  memory accesses. Users can configure the L1/SMEM carveout ratio but do
  not directly address L1.
- **Shared Memory (SMEM)**: Per-SM, part of the unified data cache.
  Accessible by all threads in a thread block (and by threads in the same
  cluster via Distributed Shared Memory on SM90+). 32 banks, each 4 bytes
  wide. Max 228 KB per SM / 227 KB per block on SM90/SM100. Also hosts
  mbarrier objects (8 bytes each).
  **User-managed**: explicitly allocated (`__shared__` or dynamic SMEM),
  read/written by kernel code. The user controls data placement and must
  handle synchronization between threads.
- **Tensor Memory (TMEM)**: Per-SM, Blackwell-only (SM100+). Dedicated on-chip
  memory for MMA accumulators and block scale factors. Not accessible via
  normal load/store — only through `tcgen05` instructions.
  **User-managed (via intrinsics)**: allocated and accessed through
  specialized `tcgen05` instructions (e.g., `tmem_alloc`, `tmem_copy`,
  `tc_gen5_mma`). Not addressable by normal ld/st. In Triton, the compiler
  handles TMEM allocation, but the user-facing kernel controls data flow
  through TLX/TTGIR ops.

```
Across-SM                              Intra-SM (one SM)
┌─────────────────────┐    ┌─────────────────────────────────────────┐
│  Global Memory (HBM)│    │  Register File (64K x 32-bit)           │
│  accessible by      │    │  per-thread, compiler-managed           │
│  all SMs / all CTAs │    ├─────────────────────────────────────────┤
└────────┬────────────┘    │  Unified Data Cache (96-256 KB)         │
         │                 │  ┌──────────────┬───────────────────┐   │
         ▼                 │  │  L1 Cache    │  Shared Memory    │   │
┌─────────────────────┐    │  │  (automatic) │  (programmable)   │   │
│     L2 Cache        │    │  │              │  up to 228 KB/SM  │   │
│     shared across   │◄──►│  └──────────────┴───────────────────┘   │
│     all SMs         │    │         ▲                               │
└─────────────────────┘    │         │ cluster addressing (SM90+)    │
                           │         ▼                               │
Across-SM (within GPC)     │  ┌───────────────────────────────┐      │
┌─────────────────────┐    │  │ Distributed Shared Memory     │      │
│  DSMEM: other CTAs' │◄──►│  │ (DSMEM, up to 8 CTAs/cluster) │      │
│  SMEM in cluster    │    │  └───────────────────────────────┘      │
└─────────────────────┘    ├─────────────────────────────────────────┤
                           │  Tensor Memory (TMEM) — SM100+ only     │
                           │  MMA accumulators, tcgen05 access only  │
                           └─────────────────────────────────────────┘
```

## Memory Spaces in Triton MLIR

Triton models three explicit memory space **resources** in its TableGen-based
MLIR dialect definitions (used for memory effect tracking on ops):

| Resource | MLIR Resource String | Defined In |
|---|---|---|
| `GlobalMemory` | `::mlir::triton::GlobalMemory` | `TritonOps.td`, `TritonGPUOps.td`, `TritonNvidiaGPUOps.td` |
| `SharedMemory` | `::mlir::triton::gpu::SharedMemory` | `TritonGPUOps.td`, `TritonNvidiaGPUOps.td` |
| `TensorMemory` | `::mlir::triton::nvidia_gpu::TensorMemory` | `TritonNvidiaGPUOps.td` only |

The `MemDescType` carries a `memorySpace` attribute to distinguish SMEM from
TMEM descriptors:
- `SharedMemorySpaceAttr` (defined in `TritonGPUAttrDefs.td`)
- `TensorMemorySpaceAttr` (defined in `TritonNvidiaGPUAttrDefs.td`)

Registers are not modeled as a memory space — they are the default home for
distributed tensor values (`RankedTensorType` with an encoding attribute).

## Hopper (SM90, Compute Capability 9.0)

Hopper introduced Thread Block Clusters, TMA, and warp group MMA (`wgmma`).

**Memory features:**
- Unified data cache: 256 KB per SM, carveout up to 228 KB SMEM
- Registers hold MMA accumulators (wgmma writes results to registers)
- No Tensor Memory (TMEM)
- TMA for bulk async copies between global memory and SMEM (1D–5D tensors)
- Distributed Shared Memory (DSMEM): threads in a cluster can access SMEM of
  other CTAs via cluster addressing
- Cluster size: up to 8 CTAs per cluster
- Hardware-accelerated mbarriers in SMEM (block and cluster scope)
- STAS (`st.async`): async register → remote SMEM within a cluster

**MMA data flow:**
```
Global ──TMA──► SMEM ──local_load──► Registers (dot operand layout)
                 │                         │
                 └── wgmma reads A,B ──────┘──► Registers (accumulator)
```
- Operand A: SMEM or registers
- Operand B: always SMEM
- Accumulator (C/D): registers
- Completion: `wgmma.wait_group` (pendings-based)

**Proxy model:** TMA and wgmma operate via the **async proxy**. A
`fence.proxy.async` is required between generic-proxy writes (e.g.,
`local_store` to SMEM) and async-proxy reads (e.g., wgmma reading SMEM).

## Blackwell (SM100, Compute Capability 10.0)

Blackwell adds Tensor Memory and `tcgen05` MMA, plus Cluster Launch Control
for persistent kernels with work stealing.

**Memory features (same as Hopper plus):**
- Unified data cache: 256 KB per SM, carveout up to 228 KB SMEM (same as Hopper)
- **Tensor Memory (TMEM)**: dedicated on-chip memory per SM for MMA accumulators
  and block scale factors. Accessed only via `tcgen05` instructions (`tcgen05.cp`,
  `tcgen05.mma`). Not addressable by normal ld/st.
- TMA with all Hopper features
- Cluster Launch Control (CLC): a CTA can cancel a pending cluster launch and
  steal its work index, enabling dynamic persistent kernels
- Supports 2-CTA MMA: distributed matmul across two CTAs in a cluster

**MMA data flow:**
```
Global ──TMA──► SMEM ──tcgen05.mma──► TMEM (accumulator)
                 │                       │
                 └── reads A,B from SMEM │
                                    tmem_load
                                         │
                                         ▼
                                   Registers (result)
```
- Operand A: SMEM
- Operand B: SMEM
- Accumulator (D): **TMEM** (not registers)
- Completion: mbarrier-based (via `tc_gen5_commit` + `wait_barrier`)

**Scaled MMA (MX formats):**
```
Global ──TMA──► SMEM ─┬─ tcgen05.mma ──► TMEM (accumulator)
                       │
                       └─ tmem_copy ────► TMEM (scales)
```
Block scale factors are copied from SMEM to TMEM via `tcgen05.cp` and
consumed by `tc_gen5_mma_scaled`. Supports FP4, FP6, FP8 with per-block
scaling.

**Tensor core data type additions over Hopper:** FP4, FP6 (Hopper: none).
SM100 retains FP64 tensor core support; SM103 does not.

## Blackwell (SM103, Compute Capability 10.3)

SM103 is part of the same GPU family as SM100 (`compute_100f`). It shares
the Blackwell memory hierarchy and `tcgen05` instruction set with SM100.

**Differences from SM100:**
- No FP64 tensor core support
- Same SM occupancy limits (24 blocks, 48 warps, 1536 threads per SM)
- Same SMEM capacity (256 KB unified cache, up to 228 KB SMEM)
- Same TMEM and TMA features

The `compute_100f` family-specific compilation target covers both SM100 and
SM103. The `compute_100a` architecture-specific target is SM100-only.

## Cluster Memory (SM90+)

Thread Block Clusters group up to 8 CTAs that are co-scheduled on the same
GPC. Within a cluster, each CTA can access other CTAs' shared memory via
**Distributed Shared Memory (DSMEM)**. Total DSMEM = cluster_size × SMEM per
block.

TTGIR ops for cluster memory access:
- `ttg.remote_shmem_store` / `ttg.async_remote_shmem_store`: write to
  another CTA's SMEM
- `ttng.map_to_remote_buffer`: create a memdesc view of a remote CTA's
  SMEM buffer (pure, no data movement)
- TMA multicast: a single TMA load writes to multiple CTAs' SMEM
  simultaneously via a bitmask

Cluster-scoped mbarriers allow a CTA to arrive on a barrier in another CTA's
SMEM, but waiting is only supported on local SMEM barriers.
