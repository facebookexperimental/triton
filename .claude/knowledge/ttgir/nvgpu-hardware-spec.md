# NVIDIA GPU Hardware Specifications

Key numbers from the CUDA Programming Guide (Release 13.2) relevant to
Triton compiler development. Focuses on Hopper (SM90) and Blackwell (SM100).

Source: CUDA Programming Guide, Tables 29-33, and architectural sections.

## Compute Capabilities

| Architecture | Compute Capability | Codename |
|---|---|---|
| Turing | 7.5 | SM75 |
| Ampere | 8.0, 8.6, 8.7 | SM80/86/87 |
| Ada Lovelace | 8.9 | SM89 |
| Hopper | 9.0 | SM90 |
| Blackwell | 10.0, 10.3 | SM100/103 |
| (unnamed) | 11.0 | SM110 |
| (unnamed) | 12.x, 12.1 | SM120/121 |

Family-specific targets: `compute_100f` covers SM100 + SM103;
`compute_110f` covers SM110; `compute_120f` covers SM120 + SM121.

## Thread / Block / Grid Limits

| Resource | All CCs |
|---|---|
| Warp size | 32 threads |
| Max threads per block | 1024 |
| Max block dimensions (x, y) | 1024 |
| Max block dimension (z) | 64 |
| Max grid dimension (x) | 2^31 - 1 |
| Max grid dimension (y, z) | 65535 |
| Grid dimensionality | 3 |
| Max resident grids per device | 128 |

## SM Occupancy Limits

| Resource | SM75 | SM80 | SM86 | SM87 | SM89 | SM90 | SM100 | SM103 | SM110 | SM120 |
|---|---|---|---|---|---|---|---|---|---|---|
| Max resident blocks/SM | 16 | 32 | 16 | 16 | 24 | 32 | 24 | 24 | 24 | 24 |
| Max resident warps/SM | 32 | 64 | 48 | 48 | 48 | 64 | 48 | 48 | 48 | 48 |
| Max resident threads/SM | 1024 | 2048 | 1536 | 1536 | 1536 | 2048 | 1536 | 1536 | 1536 | 1536 |

## Register File

| Resource | All CCs |
|---|---|
| 32-bit registers per SM | 64K (65536) |
| Max 32-bit registers per block | 64K (65536) |
| Max 32-bit registers per thread | 255 |

Register allocation is per-warp. Using fewer registers per thread allows more
warps to be resident, improving occupancy and latency hiding. Use `--maxrregcount`
or `__maxnreg__()` to cap register usage (may cause spilling to local memory).

## Shared Memory (SMEM)

| Resource | SM75 | SM80 | SM86/89 | SM87 | SM90 | SM100/103/110 | SM120 |
|---|---|---|---|---|---|---|---|
| Max SMEM per SM | 64 KB | 164 KB | 100 KB | 164 KB | 228 KB | 228 KB | 100 KB |
| Max SMEM per block | 64 KB | 163 KB | 99 KB | 163 KB | 227 KB | 227 KB | 99 KB |
| Shared memory banks | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

Kernels using >48 KB SMEM per block must use dynamic shared memory with
explicit opt-in via `cudaFuncSetAttribute`.

### Unified Data Cache Sizes and SMEM Carveout Options

| CC | Unified Cache | SMEM Capacity Options (KB) |
|---|---|---|
| 7.5 | 96 KB | 32, 64 |
| 8.0 | 192 KB | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.6, 8.9 | 128 KB | 0, 8, 16, 32, 64, 100 |
| 8.7 | 192 KB | 0, 8, 16, 32, 64, 100, 132, 164 |
| 9.0, 10.x, 11.0 | 256 KB | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 12.x | 128 KB | 0, 8, 16, 32, 64, 100 |

SMEM and L1 cache share the same physical resource (unified data cache).
More SMEM = less L1 cache. Configurable via `cudaFuncSetAttribute` with
`cudaFuncAttributePreferredSharedMemoryCarveout`.

### Bank Conflicts

- 32 banks, each 4 bytes wide
- Successive 32-bit words map to successive banks
- Conflict: multiple threads in a warp access different words in the same bank
- No conflict: all threads access different banks, or all access the same word (broadcast)
- Common fix: pad shared memory arrays by +1 column (e.g., `float smem[32][33]`)

## Other Memory

| Resource | All CCs |
|---|---|
| Max local memory per thread | 512 KB |
| Constant memory size | 64 KB |
| Constant cache per SM | 8 KB |
| Texture cache per SM | 28-256 KB (varies) |

## Thread Block Clusters (SM90+)

- Available from compute capability 9.0
- Max cluster size: **8 thread blocks** (may be lower on GPUs with <8 SMs)
- Query actual max: `cudaOccupancyMaxPotentialClusterSize`
- Enables **Distributed Shared Memory (DSMEM)**: threads can access SMEM of
  other blocks in the cluster
- Total DSMEM = cluster_size x SMEM_per_block

## Warp Groups (SM90+ PTX concept)

- A warp group = 4 consecutive warps = 128 threads
- Used by `wgmma` (warp group MMA) instructions on Hopper
- Not a CUDA C++ concept; exposed through PTX and Triton's TTGIR

## Asynchronous Barriers (mbarriers)

- Allocated in shared memory, 8 bytes each
- Hardware-accelerated from SM80+
- Split arrive/wait model with phase tracking (ping-pong parity)
- Can track both arrival counts and byte counts (for TMA/tcgen05)
- Cluster-scope barriers (SM90+): arrive from remote CTA, wait locally only
- Max arrival count: `__mbarrier_maximum_count()` (hardware-defined)

### Barrier Scopes

| Scope | Memory Location | Arrive | Wait | HW Accel | Min CC |
|---|---|---|---|---|---|
| Block | Shared memory | Yes | Yes | Yes | 8.0 |
| Cluster (local) | Shared memory | Yes | Yes | Yes | 9.0 |
| Cluster (remote) | Shared memory | Yes | No | Yes | 9.0 |
| Device | Global memory | Yes | Yes | No | 7.0 |
| System | Global/unified | Yes | Yes | No | 7.0 |

## Named Barriers (Hardware Barrier Indices)

- Use hardware barrier registers, indices 0-15 (16 barriers total)
- No SMEM allocation needed
- Used in Triton for warp-level synchronization (e.g., ping-pong scheduling
  in warp specialization)
- Lighter weight than mbarriers for intra-CTA synchronization

## Tensor Memory Accelerator (TMA) — SM90+

- Hardware unit for async bulk copies between global and shared memory
- Supports 1D to 5D tensor transfers
- Uses **tensor map** (tensor descriptor) to describe global memory layout
- Tensor map encodes: base address, dimensions, strides, element type, swizzle mode
- Supports multicast to multiple CTAs in a cluster
- Completion tracked via mbarrier

### TMA Swizzle Patterns (SM90)

| Pattern | Swizzle Width | Max Inner Dim | Repeats After | Alignment |
|---|---|---|---|---|
| 128B | 128 bytes | 128 bytes | 1024 bytes | 128 bytes |
| 64B | 64 bytes | 64 bytes | 512 bytes | 128 bytes |
| 32B | 32 bytes | 32 bytes | 256 bytes | 128 bytes |
| None | - | - | - | 16 bytes |

## Async Copy Mechanisms

| Mechanism | Direction | Min CC | Granularity |
|---|---|---|---|
| LDGSTS (`cp.async`) | Global → SMEM | 8.0 | 4, 8, or 16 bytes per thread |
| TMA (bulk tensor) | Global ↔ SMEM | 9.0 | Bulk tile (up to 5D) |
| STAS (`st.async`) | Registers → DSMEM | 9.0 | 4, 8, or 16 bytes |

### Proxy Fence Requirements

TMA and tcgen05 operations use the **async proxy**. A proxy fence
(`fence.proxy.async`) is required between generic-proxy writes (e.g.,
`local_store` to SMEM) and async-proxy reads (e.g., TMA load from SMEM,
wgmma reading SMEM operand). Without the fence, the async engine may
read stale data.

## Tensor Core Data Type Support

| CC | FP64 | TF32 | BF16 | FP16 | FP8 | FP6 | FP4 | INT8 | INT4 |
|---|---|---|---|---|---|---|---|---|---|
| 7.5 | | | | Yes | | | | Yes | Yes |
| 8.0 | Yes | Yes | Yes | Yes | | | | Yes | Yes |
| 8.6-8.7 | | Yes | Yes | Yes | | | | Yes | Yes |
| 8.9 | | Yes | Yes | Yes | Yes | | | Yes | Yes |
| 9.0 | Yes | Yes | Yes | Yes | Yes | | | Yes | |
| 10.0 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | |
| 10.3-12.x | | Yes | Yes | Yes | Yes | Yes | Yes | Yes | |

## Tensor Memory (TMEM) — SM100+ (Blackwell)

- Dedicated on-chip memory for MMA accumulators and scale factors
- 512 rows, column width depends on encoding
- Not directly addressable by normal load/store; accessed via `tcgen05` instructions
- Async copy from SMEM via `tcgen05.cp`
- MMA result written directly to TMEM (not registers like Hopper wgmma)

## Key Architectural Differences: Hopper vs Blackwell

| Feature | Hopper (SM90) | Blackwell (SM100+) |
|---|---|---|
| MMA instruction | `wgmma` (warp group) | `tcgen05.mma` |
| MMA accumulator | Registers | TMEM |
| MMA operand A | SMEM or Registers | SMEM |
| MMA operand B | SMEM | SMEM |
| MMA completion | `wgmma.wait_group` | mbarrier (via `tc_gen5_commit`) |
| Cluster Launch Control | No | Yes (work stealing) |
| Max SMEM/SM | 228 KB | 228 KB |
| Narrow type support | FP8, INT8 | FP4, FP6, FP8, INT8 |
| 2-CTA MMA | No | Yes |

## Thread Scope Coherency Points

| CUDA Scope | PTX Scope | Coherency Point |
|---|---|---|
| `thread_scope_block` | `.cta` | L1 |
| (cluster) | `.cluster` | L2 |
| `thread_scope_device` | `.gpu` | L2 |
| `thread_scope_system` | `.sys` | L2 + connected caches |

## Memory Hierarchy (Relative Ordering)

From fastest to slowest access:
1. **Registers** — per-thread, compiler-managed
2. **SMEM** — per-CTA, on-chip, same physical resource as L1
3. **TMEM** — per-CTA (Blackwell only), on-chip, accessed via tcgen05
4. **L1 cache** — per-SM, shares physical space with SMEM
5. **L2 cache** — per-GPU, shared across all SMs
6. **HBM (Global)** — off-chip DRAM

Note: Specific bandwidth/latency numbers vary by GPU SKU and are not
covered in the CUDA Programming Guide. Consult product datasheets.
