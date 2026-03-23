# TTGIR Memory Layout Ops

Ops for creating views, transforming descriptors, and converting layouts.
These ops do not move data — they reinterpret how existing memory is addressed.

## Memory Descriptor Views

All view ops are `Pure` (no side effects) and carry the `MemDescViewTrait`.
They return a new `MemDescType` pointing to the same underlying memory.

| Op | What it does | Applies to |
|---|---|---|
| `ttg.memdesc_index` | Index dim 0, reduce rank by 1 (e.g., select pipeline stage) | SMEM |
| `ttg.memdesc_subslice` | Static-offset subview | SMEM |
| `ttg.memdesc_trans` | Transpose (permute dimensions) | SMEM |
| `ttg.memdesc_reshape` | Reshape (contiguous only) | SMEM |
| `ttg.memdesc_reinterpret` | Reinterpret shape + element type (bitcast) | SMEM |
| `ttng.tmem_subslice` | Subslice along inner (column) dim only | TMEM (SM100+) |

## Cluster Buffer Mapping

`ttng.map_to_remote_buffer`: Given a local SMEM memdesc, returns a view of
the corresponding buffer in another CTA within the cluster. Pure, no data
movement. Used with distributed algorithms and 2-CTA MMA.

## TMA Descriptor Ops

| Op | Purpose |
|---|---|
| `ttng.reinterpret_tensor_descriptor` | Cast raw `!tt.ptr<i8>` to typed `!tt.tensordesc`. Pure. |
| `ttng.tensormap_create` | Create TMA descriptor on device. Takes base address, box dims, global dims, strides, element type, swizzle mode. Has global memory effects. |

TMA descriptors (`!tt.tensordesc`) are consumed by all `async_tma_*` data
transfer ops. The swizzle mode (128B/64B/32B/None) must match the SMEM
layout encoding.

## Register Layout Conversion

`ttg.convert_layout`: Converts a distributed tensor between register layouts
(e.g., `#blocked` ↔ `#mma` ↔ `#dot_op`). Pure at TTGIR level but may lower
to SMEM-mediated shuffles. Same shape and element type, different encoding.
