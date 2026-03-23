# TTGIR Tensor Core Ops

Matrix multiply-accumulate operations that execute on GPU tensor cores.

## Hopper (SM90): Warp Group MMA

**`ttng.warp_group_dot`** — Wgmma: `D = A * B + C`
- Operand A: SMEM memdesc or register tensor
- Operand B: SMEM memdesc (always)
- Accumulator C/D: register tensors
- Async mode (`isAsync=true`): result not immediately available

**`ttng.warp_group_dot_wait`** — Wait for async wgmma completion.
`pendings` specifies max outstanding ops allowed. Must pass in-flight
result tensors as `inputs` for dependency tracking.

## Blackwell (SM100): TCGen5 MMA

**`ttng.tc_gen5_mma`** — `D += A * B` on Blackwell tensor cores.
- Operand A: SMEM memdesc
- Operand B: SMEM memdesc
- Accumulator D: **TMEM** memdesc (read/written in-place)
- Async by default; completion tracked via mbarrier + `tc_gen5_commit`
- Supports 2-CTA mode (`two_ctas`) for distributed matmul
- `useD` controls accumulate vs overwrite

**`ttng.tc_gen5_mma_scaled`** — Scaled MMA with block scaling factors.
Same as `tc_gen5_mma` plus `a_scale`/`b_scale` descriptors (SMEM or TMEM)
and element type attributes (`lhs`/`rhs` — e.g., `e4m3`, `e2m1`).
Used for MX-format GEMM with FP4/FP6/FP8 narrow types.

## Architectural Comparison

| Aspect | Hopper (`warp_group_dot`, CC 9.0) | Blackwell (`tc_gen5_mma`, CC 10.0) |
|---|---|---|
| A operand | SMEM or Registers | SMEM |
| B operand | SMEM | SMEM |
| Accumulator | Registers | TMEM |
| Completion | `warp_group_dot_wait` (pendings) | mbarrier via `tc_gen5_commit` |
| Scaled MMA | N/A | `tc_gen5_mma_scaled` |
| 2-CTA mode | No | Yes |

## Memory Access Summary

| Op | Reads | Writes |
|---|---|---|
| `warp_group_dot` | A: SMEM or Reg, B: SMEM, C: Reg | D: Reg |
| `warp_group_dot_wait` | (sync only) | (sync only) |
| `tc_gen5_mma` | A: SMEM, B: SMEM, D: TMEM (if useD) | D: TMEM |
| `tc_gen5_mma_scaled` | A: SMEM, B: SMEM, scales: SMEM/TMEM, D: TMEM | D: TMEM |
