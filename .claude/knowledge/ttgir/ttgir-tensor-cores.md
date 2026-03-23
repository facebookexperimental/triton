# TTGIR Tensor Core Ops

Matrix multiply-accumulate operations that execute on GPU tensor cores.

## Hopper (SM90): Warp Group MMA

### `ttng.warp_group_dot`
Hopper wgmma (warp group matrix multiply-accumulate). Executes
`D = A * B + C` on tensor cores.

- **Operand A**: SMEM memdesc or register tensor
- **Operand B**: SMEM memdesc (always)
- **Operand C**: Register tensor (accumulator input)
- **Result D**: Register tensor (accumulator output)

Supports async mode (`isAsync=true`) where the result is not immediately
available — must call `warp_group_dot_wait` before reading.

```mlir
%d = ttng.warp_group_dot %a, %b, %c
    : !ttg.memdesc<128x64xf16, #shared, #smem>
    * !ttg.memdesc<64x128xf16, #shared, #smem>
    -> tensor<128x128xf32, #mma>
```

### `ttng.warp_group_dot_wait`
Waits until there are `pendings` or fewer outstanding async wgmma ops.
Must pass the result tensors of the in-flight dot ops as `inputs` so the
compiler can track dependencies.

```mlir
%synced = ttng.warp_group_dot_wait %d {pendings = 0}
    : tensor<128x128xf32, #mma>
```

## Blackwell (SM100): TCGen5 MMA

### `ttng.tc_gen5_mma`
Blackwell tensor core MMA. Executes `D += A * B`.

- **Operand A**: SMEM memdesc
- **Operand B**: SMEM memdesc
- **Operand D**: TMEM memdesc (accumulator, read and written in-place)
- **useD**: Whether to accumulate (true) or overwrite (false)

Key differences from Hopper:
- Accumulator lives in TMEM, not registers
- Async by default; completion tracked via mbarrier (`barriers`)
- Supports 2-CTA mode (`two_ctas`) for distributed matmul across a cluster
- Takes/produces optional async token for TMEM aliasing tracking

```mlir
%token = ttng.tc_gen5_mma %a, %b, %d token(%dep) -> !ttg.async_token,
    %useD, %pred barriers(%bar : %bar_pred)
    {is_async}
    : !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #tensor_memory>
```

### `ttng.tc_gen5_mma_scaled`
Blackwell scaled MMA with block scaling factors. Same as `tc_gen5_mma`
plus scale descriptors for A and B.

- **a_scale, b_scale**: SMEM or TMEM memdescs containing block scales
- **a_type, b_type**: Element types for scaling (e.g., e4m3, e2m1)

Used for MX-format (microscaling) GEMM where weights/activations use
narrow types (FP4, FP8) with per-block scale factors.

```mlir
ttng.tc_gen5_mma_scaled %a, %b, %d token(%dep),
    %a_scale, %b_scale, %useD, %pred
    lhs = e4m3 rhs = e2m1
    : !ttg.memdesc<...>, !ttg.memdesc<...>, !ttg.memdesc<...>,
      !ttg.memdesc<...>, !ttg.memdesc<...>
```

## Memory Access Summary

| Op | Reads | Writes |
|---|---|---|
| `warp_group_dot` | A: SMEM or Reg, B: SMEM, C: Reg | D: Reg |
| `warp_group_dot_wait` | (sync only) | (sync only) |
| `tc_gen5_mma` | A: SMEM, B: SMEM, D: TMEM (if useD) | D: TMEM |
| `tc_gen5_mma_scaled` | A: SMEM, B: SMEM, scales: SMEM/TMEM, D: TMEM | D: TMEM |
