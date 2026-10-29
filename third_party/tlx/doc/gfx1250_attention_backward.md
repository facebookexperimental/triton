# Attention backward on MI450 (gfx1250)

`third_party/tlx/ops/kernels/flash_attn/gfx1250_bwd.py` implements a
non-causal, head-dimension-128, BF16 flash-attention backward for gfx1250. It
supports MHA and GQA (`Hq % Hkv == 0`) with equal Q and KV lengths. The entry
point is `fa_backward(q, k, v, o, do, lse, sm_scale=None, *, config=None,
workspace=None)` on contiguous BHND tensors with FP32 LSE. It returns dQ, dK
and dV in BF16.

The kernel ports the delayed-dQ schedule of the gfx950 backward
(`gfx950_bwd.py`, PR #2709) to wave32 WMMA, TDM loads and the 320 KiB LDS of
gfx1250.

## Kernel structure

Each launch is three kernels:

1. **Preprocess.** `Delta = rowsum(O * dO)`. A separate small launch zeroes
   the dQ accumulator and any FP32 dK/dV partials. The zeroing must stay in
   its own launch. On MI450, a kernel that mixes the O/dO loads with these
   stores right after the previous call's atomics can stall forever when calls
   are issued back to back without synchronization.
2. **Main kernel** (`_delayed_bwd_kernel`). A workgroup owns one 256-row KV
   tile and loops over every Q tile of its query-head group. dK and dV stay in
   registers for the whole loop. dQ is accumulated with buffer atomics, either
   FP32 or packed BF16 (`packed_dq`).
3. **Convert** (`_convert_outputs`). Casts the dQ accumulator to BF16 and
   applies the softmax scale. When Q is split across workgroups, it also sums
   the FP32 dK/dV partials.

Per Q step the loop computes S = QKᵀ, P, dP = dO·Vᵀ and dS, then
dV += Pᵀ·dO and dK += dSᵀ·Q. Each step publishes dS to a two-slot LDS ring.
The dQ = dS·K product for step i runs during step i+1 (the "delayed" dQ), so
its LDS reads overlap the next step's score WMMAs instead of serializing
behind a barrier.

There are two schedules. `select_config` picks one per shape.

- **Occupancy-1 (`occ1`), the default for every BN256 shape we measured.**
  Four waves per workgroup at 1,024 VGPRs each, one workgroup per CU, and
  204 KiB of LDS. Each wave computes in 32-row phases over a 64-row TDM tile,
  with three TDM stages. V and score K stay resident in registers, and two
  bands of the dQ K operand do too (`occ1_k_bands=2`). Hardware WMMA queuing
  is enabled through `HW_REG_WAVE_SCHED_MODE` bit 2.
- **Eight-wave ping-pong.** Two groups of four waves are phase-shifted by half
  a step with `tlx.cond_barrier`, so one group's exponentials overlap the
  other group's WMMAs. It also uses partitioned LDS buffers for K/V and dS, so
  the two SIMD pairs read different 64 KiB LDS partitions. It remains the path
  for BN128 tiles (`N % 256 != 0`) and for shapes outside the occupancy-1
  coverage.

### dQ accumulation and grid shape

- `tile_major` puts KV tiles on grid X. `cluster_dq` then launches each KV
  head's tiles as one workgroup cluster (`ctas_per_cga` up to 16). Every
  contributor to a dQ row is in the same cluster, which lives on one shader
  engine, so the dQ atomics can use SE scope
  (`tlx.buffer_atomic_add(scope="cluster")`) instead of device scope. Above 16
  tiles per head, each cluster group accumulates into a private dQ slice, and
  the convert kernel sums the slices.
- `q_splits` splits each KV owner's Q range across workgroups so the grid
  fills the 192 CUs evenly. `tail_splits` splits only the KV owners of the
  incomplete final dispatch round. The other workgroups keep direct BF16 dK/dV
  stores, and only the tail owners write FP32 partials.

### Compiler support used by the kernel

- Partitioned shared encodings with dynamic subslices and TDM loads
  (`tlx.partitioned_shared_layout_encoding`).
- Explicit WMMA register layouts (`tlx.amd_wmma_layout`).
- Cluster-scope buffer atomics.
- The AMD layout-conversion sinking pass keeps conversions out of
  phase-shifted `cond_barrier` windows. Without this, ping-pong corrupts its
  outputs.

## Configuration

`BackwardConfig` fields are independent scheduling switches. `fa_backward`
validates their combinations and rejects unsupported ones, such as a
one-step ping-pong split or BM64 ping-pong without Q splits.

`select_config(n, b, hq, hk, packed_dq=...)` returns the measured policy. It
reads the CU count from the device unless `num_cus` is given. Several
dispatch-tail choices are keyed to the measured 192-CU shapes. Other CU
counts and shapes get the general rule. Called with only `n`, it returns the
base eight-wave policy.

`packed_dq=True` (the default) accumulates dQ in packed BF16. This halves
atomic traffic, and dQ error stays within the tolerances of the tests below.
`packed_dq=False` accumulates dQ in FP32.

## Performance

MI450 (Rev A0, 192 CUs, power-limited clocks), ROCm 10.0. Each point is the
median of three interleaved rounds against AITER's gfx1250 assembly backward
(`mha_bwd`, `use_asm_v3`, FP32 atomics), called through
`benchmarks/mi450_bwd_aiter_bridge.cpp`. Values are TFLOPS, counting the
backward as 5 GEMMs: 10·B·Hq·N²·128 FLOPs.

| Shape (Hq,Hkv,N) | FP32 dQ | Packed dQ | AITER | Packed / AITER |
|---|---:|---:|---:|---:|
| 16,16,1K | 872 | 916 | 695 | 1.32 |
| 64,8,1K | 975 | 1033 | 676 | 1.53 |
| 64,8,1K (B16) | 984 | 1039 | 686 | 1.51 |
| 16,16,2K | 1059 | 1098 | 786 | 1.40 |
| 64,8,2K | 1117 | 1156 | 772 | 1.50 |
| 64,8,2K (B16) | 1128 | 1166 | 776 | 1.50 |
| 16,16,4K | 1166 | 1196 | 827 | 1.45 |
| 64,8,4K | 1203 | 1231 | 816 | 1.51 |
| 64,8,4K (B16) | 1202 | 1232 | 816 | 1.51 |
| 16,16,8K | 1198 | 1238 | 838 | 1.48 |
| 64,8,8K | 1226 | 1252 | 831 | 1.51 |
| 64,8,8K (B16) | 1223 | 1250 | 823 | 1.52 |
| 16,16,16K | 1228 | 1263 | 877 | 1.44 |
| 64,8,16K | 1230 | 1256 | 876 | 1.43 |
| 64,8,16K (B15) | 1228 | 1264 | 875 | 1.44 |

The batch is 16 for MHA and 8 for GQA unless the row says otherwise. Against
AITER, the geometric-mean speedup is 1.37× (MHA, FP32 dQ), 1.42× (MHA,
packed), 1.45× (GQA, FP32) and 1.50× (GQA, packed). Every point is between
1.25× and 1.53×.

### Where the remaining time goes

At 4K and above, the main loop runs at about 83–89% of what a
TDM-pipelined BF16 GEMM reaches on the same part. Each 64-row step issues
320 WMMAs in about 4,770 cycles per wave. The loop is bounded by the
WMMA/LDS issue of one wave per SIMD: with 1,024 VGPRs per wave, it cannot
hold the working sets of two steps at once. Short sequences also lose time
to dispatch-round quantization and the dK/dV epilogue. The
`q_splits`/`tail_splits` policies recover part of that.

Approaches we measured and rejected:

- Moving the dQ reduction into LDS.
- Parking dK slices in LDS to free VGPRs for cross-step overlap.
- A persistent kernel with an asynchronous TDM dK/dV epilogue. With only two
  LDS staging slots, the in-order TDM completion counter serializes the
  stores before the next tile's loads.
- Changing the statistics layout.

## Testing

```bash
pytest python/test/unit/language/test_tlx_amd_gfx1250.py -k "attention_backward or partitioned"
```

The tests cover:

- Selector policy, including the CU-count fallbacks.
- Configuration validation.
- Assembly invariants: VGPR count, zero scratch, LDS size, WMMA count, the
  ratio of LDS reads to WMMAs, SE-scope dQ atomics, and no store waits in the
  loop.
- Correctness of the ping-pong, clustered and occupancy-1 paths against a
  PyTorch reference, including uneven Q splits and dispatch-tail splits.

Tests that compile or run kernels skip on targets other than gfx1250.
Selector and validation tests run on any host.

## Reproducing the measurements

Build the AITER bridge against an AITER checkout that provides the gfx1250
`mha_bwd` assembly:

```bash
hipcc -std=c++17 -O2 -shared -fPIC \
  -I$AITER/csrc/include benchmarks/mi450_bwd_aiter_bridge.cpp \
  -L$AITER/op_tests/cpp/mha -Wl,-rpath,$AITER/op_tests/cpp/mha -lmha_bwd \
  -o mi450-aiter-bridge.so
```

Then run a check and a paired benchmark:

```bash
export GPU_LOCK_HELD=1 AITER_ASM_DIR=$AITER/hsa
python benchmarks/mi450_fa_bwd.py --mode check --shape 2,6,2,384
python benchmarks/mi450_fa_bwd.py --mode bench --shape 8,64,8,2048 \
  --aiter-library mi450-aiter-bridge.so --repeat-ms 100
python benchmarks/mi450_fa_bwd.py --mode bench --shape 8,64,8,2048 \
  --config '{"packed_dq": false}' --aiter-library mi450-aiter-bridge.so
```

`--mode compile --artifacts DIR` compiles for an explicit gfx1250 target
without a GPU and saves the assembly together with resource usage.

## Limitations

- Non-causal only, D = 128, BF16 inputs, and N must be a multiple of the KV
  tile (256, or 128 for the BN128 path) with N ≥ 256.
- Sliding windows, dropout and variable-length batches are not supported.
- One query-head group's FP32 dQ workspace must fit in the 2 GiB signed
  buffer range: `(Hq/Hkv)·N·128 < 2²⁹`.
