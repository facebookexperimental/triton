# Attention backward on MI450 (gfx1250)

`third_party/tlx/ops/kernels/flash_attn/gfx1250_bwd.py` implements a
head-dimension-128, BF16 flash-attention backward for gfx1250, non-causal or
causal. It supports MHA and GQA (`Hq % Hkv == 0`) with equal Q and KV lengths.
The entry point is `fa_backward(q, k, v, o, do, lse, sm_scale=None,
causal=False, *, config=None, workspace=None)` on contiguous BHND tensors with
FP32 LSE. It returns dQ, dK
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

### Causal attention

Causal attention (query row `r` sees keys `0..r`) runs only the occupancy-1
kernel without dispatch-tail splits.

- KV tile `j` sees only query rows `>= 256·j`. Its pass iterates over a
  compacted outer index that skips the fully masked 64-row Q tiles of every
  head in the group. A helper maps that index back to the flattened Q tile
  for TDM loads and dQ rows. The TDM ring slot follows the compacted index.
- Only the first four outers of each head cross the diagonal. Each pass
  visits every head's diagonal outers first, in their own loop, which masks
  the exponent to `-inf` before `exp2`, so P and dS are zero above the
  diagonal. The steady loop then runs the non-causal phase with no mask or
  branch. Splitting the loop this way was worth −3% to −14% at MHA and up to
  −5% at short GQA, against a single loop with a runtime mask branch. With
  four splits, one diagonal outer per head is left, and the split loop
  spills, so that case keeps one loop in head-major order.
- The delayed dQ of a phase-0 step targets the last row tile of the previous
  compacted outer, which may belong to the previous query head.
- Work per KV tile falls linearly with `j`, and a head's tiles launch as one
  cluster that finishes with its heaviest tile. As in AITER's causal kernel,
  each workgroup therefore runs two passes, tile `i` and then tile `NT−1−i`,
  so every workgroup in a cluster does the same work. Grid X and the cluster
  are halved. Each cluster still covers every KV tile of its dQ slice, so the
  dQ atomics keep SE scope. Without pairing, causal 4K–8K ran at about 0.85×
  AITER; pairing made 4K–16K 20–36% faster.
- Pairing halves the grid, so short and GQA grids can leave the last
  dispatch round partly empty. For GQA, `q_splits` of 2 or 4 split every
  pass. Split `s` owns the outers `o` with `o % S == s` in every KV tile, so
  all contributors to a dQ row stay in one cluster, and each split writes a
  private FP32 dK/dV slice that convert sums. The selector splits in two
  when the last round is at most half full and in four on GQA grids of at
  least eight rounds: −5% to −11% at those shapes. MHA passes are too short
  to amortize a split, and lost up to 30%.
- The Q/dO ring, statistics and dS buffers are allocated once for both
  passes. V offsets come from pinned 1D ranges, and dK/dV are stored per band
  from the WMMA layout, so no layout conversion needs LDS beside the rings
  (209 KiB in total).
- The preprocess and convert kernels are unchanged.
- Skipping the dQ WMMAs of fully masked KV bands in the diagonal loop with
  uniform branches was rejected: the branches break the band schedule and
  spill at MHA (+7% to +12%), for at most −1.7% at GQA.
- The pair loop keeps more values live around each pass, so causal keeps
  one resident dQ K band (none with four splits): −0.2% to −1.2%.
- Also measured and rejected: running tiles unpaired (+15% to +75%, the
  clusters finish with their heaviest tile), issuing the next pass's Q/dO
  ring with its K load (neutral), and loading the next pass's V before the
  dK/dV stores (+30% to +44% at MHA).

The main loop has no spills. The pass prologue, drain and epilogue spill
152–164 bytes per pass, because the pass loop keeps a few more values live at
the 1,024-VGPR limit. Non-causal configurations compile to the same ISA as
before.

Causal timings from the same A0 setup, one round per shape. TFLOPS count
5·B·Hq·N²·128 FLOPs, half the non-causal convention:

| Shape (B,Hq,Hkv,N) | AITER | FP32 dQ | Packed dQ | FP32 / AITER | Packed / AITER |
|---|---:|---:|---:|---:|---:|
| 16,16,16,1K | 527 | 574 | 633 | 1.09 | 1.20 |
| 16,16,16,2K | 690 | 795 | 841 | 1.15 | 1.22 |
| 16,16,16,4K | 785 | 998 | 1041 | 1.27 | 1.32 |
| 16,16,16,8K | 843 | 1123 | 1152 | 1.33 | 1.37 |
| 16,16,16,16K | 870 | 1185 | 1223 | 1.36 | 1.41 |
| 8,64,8,1K | 511 | 570 | 617 | 1.12 | 1.21 |
| 16,64,8,1K | 524 | 618 | 675 | 1.18 | 1.29 |
| 8,64,8,2K | 667 | 798 | 863 | 1.20 | 1.29 |
| 8,64,8,4K | 770 | 910 | 991 | 1.18 | 1.29 |
| 8,64,8,8K | 834 | 1067 | 1123 | 1.28 | 1.35 |
| 8,64,8,16K | 864 | 1138 | 1199 | 1.32 | 1.39 |

AITER runs its `causal_br` assembly kernel. With equal lengths, bottom-right
causal is ordinary causal.

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
- Causal correctness for MHA and GQA from N = 512 to 8K in both dQ
  precisions and 1, 2 or 4 Q splits, with NaN-poisoned workspaces, and
  rejection of unsupported causal configs.

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

- D = 128, BF16 inputs, and N must be a multiple of the KV tile (256, or 128
  for the BN128 path) with N ≥ 256.
- Causal attention requires `N/256` in {2, 4, 8, 16} or a multiple of 32
  (paired tiles per cluster). It supports `q_splits` of 1, 2 or 4 (each
  split needs two outers of the shortest pass), but not `tail_splits`.
- Sliding windows, dropout and variable-length batches are not supported.
- One query-head group's FP32 dQ workspace must fit in the 2 GiB signed
  buffer range: `(Hq/Hkv)·N·128 < 2²⁹`.
