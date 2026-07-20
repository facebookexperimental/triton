# TLX Wave MXFP 16K ATT Report

Date: 2026-07-03

This note records the ATT methodology and findings for the TLX Wave MXFP4
kernel at `M=N=4096, K=16384` on gfx950.

## Scope

Kernel under test:

- Symbol: `_a4w4_kernel`
- Backend: `TRITON_DEFAULT_BACKEND=tlx_wave`
- Historical Wave scheduler region cap for this run: 2048 operations (the bridge override has since been removed)
- Cache root: `/tmp/tlx-a4w4-mxfp-wave-6d5b578`
- HSACO:
  `/tmp/tlx-a4w4-mxfp-wave-6d5b578/M4096_N4096_K16384/DQ6FEYYN7ZDCP3O3UCUFXXRDGCRY7HZYWVHVS5Y7STMIN7S3YULQ/_a4w4_kernel.hsaco`
- Wave submodule revision used for this run: `6d5b578`

The goal was to inspect actual hot-loop latency with AMD Advanced Thread Trace,
not infer the bottleneck from static assembly alone.

## Toolchain

Use ROCm tools from the conda environment:

```bash
CONDA_PREFIX=/home/ibutygin/miniforge3/envs/tlx-950
ROCPROF=$CONDA_PREFIX/bin/rocprofv3
HIPCC=$CONDA_PREFIX/bin/hipcc
LLVM_OBJDUMP=$CONDA_PREFIX/lib/python3.11/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump
ROCM_CORE=$CONDA_PREFIX/lib/python3.11/site-packages/_rocm_sdk_core
ROCM_DEVEL=$CONDA_PREFIX/lib/python3.11/site-packages/_rocm_sdk_devel
```

The ATT decoder path must come from the conda env:

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROCM_DEVEL/lib:$ROCM_CORE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ROCPROF_ATT_LIBRARY_PATH="$ROCM_DEVEL/lib"
```

Using the package-internal rocprof wrapper path directly caused decoder trouble
or incomplete output. The working setup used `$CONDA_PREFIX/bin/rocprofv3`.

## Runner Setup

The Python benchmark path was not used for the final ATT capture. Running the
full Python process under rocprofv3 produced large numbers of unrelated code
object registration messages and a stuck process in one attempt.

Instead, a temporary HIP runner was built from Wave's
`wave-matmul-calibrate-runner.cpp` and launched the cached HSACO directly.

The stock runner's `tlx-mxfp` ABI path was not faithful for this cached TLX
kernel because the source TTGIR runtime arguments are:

```text
a, b, c, a_scales, b_scales, M, N,
stride_am, stride_bn, stride_cm, stride_ask, stride_bsk
```

For the Python benchmark, scale tensors have shape `[rows, K / 32]` with
`stride(0)=1` and `stride(1)=rows`. The temporary runner was adjusted to:

- store scale bytes in K-group-major layout, matching the Python tensors;
- pass `stride_ask = M`;
- pass `stride_bsk = N`;
- keep `stride_am = stride_bn = K / 2`;
- keep `stride_cm = N`;
- disable output checking for the ATT capture.

Smoke-run command:

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROCM_DEVEL/lib:$ROCM_CORE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
/tmp/tlx-a4w4-att-runner/tlx-a4w4-att-runner \
  --m 4096 --n 4096 --k 16384 \
  --bm 2 --bn 2 --wave-m-tiles 8 --wave-n-tiles 8 --wave-k-tiles 2 \
  --wave-size 64 \
  --input-type mxfp4 --c-type bf16 --kernel-abi tlx-mxfp \
  --dynamic-lds 0 \
  --warmup 0 --iters 1 --no-check \
  "$HSACO" _a4w4_kernel
```

Observed one-launch smoke timing was about `163 us`, consistent with the
benchmark scale for this kernel.

## ATT Capture

Single-dispatch ATT runs only emitted code object snapshots, not decoded
`ui_output_*` and `stats_ui_output_*` files. The working capture used multiple
launches and traced an interior dispatch, matching the pattern used by the
Gluon ATT scripts.

ATT job file:

```json
{
  "jobs": [
    {
      "kernel_include_regex": "_a4w4_kernel",
      "kernel_exclude_regex": "",
      "kernel_iteration_range": "[15]",
      "advanced_thread_trace": true,
      "att_target_cu": 0,
      "att_shader_engine_mask": "0xF",
      "att_simd_select": "0x3",
      "att_buffer_size": "0x20000000"
    }
  ]
}
```

On gfx10+ `att_simd_select` is a SIMD ID, not a mask; `0x3` selects SIMD 3.

Capture command:

```bash
OUT=/tmp/tlx-a4w4-wave-att-16k-runner-conda-json-6d5b578

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROCM_DEVEL/lib:$ROCM_CORE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
ROCPROF_ATT_LIBRARY_PATH="$ROCM_DEVEL/lib" \
"$ROCPROF" \
  --att \
  --att-library-path "$ROCM_DEVEL/lib" \
  --output-format json \
  -i /tmp/tlx-a4w4-att-16k.json \
  -d "$OUT" -o out \
  -- /tmp/tlx-a4w4-att-runner/tlx-a4w4-att-runner \
       --m 4096 --n 4096 --k 16384 \
       --bm 2 --bn 2 --wave-m-tiles 8 --wave-n-tiles 8 \
       --wave-k-tiles 2 --wave-size 64 \
       --input-type mxfp4 --c-type bf16 --kernel-abi tlx-mxfp \
       --dynamic-lds 0 \
       --warmup 0 --iters 20 --no-check \
       "$HSACO" _a4w4_kernel
```

Decoded files were produced under:

```text
/tmp/tlx-a4w4-wave-att-16k-runner-conda-json-6d5b578/
```

Important files:

- `stats_ui_output_agent_43157_dispatch_16.csv`
- `ui_output_agent_43157_dispatch_16/code.json`
- `ui_output_agent_43157_dispatch_16/se*_wv*.json`
- `out_results.json`
- `out_43157_shader_engine_*.att`

The traced run reported:

```text
iters: 20
per_launch_us: 149.353
per_launch_cycles_wallclock: 328577
```

## Post Processing

Static ISA was dumped with the conda LLVM objdump:

```bash
"$LLVM_OBJDUMP" -d --mcpu=gfx950 "$HSACO" \
  > /tmp/tlx-a4w4-wave-att-16k-runner-conda-json-6d5b578/hsaco.objdump.s
```

Wave ATT import:

```bash
python3 third_party/wave/tools/wave-att-import/wave-att-import.py \
  --att-dir "$OUT" \
  --code-object "$HSACO" \
  --llvm-objdump "$LLVM_OBJDUMP" \
  --arch gfx950 \
  --trip-count 31 \
  --summary \
  --output "$OUT/per_pc.csv"

python3 third_party/wave/tools/wave-att-import/wave-att-import.py \
  --att-dir "$OUT" \
  --code-object "$HSACO" \
  --llvm-objdump "$LLVM_OBJDUMP" \
  --arch gfx950 \
  --trip-count 31 \
  --windows \
  --output "$OUT/windows.csv"

python3 third_party/wave/tools/wave-att-import/wave-att-import.py \
  --att-dir "$OUT" \
  --code-object "$HSACO" \
  --llvm-objdump "$LLVM_OBJDUMP" \
  --arch gfx950 \
  --trip-count 31 \
  --window-summary \
  --output "$OUT/window_summary.csv"
```

Import coverage was clean:

```text
static_instructions: 2185
stats_rows: 2185
stats_resolved: 2185
stats_unresolved: 0
wave_files: 4
wave_rows: 66368
wave_resolved: 66368
wave_unresolved: 0
```

The `K=16384` kernel has `K / BLOCK_K = 64` pipeline steps. The source loop is
stepped by 2 over `0..62`, so the hot loop has 31 iterations. In the ATT data,
PC `0x2ac8` appears exactly 31 times per traced wave and was used as the loop
iteration marker.

## Findings

### Loop Period

Measured from consecutive executions of PC `0x2ac8`:

```text
intervals: 120
mean cycles/iteration: 7568.6
median cycles/iteration: 7442.0
min cycles/iteration: 7380
max cycles/iteration: 7980
```

This is the actual per-wave hot-loop period observed by ATT.

### Wait Windows

The importer found 53 wait windows total. The hot loop has 19 wait windows.

Steady-state hot-loop waitcnt cost:

```text
per-wave waitcnt stall over 31 iterations: 10649.988 cycles
waitcnt stall per iteration: 343.548 cycles
```

Breakdown of hot-loop wait windows:

```text
lgkm waitcnt: 307.548 cycles/iteration
vmem waitcnt:  36.000 cycles/iteration
```

The main hot-loop wait PCs:

```text
0x2f20 s_waitcnt lgkmcnt(0): 76 cycles/iteration
0x39f4 s_waitcnt lgkmcnt(0): 76 cycles/iteration
0x3ff0 s_waitcnt lgkmcnt(0): 68 cycles/iteration
0x3534 s_waitcnt lgkmcnt(2): 50 cycles/iteration
0x2ac8 s_waitcnt lgkmcnt(0): 17.548 cycles/iteration
```

These waits are around LDS read/write and barrier groups. They are not a single
bad global-memory wait.

### Representative Iteration Mix

One representative steady-state interval contained 481 dynamic instructions:

```text
v_mfma_scale           256
ds_read_b128            64
buffer_load_lds         38
s_add_i32               36
s_nop                   23
s_barrier               20
s_waitcnt               19
ds_read_b64_tr_b8        8
ds_write                 6
s_add_u32                4
s_addc_u32               4
s_mov_b32                1
s_cmp_lt_i32             1
s_cbranch_scc1           1
```

Instruction-level ATT stall averaged over all hot-loop intervals:

```text
class                  count/iter  stall/iter  duration/iter
v_mfma_scale              256.00     2986.27       4150.03
buffer_load_lds            38.00     1372.50       1536.50
ds_read_b128               64.00      597.53        867.07
s_waitcnt                  19.00      343.53        343.53
s_barrier                  20.00      149.47        189.47
salu                       70.00       92.00        303.33
ds_write                    6.00       46.00        102.00
ds_read_b64_tr_b8           8.00       26.00         58.00
```

The largest individual per-PC stalls in the hot loop are mostly
`buffer_load_dwordx4 ... lds`, for example:

```text
0x3aec buffer_load_dwordx4 ... lds: avg stall 194.129
0x3024 buffer_load_dwordx4 ... lds: avg stall 140.516
0x3ab0 buffer_load_dwordx4 ... lds: avg stall  82.548
0x3ac4 buffer_load_dwordx4 ... lds: avg stall  63.452
0x3ad8 buffer_load_dwordx4 ... lds: avg stall  58.677
```

### Interpretation

The hot loop is not dominated by one suspicious `vmcnt(4)`.

The explicit waitcnt stall is about:

```text
343.5 / 7568.6 = 4.5% of the hot-loop period
```

Most observed hot-loop latency comes from the MFMA/LDS-heavy instruction stream:

- 256 scaled MFMA instructions per loop iteration;
- 64 `ds_read_b128` instructions per loop iteration;
- 38 buffer-load instructions in the importer bucket, of which 32 are
  `buffer_load_dwordx4 ... lds` packet loads and 6 are miscellaneous global
  loads;
- 20 barriers per loop iteration;
- visible stalls on `buffer_load_dwordx4 ... lds` and LDS read clusters.

Therefore, optimizing a single waitcnt site is unlikely to materially change
the 16K kernel. The next useful comparison is against the known-fast Wave MXFP4
4-wave/register-scale kernel, using the same ATT method, to determine whether
the TLX kernel has extra LDS traffic/barriers or whether the same instruction
mix is scheduled/placed worse.

## Recommended Next Steps

1. Inspect the TLX Wave IR immediately after the bridge and verify that memory
   tokens only encode required happens-before edges. Distinct LDS allocations
   or non-aliasing staging slots should not be serialized through one token
   stream.
2. Trace the bridge lowering for the hot-loop global-to-LDS staging region.
   The asm has a dense load cluster with high ATT stall; determine whether that
   shape is already present in Wave IR or introduced later by scheduling.
3. Trace the bridge lowering for epilogue `ttg.convert_layout` plus store. The
   asm shows repeated LDS write/wait/barrier/read micro-roundtrips that are not
   structurally required by the reference kernel.
4. Only after the bridge produces an equivalent dependency graph and op mix,
   attribute the remaining gap to Wave scheduling.
5. Do not treat the earlier suspicious `vmcnt(4)` as the root cause by itself.
   It is inside a worse dependency/order region; the surrounding structure is
   the signal.

## Wave 4-Wave MXFP4 Reference Kernel

The known-fast upstream Wave kernel was profiled with the same ATT setup. This
is the regular Wave 4-wave MXFP4 path, not the TensileLite subtile variant.

Profile:

```text
kernel_profile: gfx950-mxfp4-256x256-4wave
symbol: wmma_f16_matmul_tiled
example: matmul
kernel_abi: matmul
input_type: mxfp4
output_type: f16
mxfp4_scale_path: regs
use_dma_lds: true
bm=2 bn=2 wave_m_tiles=8 wave_n_tiles=8 wave_k_tiles=2
target_waves=1
cta_swizzle_xcds=8 cta_group_m=4
```

Generated artifacts:

```text
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/gfx950_mxfp4_4wave_k16k.mlir
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/gfx950_mxfp4_4wave_k16k.scheduled.s
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/gfx950_mxfp4_4wave_k16k.scheduled.hsaco
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/hsaco.objdump.s
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/per_pc.csv
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/windows.csv
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/window_summary.csv
```

The calibrator needed a temporary build-dir shim because this checkout has Wave
Python packages under `third_party/wave/build/wave-build`, while LLVM tools are
under `third_party/wave/build/llvm-install`:

```bash
python3 third_party/wave/tools/wave-matmul-calibrate/wave-matmul-calibrate.py \
  --chip=gfx950 \
  --build-dir=/tmp/wave-cal-build-6d5b578 \
  --kernel-profile=gfx950-mxfp4-256x256-4wave \
  --m=4096 --n=4096 --k=16384 \
  --variants=scheduled \
  --skip-hw --no-check \
  --emit-hsaco "$OUT/gfx950_mxfp4_4wave_k16k.scheduled.hsaco" \
  --emit-asm "$OUT/gfx950_mxfp4_4wave_k16k.scheduled.s" \
  --emit-mlir "$OUT/gfx950_mxfp4_4wave_k16k.mlir"
```

The stock Wave runner was used without TLX-specific argument/layout patches:

```bash
"$HIPCC" -O2 \
  third_party/wave/tools/wave-matmul-calibrate/wave-matmul-calibrate-runner.cpp \
  -o "$OUT/wave-matmul-calibrate-runner"
```

Runtime launch parameters:

```text
dynamic_lds: 57344
grid: 16,16,1
block: 256,1,1
waves_per_workgroup: 4
kernel_arg_trip_count: 63
sim_loop_trip_count: 62
```

### Reference Timing

Direct HIP-event timing with `iters=200`, `warmup=25`, random seed `0`, and
`--no-check`:

```text
samples us: 118.925, 118.419, 118.348, 119.559, 117.739
median us: 118.419
mean us:   118.598
TFLOP/s:   4642.46
```

The TFLOP/s number uses `2 * M * N * K` for `M=N=4096, K=16384`.

One 9-repeat run through `wave-matmul-calibrate.py --run-hsaco` reported a
slower median of `140.453 us`. Direct reruns before and after that calibrator
run were stable around `118-119 us`, so the report uses the direct repeat set
and treats the `140 us` run as a contended/outlier measurement.

### Reference ATT Capture

ATT job file:

```json
{
  "jobs": [
    {
      "kernel_include_regex": "wmma_f16_matmul_tiled",
      "kernel_exclude_regex": "",
      "kernel_iteration_range": "[15]",
      "advanced_thread_trace": true,
      "att_target_cu": 0,
      "att_shader_engine_mask": "0xF",
      "att_simd_select": "0x3",
      "att_buffer_size": "0x20000000"
    }
  ]
}
```

The traced run reported:

```text
iters: 20
per_launch_us: 129.433
per_launch_cycles_wallclock: 284752
```

Import coverage:

```text
static_instructions: 1696
stats_rows: 1696
stats_resolved: 1696
stats_unresolved: 0
wave_files: 4
wave_rows: 81372
wave_resolved: 81372
wave_unresolved: 0
```

### Reference Loop Period

The repeated hot-loop body in this kernel is `code.json` static indices
`602..918`, corresponding to PCs `0x25d4..0x32b4`. The decoded `per_pc.csv`
indices are offset by one because `code.json` entry 0 is the symbol/comment
record, so loop aggregation used `per_pc.csv` static indices `601..917`.

Measured from consecutive executions of PC `0x25d4`:

```text
intervals: 244
mean cycles/iteration: 3556.0
median cycles/iteration: 3566.0
min cycles/iteration: 3504
max cycles/iteration: 3652
```

This Wave reference loop carries 128 `v_mfma_scale` instructions per iteration.
The earlier TLX bridge loop carried 256 `v_mfma_scale` instructions per
iteration. Normalized to the TLX 256-MFMA loop unit:

```text
Wave reference normalized cycles: 7112.1
TLX bridge cycles:                7568.6
TLX / Wave-reference ratio:          1.06x
```

### Reference Hot-Loop Mix

Aggregated over static indices `602..918` and divided by 62 iterations:

```text
class                  count/iter  stall/iter  duration/iter
v_mfma_scale              128.00     1387.71       1997.94
ds_read_b128               32.00      313.50        441.50
s_barrier                   6.00      135.63        147.63
s_waitcnt                   6.00       90.10         90.10
salu                       91.00       28.00        368.00
buffer_load_lds            16.00       21.84        150.55
buffer_load_dword           4.00       17.48         49.48
v_add_u32_e32               4.00       13.61         29.61
v_add3_u32                  7.00       10.69         38.69
ds_read_b64_tr_b8           8.00        9.81         41.81
ds_write                    4.00        4.55         28.55
v_lshl_add_u32              1.00        0.00          4.00
```

Main loop wait PCs:

```text
0x2a48 s_waitcnt lgkmcnt(0): 46.000 cycles/iteration
0x2630 s_waitcnt lgkmcnt(3): 28.097 cycles/iteration
0x3084 s_waitcnt vmcnt(24):   4.000 cycles/iteration
0x2834 s_waitcnt lgkmcnt(0):  4.000 cycles/iteration
0x25d8 s_waitcnt lgkmcnt(0):  4.000 cycles/iteration
0x2fdc s_waitcnt vmcnt(16):   2.000 cycles/iteration
0x3014 s_waitcnt vmcnt(12):   2.000 cycles/iteration
```

The explicit waitcnt stall is about:

```text
90.1 / 3556.0 = 2.5% of the reference hot-loop period
```

### TLX Versus Reference

The reference profile uses `f16` output while the cached TLX bridge kernel in
this report used `bf16` output. Full-kernel timings are therefore not a strict
epilogue apples-to-apples comparison; the hot-loop ATT normalization below is
the more relevant signal for the bridge/scheduler question.

Normalized to 256 scaled MFMA instructions:

```text
metric                         TLX bridge       Wave 4-wave reference
loop cycles                    7568.6           7112.1
waitcnt stall                   343.5            180.2
v_mfma_scale count              256.0            256.0
ds_read_b128 count               64.0             64.0
buffer_load_lds count            38.0             32.0
ds_read_b64_tr_b8 count           8.0             16.0
s_barrier count                  20.0             12.0
s_waitcnt count                  19.0             12.0
```

The fast reference kernel is not winning by eliminating the core MFMA/LDS read
work. The biggest visible differences in the steady loop are fewer barriers,
fewer waitcnts, lower explicit waitcnt stall, and fewer global-to-LDS data
loads per 256-MFMA unit. The scale path is also different: the reference profile
uses the register scale path and wide packed scale handling, while the TLX
bridge kernel still has the bridge-generated scale/layout work described above.

## ASM Scout Classification

This section classifies the suspicious asm regions as either fundamental to the
kernel/ISA shape or likely bridge deficiencies. The goal is to avoid blaming the
scheduler before checking whether the bridge lowered the TTGIR structure
faithfully.

### Fundamental Or Expected

The following are expected for this kernel family on gfx950:

- scaled MFMA instructions in the steady-state loop;
- LDS staging for operands and corresponding `ds_read_b128` traffic;
- `ds_read_b64_tr_b8` scale reads feeding scaled MFMA;
- some `s_waitcnt` and `s_barrier` boundaries around LDS producer/consumer
  groups;
- at least one epilogue LDS roundtrip can be reasonable when the final layout
  must be transposed/vectorized before global stores.

These features are also present in the known-fast Wave 4-wave reference kernel.
They are not, by themselves, evidence of a fundamental performance problem.

### Likely Bridge Deficiencies

The TLX bridge hot loop has too many hard synchronization boundaries when
normalized to the same 256-MFMA unit:

```text
metric                         TLX bridge       Wave 4-wave reference
s_barrier count                  20.0             12.0
s_waitcnt count                  19.0             12.0
waitcnt stall                   343.5            180.2
```

The most suspicious TLX hot-loop region is:

```text
/tmp/tlx-a4w4-wave-att-16k-runner-conda-json-6d5b578/hsaco.objdump.s:848
```

That region does:

- `s_waitcnt vmcnt(20)`;
- `s_barrier`;
- eight `ds_read_b128`;
- `s_waitcnt vmcnt(4)`;
- a scale `ds_write_b32`;
- `s_waitcnt lgkmcnt(0)`;
- another `s_barrier`;
- `ds_read_b64_tr_b8`;
- a dense `buffer_load_dwordx4 ... lds` cluster;
- `s_waitcnt lgkmcnt(0)`;
- then MFMA resumes.

The literal `vmcnt(4)` is not the root cause. ATT shows the larger cost in the
surrounding LDS/global-to-LDS region, especially at `buffer_load_dwordx4 ...
lds` PCs such as `0x3024` and `0x3aec`. This points to an over-serialized
staging/dependency shape. It may still be scheduling once the dependencies are
correct, but first the bridge IR must be checked for unnecessary token joins or
false aliasing between distinct LDS staging buffers.

The reference loop has the same core operations, but a cleaner structure around:

```text
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/hsaco.objdump.s:608
```

It performs scale `ds_read_b64_tr_b8`, waits with `lgkmcnt(3)`, enters MFMA, and
later issues scale global loads and staging work without the same large
stall-heavy packet-load cluster. This makes the TLX structure look like a bridge
or dependency-graph deficiency, not a fundamental ISA limitation.

The TLX epilogue has another bridge-shaped issue:

```text
/tmp/tlx-a4w4-wave-att-16k-runner-conda-json-6d5b578/hsaco.objdump.s:1579
```

It repeatedly emits:

```text
ds_write_b128*
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128*
s_waitcnt lgkmcnt(0)
s_barrier
```

The reference epilogue around:

```text
/tmp/wave-mxfp4-4wave-16k-att-6d5b578/hsaco.objdump.s:1610
```

uses one consolidated LDS write/read/store path with a store wait ladder. The
outputs differ (`bf16` in the cached TLX bridge kernel, `f16` in the reference),
so this is not an exact epilogue apples-to-apples comparison. Still, the TLX
repeated micro-roundtrips are not fundamental to AMDGPU ISA; they are most
likely from bridge lowering of `ttg.convert_layout` plus store.

### Attribution

Current attribution:

```text
core MFMA/LDS read traffic          fundamental / expected
scale transpose LDS reads           fundamental / expected
some waits and barriers             fundamental / expected
extra hot-loop barriers/waits        bridge deficiency until proven scheduler
dense packet-load stall cluster      bridge dependency/staging deficiency first
epilogue LDS micro-roundtrips        bridge store/layout lowering deficiency
whole-function s_barrier gap         bridge/lowering structure signal
whole-function s_waitcnt count       not sufficient alone
```

The immediate bridge audit should check memory-token threading, explicit
`ttg.convert_layout` lowering, and store lowering. Scheduling should be treated
as the explanation only after the bridge emits a faithful Wave IR dependency
graph and comparable operation structure.

## Bridge Follow-Up

The first concrete bridge deficiency found from this audit was LDS dependency
granularity for static `ttg.memdesc_index` views. The emission stage already
kept distinct top-level `ttg.local_alloc` roots independent and conservatively
joined roots through `select`, but `ttg.memdesc_index` inherited the parent
allocation root even when the bridge had a proven static byte offset and slot
size. That made different static slots of one allocation look may-aliasing to
the local-memory token model.

This is a structural bridge issue, not a scheduler or hardware issue. The
generic rule is:

- dynamic or unknown `memdesc_index` views keep the parent root;
- static views get canonical byte-interval roots under the ultimate allocation;
- dependency queries include exact roots plus overlapping parent/child interval
  roots, so parent accesses remain conservative;
- `select` of memdesc values still unions all possible roots.

This keeps op lowerers faithful: no lowering path inspects users or recognizes a
specific MXFP pattern. The local-memory dependency model now has enough
information for disjoint static LDS slots to avoid unnecessary token edges while
still preserving ordering for overlapping views and parent accesses.

## Post Static-Root Barrier Audit

After the static `ttg.memdesc_index` root fix, the old whole-function barrier
gap is no longer a valid attribution. A fresh 16K dump from the TLX Wave bridge
and the TLX LLVM backend shows comparable operation shape and no extra Wave
barriers:

```text
artifact                                  count
Wave IR wave.barrier                         34
Wave object s_barrier                        34
LLVM asm s_barrier                           40
Wave object s_waitcnt                        52
LLVM asm s_waitcnt                           53
Wave object ds_read                         176
LLVM asm ds_read                            176
Wave object ds_write                         44
LLVM asm ds_write                            44
Wave object buffer/global loads              76
LLVM asm buffer/global loads                 76
Wave object scaled MFMA                     512
LLVM asm scaled MFMA                        512
```

Fresh artifacts used for this audit:

```text
/tmp/tlx-a4w4-mxfp-wave-sweep-1783110106/M4096_N4096_K16384/DQ6FEYYN7ZDCP3O3UCUFXXRDGCRY7HZYWVHVS5Y7STMIN7S3YULQ/_a4w4_kernel.wave
/tmp/tlx-a4w4-mxfp-wave-sweep-1783110106/M4096_N4096_K16384/DQ6FEYYN7ZDCP3O3UCUFXXRDGCRY7HZYWVHVS5Y7STMIN7S3YULQ/_a4w4_kernel.objdump.s
/tmp/tlx-a4w4-mxfp-llvm-sweep-1783110087/M4096_N4096_K16384/Z7KJH3F7UUMPA3YM2ATVJ64NHEJ3BUMEVZDZVHOXN57VBYPLUZFQ/_a4w4_kernel.amdgcn
```

The 34 Wave IR barriers split as:

```text
pre-loop setup                                  3
main loop async-wait/LDS-read groups            4
main loop scale store-to-transpose-load groups  6
tail/final pipeline                             5
epilogue convert/store LDS micro-roundtrips    16
```

The six hot-loop scale barriers are not invented by the bridge. They correspond
to explicit `ttg.local_store` / `ttg.local_load` scale staging in the TTGIR
source. The LLVM backend lowers the same source structure into VMEM scale loads,
LDS writes, barriers, `ds_read_b64_tr_b8`, and waits around consumption. For the
current TLX-vs-LLVM comparison, the bridge is faithful in this part of the hot
loop.

The remaining bridge-shaped issue is the epilogue `ttg.convert_layout` plus
store path, which still manifests as repeated LDS write/barrier/read/barrier
micro-roundtrips. That is distinct from the whole-function barrier count and
should be investigated as generic layout conversion/store lowering. The
remaining gap against the upstream Wave 4-wave MXFP reference should be
attributed to source structure, scheduling, or layout/store lowering only after
comparing the TLX TTGIR and bridge Wave IR against that reference; it is not
explained by extra full-kernel barriers versus TLX LLVM.

The fresh LLVM asm also contains repeated LDS exchange barriers around the two
final `ttg.convert_layout` results and their buffer stores. In the LLVM artifact,
the analogous barrier clusters appear around lines 2136..2229 for the left
output and 2623..2653 for the right output. In the Wave object, the matching
clusters appear around lines 1583..1630 and 2055..2097. This means the
epilogue LDS roundtrips are a TLX TTGIR/source-layout issue relative to the
upstream Wave 4-wave reference, not evidence that the bridge is adding extra
barriers compared with TLX LLVM for the same input.
