# FP16 GEMM Kernel Optimization on AMD GFX9 (TLX)

This directory presents a **step-by-step optimization journey of an FP16 GEMM kernel
written in TLX**, targeting **AMD GFX9 GPUs** (developed on gfx950 / MI355).

Rather than showing a single "final" kernel, it documents **how high performance is
achieved**—from a naive baseline to a near-optimal design—covering **memory movement,
layout design, latency hiding, and instruction scheduling** along the way. Each version
(`v0` → `v9`) introduces **one core optimization concept**, so the diff between two
consecutive steps *is* the lesson.

It is the TLX counterpart to the Gluon `gfx9_gemm/a16w16` tutorial: same shapes, same
`v*` progression, so the two can be read side by side to compare the languages.

fp16, 256×256×64 tiles, 4 warps, `matrix_instr_nonkdim=16`.
Shape: M=N=4096, K=8192 (matching the Gluon benchmark).

## Results (TFLOPS, K=8192)

All versions use `num_stages=1` with manual pipeline management.

Measurement: `do_bench` steady-state (warm up, then median of 9 runs) on a single
idle gfx950 (MI355), M=N=4096, K=8192, fp16. Both columns measured in the same
process on the same GPU for an apples-to-apples comparison. **rocBLAS steady ≈ 1160T.**
Absolute TFLOPS vary a few % chip-to-chip (clock); the **%rocBLAS ratio is the stable
metric**, so treat the absolute column as "this GPU, this run."

Note: the machine has 8 GPUs and is shared — pin to an idle one (`HIP_VISIBLE_DEVICES=<n>`)
for stable numbers, and discard the first run (a cold/boost-clock run reads ~15-20%
high and is not sustained — steady-state under load is the number reported here).

| Version | Gluon | TLX | %rocBLAS | Description |
|---------|------:|----:|---------:|-------------|
| v0_naive | —¹ | 614 | 53% | Baseline: `tl.load` + `tl.dot` + `tl.store`, no pipelining |
| v1_buffer_load | 497 | 608 | 52% | `buffer_load` for loads, `tl.store` for epilogue |
| v2_async_copy | 648 | 652 | 56% | `buffer_load_to_local` direct-to-LDS + explicit swizzled `order=[0,1]` for B |
| v3_lds | 690 | 652 | 56% | Padded shared layout `[(512,16)]` (explicit, the teaching step) |
| v4_global_prefetch | 866 | 836 | 72% | Manual 2-stage pipeline + inferred padded layout |
| v5_local_prefetch | 868 | 836 | 72% | Manual 3-stage pipeline + inferred padded layout |
| v6_loop_unroll | 833 | 859 | 74% | Step-2 loop unrolling, alternating register sets |
| v7_slice | 958 | 913² | 79%² | N-sliced B, manual 4-region pipeline; conflict-free LDS layout, AGPR accumulator pins, scalar K offsets |
| v8_warp_pipeline | — | **1005** | **86%** | v7 + `warp_pipeline_stage` for `s_setprio` mem/MFMA interleave (8 warps) |
| v9_beyond_hotloop | —¹ | **1026** | **89%** | v8 + PID remap (8 XCDs) + workgroup swizzle (GROUP_SIZE_M=4) |

The v8/v9 numbers include the post-misched flag, which is **baked into their kernel
source** (see below) — no env var needed.

**post-misched (baked in):** v8/v9 set `TRITON_DISABLE_POST_MISCHED=1` from their kernel
source (`os.environ.setdefault(...)`, with a comment). The LLVM post-RA machine scheduler
would otherwise re-order the hand-tuned mem/MFMA interleave of the warp pipeline; disabling
it preserves the schedule (~+1-2%). It's a generic LLVM scheduling flag
(`enable-post-misched=false`) that only affects codegen for the current compile — override
with `TRITON_DISABLE_POST_MISCHED=0` to see the difference.

Note the two final steps are ordered hot-loop-first: **v8** finishes the hot loop
(`warp_pipeline_stage`, the big +~90T jump over v7), then **v9** does the grid-level
"beyond the hot loop" scheduling last (PID remap + swizzle, +~30T). The biggest single
win is the warp pipeline; the grid remap is the finishing polish.

¹ Gluon v0 and `beyond_hotloop` don't compile on this Triton build (Gluon API drift:
`convert_layout` / `extract_slice` signatures changed). `v8_warp_pipeline` is TLX-only
(Gluon has no warp-pipeline step). The upstream Gluon tutorial reports `beyond_hotloop`
≈ 1137T base and up to ~1405T with its custom `llirSched` + `amdgcnSched` scheduler
passes — that scheduler control is the main remaining headroom over the LLVM backend's
default scheduling.

² Measured before v7 gained its explicit LDS layout, accumulator pins and scalar K offsets.
For the current kernel, unscheduled and with the LLIR scheduler plugin, see
[v7 and the LLIR scheduler](#v7-and-the-llir-scheduler).

### Layout inference

From v4 to v6 the padded shared layout is **inferred by the compiler** from the dot
operands — `tlx.local_alloc(...)` with no `layout=` argument produces the exact same
`padded_shared<[512:+16]>` encoding (and identical assembly) as the hand-written
`tlx.padded_shared_layout.with_identity_for([(512,16)], ...)`. v3 keeps the explicit
form to teach what the layout is; v2 keeps an explicit *swizzled* layout to stay the
pre-padding step. See `InsertRequireLayout.cpp`.

v7 pins explicit offset bases again. For direct-to-LDS buffers the inferred layout keeps
plain row-major order, which costs 4 bank conflicts per LDS instruction; the bit
permutation v7 uses measures 0 (see [below](#v7-and-the-llir-scheduler)).

## v7 and the LLIR scheduler

v7 is the last 4-wave step: one wave both loads and computes. v8 closes the remaining
scheduling gap with hardware (8 waves and `warp_pipeline_stage`). The other way is the one the
Gluon tutorial uses: a compiler pass that interleaves the MFMAs with the memory work inside each
wave. That pass, the LLIR scheduler, is available here as an out-of-tree LLVM pass plugin in
[`../plugins/llir_scheduler`](../plugins/llir_scheduler/README.md).

For the pass to deliver, v7 has to be the same kernel as the Gluon `v7_sliceN`. Three details
of the kernel source matter:

| | Without it | In v7 |
|---|---|---|
| LDS layout | inferred padded layout, plain row-major order: 4 bank conflicts per LDS instruction | explicit conflict-free offset bases: 0 |
| Accumulator | LLVM splits it across VGPRs and AGPRs: ~140 `v_accvgpr` copies in the hot loop | `tlx.amd_dot(..., cd_regclass="a")` pins every MFMA tile to AGPRs: 0 copies |
| K offset | added to every element of the i32 offset tensors: vector adds in the hot loop | advances the scalar base pointer: none |

Measured with this `bench.py` on one idle gfx950 (MI355X), fp16, M=N=4096, K=8192, two rounds.
`triton` is the default `do_bench` median; `batched` is `--timing-mode batched`. This is a
different GPU from the table at the top, so compare by the ratio to rocBLAS. "Before" is v7
with the inferred layout, no pins and tensor K offsets.

| Kernel | `triton` TFLOPS | %rocBLAS | `batched` TFLOPS | %rocBLAS | `SQ_LDS_BANK_CONFLICT` per launch |
|---|---:|---:|---:|---:|---:|
| rocBLAS | 1393 - 1421 | | 1461 - 1470 | | |
| v7 before | 1088 / 1085 | 77% | 1164 / 1161 | 79% | 1.68e7 |
| v7 before + plugin | 1223 / 1225 | 87% | 1255 / 1255 | 86% | 1.68e7 |
| v7 | 1054 / 1059 | 75% | 1226 / 1226 | 84% | 0 |
| v7 + plugin | **1368 / 1412** | **99%** | **1496 / 1495** | **102%** | 0 |

The three details are not a consistent win on their own: without the scheduler, LLVM's default
schedule still decides the outcome. They are what lets the scheduler deliver, and the scheduled
hot loop is then instruction-for-instruction the scheduled Gluon v7 loop (256 MFMA, 64
`ds_read_b128`, 32 direct-to-LDS loads, 0 `v_accvgpr`, 0 vector ALU).

The plugin is for 4-wave kernels. Do not load it for v8 / v9 or other `warp_pipeline_stage`
kernels: it declines their regions but still pads them, which measured 2-5% slower.

```bash
../plugins/llir_scheduler/build.sh
python bench.py --version 7 --K 8192                                     # unscheduled
LLVM_PASS_PLUGIN_PATH=$PWD/../plugins/llir_scheduler/libLlirSched.so \
  python bench.py --version 7 --K 8192                                   # scheduled
```

## Compiler Fixes Applied

**Buffer-op coalescing**: the AMD `tritonamdgpu-coalesce-buffer-ops` pass already
coalesces `buffer_load`/`buffer_store` but early-returned on `buffer_load_to_local`.
Extended it to coalesce `buffer_load_to_local` too — it picks the contiguous order
and vectorization width from the i32 offset tensor's axis info (the offset tensor
drives the global-load addressing; the op's result is a shared memdesc, so there is
no register tensor to rewrite). Without this, v2–v9 fail to legalize
(`unrealized_conversion_cast`).

## Key Lesson: `other=0.0` hurts performance

Passing `other=0.0` to `buffer_load` causes 2x regression (319T → 590T) because
the compiler generates extra register copies to implement the fallback value for
masked-out lanes. On AMD, `buffer_load` with mask already returns 0 for masked
lanes, so `other` is redundant.

## TLX vs Gluon: Key Differences

All versions use `num_stages=1` with manual pipeline management (matching Gluon's approach).

- **v0-v3**: TLX matches or exceeds Gluon — layout propagation works well.
- **v4-v7**: Manual pipelines reach ~72-77% of rocBLAS, a few % behind Gluon. The gap
  is LLVM-backend instruction scheduling — Gluon's published best uses custom
  `llirSched` / `amdgcnSched` passes that the default backend scheduler doesn't match.
  v7 can now be scheduled by the same LLIR scheduler, loaded as a pass plugin: 99% of
  rocBLAS at 4 waves (see [v7 and the LLIR scheduler](#v7-and-the-llir-scheduler)).
- **v8**: `warp_pipeline_stage` (`s_setprio` mem/compute interleaving) is the big
  hot-loop win — 1005T / 86% of rocBLAS, ahead of Gluon v7 (958T) measured here.
  TLX-only (Gluon has no warp-pipeline step).
- **v9**: PID remapping (8 XCDs) + workgroup swizzle adds grid-level L2 reuse on top of
  v8 (→ 1026T / 89%) — the final step, matching Gluon's `beyond_hotloop` concept. TLX's
  best on this build.

## Running

```bash
cd third_party/tlx/tutorials/gfx9_gemm/a16w16
HIP_VISIBLE_DEVICES=1 python bench.py --version 9 --K 8192   # v9 best (~1025T / 89%); post-misched baked into source
HIP_VISIBLE_DEVICES=1 TRITON_DISABLE_POST_MISCHED=0 python bench.py --version 9 --K 8192  # disable it to see the difference
for v in 0 1 2 3 4 5 6 7 8 9; do python bench.py --version $v --K 8192; done  # All
```
