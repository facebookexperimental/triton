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
| v7_slice | 958 | 913 | 79% | N-sliced B + inferred padded layout, manual 4-region pipeline |
| v8_warp_pipeline | — | **1007** | **87%** | v7 + `warp_pipeline_stage` for `s_setprio` mem/MFMA interleave (8 warps) |
| v9_beyond_hotloop | —¹ | **1038** | **89%** | v8 + PID remap (8 XCDs) + workgroup swizzle (GROUP_SIZE_M=4) |

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

### Layout inference

From v4 onward the padded shared layout is **inferred by the compiler** from the dot
operands — `tlx.local_alloc(...)` with no `layout=` argument produces the exact same
`padded_shared<[512:+16]>` encoding (and identical assembly) as the hand-written
`tlx.padded_shared_layout.with_identity_for([(512,16)], ...)`. v3 keeps the explicit
form to teach what the layout is; v2 keeps an explicit *swizzled* layout to stay the
pre-padding step. See `InsertRequireLayout.cpp`.

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
- **v8**: `warp_pipeline_stage` (`s_setprio` mem/compute interleaving) is the big
  hot-loop win — 1007T / 87% of rocBLAS, ahead of Gluon v7 (958T) measured here.
  TLX-only (Gluon has no warp-pipeline step).
- **v9**: PID remapping (8 XCDs) + workgroup swizzle adds grid-level L2 reuse on top of
  v8 (→ 1038T / 89%) — the final step, matching Gluon's `beyond_hotloop` concept. TLX's
  best on this build.

## Running

```bash
cd third_party/tlx/tutorials/gfx9_gemm/a16w16
HIP_VISIBLE_DEVICES=1 python bench.py --version 9 --K 8192   # v9 best (~1040T / 89%); post-misched baked into source
HIP_VISIBLE_DEVICES=1 TRITON_DISABLE_POST_MISCHED=0 python bench.py --version 9 --K 8192  # disable it to see the difference
for v in 0 1 2 3 4 5 6 7 8 9; do python bench.py --version $v --K 8192; done  # All
```
