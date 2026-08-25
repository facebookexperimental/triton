We introduce TorchTLX to boost PT2 performance by pushing TLX primitives deeper into inductor template and fusion infra.

This work runs in parallel with our ongoing Triton compiler investments. TorchTLX lets us rapidly validate hardware-specific optimizations and bring forward wins while feeding proven techniques back to the Triton compiler.

## Overview

PT2 flows through Dynamo for graph capture and Inductor for optimization and code gen. Within Inductor, a scheduler decides which operations to fuse and then delegates to a backend-specific codegen engine, selecting among template candidates (Triton/CUTLASS) and cuBLAS via autotuning.

However, low-level hardware primitives such as tile shapes, warp spec roles and TMA directionality are invisible to the scheduler, limiting its ability to leverage modern GPU architecture when making fusion decisions.

Meanwhile, standalone TLX matmul kernels are matching or slightly exceeding cuBLAS. Unlike opaque cuBLAS kernels, a TLX kernel at parity opens the door to richer epilogue fusions.

In summary, TorchTLX integrates TLX as a low-level backend for Inductor, introducing significant changes to template selection, fusion logic and kernel codegen.

## Knob

`TORCHINDUCTOR_TLX_MODE` (unset by default)
- unset / any other value - TLX not considered (standard inductor behavior; the config value is `None`)
- allow - TLX added to candidates, competes via autotuning
- force - TLX templates enabled + forced epilogue fusion

Only `allow` and `force` engage TLX; every other value (including an unset env var or a legacy `default`) maps to `None`, leaving TLX off. TLX is additionally a no-op unless the active Triton is the fbtriton fork.

## TorchTLX Templates

TorchTLX templates are jinja-based kernel definitions that encode our best-performing TLX kernels into Inductor's template infra. We have shipped a GEMM template targeting Blackwell, with a flex attention template in active development. The underlying TLX GEMM kernel incorporates a series of hardware optimizations: warp specialization, CLC-based persistent execution, multi-CTA cooperation, epilogue subslicing, data partitioning and dynamic split-K.

Our performance strategy operates on two parallel tracks: improving standalone TLX kernels upstream and closing any remaining gap when running through the Inductor template path.

## Naming Conventions

Names carry the architecture, because in this codebase "which GPU" is the question
that decides almost everything: which template is offered, which config table is
valid, and how much on-chip memory a tile may use. A name that hides the arch
invites a config tuned for one part to be applied to another.

**Arch ids** follow `triton._internal_testing`, so the two agree on what "the same
target" means: `sm90`, `sm100`, `sm107` on NVIDIA (compute capability) and
`gfx942`, `gfx950`, `gfx1250` on AMD. One caution that module documents — match
`gfx1250` before any `gfx12` prefix rule, or an RDNA4 check swallows it.

**Templates are named for the arch they were tuned and validated on**, and the same
name appears at all three layers:

| jinja file | uid (appears in generated kernel names) | Python object |
|---|---|---|
| `blackwell_gemm_ws.py.jinja` | `tlx_blackwell_gemm_ws` | `blackwell_gemm_ws_template` |
| `gfx950_addmm_warppipe.py.jinja` | `tlx_gfx950_addmm_warppipe` | `gfx950_addmm_warppipe_template` |

The file drops the `tlx_` prefix (the directory is already TLX); the uid keeps it.
Heuristic classes follow the template they serve: `BlackwellGemmWSConfigHeuristic`,
`Gfx950AddMMWarpPipeConfigHeuristic`.

Note that **registration is deliberately broader than the name**. The warp-pipe
heuristics register for all of ROCm and the Blackwell GEMM for all of CUDA, so a
config still has to fit the *actual* target's budget — gfx942 has a quarter of
gfx950's LDS. Validate against that target's limits rather than assuming the one
in the name.

**Vendor names are for the dispatch layer only** — the code that branches on
vendor rather than on part: `append_tlx` → `_append_tlx_{nvidia,amd}`,
`_AmdArch`, `IS_ROCM`. Say `amd`, not `ROCm`, for the vendor; ROCm is the build.

**Each vendor keeps its own vocabulary** in the arch classes, with two
vendor-agnostic accessors so callers need not care:

| concept | NVIDIA | AMD | vendor-agnostic |
|---|---|---|---|
| on-chip memory | `smem_bytes` | `lds_bytes` | `on_chip_bytes()` |
| processors | `num_sms` | `num_cus` | `processor_count()` |

Finally, **an absent attribute means the concept does not exist there, or is not
yet measured** — there are no sentinel values. `tmem_columns` appears first on
`Sm100`, `num_xcds` only on AMD, and `Gfx1250` declares neither an LDS size nor a
CU count. Reading one is an `AttributeError` at the call site that should have
gated, never a plausible-looking `0` or a neighbouring part's number.

## TorchTLX Fusions

TorchTLX supports epilogue fusion through two store mechanisms.
Inductor's standard epilogue fusion works out of box when a TLX templated kernel uses tl.store. The scheduler detects fusible downstream ops (e.g., bias add, ReLU) and injects them into the template's `{{store_output}}` hook at codegen time.

```
                        tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                        offs_cn = offs_bn + slice_id * slice_size
                        _tmp_var1 = offs_am
                        _tmp_var0 = (_tmp_var1 + tl.arange(0, BLOCK_M_SPLIT))[:, None]
                        _tmp_var2 = _tmp_var0 < 4096
                        _tmp_var4 = offs_cn
                        _tmp_var3 = (_tmp_var4 + tl.arange(0, slice_size))[None, :, ]
                        _tmp_var5 = _tmp_var3 < 4096
                        _tmp_var6 = _tmp_var2 & _tmp_var5
                        xindex = _tmp_var3 + 4096*_tmp_var0
                        tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(xindex, [BLOCK_M_SPLIT, slice_size])), _tmp_var6, eviction_policy='evict_last').to(tl.float32)
                        tmp1 = result + tmp0
                        tmp2 = tl.full([1], 0, tl.int32)
                        tmp3 = triton_helpers.maximum(tmp2, tmp1)
                        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M_SPLIT, slice_size])), tmp3, _tmp_var6)
```

For templates using tlx.async_descriptor_store, we introduce compute_epilogue to run the fused epilogue ops and assign the result to a variable, letting the template manually stage through SMEM and issue async_descriptor_store to preserve the async pipeline overlap with MMA warps.

```
                        tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                        offs_cn = offs_bn + slice_id * slice_size
                        xoffset = offs_am
                        xindex = (xoffset + tl.arange(0, XBLOCK))[:, None]
                        xmask = xindex < 4096
                        yoffset = offs_cn
                        yindex = (yoffset + tl.arange(0, YBLOCK))[None, :, ]
                        ymask = yindex < 4096
                        fused_c = result
                        tmp0 = tl.load(in_ptr2 + (yindex + 4096*xindex), None, eviction_policy='evict_last').to(tl.float32)
                        tmp1 = result + tmp0
                        tmp2 = tl.full([1, 1], 0, tl.int32)
                        tmp3 = triton_helpers.maximum(tmp2, tmp1)
                        fused_c = tmp3
                        c_smem = c_smem_buffers[(group_id * EPILOGUE_SUBTILE + slice_id) % NUM_EPILOGUE_SMEM_BUFFERS]
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(c_smem, fused_c.to(A.dtype.element_ty))
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(desc_c, c_smem, [offs_am, offs_cn], eviction_policy="evict_first")
```

## Vision for Next-gen Hardware

As new GPU architectures arrive, TorchTLX offers two key advantages.

First, it streamlines access to state-of-the-art hardware intrinsics. Given that TLX provides a Triton-native API for features like TMA, WS, CLC, 2-CTA, etc., once the performance is maximized in the standalone TLX kernel, integrating those wins into the PT2 stack through TorchTLX requires a fairly small amount of additional work.

Second, it scales better than manual agentic kernel authoring. There is growing interest across the industry in agentic kernel authoring at scale. We believe making the PT2 stack natively aware of hardware architecture features might be a more scalable path. Once the plumbing is in place, every new kernel and fusion pattern benefits automatically.

## Source and tests

Templates, heuristics and codegen live in
[`third_party/tlx/language/tlx/inductor/`](third_party/tlx/language/tlx/inductor/).

```bash
pytest python/test/unit/language/test_torchtlx_templates.py
pytest python/test/unit/language/test_torchtlx_fusions.py
```

In an fbtriton git checkout, [`scripts/run_torchtlx_tests.sh`](scripts/run_torchtlx_tests.sh)
runs both. CI coverage is described in [CI](.github/workflows/).
