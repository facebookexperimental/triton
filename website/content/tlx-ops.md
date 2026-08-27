`tlx.ops` is a curated library of production-ready GPU kernels written in TLX, shipped as part of the FBTriton wheel and callable directly from Python.

## What tlx.ops is

TLX ships dozens of kernels under `third_party/tlx/tutorials/`. They are fast and well understood, but they were never an API: the only way to call one was to import the tutorial module and reach for whichever host wrapper it happened to export, along with its config lists and prune hooks. Which file to import, and which of several variants was the good one, was tribal knowledge.

`tlx.ops` turns that body of work into a library.

- **Promotion, not rewriting.** Kernels are promoted from `tutorials/` once they are performant and proven in production. The tutorial copy stays where it is and is frozen; the promoted copy is where fixes and performance work land.
- **One blessed kernel per (op, arch).** Where several tutorial variants exist for the same operation on the same part, the catalog picks the best one. Callers do not choose between `ws`, `clc` and `2cta` — that taxonomy stays internal, which also means a kernel can be re-blessed without an API change.
- **A higher bar on the way in.** Promotion requires correctness coverage in CI. Every cataloged op runs a hardware-agnostic shape sweep across dtypes and memory layouts on real hardware.
- **`tutorials/` becomes a playground.** As ops take over coverage, the tutorial tree is intended to become a scratch space for experiments rather than a de facto library, with CI attention moving to `tlx.ops`.

## Interface

```python
from triton.tlx.ops import mm as tlx_mm

out = tlx_mm(a, b)
```

Architecture is detected from the current device, so the same code runs wherever the op is available. If an op has no implementation for the current GPU it raises `UnsupportedOp`, naming the architectures where it does exist — it never silently falls back to torch, because a silent fallback turns "TLX is not running here" into an unexplained performance cliff.

An architecture can be pinned explicitly, which is mainly useful in tests and benchmarks:

```python
out = tlx_mm(a, b, arch="sm100")
```

Architectures are named the way PTX and the ROCm toolchain name them — `sm90`, `sm100`, `gfx942`, `gfx950` — rather than by marketing name. The same strings appear in the directory layout and in dispatch, so there is one vocabulary rather than a mapping table between three.

## Structure

```
third_party/tlx/
    ops/                          -> (symlink) python/triton/tlx/ops
        __init__.py               the public functions; this file is the API
        _catalog.py               (op, arch) -> implementation, plus dispatch
        kernels/
            mm/                   sm90.py  sm100.py  gfx942.py  gfx950.py
            addmm/                gfx942.py
            flash_attn/           sm90.py  sm100.py
            hstu_attn/            sm100.py  gfx942.py
            ...
    tutorials/                    unchanged; frozen once promoted
```

Kernels are grouped by op, then by architecture, so an op's coverage is visible by listing one directory. A file named for an architecture is an entry point; underscore-prefixed siblings such as `_util.py` or `_reference.py` are private helpers for that op.

The catalog itself is a static table mapping `(op, arch)` to an import path as a plain string, resolved on first call. Importing `triton.tlx.ops` therefore never imports a kernel module or builds an autotune config table, and an architecture you are not running is never touched.

## Current availability

The library is new and the catalog is being filled in incrementally. Available today, all on Blackwell:

| Op | Architecture | Notes |
|---|---|---|
| `mm` | `sm100` | fp16 / bf16, either operand row- or column-major |
| `flash_attn` | `sm100` | forward and backward, causal and non-causal |
| `hstu_attn` | `sm100` | ragged sequences, SiLU-scaled scores |

Cataloged and queued for promotion, in rough priority order: `mm`, `addmm` and `bmm` on `sm90`, `gfx942` and `gfx950`; `flash_attn` on the AMD parts; then the workload ops — `gdpa`, `grouped_gemm`, `addmm_glu`, `paged_decode`, `ikbo_fa`, `ikbo_lce`, `multi_cta_layernorm`, `cross_attention` and `bmm_shared_a`.

An op is either in the catalog for a given architecture or it is not; there is no partially-working state. `UnsupportedOp` tells you which architectures a given op currently covers.
