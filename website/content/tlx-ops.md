Production-ready kernels promoted from the TLX tutorials into the FBTriton op library.

## What tlx.ops is

- Promote performant and prod-impactful ops from TLX tutorials into the FBTriton op library.
- Catalog existing OSS TLX kernels and pick the best kernel per (arch, op).
- Enforce the highest quality bar for both code change and CI.
- Demote `tutorials/` to a playground scratch with no CI coverage.

## Interface

```python
from triton.tlx.ops import mm as tlx_mm

out = tlx_mm(a, b, arch="sm100")
```

`sm100` rather than `blackwell`, to align with the PTX convention.

## Structure

```
third_party/tlx/
    ops/                    -> (symlink) python/triton/tlx/ops
        kernels/
            mm/          sm90.py  sm100.py  gfx942.py  gfx950.py
            addmm/       gfx942.py
            bmm/         gfx942.py
            ...
```

## Current availability

Implemented today:

```
mm/sm100.py
flash_attn/sm100.py
hstu_attn/sm100.py
kda/sm100.py
```

Current kernel selections for the rest of the catalog:

```
kernels/
    mm/                   sm90.py  sm100.py  gfx942.py  gfx950.py
    addmm/                gfx942.py
    bmm/                  gfx942.py
    flash_attn/           sm90.py  sm100.py            (fwd + bwd)
    hstu_attn/            sm100.py  gfx942.py
                          _util.py  _stubs.py  _reference.py
    kda/                  sm100.py
    gdpa/                 sm100.py  gfx950.py
    grouped_gemm/         sm100.py  gfx950.py
    addmm_glu/            gfx950.py
    paged_decode/         gfx950.py
    ikbo_fa/              gfx950.py
    ikbo_lce/             gfx950.py
    multi_cta_layernorm/  sm100.py
    cross_attention/      sm100.py
    bmm_shared_a/         gfx950.py
```
