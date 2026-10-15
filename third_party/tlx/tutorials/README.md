# Adaptive FlashAttention tutorial

[`amd_fa_adaptive.py`](amd_fa_adaptive.py) implements adaptive and
fixed-reference non-causal FlashAttention for BF16 tensors on gfx950.  Its
module documentation contains the online-softmax equations, numerical
contracts, pipeline structure, and detailed variant guidance.

The general TLX APIs used to express its register ownership and uniform votes
are documented in the repository [root README](../../../README.md#other-operations).

## API

```python
from third_party.tlx.tutorials.amd_fa_adaptive import attention

# General-purpose adaptive reference tracking.
out = attention(q, k, v)

# Fixed reference for comparison; require proven bounds and profile the target.
out = attention(q, k, v, qk_max_abs=1.0)
```

See the tutorial module docstring for the online-softmax equations, numerical
contracts, pipeline structure, and guidance for selecting the adaptive or
fixed-reference specialization.  The current gfx950 LLVM code is fastest with
adaptive reference tracking; the bounded specialization is not a performance
shortcut unless measurements on the exact target prove otherwise.  Run
`python third_party/tlx/tutorials/amd_fa_adaptive_bench.py --help` for the
correctness/performance driver.

## Buffer Store Contiguity

```python
tlx.buffer_store(
    stored_value,
    ptr,
    offsets,
    mask=None,
    cache=None,
    contiguity=1,
)
```

`contiguity` is a correctness contract, not a best-effort hint. For every group
of `contiguity` elements owned by one thread, the caller promises that the
offsets are consecutive, the mask predicate is identical, and the first address
meets the packed access's alignment. The compiler verifies that the width is a
positive power of two, divides the number of elements owned by each thread, and
does not exceed the physical adjacency of the pinned layout; it cannot verify the
runtime address or mask promises. An incorrect promise can change which
elements are written.

Pinning `stored_value` anchors the store layout; the store's operand constraint
keeps offsets and mask in the same encoding. The final instruction width is
also capped by AMD's 128-bit buffer-store limit. For example, BF16
`contiguity=16` is emitted as legal eight-element `buffer_store_dwordx4`
operations. The default `contiguity=1` retains the existing scalar-contract
behavior.

```python
store_layout: tl.constexpr = tlx.amd_mfma_layout(
    version=4,
    instr_shape=[16, 16, 32],
    transposed=True,
    warps_per_cta=[2, 2],
)
mask_boundary = mask_boundary - mask_boundary % 4
mask = offsets < mask_boundary
values = tlx.require_layout(values, store_layout)
tlx.buffer_store(
    values,
    out_ptr,
    offsets,
    mask=mask,
    contiguity=4,
)
```
