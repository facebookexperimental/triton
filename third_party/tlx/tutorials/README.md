# AMD attention tutorials

## Adaptive FlashAttention

[`amd_fa_adaptive.py`](amd_fa_adaptive.py) implements adaptive and
fixed-reference non-causal FlashAttention for BF16 tensors on gfx950.  Its
module documentation contains the online-softmax equations, numerical
contracts, pipeline structure, and detailed variant guidance.

The general TLX APIs used to express its register ownership and uniform votes
are documented in the repository [root README](../../../README.md#other-operations).

### API

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

## Packed variable-length FlashAttention backward

[`amd_fa_varlen_bwd.py`](amd_fa_varlen_bwd.py) provides a gfx950 specialization
for packed BF16 THD backward with head dimension 128.  Non-causal mode supports
MHA/GQA with `Hq % Hkv == 0`; causal mode supports MHA self-attention with
identical Q/KV cumulative offsets and `Hq == Hkv`.  Prepare a plan once for
immutable cumulative sequence offsets, then reuse it for every backward
invocation with the same packing:

```python
from triton.language.extra.tlx.tutorials.amd_fa_varlen_bwd import (
    fa_varlen_backward,
    prepare_varlen_backward,
    validate_varlen_backward_plan,
)

plan = prepare_varlen_backward(
    cu_seqlens_q,
    cu_seqlens_k,
    q.shape[0],
    k.shape[0],
    max_seqlen_q,
    max_seqlen_k,
)
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale)

# Use FP32 dQ accumulation; inputs and returned gradients stay BF16.
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale, dq_atomic_fp32=True)

# Causal packed self-attention. Q and KV offsets and head counts must match.
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale, causal=True)
```

When all four totals/maxima are supplied, plan preparation requires contiguous
offsets, asynchronously validates them, and builds compact BM16/BN128 schedules
plus a masked BM32/BN256 schedule on the GPU. The legacy two-argument call also
accepts strided rank-1 offsets, cloning them into contiguous plan storage before
copying them to the CPU to infer metadata, and therefore synchronizes. Invalid
zero-start, monotonicity, terminal-total, or maximum-length metadata sets a
device error flag that suppresses schedule consumption. The metadata-supplied
path does not copy sequence offsets or task counts to the host. Prepare and
consume a plan on the same CUDA stream. For untrusted offsets, call
`validate_varlen_backward_plan(plan)` once before reuse; this explicitly
synchronizes the error flag and raises `ValueError` on invalid metadata. Pass
`causal=True` to this validator when separately allocated Q/KV offset tensors
must also be checked for value equality. The frozen plan prevents field
rebinding, but its PyTorch
tensors are still mutable; treat every plan-owned offset and schedule tensor as
immutable after preparation. Keep plan construction outside the timed or
repeated execution path. Every sequence must have at least one query and one
key/value token.
`lse` is contiguous FP32 with shape `(query_heads, total_q)`.  In causal mode,
V may use the TritonBench-style `v_storage[:, 0]` view: the head and D axes must
remain dense while the token stride may include gaps.  Returned `dv` is always
contiguous.

The default dQ path uses BF16 accumulation and atomics. The FP32 option enables
FP32 accumulation, with FP32 atomics on most routes. Eligible aligned non-causal
MHA and GQA cases use a shared interleaved schedule: each KV tile reuses K/V
across its query heads and retains their dK/dV contributions in FP32 before
storing BF16 outputs. For the batch-19 prefix configuration
`(total_q, total_kv, max_q, max_kv) = (50754, 100696, 5662, 10414)`, the eager
Hq/Hkv=12/4 and 64/8 paths can materialize BF16 dS and compute dQ in a separate
consumer. One producer owns each KV tile, accumulates every query chunk and
sibling head in FP32, and stores BF16 dK/dV directly. The consumer scales each
KV256 dQ partial before adding it to the FP32 total, with one final BF16
conversion.

Both split paths require non-causal FP32 mode and 16-byte aligned Q/K/V/dO
bases. Hq/Hkv=12/4 uses a fixed-pitch dS temporary of about **11.974 GiB**.
Hq/Hkv=64/8 uses compact per-sequence rectangles padded to Q16 and KV256;
the seeded prefix benchmark uses about **36.59 GiB**. The compact layout
requires metadata attached by legacy plan preparation from the two cumulative
sequence-length tensors. Plans prepared with device metadata, and older plans
without compact dS metadata, retain the H64 FP32-atomic route.

Capture on the current stream of Q's device, or `torch.cuda.OutOfMemoryError`
from the optional dS allocation, selects the existing FP32-atomic route.
H12's fallback uses a 0.292 GiB dQ accumulator and two owners with FP32 dK/dV
partials followed by a reduction. H64's fallback uses a 1.558 GiB dQ accumulator
and one owner with direct BF16 dK/dV stores. These memory figures exclude all
other live tensors. Other allocation, capture-query, device-context, and kernel
errors propagate without retry.

Other shapes, causal attention, and BF16 dQ accumulation retain their existing
paths, except that misaligned inputs bypass wide aligned copies through the
generic BM16 kernel.

In non-causal mode the Q and KV offsets are independent.  To model an
extend-attention workload, pack only the extend tokens in Q and pack the full
context in K/V, using `q_len = extend_len` and
`kv_len = prefix_len + extend_len` for each request.  Causal cross-attention
and causal GQA are rejected by the current specialization.

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
