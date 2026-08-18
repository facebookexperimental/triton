# TLX extensions used by adaptive FlashAttention

The adaptive FlashAttention tutorial at
[`amd_fa_adaptive.py`](amd_fa_adaptive.py) demonstrates four
APIs for expressing physical ownership without exposing a lane-mask ballot.
These APIs are general TLX facilities; none is specific to attention.

## Exact distributed linear layouts

```python
layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
    reg_bases=[[0, 1], [0, 2]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
    warp_bases=[[64, 0], [128, 0], [256, 0]],
    block_bases=[],
    shape=[512, 4],
)
pinned = tlx.require_layout(value, layout)
```

`distributed_linear_layout_encoding.make` constructs a Triton distributed
linear encoding directly from physical input bases:

- `reg_bases` describes values held by one lane.
- `lane_bases` describes how lane-id bits move through the logical tensor.
- `warp_bases` describes how warp-id bits move through the logical tensor.
- `block_bases` describes how program-block bits move through the tensor.
- `shape` names the logical output dimensions and their extents.

Every basis vector has one component per logical dimension.  The resulting map
must cover the requested shape; construction fails if the bases are not
surjective.  Use this API when an instruction ABI or register-level algorithm
requires exact physical ownership.  Prefer `tlx.layout(shape=..., stride=...)`
when a normal logical shape/stride description is sufficient, because it
leaves more freedom to layout inference.

## Ending an explicit layout contract

```python
ordinary = tlx.release_layout(pinned)
```

`tlx.release_layout` marks the point where downstream operations no longer
have to preserve the source's explicit encoding.  It does not move data by
itself; layout propagation decides whether a conversion is required at the
boundary.  Releasing an already ordinary tensor is legal and folds away.

Keep ownership pins local.  Pin around the instruction packet, cast, or
physical permutation that needs them, then release before returning to
shape-level code.  This prevents a register ABI choice from accidentally
becoming a whole-kernel layout constraint.

## Casting without losing ownership

```python
probability_bf16 = tlx.cast_preserve_layout(probability_f32, tl.bfloat16)
bits = tlx.cast_preserve_layout(probability_f32, tl.int32, bitcast=True)
```

Ordinary frontend cast inference may select a fresh result encoding.
`tlx.cast_preserve_layout` instead changes only the element type and retains
the source tensor's current IR encoding.  It supports floating-point
extension/truncation, floating-point conversion with
`fp_downcast_rounding=...`, and equal-width bitcasts with `bitcast=True`.
Passing the existing dtype is an identity.

Use it only inside a region whose physical ownership is intentional, such as
converting exponentiated score registers before MFMA consumption.  Release the
layout after that region rather than carrying the pin through unrelated code.

## Warp-uniform boolean votes

```python
all_safe = tlx.warp_all(per_lane_safe)
any_active = tlx.warp_any(per_lane_active)
```

`tlx.warp_all` and `tlx.warp_any` consume a distributed predicate containing
exactly one boolean per physical lane and return one scalar `i1` result that is
uniform for the warp.  Non-boolean inputs are compared with zero first.

These operations intentionally expose only the semantic reduction.  Use
`warp_all` when a group may take a fast path only if every lane satisfies a
contract, and `warp_any` when one participating lane is enough to trigger
work.  They are not tensor-axis reductions: `tl.all` and `tl.max` reduce a
logical tensor dimension and preserve tensor layout semantics, while the warp
votes reduce physical lane participation and produce a scalar suitable for a
uniform branch.

The adaptive attention kernel uses `warp_all` after duplicating its 32 logical
row decisions across the two halves of a 64-lane warp.  One unsafe row then
forces a uniform softmax rebase without materializing or interpreting a ballot
bit mask.

## Tutorial API

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
