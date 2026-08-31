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
for packed BF16 THD, non-causal MHA backward with head dimension 128.  Prepare a
plan once for immutable cumulative sequence offsets, then reuse it for every
backward invocation with the same packing:

```python
from triton.language.extra.tlx.tutorials.amd_fa_varlen_bwd import (
    fa_varlen_backward,
    prepare_varlen_backward,
)

plan = prepare_varlen_backward(cu_seqlens_q, cu_seqlens_k)
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale)
```

Plan preparation validates and copies `cu_seqlens_q` and `cu_seqlens_k` to the
host once to build compact BM16/BN128 schedules, then owns cloned device
offsets.  The frozen plan prevents field rebinding, but its PyTorch tensors are
still mutable; treat every plan-owned offset and schedule tensor as immutable
after preparation.  Keep plan construction outside the timed or repeated
execution path.  Every sequence must have at least one query and one key/value
token.  `lse` is contiguous FP32 with shape `(heads, total_q)`.
