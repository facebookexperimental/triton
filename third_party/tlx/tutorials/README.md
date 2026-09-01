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
)

plan = prepare_varlen_backward(cu_seqlens_q, cu_seqlens_k)
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale)

# Causal packed self-attention. Q and KV offsets and head counts must match.
dq, dk, dv = fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale, causal=True)
```

Plan preparation validates and copies `cu_seqlens_q` and `cu_seqlens_k` to the
host once to build compact BM16/BN128 schedules plus a masked BM32/BN256
schedule for long non-causal split-GQA, then owns cloned device offsets.  The
frozen plan prevents field rebinding, but its PyTorch tensors are still mutable;
treat every plan-owned offset and schedule tensor as immutable after
preparation.  Keep plan construction outside the timed or repeated execution
path.  Every sequence must have at least one query and one key/value token.
`lse` is contiguous FP32 with shape `(query_heads, total_q)`.  In causal mode,
V may use the TritonBench-style `v_storage[:, 0]` view: the head and D axes must
remain dense while the token stride may include gaps.  Returned `dv` is always
contiguous.

In non-causal mode the Q and KV offsets are independent.  To model an
extend-attention workload, pack only the extend tokens in Q and pack the full
context in K/V, using `q_len = extend_len` and
`kv_len = prefix_len + extend_len` for each request.  Causal cross-attention
and causal GQA are rejected by the current specialization.
