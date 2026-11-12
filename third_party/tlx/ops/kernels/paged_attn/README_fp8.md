# Per-head FP8 paged decode

The public interfaces are `triton.tlx.ops.paged_attention_decode_fp8` and
`paged_attention_decode_fp8_workspace`. Catalog dispatch selects gfx1250 from
the query tensor's device. Other architectures raise `UnsupportedOp`; callers
do not select an architecture or kernel variant. Only the `heuristic` search
space is available. The existing BF16/FP16 `paged_attention_decode` is unchanged.

This operator consumes **plain E4M3FN with per-head multipliers**, not E8M0
block-scaled MXFP8. Its inputs are:

| Tensor | Shape / dtype |
|---|---|
| Q | `[batch * query_length, query_heads, D]`, BF16 or E4M3FN |
| K | `[blocks, kv_heads, D/16, page_size, 16]`, contiguous E4M3FN |
| V | `[blocks, kv_heads, page_size/16, D, 16]`, contiguous E4M3FN |
| context_lens | `[batch]`, contiguous int32/int64 |
| block_tables | `[batch, max_pages]`, int32/int64 with contiguous rows |
| k_scale / v_scale | `[kv_heads]`, contiguous FP32 dequantization multipliers |
| q_scale | `[query_heads]`, contiguous FP32, required only for FP8 Q |
| output | Q's shape, contiguous BF16, before output projection |

`D` is 64 or 128; page size is a positive multiple of 16. Query heads must
be divisible by KV heads. A query head maps to
`query_head // (query_heads / kv_heads)`. Thus the same operator handles MHA,
GQA and MQA. All tensors share a device. Scale values must be positive and
finite and the logical inputs finite. These device-side value preconditions
are not read back to the host.

`context_lens` includes the queries' K/V. Causal queries align with the last
`query_length` context positions. For independent decode requests, set
`query_length=1`; for noncausal queries over one shared cache, set
`query_length=T` and `causal=False`. The original workload's exact head mapping
and T semantics remain undisclosed; the tested mappings are explicit synthetic
controls. A positive `sliding_window` includes the current key and requires
causal attention. Zero disables the window. Empty or fully masked rows return
zero. Context lengths must lie in `[0, max_context_len]`, and referenced block
table entries must identify allocated pages.

```python
import torch
from triton.tlx import ops

workspace = ops.paged_attention_decode_fp8_workspace(
    q, key_cache, value_cache, block_tables, max_context_len=2304,
)
out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
ops.paged_attention_decode_fp8(
    q, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale,
    max_context_len=2304, q_scale=None, out=out, workspace=workspace,
)
```

Allocate the workspace/output before graph capture. Workspace planning must use
the same query/cache shapes, context bound, query length, window and optional
`num_splits` override as attention. The helper returns `None` for one split;
otherwise it returns opaque FP32 capacity buffers with shapes
`[tokens, query_heads, splits, D]` and `[tokens, query_heads, splits]`. For at least
eight splits, raw-V normalized partials occupy FP16 views inside that capacity;
softmax, LSE and final merge arithmetic stay FP32. No dequantized global cache
or output-projection GEMM is introduced.

The implementation uses packed TDM transfers and grouped WMMA on page64/128,
with masked matrix/vector paths for other supported pages. Kernel and launch
policy selection are private. Explicit split counts from 1 through 1024 remain
available through the public operator and workspace helper.

Run public correctness tests on gfx1250:

```bash
python -m pytest -s --tb=short python/test/unit/tlx_ops/test_paged_attn_fp8_gfx1250.py
```

The same file includes device-free validation, dispatch, workspace-planning and
forward-only contract tests. Numerical tests use an independent dense FP32
oracle on the actual quantized inputs (`atol=5e-4`, `rtol=1e-2`), unequal head
scales, shuffled pages, poisoned tails, both Q dtypes, ragged/empty contexts,
causal/SWA masks, T sweeps, 128K/256K contexts and tiny aggregate probabilities.
They also exercise caller output/workspace reuse and CUDA-graph replay through
the public operator. The old tutorial entry points are removed without a shim.

Benchmark the public API, including split/merge, with packing, quantization and
reference checks outside timing:

```bash
python -m triton.tlx.ops.kernels.paged_attn.bench_fp8 --heads 16 --kv-heads 2 --dim 64 --length 2304
python -m triton.tlx.ops.kernels.paged_attn.bench_fp8 --heads 64 --kv-heads 64 --dim 128 --length 32768
```

The benchmark reports warm CUDA-graph latency and useful KV TB/s: unique active
K/V bytes divided by full attention time. This is effective throughput, not a
measurement of physical HBM traffic or a peak-bandwidth claim. Compare runs on
the same hardware with matching precision, scale, mask and cache protocols.
