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

## gfx1250 TDM kernels

The gfx1250 tutorials use TDM descriptor transfers and WMMA, including fused
loads with explicit warp masks. These kernels were ported from `tlx-amd-meta`:

| Module | Entry points and supported schedules |
| --- | --- |
| [`amd_tdm_gemm_pipelined.py`](amd_tdm_gemm_pipelined.py) | `matmul` retains the config-based API; `matmul_tdm_pipelined_single_warp_per_simd_schedule` adds the sliced-K schedule and transposed B. Both use TDM output stores. |
| [`amd_mxfp_gemm_tdm_pipelined.py`](amd_mxfp_gemm_tdm_pipelined.py) | `matmul` retains the config-based API; `mxgemm_tdm_pipelined` supports FP8/FP4, optional A scales, baseline/sliceK/sliceNK/sliceMNK, and none/2way/4way/partial TDM fusion. `TDM_SPLIT` splits the operand LDS buffers. |
| [`amd_fa_tdm_pipelined.py`](amd_fa_tdm_pipelined.py) | `attn_fwd_tdm_pipelined` implements non-causal BF16 attention with FP32 output and a hand-written K/V pipeline. |
| [`amd_grouped_gemm_gfx1250/`](amd_grouped_gemm_gfx1250/README.md) | `grouped_gemm_phase0` accepts ragged pointer tables; `grouped_gemm_tdm` accepts packed A `[sum(M_g), K]`, B `[G, N, K]`, and int32 group offsets. |

The grouped TDM path requires each group M to be a multiple of `block_m`,
N to be a multiple of `block_n`, and K to be a multiple of 128 with enough
tiles to fill the ring. Cross-tile prefetch requires dedicated C staging,
two input buffers, and an even number of K tiles. The existing tile-ranking
heuristic is carried over from the reference kernels; benchmark on the target
system when choosing a configuration.

For a functional grouped GEMM check:

```bash
python third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py \
  --m_list 256,128 -N 512 -K 512 -BM 128 -BN 256 \
  --num_programs 2 --dedicated_c_buffer --cross_tile_prefetch \
  --benchmark_mode none --check
```

The [grouped benchmark driver](amd_grouped_gemm_gfx1250/bench.py) runs regular
shape sweeps in separate processes; use `--dry-run` to inspect the commands.
The GEMM and MXFP modules also provide benchmark CLIs (`--help`). Compile
checks live in `python/test/unit/language/test_tlx_codegen.py`; gfx1250 runtime
checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`. The port
uses `clamp_bounds=False` explicitly to retain the reference descriptor
positioning semantics, including predicates and pre-clamped descriptors.
