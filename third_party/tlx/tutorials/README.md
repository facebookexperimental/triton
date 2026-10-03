# AMD attention tutorials

## PyTorch dense FlashAttention backward

The packaged gfx950 BF16 backward kernels live in
[`../ops/kernels/flash_attn/gfx950_bwd.py`](../ops/kernels/flash_attn/gfx950_bwd.py).
Current PyTorch releases can opt into them through the FlashAttention provider
registry:

```python
import triton.tlx.pytorch
from torch.nn.attention import activate_flash_attention_impl

activate_flash_attention_impl("TLX_GFX950_BWD")
```

Activation is process-global and replaces only
`aten::_scaled_dot_product_flash_attention_backward`; PyTorch continues to run
the forward kernel. The provider selects TLX only for correctness- and
performance-validated dense BF16 gfx950 routes. Dropout, deterministic mode,
unsupported layouts/shapes, and routes without a measured advantage call the
CUDA kernel captured at activation. Use
`torch.nn.attention.restore_flash_attention_impl()` to remove the override.
The automatic set is an exact allow-list of D64, D128, and D256 signatures that
won through the activated provider on MI350X; unmeasured shapes and known
losing members of the same kernel families fall back. For short D128
`(16, 27, 200, 128)`, the default split route improves forward plus backward by
1.15x non-causal and 1.35x causal. The exact-D128 opt-in improves it by 1.34x
and 1.39x, respectively. Both measured routes are eligible; unmeasured
persistent experiments and register-allocation overrides fall back.
The measured expansion also enables 13 dense D64 signatures from the long
MHA/GQA and rectangular-GQA kernels, with forward-plus-backward speedups of
1.10x--1.59x over PyTorch's CK/AITER path, plus three interleaved D128
signatures: non-causal MHA at N=1024/2048 (1.12x/1.07x) and causal GQA8 at
N=1024 (1.05x). Neighboring D128 shapes that did not clear the performance
gate remain on the native provider. For causal rectangular attention, use
PyTorch's `causal_lower_right(SQ, SKV)` bias; plain `is_causal=True` rejects
unequal sequence lengths before reaching FlashAttention.

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

## Symmetric-memory all-gather (Blackwell + MI350)

[`blackwell_dist_all_gather.py`](distributed/blackwell_dist_all_gather.py) and
[`amd_dist_all_gather.py`](distributed/amd_dist_all_gather.py) implement the same
forward-only, single-node all-gather on top of
`torch.distributed._symmetric_memory`. The pattern, identical on both
architectures:

1. Each rank allocates its shard with `symm_mem.empty()` and exchanges
   handles with `symm_mem.rendezvous()`, which publishes every rank's base
   pointer as an int64 `buffer_ptrs` table.
2. Triton kernels read peer shards with plain `tl.load` on the published
   pointers and `tl.store` into the local output slice; no collective
   library sits in the data path.
3. Phase boundaries are `handle.barrier(channel=...)` calls, alternating two
   channels so consecutive phases never share one.

The public entry point (`symm_mem_all_gather_into_tensor`) picks one of three
strategies from the per-rank payload: small payloads use concurrent fanout
peer reads, large even-length payloads use ring-ordered phased int32 reads
(two BF16 elements per 4-byte transaction), and odd BF16 lengths fall back to
phased BF16 reads to preserve slice alignment. Correctness is checked
bit-for-bit against the vendor collective plus determinism iterations.

Architecture deltas, all host-side (the three kernels are arch-agnostic):

| | Blackwell (NVLink) | MI350 (xGMI) |
| --- | --- | --- |
| Collective backend | NCCL (`backend="nccl"`) | RCCL (same string on ROCm builds) |
| Multicast | Disabled via `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` | Not set |
| Arch gate | `get_device_capability()[0] >= 10` | `gcnArchName` prefix `gfx950` |
| Correctness baseline | `_c10d_functional.all_gather_into_tensor` | Same where available, else `dist.all_gather_into_tensor` |
| Tuning | 64 MiB crossover, `num_warps=8`, `BLOCK_SIZE=4096`, with measured B200 table in the docstring | Same defaults carried over; xGMI retuning is future work |

Multi-node is out of scope: the pattern assumes intranode peer visibility,
same as the Blackwell original. The
[`test_symm_mem_dist_all_gather`](testing/test_correctness.py) unit test
covers both tutorials: it `mp.spawn`s two ranks, runs the 1 MiB / 64 MiB /
64 MiB+1-element correctness cases through rendezvous, barriers, and the
kernels, and skips unless 2+ Blackwell or gfx950 GPUs are visible. Run it
with `pytest -k test_symm_mem_dist_all_gather` from
`third_party/tlx/tutorials/testing`, or run either tutorial directly:

```bash
torchrun --standalone --nproc_per_node=2 \
  third_party/tlx/tutorials/distributed/blackwell_dist_all_gather.py --mode correctness

HIP_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  third_party/tlx/tutorials/distributed/amd_dist_all_gather.py --mode correctness
```
