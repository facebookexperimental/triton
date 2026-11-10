FP8 KV decode on gfx1250 (preliminary)
====================================

``amd_pa_decode_fp8.py`` establishes a plain E4M3FN cache contract for the
MI450X-MC work. It is separate from the existing gfx950 BF16/FP16 decode
operator. It does not change production dispatch or require compiler changes.

Layout and scales
-----------------

The layout follows the existing 5D tutorial. For one-byte FP8, X is 16::

    Q                  [batch * query_length, query_heads, head_dim]
    K                  [blocks, kv_heads, head_dim / 16, page_size, 16]
    V                  [blocks, kv_heads, page_size / 16, head_dim, 16]
    block_tables       [batch, max_pages]
    context_lens       [batch]
    k_scale, v_scale   [kv_heads]      float32
    q_scale            [query_heads]   float32, required only for FP8 Q
    O                  same shape as Q, bfloat16

Q may be BF16 or ``torch.float8_e4m3fn``. K/V must be contiguous
``torch.float8_e4m3fn`` tensors in the layouts above. ``head_dim`` is 64 or 128;
page size is a positive multiple of 16. Query heads must be divisible by KV
heads. Each query head maps to ``query_head // (query_heads / kv_heads)``.

Scales are dequantization multipliers, positive and finite. They are constant
across tokens and pages for each head. There are no MXFP8 block scales or
E8M0 scale bytes. FP32 softmax follows QK; V's head scale is applied to the
normalized weighted sum.

``context_lens`` includes the current queries' K/V. Causal query positions
are aligned to the end of each request's context. ``window_size`` includes
the current position: 1 attends only to that position; 0 disables the window.
An empty context or fully masked query returns zeros. Referenced block-table
entries and lengths must be valid; the launcher does not copy metadata to the
host to inspect its values. Unreferenced blocks are never accessed. The packed-page path may fetch unused
slots inside an allocated page; invalid columns are masked and invalid V values
are zeroed before PV, so poisoned tail slots cannot affect the result.

Mapping the preliminary workload table
-------------------------------------

The dense ``[T,8192]`` view needs an explicit head interpretation. For example,
128 heads of dimension 64 produce that width. The folded ``[1,16,64]`` example
has width 1024 and represents a different head configuration. The launcher
therefore accepts explicit head axes rather than inferring a model from 8192.

``T`` can mean independent requests (use ``batch=T, query_length=1``) or query
rows over one context (use ``batch=1, query_length=T``). The latter supports
right-aligned causal queries, or a dense noncausal sweep with ``causal=False``.
In particular, the supplied T-sweep reaches 8192 while L remains 2304, so it
cannot be interpreted as 8192 appended causal tokens without more context.
The embedded T-sweep tests explicitly choose noncausal queries over a shared
cache; they do not settle the model's intended masking semantics.

The output is the attention result before an output projection. The factorized
``[T,1024] x [1024,8192]`` O column does not supply projection weights or an
unambiguous contract, so that GEMM is outside this preliminary kernel.

Usage
-----

Import from this tutorial file, then call::

    out = fp8_kv_decode(
        q, key_cache, value_cache, context_lens, block_tables,
        k_scale, v_scale,
        max_context_len=2304,
        query_length=1,
        q_scale=q_scale,  # None for BF16 Q; [query_heads] for FP8 Q
        window_size=0,
        variant="auto",       # default: measured shape policy
        num_splits=None,       # allow the policy to select the split count
    )

The default ``variant="auto"`` chooses a retained candidate from explicit
physical shapes and the window. The tuning assumptions are synthetic, not a
claim about the undisclosed model's head mapping. ``select_decode_config``
returns that policy's ``variant``, ``block_n``, ``block_m``, ``num_warps`` and
``num_splits`` so callers can inspect or override it. Explicit split counts
from 1 through 1024 are supported; workspace size grows with that count. Explicit split counts
always take precedence.

Pass a contiguous BF16 ``out`` with Q's shape to reuse the output allocation.
For split counts greater than one, ``workspace`` is a pair of contiguous FP32
tensors with shapes ``[tokens, query_heads, num_splits, head_dim]`` and
``[tokens, query_heads, num_splits]``. Its size is
``tokens * query_heads * num_splits * (head_dim + 1) * 4`` bytes.
For S>=8, normalized raw-V partial values use FP16 and the V head scale
is applied in the final FP32 merge. Their magnitude is bounded by the largest
finite E4M3 value (448). Internally allocated partial storage uses two bytes
per element; supplied FP32 workspace is a capacity buffer whose first half is
reinterpreted for these values. Workspace contents are opaque. LSE remains
FP32, and lower split counts retain FP32 partials.

With one split, candidates write the final result directly and require no
partial workspace or merge kernel. ``variant="baseline"`` preserves the
original allocation and unconditional-merge behavior for comparison.

Implementation and validation
-----------------------------

``amd_pa_decode_fp8_optimized.py`` contains the retained candidates:

* ``matrix_tdm_flat`` uses TLX packed-page tensor descriptors, two LDS buffers,
  grouped query rows, WMMA QK/PV and a shape-specific split merge. Full-page V
  masking is peeled using uniform control flow; ragged tails remain sanitized.
* ``matrix`` uses masked TLX K staging and grouped WMMA for page sizes outside
  the flat descriptor path. Native FP8 QK is limited to FP8 Q with D64; other
  inputs use BF16 QK. This avoids the referenced gfx1250 larger-FP8-contraction
  concern. ``matrix_fp8`` is a historical alias for this current matrix policy.
* Single-pass PV uses FP16 probabilities multiplied by 32768 and FP16 raw V.
  FP32 accumulation, the FP32 softmax denominator, and inverse scaling in the
  output epilogue preserve the strict error gate without an extra PV product.
  ``matrix_precise`` retains compensated two-component BF16 PV as a control.
* ``vector`` uses TLX register-layout constraints and FP32 reductions, with
  splits over the active window. ``vector_page`` instead loads contiguous
  packed-page bytes into registers. The policy uses a one-split flat matrix path for tiny windows when the page
  layout supports it, and vector for other small-page layouts.
* ``baseline`` is the frozen vector-reduction design with TLX K/V LDS staging.

The fast paths retain separate per-KV-head K/V scales and per-query-head Q
scales. No dequantized global cache or output-projection GEMM is introduced.
Rejected descriptor layouts, deeper buffering and mixed-precision experiments
are preserved in the experiment snapshots and Markdown log.

Run correctness only::

    python -m pytest -s --tb=short third_party/tlx/tutorials/amd_pa_decode_fp8.py

Tests compare to independent dense FP32 attention on the quantized inputs,
using unequal non-unit head scales and shuffled physical pages. They cover
BF16/FP8 Q, MHA/GQA, partial and empty contexts, causal multi-query decode,
sliding windows, split reduction, 128K/256K contexts, an 8192-wide dense view,
and the T-sweep on the folded 16-head configuration. Unused tail slots contain
NaNs to expose accidental reads. These are correctness checks, not performance
measurements or evidence for every combination in the workload table.

The current optimization environment is an isolated
``rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261101`` container with Torch
2.12.0+rocm10.1.0 and this checkout's source-built TLX 3.9 runtime. The agent
exposes gfx1250, 96 CUs and 144 GiB (DPX/NPS2). Earlier correctness snapshots
used fbtriton3.7.4; cross-compiler calibration is excluded from speedup claims.

Reference adapters, numerical gates, graph timing, exact pins, ATT captures,
raw results and saved comparison charts live in
``experiments/fp8-kv-decode/``. See ``FP8_KV_DECODE_ATTN.md`` and ``REPRODUCE.md``.
A standard Triton wheel without TLX cannot execute this file.
