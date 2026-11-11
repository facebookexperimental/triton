FP8 KV decode on gfx1250 (preliminary)
====================================

``amd_pa_decode_fp8.py`` implements plain E4M3FN KV decode with per-head
scales. It is separate from the existing BF16/FP16 decode operator and does
not change production dispatch or require compiler changes. Output projection
is outside this tutorial.

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
  Selected BF16-Q single-token paths prefill both existing buffers before the
  first compute step. Later iterations resume the ordinary staggered K/V
  schedule. Short contexts, windows, FP8 Q and multi-query paths use the
  ordinary copy schedule. Dense split selection includes a parallelism floor for
  smaller KV-head grids.
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

Run correctness only::

    python -m pytest -s --tb=short third_party/tlx/tutorials/amd_pa_decode_fp8.py

Tests compare to independent dense FP32 attention on the quantized inputs,
using unequal non-unit head scales and shuffled physical pages. They cover
BF16/FP8 Q, MHA/GQA, partial and empty contexts, causal multi-query decode,
sliding windows, split reduction, 128K/256K contexts, an 8192-wide dense view,
and the T-sweep on the folded 16-head configuration. Unused tail slots contain
NaNs to expose accidental reads. These are correctness checks, not performance
measurements or evidence for every combination in the workload table.
Automatic dispatch is also checked without explicit split overrides across
additional MQA/GQA/MHA head counts, ragged batches and independent decode batches.

A standard Triton wheel without TLX cannot execute this tutorial.
