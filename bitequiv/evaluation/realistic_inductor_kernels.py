"""Realistic TorchInductor kernel corpus — diversity + reduction-equivalence gap analysis.

These are not synthetic test kernels like the ones in ``eval_kernels.py``. Every kernel here
is a **real Triton kernel that torch.compile / TorchInductor emits** — collected to answer two
questions at once:
  1. GAP ANALYSIS (the original 8, GROUP 1): "how far is the shipped reduction-equivalence
     checker from what a real customer actually ships?"
  2. DIVERSITY / BREADTH (GROUPS 2+): "across the whole ZOO of kernel TYPES a real PyTorch
     workload produces — GEMM, warp-specialized, attention, scan, cooperative reduction,
     pointwise/foreach — where does bitwise-equivalence work apply, and where is it out of
     scope (an M3 / future target)?" This makes the corpus concrete evidence of how much the
     project can contribute to real Triton kernels, not just reductions.

    ⚠️  This module is a REFERENCE corpus. It is intentionally NOT wired into ``evaluate.py``:
        the bodies are inspected, never compiled here (they hard-code shapes and reference
        Inductor helpers). Treat every kernel as "realistic customer usage → target", not as
        an in-scope requirement. Each future checker improvement can be measured against it.

How these kernels were collected (reproducible)
-----------------------------------------------
Two ways, recorded per kernel in its banner:
  * HARVEST — copied verbatim from Inductor's own in-tree tests (e.g. ``test_cuda_repro.py``,
    ``test_static_triton_launcher.py`` under ``caffe2/test/inductor/``).
  * GENERATE — emitted by running a tiny ``torch.compile`` snippet under
    ``TORCH_LOGS=output_code`` on an **H100 (sm_90)** and copying the kernel out of the
    inductor cache. GEMM used ``config.max_autotune_gemm_backends="TRITON"``; the split-scan,
    cooperative reduction, and combo/foreach kernels used the matching config knob. Generation
    RAN each kernel (so GROUP 2..7 Hopper kernels are verified runnable + numerically correct).
Faithfulness rule (as with the original 8): body copied verbatim; the trailing async-compile
wrapper stripped; the Inductor ``@triton_heuristics.<cat>(...)`` decorator replaced with a
plain ``@triton.jit`` and its autotune config kept as a ``# Autotune:`` comment in the banner.
The one exception is the Blackwell WS kernel (GROUP 3): it is TEMPLATE-DERIVED (rendered from
the Inductor jinja template) and NOT runnable on this sm_90 host — it needs sm_100/Blackwell.

GPU-architecture note (does the .py source differ per GPU?)
-----------------------------------------------------------
Mostly NO. TorchInductor emits ONE arch-agnostic ``@triton.jit`` body; the GPU shows up only as
(a) autotune configs (num_warps / num_stages / num_consumer_groups / warp_specialize /
epilogue_subtile), (b) capability guards, and (c) the wgmma-vs-tcgen05 MMA choice the compiler
makes below the source. Source-level arch divergence appears ONLY in low-level warp-specialized
kernels — hence GROUP 3 carries the one genuinely arch-specific body (Blackwell device-TMA +
``warp_specialize=True`` + tcgen05 accumulator path) next to the arch-agnostic Hopper GEMMs.

How the scope verdicts were produced
------------------------------------
Each kernel was checked against the two shipped checkers by reading their source
(``bitequiv/ptx_reduction.py`` + engine ``bitequiv/ptx/``, and ``bitequiv/ttgir_reduction.py``
+ ``lib/Analysis/ReductionOrder.cpp``), and the verdict was adversarially re-verified. Recall
what each checker does:
  * **PTX checker** — reconstructs the FP reduction TREE from PTX: roots = ``st.global``
    values, leaves = ``ld.global`` loads labelled by AFFINE load-address arithmetic; the
    num_warps/layout-invariant "collapse" (``ITreeReduce``) fires only for a SINGLE-output,
    fully-affine, BALANCED **add** tree (``_REDUCE_FP == {add}``). Everything else stays a
    verbatim, sound-but-layout-bearing tree (it over-splits, never over-merges). ``tl.dot`` /
    wgmma / tcgen05 are NOT modeled — they are fingerprinted as an opaque ``unanalyzed-mma``.
  * **TTGIR checker** — one signature per ``tt.reduce`` from ``toLinearLayout`` + the combine
    region's op-name sequence. FMA-contraction-blind; ``tt.dot``/mma get only an
    ``unanalyzed-mma`` guard key; and it NEVER walks ``tt.scan``, ``tt.load``, addresses, a
    loop-carried accumulate, or value provenance.

Inline scope tags (the "out of which checker's scope?" flag the corpus is for)
------------------------------------------------------------------------------
    [scope: PTX]     — essential syntax the PTX tree-checker cannot model / degrades on
    [scope: TTGIR]   — essential syntax the TTGIR layout-checker cannot model / degrades on
    [scope: BOTH]    — out of scope for both checkers
    [scope: neither] — handled SOUNDLY by both (flagged only to pre-empt "surely this breaks it")
severity in parentheses:
    (SOUNDNESS GAP)      — the checker can return a WRONG "equivalent" (over-merge). The scan /
                           split-scan kernels hit this. This is the dangerous class.
    (sound; over-splits) — never wrong; the checker just refuses the num_warps-invariant merge,
                           so it loses tuning freedom. The open work is PRECISION, not soundness.
    (out — M3/scan)      — deliberately not modeled by a REDUCTION checker (MMA is the M3 target;
                           scan is the scan-model target). Sound today (over-splits), no gap.

=========================================================================================
GROUP 1 — reduction fusions (the original gap-analysis 8). TTGIR / PTX / overall scope.
=========================================================================================
  KERNEL                     TTGIR   PTX     OVERALL WHY NOT FULLY SUPPORTED YET
  -------------------------  ------  ------  ------  ----------------------------------------
  A_rms_norm_fwd             in      partial partial PTX: the one reduction result fans out to 2 st.global roots, so the
                                                     num_warps-invariant collapse is refused (sound; over-splits).
  B_layernorm_welford_gather partial partial partial PTX: data-dependent GATHER load -> opaque leaf (provenance lost);
                                                     welford combine is FMA-blind on BOTH (structural only); 6 roots.
  C_rms_norm_bwd_2reduce     in      partial partial no soundness gap. PTX over-splits (3 roots + non-layout-invariant
                                                     leaves); TTGIR emits 2 positional sigs (sound).
  D_masked_global_sum        in      partial partial PTX: tl.where tail-mask -> selp INSIDE the leaf, so the maximal
                                                     balanced region will not collapse (sound; over-splits).
  E_triu_masked_rowsum       in      partial partial PTX: multi-output per-row (roots>1) -> collapse refused
                                                     (sound; over-splits).
  F_cumsum_scan              out     out     NEITHER *** SOUNDNESS GAP *** tt.scan not modeled: TTGIR emits () and may
                                                     OVER-MERGE; PTX keeps the scan verbatim/opaque. int64, not FP.
  G_plain_sum_looped         in      partial partial TTGIR: loop accumulate is not a tt.reduce (safe only via sAxis
                                                     over-split). PTX: collapse gated by _balance_pass (R0_BLOCK<128).
  H_mean_permute             in      partial partial PTX: multi-output per-row at XBLOCK>1 -> collapse refused (sound).
                                                     NB the "permute"/mean-divide live in OTHER ops, not this body.

=========================================================================================
GROUPS 2+ — diverse realistic kernel TYPES (breadth). Verdicts are per the reduction checkers.
=========================================================================================
  KERNEL                     TTGIR   PTX     OVERALL WHAT IT ADDS / WHY (NOT) IN SCOPE
  -------------------------  ------  ------  ------  ----------------------------------------
  GROUP 2 — GEMM (tl.dot; M3 MMA-equivalence target)
  GEMM_plain_mm              guard   out     out(M3) tl.dot only guarded (unanalyzed-mma) on both; no FP reduce tree.
  GEMM_addmm_bias            guard   out     out(M3) + a bias-add epilogue fused after the dot (still MMA-dominated).
  GEMM_batched_bmm           guard   out     out(M3) 3D batched dot; batch over grid.y/z. Same MMA scope story.
  GROUP 3 — warp-specialized / GPU-arch diversity
  WS_blackwell_ws_tma_mm     guard   out     out(M3) device-TMA + warp_specialize=True + tcgen05 path (sm_100 only;
                                                     TEMPLATE-DERIVED, NOT runnable here). WS invisible to TTGIR;
                                                     PTX can't follow named-barrier partitions -> over-splits.
  GROUP 4 — attention (online softmax + 2 dots)
  ATTN_flex_attention_fwd    partial partial partial the tl.max/tl.sum softmax reduces ARE keyed by TTGIR (in); the two
                                                     tl.dot are guard-only; the running rescale is a loop recurrence
                                                     TTGIR is blind to (safe only by side effect -> tag it).
  GROUP 5 — scan / cross-lane
  SCAN_split_lookback_cumsum out     out     NEITHER *** SOUNDNESS GAP *** decoupled-lookback split scan: TTGIR keys
                                                     only the block-sum reduce, blind to the scan + cross-block carry.
  GROUP 6 — cooperative / cross-CTA reduction
  COOP_xgrid_sum             in*     partial partial a real tt.reduce (in) but the cross-CTA workspace carry +
                                                     x_grid_barrier are opaque to PTX / safe-by-side-effect on TTGIR.
  GROUP 7 — pointwise / foreach (order-insensitive baseline)
  PW_reflection_pad_add      neither neither neither no reduction -> TTGIR () trivially-equal; PTX pure per-element DAG.
  PW_cat_masked              neither neither neither tl.where cat masks; still no cross-lane FP order to constrain.
  FOREACH_combo_add          neither neither neither 4 independent pointwise blocks in one combo/foreach kernel.
  GROUP 8 — non-FP reduction
  RED_any_isinf              in      partial partial boolean OR-reduce: TTGIR keys the combine; PTX won't collapse a
                                                     non-add reduce (_REDUCE_FP={add}). Sound; int, no FP order.

Big-picture takeaway
--------------------
  * GROUP 1: only ``F_cumsum_scan`` is a true SOUNDNESS gap; ``B_layernorm_welford_gather`` is
    the one PTX soundness RISK (gather -> opaque leaf); the other six are already SOUND and only
    OVER-SPLIT (the work there is PRECISION / tuning freedom, not correctness).
  * GROUPS 2+: the second SOUNDNESS gap is ``SCAN_split_lookback_cumsum`` (same tt.scan blind
    spot as F, now with a cross-block carry). GEMM/attention/WS are the M3 (MMA-equivalence)
    frontier — sound today (over-split / guarded), not yet modeled. Pointwise/foreach are the
    trivially-sound floor (no cross-lane FP order to constrain).
  * Net roadmap this corpus points at: (1) model ``tt.scan`` (F + split-scan); (2) model
    gather-fed reductions; (3) recover the num_warps-invariant collapse for multi-output /
    tail-masked / accumulate-then-reduce trees; (4) real FMA/welford numerics on TTGIR;
    (5) MMA / tensor-core equivalence for GEMM + attention + warp specialization (M3).

Nothing in this file is meant to be launched here; the bodies are kept faithful (only the
Inductor ``@triton_heuristics(...)`` decorator was replaced with a plain ``@triton.jit`` and the
trailing async-compile wrapper stripped) so they stay inspectable and can be wired into
``evaluate.py`` later if/when the checker grows to cover them.
"""

import triton
import triton.language as tl

# The Inductor bodies below reference these exactly as emitted. This module is a REFERENCE
# corpus — the kernels are never compiled here — so tolerate a missing torch/inductor at import.
try:
    from torch._inductor.runtime import triton_helpers  # welford_reduce / welford
    from torch._inductor.runtime.triton_helpers import libdevice  # rsqrt
    from torch._inductor.runtime.triton_helpers import math as tl_math  # noqa: F401 (some variants use it)
except Exception:  # noqa: BLE001 - reference-only module; bodies are not compiled at import
    triton_helpers = libdevice = tl_math = None


@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    """The plain-add combine Inductor passes to ``tl.associative_scan`` in ``F_cumsum_scan``."""
    tmp0 = arg0_0 + arg1_0
    return tmp0


# #########################################################################################
# ## GROUP 1 — REDUCTION FUSIONS (the original gap-analysis 8: A..H).                     ##
# ## Verbatim TorchInductor reduction fusions. This is the reduction-equivalence checker's ##
# ## home turf; see the GROUP 1 table in the module docstring for per-kernel verdicts.    ##
# #########################################################################################
# ========================================================================================= #
# A. RMS-norm forward  (source: triton_per_fused__fused_rms_norm__to_copy_mul_3)
# -----------------------------------------------------------------------------------------
# What it computes: rstd = rsqrt(mean(x^2)+eps); out = x*rstd*weight. The core of every
# transformer block in this model. ONE pure add-reduction of x*x; the rest is pointwise.
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.
# Novelty vs suite: the reduction tree itself overlaps `dot` (x*y) / `sum_dim1_simple`. The
# genuine (PTX-only) novelty: a single-output-collapsible add-tree that LOSES the num_warps-
# invariant collapse purely because the one reduction result fans out into TWO stores.
# ========================================================================================= #
@triton.jit
def A_rms_norm_fwd(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1024
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256 * x0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    # [scope: neither] in-reduction square: PTX sees mul(L,L), which IS layout-invariant; TTGIR
    # does not see it (it lives outside the tt.reduce combine) and defers FMA/mul to the PTX sibling.
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)  # the one add-reduction (TTGIR: in scope)
    tmp7 = tl.full([1, 1], 256.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-05, tl.float32)
    tmp10 = tmp8 + tmp9
    # [scope: neither] rsqrt + affine epilogue: not part of the reduction tree; TTGIR never walks
    # it, PTX walks THROUGH it (div/add as FpOp, rsqrt as opaque pass-through above the root).
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    # [scope: PTX] (sound; over-splits) the SAME reduction result reaches TWO st.global roots
    # (rstd scalar + normalized tensor) -> len(roots)!=1 -> the num_warps-invariant ITreeReduce
    # collapse is refused. The tree stays verbatim/layout-bearing (correct, just no merge).
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr0 + (r0_1 + 256 * x0), tmp15, xmask)


# ========================================================================================= #
# B. LayerNorm forward over a GATHERED input + sigmoid gate
#    (inductor: triton_red_fused_index_select_mul_native_layer_norm_sigmoid_32)
# -----------------------------------------------------------------------------------------
# What it computes: welford mean/var of an index_select-GATHERED row, then rsqrt, then a
# second pass that re-gathers, normalizes, affine-scales and sigmoid-gates. This is the most
# complex kernel in the corpus (two-pass, welford, data-dependent gather).
# Verified scope:  TTGIR = partial,  PTX = partial,  overall = partial.
# Novelty vs suite: the suite's `welford` reduces a DIRECT affine load; this reduces a GATHER,
# fusing two documented gaps at once (welford FMA-blindness + non-affine gather leaf).
# ========================================================================================= #
@triton.jit
def B_layernorm_welford_gather(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
                               out_ptr4, ks0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')  # the gather INDEX (data-dependent)
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = (ks0).to(tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp4 < ks0")
        # [scope: PTX] (SOUNDNESS RISK) GATHER: the address term `128*tmp4` traces back to a
        # ld.global value, which AffineEval cannot model -> the leaf address is opaque, its
        # coordinate/column-image is lost. The reduction leaf cannot be mapped to a static input
        # element -> verbatim compare + the documented cross-module opaque-leaf collision risk.
        # (TTGIR is unaffected: it never inspects load addresses.)
        tmp6 = tl.load(in_ptr1 + (r0_1 + 128 * tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        # [scope: BOTH] (sound; over-splits) welford combine holds multiplies/divides. TTGIR keys
        # only on the combine op-NAME sequence (FMA/contraction-blind -> structural match only);
        # PTX's collapse is add-only, so welford is never the balanced add-tree it can merge.
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0)
        tmp8_mean = tl.where(r0_mask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask & xmask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)  # cross-lane welford reduce
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp14 = tl.full([1, 1], 128.0, tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tl.full([1, 1], 1e-05, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(in_out_ptr0 + (x0), tmp18, xmask)
    # [scope: neither] second pass = pure pointwise + sigmoid EPILOGUE (no reduction here). Note
    # this pass adds 4 more st.global roots; together with the 2 above that is 6 roots, which
    # would block the num_warps-invariant collapse anyway (single-output-only).
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = (ks0).to(tl.int32)
        tmp20 = tmp0 + tmp19
        tmp21 = tmp0 < 0
        tmp22 = tl.where(tmp21, tmp20, tmp0)
        tl.device_assert(((0 <= tmp22) & (tmp22 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp22 < ks0")
        tmp24 = tl.load(in_ptr1 + (r0_1 + 128 * tmp22), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tmp24 - tmp8
        tmp26 = tmp25 * tmp18
        tmp28 = tmp26 * tmp27
        tmp30 = tmp28 + tmp29
        tmp31 = tl.sigmoid(tmp30)
        tmp32 = tmp24 * tmp31
        tl.store(out_ptr1 + (r0_1 + 128 * x0), tmp30, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 128 * x0), tmp31, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_1 + 128 * x0), tmp24, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 128 * x0), tmp32, r0_mask & xmask)


# ========================================================================================= #
# C. RMS-norm BACKWARD, two fused reductions  (source:
#    triton_per_fused__fused_rms_norm__fused_rms_norm_backward__to_copy_add_div_expand_mul_neg_
#    sigmoid_sigmoid_backward_sub_sum_33)
# -----------------------------------------------------------------------------------------
# What it computes: the RMS-norm backward pass. TWO add-reductions in one kernel where the
# SECOND reduction's input depends on the FIRST reduction's result (a reduce-of-reduce), wrapped
# in a full sigmoid_backward pointwise chain. This is the single biggest family in the report.
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.  NO soundness gap.
# Novelty vs suite: first kernel with a genuine reduce-of-reduce data dependency (welford has 2
# outputs from ONE reduce; cond_reduce's 3 sums are independent). Both checkers stay sound; PTX
# just over-splits (3 roots + non-layout-invariant leaves: sigmoid in reduce 1, the shfl/smem
# partial from reduce 1 inside reduce 2). TTGIR emits 2 independent positional signatures.
# ========================================================================================= #
@triton.jit
def C_rms_norm_bwd_2reduce(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr5, xnumel,
                           r0_numel, XBLOCK: tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256 * x0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (r0_1 + 256 * x0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp1 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp1
    tmp16 = tl.full([1, 1], 1.0, tl.float32)
    tmp17 = tmp16 - tmp10
    tmp18 = tmp10 * tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19 * tmp6
    tmp21 = tmp5 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    # [scope: neither] REDUCTION 1 (add-tree). Handled soundly by both. Its leaf subtree carries a
    # sigmoid (opaque, not pure-elementwise), so PTX cannot layout-invariant-collapse it anyway.
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tl.full([1, 1], 0.00390625, tl.float32)
    tmp27 = tmp5 * tmp26
    tmp28 = tmp27 * tmp25  # <-- reduce-of-reduce: reduction 2's input depends on reduction 1 (tmp25)
    tmp29 = tmp20 - tmp28
    tmp30 = tmp29 * tmp4
    tmp31 = -tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    # [scope: neither] REDUCTION 2, fed by reduction 1. TTGIR gives it an INDEPENDENT positional
    # signature (the def-use link to reduce 1 is not tracked, which is conservative/sound). PTX
    # reconstructs the combined DAG soundly but over-splits (reduce-1's shfl/smem partial sits
    # inside reduce-2's leaves -> not layout-invariant). No wrong-merge risk either way.
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = tmp14 * tmp10
    tmp37 = tmp36 + tmp30
    tmp38 = tmp35 * tmp26
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 256 * x0), tmp10, xmask)
    tl.store(out_ptr1 + (r0_1 + 256 * x0), tmp12, xmask)
    tl.store(out_ptr5 + (r0_1 + 256 * x0), tmp40, xmask)


# ========================================================================================= #
# D. Masked global sum to a scalar  (source: triton_per_fused__to_copy_mse_loss_view_56)
# -----------------------------------------------------------------------------------------
# What it computes: sum a length-608 vector to one scalar (the tail of an MSE-loss reduction).
# ONE add-reduction; the twist is a NON-power-of-two extent (608) zero-padded into a 1024 tile.
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.
# Novelty vs suite: first kernel where a `tl.where` tail-mask sits INSIDE the reduction leaf
# subtree; empirically it lowers to `selp` and gives 4 distinct PTX descriptors across num_warps.
# ========================================================================================= #
@triton.jit
def D_masked_global_sum(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1
    r0_numel = 608
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)  # affine, in scope
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    # [scope: PTX] (sound; over-splits) this tail-mask (608 live of a 1024 tile) lowers to `selp`
    # INSIDE the reduction's boundary leaves; `selp` is not pure-elementwise, so the maximal
    # balanced region containing it will not collapse -> the reduction stays layout-bearing (a
    # smaller pure sub-region still collapses). Sound, but not num_warps-invariant.
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)  # the one add-reduction (TTGIR: in scope)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)


# ========================================================================================= #
# E. Masked per-row sum, odd extent  (source: triton_per_fused_expand_mul_select_sum_triu_view_1)
# -----------------------------------------------------------------------------------------
# What it computes: a per-row sum over an ODD live extent (235) padded to 256 (the reduce tail of
# a triangular/causal-mask op; the triu/select live in other pointwise kernels).
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.
# Novelty vs suite: odd non-power-of-two live extent with identity masking, in a per-row
# (multi-output) single-tile reduce — the pure multi-output-collapse-guard case (no loop fence).
# ========================================================================================= #
@triton.jit
def E_triu_masked_rowsum(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 592
    r0_numel = 235
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 235 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    # [scope: neither] identity-pad of the odd extent 235 into the 256-wide axis (masked lanes add
    # 0.0). Both checkers handle this: TTGIR ignores masks by design, PTX rides it as a per-leaf flag.
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)  # the one add-reduction (TTGIR: in scope)
    # [scope: PTX] (sound; over-splits) per-row store => one st.global root PER row (roots>1 when
    # XBLOCK>1) => the num_warps-invariant collapse is refused (the multi-output G3 guard).
    tl.store(out_ptr0 + (x0), tmp4, xmask)


# ========================================================================================= #
# F. Cumulative sum via a SCAN  (source: triton_per_fused_cumsum_ge_scalar_tensor_sub_where_4)
# -----------------------------------------------------------------------------------------
# What it computes: a prefix/cumulative sum (int64) then a pointwise int epilogue. The only
# cross-lane op is `tl.associative_scan`, which lowers to `tt.scan` — NOT `tt.reduce`.
# Verified scope:  TTGIR = out,  PTX = out,  overall = NEITHER.  *** THE ONE SOUNDNESS GAP ***
# Novelty vs suite: the first pure-scan kernel — the suite is entirely tt.reduce-based. This is
# the clean boundary case: it marks exactly where "reduction equivalence" stops.
# ========================================================================================= #
@triton.jit
def F_cumsum_scan(in_out_ptr0, in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_first')
    # [scope: BOTH] int64 accumulation: both checkers are FP-order tools; an integer add is not
    # an FP reduce node in PTX and would only be an int op-name in TTGIR's combine key.
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    # [scope: BOTH] (SOUNDNESS GAP) tt.scan is NOT modeled. TTGIR never walks triton::ScanOp, so
    # the descriptor is () -> two kernels that differ only in a scan can be wrongly called EQUAL
    # (an over-merge). PTX has no prefix-scan model: the scan's cross-lane shfl/carry falls to
    # opaque nodes matched only byte-for-byte. This kernel's whole cross-lane story is invisible.
    tmp3, = tl.associative_scan((tmp2, ), 1, _triton_helper_fn_add0)
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = (1 + ks0 + ((-1) * ks1)).to(tl.int64)
    tmp9 = (tmp8).to(tl.int64)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp7, None)
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp10, None)


# ========================================================================================= #
# G. Looped bias-gradient sum  (source: triton_red_fused_sum_60)
# -----------------------------------------------------------------------------------------
# What it computes: accumulate each R0_BLOCK-wide chunk into a running [XBLOCK, R0_BLOCK] buffer
# across an r0 loop, THEN tree-reduce the buffer once (accumulate-then-reduce). A common
# bias/grad reduction. r0_numel is hard-coded 128, so R0_BLOCK==128 => loop runs once.
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.
# Novelty vs suite: `sum_dim1_persistent` does reduce-THEN-accumulate; this does accumulate-THEN-
# reduce (a different composition), and the R0_BLOCK==128 boundary flips PTX collapse on/off.
# ========================================================================================= #
@triton.jit
def G_plain_sum_looped(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 5760
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 128
    x0 = (xindex % 128)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = (r0_2 + 128 * x1).to(tl.int32)
        tmp1 = tl.full([1, 1], 5744, tl.int32)
        tmp2 = tmp0 < tmp1  # [scope: neither] boundary MASK (a predicate, not an address) — handled
        tmp3 = tl.load(in_ptr0 + (x0 + 128 * r0_2 + 16384 * x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first',
                       other=0.0).to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        # [scope: TTGIR] the cross-chunk running accumulate (arith.addf + select) is NOT a
        # tt.reduce, so the TTGIR checker is blind to its order (it stays safe only by side effect:
        # different R0_BLOCK changes sAxis on the final reduce and over-splits).
        # [scope: PTX] (sound; over-splits) this loop-carried add is a left-fold chain; _balance_pass
        # rejects it as unbalanced, so no ITreeReduce collapse when R0_BLOCK<128 (fence-guarded via
        # loops=). At R0_BLOCK==128 the loop runs once and the final tree CAN collapse.
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]  # [scope: neither] the final tree-reduce; in scope for both
    tl.store(out_ptr0 + (x3), tmp5, xmask)


# ========================================================================================= #
# H. Per-row mean over a contiguous axis  (source: triton_per_fused__to_copy_mean_permute_70)
# -----------------------------------------------------------------------------------------
# What it computes: a per-row sum over a CONTIGUOUS reduce axis. NB: despite the name, the
# "permute" and the mean-divide are SEPARATE upstream/downstream ops — this body is a plain sum.
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial.
# Novelty vs suite: a real compiler-emitted per-row reduce where XBLOCK is an autotunable
# constexpr, so it straddles the single-output vs multi-output boundary of the PTX collapse.
# ========================================================================================= #
@triton.jit
def H_mean_permute(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256 * x0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)  # the one add-reduction (TTGIR: in scope)
    # [scope: PTX] (sound; over-splits) per-row store => multiple st.global roots when XBLOCK>1 =>
    # collapse refused. At the tuned XBLOCK==1 it is single-output and the collapse CAN fire.
    tl.store(out_ptr0 + (x0), tmp5, xmask)


# #########################################################################################
# ## GROUP 2 - GEMM / MATMUL (tl.dot templates; M3 MMA-equivalence target).
# ## Real triton_tem_fused_* templates torch.compile emits for a @ b / addmm / bmm on H100
# ## (config.max_autotune_gemm_backends='TRITON' so a Triton kernel wins over cuBLAS).
# ## Generated + RAN on this H100 (numerically exact vs eager). tl.dot is the M3 frontier:
# ## neither reduction checker models the tensor-core contraction (see per-line tags).
# #########################################################################################

# ========================================================================================= #
# GEMM-1. Plain tiled matmul  (source: triton_tem_fused_mm_0 | repro: (a@b), fp16 1024x512@512x1024, H100)
# Kernel type: TEMPLATE / GEMM (grouped-L2 swizzle, K-loop tl.dot accumulate).
# Autotune: num_stages=4, num_warps=4, BLOCK_M=64 BLOCK_N=128 BLOCK_K=128 GROUP_M=8 ALLOW_TF32=False.
# What it computes: C = A @ B, the canonical tiled GEMM every transformer FFN/attention proj hits.
# Verified scope:  TTGIR = guard-only,  PTX = out,  overall = out (M3 MMA target).
# Novelty vs suite: the first tl.dot kernel - the whole GROUP 1 is tt.reduce-only. Marks where
# reduction equivalence ends and MMA equivalence (M3) begins.
# ========================================================================================= #
@triton.jit
def GEMM_plain_mm(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    OUT_DTYPE : tl.constexpr = tl.float16
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1024
    N = 1024
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1024
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + (xindex))


        # [scope: BOTH] (out - M3) the K-loop tensor-core contraction. Neither reduction checker
        # models tl.dot: PTX fingerprints it as an opaque unanalyzed-mma (the accumulator becomes an
        # opaque node), TTGIR emits only the unanalyzed-mma guard key. Sound today (over-splits); the
        # MMA-equivalence M3 target.
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


# ========================================================================================= #
# GEMM-2. Matmul + bias epilogue  (source: triton_tem_fused_addmm_0 | repro: addmm(bias,a,b), H100)
# Kernel type: TEMPLATE / GEMM with a fused bias-add epilogue.
# Autotune: num_stages=5, num_warps=4, BLOCK_M=64 BLOCK_N=32 BLOCK_K=128 GROUP_M=8 ALLOW_TF32=False.
# What it computes: C = bias + A @ B (nn.Linear). Shows a pointwise epilogue fused onto the dot.
# Verified scope:  TTGIR = guard-only,  PTX = out,  overall = out (M3 MMA target).
# Novelty vs suite: MMA followed by a fused elementwise epilogue - the epilogue is order-free
# (neither checker constrains it), the dot is the out-of-scope part.
# ========================================================================================= #
@triton.jit
def GEMM_addmm_bias(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    OUT_DTYPE : tl.constexpr = tl.float16
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 512
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 512
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + (xindex))


        # [scope: BOTH] (out - M3) the K-loop tensor-core contraction. Neither reduction checker
        # models tl.dot: PTX fingerprints it as an opaque unanalyzed-mma (the accumulator becomes an
        # opaque node), TTGIR emits only the unanalyzed-mma guard key. Sound today (over-splits); the
        # MMA-equivalence M3 target.
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    # [scope: neither] the fused bias-add epilogue (a pure per-element add after the dot) is
    # not a reduction; it rides above the unanalyzed-mma root and neither checker constrains it.
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


# ========================================================================================= #
# GEMM-3. Batched matmul  (source: triton_tem_fused_bmm_0 | repro: bmm(8x512x256, 8x256x512), H100)
# Kernel type: TEMPLATE / batched GEMM (batch mapped onto grid.y/z; pointer-walk K-loop).
# Autotune: num_stages=3, num_warps=4, BLOCK_M=64 BLOCK_N=128 BLOCK_K=64 GROUP_M=8 BATCH=8.
# What it computes: C[b] = A[b] @ B[b] for each batch b (attention scores / batched projections).
# Verified scope:  TTGIR = guard-only,  PTX = out,  overall = out (M3 MMA target).
# Novelty vs suite: adds a 3D/batched shape - a different tiling (batch over grid.y/z) around the
# same unanalyzed-mma dot; broadens the shape diversity of the corpus.
# ========================================================================================= #
@triton.jit
def GEMM_batched_bmm(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    OUT_DTYPE : tl.constexpr = tl.float16
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 512
    K = 256
    BATCH = 8

    stride_aq = 131072
    stride_am = 256
    stride_ak = 1

    stride_bq = 131072
    stride_bk = 512
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)

    # Reconstruct batch index from grid_y/grid_z split (handles batch > 65535)
    # [scope: neither] the batch index rides grid.y/z (batch can exceed 65535); it changes which
    # matrices are multiplied, not the intra-dot FP order. Same MMA scope story as plain mm.
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    # Clamp to valid range for safe pointer arithmetic; out-of-bounds CTAs are
    # masked off at the store below.
    idx_q_clamped = tl.minimum(idx_q, BATCH - 1)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q_clamped*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q_clamped*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        # [scope: BOTH] (out - M3) the K-loop tensor-core contraction. Neither reduction checker
        # models tl.dot: PTX fingerprints it as an opaque unanalyzed-mma (the accumulator becomes an
        # opaque node), TTGIR emits only the unanalyzed-mma guard key. Sound today (over-splits); the
        # MMA-equivalence M3 target.
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N) & (idx_q < BATCH)

    # cast accumulator to output dtype
    acc = acc.to(OUT_DTYPE)
    # inductor generates a suffix
    xindex = idx_n + 512*idx_m + 262144*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


# #########################################################################################
# ## GROUP 3 - WARP-SPECIALIZED / GPU-ARCH DIVERSITY.
# ## The one genuinely arch-specific body: source-level Hopper-vs-Blackwell divergence only shows
# ## up in low-level warp-specialized kernels. This is TEMPLATE-DERIVED from the Inductor jinja
# ## triton_blackwell_ws_persistent_device_tma_mm.py.jinja, rendered with representative sm_100
# ## constexprs. It is NOT verbatim Inductor output and NOT runnable on this sm_90 host (needs
# ## sm_100 / Blackwell for tcgen05 + device TMA); the verify harness SKIPS it.
# #########################################################################################

# ========================================================================================= #
# WS-1. Blackwell warp-specialized persistent device-TMA matmul  (template-derived; sm_100 only)
# Kernel type: TEMPLATE / GEMM + persistent (grid==NUM_SMS) + device TMA + warp specialization.
# Autotune: WARP_SPECIALIZE=True FLATTEN=True EPILOGUE_SUBTILE=2 NUM_SMS=148 BLOCK_M/N=128 BLOCK_K=64.
# What it computes: C = A @ B as a persistent tile loop; TMA bulk-loads A/B tiles, tcgen05 MMA
# accumulates in TMEM, warp_specialize splits the loop into async producer/consumer partitions.
# Verified scope:  TTGIR = guard-only,  PTX = out,  overall = out (M3 MMA + warp-spec target).
# Novelty vs suite: the ONLY kernel whose .py source is arch-specific. Shows the two axes of GPU
# diversity: warp_specialize=True + device-TMA descriptors + the tcgen05/TMEM accumulator path
# (vs Hopper's arch-agnostic register-accumulator GEMM in GROUP 2).
# ========================================================================================= #
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, grid_m, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % GROUP_M)
    pid_n = (tile_id % num_pid_in_group) // GROUP_M
    return pid_m, pid_n


@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    tl.static_assert(SUBTILE_FACTOR > 0, "SUBTILE_FACTOR must be positive")
    tl.static_assert((SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0, "SUBTILE_FACTOR must be a power of 2")
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        left_subtiles = _subtile_accumulator(left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2)
        right_subtiles = _subtile_accumulator(right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2)
        return left_subtiles + right_subtiles


@triton.jit
def WS_blackwell_ws_tma_mm(A, B, out_ptr0):
    # Constexprs the Inductor Blackwell WS template heuristic injects (representative sm_100 config).
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8
    NUM_SMS: tl.constexpr = 148
    WARP_SPECIALIZE: tl.constexpr = True
    FLATTEN: tl.constexpr = True
    EPILOGUE_SUBTILE: tl.constexpr = 2
    A_ROW_MAJOR: tl.constexpr = True
    B_ROW_MAJOR: tl.constexpr = True
    ALLOW_TF32: tl.constexpr = True
    M = 4096
    N = 4096
    K = 4096
    if M * N == 0:
        return
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n
    stride_am = 4096
    stride_ak = 1
    stride_bk = 4096
    stride_bn = 1
    # [scope: BOTH] (out - M3) device-side TMA descriptors: the reduction checkers do not
    # model make_tensor_descriptor / load_tensor_descriptor. PTX sees the TMA loads as opaque
    # leaves; TTGIR never inspects loads. TMA is a Hopper+/Blackwell hardware bulk-copy unit.
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[stride_am, 1] if A_ROW_MAJOR else [stride_ak, 1],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[stride_bk, 1] if B_ROW_MAJOR else [stride_bn, 1],
        block_shape=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
    )
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_M * grid_n
    # [scope: PTX] (sound; over-splits) warp_specialize=True partitions the loop body into
    # producer/consumer async partitions behind named-barrier sync. The flat PTX reader cannot
    # follow the partition/barrier control flow -> extra opaque nodes -> collapse refused.
    # [scope: TTGIR] warp specialization is invisible unless it changes a tt.reduce operand's
    # layout; here there is no tt.reduce at all, only tt.dot (guarded), so it is a no-op key.
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, grid_m, GROUP_M, NUM_SMS)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_am_desc = offs_am.to(tl.int32)
        offs_bn_desc = offs_bn.to(tl.int32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            offs_k_desc = offs_k.to(tl.int32)
            a = tl.load_tensor_descriptor(
                a_desc,
                [offs_am_desc, offs_k_desc]
                if A_ROW_MAJOR
                else [offs_k_desc, offs_am_desc],
            )
            b = tl.load_tensor_descriptor(
                b_desc,
                [offs_k_desc, offs_bn_desc]
                if B_ROW_MAJOR
                else [offs_bn_desc, offs_k_desc],
            )
            # [scope: BOTH] (out - M3) the K-axis tensor-core contraction. On sm_100 this lowers
            # to a tcgen05 MMA with the accumulator living in TMEM; neither checker models it
            # (PTX fingerprints it unanalyzed-mma; TTGIR emits only the unanalyzed-mma guard).
            accumulator += tl.dot(
                a if A_ROW_MAJOR else a.T,
                b if B_ROW_MAJOR else b.T,
                allow_tf32=ALLOW_TF32,
            )
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, grid_m, GROUP_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_M
        offs_cn = pid_n * BLOCK_N
        subtiles = _subtile_accumulator(accumulator, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
        for i in tl.static_range(EPILOGUE_SUBTILE):
            subtile = subtiles[i]
            offs_cn_i = offs_cn + i * (BLOCK_N // EPILOGUE_SUBTILE)
            # store_output rendered as a masked global store of this subtile.
            idx_m = offs_cm + tl.arange(0, BLOCK_M)[:, None]
            idx_n = offs_cn_i + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILE)[None, :]
            mask = (idx_m < M) & (idx_n < N)
            tl.store(out_ptr0 + (idx_n + N * idx_m), subtile.to(tl.float16), mask)


# #########################################################################################
# ## GROUP 4 - ATTENTION (online softmax + two matmuls).
# ## A complete FlexAttention forward torch.compile emits (flex_attention(q,k,v)); generated + RAN
# ## on this H100. Kept verbatim in full (main kernel + its 6 @triton.jit helpers) so it is a
# ## faithful, self-contained flash-attention. Most of the body is Inductor plumbing; the
# ## load-bearing cross-lane ops are tagged inside forward_block_mn (two tl.dot + the tl.max /
# ## tl.sum softmax reduces + the online-softmax rescale recurrence).
# #########################################################################################

# ========================================================================================= #
# ATTN-1. FlexAttention forward  (source: triton_tem_fused__to_copy_flex_attention_ones_slice_... | repro: flex_attention 2x4x512x64, H100)
# Kernel type: TEMPLATE / attention = 2 tl.dot (q@k^T, p@v) + online-softmax (tl.max, tl.sum).
# Autotune: num_stages=3, num_warps=4, BLOCK_M=64 BLOCK_N=128 SM_SCALE=0.125 (USE_TMA=False here).
# What it computes: O = softmax(scale * Q @ K^T) @ V with a running (m_i,l_i,acc) online softmax.
# Verified scope:  TTGIR = partial,  PTX = partial,  overall = partial (the two dots are out/M3).
# Novelty vs suite: the first kernel that MIXES reductions and MMA. The softmax tl.max/tl.sum ARE
# reduction nodes the checkers see; the two dots are unanalyzed-mma; and the acc/l_i rescale is a
# loop recurrence TTGIR is blind to (safe only by side effect - flagged as a soundness caution).
# ========================================================================================= #
@triton.jit
def ATTN_flex_attention_fwd(arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    MAX = arg_MAX
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = 131072, 32768, 64, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 131072, 32768, 64, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 131072, 32768, 64, 1

    ZQ = 2
    HQ = 4
    Q_LEN = 512
    ZKV = 2
    KV_LEN = 512

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0).to(INDEX_DTYPE)
    off_zq = tl.program_id(1).to(INDEX_DTYPE)
    off_hq = tl.program_id(2).to(INDEX_DTYPE)

    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    # Setting up the TMA descriptors for Q, K, V
    desc_q = None
    desc_k = None
    desc_v = None

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = 1
    stride_kv_idx_h = 1
    stride_kv_idx_m = 1

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950
    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    q = load_checked_2d(Q, offs_m, offs_k, stride_qm, stride_qk, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)

    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))


    # K and V pointers will be passed directly to forward_inner

    offs_n = kv_start + tl.arange(0, BLOCK_N)


    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, K, V,
        desc_k, desc_v, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_start,
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        stride_kk, stride_kn, stride_vn, stride_vk,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_full_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_full_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        # K and V pointers will be passed directly to forward_inner
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
            q, K, V,
            desc_k, desc_v, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_start,
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            stride_kk, stride_kn, stride_vn, stride_vk,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1).to(INDEX_DTYPE)
    idx_hq = tl.program_id(2).to(INDEX_DTYPE)
    idx_m = offs_m[:, None].to(INDEX_DTYPE)
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :].to(INDEX_DTYPE)

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    tl.static_assert(acc.shape == [BLOCK_M, V_HEAD_DIM_ROUNDED])
    xindex = idx_d + 64*idx_m + 32768*idx_hq + 131072*idx_zq
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, V_HEAD_DIM_ROUNDED])), acc, mask)

    if OUTPUT_LOGSUMEXP:
        off_hz = off_zq * HQ + off_hq
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)

    if OUTPUT_MAX:
        off_hz = off_zq * HQ + off_hq
        max_ptrs = MAX + off_hz * Q_LEN + offs_m
        if IS_DIVISIBLE:
            tl.store(max_ptrs, m_i)
        else:
            tl.store(max_ptrs, m_i, mask=offs_m < Q_LEN)


# Utility triton funcs
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_LEN: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)


# Common Imports
@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    kv_offset,
    MATMUL_PRECISION, RCP_LN2,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824
    INDEX_DTYPE : tl.constexpr = tl.int32


    # -- load k --
    # NB reversed order to since K is transposed
    kv_base_offset = kv_start + kv_offset

    # Load K as [BLOCK_N, QK_HEAD_DIM_ROUNDED] then transpose to [QK_HEAD_DIM_ROUNDED, BLOCK_N]
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_n_load = kv_base_offset + tl.arange(0, BLOCK_N)
    k = load_checked_2d(K, offs_n_load, offs_k, stride_kn, stride_kk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)

    k = tl.trans(k)
    k = k.to(q.dtype)
    # -- compute qk ---
    # [scope: BOTH] (out - M3) first attention matmul q @ k^T; unanalyzed-mma on both checkers.
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = tl.full([1], True, tl.int1)
        mask_mod_output = tmp1


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    # [scope: TTGIR] (in) the softmax max-reduce tl.max(...,1) IS a tt.reduce -> TTGIR keys it
    # (combine=maximum). [scope: PTX] max is in _FP_KINDS but _REDUCE_FP={add} only, so PTX never
    # collapses a max reduction -> it stays a verbatim layout-bearing tree (sound; over-splits).
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    # [scope: TTGIR] (in) tl.sum(p,1) is a keyed add tt.reduce, but folding it into the running
    # l_i*alpha denominator makes the RECURRENCE itself not-a-tt.reduce -> invisible to TTGIR.
    # [scope: PTX] loop-carried fma acc-chain -> not the balanced add-tree it can collapse.
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    # [scope: TTGIR] (soundness caution) online-softmax rescale of the accumulator: a loop
    # recurrence TTGIR does not model; sound here only because the max/sum reduces it DOES see
    # change their layout keys per config (same safe-by-side-effect story as G_plain_sum_looped).
    acc = acc * alpha[:, None]
    # Calculate offsets for V loading - reuse kv_base_offset from K loading
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
    v = load_checked_2d(V, offs_n_load, offs_v, stride_vn, stride_vk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
    # [scope: BOTH] (out - M3) second attention matmul p @ v accumulated into acc; unanalyzed-mma.
    acc = tl.dot(p.to(MATMUL_PRECISION), v.to(q.dtype), acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K, V,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824
    INDEX_DTYPE : tl.constexpr = tl.int32


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )


        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        kv_offset += offset


    return acc, l_i, m_i


# #########################################################################################
# ## GROUP 5 - SCAN / CROSS-LANE (the second soundness gap).
# ## A decoupled-lookback SPLIT scan torch.compile emits for a long-axis torch.cumsum; generated +
# ## RAN on this H100. Reuses _triton_helper_fn_add0 (defined for F_cumsum_scan above). This is the
# ## harder cousin of F: it adds a real per-block tt.reduce AND a cross-block atomic carry, so the
# ## TTGIR checker keys the harmless block-sum while staying blind to the parts that actually differ.
# #########################################################################################

# ========================================================================================= #
# SCAN-1. Decoupled-lookback split scan  (source: triton_spl_fused_cumsum_0 | repro: cumsum(8x40000), H100)
# Kernel type: SPLIT_SCAN = per-block tl.reduce + tl.associative_scan + cross-block lookback carry.
# Autotune: split_scan grid; R0_BLOCK autotuned; workspace ws_ptr holds the decoupled-lookback state.
# What it computes: prefix sum over a reduction axis too long for one block, via Merrill-Garland
# decoupled lookback (each block sums, publishes, then looks back over peers' partials).
# Verified scope:  TTGIR = out,  PTX = out,  overall = NEITHER.  *** SECOND SOUNDNESS GAP ***
# Novelty vs suite: extends F_cumsum_scan (single-block scan) with a CROSS-BLOCK carry. TTGIR keys
# only the block-sum reduce and is blind to both the scan and the lookback carry -> over-merge risk.
# ========================================================================================= #
@triton.jit
def SCAN_split_lookback_cumsum(in_ptr0, out_ptr0, ws_ptr, xnumel, r0_numel, R0_BLOCK : tl.constexpr):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    r0_numel = 40000
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(1) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    r0_offset = tl.program_id(0) * R0_BLOCK
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 40000*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.num_programs(0)
    tmp2 = ws_ptr.to(tl.pointer_type(tl.uint64)) + xoffset * 1 * tmp1
    tmp3 = tmp0.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    # [scope: neither] the per-block partial SUM is a real tt.reduce -> TTGIR keys THIS (add). It
    # is the ONLY part of the split-scan the reduction checker can actually see.
    tmp5 = tl.reduce(tmp4, 0, _triton_helper_fn_add0)
    # [scope: BOTH] (SOUNDNESS GAP) the decoupled-lookback CROSS-BLOCK carry (atomic workspace +
    # lookback over peer partials). TTGIR never walks it; PTX sees only opaque atomics -> two
    # split-scans that differ in carry order can go undetected (over-merge risk).
    tmp6 = triton_helpers.exclusive_scan_decoupled_lookback(
        tmp2,
        tmp5,
        tl.program_id(0),
        _triton_helper_fn_add0,
        DTYPE_VALUE_AS_UINT=tl.uint32,
        DTYPE_PACK=tl.uint64,
    )
    # [scope: BOTH] (SOUNDNESS GAP) tt.scan is not modeled (same blind spot as F_cumsum_scan):
    # TTGIR emits () for the scan; PTX keeps the prefix shfl/carry opaque. The prefix-sum order
    # is invisible, so two differing scans can be wrongly called equal.
    tmp7 = tl.associative_scan(tmp4, 0, _triton_helper_fn_add0)
    tmp8 = _triton_helper_fn_add0(tmp6, tmp7)
    tmp9 = tl.where(roffset == 0, tmp7, tmp8)
    tl.store(out_ptr0 + (r0_1 + 40000*x0), tmp9, r0_mask)


# #########################################################################################
# ## GROUP 6 - COOPERATIVE / CROSS-CTA REDUCTION.
# ## A cooperative (cross-CTA) reduction torch.compile emits for a sum over a very long axis with few
# ## rows (config.triton.force_cooperative_reductions); generated + RAN on this H100. The reduction is
# ## SPLIT across CTAs (RSPLIT), each does a local tree-reduce, then a grid barrier + shared workspace
# ## combine the per-CTA partials - a reduction ordering story that spans thread blocks.
# #########################################################################################

# ========================================================================================= #
# COOP-1. Cross-CTA cooperative sum  (source: triton_unk_fused_sum_0 | repro: x.sum(-1) 8x131072, H100)
# Kernel type: cooperative_reduction = per-CTA loop accumulate + tl.sum, then x_grid_barrier + peer sum.
# Autotune: launch_cooperative_grid=True; RSPLIT (CTAs per reduction) + R0_BLOCK autotuned; sem/ws ptrs.
# What it computes: out[x] = sum over a 131072-long axis, split across RSPLIT CTAs that combine via a
# grid barrier + a shared workspace (a reduction whose tree spans multiple thread blocks).
# Verified scope:  TTGIR = in (per tt.reduce),  PTX = partial,  overall = partial.
# Novelty vs suite: the first CROSS-CTA reduction - GROUP 1 is all single-CTA. The intra-CTA tl.sum
# is keyed, but the cross-CTA carry (barrier + workspace) is opaque to PTX / safe-by-side-effect on TTGIR.
# ========================================================================================= #
@triton.jit
def COOP_xgrid_sum(in_ptr0, out_ptr0, sem_ptr, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr, RSPLIT : tl.constexpr):
    xnumel = 8
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
    RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
    HAS_RSPLIT: tl.constexpr = RSPLIT > 1
    rsplit_id = tl.program_id(0)
    num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
    rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
    rsplit_start = rsplit_chunk * rsplit_id
    rsplit_end = rsplit_chunk * (rsplit_id + 1)
    rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    rsplit_arange = tl.arange(0, RSPLIT_NEXT_POWER_OF_2)[None, :]
    rsplit_mask = xmask if RSPLIT_IS_POWER_OF_2 else ((rsplit_arange < RSPLIT) & xmask)
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(rsplit_start, rsplit_end, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        # [scope: PTX] (sound; over-splits) per-CTA loop-carried partial accumulate (a left-fold);
        # _balance_pass rejects it as unbalanced, so no ITreeReduce collapse.
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    # [scope: neither] the intra-CTA tree-reduce IS a tt.reduce -> keyed by TTGIR (add), a real
    # reduction node both checkers understand.
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    if HAS_RSPLIT:
        tmp2_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp2_ws + (xindex * RSPLIT + rsplit_id), tmp2, xindex < xnumel)
    if HAS_RSPLIT:
        # [scope: BOTH] the CROSS-CTA carry: a grid barrier + a shared workspace where each CTA writes
        # its partial and re-reduces the peers' partials (the tl.sum below). PTX cannot follow the
        # barrier / global workspace (opaque); TTGIR sees only the tt.reduce nodes and is blind to the
        # cross-CTA ordering (safe by side effect: a different RSPLIT changes the peer-reduce key).
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp2_peers = tl.load(tmp2_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp2 = tl.sum(tmp2_peers, 1)[:, None]
    if rsplit_id == (0 % RSPLIT):
        tl.store(out_ptr0 + (x0), tmp2, xmask)


# #########################################################################################
# ## GROUP 7 - POINTWISE / FOREACH (the order-insensitive floor).
# ## Pure elementwise kernels: no reduction, so no cross-element FP order to constrain. These are the
# ## 'trivially sound' baseline - TTGIR returns () (equal), PTX builds a per-element DAG with no
# ## cross-lane node. The first two are harvested verbatim from Inductor's own tests; the third is a
# ## combo/foreach kernel generated on this H100 (config.combo_kernels).
# #########################################################################################

# ========================================================================================= #
# PW-1. Reflection-pad + add  (source: triton_poi_fused_add_reflection_pad2d_0, test_cuda_repro.py:2480)
# Kernel type: POINTWISE (reflection-pad addressing via tl_math.abs + a tensor add).
# What it computes: out = pad_reflect(x) + pad_reflect(y); the interesting part is the folded
# abs()-based reflection index arithmetic in the load address (a PTX AffineEval stress test).
# Verified scope:  TTGIR = neither,  PTX = neither,  overall = neither (trivially sound).
# Novelty vs suite: a pure pointwise kernel with NON-trivial affine addressing but zero reduction -
# the baseline where both checkers correctly say 'equivalent for every config'.
# ========================================================================================= #
@triton.jit
def PW_reflection_pad_add(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x1 = ((xindex // 20) % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    # [scope: neither] pure pointwise, no reduction. TTGIR returns () (trivially equal - an
    # elementwise kernel has no cross-element FP order to constrain); PTX builds a per-element DAG
    # with no cross-lane node. The only interesting bit is the tl_math.abs reflection addressing.
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)


# ========================================================================================= #
# PW-2. torch.cat via masks  (source: triton_poi_fused_cat_0, test_static_triton_launcher.py:465)
# Kernel type: POINTWISE (a two-operand concat lowered to tl.where masks + int64 index math).
# What it computes: out = cat(x*4, y+10) - a concat expressed as masked selects over one index space.
# Verified scope:  TTGIR = neither,  PTX = neither,  overall = neither (trivially sound).
# Novelty vs suite: shows tl.where-based control flow (concat/masking) in a pointwise body; still no
# cross-lane FP order, so it stays in the trivially-sound floor.
# ========================================================================================= #
@triton.jit
def PW_cat_masked(
    in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(
        in_ptr0 + (x0), xmask & tmp4, eviction_policy="evict_last", other=0.0
    )
    tmp6 = 4.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp13 = tl.load(
        in_ptr1 + (x0 + ((-1) * ks0)),
        xmask & tmp10,
        eviction_policy="evict_last",
        other=0.0,
    )
    tmp14 = 10.0
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    # [scope: neither] a torch.cat lowered to tl.where masks; still pure pointwise, no cross-lane
    # FP order. Both checkers: trivially equivalent across configs (TTGIR () / PTX pure DAG).
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x0), tmp18, xmask)


# ========================================================================================= #
# PW-3. Combo / foreach add  (source: triton_for_fused_0 | repro: torch._foreach_add of 4 tensors, H100)
# Kernel type: FOREACH / combo kernel (several independent pointwise ops horizontally fused).
# What it computes: 4 independent elementwise adds dispatched by program-id range in ONE launch.
# Verified scope:  TTGIR = neither,  PTX = neither,  overall = neither (trivially sound).
# Novelty vs suite: the horizontally-fused (foreach) structure - one kernel, several unrelated
# pointwise sub-kernels selected by pid. No reduction anywhere.
# ========================================================================================= #
@triton.jit
def FOREACH_combo_add(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3):
    # [scope: neither] a combo/foreach kernel: FOUR independent pointwise adds dispatched by pid
    # range in one launch. No reduction anywhere -> trivially sound for both checkers.
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(4096, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(4096, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(4096, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(4096, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 4096
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)[:]
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 4096
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)[:]
        x1 = xindex
        tmp3 = tl.load(in_ptr2 + (x1), None)
        tmp4 = tl.load(in_ptr3 + (x1), None)
        tmp5 = tmp3 + tmp4
        tl.store(out_ptr1 + (x1), tmp5, None)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 4096
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)[:]
        x2 = xindex
        tmp6 = tl.load(in_ptr4 + (x2), None)
        tmp7 = tl.load(in_ptr5 + (x2), None)
        tmp8 = tmp6 + tmp7
        tl.store(out_ptr2 + (x2), tmp8, None)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        xnumel = 4096
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)[:]
        x3 = xindex
        tmp9 = tl.load(in_ptr6 + (x3), None)
        tmp10 = tl.load(in_ptr7 + (x3), None)
        tmp11 = tmp9 + tmp10
        tl.store(out_ptr3 + (x3), tmp11, None)
    else:
        pass


# #########################################################################################
# ## GROUP 8 - NON-FP REDUCTION.
# ## A boolean any(isinf(x)) reduction harvested verbatim from Inductor's tests. It IS a tt.reduce,
# ## so TTGIR keys it - but the combine is OR, not add, so the PTX add-only collapse never fires, and
# ## being integer there is no FP order to preserve. A useful 'reduction that is not an FP add-tree'.
# #########################################################################################

# ========================================================================================= #
# RED-1. Boolean any(isinf)  (source: triton_red_fused_any_isinf_0, test_static_triton_launcher.py:351)
# Kernel type: REDUCTION (looped OR-reduce of an isinf predicate via triton_helpers.any).
# What it computes: out = any(isinf(x)) over the reduction axis (a NaN/Inf guard reduction).
# Verified scope:  TTGIR = in,  PTX = partial,  overall = partial (sound; non-FP).
# Novelty vs suite: the first NON-add reduction - TTGIR keys the OR combine; PTX will not collapse it
# (_REDUCE_FP={add}); and since it is integer/boolean there is no FP association order to enforce.
# ========================================================================================= #
@triton.jit
def RED_any_isinf(
    in_ptr0,
    out_ptr0,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 1  # noqa: F841
    rnumel = r0_numel  # noqa: F841
    RBLOCK: tl.constexpr = R0_BLOCK  # noqa: F841
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # noqa: F841
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)  # noqa: F841
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base  # noqa: F841
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset  # noqa: F841
        rindex = r0_index  # noqa: F841
        r0_0 = r0_index
        tmp0 = tl.load(
            in_ptr0 + (r0_0), r0_mask, eviction_policy="evict_first", other=0.0
        )
        tmp1 = libdevice.isinf(tmp0).to(tl.int1)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 | tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    # [scope: TTGIR] (in) the boolean OR-reduction is a tt.reduce -> TTGIR keys its combine
    # (or/any). [scope: PTX] a non-add reduce: _REDUCE_FP={add} means PTX never collapses it ->
    # it stays layout-bearing (sound; over-splits). Being integer, there is no FP order anyway.
    tmp3 = triton_helpers.any(_tmp3.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


# =========================================================================================
# Corpus tuples. Values are the raw @triton.jit functions (NOT KernelSpecs) - this module is a
# reference/gap-analysis corpus, deliberately not wired into evaluate.py. See the module
# docstring for per-kernel scope verdicts and how each kernel was collected.
# =========================================================================================
REDUCTION_FUSION_KERNELS = (
    A_rms_norm_fwd,
    B_layernorm_welford_gather,
    C_rms_norm_bwd_2reduce,
    D_masked_global_sum,
    E_triu_masked_rowsum,
    F_cumsum_scan,
    G_plain_sum_looped,
    H_mean_permute,
)

# Back-compat alias: this name used to hold exactly the original 8 reduction fusions.
COMPLEX_FUSION_KERNELS = REDUCTION_FUSION_KERNELS

GEMM_KERNELS = (
    GEMM_plain_mm,
    GEMM_addmm_bias,
    GEMM_batched_bmm,
)

# NOT runnable on sm_90 (needs Blackwell / sm_100); the verify harness skips it.
WARP_SPECIALIZED_KERNELS = (WS_blackwell_ws_tma_mm,)

ATTENTION_KERNELS = (ATTN_flex_attention_fwd,)

SCAN_KERNELS = (SCAN_split_lookback_cumsum,)

COOPERATIVE_REDUCTION_KERNELS = (COOP_xgrid_sum,)

POINTWISE_KERNELS = (
    PW_reflection_pad_add,
    PW_cat_masked,
    FOREACH_combo_add,
)

NONFP_REDUCTION_KERNELS = (RED_any_isinf,)

# Everything, groups in table order.
REALISTIC_KERNELS = (
    REDUCTION_FUSION_KERNELS
    + GEMM_KERNELS
    + WARP_SPECIALIZED_KERNELS
    + ATTENTION_KERNELS
    + SCAN_KERNELS
    + COOPERATIVE_REDUCTION_KERNELS
    + POINTWISE_KERNELS
    + NONFP_REDUCTION_KERNELS
)
