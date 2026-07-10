"""Complex fusion reduction kernels — the ULTIMATE-GOAL corpus (NOT current M1 scope).

These are not synthetic test kernels like the ones in ``eval_kernels.py``. They are
verbatim Triton kernels emitted by **torch.compile / TorchInductor** on real production
models, collected purely for GAP ANALYSIS: "how far is the current reduction-equivalence
checker from what a real customer actually ships?". They are deliberately messy and fused.

    ⚠️  We do NOT expect the checker to fully support these today, and this module is
        intentionally NOT wired into ``evaluate.py`` yet. It is a reference corpus that
        marks the ultimate target (full precision AND soundness on real customer kernels),
        against which each future checker improvement can be measured. Treat every kernel
        here as "realistic customer usage → ultimate goal", not as an in-scope requirement.

Provenance
----------
  These are all standard reduction fusions produced by ``torch.compile`` / TorchInductor on
  real production workloads — nothing exotic: LayerNorm / RMS-norm forward and backward,
  gather+norm+sigmoid, masked and looped sums, and a cumulative-sum scan.
  ``B_layernorm_welford_gather`` is a LayerNorm + index_select + sigmoid fusion; the rest are
  RMS-norm-heavy reduction fusions whose many near-duplicate variants reduce to these 8
  distinct reduction SHAPES. (Source-workload links are intentionally omitted.)

How the scope verdicts below were produced
------------------------------------------
Each kernel was checked against the two shipped checkers by reading their source
(``bitequiv/ptx_reduction.py`` + engine ``bitequiv/ptx/``, and ``bitequiv/ttgir_reduction.py``
+ ``lib/Analysis/ReductionOrder.cpp``), and the verdict was adversarially re-verified. Recall
what each checker does:
  * **PTX checker** — reconstructs the FP reduction TREE from PTX: roots = ``st.global``
    values, leaves = ``ld.global`` loads labelled by AFFINE load-address arithmetic; the
    num_warps/layout-invariant "collapse" (``ITreeReduce``) fires only for a SINGLE-output,
    fully-affine, BALANCED **add** tree. Everything else stays a verbatim, sound-but-layout-
    bearing tree (it over-splits, never over-merges on what it can see).
  * **TTGIR checker** — one signature per ``tt.reduce`` from ``toLinearLayout`` + the combine
    region's op-name sequence. It is FMA-contraction-blind and it NEVER walks ``tt.scan``,
    ``tt.load``, addresses or value provenance.

Inline scope tags (this is the "out of which checker's scope?" flag the corpus is for)
--------------------------------------------------------------------------------------
    [scope: PTX]     — essential syntax the PTX tree-checker cannot model / degrades on
    [scope: TTGIR]   — essential syntax the TTGIR layout-checker cannot model / degrades on
    [scope: BOTH]    — out of scope for both checkers
    [scope: neither] — handled SOUNDLY by both (flagged only to pre-empt "surely this breaks it")
severity in parentheses:
    (SOUNDNESS GAP)      — the checker can return a WRONG "equivalent" (over-merge). ONLY the
                           scan kernel hits this. This is the dangerous class.
    (sound; over-splits) — never wrong; the checker just refuses the num_warps-invariant merge,
                           so it loses tuning freedom. This is the checker working as designed —
                           the open work here is PRECISION, not soundness.

=========================================================================================
SUMMARY TABLE (verified TTGIR / PTX / overall scope + why it is not fully supported yet)
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

Big-picture takeaway (what the ultimate goal actually requires)
--------------------------------------------------------------
  * Only ONE kernel (``F_cumsum_scan``) is a true SOUNDNESS gap — the TTGIR checker can wrongly
    call two scans equivalent. Closing the scan gap is the single correctness item here.
  * ``B_layernorm_welford_gather`` carries the one real PTX soundness RISK: a data-dependent
    GATHER load whose leaf loses provenance (documented opaque-leaf collision risk).
  * The other SIX are already SOUND — the checker never wrong-merges them; it only OVER-SPLITS
    (refuses the num_warps-invariant collapse under multi-output / masked-tail / loop-accumulate
    shapes). So for those, the ultimate-goal work is recovering PRECISION (tuning freedom), not
    fixing correctness.
  * Net roadmap this corpus points at: (1) model ``tt.scan``; (2) model gather-fed reductions;
    (3) recover the num_warps-invariant collapse for multi-output / tail-masked / accumulate-then-
    reduce trees; (4) add real FMA/welford-combine numerics (today only structural on TTGIR).

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


# All kernels in this corpus, in table order. Values are the raw @triton.jit functions (NOT
# KernelSpecs) — this module is a reference/gap-analysis corpus and is deliberately not wired
# into evaluate.py. See the module docstring for the per-kernel scope verdicts.
COMPLEX_FUSION_KERNELS = (
    A_rms_norm_fwd,
    B_layernorm_welford_gather,
    C_rms_norm_bwd_2reduce,
    D_masked_global_sum,
    E_triu_masked_rowsum,
    F_cumsum_scan,
    G_plain_sum_looped,
    H_mean_permute,
)
