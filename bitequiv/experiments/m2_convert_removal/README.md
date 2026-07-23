# M2 convert-removal experiment (approach A vs approach B)

Evidence artifacts for one M2 design decision: **keep A (convert-based); reject B
(convert removal).** Full write-up: `M2_DESIGN.en.md` / `M2_DESIGN.zh.md` §5.3–§5.5
and the standalone protocol `M2_approach_B_experiment.md` (both in the bitequiv
knowledge base).

## The question

The M2 pass `tritongpu-optimize-reduction-layout` inserts one `ttg.convert_layout`
on a reduction's operand (coalesced load → reduce-friendly layout). Approach B
asked: can we remove that convert by making the *load* produce the reduce-friendly
layout directly (pinning it upstream, à la `OptimizeThreadLocality`)?

## The two kernels here

Real compiled TTGIR for `sum_2d_col` [256×32], reduce axis 0 (M non-contiguous,
C contiguous), `num_warps=4`, `reduction_ordering="inner_tree"`:

- `sum_2d_col.A.ttgir` — **approach A**: coalesced load → `ttg.convert_layout` →
  warp-synchronous reduce (what the pass emits).
- `sum_2d_col.B.ttgir` — **approach B**: load *directly* in the reduce-friendly
  layout (uncoalesced) → reduce, **no convert**.

The only difference is the load layout and the presence of the convert. B removes
the convert but forces an uncoalesced load: 32 lanes on the strided M axis ≈ 32×
the memory transactions.

## Result (GB300, `do_bench` min-of-medians; 6 kernels × num_warps ∈ {2,4,8} = 17 configs)

- A: **1.6–5.3×** over base, 0 regressions.
- B: **0.41–1.92×** over base (usually *slower* than base); **2.5–4.8× slower than A**.
- All 17 configs: `base == A == B` bytewise (`inner_tree` makes layout a free knob).

**Conclusion.** For a reduction over a non-contiguous axis, "no convert" is only
reachable by an uncoalesced load, which costs more than the cross-warp stage it
removes. A's convert is the cheaper of the two unavoidable bridges between a
coalesced load layout and a warp-synchronous reduce layout. A chosen; B rejected.

## Reproduce

`bexp.py` builds the B variant from A's ideal layout, injects it via a
`make_ttgir` patch (same hook `evaluate_opt` uses for A), and times base/A/B.
Run from an activated venv on a GPU box; see `M2_approach_B_experiment.md` for the
exact command, environment, and per-config data.
