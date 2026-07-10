"""Static PTX reduction-equivalence — a standalone layout-equivalence checker (M1-PTX).

**No runtime.** Two autotuning configs are bitwise-equivalent *reductions* iff the
kernels they compile to reduce the same elements, in the same floating-point association
order, with the same rounding/fusion. This module decides that **from PTX alone** — it is
an *independent* check level, a peer of the TTGIR checker (sibling ``bitequiv-m1``
branch), not a backstop layered after it. The engine reconstructs the reduction *tree*
from PTX and labels each leaf with the tensor element it reads (recovered from the
load-address arithmetic), so the PTX signature captures the **layout** (which element each
thread/lane/register holds), the **within-thread fold order**, the **within-/cross-warp
butterfly tree**, *and* the **rounding/FMA-contraction** decided below TTGIR (``fma.rn.f32``
vs ``mul``+``add``). Same signature => same PTX reduction => same bits. No kernel launch,
no input tensors, no reference output.

Signatures are conservative — anything the engine cannot prove is modeled becomes an
opaque token that matches only an identical one. This makes it **sound relative to the
reconstructed result-DAG at PTX** (it over-splits rather than over-merges on what it can
see), *not* an absolute proof of bit-equality: see :mod:`bitequiv.ptx_reduction` for the
exact conditions and residuals (PTX->SASS, walk-completeness, opaque-token collisions). It
is a cheap static pre-filter; the runtime bit-comparison is the ground truth. Being
standalone does not remove composition: ``level=["ttgir", "ptx"]`` still AND-s the two
independent checkers into one strictly-more-conservative key where both branches' code
coexists.

The reconstruction engine lives in :mod:`bitequiv.ptx_reduction` (+ :mod:`bitequiv.ptx`);
this module keeps the autotuner-facing API (signature/prune/registry). The human-readable
design is in ``bitequiv/design-doc.md``.
"""
from collections import OrderedDict

from .ptx_reduction import ptx_reduction_descriptor


def ptx_reduction_signature(ptx):
    """Standalone reduction-tree signature reconstructed from **PTX**.

    Thin alias for :func:`bitequiv.ptx_reduction.ptx_reduction_descriptor`: reconstructs
    the floating-point reduction tree (layout-labeled leaves + fold order + butterfly
    shuffles + rounding/fusion modifiers) into a conservative, hashable signature. Detects
    layout differences (``num_warps``/``sizePerThread``/transpose) *and* the fusion/rounding
    decided below TTGIR (e.g. a ``tl.sum(x*y)`` dot with ``enable_fp_fusion`` on vs off —
    identical TTGIR, different PTX). ``()`` when there is no reduction in the PTX.
    """
    return ptx_reduction_descriptor(ptx)


def _ptx_equivalence_key(config, asm, metadata):
    return ptx_reduction_signature(asm.get("ptx", ""))


# Registry of equivalence *key functions* by IR level, consumed by
# ``reduction_equivalence_prune(level)``. Each value is a
# ``(config, asm, metadata) -> Hashable`` key fn; configs with equal keys are
# bitwise-equivalent at that level. This branch ships only the PTX level; the TTGIR
# checker lives on the separate ``bitequiv-m1`` branch.
CHECKERS = {
    "ptx": _ptx_equivalence_key,
}

_UNSET = object()  # sentinel: a config's key may legitimately be None/falsy


class _IRBasedPrune:
    """``ir_config_prune`` predicate built from an equivalence *key function*, keeping
    only configs whose key matches a chosen **reference**.

    Registered via ``prune_configs_by={"ir_config_prune": <this>}``. The autotuner
    compiles each config once (``run(warmup=True)``, no launch) and calls this with
    ``(config, asm, metadata, reference)``, where ``reference`` is the
    ``(config, asm, metadata)`` of the reference config (by default the first config).
    We map each config to a hashable key via ``key_fn`` and KEEP it iff its key equals
    the reference key — bitwise-equivalent by construction, no kernel launch, no
    reference output.

    The reference is resolved per call (so it is correct across multiple autotuning
    keys) in this order:
      1. an explicit external anchor passed to the constructor (``reference=`` — a
         pre-compiled kernel with ``.asm``, an ``asm`` dict, or raw IR text), else
      2. the ``reference`` the autotuner passes (the first config, by default), else
      3. (direct / 3-arg use with no reference) the first config seen.

    Introspection (reset at the start of each prune run):
      * ``.classes`` — ``{key: [Config, ...]}`` equivalence classes seen.
      * ``.pruned``  — ``{Config: reason}`` for configs dropped as non-equivalent.
    """

    def __init__(self, key_fn, reference=None):
        self.key_fn = key_fn
        self.user_reference = reference  # explicit external anchor (compiled kernel / asm dict / IR text)
        self.classes = OrderedDict()
        self.pruned = OrderedDict()
        self._anchor_key = _UNSET  # cached key of the explicit anchor (constant across calls)
        self._first_key = _UNSET  # first-seen fallback when no reference is provided at all
        self._last_reference = _UNSET  # identity of the current run's reference (to reset per run)

    def _anchor_key_of(self, ref):
        """Reference key for an explicit external anchor not in the tuning set."""
        if hasattr(ref, "asm"):  # CompiledKernel-like
            return self.key_fn(None, getattr(ref, "asm", {}) or {}, getattr(ref, "metadata", None))
        if isinstance(ref, dict):  # an asm dict
            return self.key_fn(None, ref, None)
        if isinstance(ref, str):  # raw IR text (the checker reads the level it needs)
            return self.key_fn(None, {"ptx": ref}, None)
        raise ValueError(f"unsupported reference anchor {type(ref).__name__}; pass a compiled kernel, an asm "
                         "dict, or IR text")

    def _reference_key(self, reference):
        if self.user_reference is not None:  # explicit external anchor wins (cached)
            if self._anchor_key is _UNSET:
                self._anchor_key = self._anchor_key_of(self.user_reference)
            return self._anchor_key
        if reference is not None:  # autotuner-passed reference: (config, asm, metadata)
            return self.key_fn(*reference)
        return _UNSET  # nothing provided -> first-seen fallback

    def __call__(self, config, asm, metadata, reference=None):
        if reference is not None and reference is not self._last_reference:
            self.classes = OrderedDict()  # new prune run (fresh reference object) -> reset introspection
            self.pruned = OrderedDict()
            self._first_key = _UNSET
            self._last_reference = reference
        key = self.key_fn(config, asm, metadata)
        self.classes.setdefault(key, []).append(config)
        reference_key = self._reference_key(reference)
        if reference_key is _UNSET:  # no reference at all: first config defines the order
            if self._first_key is _UNSET:
                self._first_key = key
            reference_key = self._first_key
        keep = key == reference_key
        if not keep:
            self.pruned[config] = "not-equivalent-to-reference"
        return keep


def ir_based_prune_configs(key_fn, reference=None):
    """Turn an equivalence *key function* into an ``ir_config_prune`` predicate.

    ``key_fn(config, asm, metadata) -> Hashable`` returns a config's equivalence key
    from its compiled artifact (e.g. a PTX reduction-order signature). The returned
    predicate keeps only configs whose key matches the reference; pass it as
    ``prune_configs_by={"ir_config_prune": ir_based_prune_configs(key_fn)}`` and keep a
    handle to read ``.classes`` / ``.pruned`` afterwards. ``reference`` selects the
    anchor (see :class:`_IRBasedPrune`): ``None`` = the first config (supplied by the
    autotuner), or an external pre-compiled kernel / asm dict / IR text not in the set.
    """
    return _IRBasedPrune(key_fn, reference)


def reduction_equivalence_prune(level="ptx", reference=None):
    """Static reduction-order equivalence prune at the given IR ``level``.

    ``level`` is ``"ptx"`` (default) or an ordered list of level names (only ``"ptx"``
    is available on this branch; the TTGIR checker lives on ``bitequiv-m1``). Multiple
    levels are combined into a single key (the per-level signatures as a tuple), so a
    config is kept iff it matches the reference at *every* level. ``reference`` chooses
    the anchor order: ``None`` (the first config, supplied by the autotuner) or an
    external pre-compiled kernel / asm dict / IR text. Built on
    :func:`ir_based_prune_configs`; returns a predicate exposing ``.classes`` /
    ``.pruned``.
    """
    names = [level] if isinstance(level, str) else list(level)
    checkers = []
    for name in names:
        if name not in CHECKERS:
            raise ValueError(f"unknown equivalence level {name!r}; available: {sorted(CHECKERS)}")
        checkers.append(CHECKERS[name])
    if len(checkers) == 1:
        key_fn = checkers[0]
    else:

        def key_fn(config, asm, metadata):
            return tuple(checker(config, asm, metadata) for checker in checkers)

    return ir_based_prune_configs(key_fn, reference)
