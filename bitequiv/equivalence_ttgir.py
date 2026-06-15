"""Static TTGIR reduction-order equivalence (Starter T3 redesign / M1 core).

**No runtime.** Two autotuning configs are bitwise-equivalent *reductions* iff the
kernels they compile to reduce the same elements in the same tree shape. After
layout assignment that tree shape is fully determined at **TTGIR** by:

  * the **data layout of the reduced operand** along the reduction axis
    (``sizePerThread[axis]``, ``threadsPerWarp[axis]``, ``warpsPerCTA[axis]`` and
    the ``order``) — this fixes which element lives in which register/lane/warp,
    hence the intra-thread / intra-warp / cross-warp combination structure;
  * the ``reduction_ordering`` attribute on the ``tt.reduce`` op
    (``unordered`` vs ``inner_tree`` — count-down-bfly + left-fold vs
    count-up-bfly + balanced tree); and
  * the combine op (``arith.addf`` / ``mulf`` / ``fma`` ...).

So we **statically compare the compiled IRs**: extract a *reduction-order
signature* from each config's TTGIR and group configs by it. Same signature =>
same reduction order => same bits. This is the opposite of a runtime output
check: no kernel launch, no input tensors, no reference output — and it is
input-independent (the op *order* is the same for every input).

Caveat on IR level: TTGIR fixes the *association order* but is blind to **FMA
contraction**, which is decided below TTGIR. For pure add/min/max reductions that
costs nothing; once a multiply feeds the reduction (Welford, dot-product, GEMM/M3)
a PTX/``--fmad`` backstop is also needed (provided separately on the
``bitequiv-m1-ptx`` branch). Signatures are conservative — they never call non-equivalent configs equivalent
(a textual layout difference yields a different signature), so the kept set is always
a safe subset.

The reduction-order engine now lives in :mod:`bitequiv.reduction_tree` (a logical
mixed-radix / GF(2) thread<->data descriptor + a canonical-tree test oracle); this
module keeps the autotuner-facing API (signature/prune/registry). The human-readable
design is in ``bitequiv/design-doc.md``.
"""
from collections import OrderedDict

from .reduction_tree import reduction_descriptor, reductions_equivalent


def reduction_signature(ttgir):
    """Static reduction-order signature of a TTGIR module (one entry per
    ``tt.reduce``).

    Thin alias for :func:`bitequiv.reduction_tree.reduction_descriptor` — the
    conservative, shape-aware logical-relation descriptor. Two TTGIRs with equal
    signatures perform identical reduction trees (sound); ``()`` when there is no
    reduction. See ``bitequiv/design-doc.md``
    for how the descriptor encodes the thread<->data map and how equality is decided.
    """
    return reduction_descriptor(ttgir)


def same_reduction_order(ttgir_a, ttgir_b):
    """True iff the two TTGIRs reduce in the same (bitwise-equivalent) order.
    Thin alias for :func:`bitequiv.reduction_tree.reductions_equivalent`."""
    return reductions_equivalent(ttgir_a, ttgir_b)


def reduction_equivalence_key(config, asm, metadata):
    """Equivalence *key function* for the TTGIR level: maps a config's compiled
    artifact to its static reduction-order signature.

    Used by :func:`reduction_equivalence_prune` / :func:`ir_based_prune_configs` to
    keep only configs bitwise-equivalent to the reference (first) config — no kernel
    launch, no reference output.
    """
    return reduction_signature(asm.get("ttgir", ""))


# Registry of equivalence *key functions* by IR level, consumed by
# ``reduction_equivalence_prune(level)``. Each value is a
# ``(config, asm, metadata) -> Hashable`` key fn; configs with equal keys are
# bitwise-equivalent at that level. This branch ships only the TTGIR level; the PTX
# backstop lives on the separate ``bitequiv-m1-ptx`` branch.
CHECKERS = {
    "ttgir": reduction_equivalence_key,
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
            return self.key_fn(None, {"ttgir": ref}, None)
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
    from its compiled artifact (e.g. a reduction-order signature). The returned
    predicate keeps only configs whose key matches the reference; pass it as
    ``prune_configs_by={"ir_config_prune": ir_based_prune_configs(key_fn)}`` and keep a
    handle to read ``.classes`` / ``.pruned`` afterwards. ``reference`` selects the
    anchor (see :class:`_IRBasedPrune`): ``None`` = the first config (supplied by the
    autotuner), or an external pre-compiled kernel / asm dict / IR text not in the set.
    """
    return _IRBasedPrune(key_fn, reference)


def reduction_equivalence_prune(level="ttgir", reference=None):
    """Static reduction-order equivalence prune at the given IR ``level``.

    ``level`` is ``"ttgir"`` (default) or an ordered list of level names (only
    ``"ttgir"`` is available on this branch; the PTX backstop lives on
    ``bitequiv-m1-ptx``). Multiple levels are combined into a single key (the per-level
    signatures as a tuple), so a config is kept iff it matches the reference at *every*
    level. ``reference`` chooses the anchor order: ``None`` (the first config, supplied
    by the autotuner) or an external pre-compiled kernel / asm dict / IR text. Built on
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


def classify(named_ttgirs):
    """Group ``{label: ttgir}`` into equivalence classes by reduction signature.

    Returns ``OrderedDict{signature: [labels...]}`` (insertion order preserved),
    so labels sharing a signature are bitwise-equivalent reductions.
    """
    classes = OrderedDict()
    for label, ttgir in named_ttgirs.items():
        classes.setdefault(reduction_signature(ttgir), []).append(label)
    return classes
