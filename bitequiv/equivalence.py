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
a PTX/``--fmad`` backstop is also needed (the ``ptx`` checker below is the slot for
it). Signatures are conservative — they never call non-equivalent configs equivalent
(a textual layout difference yields a different signature), so the kept set is always
a safe subset.
"""
import re
from collections import OrderedDict

# `#name = #ttg.blocked<{...}>` (also #ttg.slice / #ttg.nvidia_mma / #ttg.linear ...)
_ENC_DEF = re.compile(r"^#(\w+)\s*=\s*(#ttg\.\w+<.*>)\s*$", re.M)

# A tt.reduce op: capture (attrs, region-body, operand-encoding-name).
#   "tt.reduce"(%v) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({ ...region... })
#       : (tensor<64x1024xf32, #blocked>) -> tensor<64xf32, #ttg.slice<...>>
_REDUCE = re.compile(r'"tt\.reduce"\([^)]*\)\s*<\{([^}]*)\}>\s*\(\{(.*?)\}\)\s*:\s*'
                     r"\(tensor<[^,>]*,\s*(#\w+)>\)", re.S)

_ARRAY = lambda key, body: (lambda m: tuple(int(v) for v in m.group(1).replace(" ", "").split(",") if v != "")
                            if m else None)(re.search(key + r"\s*=\s*\[([0-9, ]*)\]", body))


def _encoding_defs(ttgir):
    return {name: body for name, body in _ENC_DEF.findall(ttgir)}


def _layout_along_axis(enc_body, axis):
    """The layout facts that determine the reduction tree along ``axis`` for a
    blocked encoding. Returns a hashable tuple; falls back to the raw encoding
    body for non-blocked encodings (#mma/#linear — conservative: exact-match)."""
    spt = _ARRAY("sizePerThread", enc_body)
    tpw = _ARRAY("threadsPerWarp", enc_body)
    wpc = _ARRAY("warpsPerCTA", enc_body)
    order = _ARRAY("order", enc_body)
    if spt is None or tpw is None or wpc is None or axis is None or axis >= len(spt):
        # Not a plain blocked layout we can project (e.g. #mma) — compare verbatim.
        return ("raw", " ".join(enc_body.split()))
    return (
        ("sizePerThread@axis", spt[axis]),
        ("threadsPerWarp@axis", tpw[axis]),
        ("warpsPerCTA@axis", wpc[axis]),
        ("order", order),
    )


def reduction_signature(ttgir):
    """Static reduction-order signature of a TTGIR module.

    Returns a tuple with one entry per ``tt.reduce`` op:
    ``(axis, reduction_ordering, combine_ops, layout_along_axis)``.
    Two TTGIRs with equal signatures perform identical reduction trees.
    Returns ``()`` when there is no reduction (caller decides what that means).
    """
    defs = _encoding_defs(ttgir)
    sigs = []
    for attrs, region, enc_name in _REDUCE.findall(ttgir):
        am = re.search(r"axis\s*=\s*(\d+)", attrs)
        om = re.search(r'reduction_ordering\s*=\s*"(\w+)"', attrs)
        axis = int(am.group(1)) if am else None
        ordering = om.group(1) if om else None
        combine = tuple(sorted(set(re.findall(r"(?:arith|math)\.\w+", region))))
        if ordering == "inner_tree":
            # inner_tree enforces one canonical order over original element indices,
            # independent of layout -> all such configs are equivalent regardless of
            # warpsPerCTA/sizePerThread. Exclude layout from the signature.
            layout = ("layout-invariant(inner_tree)", )
        else:
            layout = _layout_along_axis(defs.get(enc_name.lstrip("#"), enc_name), axis)
        sigs.append((axis, ordering, combine, layout))
    return tuple(sigs)


def same_reduction_order(ttgir_a, ttgir_b):
    """True iff the two TTGIRs reduce in the same (bitwise-equivalent) order."""
    return reduction_signature(ttgir_a) == reduction_signature(ttgir_b)


def reduction_equivalence_key(config, asm, metadata):
    """Equivalence *key function* for the TTGIR level: maps a config's compiled
    artifact to its static reduction-order signature.

    Used by :func:`reduction_equivalence_prune` / :func:`ir_based_prune_configs` to
    keep only configs bitwise-equivalent to the reference (first) config — no kernel
    launch, no reference output.
    """
    return reduction_signature(asm.get("ttgir", ""))


def ptx_reduction_signature(ptx):
    """STUB — static reduction-order signature reconstructed from **PTX** (the
    FMA / below-TTGIR backstop). Not implemented yet.

    This is the slot for the future lane-parametric PTX reconstruction (parse the
    ``shfl.sync.bfly`` + ``add``/``fma`` tree). Filling this in is the *only* change
    needed to enable PTX-level equivalence (``reduction_equivalence_prune("ptx")``)
    end to end — no autotuner or call-site changes.
    """
    raise NotImplementedError("PTX-level reduction equivalence is not implemented yet; only TTGIR-level "
                              "(reduction_equivalence_prune(\"ttgir\")) is available.")


def _ptx_equivalence_key(config, asm, metadata):
    return ptx_reduction_signature(asm.get("ptx", ""))


# Registry of equivalence *key functions* by IR level, consumed by
# ``reduction_equivalence_prune(level)``. Each value is a
# ``(config, asm, metadata) -> Hashable`` key fn; configs with equal keys are
# bitwise-equivalent at that level. ``"both"`` combines ttgir + ptx into one key
# (cheap TTGIR signature + the below-TTGIR ground-truth backstop).
CHECKERS = {
    "ttgir": reduction_equivalence_key,
    "ptx": _ptx_equivalence_key,
}

_UNSET = object()  # sentinel: a config's key may legitimately be None/falsy


class _IRBasedPrune:
    """Stateful ``ir_config_prune`` predicate built from an equivalence *key
    function*.

    Usable directly as ``prune_configs_by={"ir_config_prune": <this>}``: the
    autotuner compiles each config once (``run(warmup=True)``, no launch) and calls
    this for each, in config order, with ``(config, asm, metadata)``. We map the
    config to a hashable key via ``key_fn`` and KEEP it iff the key matches the
    **first-seen (reference) config's** key — so the surviving set is bitwise-
    equivalent to ``configs[0]`` by construction, with no kernel launch and no
    reference output.

    Introspection (replaces the autotuner's old ``equivalence_classes`` /
    ``pruned_by_equivalence``):
      * ``.classes`` — ``{key: [Config, ...]}`` equivalence classes seen.
      * ``.pruned``  — ``{Config: reason}`` for configs dropped as non-equivalent.
    """

    def __init__(self, key_fn):
        self.key_fn = key_fn
        self.reference = _UNSET
        self.classes = OrderedDict()
        self.pruned = OrderedDict()

    def __call__(self, config, asm, metadata):
        key = self.key_fn(config, asm, metadata)
        self.classes.setdefault(key, []).append(config)
        if self.reference is _UNSET:
            self.reference = key  # first config defines the reference order
        keep = key == self.reference
        if not keep:
            self.pruned[config] = "not-equivalent-to-reference"
        return keep


def ir_based_prune_configs(key_fn):
    """Turn an equivalence *key function* into an ``ir_config_prune`` predicate.

    ``key_fn(config, asm, metadata) -> Hashable`` returns a config's equivalence key
    from its compiled artifact (e.g. a reduction-order signature). The returned
    callable keeps only configs whose key matches the reference (first) config; pass
    it as ``prune_configs_by={"ir_config_prune": ir_based_prune_configs(key_fn)}``.
    Keep a handle to it to read ``.classes`` / ``.pruned`` after autotuning.
    """
    return _IRBasedPrune(key_fn)


def reduction_equivalence_prune(level="ttgir"):
    """Static reduction-order equivalence prune at the given IR ``level``.

    ``level`` is ``"ttgir"`` (default), ``"ptx"`` (stub — raises until implemented),
    ``"both"``, or an ordered list of level names. Multiple levels are combined into a
    single key (the per-level signatures as a tuple), so a config is kept iff it
    matches the reference at *every* level. Built on :func:`ir_based_prune_configs`;
    returns an ``ir_config_prune`` predicate exposing ``.classes`` / ``.pruned``.
    """
    names = ["ttgir", "ptx"] if level == "both" else ([level] if isinstance(level, str) else list(level))
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

    return ir_based_prune_configs(key_fn)


def classify(named_ttgirs):
    """Group ``{label: ttgir}`` into equivalence classes by reduction signature.

    Returns ``OrderedDict{signature: [labels...]}`` (insertion order preserved),
    so labels sharing a signature are bitwise-equivalent reductions.
    """
    classes = OrderedDict()
    for label, ttgir in named_ttgirs.items():
        classes.setdefault(reduction_signature(ttgir), []).append(label)
    return classes
