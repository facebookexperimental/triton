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

Caveat (see knowledge-base/equivalence-check-level-ttgir-vs-ptx.md): TTGIR fixes
the *association order* but is blind to **FMA contraction**, which is decided
below TTGIR. For pure add/min/max reductions that costs nothing; once a multiply
feeds the reduction (Welford, dot-product, GEMM/M3) a PTX/``--fmad`` backstop is
also needed. Signatures are conservative — they never call non-equivalent configs
equivalent (a textual layout difference yields a different signature), so the kept
set is always a safe subset.
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
    """Adapter for the autotuner ``equivalence_fn`` hook
    (``prune_configs_by={"equivalence_fn": reduction_equivalence_key}``).

    Maps a config's compiled artifact to its static reduction-order signature, so
    the autotuner keeps only configs bitwise-equivalent to the reference (first)
    config — no kernel launch, no reference output.
    """
    return reduction_signature(asm.get("ttgir", ""))


def classify(named_ttgirs):
    """Group ``{label: ttgir}`` into equivalence classes by reduction signature.

    Returns ``OrderedDict{signature: [labels...]}`` (insertion order preserved),
    so labels sharing a signature are bitwise-equivalent reductions.
    """
    classes = OrderedDict()
    for label, ttgir in named_ttgirs.items():
        classes.setdefault(reduction_signature(ttgir), []).append(label)
    return classes
