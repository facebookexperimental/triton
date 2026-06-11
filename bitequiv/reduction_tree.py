"""Logical-relation core for TTGIR reduction equivalence (M1).

Given two compiled **TTGIR** modules, decide whether their reductions are
*bitwise-equivalent* — i.e. whether they combine the same elements in the same
floating-point association order. We answer this **statically** (no kernel launch,
no inputs, no reference output) from the layout the compiler assigned, because
after layout assignment the reduction tree is fully determined at TTGIR.

The design and the algorithm are documented for humans in
``bitequiv/design-doc.md`` (the project design doc) — keep that doc in sync when
this module's algorithm changes.

Two layers live here:

* **Conservative descriptor (production).** ``reduction_descriptor(ttgir)`` /
  ``reductions_equivalent(a, b)``. Each ``tt.reduce`` is reduced to a small,
  hashable descriptor built from the *logical relation* — the mixed-radix
  thread<->data map ``(sizePerThread, threadsPerWarp, warpsPerCTA, order)`` along
  the reduce axis, the operand **shape** along that axis, the ``reduction_ordering``
  attribute, and the combine op. We never enumerate a per-thread table. The
  descriptor includes every field that can change the tree, so it is **sound**
  (never declares two configs equivalent when their bits differ). It is
  deliberately *conservative*: it may over-split genuinely-equivalent configs
  (e.g. a 2-lane warp shuffle vs a 2-warp shared-memory combine, or masked-zero
  padding). That gap is acceptable for the first cut and is measured by the oracle.

* **Canonical-tree oracle (tests).** ``oracle_tree`` / ``trees_equivalent``
  materialize the actual parenthesization over element indices for the
  contiguous-``#blocked`` case and canonicalize it (commutative children sorted,
  trivial levels elided, phase origin erased). This is TTGIR-level ground truth
  for the association order; tests use it to confirm the descriptor is a *sound
  refinement* (``descriptor-equal => oracle-equal``) and to record exactly which
  true-equivalences the conservative descriptor misses.

Scope of the first cut: single-operand reductions over ``#blocked`` operands,
any axis (1-D / 2-D). Out of scope (conservatively handled or deferred):
multi-operand reduces (argmax/argmin/Welford), ``#mma``/``#linear`` operands
(compared verbatim), masking/identity padding, and the PTX/FMA backstop.
"""
import re
from collections import namedtuple

# `#name = #ttg.blocked<{...}>` (also #ttg.slice / #ttg.nvidia_mma / #ttg.linear ...)
_ENC_DEF = re.compile(r"^#(\w+)\s*=\s*(#ttg\.\w+<.*>)\s*$", re.M)

# A single-operand tt.reduce op. Captures (attrs, region-body, operand-shape+dtype,
# operand-encoding-name):
#   "tt.reduce"(%v) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({ ...region... })
#       : (tensor<64x1024xf32, #blocked>) -> tensor<64xf32, #ttg.slice<...>>
# NOTE: the leading `\(tensor<...>\)` with a single operand type is intentional —
# multi-operand reduces (argmax/argmin/Welford) have `(tensor<...>, tensor<...>)`
# and are *not* matched here (a named follow-up).
_REDUCE = re.compile(r'"tt\.reduce"\([^)]*\)\s*<\{([^}]*)\}>\s*\(\{(.*?)\}\)\s*:\s*'
                     r"\(tensor<([^,>]*),\s*(#\w+)>\)", re.S)


def _array(key, body):
    """Parse an MLIR int array attribute ``key = [a, b, ...]`` from ``body``."""
    m = re.search(key + r"\s*=\s*\[([0-9, ]*)\]", body)
    if not m:
        return None
    return tuple(int(v) for v in m.group(1).replace(" ", "").split(",") if v != "")


def _encoding_defs(ttgir):
    return {name: body for name, body in _ENC_DEF.findall(ttgir)}


def _parse_dims(shape_dtype):
    """``"64x1024xf32"`` -> ``(64, 1024)`` (drops the trailing element type)."""
    return tuple(int(tok) for tok in shape_dtype.split("x") if tok.isdigit())


# One parsed reduction. `dims` is the operand shape; `s_axis` its extent on the
# reduce axis; `enc_body` the operand's encoding definition body (or the name if
# the def was not found).
ReduceInfo = namedtuple("ReduceInfo", "axis ordering combine dims s_axis enc_name enc_body")


def parse_reductions(ttgir):
    """Parse every single-operand ``tt.reduce`` in ``ttgir`` into a ``ReduceInfo``.

    ``reduction_ordering`` is normalized: an absent/empty attribute means the
    default (count-down butterfly + left fold), so it is reported as
    ``"unordered"`` — identical in meaning, and treated identically downstream.
    """
    defs = _encoding_defs(ttgir)
    out = []
    for attrs, region, shape_dtype, enc_name in _REDUCE.findall(ttgir):
        am = re.search(r"axis\s*=\s*(\d+)", attrs)
        om = re.search(r'reduction_ordering\s*=\s*"(\w+)"', attrs)
        axis = int(am.group(1)) if am else None
        ordering = om.group(1) if om else "unordered"  # absent/empty == unordered
        combine = tuple(sorted(set(re.findall(r"(?:arith|math)\.\w+", region))))
        dims = _parse_dims(shape_dtype)
        s_axis = dims[axis] if (axis is not None and axis < len(dims)) else None
        out.append(ReduceInfo(axis, ordering, combine, dims, s_axis, enc_name, defs.get(enc_name.lstrip("#"),
                                                                                        enc_name)))
    return out


def _layout_term(info):
    """The layout part of the descriptor for one reduction.

    * ``inner_tree`` => layout-invariant by construction (the order is a fixed
      balanced tree over original indices), so the layout is dropped — but the
      operand extent ``s_axis`` is kept, because a different number of leaves is a
      different canonical tree.
    * ``#blocked`` => the projected per-axis radices + ``order`` + ``s_axis``. This
      fully determines the tree shape and is sound (different => different bits).
    * anything else (``#mma`` / ``#linear`` / ``#ttg.slice`` operand, or an axis we
      cannot project) => verbatim encoding text (conservative exact-match).
    """
    if info.ordering == "inner_tree":
        return ("inner_tree-invariant", info.s_axis)
    spt = _array("sizePerThread", info.enc_body)
    tpw = _array("threadsPerWarp", info.enc_body)
    wpc = _array("warpsPerCTA", info.enc_body)
    order = _array("order", info.enc_body)
    if (spt and tpw and wpc and info.axis is not None and info.axis < len(spt)):
        return ("blocked", info.s_axis, spt[info.axis], tpw[info.axis], wpc[info.axis], order)
    return ("raw", " ".join(info.enc_body.split()), info.s_axis)


def reduction_descriptor(ttgir):
    """Conservative, hashable reduction-order descriptor of a TTGIR module.

    Returns one entry per ``tt.reduce`` (in textual order):
    ``(axis, ordering, combine, layout_term)``. Two TTGIRs with equal descriptors
    perform identical reduction trees (sound); equal trees may occasionally yield
    different descriptors (conservative). ``()`` when there is no reduction.
    """
    return tuple((info.axis, info.ordering, info.combine, _layout_term(info)) for info in parse_reductions(ttgir))


def reductions_equivalent(ttgir_a, ttgir_b):
    """**The public check.** True iff the two TTGIRs reduce in the same
    (bitwise-equivalent) order, decided statically from the logical relation."""
    return reduction_descriptor(ttgir_a) == reduction_descriptor(ttgir_b)


# ---------------------------------------------------------------------------
# Canonical-tree oracle (tests only). Ground truth for the association order at
# TTGIR; not used by the production check above.
# ---------------------------------------------------------------------------
def _combine(a, b):
    """A commutative binary node. Children are canonicalized by sorting (FP add/mul
    are commutative: ``a+b == b+a`` bit-for-bit), but the binary grouping is
    preserved (``(a+b)+c != a+(b+c)``)."""
    return tuple(sorted((a, b), key=repr))


def _seq_fold(items):
    """Left fold (the unordered within-thread accumulation)."""
    acc = items[0]
    for x in items[1:]:
        acc = _combine(acc, x)
    return acc


def _balanced(items):
    """Balanced pairwise tree over ``items`` in order (the ``inner_tree`` shape)."""
    items = list(items)
    while len(items) > 1:
        items = [_combine(items[i], items[i + 1]) if i + 1 < len(items) else items[i] for i in range(0, len(items), 2)]
    return items[0]


def _butterfly(items):
    """Count-down butterfly (XOR) reduction over a power-of-two ``items``; returns
    the canonical tree accumulated at index 0. Lane ``L`` combines with ``L ^ off``
    for ``off = n/2, ..., 1`` — the unordered within-warp / cross-warp shape."""
    n = len(items)
    if n == 1:
        return items[0]
    assert n & (n - 1) == 0, "butterfly width must be a power of two"
    vals = list(items)
    off = n // 2
    while off >= 1:
        vals = [_combine(vals[L], vals[L ^ off]) for L in range(n)]
        off //= 2
    return vals[0]


def oracle_tree(c, t, w, s, ordering):
    """Materialize the canonical reduction tree over element indices ``0..s-1`` for
    the **contiguous-``#blocked``** case (``order`` has the reduce axis innermost).

    Phases mirror ``ReduceOpToLLVM.cpp``: within-thread sequential fold, within-warp
    count-down butterfly over ``t`` lanes, cross-warp count-down butterfly over
    ``w`` warps. ``inner_tree`` ignores the layout (balanced tree over original
    indices). Requires ``t``/``w`` powers of two and ``s`` divisible by ``c*t*w``
    (true for the fixtures); raises otherwise."""
    if ordering == "inner_tree":
        return _balanced(range(s))
    tile = c * t * w
    if s % tile != 0:
        raise ValueError(f"oracle: s={s} not divisible by c*t*w={tile} (masking not modeled)")
    # element i -> slot=i%c, lane=(i//c)%t, warp=(i//(c*t))%w, group=i//(c*t*w)
    per_thread = {}
    for i in range(s):
        lane = (i // c) % t
        warp = (i // (c * t)) % w
        per_thread.setdefault((warp, lane), []).append(i)
    thread_val = {k: _seq_fold(sorted(v)) for k, v in per_thread.items()}
    warp_val = [_butterfly([thread_val[(warp, lane)] for lane in range(t)]) for warp in range(w)]
    return _butterfly(warp_val)


def trees_equivalent(a, b):
    """True iff two oracle trees are the same canonical parenthesization."""
    return a == b


def oracle_tree_from_blocked(info):
    """Build the oracle tree for a parsed ``#blocked`` ``ReduceInfo`` (contiguous
    axis). Returns ``None`` if the operand is not a projectable ``#blocked`` layout."""
    spt = _array("sizePerThread", info.enc_body)
    tpw = _array("threadsPerWarp", info.enc_body)
    wpc = _array("warpsPerCTA", info.enc_body)
    if not (spt and tpw and wpc and info.axis is not None and info.axis < len(spt)):
        return None
    a = info.axis
    return oracle_tree(spt[a], tpw[a], wpc[a], info.s_axis, info.ordering)
