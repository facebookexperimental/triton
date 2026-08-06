"""ISA-neutral reduction-tree canonicalizer: the post-order walks, the Merkle ``tree_hash``,
and the balanced (inner_tree) collapse that recovers the layout freedom.

Operates purely on the :mod:`bitequiv.core.treeir` node model, so it holds no PTX parser or
instruction knowledge. Moved VERBATIM out of :mod:`bitequiv.ptx.builder`, which keeps the
backward PTX def-use walk (``_Builder`` / ``build_trees`` / ``entry_signatures``) and imports
this canonicalizer."""

import hashlib

from bitequiv.core.treeir import (FpOp, ITreeReduce, Leaf, LoopReduce, Mma, OpaqueLeaf, OpaqueOp, ShflCombine,
                                  SmemExchange)

# Reduce combine ops whose BALANCED tree we collapse to a layout-invariant signature.
# add (sum), min, max: all associative + commutative, so a balanced (inner_tree) tree of them is
# layout-invariant. min/max are DELIBERATELY not in treeir._COMMUTATIVE (their children are never
# sorted) so a NaN payload rides positionally — inner_tree fixes ONE balanced tree at any layout,
# so the pairing (hence NaN propagation) is identical -> bit-identical. mul is excluded (product
# reductions are rare + carry an fma-contraction ambiguity).
_REDUCE_FP = frozenset({"add", "min", "max"})

# Reduce ops whose result is bit-identical for ANY tree shape / association order — they select an
# input verbatim and NEVER round (unlike add/mul). So a min/max reduction over one element SET is
# bit-identical regardless of layout (num_warps) or a balanced-vs-left-fold shape. This is what lets
# the EXTENT-FREE collapse merge a min/max LEFT-FOLD across num_warps (col_max), keyed on the reduced
# column SET alone. (NaN rides as "missing" in PTX min/max -> the reduction is the min/max of the
# non-NaN inputs, still order-invariant; the empirical fuzzer is the backstop.)
_ORDER_INVARIANT_OPS = frozenset({"min", "max"})

# Canonical, num_warps-INVARIANT extent key for a COMPLETE cross-thread min/max reduction (see the
# min/max branch of `collapse_balanced`). A cross-warp min/max read resolves to one representative
# partial, so the reconstructed column SET covers only one warp's slice -> num_warps-VARYING even
# though the TRUE reduced multiset is config-invariant. min/max is order- AND shape-invariant, so its
# result is fixed by that multiset alone, and every autotuner knob preserves it -> a complete
# cross-thread min/max reduction is bit-identical across all of a kernel's configs. Keying such a
# region on this constant (dropping the varying residual) recovers the num_warps freedom soundly.
_ORDER_INVARIANT_COLS = "oi"


def _reduce_op(node):
    """Reduce-op token (kind + modifiers) of a reduce node — an ``FpOp`` or a ``ShflCombine`` — or
    ``None`` for a ``SmemExchange`` (a phase marker with no op of its own; its op comes from its
    child). Unlike the old FpOp-only detection this also reads a ``ShflCombine``'s op, so a PURE
    cross-thread butterfly (1 element/thread, no within-thread fold) has a recoverable reduce op."""
    if isinstance(node, FpOp):
        return node.kind + "".join(node.mods)
    if isinstance(node, ShflCombine):
        return node.kind + "".join(node.mods)
    return None


def _forest_postorder(roots):
    """Nodes of a whole FOREST in post-order (children before parents), each visited once even
    when the roots share subtrees. Iterative, so deep folds never hit Python's recursion limit.

    Walking (and re-materializing) per root costs O(roots x nodes) on a DAG whose roots share a
    long chain — a prefix scan, where ``root[i]`` contains ``root[i-1]``, is the worst case. One
    shared walk makes it O(unique nodes). The emission ORDER differs from the concatenated
    per-root walks; that is immaterial because every consumer computes a node's value from its
    children's values, so only children-before-parents matters."""
    order, seen = [], set()
    stack = [(r, False) for r in reversed(roots)]
    while stack:
        node, done = stack.pop()
        if id(node) in seen:
            continue
        if not done:
            stack.append((node, True))
            for c in node.children:
                if id(c) not in seen:
                    stack.append((c, False))
        else:
            seen.add(id(node))
            order.append(node)
    return order


def _postorder(root):
    """Nodes of one tree in post-order (children before parents), each visited once even in a
    shared DAG."""
    return _forest_postorder([root])


def tree_sig(root):
    """Full canonical signature string of a tree (readable; for small trees / debugging).
    A shared DAG inlines a shared subtree at each reference, so this can be very large for a
    big reduction — :func:`tree_hash` is what the descriptor uses."""
    memo = {}
    for node in _postorder(root):
        memo[id(node)] = node.sig_local([memo[id(c)] for c in node.children])
    return memo[id(root)]


def tree_hashes(roots):
    """:func:`tree_hash` for EVERY root of one shared DAG, hashing each unique node ONCE.

    A per-root call gives each root a fresh memo, so a subtree shared by k roots is re-hashed k
    times. The hash is a pure bottom-up Merkle fold — a node's hash is a function of its own local
    string and its children's hashes, never of which root reached it — so one shared memo yields
    byte-identical hashes for a fraction of the work."""
    h = {}
    for node in _forest_postorder(roots):
        local = node.sig_local([h[id(c)] for c in node.children])
        h[id(node)] = hashlib.sha1(local.encode()).hexdigest()[:16]
    return [h[id(r)] for r in roots]


def tree_hash(root):
    """Compact canonical signature: a Merkle hash where each node hashes its local string
    over its children's hashes. Equal trees (including shared DAGs) -> equal hash, computed
    once per unique node, so the descriptor stays O(1)-sized regardless of reduction size."""
    return tree_hashes([root])[0]


def _coordfree_sig(node):
    """Canonical signature of a (leaf-boundary) subtree with Leaf coordinates blanked, so two
    leaves computing the same thing at different layout positions compare equal."""
    memo = {}
    for n in _postorder(node):
        memo[id(n)] = "L[]" if isinstance(n, Leaf) else n.sig_local([memo[id(c)] for c in n.children])
    return memo[id(node)]


def _leaf_cols_union(node):
    """Union of every descendant Leaf's column image, or None if any is unrecoverable."""
    out = set()
    for n in _postorder(node):
        if isinstance(n, Leaf):
            if n.cols is None:
                return None
            out |= n.cols
    return out


def _is_reduce_node(n):
    return ((isinstance(n, FpOp) and not n.fused and n.kind in _REDUCE_FP and len(n.children) == 2)
            or (isinstance(n, ShflCombine) and n.kind in _REDUCE_FP) or isinstance(n, SmemExchange))


def _balance_pass(order):
    """Per node: (balanced, height, level, optok). A subtree is a balanced reduction iff every reduce
    node has children that are balanced reductions of one consistent op AND (for the 2-ary fp
    combine) their heights differ by <= 1 — the AVL property that separates inner_tree's balanced
    tree from unordered's left-fold. Boundary leaves are trivial arity-1 balanced reductions.

    ``height`` is the REDUCTION height — log2 of the number of elements folded into the node — so a
    ``SmemExchange`` is transparent: staging a partial through shared memory does not combine
    anything, and counting it inflated the height by the number of exchanges, which is exactly a
    num_warps artifact (num_warps=1 needs no exchange at all, so the same total reduction came out
    two levels shorter and never matched the multi-warp configs). Dropping the artifact also removes
    accidental MERGES it caused, where one config's extra exchange made up for a genuinely shorter
    reduction.

    ``level`` is the raw node depth (every reduce node counts, exchanges included). It is what the
    butterfly-direction readers (:func:`_is_count_up`, :func:`_shfl_dir`) group shuffle steps by, so
    that the within-warp run stays separated from the cross-warp run by the exchange between them.

    ``order`` is a post-order node list (one tree's or a whole forest's). Every value is a pure
    function of the node's own subtree, so a forest-wide pass gives each node exactly the values a
    per-tree pass would."""
    bal, ht, lvl, optok = {}, {}, {}, {}
    for n in order:
        if _is_reduce_node(n):
            kids = list(n.children)
            ops = {optok[id(c)] for c in kids if optok[id(c)] is not None}
            self_op = _reduce_op(n)  # FpOp OR ShflCombine op (None for SmemExchange)
            if self_op is not None:
                ops.add(self_op)
            op = next(iter(ops)) if len(ops) == 1 else None  # single consistent op, else None (never empty-iter)
            kids_bal = all(bal[id(c)] for c in kids)
            if isinstance(n, FpOp) and n.kind not in _ORDER_INVARIANT_OPS:
                # add (etc): the SHAPE matters (rounding) -> require the AVL balance property, so an
                # unordered left-fold stays uncollapsed (num_warps-bearing).
                bal[id(n)] = kids_bal and op is not None and abs(ht[id(kids[0])] - ht[id(kids[1])]) <= 1
            else:  # min/max FpOp (shape-invariant), or a 1-ary shfl/smem step: no AVL check needed
                bal[id(n)] = kids_bal and op is not None
            # A shared exchange RELOCATES a partial between threads; it combines nothing, so it
            # multiplies the reduced element count by one and adds no reduction height.
            grow = 0 if isinstance(n, SmemExchange) else 1
            ht[id(n)] = max((ht[id(c)] for c in kids), default=0) + grow
            lvl[id(n)] = max((lvl[id(c)] for c in kids), default=0) + 1
            optok[id(n)] = op
        else:
            bal[id(n)], ht[id(n)], lvl[id(n)], optok[id(n)] = True, 0, 0, None
    return bal, ht, lvl, optok


def _has_opaque(node):
    return any(isinstance(n, (OpaqueLeaf, OpaqueOp)) for n in _postorder(node))


def _addr_resolved(node):
    """True iff every ``Leaf`` in ``node`` has a FULLY-affine address (no ``opq(`` token in its
    coord) — i.e. the affine engine, with its ``%tid`` bit-basis (:func:`bitequiv.ptx.affine`),
    pinned the exact element each thread loads. This is the "affine linchpin" gate for the
    single-leaf add collapse below: only when the addressing is fully resolved is the reduced
    element identity known, so a lone cross-thread add leaf may soundly collapse; an unresolved
    (opaque / swizzled) address keeps the conservative >= 2 guard (fail-closed, over-split)."""
    for n in _postorder(node):
        if isinstance(n, Leaf) and (n.coord is None or "opq(" in n.coord):
            return False
    return True


def _is_count_up(node, lvl):
    """True iff the butterfly shuffles reduce COUNT-UP — the offset DOUBLES leaf->root (1, 2, 4, 8, ...).
    That is the ``inner_tree`` signature: a count-up balanced tree fixes the reduction to the layout-
    invariant canonical order, so it is bit-identical across ``num_warps``. ``unordered`` is count-DOWN
    (16, 8, ..., 1 leaf->root). For a PURE cross-thread reduction (one boundary leaf, no within-thread
    fold to reveal the left-fold) the butterfly direction is the ONLY signal that separates the
    num_warps-invariant order from the layout-dependent one, so the single-leaf collapse MUST gate on it
    or it over-merges ``unordered`` across num_warps.

    Examine EVERY maximal run of shuffles at consecutive LEVELS — the within-warp butterfly AND the
    cross-warp run (which sits higher, separated by a shared-memory exchange). Both carry the same
    direction. inner_tree makes every run increasing; unordered makes every run decreasing. Return True
    iff at least one run of >= 2 steps is strictly increasing AND no run is non-increasing. Reading only
    the TOP run (old behavior) missed the direction at low num_warps, where the cross-warp run is a
    single (direction-less) step and only the within-warp butterfly proves count-up — so nw=2 stayed
    over-split. A count-DOWN run (unordered), a non-monotone run, a dynamic offset, or a level carrying
    two offsets -> False (cannot prove inner_tree -> do not collapse -> sound over-split). Runs are cut
    at LEVEL (not reduction height) gaps on purpose: the exchange between the within-warp and the
    cross-warp butterfly is what separates their two runs, and it carries no reduction height."""
    by_ht = {}
    for n in _region_nodes(node):
        if isinstance(n, ShflCombine):
            try:
                off = int(n.offset)
            except (TypeError, ValueError):
                return False  # dynamic / unresolved offset
            by_ht.setdefault(lvl[id(n)], set()).add(off)
    if not by_ht or any(len(offs) != 1 for offs in by_ht.values()):
        return False  # no shuffles, or a height carrying two different offsets -> ambiguous
    flat = {h: next(iter(offs)) for h, offs in by_ht.items()}
    heights = sorted(flat)
    runs, cur = [], [heights[0]]  # maximal runs of CONSECUTIVE heights (ascending height = leaf->root)
    for h in heights[1:]:
        if h == cur[-1] + 1:
            cur.append(h)
        else:
            runs.append(cur)
            cur = [h]
    runs.append(cur)
    saw_up = False
    for run in runs:
        offs = [flat[h] for h in run]  # leaf->root
        if len(offs) < 2:
            continue  # a single (cross-warp) step: direction-less on its own -> no evidence, skip
        if all(0 < offs[i] < offs[i + 1] for i in range(len(offs) - 1)):
            saw_up = True
        else:
            return False  # count-DOWN (unordered) or non-monotone -> not provably inner_tree
    return saw_up


# Pure per-element ops that are safe to keep VERBATIM inside a collapsed reduction's leaf: each is
# a deterministic function of ONE element (cast/copy/abs/neg + the transcendentals a softmax /
# rmsnorm / layernorm applies before the reduce), hence layout-invariant. Its token rides verbatim
# into leaf_sig (compared, never dropped), so equal leaf_sig => same op; a config WITHOUT the op
# gets a different leaf_sig and never merges.
_PURE_ELT = ("cvt", "mov", "abs", "neg", "ex2", "lg2", "exp", "log", "sqrt", "rsqrt", "rcp", "sin", "cos", "tanh")


def _tok_is_pure(token):
    """True iff ``token``'s OPCODE (the part before the first '.') is a pure per-element op. Matched
    at the opcode boundary (``token == p`` or ``token.startswith(p + '.')``), never a bare prefix,
    so 'neg' cannot alias a hypothetical 'negX' and 'exp' cannot alias an unrelated 'expand'."""
    return any(token == p or token.startswith(p + ".") for p in _PURE_ELT)


def _is_const_token(token):
    """True iff an ``OpaqueLeaf``'s token is a compile-time CONSTANT — a literal immediate the
    affine evaluator could not parse as an integer (``opq(imm(0f3FB8AA3B))``, a float literal) or one
    it could (a bare decimal). A constant is the same value in every thread and in every config, so
    unlike a lost-provenance value it cannot hide a layout dependence."""
    return "imm(" in token or token.lstrip("-").isdigit()


def _leaf_layout_invariant(node):
    """A reduction boundary-leaf subtree is layout-invariant (safe to collapse over) iff it has
    NO cross-thread op (Shfl/Smem), NO lost-provenance OpaqueLeaf (a literal CONSTANT is fine — see
    :func:`_is_const_token`), no fused-fma ACC-chain (a layout-dependent accumulation, not a
    per-element value), and every OpaqueOp is a pure per-element op (cvt/mov/abs/neg) kept verbatim.
    This admits the bf16 promote (cvt.f32.bf16) and a single product mul(L,L) / fma(L,L;const) while
    refusing anything whose value could depend on the thread/lane layout.

    An ``ITreeReduce`` is accepted: it is a NESTED reduction this same pass already certified as
    layout-invariant, and its token rides verbatim into the outer ``leaf_sig``, so an outer reduction
    over inner reductions (softmax's ``sum(exp(x - max(x)))``) collapses without losing the inner
    key. Callers pass the already-COLLAPSED leaf, so this is the only place such a node appears."""
    for n in _postorder(node):
        if isinstance(n, (ShflCombine, SmemExchange, Mma, LoopReduce)):
            return False  # cross-thread / opaque-chunk / loop-fold node: not a per-element leaf
        if isinstance(n, OpaqueLeaf) and not _is_const_token(n.token):
            return False
        if isinstance(n, OpaqueOp) and not _tok_is_pure(n.token):
            return False
        if isinstance(n, FpOp) and n.fused and len(n.children) == 3 and isinstance(
                n.children[2], (FpOp, ShflCombine, SmemExchange, ITreeReduce)):
            return False  # fma acc-chain (a*b + running_acc) is layout-dependent, not a leaf
    return True


def _shfl_dir(node, lvl):
    """The butterfly DIRECTION: the distinct offsets of the FIRST (min-level, nearest-leaves)
    shuffle step. count-up (inner_tree) starts at offset 1; count-down (unordered) starts at 16.
    They pair lanes differently -> different bits, so this must be in the token. It is
    num_warps-INVARIANT (the within-warp butterfly's first step is offset 1 for inner_tree at any
    num_warps; cross-warp shuffles sit at a higher level), so it distinguishes the orderings
    WITHOUT blocking num_warps recovery (the full offset sequence would, since cross-warp shuffle
    count scales with num_warps). Scoped to THIS region (:func:`_region_nodes`): a butterfly inside a
    boundary leaf belongs to that nested reduction, and letting it leak in made the direction key
    depend on which hardware form the nested reduce took (col_max's `redux.sync` at num_warps=32 vs a
    shuffle butterfly below), splitting an outer sum whose own direction is identical."""
    shfls = [(lvl[id(n)], str(n.offset)) for n in _region_nodes(node) if isinstance(n, ShflCombine)]
    if not shfls:
        return ()
    lo = min(h for h, _ in shfls)
    return tuple(sorted({off for h, off in shfls if h == lo}))


def _region_nodes(node):
    """Every reduce node of the reduction REGION rooted at ``node`` — the walk descends only through
    reduce nodes, so a nested reduction hiding inside a BOUNDARY leaf is not part of this region."""
    out, stack, seen = [], [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        for c in n.children:
            if _is_reduce_node(c):
                stack.append(c)
    return out


def _region_walk(node):
    """``(reduce op token, boundary leaves)`` of the reduction REGION rooted at ``node``, or
    ``(None, ...)`` when the region mixes two reduce ops.

    The walk descends ONLY through reduce nodes, exactly like the op propagation in
    :func:`_balance_pass`: a non-reduce child ends the region and is a boundary leaf, and whatever
    lives INSIDE that leaf is the leaf's business. Scanning the whole post-order instead (the old
    behaviour) reached back into the leaves and saw their own reduce nodes, so any reduction fed by a
    DIFFERENT-op or different-width sub-reduction — softmax's ``sum(exp(x - max(x)))``, col_max's
    within-thread ``max.f16`` fold under a cross-warp ``max.f32`` — looked like a two-op region and
    never collapsed, which is precisely where the num_warps split lived."""
    op, leaves = None, []
    for n in _region_nodes(node):
        tok = _reduce_op(n)  # FpOp OR ShflCombine (a pure cross-thread butterfly carries its op here)
        if tok is not None:
            if op is None:
                op = tok
            elif op != tok:
                return None, leaves
        for c in n.children:  # a boundary leaf is a non-reduce child of a reduce node
            if not _is_reduce_node(c):
                leaves.append(c)
    return op, leaves


def _collapse_info(node, lvl=None, collapsed=None):
    """(reduce_op_token, uniform_coord_free_leaf_sig, reduced_column_SET) for a collapsible reduction,
    or None if it cannot be SOUNDLY collapsed. The BOUNDARY leaves are the non-reduce CHILDREN of the
    region's reduce nodes (:func:`_region_walk`).

    ``collapsed`` maps a node id to its ALREADY-COLLAPSED form (the post-order output map of
    :func:`_collapse_regions`). The boundary-leaf signature is taken from there, so a leaf that is
    itself a nested reduction contributes its layout-invariant ``ITreeReduce`` token instead of its
    physical fold — which is what makes an outer reduction over it num_warps-invariant too. The
    column image and the address-resolution gate still read the RAW leaf (more leaves visible ->
    strictly more conservative).

    Guards: a single consistent reduce op (read from FpOp AND ShflCombine, so a pure cross-thread
    butterfly counts); >= 1 boundary leaf; every boundary is layout-invariant (pure-elt); the SAME
    coord-free computation across boundaries (dropping coords is then safe); the column image is
    recoverable for every leaf (addressing understood, so the config-invariant reduced element SET is
    known — the extent key for the order-invariant collapse).

    A MULTI-element boundary (``mul(a_i, b_i)``, a dot's product of two distinct arrays) is now
    admitted: within one kernel the PAIRING (which A element multiplies which B element) is fixed by
    the source and never changes with num_warps / num_stages / ordering / fp_fusion, so the multiset
    of leaf values — hence a balanced reduction over it — is config-invariant. Every leaf's column
    image being recoverable is the config-invariance proof. (The old ``_single_element`` guard refused
    this because coord-blanking cannot prove the pairing; but the pairing is a source fact, not a
    layout fact, so it is invariant across the configs actually compared.)"""
    op, raw = _region_walk(node)
    if op is None or not raw:
        return None
    leaves = [collapsed[id(l)] for l in raw] if collapsed is not None else raw
    # Leaf-count guard depends on the op's shape-sensitivity. A SHAPE-INVARIANT op (min/max) keys on
    # the reduced element SET (cols) below, so a lone coord-blanked leaf is fine — a pure cross-thread
    # min/max butterfly (1 leaf/thread) is a real reduction and MAY collapse. A SHAPE-DEPENDENT op
    # (add) keys on height + shfl_dir, NOT the element set, so a lone coord-blanked leaf would merge
    # reductions over DIFFERENT element sets that share the height/direction (they differ in bits) ->
    # keep the >= 2 guard for add UNLESS the sole leaf's address is FULLY resolved to affine
    # (`_addr_resolved`) AND the tree reduces COUNT-UP (`_is_count_up`): with the `%tid` bit-basis the
    # exact reduced element set is then pinned by the coord, and count-up is the inner_tree canonical
    # order (bit-identical across num_warps), so a pure cross-thread add for an axis-0 / outer-axis
    # reduction (sum_2d_axis0 etc., whose whole reduction is cross-thread -> one boundary leaf) soundly
    # recovers num_warps. Count-DOWN (unordered) or an UNRESOLVED (opaque/swizzled) address keeps the
    # old >= 2 fail-close (over-split, sound) — count-up vs count-down is the ONLY signal separating the
    # num_warps-invariant order from the layout-dependent one when there is no within-thread fold. This
    # is the affine linchpin. (lvl is None on the pre-collapse `_reduces_without_leaf` scan -> stay strict.)
    if (op.split(".")[0] not in _ORDER_INVARIANT_OPS and len(raw) < 2
            and not (_addr_resolved(raw[0]) and lvl is not None and _is_count_up(node, lvl))):
        return None
    if not all(_leaf_layout_invariant(l) for l in leaves):
        return None  # a layout-dependent / lost-provenance leaf -> do not collapse
    if len({_coordfree_sig(l) for l in leaves}) != 1:
        return None  # non-uniform leaf computations -> conservative
    unions = [_leaf_cols_union(l) for l in raw]  # config-invariant column image of every leaf
    if any(u is None for u in unions):
        return None  # addressing not understood for some leaf -> cannot prove the extent -> conservative
    return (op, _coordfree_sig(leaves[0]), frozenset().union(*unions))


def output_coordfree_key(node):
    """Coord-free dedup key for one output of a MULTI-output entry (a COLLAPSED tree), or None if the
    tree is not safe to coord-blank.

    A multi-output entry (softmax / layernorm / rmsnorm: one output per row, many rows per block) is
    kept VERBATIM by the descriptor today, so num_warps (which redistributes rows across threads)
    never merges. But when EVERY output is a clean per-element function of its own element over
    num_warps-invariant reductions — ``out[i] = f(x_i, ITreeReduce(row_i))`` — blanking the leaf coord
    makes all outputs share ONE key, and deduping that key across the entry gives a num_warps-invariant
    descriptor (the per-thread output COUNT varies with num_warps, but the SET of distinct output
    computations does not). An unordered per-row reduction keeps its ShflCombine offsets in the
    coord-free string (num_warps-bearing) so it still splits — inner_tree recovers, unordered does not.

    SAFE only when the output's VALUE structure is fully modeled: no lost-provenance ``OpaqueLeaf``
    (except a CONSTANT immediate, which is num_warps-invariant), and every ``OpaqueOp`` is a pure
    per-element op (cvt / exp / mov / ...). A Leaf's COORD may be opaque (a swizzle / multi-dim address
    the affine pass could not pin) — that is only ADDRESSING (which element), and blanking it is the
    whole point; faithful reconstruction already guarantees the value DEPENDENCY is captured, so what
    remains is an elementwise map over num_warps-invariant reductions. A tree with an unmodeled value
    op / lost non-constant leaf returns None and the caller falls back to the verbatim (sound,
    over-split) descriptor. This blanks more than the guarded reduction collapse (in particular it does
    not re-prove the elementwise assumption for a swizzled address), so it is gated on the empirical
    fuzzer (0 over-merge) rather than statically proven."""
    return output_coordfree_keys([node])[0]


def _coordfree_blankable(n):
    """Is THIS node (ignoring its children) safe to coord-blank? See :func:`output_coordfree_key`."""
    if isinstance(n, OpaqueLeaf) and not _is_const_token(n.token):
        return False  # a lost-provenance (non-constant) value -> unsafe to coord-blank
    if isinstance(n, OpaqueOp) and not _tok_is_pure(n.token):
        return False  # an unmodeled value op -> unsafe
    if isinstance(n, Leaf) and n.coord is None:
        return False  # no coord at all (not even opaque) -> nothing to blank soundly
    return True


def output_coordfree_keys(nodes):
    """:func:`output_coordfree_key` for EVERY output tree of one entry, sharing the safety scan and
    the signature memo across them, so a subtree several outputs share is scanned and serialized
    once instead of once per output. Both are pure bottom-up folds, so the returned keys are
    byte-identical to the per-tree calls. Signatures are built only for the trees that passed the
    scan (an unsafe tree returns ``None`` and its string is never needed)."""
    order = _forest_postorder(nodes)
    safe = {}
    for n in order:
        safe[id(n)] = _coordfree_blankable(n) and all(safe[id(c)] for c in n.children)
    sig, wanted = {}, [n for n in nodes if safe[id(n)]]
    for n in _forest_postorder(wanted):
        sig[id(n)] = "L[]" if isinstance(n, Leaf) else n.sig_local([sig[id(c)] for c in n.children])
    return [sig[id(n)] if safe[id(n)] else None for n in nodes]


def _rebuild(n, kids):
    """Reconstruct node n with already-processed children kids."""
    if isinstance(n, (Leaf, OpaqueLeaf, ITreeReduce)):
        return n
    if isinstance(n, FpOp):
        return FpOp(n.kind, n.mods, tuple(kids), n.fused)
    if isinstance(n, OpaqueOp):
        return OpaqueOp(n.token, tuple(kids))
    if isinstance(n, ShflCombine):
        return ShflCombine(n.offset, n.kind, n.mods, kids[0])
    if isinstance(n, SmemExchange):
        return SmemExchange(kids[0])
    if isinstance(n, Mma):
        return Mma(n.token, n.flags, tuple(kids))
    if isinstance(n, LoopReduce):
        return LoopReduce(n.op, kids[0], n.key)
    return n


def collapse_balanced(root):
    """Replace each MAXIMAL collapsible reduction with a layout-invariant ITreeReduce node.

    Two collapse modes, by op shape-sensitivity:
    * SHAPE-DEPENDENT (add): only a BALANCED tree collapses (inner_tree; unordered's left-fold stays
      intact), keyed on height (= log2 total elems, num_warps-invariant) + butterfly direction. A
      different chunk size (BLOCK_N) -> different height -> distinct; fp_fusion -> different leaf/fold
      structure -> distinct.
    * SHAPE-INVARIANT (min/max, which never round): ANY shape collapses (a left-fold too), keyed on the
      reduced column SET (extent-safe) with height/direction dropped. Recovers a min/max left-fold
      across num_warps WHEN the reconstructed column set is num_warps-invariant; when it is not (a
      cross-warp reduction resolved to a representative), the set varies -> distinct -> stays
      over-split (sound, no recovery).

    Iterative (no recursion on deep folds). Use :func:`collapse_balanced_forest` for a multi-output
    entry — calling this per output rebuilds every shared subtree once per output."""
    order = _postorder(root)
    ht, lvl, collapsible = _collapse_prep(order)
    marks = {}
    for n in order:
        if collapsible[id(n)]:
            for c in n.children:
                marks[id(c)] = True
    return _collapse_regions(order, [root], ht, lvl, collapsible, marks)[0]


def _collapse_prep(order):
    """(height, level, collapsible) per node of a post-order list. Collapse every MAXIMAL balanced
    reduction region, including per-chunk reductions nested under a persistent kernel's cross-chunk
    loop. The ITreeReduce token keeps the reduction height (= log2(BLOCK_N)+const,
    num_warps-invariant, BLOCK_N-monotone), so different chunk sizes stay in distinct classes."""
    bal, ht, lvl, optok = _balance_pass(order)
    return ht, lvl, {id(n): (_is_reduce_node(n) and bal[id(n)] and optok[id(n)] is not None) for n in order}


def collapse_balanced_forest(roots):
    """:func:`collapse_balanced` for EVERY output root of one value-DAG, rebuilding each unique node
    ONCE.

    A per-root call gives each root a fresh output map, so a subtree shared by k roots is
    re-materialized k times: O(roots x nodes) new node objects, all live at once (measured: a
    512-root prefix scan expanded a 3,719-node DAG into 631,169 objects, ~2.5 n^2). Every quantity
    the collapse computes is a pure bottom-up function of a node's own subtree — ``_balance_pass``,
    ``collapsible``, ``_collapse_info``, ``_shfl_dir``, ``_rebuild`` — with ONE exception: whether a
    node is a region ROOT also depends on its PARENTS, which are a per-root fact. That flag is
    resolved by :func:`_shared_region_marks`, which falls back to the exact per-root pass when the
    roots disagree."""
    order = _forest_postorder(roots)
    ht, lvl, collapsible = _collapse_prep(order)
    marks = _shared_region_marks(order, roots, collapsible)
    if marks is None:
        return [collapse_balanced(r) for r in roots]
    return _collapse_regions(order, roots, ht, lvl, collapsible, marks)


# Cap on the root x node bitmasks :func:`_shared_region_marks` builds (bits, ~8 MB). Beyond it the
# per-root pass is used instead — same descriptors, just the old cost.
_MAX_REGION_MARK_BITS = 1 << 26


def _shared_region_marks(order, roots, collapsible):
    """One ``has_coll_parent`` map valid for EVERY root, or ``None`` when the roots disagree.

    :func:`collapse_balanced` treats a collapsible node as a region ROOT only when no collapsible
    PARENT of it is reachable from that root, so the same shared node can legitimately be a region
    root under one output and be swallowed by a larger region under another. With one shared memo a
    node is rebuilt once, so whichever answer came first would silently win for everyone — that
    would move a collapse boundary and change a descriptor. Decide it exactly with two bitmasks over
    the roots: ``reach[n]`` = the roots that reach n; ``mark[n]`` = the roots that reach some
    collapsible parent of n. Root R sees ``has_coll_parent[n] = bit R of mark[n]``, so the roots
    agree on n iff ``mark[n] & reach[n]`` is empty or is all of ``reach[n]``.

    Only COLLAPSIBLE nodes are checked: ``region_root = collapsible[n] and not has_coll_parent[n]``
    short-circuits, so the flag is never read anywhere else. That matters — a prefix scan does have
    nodes whose parents differ per root, but none of them is collapsible, so the shared path stays
    live exactly where it pays off."""
    if len(roots) * len(order) > _MAX_REGION_MARK_BITS:
        return None
    reach = {}
    for i, r in enumerate(roots):
        reach[id(r)] = reach.get(id(r), 0) | (1 << i)
    for n in reversed(order):  # reversed post-order visits every parent before its children
        m = reach.get(id(n), 0)
        if m:
            for c in n.children:
                reach[id(c)] = reach.get(id(c), 0) | m
    mark = {}
    for n in order:
        if collapsible[id(n)]:
            m = reach.get(id(n), 0)
            for c in n.children:
                mark[id(c)] = mark.get(id(c), 0) | m
    out = {}
    for n in order:
        if not collapsible[id(n)]:
            continue
        m = mark.get(id(n), 0) & reach[id(n)]
        if m:
            if m != reach[id(n)]:
                return None  # some root reaches n without reaching any collapsible parent of it
            out[id(n)] = True
    return out


def _collapse_regions(order, roots, ht, lvl, collapsible, has_coll_parent):
    """Collapsed tree of each root, sharing one output map over ``order`` (see the two callers)."""
    new = {}
    for n in order:
        region_root = collapsible[id(n)] and not has_coll_parent.get(id(n), False)
        info = _collapse_info(n, lvl, new) if region_root else None
        if info is not None:
            op, leaf_sig, cols = info
            if op.split(".")[0] in _ORDER_INVARIANT_OPS:
                # min/max: SHAPE-invariant (never rounds). The physical height + butterfly direction
                # are bit-irrelevant, so DROP them and key on the reduced element SET — this collapses
                # even a LEFT-FOLD / cross-warp min/max across num_warps, while a different reduced set
                # (different extent) still yields a distinct key. BUT for a genuine CROSS-THREAD
                # reduction (a butterfly / cross-warp exchange present) the reconstructed column set is
                # itself num_warps-VARYING: a cross-warp read resolves to one representative partial, so
                # the union covers only one warp's slice (col_max: nw1=62, nw2=30, nw4=14 cols) even
                # though the TRUE reduced multiset is config-invariant. Since min/max is order- AND
                # shape-invariant its result is fixed by that multiset alone, and every autotuner knob
                # preserves the multiset (layout/order/tiling only), so drop the varying residual too
                # and key on (op, leaf_sig) via `_ORDER_INVARIANT_COLS` — recovering col_max soundly. A
                # pure within-thread min/max (no cross-thread op — a partial / elementwise max) keeps
                # its column key (conservative: it is not a complete reduction over the tile).
                cross_thread = any(isinstance(x, (ShflCombine, SmemExchange)) for x in _postorder(n))
                key = _ORDER_INVARIANT_COLS if cross_thread else _cols_key(cols)
                new[id(n)] = ITreeReduce(op, leaf_sig, 0, (), cols=key)
            else:
                # add (etc): the SHAPE matters. Keep height (= log2 of the total fan-in, which is
                # num_warps-invariant for a balanced tree — a 1-leaf pure cross-thread butterfly
                # included) + the butterfly direction, so unordered stays split and inner_tree merges.
                new[id(n)] = ITreeReduce(op, leaf_sig, ht[id(n)], _shfl_dir(n, lvl))
        else:
            new[id(n)] = _rebuild(n, [new[id(c)] for c in n.children])
    return [new[id(r)] for r in roots]


def _cols_key(cols):
    """Compact canonical key for a reduced column-image SET — the extent key of the order-invariant
    (min/max) collapse. Same reduced set across configs -> same key (merge); a different set (a
    different reduce extent) -> a different key (split)."""
    return hashlib.sha1(repr(tuple(sorted(cols))).encode()).hexdigest()[:16]
