# The reduction collapse is MOSTLY architecture-neutral, so the AMD checker REUSES the shared
# `bitequiv.core.canonicalize` (extracted on the NV side, D114729647) and keeps ONLY the AMD-specific
# pieces here: the divergent leaf predicates (`_is_const_token` adds `fpconst:`, `_tok_is_pure` OR's
# AMD mnemonic prefixes, `_is_count_up` dedups packed offsets), every function that transitively
# CALLS them (`_leaf_layout_invariant`/`_coordfree_blankable`/`_collapse_info`/`_collapse_regions`/
# `output_coordfree_key(s)`/`collapse_balanced`/`collapse_balanced_forest` — Python binds those calls
# to THIS module's names, so importing the core versions would silently use the NV predicates), and
# the AMD-only cross-warp recovery hooks (`_dir_key`, `_fold_equal_height`, `_collapse_complete`,
# `_reduction_extent`, ...). Nothing here is pushed back into `bitequiv.core` — NV has priority.
"""AMD-specific reduction collapse layered on the shared `bitequiv.core` tree IR + helpers.

See the module comment above for exactly which symbols are AMD-local and why. Everything else is
imported from `bitequiv.core.canonicalize`; a few of those (`_coordfree_sig`, `_forest_postorder`,
`tree_hash`, `tree_hashes`) are re-exported because the AMD forward interpreter imports them from
this module path."""
from bitequiv.core.canonicalize import (  # noqa: F401 -- several are re-exported for the interpreter
    _ORDER_INVARIANT_COLS,
    _ORDER_INVARIANT_OPS,
    _PURE_ELT,
    _addr_resolved,
    _balance_pass,
    _collapse_prep,
    _cols_key,
    _coordfree_sig,
    _forest_postorder,
    _is_reduce_node,
    _leaf_cols_union,
    _postorder,
    _rebuild,
    _region_nodes,
    _region_walk,
    _shared_region_marks,
    _shfl_dir,
    tree_hash,
    tree_hashes,
)

from bitequiv.amdgcn.core.treeir import (
    FpOp,
    ITreeReduce,
    Leaf,
    LoopReduce,
    Mma,
    OpaqueLeaf,
    OpaqueOp,
    ShflCombine,
    SmemExchange,
)



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
        # Dedup CONSECUTIVE-equal offsets: a packed / SIMD-2 (v_pk) reduction repeats each butterfly
        # step once per lane, so the same offset lands at consecutive levels (1,1,2,2,4,4,... instead
        # of 1,2,4,...). That is still count-up; deduping recovers the true doubling sequence. A
        # count-DOWN run (32,32,16,16,...) dedups to 32,16,... and still fails the strict-increase
        # check, so unordered is NOT wrongly admitted.
        ded = [offs[0]]
        for o in offs[1:]:
            if o != ded[-1]:
                ded.append(o)
        if len(ded) < 2:
            continue  # a single (cross-warp) step: direction-less on its own -> no evidence, skip
        if all(0 < ded[i] < ded[i + 1] for i in range(len(ded) - 1)):
            saw_up = True
        else:
            return False  # count-DOWN (unordered) or non-monotone -> not provably inner_tree
    return saw_up


# Pure per-element ops that are safe to keep VERBATIM inside a collapsed reduction's leaf: each is
# a deterministic function of ONE element (cast/copy/abs/neg + the transcendentals a softmax /
# rmsnorm / layernorm applies before the reduce), hence layout-invariant. Its token rides verbatim
# into leaf_sig (compared, never dropped), so equal leaf_sig => same op; a config WITHOUT the op
# gets a different leaf_sig and never merges.


_AMD_PURE_PREFIXES = (
    "v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_", "v_cvt_",
    "v_rndne_", "v_floor_", "v_ceil_", "v_trunc_", "v_fract_",
    # exp / div / rsqrt SOFTWARE-sequence primitives — each is a deterministic per-element step of a
    # unary transcendental (range reduction + polynomial + fixup), so it is layout-invariant. Lets the
    # multi-output coord-free key see through AMD's software exp/div (NV has 1-op ex2/rcp). Fuzzer-gated.
    "v_ldexp_", "v_frexp_", "v_cndmask_", "v_cmp_", "v_div_scale_", "v_div_fmas_", "v_div_fixup_",
    "v_med3_", "v_bfe_", "v_bfrev_", "v_readfirstlane_", "v_not_", "v_and_b32", "v_or_b32", "v_xor_b32",
    "v_lshlrev_", "v_lshrrev_", "v_ashrrev_",
)


def _tok_is_pure(token):
    """True iff ``token``'s OPCODE is a pure per-element op. PTX form matched at the '.' boundary
    (``token == p`` / ``token.startswith(p + '.')``); AMD form matched by opcode prefix."""
    return (any(token == p or token.startswith(p + ".") for p in _PURE_ELT)
            or token.startswith(_AMD_PURE_PREFIXES))


def _is_const_token(token):
    """True iff an ``OpaqueLeaf``'s token is a compile-time CONSTANT — a literal immediate the
    affine evaluator could not parse as an integer (``opq(imm(0f3FB8AA3B))``, a float literal) or one
    it could (a bare decimal). A constant is the same value in every thread and in every config, so
    unlike a lost-provenance value it cannot hide a layout dependence. ``fpconst:`` is a decoded
    float constant the interpreter recognized (e.g. the ``-0.0`` reduction identity from
    ``v_bfrev_b32 R, 1``)."""
    return "imm(" in token or token.startswith("fpconst:") or token.lstrip("-").isdigit()


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


def _dir_key(node, lvl):
    """The num_warps-INVARIANT direction key for an add reduction's ITreeReduce token. When the
    butterfly is provably count-up (inner_tree, :func:`_is_count_up`) the RESULT is num_warps-
    invariant regardless of the physical first offset, so we key on a canonical ``("up",)`` marker
    instead of :func:`_shfl_dir` — whose raw first offset itself VARIES across num_warps for a
    cross-warp reduction (a 2D axis-0 sum starts its within-warp butterfly at offset 8 for num_warps=1
    but offset 1 for num_warps=16, which spuriously split two bit-identical inner_tree configs). A
    non-count-up (unordered / unprovable) reduction keeps the raw ``_shfl_dir`` offsets, so unordered
    still splits by num_warps (its bits DO vary) and never merges with a count-up sibling."""
    return ("up", ) if _is_count_up(node, lvl) else _shfl_dir(node, lvl)


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
    out = _collapse_regions(order, [root], ht, lvl, collapsible, marks)
    return _collapse_complete(_fold_equal_height(out))[0]


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
    return _collapse_complete(_fold_equal_height(_collapse_regions(order, roots, ht, lvl, collapsible, marks)))


# Cap on the root x node bitmasks :func:`_shared_region_marks` builds (bits, ~8 MB). Beyond it the
# per-root pass is used instead — same descriptors, just the old cost.


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
                # The equal-height Horner fold (:func:`_fold_equal_height`) later merges two of these
                # into one level up, recovering the num_warps freedom on a reconstructed add-chain.
                # _dir_key: canonical ("up",) for count-up (its raw first offset varies with num_warps
                # on a cross-warp reduction), raw offsets otherwise -> unordered still splits.
                new[id(n)] = ITreeReduce(op, leaf_sig, ht[id(n)], _dir_key(n, lvl))
        else:
            new[id(n)] = _rebuild(n, [new[id(c)] for c in n.children])
    return [new[id(r)] for r in roots]


def _is_add_neutral(n):
    """True iff ``n`` is the additive identity ``-0.0`` (``x + -0.0 == x`` for EVERY x, including the
    signed zeros — unlike ``+0.0``, which flips ``-0.0 + +0.0`` to ``+0.0``). Codegen seeds a sum
    accumulator with ``-0.0`` via ``v_bfrev_b32 R, 1`` (bit-reverse of 1 = ``0x80000000``); the
    interpreter tags that as ``fpconst:-0.0``. Stripping it is bit-exact and it otherwise breaks the
    Horner-chain fold below (it sits between two equal-height partials)."""
    return isinstance(n, OpaqueLeaf) and n.token == "fpconst:-0.0"


def _merge_equal_itree(a, b, mods):
    """If ``add(a, b)`` provably equals ONE balanced count-up (inner_tree) reduction of height+1,
    return that ``ITreeReduce``; else ``None``.

    A balanced inner_tree over 2^(h+1) elements IS ``add(left_half, right_half)`` where each half is
    the balanced count-up reduction of 2^h elements. So two equal-height count-up add ``ITreeReduce``
    of the SAME leaf computation, added, ARE the next level up — this is exactly the inner_tree
    contract the height-keyed collapse already trusts (same height + count-up + leaf_sig => same
    bits). Applied bottom-up it folds a left-leaning Horner chain ``(((h+h)+h1)+h2)+...`` (how a low
    num_warps reconstructs the same logical tree) into the single balanced node a high num_warps
    yields — the last num_warps split on plain sums.

    Guards (all necessary for soundness): both are add ``ITreeReduce`` (not the min/max cols path);
    same op + leaf_sig + height + butterfly direction; count-up (first shuffle offset ``1`` — an
    unordered/count-down fold must NOT merge); distinct objects (``a is b`` would be ``x+x`` doubling,
    not a two-half partition). This is the SAME trust the height-keyed collapse already relies on:
    under the inner_tree contract two distinct partials of equal height + same leaf computation reduce
    DISJOINT halves of the logical set (the codegen always partitions), so merging them to height+1 is
    bit-exact. The element-column image is NOT used here — for a reduction both halves write the SAME
    output column, and that image is itself num_warps-varying, so it cannot prove a partition."""
    if not (isinstance(a, ITreeReduce) and isinstance(b, ITreeReduce)) or a is b:
        return None
    if a.cols is not None or b.cols is not None:
        return None  # min/max shape-invariant nodes key on the set, not height -> not this fold
    if a.op != b.op or not a.op.startswith("add") or a.leaf_sig != b.leaf_sig:
        return None
    if a.height != b.height or a.shfl_seq != b.shfl_seq:
        return None
    if a.shfl_seq not in (("up", ), ("1", )):
        return None  # only count-up (inner_tree) folds; unordered keeps its physical shape. ("up",)
        #             is the canonical count-up marker (_dir_key); ("1",) a raw count-up first offset.
    return ITreeReduce(a.op, a.leaf_sig, a.height + 1, a.shfl_seq)


def _fold_equal_height(roots):
    """Bottom-up Horner-chain fold over already-collapsed trees (:func:`_merge_equal_itree` +
    additive-neutral strip). Memoized over the forest so a shared subtree folds once; byte-identical
    to per-root folding. Recovers the num_warps freedom on a plain sum whose low-num_warps codegen
    reconstructs the balanced tree as a left-leaning add-chain of unequal-height partials."""
    new = {}
    for n in _forest_postorder(roots):
        kids = [new[id(c)] for c in n.children]
        folded = None
        if isinstance(n, FpOp) and n.kind == "add" and not n.fused and len(kids) == 2:
            a, b = kids
            if _is_add_neutral(a):
                folded = b
            elif _is_add_neutral(b):
                folded = a
            else:
                folded = _merge_equal_itree(a, b, n.mods)
        new[id(n)] = folded if folded is not None else _rebuild(n, kids)
    return [new[id(r)] for r in roots]


def _reduction_extent(root):
    """Total elements a collapsed reduction subtree reduces — the num_warps-INVARIANT extent (a low
    num_warps folds more within-thread + fewer cross-warp exchanges, a high one the reverse, but the
    PRODUCT is the fixed tile size). Recurses only through reduce nodes: a butterfly ``ShflCombine``
    doubles the fan-in (combine with a shuffled copy), a fan-in-1 ``SmemExchange`` relocates (no
    change), an ``add`` sums its children, a boundary leaf is one value and a nested ``ITreeReduce``
    (add) is its ``2**height``. ``None`` if any node is not a pure-add reduction step (fma/mul/opaque
    -> cannot count soundly)."""
    order, stack, seen = [], [(root, False)], set()
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
            continue
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.append((n, True))
        if _is_reduce_node(n) and not isinstance(n, ITreeReduce):
            for c in n.children:
                stack.append((c, False))
    ext = {}
    for n in order:
        if isinstance(n, ITreeReduce):
            ext[id(n)] = (1 << n.height) if n.cols is None else None  # min/max cols node: no count
        elif not _is_reduce_node(n):
            ext[id(n)] = 1  # a boundary leaf (loaded element / product) is one reduced value
        elif isinstance(n, ShflCombine):
            c = ext.get(id(n.children[0]))
            ext[id(n)] = None if c is None else 2 * c
        elif isinstance(n, SmemExchange):
            ext[id(n)] = ext.get(id(n.children[0]))
        elif isinstance(n, FpOp) and n.kind == "add" and not n.fused:
            cs = [ext.get(id(c)) for c in n.children]
            ext[id(n)] = None if any(c is None for c in cs) else sum(cs)
        else:
            ext[id(n)] = None
    return ext[id(root)]


def _complete_countup_info(node, lvl):
    """(op, uniform_leaf_sig, log2_extent) for a COMPLETE count-up (inner_tree) reduction whose
    physical shape (within-thread fold split vs cross-warp exchange count) varies with num_warps but
    whose reduced element SET is the fixed tile — or ``None``.

    This is the cross-warp analog of the balanced within-warp collapse: an inner_tree reduction is
    bit-determined by (op, leaf computation, element COUNT) regardless of how the count is split
    across threads, so keying on the num_warps-invariant extent (:func:`_reduction_extent`) recovers
    num_warps even when the reconstructed tree is an unbalanced pipelined/packed/butterfly mix that
    neither the balanced collapse nor the Horner fold can flatten. Gates (all needed for soundness):
    a single add op; uniform coord-free leaf computation; provably count-up (``_is_count_up`` -> NOT
    unordered, whose bits DO depend on num_warps); a clean power-of-two extent (an unmodelable fan-in
    makes it ``None`` / non-pow2 -> fail closed). Emitted as the SAME ``ITreeReduce`` key a balanced
    tree of that height uses, so a within-warp and a cross-warp reduction of the same extent + leaf
    merge (correct: inner_tree makes them bit-identical)."""
    op, raw = _region_walk(node)
    if op is None or not raw or op.split(".")[0] in _ORDER_INVARIANT_OPS:
        return None
    if not all(_leaf_layout_invariant(l) for l in raw):
        return None
    if len({_coordfree_sig(l) for l in raw}) != 1:
        return None
    if not _is_count_up(node, lvl):
        return None  # unordered / unprovable direction -> keep physical shape (sound over-split)
    ext = _reduction_extent(node)
    if ext is None or ext < 2 or (ext & (ext - 1)) != 0:
        return None  # need a clean power-of-two total fan-in
    return (op, _coordfree_sig(raw[0]), ext.bit_length() - 1)


def _collapse_complete(roots):
    """Post-pass that collapses a COMPLETE count-up reduction the balanced pass + Horner fold left
    uncollapsed (a cross-warp pipelined/packed mix) to one extent-keyed ``ITreeReduce``. Region roots
    = reduce nodes not consumed by a reduce parent; ``lvl`` (butterfly level) is read on the INPUT
    trees, so the replacement is decided before any rebuild disturbs node identity."""
    order = _forest_postorder(roots)
    _, _, lvl, _ = _balance_pass(order)
    parent_reduce = set()
    for n in order:
        if _is_reduce_node(n):
            for c in n.children:
                if _is_reduce_node(c):
                    parent_reduce.add(id(c))
    replace = {}
    for n in order:
        if _is_reduce_node(n) and not isinstance(n, ITreeReduce) and id(n) not in parent_reduce:
            info = _complete_countup_info(n, lvl)
            if info is not None:
                op, leaf_sig, height = info
                replace[id(n)] = ITreeReduce(op, leaf_sig, height, ("up", ))
    if not replace:
        return roots
    new = {}
    for n in order:
        new[id(n)] = replace[id(n)] if id(n) in replace else _rebuild(n, [new[id(c)] for c in n.children])
    return [new[id(r)] for r in roots]
