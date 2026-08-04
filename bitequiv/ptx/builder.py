"""Backward reduction-tree builder.

Walk backward from each ``st.global`` result through the floating-point DAG (def-use),
the butterfly shuffles, and the shared-memory exchanges, down to the global-load leaves.
Produces one tree per result store; :func:`entry_signatures` serializes them.

Both the build and the serialization are **iterative** (explicit work stacks), because a
within-thread fold over a large reduction (e.g. 8192 elements) is a left-fold chain as deep
as the element count — recursion would blow Python's stack. The build is conservative: any
value it cannot model becomes an ``OpaqueLeaf`` whose token is the structural expression, so
two trees compare equal only when provably the same computation.
"""

import hashlib

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.ptx.affine import AffineEval, canon, reqntid_of
from bitequiv.ptx.leaves import leaf_coord, leaf_columns
from bitequiv.ptx.linker import Def, DefUse
from bitequiv.ptx.treeir import (FpOp, ITreeReduce, Leaf, LoopReduce, Mma, OpaqueLeaf, OpaqueOp,
                                 ShflCombine, SmemExchange)

_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_FP_KINDS = frozenset({"add", "sub", "mul", "div", "min", "max"})

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


def _is_fp(inst):
    return bool(inst.modifiers) and inst.modifiers[-1] in _FP_WIDTHS


def _offset(operand):
    txt = getattr(operand, "text", None) or getattr(operand, "name", None) or str(operand)
    txt = txt.strip()
    return int(txt) if (txt.lstrip("-")).isdigit() else txt


class _Builder:

    def __init__(self, func):
        self.du = DefUse(func)
        _ntid = reqntid_of(func)
        self.ev = AffineEval(self.du, _ntid)
        # Separate evaluator for column-image extraction: absorbs unmodellable row math into
        # droppable symbols so an opaque row base never hides the tid-dependent column offset.
        self.colev = AffineEval(self.du, _ntid, absorb_opaque=True)
        self._memo = {}  # id(Def) -> Node
        self._plans = {}  # id(Def) -> (tag, aux, child_specs)  (child_spec: Def | Node)
        # Shared-memory stores, in ascending stream order, for exchange matching.
        self._smem_stores = [(i, inst)
                             for i, inst in enumerate(self.du.insts)
                             if inst.opcode == "st" and ".shared" in inst.modifiers]

    # -- result roots ---------------------------------------------------------

    def roots(self):
        """The stored result register(s): the value operand of each ``st.global``."""
        out = []
        for inst in self.du.insts:
            if inst.opcode == "st" and ".global" in inst.modifiers and len(inst.operands) >= 2:
                val = inst.operands[1]
                regs = (val.elements if isinstance(val, VectorOperand) else (val, ))
                for r in regs:
                    if isinstance(r, RegisterOperand):
                        out.append((r.name, self.du.index_of(inst)))
        return out

    # -- iterative build ------------------------------------------------------

    def build(self, root_reg, root_before):
        root = self._resolve(root_reg, root_before)
        if not isinstance(root, Def):
            return root  # a terminal node (no producing instruction)
        stack = [root]
        while stack:
            d = stack[-1]
            if id(d) in self._memo:
                stack.pop()
                continue
            specs = self._plan(d)
            pending = [c for c in specs if isinstance(c, Def) and id(c) not in self._memo]
            if pending:
                stack.extend(pending)
                continue
            kids = [self._memo[id(c)] if isinstance(c, Def) else c for c in specs]
            self._memo[id(d)] = self._assemble(d, kids)
            stack.pop()
        return self._memo[id(root)]

    def _resolve(self, reg, before_index):
        """A source register -> its producing ``Def``, or a terminal ``Node`` when it has
        no in-body producer."""
        d = self.du.last_writer(reg, before_index)
        return d if d is not None else OpaqueLeaf(canon(self.ev.of_reg(reg, before_index)))

    def _child(self, operand, at):
        if isinstance(operand, RegisterOperand):
            return self._resolve(operand.name, at)
        return OpaqueLeaf(canon(self.ev.of_operand(operand, at)))

    # -- planning (what are this Def's children) ------------------------------

    def _plan(self, d):
        p = self._plans.get(id(d))
        if p is not None:
            return p[2]
        inst = d.inst
        op, mods, at = inst.opcode, inst.modifiers, d.index
        if op == "ld" and ".global" in mods:
            slot = d.slot or 0
            p = ("leaf", (leaf_coord(self.ev, self.du, inst, slot), leaf_columns(self.colev, self.du, inst, slot)), [])
        elif op == "ld" and ".shared" in mods:
            p = self._plan_smem(d)
        elif op == "mov" and len(inst.operands) == 2 and isinstance(inst.operands[1], RegisterOperand):
            p = ("mov", None, [self._resolve(inst.operands[1].name, at)])
        elif op == "fma" and _is_fp(inst) and len(inst.operands) == 4:
            p = ("fma", mods, [self._child(o, at) for o in inst.operands[1:]])
        elif op in _FP_KINDS and _is_fp(inst) and len(inst.operands) == 3:
            p = self._plan_fp_binary(inst, at)
        else:
            # Unmodeled op: keep it as an internal node whose children are its register
            # operands (built iteratively); non-register operands ride in the token. This
            # faithfully captures e.g. an f64 half-assembly (or.b64) or a cvt over the
            # reduction without dragging the affine evaluator down a deep value chain.
            reg_ops = [o for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
            others = [o for o in inst.operands[1:] if not isinstance(o, RegisterOperand)]
            tok = inst.opcode + "".join(inst.modifiers) + ("" if d.slot is None else f"#{d.slot}")
            if others:
                tok += "{" + ",".join(canon(self.ev.of_operand(o, at)) for o in others) + "}"
            p = ("opaqueop", tok, [self._resolve(o.name, at) for o in reg_ops])
        self._plans[id(d)] = p
        return p[2]

    def _plan_fp_binary(self, inst, at):
        a, b = inst.operands[1], inst.operands[2]
        if inst.opcode in _REDUCE_FP:  # a butterfly `op(p, shfl.bfly(p,off))` for any reduce op
            shf = self._shuffle_of(a, b, at) or self._shuffle_of(b, a, at)
            if shf is not None:
                partial_reg, off = shf
                return ("shfl", (off, inst.opcode, inst.modifiers), [self._resolve(partial_reg, at)])
        return (inst.opcode, inst.modifiers, [self._child(a, at), self._child(b, at)])

    def _shuffle_of(self, maybe_shfl, partial, at):
        """If ``maybe_shfl`` is ``shfl.bfly(partial, OFF)``, return (partial_reg, OFF)."""
        if not isinstance(maybe_shfl, RegisterOperand) or not isinstance(partial, RegisterOperand):
            return None
        d = self.du.last_writer(maybe_shfl.name, at)
        if d is None or d.inst.opcode != "shfl" or ".bfly" not in d.inst.modifiers:
            return None
        ops = d.inst.operands
        if len(ops) >= 3 and isinstance(ops[1], RegisterOperand) and ops[1].name == partial.name:
            return partial.name, _offset(ops[2])
        return None

    def _plan_smem(self, d):
        """Match a shared load to the most recent prior shared store (single-reduction
        canonical pattern); the stored value's subtree is the exchange child."""
        store = None
        for i, inst in self._smem_stores:
            if i < d.index:
                store = inst
            else:
                break
        if store is None or len(store.operands) < 2 or not isinstance(store.operands[1], RegisterOperand):
            return ("opaque", "smem-unmatched", [])
        return ("smem", None, [self._resolve(store.operands[1].name, self.du.index_of(store))])

    # -- assembly (build this Def's Node from its built children) --------------

    def _assemble(self, d, kids):
        tag, aux, _ = self._plans[id(d)]
        if tag == "leaf":
            coord, cols = aux
            return Leaf(coord, cols)
        if tag == "opaque":
            return OpaqueLeaf(aux)
        if tag == "mov":
            return kids[0]
        if tag == "smem":
            return SmemExchange(kids[0])
        if tag == "shfl":
            off, kind, mods = aux
            return ShflCombine(off, kind, mods, kids[0])
        if tag == "fma":
            return FpOp("fma", aux, tuple(kids), fused=True)
        if tag == "opaqueop":
            return OpaqueOp(aux, tuple(kids))
        return FpOp(tag, aux, tuple(kids))  # tag == fp kind (add/sub/mul/div/min/max)


def _postorder(root):
    """Nodes of a tree in post-order (children before parents), each visited once even in a
    shared DAG. Iterative, so deep folds never hit Python's recursion limit."""
    order, seen = [], set()
    stack = [(root, False)]
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


def tree_sig(root):
    """Full canonical signature string of a tree (readable; for small trees / debugging).
    A shared DAG inlines a shared subtree at each reference, so this can be very large for a
    big reduction — :func:`tree_hash` is what the descriptor uses."""
    memo = {}
    for node in _postorder(root):
        memo[id(node)] = node.sig_local([memo[id(c)] for c in node.children])
    return memo[id(root)]


def tree_hash(root):
    """Compact canonical signature: a Merkle hash where each node hashes its local string
    over its children's hashes. Equal trees (including shared DAGs) -> equal hash, computed
    once per unique node, so the descriptor stays O(1)-sized regardless of reduction size."""
    h = {}
    for node in _postorder(root):
        local = node.sig_local([h[id(c)] for c in node.children])
        h[id(node)] = hashlib.sha1(local.encode()).hexdigest()[:16]
    return h[id(root)]


def build_trees(func):
    """List of reconstructed reduction-tree roots for one ``.entry`` (one per result store)."""
    b = _Builder(func)
    return [b.build(reg, idx) for reg, idx in b.roots()]


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


def _balance_pass(root):
    """Per node: (balanced, height, optok). A subtree is a balanced reduction iff every reduce
    node has children that are balanced reductions of one consistent op AND (for the 2-ary fp
    combine) their heights differ by <= 1 — the AVL property that separates inner_tree's balanced
    tree from unordered's left-fold. Boundary leaves are trivial arity-1 balanced reductions."""
    bal, ht, optok = {}, {}, {}
    for n in _postorder(root):
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
            ht[id(n)] = max((ht[id(c)] for c in kids), default=0) + 1
            optok[id(n)] = op
        else:
            bal[id(n)], ht[id(n)], optok[id(n)] = True, 0, None
    return bal, ht, optok


def _has_opaque(node):
    return any(isinstance(n, (OpaqueLeaf, OpaqueOp)) for n in _postorder(node))


# Pure per-element ops that are safe to keep VERBATIM inside a collapsed reduction's leaf: each is
# a deterministic function of ONE element (cast/copy/abs/neg + the transcendentals a softmax /
# rmsnorm / layernorm applies before the reduce), hence layout-invariant. Its token rides verbatim
# into leaf_sig (compared, never dropped), so equal leaf_sig => same op; a config WITHOUT the op
# gets a different leaf_sig and never merges.
_PURE_ELT = ("cvt", "mov", "abs", "neg", "ex2", "lg2", "exp", "log",
             "sqrt", "rsqrt", "rcp", "sin", "cos", "tanh")


def _tok_is_pure(token):
    """True iff ``token``'s OPCODE (the part before the first '.') is a pure per-element op. Matched
    at the opcode boundary (``token == p`` or ``token.startswith(p + '.')``), never a bare prefix,
    so 'neg' cannot alias a hypothetical 'negX' and 'exp' cannot alias an unrelated 'expand'."""
    return any(token == p or token.startswith(p + ".") for p in _PURE_ELT)


def _leaf_layout_invariant(node):
    """A reduction boundary-leaf subtree is layout-invariant (safe to collapse over) iff it has
    NO cross-thread op (Shfl/Smem), NO lost-provenance OpaqueLeaf, no fused-fma ACC-chain (a
    layout-dependent accumulation, not a per-element value), and every OpaqueOp is a pure
    per-element op (cvt/mov/abs/neg) kept verbatim. This admits the bf16 promote (cvt.f32.bf16)
    and a single product mul(L,L) / fma(L,L;const) while refusing anything whose value could
    depend on the thread/lane layout."""
    for n in _postorder(node):
        if isinstance(n, (ShflCombine, SmemExchange, Mma, LoopReduce)):
            return False  # cross-thread / opaque-chunk / loop-fold node: not a per-element leaf
        if isinstance(n, OpaqueLeaf):
            return False
        if isinstance(n, OpaqueOp) and not _tok_is_pure(n.token):
            return False
        if isinstance(n, FpOp) and n.fused and len(n.children) == 3 and isinstance(
                n.children[2], (FpOp, ShflCombine, SmemExchange, ITreeReduce)):
            return False  # fma acc-chain (a*b + running_acc) is layout-dependent, not a leaf
    return True


def _shfl_dir(node, ht):
    """The butterfly DIRECTION: the distinct offsets of the FIRST (min-height, nearest-leaves)
    shuffle step. count-up (inner_tree) starts at offset 1; count-down (unordered) starts at 16.
    They pair lanes differently -> different bits, so this must be in the token. It is
    num_warps-INVARIANT (the within-warp butterfly's first step is offset 1 for inner_tree at any
    num_warps; cross-warp shuffles sit at higher height), so it distinguishes the orderings
    WITHOUT blocking num_warps recovery (the full offset sequence would, since cross-warp shuffle
    count scales with num_warps)."""
    shfls = [(ht[id(n)], str(n.offset)) for n in _postorder(node) if isinstance(n, ShflCombine)]
    if not shfls:
        return ()
    lo = min(h for h, _ in shfls)
    return tuple(sorted({off for h, off in shfls if h == lo}))


def _collapse_info(node):
    """(reduce_op_token, uniform_coord_free_leaf_sig, reduced_column_SET) for a collapsible reduction,
    or None if it cannot be SOUNDLY collapsed. The BOUNDARY leaves are the non-reduce CHILDREN of
    reduce nodes (NOT every non-reduce postorder node — collecting the latter double-counts a
    boundary's own internal nodes, e.g. it sees both ``cvt(L)`` and its child ``L`` -> two coord-free
    sigs -> a spurious bail; that is why the baseline never collapsed cvt/exp/mul-fed reductions).

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
    op, leaves = None, []
    for n in _postorder(node):
        if not _is_reduce_node(n):
            continue
        tok = _reduce_op(n)  # FpOp OR ShflCombine (a pure cross-thread butterfly carries its op here)
        if tok is not None:
            if op is None:
                op = tok
            elif op != tok:
                return None
        for c in n.children:  # a boundary leaf is a non-reduce child of a reduce node
            if not _is_reduce_node(c):
                leaves.append(c)
    if op is None or not leaves:
        return None
    # Leaf-count guard depends on the op's shape-sensitivity. A SHAPE-INVARIANT op (min/max) keys on
    # the reduced element SET (cols) below, so a lone coord-blanked leaf is fine — a pure cross-thread
    # min/max butterfly (1 leaf/thread) is a real reduction and MAY collapse. A SHAPE-DEPENDENT op
    # (add) keys on height + shfl_dir, NOT the element set, so a lone coord-blanked leaf would merge
    # reductions over DIFFERENT element sets that share the height/direction (they differ in bits) ->
    # keep the >= 2 guard for add (a within-thread fold pins the layout enough for the eval's
    # same-element-set configs; a 1-leaf pure cross-thread add stays over-split, sound).
    if op.split(".")[0] not in _ORDER_INVARIANT_OPS and len(leaves) < 2:
        return None
    if not all(_leaf_layout_invariant(l) for l in leaves):
        return None  # a layout-dependent / lost-provenance leaf -> do not collapse
    if len({_coordfree_sig(l) for l in leaves}) != 1:
        return None  # non-uniform leaf computations -> conservative
    unions = [_leaf_cols_union(l) for l in leaves]  # config-invariant column image of every leaf
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
    for n in _postorder(node):
        if isinstance(n, OpaqueLeaf) and "imm(" not in n.token:
            return None  # a lost-provenance (non-constant) value -> unsafe to coord-blank
        if isinstance(n, OpaqueOp) and not _tok_is_pure(n.token):
            return None  # an unmodeled value op -> unsafe
        if isinstance(n, Leaf) and n.coord is None:
            return None  # no coord at all (not even opaque) -> nothing to blank soundly
    return _coordfree_sig(node)


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

    Iterative (no recursion on deep folds)."""
    bal, ht, optok = _balance_pass(root)
    # Collapse every MAXIMAL balanced reduction region, including per-chunk reductions nested
    # under a persistent kernel's cross-chunk loop. The ITreeReduce token keeps the reduction
    # height (= log2(BLOCK_N)+const, num_warps-invariant, BLOCK_N-monotone), so different chunk
    # sizes stay in distinct classes.
    collapsible = {id(n): (_is_reduce_node(n) and bal[id(n)] and optok[id(n)] is not None) for n in _postorder(root)}
    has_coll_parent = {}
    for n in _postorder(root):
        if collapsible[id(n)]:
            for c in n.children:
                has_coll_parent[id(c)] = True
    new = {}
    for n in _postorder(root):
        region_root = collapsible[id(n)] and not has_coll_parent.get(id(n), False)
        info = _collapse_info(n) if region_root else None
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
                new[id(n)] = ITreeReduce(op, leaf_sig, ht[id(n)], _shfl_dir(n, ht))
        else:
            new[id(n)] = _rebuild(n, [new[id(c)] for c in n.children])
    return new[id(root)]


def _cols_key(cols):
    """Compact canonical key for a reduced column-image SET — the extent key of the order-invariant
    (min/max) collapse. Same reduced set across configs -> same key (merge); a different set (a
    different reduce extent) -> a different key (split)."""
    return hashlib.sha1(repr(tuple(sorted(cols))).encode()).hexdigest()[:16]


def entry_signatures(func):
    """Canonical, sorted, compact tree signatures for one ``.entry`` (one per result store).

    G3 (2-D / multi-output guard): the layout-drop collapse is num_warps-invariant ONLY when ALL
    of the entry's threads reduce into ONE output. With MULTIPLE outputs (e.g. a 2-D tile reduced
    per-row, ROWS_PER_BLOCK>1), the threads/lanes are PARTITIONED among the outputs and num_warps
    re-partitions them, re-associating each row's sum differently -> different bits. Collapsing
    there over-merges (measured: reduce2d merges unordered~inner_tree). So collapse only
    single-output entries; multi-output entries keep their verbatim layout-bearing trees (sound,
    over-split)."""
    roots = build_trees(func)
    if len(roots) == 1:
        roots = [collapse_balanced(roots[0])]
    return tuple(sorted(tree_hash(t) for t in roots))
