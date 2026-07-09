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
from bitequiv.ptx.treeir import FpOp, ITreeReduce, Leaf, OpaqueLeaf, OpaqueOp, ShflCombine, SmemExchange

_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_FP_KINDS = frozenset({"add", "sub", "mul", "div", "min", "max"})

# Reduce combine ops whose BALANCED tree we collapse to a layout-invariant signature.
# Restricted to add (sum-style) for now; mul/min/max are future work.
_REDUCE_FP = frozenset({"add"})


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
        if inst.opcode == "add":
            shf = self._shuffle_of(a, b, at) or self._shuffle_of(b, a, at)
            if shf is not None:
                partial_reg, off = shf
                return ("shfl", (off, inst.modifiers), [self._resolve(partial_reg, at)])
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
            off, mods = aux
            return ShflCombine(off, "add", mods, kids[0])
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
            if isinstance(n, FpOp):
                ops.add(n.kind + "".join(n.mods))
            op = next(iter(ops)) if len(ops) == 1 else None  # single consistent op, else None (never empty-iter)
            kids_bal = all(bal[id(c)] for c in kids)
            if isinstance(n, FpOp):
                bal[id(n)] = kids_bal and op is not None and abs(ht[id(kids[0])] - ht[id(kids[1])]) <= 1
            else:  # shfl / smem: 1-ary cross-thread step; balance carried from its child
                bal[id(n)] = kids_bal and op is not None
            ht[id(n)] = max((ht[id(c)] for c in kids), default=0) + 1
            optok[id(n)] = op
        else:
            bal[id(n)], ht[id(n)], optok[id(n)] = True, 0, None
    return bal, ht, optok


def _has_opaque(node):
    return any(isinstance(n, (OpaqueLeaf, OpaqueOp)) for n in _postorder(node))


# Pure per-element ops that are safe to keep VERBATIM inside a collapsed reduction's leaf: they
# are deterministic functions of one element (cast/copy/abs/neg), hence layout-invariant. Their
# token rides verbatim into leaf_sig (compared, never dropped), so equal leaf_sig => same op.
_PURE_ELT = ("cvt", "mov", "abs", "neg")


def _leaf_layout_invariant(node):
    """A reduction boundary-leaf subtree is layout-invariant (safe to collapse over) iff it has
    NO cross-thread op (Shfl/Smem), NO lost-provenance OpaqueLeaf, no fused-fma ACC-chain (a
    layout-dependent accumulation, not a per-element value), and every OpaqueOp is a pure
    per-element op (cvt/mov/abs/neg) kept verbatim. This admits the bf16 promote (cvt.f32.bf16)
    and a single product mul(L,L) / fma(L,L;const) while refusing anything whose value could
    depend on the thread/lane layout."""
    for n in _postorder(node):
        if isinstance(n, (ShflCombine, SmemExchange)):
            return False
        if isinstance(n, OpaqueLeaf):
            return False
        if isinstance(n, OpaqueOp) and not any(n.token.startswith(p) for p in _PURE_ELT):
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
    """(reduce_op_token, uniform_coord_free_leaf_sig) for a balanced reduction, or None if it
    cannot be SOUNDLY collapsed. Guards: a single consistent reduce op; >= 2 boundary leaves;
    every boundary leaf has the SAME coord-free computation (so dropping coords is safe); and NO
    opaque/unmodelled node anywhere (we only collapse fully-understood reductions). Soundness
    also relies on the column image being recoverable for every leaf (proves the addressing is
    understood), even though the size key itself uses the layout-invariant tree height."""
    op, leaves = None, []
    for n in _postorder(node):
        if isinstance(n, FpOp) and not n.fused and n.kind in _REDUCE_FP and len(n.children) == 2:
            tok = n.kind + "".join(n.mods)
            if op is None:
                op = tok
            elif op != tok:
                return None
        elif _is_reduce_node(n):
            continue  # shfl / smem structural reduce node
        else:
            leaves.append(n)  # boundary leaf computation
    if op is None or len(leaves) < 2:
        return None
    if not all(_leaf_layout_invariant(l) for l in leaves):
        return None  # a layout-dependent / lost-provenance leaf -> do not collapse
    if len({_coordfree_sig(l) for l in leaves}) != 1:
        return None  # non-uniform leaf computations -> conservative
    for l in leaves:  # require addressing understood for every leaf (column image recoverable)
        if _leaf_cols_union(l) is None:
            return None
    return (op, _coordfree_sig(leaves[0]))


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
    return n


def collapse_balanced(root):
    """Replace each MAXIMAL balanced reduction with a layout-invariant ITreeReduce node.

    Sound: only balanced reductions collapse (inner_tree; unordered's left-fold stays intact),
    and only when the column image + a uniform leaf computation are fully recovered. Different
    chunk sizes (BLOCK_N) -> different column image -> distinct; fp_fusion -> different leaf/fold
    structure -> distinct. Iterative (no recursion on deep folds)."""
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
            op, leaf_sig = info
            new[id(n)] = ITreeReduce(op, leaf_sig, ht[id(n)], _shfl_dir(n, ht))
        else:
            new[id(n)] = _rebuild(n, [new[id(c)] for c in n.children])
    return new[id(root)]


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
