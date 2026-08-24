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

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.core.affine_algebra import canon
from bitequiv.core.canonicalize import _REDUCE_FP, collapse_balanced, tree_hashes
from bitequiv.core.treeir import FpOp, Leaf, OpaqueLeaf, OpaqueOp, ShflCombine, SmemExchange
from bitequiv.ptx.affine import AffineEval, reqntid_of
from bitequiv.ptx.leaves import leaf_coord, leaf_columns
from bitequiv.ptx.linker import Def, DefUse

_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_FP_KINDS = frozenset({"add", "sub", "mul", "div", "min", "max"})


def _is_fp(inst):
    return bool(inst.modifiers) and any(m in _FP_WIDTHS for m in inst.modifiers)


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


def build_trees(func):
    """List of reconstructed reduction-tree roots for one ``.entry`` (one per result store)."""
    b = _Builder(func)
    return [b.build(reg, idx) for reg, idx in b.roots()]


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
    return tuple(sorted(tree_hashes(roots)))
