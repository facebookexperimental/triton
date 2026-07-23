"""Forward interpreter — Phase 0 driver (single-block within-thread + within-warp reduction).

Walk ``linearize(func)`` in program order over a symbolic thread, maintaining ``regs: name -> Node``
(the value-DAG node each register currently holds). Each instruction is a transfer function that
looks its operands up in ``regs`` (forward) instead of resolving them backward. The produced value-
DAG reuses :mod:`bitequiv.ptx.treeir`, so the same ``collapse_balanced`` + ``tree_hash`` yield a
descriptor directly comparable to the backward ``entry_signatures`` (the Phase-0 cross-check).

Scope of Phase 0: ``ld.global`` (scalar + vector slots), the fp combines (``add/sub/mul/div/min/max``,
``fma``), ``mov`` pass-through, the within-warp butterfly ``add(p, shfl.bfly(p, off))`` -> ``ShflCombine``,
and ``st.global`` roots. Cross-warp shared memory, MMA, loops, and branches are later phases; anything
unmodeled becomes an ``Opaque`` (sound floor). Integer/address ops do not touch the value state (their
values live in the affine domain, resolved on demand for leaf coordinates).
"""

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.ptx.affine import AffineEval, canon, reqntid_of
from bitequiv.ptx.builder import collapse_balanced, tree_hash
from bitequiv.ptx.leaves import leaf_coord, leaf_columns
from bitequiv.ptx.linker import DefUse, _def_regs, linearize
from bitequiv.ptx.treeir import FpOp, Leaf, OpaqueLeaf, OpaqueOp, ShflCombine

_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_FP_KINDS = frozenset({"add", "sub", "mul", "div", "min", "max"})


def _is_fp(inst):
    return bool(inst.modifiers) and inst.modifiers[-1] in _FP_WIDTHS


def _offset(operand):
    txt = getattr(operand, "text", None) or getattr(operand, "name", None) or str(operand)
    txt = txt.strip()
    return int(txt) if (txt.lstrip("-")).isdigit() else txt


class _Shuffle:
    """Transient marker in the register state: ``child``'s value as seen from lane^offset (a
    butterfly partner). NOT a treeir node — it lives only in ``regs`` until the following combine
    consumes it, letting the forward combine transfer recognize ``add(p, shfl(p, off))`` ->
    ``ShflCombine(off, p)`` (the same detection the backward ``_plan_fp_binary`` does)."""

    __slots__ = ("child", "offset")

    def __init__(self, child, offset):
        self.child = child
        self.offset = offset


class ForwardInterp:
    """Forward value-DAG reconstruction over one ``.entry``."""

    def __init__(self, func):
        self.flat = linearize(func)
        self.du = DefUse(func)  # reused for leaf addresses + not-produced-in-body operand fallback
        ntid = reqntid_of(func)
        self.ev = AffineEval(self.du, ntid)
        self.colev = AffineEval(self.du, ntid, absorb_opaque=True)
        self.regs = {}  # reg name -> value-DAG Node (or a transient _Shuffle marker)
        # Sound floor: set False when the walk hits a cross-thread structure this phase does not
        # model faithfully (a cross-warp shared load). The reconstructed tree is then untrustworthy
        # for the verbatim hash (it would over-merge across num_warps), so forward_descriptor falls
        # back to the conservative per-config fingerprint. Phase 2 models these -> faithful stays True.
        self.faithful = True

    # -- operand lookup -------------------------------------------------------

    def _val(self, operand, at):
        """Current forward value of an operand (may be a ``_Shuffle`` marker), or an ``OpaqueLeaf``
        for a non-register / not-yet-produced value (a kernel input used before its producer)."""
        if isinstance(operand, RegisterOperand):
            v = self.regs.get(operand.name)
            if v is not None:
                return v
            return OpaqueLeaf(canon(self.ev.of_reg(operand.name, at)))
        return OpaqueLeaf(canon(self.ev.of_operand(operand, at)))

    def _node(self, operand, at):
        """Like :meth:`_val` but collapses a ``_Shuffle`` marker to its child (a non-combine
        consumer of a shuffled value just sees the underlying partial)."""
        v = self._val(operand, at)
        return v.child if isinstance(v, _Shuffle) else v

    # -- driver ---------------------------------------------------------------

    def run(self):
        """Forward-simulate the entry; return the value-DAG root of each ``st.global`` value."""
        roots = []
        for at, inst in enumerate(self.flat):
            op, mods = inst.opcode, inst.modifiers
            if op == "st" and ".global" in mods and len(inst.operands) >= 2:
                val = inst.operands[1]
                elts = val.elements if isinstance(val, VectorOperand) else [val]
                roots.extend(self._node(e, at) for e in elts if isinstance(e, RegisterOperand))
                continue
            defs = _def_regs(inst)
            if not defs:
                continue
            if op == "ld" and ".global" in mods:  # per-slot leaves (scalar: slot None -> 0)
                for name, slot in defs:
                    s = slot or 0
                    self.regs[name] = Leaf(leaf_coord(self.ev, self.du, inst, s),
                                           leaf_columns(self.colev, self.du, inst, s))
                continue
            node = self._transfer(inst, at)
            if node is not None:
                for name, _slot in defs:
                    self.regs[name] = node
        return roots

    def _transfer(self, inst, at):
        """Value produced by a non-load, non-store instruction, or ``None`` when it writes no
        value-domain register (an integer/address op)."""
        op, mods = inst.opcode, inst.modifiers
        if (op == "ld" and ".shared" in mods) or op == "ldmatrix":
            self.faithful = False  # cross-warp shared-memory exchange not modeled yet (Phase 2)
        if op == "shfl" and ".bfly" in mods and len(inst.operands) >= 3:
            return _Shuffle(self._node(inst.operands[1], at), _offset(inst.operands[2]))
        if op == "mov" and len(inst.operands) == 2 and isinstance(inst.operands[1], RegisterOperand):
            return self._val(inst.operands[1], at)  # pass-through (preserves a _Shuffle marker)
        if op == "fma" and _is_fp(inst) and len(inst.operands) == 4:
            return FpOp("fma", mods, tuple(self._node(o, at) for o in inst.operands[1:]), fused=True)
        if op in _FP_KINDS and _is_fp(inst) and len(inst.operands) == 3:
            return self._combine(inst, at)
        # Any other unmodeled op (packed `.f32x2`, `mov.b64` pack/unpack, an f64 half-assembly, a
        # cvt, ...): keep it as an OpaqueOp NODE with its register operands as children + non-register
        # operands in the token, EXACTLY like the backward else-branch, so the value-DAG matches. A
        # pure integer/address op is stored too but never pulled into a value tree (value ops read
        # value operands; addresses go through the affine domain), so this is harmless and lazy at
        # the descriptor level (the descriptor only traverses the roots' value-DAG).
        reg_ops = [o for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
        others = [o for o in inst.operands[1:] if not isinstance(o, RegisterOperand)]
        tok = op + "".join(mods)
        if others:
            tok += "{" + ",".join(canon(self.ev.of_operand(o, at)) for o in others) + "}"
        return OpaqueOp(tok, tuple(self._node(o, at) for o in reg_ops))

    def _combine(self, inst, at):
        a, b = inst.operands[1], inst.operands[2]
        if inst.opcode == "add":  # within-warp butterfly: add(p, shfl.bfly(p, off))
            va, vb = self._val(a, at), self._val(b, at)
            for x, y in ((va, vb), (vb, va)):
                if isinstance(y, _Shuffle) and y.child is x:
                    return ShflCombine(y.offset, "add", inst.modifiers, x)
        return FpOp(inst.opcode, inst.modifiers, tuple(self._node(o, at) for o in (a, b)))

    def fingerprint(self):
        """Conservative per-config key used when the reconstruction is not faithful. Folds the
        layout-BEARING, per-config-varying facts so two configs the fuzzer separates get distinct
        keys (never an over-merge): ``reqntid`` (num_warps), the ORDERED ``shfl.bfly`` offset
        sequence (inner_tree's count-up vs unordered's count-down, and the cross-warp steps that
        scale with num_warps), the shared-store width multiset, and the fp-combine count. It
        over-splits (loses num_warps recovery on the unmodeled idiom), which later phases lift."""
        ntid = ",".join(f"{k}{v}" for k, v in sorted(self.ev.reqntid.items())) or "?"
        shfl, stores, fp = [], {}, 0
        for inst in self.flat:
            is_fp = inst.modifiers and any(t in inst.modifiers[-1] for t in (".f16", ".f32", ".f64", ".bf16"))
            if inst.opcode == "shfl" and ".bfly" in inst.modifiers and len(inst.operands) >= 3:
                shfl.append(str(_offset(inst.operands[2])))
            elif inst.opcode == "st" and ".shared" in inst.modifiers:
                w = "".join(m for m in inst.modifiers if m != ".shared") or ".b32"
                stores[w] = stores.get(w, 0) + 1
            elif inst.opcode in _FP_KINDS and is_fp:
                fp += 1
        stores_s = ",".join(f"{w}x{n}" for w, n in sorted(stores.items()))
        return f"fwd-incomplete|ntid={ntid}|shfl={','.join(shfl)}|st={stores_s}|fp={fp}"


def forward_descriptor(func):
    """Per-entry forward descriptor: the collapsed + Merkle-hashed output trees (reusing the backward
    ``collapse_balanced`` + ``tree_hash`` so it is directly comparable to
    :func:`bitequiv.ptx.builder.entry_signatures`) when the reconstruction is FAITHFUL, else a single
    conservative per-config fingerprint (the sound floor)."""
    interp = ForwardInterp(func)
    roots = interp.run()
    if not interp.faithful:
        return (interp.fingerprint(), )
    collapsed = [collapse_balanced(t) for t in roots]
    return tuple(sorted(tree_hash(t) for t in collapsed))


def forward_module_descriptor(ptx):
    """Module-level forward descriptor, mirroring :func:`bitequiv.ptx_reduction.ptx_reduction_descriptor`
    so the eval framework can drive the forward checker via ``--checker
    bitequiv.ptx.forward.interp:forward_module_descriptor``. One canonical signature per ``.entry``
    (sorted). An entry with no reconstructed reduction keeps its launch geometry (``ntid``) as the
    empty-signature guard, so different geometries never collapse to an empty match."""
    from pyptx.ir.nodes import Function
    from pyptx.parser import parse
    from pyptx.parser.parser import ParseError

    from bitequiv.ptx_reduction import _ensure_header, _Unparseable
    if not ptx:
        return ()
    try:
        module = parse(_ensure_header(ptx))
    except ParseError:
        return _Unparseable()
    out = []
    for f in module.directives:
        if not (isinstance(f, Function) and f.is_entry):
            continue
        sig = forward_descriptor(f)
        if sig:
            out.append("|".join(sig))
        else:
            ntid = reqntid_of(f)
            out.append("ntid=" + (",".join(f"{k}{v}" for k, v in sorted(ntid.items())) if ntid else "?"))
    return tuple(sorted(out))
