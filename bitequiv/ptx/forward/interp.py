"""Forward interpreter — the value-DAG reconstruction driver (Phases 0-3).

Walk ``linearize(func)`` in program order over a symbolic thread, maintaining ``regs: name -> Node``
(the value-DAG node each register currently holds) plus transient markers (``_Shuffle`` butterfly
partners, ``_Packed`` f32x2 lane pairs). Each instruction is a transfer function that looks its
operands up in ``regs`` (forward) instead of resolving them backward. The produced value-DAG reuses
:mod:`bitequiv.ptx.treeir`, so the same ``collapse_balanced`` + ``tree_hash`` yield a descriptor
directly comparable to the backward ``entry_signatures`` (the cross-check oracle).

Modeled so far: ``ld.global`` (scalar + vector slots) -> leaves; the fp combines
(``add/sub/mul/div/min/max``, ``fma``) and the within-warp butterfly ``op(p, shfl.bfly(p, off))`` ->
``ShflCombine`` for any reduce op; ``mov`` pass-through; the cross-warp shared exchange
(``st.shared`` -> ``ld.shared`` -> ``SmemExchange``, the forward SmemModel); and packed ``.f32x2``
reductions (``mov.b64`` pack/unpack + ``.f32x2`` combine/fma decomposed per lane). ``st.global``
values are the roots (a packed store expands to one root per f32 lane). MMA, data-dependent branches,
and loop back-edges are later phases; anything unmodeled becomes an ``Opaque`` and, if it would strip
a transient marker (losing a reduction ordering / packed structure), trips ``faithful=False`` — the
sound floor that falls back to the conservative fingerprint. Integer/address ops do not touch the
value state (their values live in the affine domain, resolved on demand for leaf coordinates).
"""

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.ptx.affine import AffineEval, canon, reqntid_of
from bitequiv.ptx.builder import collapse_balanced, tree_hash
from bitequiv.ptx.leaves import leaf_coord, leaf_columns
from bitequiv.ptx.linker import DefUse, _def_regs, linearize
from bitequiv.ptx.treeir import FpOp, Leaf, OpaqueLeaf, OpaqueOp, ShflCombine, SmemExchange

_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_FP_KINDS = frozenset({"add", "sub", "mul", "div", "min", "max"})
# Packed 2-wide f32 (two f32 lanes in one 64-bit register, assembled/split by `mov.b64`). On
# sm_90+/sm_100 the compiler folds an f32 reduction into `.f32x2` ops, burying the per-lane shfl
# butterflies inside `mov.b64`/`add.f32x2` — idiom 2 decomposes these back to per-lane scalar trees.
_PACKED_WIDTH = ".f32x2"


def _is_fp(inst):
    return bool(inst.modifiers) and inst.modifiers[-1] in _FP_WIDTHS


def _is_packed(inst):
    return bool(inst.modifiers) and inst.modifiers[-1] == _PACKED_WIDTH


def _offset(operand):
    txt = getattr(operand, "text", None) or getattr(operand, "name", None) or str(operand)
    txt = txt.strip()
    return int(txt) if (txt.lstrip("-")).isdigit() else txt


class _Shuffle:
    """Transient marker in the register state: ``child``'s value as seen from lane^offset (a
    butterfly partner). NOT a treeir node — it lives only in ``regs`` until the following combine
    consumes it, letting the forward combine transfer recognize ``add(p, shfl(p, off))`` ->
    ``ShflCombine(off, p)`` (the same detection the backward ``_plan_fp_binary`` does).

    A ``_Shuffle`` may wrap a ``_Packed`` (``shfl.bfly`` of a 64-bit packed register shuffles BOTH
    f32 lanes together); ``_packed_lane`` distributes it over the lanes as ``_Shuffle(lane, off)``."""

    __slots__ = ("child", "offset")

    def __init__(self, child, offset):
        self.child = child
        self.offset = offset


class _Packed:
    """Transient marker: a 64-bit register holding TWO f32 lanes (``lanes = (lo_node, hi_node)``),
    as assembled by ``mov.b64 rd, {lo, hi}``. Like ``_Shuffle`` it is NOT a treeir node — it lives
    in ``regs`` only until a ``.f32x2`` combine reduces it per lane, a ``mov.b64 {lo,hi}, rd``
    scatters it back to two scalar registers, or an ``st.global`` expands it into two roots. A
    ``_Packed`` reaching a SCALAR treeir position (``_scalar``) is unmodeled -> fail closed."""

    __slots__ = ("lanes", )

    def __init__(self, lanes):
        self.lanes = tuple(lanes)


class ForwardInterp:
    """Forward value-DAG reconstruction over one ``.entry``."""

    def __init__(self, func):
        self.flat = linearize(func)
        self.du = DefUse(func)  # reused for leaf addresses + not-produced-in-body operand fallback
        ntid = reqntid_of(func)
        self.ev = AffineEval(self.du, ntid)
        self.colev = AffineEval(self.du, ntid, absorb_opaque=True)
        self.regs = {}  # reg name -> value-DAG Node (or a transient _Shuffle marker)
        # Forward shared-memory model: (stream_index, stored_value_node) for each scalar shared store,
        # in program order. A shared load resolves to the most-recent prior store's value subtree
        # (the single-reduction canonical cross-warp exchange), wrapped in a SmemExchange. Because we
        # walk FORWARD, the store's value is already in the register state when the load reads it —
        # the cross-warp fan-in is captured by construction, not inverted/guessed as in the backward
        # resolver. (Vector shared stores + address-matched multi-buffer resolution are Phase 2b.)
        self.smem_stores = []
        # Sound floor: set False when the walk hits a cross-thread structure this phase does not
        # model faithfully (a cross-warp shared load). The reconstructed tree is then untrustworthy
        # for the verbatim hash (it would over-merge across num_warps), so forward_descriptor falls
        # back to the conservative per-config fingerprint. Phase 2 models these -> faithful stays True.
        self.faithful = True

    # -- operand lookup -------------------------------------------------------

    def _val(self, operand, at):
        """Current forward value of an operand (may be a ``_Shuffle`` / ``_Packed`` marker), or an
        ``OpaqueLeaf`` for a non-register / not-yet-produced value (a kernel input used before its
        producer)."""
        if isinstance(operand, RegisterOperand):
            v = self.regs.get(operand.name)
            if v is not None:
                return v
            return OpaqueLeaf(canon(self.ev.of_reg(operand.name, at)))
        return OpaqueLeaf(canon(self.ev.of_operand(operand, at)))

    def _scalar(self, v):
        """Coerce a register value to a SCALAR treeir node: a ``_Shuffle`` collapses to its child (a
        non-combine consumer just sees the underlying partial); a ``_Packed`` reaching a scalar
        position is an idiom we do not model there -> fail closed with an opaque placeholder."""
        if isinstance(v, _Shuffle):
            return v.child
        if isinstance(v, _Packed):
            self.faithful = False
            return OpaqueLeaf("packed-scalar")
        return v

    def _deref(self, v):
        """Collapse a transient ``_Shuffle`` to its child but LEAVE a ``_Packed`` intact — used
        where a marker is carried further (the shfl source, a packed lane, an ``st.global`` root)."""
        return v.child if isinstance(v, _Shuffle) else v

    def _node(self, operand, at):
        """Scalar treeir node for ``operand`` (see :meth:`_scalar`)."""
        return self._scalar(self._val(operand, at))

    # -- driver ---------------------------------------------------------------

    def run(self):
        """Forward-simulate the entry; return the value-DAG root of each ``st.global`` value."""
        roots = []
        for at, inst in enumerate(self.flat):
            op, mods = inst.opcode, inst.modifiers
            if op == "st" and ".global" in mods and len(inst.operands) >= 2:
                val = inst.operands[1]
                elts = val.elements if isinstance(val, VectorOperand) else [val]
                for e in elts:
                    if isinstance(e, RegisterOperand):
                        roots.extend(self._root_nodes(e, at))  # a _Packed store -> one root per lane
                continue
            if op == "st" and ".shared" in mods and len(inst.operands) >= 2:
                val = inst.operands[1]  # scalar shared store: record its value for later loads
                if isinstance(val, RegisterOperand):
                    self.smem_stores.append((at, self._node(val, at)))
                continue
            if op == "mov" and ".b64" in mods and len(inst.operands) == 2 and self._b64_mov(inst, at):
                continue  # mov.b64 pack ({lo,hi}->rd) / unpack (rd->{lo,hi}) handled in place
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
        if op == "ld" and ".shared" in mods:
            src = self._match_smem(at)
            if src is not None:
                return SmemExchange(src)  # cross-warp exchange: read the stored partial's subtree
            self.faithful = False  # unmatched shared load -> fail closed (opaque fallback below)
        elif op == "ldmatrix":
            self.faithful = False  # ldmatrix hardware-transpose relocation -> Phase 2b
        if op == "shfl" and ".bfly" in mods and len(inst.operands) >= 3:
            # preserve a _Packed source (a b64 shfl shuffles BOTH lanes); _deref only unnests _Shuffle
            return _Shuffle(self._deref(self._val(inst.operands[1], at)), _offset(inst.operands[2]))
        if op == "mov" and len(inst.operands) == 2 and isinstance(inst.operands[1], RegisterOperand):
            return self._val(inst.operands[1], at)  # pass-through (preserves a _Shuffle / _Packed)
        if _is_packed(inst) and op in _FP_KINDS and len(inst.operands) == 3:
            return self._packed_combine(op, mods, inst.operands[1], inst.operands[2], at)
        if _is_packed(inst) and op == "fma" and len(inst.operands) == 4:
            return self._packed_fma(mods, inst.operands[1:], at)
        if op == "fma" and _is_fp(inst) and len(inst.operands) == 4:
            return FpOp("fma", mods, tuple(self._node(o, at) for o in inst.operands[1:]), fused=True)
        if op in _FP_KINDS and _is_fp(inst) and len(inst.operands) == 3:
            return self._combine(inst, at)
        # Any other unmodeled op (an f64 half-assembly `or.b64`, a `cvt`, ...): keep it as an
        # OpaqueOp NODE with its register operands as children + non-register operands in the token,
        # EXACTLY like the backward else-branch, so the value-DAG matches. A pure integer/address op
        # is stored too but never pulled into a value tree (value ops read value operands; addresses
        # go through the affine domain), so this is harmless and lazy at the descriptor level.
        # An unmodeled op that CONSUMES a transient marker (a _Shuffle butterfly partner, or a
        # _Packed f32x2 pair — directly or via a vector operand) would have `_scalar` silently STRIP
        # it -> the reduction ORDERING (count-up vs count-down) or the packed lane structure is lost
        # -> OVER-MERGE. Idiom 2 RECONSTRUCTS the packed reductions above (`.f32x2` combine, `mov.b64`
        # pack/unpack); a marker still reaching HERE is genuinely unmodeled -> fail closed, and the
        # fingerprint's ordered shfl sequence + store widths then distinguish the configs. A benign
        # opaque with NO marked operand (an f64 `or.b64`, a `cvt`) keeps faithful=True and recovers.
        def _marked(o):
            if isinstance(o, RegisterOperand):
                return isinstance(self._val(o, at), (_Shuffle, _Packed))
            if isinstance(o, VectorOperand):
                return any(isinstance(e, RegisterOperand)
                           and isinstance(self._val(e, at), (_Shuffle, _Packed)) for e in o.elements)
            return False

        if any(_marked(o) for o in inst.operands[1:]):
            self.faithful = False
        reg_ops = [o for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
        others = [o for o in inst.operands[1:] if not isinstance(o, RegisterOperand)]
        tok = op + "".join(mods)
        if others:
            tok += "{" + ",".join(canon(self.ev.of_operand(o, at)) for o in others) + "}"
        return OpaqueOp(tok, tuple(self._node(o, at) for o in reg_ops))

    def _combine(self, inst, at):
        """Scalar fp binary combine — see :meth:`_combine_nodes`."""
        return self._combine_nodes(inst.opcode, inst.modifiers,
                                   self._val(inst.operands[1], at), self._val(inst.operands[2], at))

    def _combine_nodes(self, op, mods, va, vb):
        """Combine two already-looked-up register values (each a node or a transient marker).
        A within-warp butterfly `op(p, shfl.bfly(p, off))` -> ShflCombine for ANY reduce op (add,
        min, max, ...): the offset + op ride verbatim into the node, sound even before the collapse
        treats min/max as reduce ops (it over-splits, never over-merges). A `_Shuffle` NOT consumed
        as a butterfly (a cross-lane idiom we don't model — e.g. a max butterfly whose partner isn't
        the sibling) would be silently stripped -> the cross-lane structure lost -> OVER-MERGE; fail
        closed instead. Reused PER LANE by the packed `.f32x2` combine."""
        for x, y in ((va, vb), (vb, va)):
            if isinstance(y, _Shuffle) and y.child is x:
                return ShflCombine(y.offset, op, mods, self._deref(x))
        if isinstance(va, _Shuffle) or isinstance(vb, _Shuffle):
            self.faithful = False
        return FpOp(op, mods, (self._scalar(va), self._scalar(vb)))

    def _packed_lane(self, v, i):
        """Lane `i` (0=lo, 1=hi) of a packed value: a `_Packed` -> its lane node; a `_Shuffle` of a
        `_Packed` (a b64 butterfly shuffling both lanes) -> `_Shuffle(lane, off)` so the per-lane
        combine still sees the butterfly partner. Anything else -> None (the caller fails closed)."""
        if isinstance(v, _Packed) and i < len(v.lanes):
            return v.lanes[i]
        if isinstance(v, _Shuffle) and isinstance(v.child, _Packed) and i < len(v.child.lanes):
            return _Shuffle(v.child.lanes[i], v.offset)
        return None

    def _packed_combine(self, op, mods, a, b, at):
        """A `.f32x2` binary combine -> a `_Packed` of the two per-lane scalar combines (each reusing
        the butterfly detection). If either operand is not packed the idiom is unmodeled -> fail
        closed with an opaque, so the descriptor never merges across a lost packed structure."""
        va, vb = self._val(a, at), self._val(b, at)
        lanes = [(self._packed_lane(va, i), self._packed_lane(vb, i)) for i in (0, 1)]
        if any(x is None for pair in lanes for x in pair):
            self.faithful = False
            return OpaqueOp(op + "".join(mods), (self._node(a, at), self._node(b, at)))
        return _Packed(self._combine_nodes(op, mods, x, y) for x, y in lanes)

    def _packed_fma(self, mods, operands, at):
        """A `fma.f32x2` -> a `_Packed` of the two per-lane fused fmas. Fail closed if an operand is
        not packed, or if a lane carries an unconsumed `_Shuffle` (an fma over a cross-lane partial
        we do not model)."""
        vals = [self._val(o, at) for o in operands]
        out = []
        for i in (0, 1):
            lane = [self._packed_lane(v, i) for v in vals]
            if any(x is None for x in lane):
                self.faithful = False
                return OpaqueOp("fma" + "".join(mods), tuple(self._node(o, at) for o in operands))
            if any(isinstance(x, _Shuffle) for x in lane):
                self.faithful = False
            out.append(FpOp("fma", mods, tuple(self._scalar(x) for x in lane), fused=True))
        return _Packed(out)

    def _b64_mov(self, inst, at):
        """Handle `mov.b64` PACK (`rd, {lo, hi}` -> a `_Packed`) / UNPACK (`{lo, hi}, rd` -> scatter
        the `_Packed` lanes to two scalar regs). Returns True if consumed; False for a scalar-to-
        scalar b64 copy, which the generic `mov` transfer preserves (markers and all)."""
        dst, src = inst.operands[0], inst.operands[1]
        if isinstance(dst, RegisterOperand) and isinstance(src, VectorOperand) and len(src.elements) == 2:
            self.regs[dst.name] = _Packed(self._node(e, at) for e in src.elements)
            return True
        if isinstance(dst, VectorOperand) and isinstance(src, RegisterOperand):
            v = self._val(src, at)
            lanes = v.lanes if isinstance(v, _Packed) else None
            for i, e in enumerate(dst.elements):
                if not isinstance(e, RegisterOperand):
                    continue
                if lanes is not None and i < len(lanes):
                    self.regs[e.name] = self._deref(lanes[i])
                else:  # a b64 never packed here (raw 64-bit load / unmodeled source) -> fail closed
                    self.faithful = False
                    self.regs[e.name] = OpaqueLeaf(canon(self.ev.of_reg(e.name, at)))
            return True
        return False

    def _root_nodes(self, operand, at):
        """Root node(s) for an `st.global` value operand: a `_Packed` store yields one scalar root
        per f32 lane (a packed two-output store), else the single scalar node."""
        v = self._val(operand, at)
        if isinstance(v, _Packed):
            return [self._deref(l) for l in v.lanes]
        return [self._scalar(v)]

    def _match_smem(self, at):
        """Value subtree of the most-recent shared store before ``at`` (the single-reduction
        canonical cross-warp exchange), or ``None`` if there is none. Because the walk is forward,
        the stored value is already reconstructed — Phase 2b adds address-matched multi-buffer
        resolution + vector/ldmatrix relocation; this covers the common one-buffer exchange."""
        node = None
        for i, n in self.smem_stores:
            if i < at:
                node = n
            else:
                break
        return node

    def fingerprint(self):
        """Conservative per-config key used when the reconstruction is not faithful. Folds the
        layout-BEARING, per-config-varying facts so two configs the fuzzer separates get distinct
        keys (never an over-merge): ``reqntid`` (num_warps), the ORDERED ``shfl.bfly`` offset
        sequence (inner_tree's count-up vs unordered's count-down, and the cross-warp steps that
        scale with num_warps), the shared-store width multiset, and the fp-combine count. It
        over-splits (loses num_warps recovery on the unmodeled idiom), which later phases lift."""
        ntid = ",".join(f"{k}{v}" for k, v in sorted(self.ev.reqntid.items())) or "?"
        shfl, stores, fp, fma = [], {}, 0, 0
        for inst in self.flat:
            is_fp = inst.modifiers and any(t in inst.modifiers[-1] for t in (".f16", ".f32", ".f64", ".bf16"))
            if inst.opcode == "shfl" and ".bfly" in inst.modifiers and len(inst.operands) >= 3:
                shfl.append(str(_offset(inst.operands[2])))
            elif inst.opcode == "st" and ".shared" in inst.modifiers:
                w = "".join(m for m in inst.modifiers if m != ".shared") or ".b32"
                stores[w] = stores.get(w, 0) + 1
            elif inst.opcode == "fma" and is_fp:
                fma += 1  # counted SEPARATELY from mul+add: fp_fusion on (fma) vs off (mul+add) can
                #           have the same TOTAL op count (a coincidental collision), so keying fma
                #           apart is what keeps fp_fusion split in the fail-closed floor.
            elif inst.opcode in _FP_KINDS and is_fp:
                fp += 1
        stores_s = ",".join(f"{w}x{n}" for w, n in sorted(stores.items()))
        return f"fwd-incomplete|ntid={ntid}|shfl={','.join(shfl)}|st={stores_s}|fp={fp}|fma={fma}"


def forward_descriptor(func):
    """Per-entry forward descriptor: the collapsed + Merkle-hashed output trees (reusing the backward
    ``collapse_balanced`` + ``tree_hash`` so it is directly comparable to
    :func:`bitequiv.ptx.builder.entry_signatures`) when the reconstruction is FAITHFUL, else a single
    conservative per-config fingerprint (the sound floor)."""
    interp = ForwardInterp(func)
    roots = interp.run()
    if not interp.faithful:
        return (interp.fingerprint(), )
    # G3 guard (mirrors the backward `entry_signatures`): the layout-drop collapse is num_warps-
    # INVARIANT only when ALL of the entry's threads reduce into ONE output. With MULTIPLE outputs
    # (a 2-D / multi-axis tile reduced per row), num_warps RE-PARTITIONS the threads among the outputs
    # and re-associates each output's reduction, so collapsing there OVER-MERGES — measured: sum_4d /
    # softmax / rmsnorm inner_tree vs unordered collapsed to the same descriptor at the heavy grid.
    # So collapse only a single-output entry; keep a multi-output entry's trees VERBATIM (sound,
    # over-split — num_warps recovery for multi-output is later, layout-aware work).
    collapsed = [collapse_balanced(roots[0])] if len(roots) == 1 else roots
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

    from bitequiv.ptx_reduction import _ensure_header, _loop_steps, _mma_guard, _Unparseable
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
        parts = list(forward_descriptor(f))
        if not parts:  # no reconstructed reduction -> keep launch geometry as the empty-sig guard
            ntid = reqntid_of(f)
            parts.append("ntid=" + (",".join(f"{k}{v}" for k, v in sorted(ntid.items())) if ntid else "?"))
        steps = _loop_steps(f)  # BLOCK_N cross-chunk fence: the straight-line walk cannot follow the
        if steps:               # loop back-edge, so a looped/persistent reduction (sum_dim1_persistent)
            parts.append("loops=" + ",".join(map(str, steps)))  # would over-merge without this (sound)
        guard = _mma_guard(f)  # conservative MMA fence until Phase 4 models the collective op forward
        if guard:
            parts.append(guard)
        out.append("|".join(parts))
    return tuple(sorted(out))
