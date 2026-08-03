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
values are the roots (a packed store expands to one root per f32 lane). A loop-carried
``acc = acc + chunk`` (a chunked / persistent reduction) is summarized as a ``LoopReduce`` fold over
one iteration's chunk (via :mod:`bitequiv.ptx.forward.loops`), recovering num_warps for looped
reductions. MMA-accumulation loops (GEMM K-fold) and data-dependent branches are later phases;
anything unmodeled becomes an ``Opaque`` and, if it would strip a transient marker (losing a
reduction ordering / packed structure), trips ``faithful=False`` — the sound floor that falls back to
the conservative fingerprint. Integer/address ops do not touch the value state (their values live in
the affine domain, resolved on demand for leaf coordinates).
"""

import hashlib

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.ptx.affine import AffineEval, canon, reqntid_of
from bitequiv.ptx.builder import _postorder, collapse_balanced, output_coordfree_key, tree_hash
from bitequiv.ptx.forward.cfg import has_unknown_control
from bitequiv.ptx.forward.loops import (find_loops, loop_accumulates, loop_carried_accumulators,
                                        loop_self_increments)
from bitequiv.ptx.forward.predicate import PredicateDecoder
from bitequiv.ptx.leaves import leaf_coord, leaf_columns
from bitequiv.ptx.linker import DefUse, _def_regs, linearize
from bitequiv.ptx.mma import _is_mma, _mma_fence
from bitequiv.ptx.treeir import (FpOp, Leaf, LoopReduce, Mma, OpaqueLeaf, OpaqueOp, ShflCombine,
                                 SmemExchange)

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
        # Phase 5 control-flow floor: the straight-line walk DROPS predicates / branches, which is
        # sound only when all control flow is thread-structural. An unresolved branch (brx / computed
        # target) or a predicate that depends on runtime DATA (a load / MMA / atomic) makes the trace
        # untrustworthy -> fail closed. A purely tid/param/lane-derived guard stays dropped (safe).
        self._unknown_ctrl = has_unknown_control(func)
        self.pred = PredicateDecoder(self.ev, self.du)
        self._ddb = None  # cached _has_data_dependent_branch() result
        # Phase 5 (loops): a loop-carried `acc = acc + chunk` is summarized as a LoopReduce fold
        # instead of the one-iteration DAG the straight-line walk would leave. Map the accumulating
        # instruction -> (acc register, LoopReduce key = the loop's chunk step). Only ACCUMULATING
        # loops (a running fp total), so an output-tiling loop is untouched. `linearize` and
        # `find_loops` walk `func.body` identically, so the accumulating `inst` objects are shared.
        self._acc = {}
        insts, loops = find_loops(func)
        for lp in set(loops):
            if not loop_accumulates(insts, lp, loops):
                continue
            key = (loop_self_increments(insts, lp, loops), )
            for acc_name, inst in loop_carried_accumulators(insts, lp, loops):
                self._acc[id(inst)] = (acc_name, key)

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

    def _has_data_dependent_branch(self):
        """True if control flow makes the straight-line trace unsound: an unresolved branch (brx /
        computed target), or any predicated instruction whose guard depends on runtime DATA (a load /
        MMA / atomic). A purely thread-structural guard (tid / param / lane arithmetic) is safe to
        drop and does not trip this. Cached — used by both :meth:`run` and :meth:`fingerprint`."""
        if self._ddb is None:
            self._ddb = self._unknown_ctrl
            if not self._ddb:
                for i, inst in enumerate(self.flat):
                    p = getattr(inst, "predicate", None)
                    if p is not None and not self.pred.decode(p.register, i).is_structural:
                        self._ddb = True
                        break
        return self._ddb

    def run(self):
        """Forward-simulate the entry; return the value-DAG root of each ``st.global`` value."""
        if self._has_data_dependent_branch():
            self.faithful = False  # Phase 5 floor: data-dependent / unresolved control flow
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
                else:
                    # A VECTOR st.shared writes a packed multi-element slot this phase does not model.
                    # Skipping it silently is UNSOUND: a later scalar ld.shared would then resolve to a
                    # STALE earlier store while faithful stays True (the tree is wrong but TRUSTED, so
                    # two bit-different configs can collapse to one descriptor -> OVER-MERGE). Fail
                    # closed instead. (The DCR / reviewer soundness concern on D112727538. The remaining
                    # address-agnostic multi-buffer case among all-scalar stores -- resolving a scalar
                    # load to a DIFFERENT scalar buffer -- needs address-matched resolution and is a
                    # follow-up, bundled with the multi-output cross-warp recovery.)
                    self.faithful = False
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
        if _is_mma(inst) or (op == "tcgen05" and ".ld" in mods):
            # The tensor-core output is a hardware-opaque accumulator boundary leaf: an Mma node,
            # NOT a reconstructable per-element tree (its internal K-fold is not FP-order-recoverable).
            # On Hopper the wgmma matmul writes its accumulator regs directly; on Blackwell the
            # tcgen05.mma writes tensor memory (no reg def) and tcgen05.ld reads it back into regs, so
            # BOTH are the MMA-output boundary. Modeling it lets an epilogue reduction OVER the MMA
            # output (gemm_reduce_sum / softmax / layernorm) reconstruct with Mma leaves, which
            # `_epilogue_reduces_mma` detects so `_mma_entry_descriptor` rides the reduction fingerprint
            # (else those within-thread-fold reductions are invisible and over-merge across the order).
            # Over-identifying an MMA output only ever ADDS a splitting term (sound, monotone).
            return Mma("mma-out")
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
            acc = self._acc.get(id(inst))
            if acc is not None and op == "add":
                return self._loop_reduce(inst, acc, at)  # loop-carried add-accumulation -> fold summary
            return self._combine(inst, at)
        # Any other unmodeled op (an f64 half-assembly `or.b64`, a `cvt`, ...): keep it as an
        # OpaqueOp NODE with its register operands as children + non-register operands in the token,
        # EXACTLY like the backward else-branch, so the value-DAG matches. A pure integer/address op
        # is stored too but never pulled into a value tree (value ops read value operands; addresses
        # go through the affine domain), so this is harmless and lazy at the descriptor level. If an
        # operand still carries a transient marker HERE it is genuinely unmodeled (the packed
        # reductions were reconstructed above) -> fail closed (see `_consumes_marker`).
        if any(self._consumes_marker(o, at) for o in inst.operands[1:]):
            self.faithful = False
        reg_ops = [o for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
        others = [o for o in inst.operands[1:] if not isinstance(o, RegisterOperand)]
        tok = op + "".join(mods)
        if others:
            tok += "{" + ",".join(canon(self.ev.of_operand(o, at)) for o in others) + "}"
        return OpaqueOp(tok, tuple(self._node(o, at) for o in reg_ops))

    def _consumes_marker(self, operand, at):
        """True if ``operand`` (a register, or a vector of registers) currently holds a transient
        ``_Shuffle`` / ``_Packed`` marker. An unmodeled op that consumes one would have ``_scalar``
        silently STRIP it -> the reduction ORDERING (count-up vs count-down) or the packed lane
        structure is lost -> OVER-MERGE. The caller fails closed when this is True; the fingerprint's
        ordered shfl sequence + store widths then distinguish the configs. A benign opaque with NO
        marked operand (an f64 ``or.b64``, a ``cvt``) keeps ``faithful`` True and still recovers."""
        if isinstance(operand, RegisterOperand):
            return isinstance(self._val(operand, at), (_Shuffle, _Packed))
        if isinstance(operand, VectorOperand):
            return any(isinstance(e, RegisterOperand)
                       and isinstance(self._val(e, at), (_Shuffle, _Packed)) for e in operand.elements)
        return False

    def _loop_reduce(self, inst, acc, at):
        """A loop-carried ``acc = acc + chunk`` -> ``LoopReduce(add, chunk, key)``, summarizing the
        whole fold instead of leaving the one-iteration ``add(seed, chunk)`` the straight-line walk
        produces. The pre-loop SEED is dropped: when autotuning a single kernel every config shares
        it, so it can never separate two configs (the module docstring's identical-outside-the-
        computation assumption). This RECOVERS num_warps for a chunked / persistent reduction — the
        chunk's within-block reduction collapses to a num_warps-invariant ITreeReduce and the key
        (chunk step) is num_warps-invariant, so configs differing only in num_warps get one LoopReduce.
        The key is chunk-BEARING (the step multiset), so a different BLOCK_N still splits (sound)."""
        acc_name, key = acc
        chunk = next((o for o in inst.operands[1:]
                      if not (isinstance(o, RegisterOperand) and o.name == acc_name)), None)
        if chunk is None:  # `acc = acc + acc` (not a chunk fold) -> fall back to a plain combine
            return self._combine(inst, at)
        return LoopReduce(inst.opcode + "".join(inst.modifiers), self._node(chunk, at), key)

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
        # Control-flow term (Phase 5): the predicated-instruction count + a data-dependent flag, so two
        # fail-closed configs that differ in control flow still split (extra key -> monotone sound).
        cond = sum(1 for inst in self.flat if getattr(inst, "predicate", None) is not None)
        dd = 1 if self._has_data_dependent_branch() else 0
        return (f"fwd-incomplete|ntid={ntid}|shfl={','.join(shfl)}|st={stores_s}"
                f"|fp={fp}|fma={fma}|cf={cond},dd{dd}")


def forward_descriptor(func):
    """Per-entry forward descriptor: the collapsed + Merkle-hashed output trees (reusing the backward
    ``collapse_balanced`` + ``tree_hash`` so it is directly comparable to
    :func:`bitequiv.ptx.builder.entry_signatures`) when the reconstruction is FAITHFUL, else a single
    conservative per-config fingerprint (the sound floor)."""
    interp = ForwardInterp(func)
    roots = interp.run()
    if not interp.faithful:
        return (interp.fingerprint(), )
    if len(roots) == 1:
        return (tree_hash(collapse_balanced(roots[0])), )
    # MULTI-output: num_warps redistributes the entry's rows across threads, so the forward per-thread
    # trees (and their COUNT) are num_warps-bearing -> a verbatim descriptor over-splits. When every
    # output is a CLEAN per-element function of its own element over num_warps-invariant reductions
    # (softmax / layernorm / rmsnorm), coord-blank each output + dedup: the SET of distinct output
    # computations is num_warps-invariant even though the per-thread count is not. An unordered per-row
    # reduction keeps its ShflCombine offsets in the key -> stays split (correct). If ANY output is not
    # cleanly reconstructed (opaque leaf / unrecovered coord), fall back to the verbatim trees (sound,
    # over-split). Replaces the old blanket multi-output G3 guard; gated on the fuzzer (0 over-merge).
    collapsed = [collapse_balanced(r) for r in roots]
    keys = [output_coordfree_key(t) for t in collapsed]
    if all(k is not None for k in keys):
        return tuple(sorted({hashlib.sha1(k.encode()).hexdigest()[:16] for k in keys}))
    return tuple(sorted(tree_hash(t) for t in collapsed))


def _fence_str(fence):
    """Canonical string for an :func:`bitequiv.ptx.mma._mma_fence` tuple (frozensets sorted so the
    string is deterministic across runs). The f32 epilogue rides as PRESENCE ``(has_fma, has_addmul)``
    — fma kept apart from add/mul so enable_fp_fusion on/off never collide, but NOT counted (the count
    is M/N-tile-scaled and would over-split equivalent re-tilings)."""
    if fence[0] == "mma":
        _, tokens, flags, epi = fence
        return (f"mma|tok={'/'.join(sorted(tokens))}|fl={','.join(sorted(flags))}"
                f"|epi=fma{epi[0]},addmul{epi[1]}")
    _, counts, flags, fa = fence  # mma-fp8 fallback: raw per-token counts + (fma, add/mul) (more split)
    counts_s = "/".join(f"{t}x{n}" for t, n in counts)
    return f"mma-fp8|tok={counts_s}|fl={','.join(sorted(flags))}|fma_addmul={fa[0]},{fa[1]}"


def _epilogue_reduces_mma(sinks):
    """True iff any sink value-DAG REDUCES over the MMA output — a reduce node (a 2-ary add/min/max
    ``FpOp`` whose BOTH children carry an ``Mma`` leaf, or a ``ShflCombine`` / ``SmemExchange`` over an
    Mma-bearing subtree). This is the ``tl.sum`` / ``tl.max`` epilogue over ``C = A @ B``
    (gemm_reduce_sum / softmax / layernorm), whether it lowers to a within-thread fold (no shfl), a
    cross-lane butterfly, or a cross-warp shared exchange. It EXCLUDES an elementwise epilogue
    (``relu(acc*alpha + bias)``: the bias add has one non-Mma child, relu's max pairs with a constant)
    and the split-K accumulation (a 1-ary ``LoopReduce`` over an Mma chunk), so the pure GEMMs keep
    their clean tiling-invariant fence. When it fires, :func:`_mma_entry_descriptor` rides the reduction
    fingerprint so configs differing only in the reduction ORDER split (they otherwise share one fence
    and the within-thread fold is invisible -> over-merge across unordered / inner_tree).

    ``sinks`` is EVERY reachable value-DAG root, not just the ``st.global`` stores: a softmax /
    layernorm output is the full tile written through ``ldmatrix`` (a transposed read the walk fails
    closed on, so the ``st.global`` root is an opaque relocation), which buries the ``sum(e)`` reduction
    mid-computation. Scanning the recorded shared-store values and the live registers too surfaces it.
    The ``Mma``-bearing walk is memoized across all sinks (shared subtrees walked once)."""
    has_mma = {}
    for root in sinks:
        for n in _postorder(root):
            if id(n) in has_mma:
                continue
            has_mma[id(n)] = isinstance(n, Mma) or any(has_mma[id(c)] for c in n.children)
            if isinstance(n, (ShflCombine, SmemExchange)) and has_mma[id(n)]:
                return True
            if (isinstance(n, FpOp) and not n.fused and n.kind in ("add", "min", "max")
                    and len(n.children) == 2 and all(has_mma[id(c)] for c in n.children)):
                return True
    return False


def _mma_entry_descriptor(func, fence):
    """Descriptor for an entry containing MMA (tensor-core) ops, via the tiling-invariant fence.

    A PURE GEMM (MMA, no reduction over the output) is the fence ALONE -> num_warps + BLOCK_M/BLOCK_N
    are free (recovered), with ``loops=`` kept (BLOCK_K is a real, bit-relevant K split). An MMA + a
    reduction OVER the MMA output (FA softmax, or gemm_reduce_sum / softmax / layernorm) is FAIL-CLOSED:
    the fence AND the reduction fingerprint both ride, so configs differing in EITHER the MMA shape OR
    the reduction order split (sound, never over-merged). The reduction is detected structurally
    (:func:`_epilogue_reduces_mma`) rather than by ``shfl.bfly`` presence, so a within-thread epilogue
    fold (no shuffle) — which the shfl-only trigger missed, over-merging unordered vs inner_tree — is
    now caught. The reduction fingerprint carries num_warps back in that case; the pure GEMM drops it
    deliberately (num_warps is a free re-tiling of the same dot products)."""
    from bitequiv.ptx.forward.loops import reduction_trip_signature
    from bitequiv.ptx_reduction import _loop_steps
    parts = [_fence_str(fence)]
    interp = ForwardInterp(func)
    roots = interp.run()
    # Ride the reduction fingerprint when the entry reduces over the MMA output. Detected two ways,
    # OR'd so the trigger is a strict SUPERSET of the old ``shfl.bfly`` one (can only ADD splitting):
    # (1) any ``shfl.bfly`` — a cross-lane / cross-warp reduction (the reconstructed tree may fail
    #     closed on an unmatched shared load, hiding the Mma leaves, so the structural detector alone
    #     under-fires here); (2) ``_epilogue_reduces_mma`` — a within-thread fold (no shuffle) over the
    #     MMA output, which (1) misses and which otherwise over-merges across the reduction order. The
    #     detector scans every reachable value-DAG (roots + recorded shared stores + live registers),
    #     since a softmax / layernorm reduction is buried behind the ldmatrix output relocation.
    sinks = list(roots)
    sinks += [v for _, v in interp.smem_stores if hasattr(v, "children")]
    sinks += [v for v in interp.regs.values() if hasattr(v, "children")]
    has_shfl = any(inst.opcode == "shfl" and ".bfly" in inst.modifiers for inst in linearize(func))
    if has_shfl or _epilogue_reduces_mma(sinks):
        parts.append("mma+red|" + interp.fingerprint())
    steps = _loop_steps(func)  # BLOCK_K split (+ the tcgen05 / fp8 K carried here, not in the token)
    if steps:
        parts.append("loops=" + ",".join(map(str, steps)))
    sig = reduction_trip_signature(func)
    if sig is not None:
        # A GEMM whose K sum is regrouped by an OUTER reduction loop (split-K: partials combined) is
        # NOT bitwise-equivalent across split counts, yet its static MMA/op structure is identical, so
        # the tiling-invariant fence over-merges them (num_splits 1==2, 4==8). Fail closed on the
        # loop-control fingerprint (the setp trip constants), which differs per split count -> sound.
        # A plain single-K-loop GEMM has no such nested reduction -> sig is None -> fence unchanged.
        # Precise per-split recovery (nested LoopReduce) is the follow-up.
        parts.append("splits=" + hashlib.sha1(repr(sig).encode()).hexdigest()[:8])
    return "|".join(parts)


def forward_module_descriptor(ptx):
    """Module-level forward descriptor, mirroring :func:`bitequiv.ptx_reduction.ptx_reduction_descriptor`
    so the eval framework can drive the forward checker via ``--checker
    bitequiv.ptx.forward.interp:forward_module_descriptor``. One canonical signature per ``.entry``
    (sorted). An entry with no reconstructed reduction keeps its launch geometry (``ntid``) as the
    empty-signature guard, so different geometries never collapse to an empty match."""
    from pyptx.ir.nodes import Function
    from pyptx.parser import parse
    from pyptx.parser.parser import ParseError

    from bitequiv.ptx_reduction import _ensure_header, _loop_steps, _Unparseable
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
        fence = _mma_fence(f)  # Phase 4: an MMA entry takes the tiling-invariant fence composition
        if fence is not None:
            out.append(_mma_entry_descriptor(f, fence))
            continue
        parts = list(forward_descriptor(f))
        if not parts:  # no reconstructed reduction -> keep launch geometry as the empty-sig guard
            ntid = reqntid_of(f)
            parts.append("ntid=" + (",".join(f"{k}{v}" for k, v in sorted(ntid.items())) if ntid else "?"))
        steps = _loop_steps(f)  # BLOCK_N cross-chunk fence: the straight-line walk cannot follow the
        if steps:               # loop back-edge, so a looped/persistent reduction (sum_dim1_persistent)
            parts.append("loops=" + ",".join(map(str, steps)))  # would over-merge without this (sound)
        out.append("|".join(parts))
    return tuple(sorted(out))
