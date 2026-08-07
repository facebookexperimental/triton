"""Forward interpreter for AMDGCN — the value-DAG reconstruction driver, AMD peer of
``bitequiv.ptx.forward.interp``.

Walk ``linearize(func)`` in program order over a symbolic thread, maintaining ``regs: name ->
Node`` (the value-DAG node each register currently holds) plus a transient ``_Shuffle`` marker for a
DPP butterfly partner. Each instruction is a transfer function that looks its operands up in
``regs`` and produces a value-DAG node reusing the ISA-neutral treeir + collapse
(:mod:`bitequiv.amdgcn.core`). Two AMDGCN modules reduce in the same bitwise order iff their
collapsed tree hashes are equal.

AMDGCN butterfly shape: a within-wave reduction is a separate ``v_mov_b32_dpp`` (the lane
permutation) feeding a plain ``v_add_f32`` (the combine) — the AMD echo of PTX's ``shfl.bfly`` + add.
So a DPP move becomes a ``_Shuffle`` marker and the following fp combine, if it combines ``x`` with
``shuffle(x)``, becomes a ``ShflCombine``. The DPP control is decoded to an integer step
(``row_shr:1/2/4/8`` -> 1/2/4/8, ``row_bcast:15/31`` -> 16/32) so the balanced-collapse count-up
check recovers ``num_warps``.

Soundness floor: the interpreter sets ``faithful = False`` on any cross-thread / control structure it
cannot model exactly (a loop back-edge, a cross-lane ``ds_read``/``v_readlane``, an unconsumed
shuffle). When not faithful the descriptor is a conservative ``fingerprint`` that can only equal an
identical config — the checker over-splits, never over-merges. GEMM (any MFMA) is decided by the sound
matrix-core fence (:mod:`bitequiv.amdgcn.mma`).

Still fail-closed (recovery-loop targets): cross-warp LDS exchange (``ds_write``/``ds_read`` address
model + ``v_readlane``), loop-carried ``LoopReduce``, packed ``v_pk_*`` per-lane decomposition, and
multi-output coord-free recovery.
"""
from __future__ import annotations

import hashlib

from bitequiv.amdgcn.affine import AffineEval, canon, reqntid_of, thread_image
from bitequiv.amdgcn.core.affine_algebra import Affine, _parse_int
from bitequiv.amdgcn.core.canonicalize import (
    _coordfree_sig,
    _forest_postorder,
    collapse_balanced,
    output_coordfree_keys,
    tree_hash,
    tree_hashes,
)
from bitequiv.amdgcn.core.treeir import FpOp, Leaf, LoopReduce, Mma, OpaqueLeaf, OpaqueOp, ShflCombine, SmemExchange
from bitequiv.amdgcn.forward.cfg import has_backedge, has_unknown_control
from bitequiv.amdgcn.forward.loops import find_loops, loop_increments, loop_step_sizes, loop_steps
from bitequiv.amdgcn.leaves import leaf_coord, leaf_columns
from bitequiv.amdgcn.linker import DefUse, _def_regs, linearize
from bitequiv.amdgcn.mma import is_mma, mma_fence, mma_token_counts
from bitequiv.amdgcn.parser import ImmediateOperand, RegisterOperand, VectorOperand, parse

_WIDTHS = {"f32": ".f32", "f64": ".f64", "bf16": ".bf16", "f16": ".f16"}

# Scalar / DPP fp combine base mnemonics -> reduce-op kind.
_FP_KIND = {
    "v_add_f32": "add", "v_add_f16": "add", "v_add_f64": "add", "v_add_bf16": "add",
    "v_sub_f32": "sub", "v_sub_f16": "sub", "v_subrev_f32": "sub", "v_subrev_f16": "sub",
    "v_mul_f32": "mul", "v_mul_f16": "mul", "v_mul_f64": "mul",
    "v_max_f32": "max", "v_max_f16": "max", "v_max_f64": "max", "v_maxnum_f32": "max", "v_maxnum_f16": "max",
    "v_min_f32": "min", "v_min_f16": "min", "v_min_f64": "min", "v_minnum_f32": "min", "v_minnum_f16": "min",
}
_FMA = frozenset({
    "v_fma_f32", "v_fmac_f32", "v_fma_f16", "v_fmac_f16", "v_fma_f64", "v_fmac_f64",
    "v_fma_legacy_f32", "v_fmac_legacy_f32",
})

# Packed (SIMD-2) fp combines: v_pk_add_f32 v[0:1], v[2:3], v[4:5] is TWO independent same-slot
# scalar combines (slot 0 = lo halves, slot 1 = hi halves) -> decomposed per lane.
_PK_KIND = {
    "v_pk_add_f32": "add", "v_pk_mul_f32": "mul", "v_pk_fma_f32": "fma", "v_pk_sub_f32": "sub",
    "v_pk_add_f16": "add", "v_pk_mul_f16": "mul", "v_pk_fma_f16": "fma", "v_pk_max_f32": "max",
    "v_pk_min_f32": "min",
}


_MAXMIN3 = frozenset({"v_max3_f32", "v_max3_f16", "v_min3_f32", "v_min3_f16"})
_FMAC = frozenset({"v_fmac_f32", "v_fmac_f16", "v_fmac_f64", "v_fmac_legacy_f32"})


def _has_clamp(inst):
    """clamp saturates the RESULT (not exact), so a packed op with clamp can't be split into plain
    combines -> opaque. (op_sel reorders halves [_pk_half_map]; neg is an EXACT sign flip
    [_pk_neg_map wraps the half in a `neg` node] — neither forces opaque.)"""
    return any(m.startswith("clamp") for m in inst.modifiers)


def _pk_neg_map(inst, nsrc):
    """Which source half is NEGATED per result lane, from neg_lo/neg_hi. Returns ``[lane0_neg,
    lane1_neg]`` (per-source 0/1). A negated half is ``a + (-b) == a - b`` bit-exactly."""
    lo = [0] * nsrc
    hi = [0] * nsrc
    for m in inst.modifiers:
        if m.startswith("neg_lo:") and "[" in m:
            vals = _parse_bracket(m)
            for i in range(min(nsrc, len(vals))):
                lo[i] = vals[i]
        elif m.startswith("neg_hi:") and "[" in m:
            vals = _parse_bracket(m)
            for i in range(min(nsrc, len(vals))):
                hi[i] = vals[i]
    return [lo, hi]


def _parse_bracket(mod):
    """The int list of a ``key:[a,b,..]`` modifier (e.g. ``op_sel:[1,0]`` -> [1, 0])."""
    inside = mod[mod.index("[") + 1:mod.rindex("]")]
    return [int(x) for x in inside.split(",") if x.strip().lstrip("-").isdigit()]


def _pk_half_map(inst, nsrc):
    """Which source half feeds each result lane, from the op_sel modifiers. Returns
    ``[lane0_halves, lane1_halves]`` where each is a per-source 0(lo)/1(hi) list. Default (no op_sel):
    lane0 takes every lo half, lane1 every hi half — the plain same-slot packed op."""
    lo = [0] * nsrc
    hi = [1] * nsrc
    for m in inst.modifiers:
        if m.startswith("op_sel:") and "[" in m:
            vals = _parse_bracket(m)
            for i in range(min(nsrc, len(vals))):
                lo[i] = vals[i]
        elif m.startswith("op_sel_hi:") and "[" in m:
            vals = _parse_bracket(m)
            for i in range(min(nsrc, len(vals))):
                hi[i] = vals[i]
    return [lo, hi]

# Cross-lane ops still unmodeled -> fail closed. (ds_read/v_readlane -> exchanges;
# ds_bpermute/ds_permute/ds_swizzle -> butterfly shuffles, gated by the self-butterfly check in
# _combine_nodes so a non-self permute stays a plain FpOp.)
_CROSS_LANE = ("v_writelane", )
_LOAD_PREFIXES = ("buffer_load", "global_load", "flat_load", "scratch_load")
_STORE_PREFIXES = ("buffer_store", "global_store", "flat_store", "scratch_store")

# quad_perm patterns that are butterfly XOR steps (adjacent-lane pairings).
_QUAD_STEP = {"[1,0,3,2]": 1, "[2,3,0,1]": 2, "[0,1,2,3]": 0, "[3,2,1,0]": 3}  # [3,2,1,0] = quad
#   reverse (lane i <-> i^3); a within-quad butterfly step. The exact value only feeds the count-up
#   DIRECTION check (a reduction keys on extent, not the offset), so 3 keeps a count-up run monotone.


def _strip(op):
    for suf in ("_e64", "_e32"):
        if op.endswith(suf):
            return op[:-len(suf)]
    return op


def _dpp_base(op):
    for suf in ("_e64_dpp", "_e32_dpp", "_dpp"):
        if op.endswith(suf):
            return _strip(op[:-len(suf)])
    return _strip(op)


def _fp_width(base):
    for k, v in _WIDTHS.items():
        if base.endswith("_" + k) or ("_" + k + "_") in base:
            return v
    return ".f32"


def _is_fp_arith(op):
    """True for a floating-point arithmetic op (scalar, packed ``v_pk_*``, or DPP): carries an fp
    width AND is an add/sub/mul/fma/max/min/div/dot. Integer address math (``v_add_lshl_u32``) is
    excluded. Used by the fingerprint so any fp-op-choice difference (fusion's ``v_pk_fma`` vs
    ``v_pk_mul`` + ``v_pk_add``) is captured."""
    return (any(w in op for w in ("_f32", "_f16", "_f64", "_bf16"))
            and any(k in op for k in ("add", "sub", "mul", "fma", "fmac", "max", "min", "div", "dot")))


def _fp_op_kind(base):
    """Reduce-op kind of an fp-arithmetic mnemonic (order matters: check compound names first)."""
    if "fma" in base or "fmac" in base:
        return "fma"
    if "subrev" in base or "sub" in base:
        return "sub"
    if "add" in base:
        return "add"
    if "mul" in base:
        return "mul"
    if "max" in base:
        return "max"
    if "min" in base:
        return "min"
    if "div" in base:
        return "div"
    return "other"


def _steps_count_up(steps):
    """True iff the integer DPP butterfly steps go COUNT-UP (inner_tree): the smallest step first
    appears before the largest. A count-down (unordered) butterfly has the largest first. Requires at
    least two distinct steps. This is the num_warps-invariance signal (D113540598: a count-up balanced
    butterfly is bitwise reproducible across num_warps)."""
    ints = [s for s in steps if isinstance(s, int) and s > 0]
    if len(set(ints)) < 2:
        return False
    lo, hi = min(ints), max(ints)
    return ints.index(lo) < ints.index(hi)


def _dpp_control(inst):
    """The DPP lane-permutation control (``row_shr:8`` / ``row_bcast:31`` / ``quad_perm:[..]``) —
    the AMD echo of a PTX ``shfl.bfly`` offset. Drops the mask / carry noise."""
    keep = [m for m in inst.modifiers if not m.startswith(("row_mask", "bank_mask", "bound_ctrl", "fi:"))]
    return " ".join(keep) if keep else "dpp"


def _dpp_offset(control):
    """Decode a DPP control to an INTEGER butterfly step (so the collapse count-up check works), or
    the control string verbatim if it is not a recognizable doubling step. ``row_shr/shl/ror:N`` ->
    N; ``row_bcast:15/31`` -> 16/32 (the 16-/32-lane cross-boundary); ``quad_perm`` adjacent swaps ->
    1/2; ``row_half_mirror`` -> 8, ``row_mirror`` -> 16."""
    c = control.strip()
    for pfx in ("row_shr:", "row_shl:", "row_ror:"):
        if c.startswith(pfx):
            try:
                return int(c[len(pfx):])
            except ValueError:
                return c
    if c.startswith("row_bcast:"):
        try:
            return int(c[len("row_bcast:"):]) + 1
        except ValueError:
            return c
    if c.startswith("quad_perm:"):
        return _QUAD_STEP.get(c[len("quad_perm:"):], c)
    if c == "row_half_mirror":
        return 8
    if c == "row_mirror":
        return 16
    return c


def _is_load(op):
    return op.startswith(_LOAD_PREFIXES)


def _is_store(op):
    return op.startswith(_STORE_PREFIXES)


def _is_mov_dpp(op):
    return op.endswith("_dpp") and _dpp_base(op).startswith(("v_mov", "s_mov"))


def _is_cross_lane(op):
    if op.startswith(_CROSS_LANE):
        return True
    if op.endswith("_dpp"):  # a DPP op that is neither a mov-shuffle nor a recognized fp combine
        base = _dpp_base(op)
        return base not in _FP_KIND and base not in _FMA and not base.startswith(("v_mov", "s_mov"))
    return False


def _store_data(inst):
    """The stored VALUE operand: ``buffer_store`` puts data in operand[0]; ``global_store`` /
    ``flat_store`` put the address in operand[0] and the data in operand[1]."""
    if inst.opcode.startswith("buffer_store"):
        return inst.operands[0] if inst.operands else None
    return inst.operands[1] if len(inst.operands) > 1 else None


def _ntid_str(func):
    return ",".join(f"{k}{v}" for k, v in sorted(reqntid_of(func).items())) or "?"


class _Shuffle:
    """A butterfly partner living only in ``regs``: ``child``'s value permuted by a DPP ``offset``.
    Consumed by the next fp combine into a ``ShflCombine``; if it reaches any other position the
    cross-lane structure would be lost, so the interpreter fails closed."""

    __slots__ = ("child", "offset")

    def __init__(self, child, offset):
        self.child = child
        self.offset = offset


class ForwardInterp:

    def __init__(self, func):
        self.func = func
        self.flat = linearize(func)
        self.du = DefUse(func)
        self.reqntid = reqntid_of(func)
        self.ev = AffineEval(self.du, self.reqntid)
        self.colev = AffineEval(self.du, self.reqntid, absorb_opaque=True)
        self.regs = {}
        # Cross-warp LDS exchange model: stores since the last barrier, and the last closed phase a
        # load reads from. Each entry is (slot_image | None, value_node).
        self.smem_phase = []
        self.smem_closed = []
        self._fails = []
        self._out_reach = self._compute_out_reach()
        self.faithful = not has_unknown_control(func)
        if not self.faithful:
            self._fails.append("unknown_control")
        # Loop-carried reductions: a loop with a recognized fp accumulator is summarized as a
        # LoopReduce (id(inst) -> chunk-step key); a loop with NO recognized accumulator can't be
        # walked straight-line faithfully -> fail closed.
        self._loop_acc = {}
        for header, latch in find_loops(func):
            key = (loop_increments(func, header, latch), )
            accs = self._loop_accumulators(header, latch)
            if accs:
                for iid in accs:
                    self._loop_acc[iid] = key
            else:
                self._fail("unmodeled-loop")

    def _compute_out_reach(self):
        """Registers whose value transitively feeds a global-store data operand (an OVER-approximate
        backward def-use closure — a single reverse pass; over-approximation only ever adds fail-close
        conservatism, so it stays sound). Used so an unmodeled cross-lane op that only computes an
        ADDRESS or a dead value does not spuriously fail the reduction closed."""
        reach = set()
        for inst in self.flat:
            if _is_store(inst.opcode):
                data = _store_data(inst)
                elems = data.elements if isinstance(data, VectorOperand) else ([data] if data else [])
                reach.update(e.name for e in elems if isinstance(e, RegisterOperand))
        for inst in reversed(self.flat):
            if any(r in reach for r, _ in _def_regs(inst)):
                for o in inst.operands[1:]:
                    if isinstance(o, RegisterOperand):
                        reach.add(o.name)
                    elif isinstance(o, VectorOperand):
                        reach.update(e.name for e in o.elements)
        return reach

    def _feeds_output(self, inst):
        return any(r in self._out_reach for r, _ in _def_regs(inst))

    def _fail(self, why):
        self.faithful = False
        self._fails.append(why)

    # -- operand lookup -------------------------------------------------------

    def _val(self, operand, at):
        if isinstance(operand, RegisterOperand):
            v = self.regs.get(operand.name)
            return v if v is not None else OpaqueLeaf(canon(self.ev.of_reg(operand.name, at)))
        if isinstance(operand, (ImmediateOperand, VectorOperand)):
            return OpaqueLeaf(canon(self.ev.of_operand(operand, at)))
        return OpaqueLeaf("operand?")

    def _deref(self, v):
        return v.child if isinstance(v, _Shuffle) else v

    def _scalar(self, v):
        """Coerce a value to a scalar tree node. A ``_Shuffle`` reaching a scalar position is an
        unconsumed butterfly partner whose cross-lane structure would be lost -> fail closed. (A
        broadcast-deref that skips the failure is sound on the tested kernels but needs the packed +
        multi-phase-LDS reconstruction to actually pay off, so it stays conservative for now.)"""
        if isinstance(v, _Shuffle):
            self._fail("unconsumed-shuffle")
            return v.child
        return v

    def _bind(self, inst, node):
        for reg, _slot in _def_regs(inst):
            self.regs[reg] = node

    # -- main walk ------------------------------------------------------------

    def run(self):
        roots = []
        for at, inst in enumerate(self.flat):
            op = inst.opcode
            if _is_store(op):
                roots.extend(self._root_nodes(inst, at))
                continue
            if op.startswith("s_barrier"):  # a barrier closes the current LDS exchange phase
                self.smem_closed, self.smem_phase = self.smem_phase, []
                continue
            if op.startswith("ds_write"):
                self._record_store(inst, at)
                continue
            if op.startswith("ds_read"):
                for reg, node in self._resolve_smem(inst, at):
                    self.regs[reg] = node
                continue
            if op.startswith(("ds_bpermute", "ds_permute")):  # cross-lane permute = a butterfly shuffle
                data = inst.operands[2] if len(inst.operands) > 2 else None
                child = self._deref(self._val(data, at)) if data is not None else OpaqueLeaf("bperm?")
                self._bind(inst, _Shuffle(child, self._lane_addr_offset(inst, at)))
                continue
            if op.startswith("ds_swizzle"):  # intra-wave swizzle = a butterfly shuffle
                data = inst.operands[1] if len(inst.operands) > 1 else None
                child = self._deref(self._val(data, at)) if data is not None else OpaqueLeaf("swz?")
                self._bind(inst, _Shuffle(child, _dpp_offset(self._swizzle_ctrl(inst))))
                continue
            if op.startswith(("v_readlane", "v_readfirstlane")):  # cross-lane read = fan-in-1 relocation
                srcop = inst.operands[1] if len(inst.operands) > 1 else None
                src = self._scalar(self._val(srcop, at)) if srcop is not None else OpaqueLeaf("readlane?")
                if isinstance(src, OpaqueLeaf):  # relocating an opaque value: keep it opaque
                    # An index/address value (the source resolves to a provable Affine, e.g.
                    # readfirstlane(%tid.x) = the warp base) is not a lost reduction -> never fail.
                    is_addr = (isinstance(srcop, RegisterOperand)
                               and isinstance(self.ev.of_reg(srcop.name, at), Affine))
                    if self._feeds_output(inst) and not is_addr:
                        self._fail("readlane-opaque")
                    self._bind(inst, src)
                else:
                    self._bind(inst, SmemExchange(src))
                continue
            if _is_mov_dpp(op):
                self._bind(inst, self._shuffle(inst, at))
                continue
            if _is_cross_lane(op):
                if self._feeds_output(inst):
                    self._fail("crosslane:" + _strip(op))
                self._bind(inst, OpaqueLeaf("crosslane:" + _strip(op)))
                continue
            if _is_load(op):
                self._bind_load(inst, at)
                continue
            if op.startswith("v_pk_mov"):  # packed move (repack halves) — transparent for the value-DAG
                self._bind_pk_mov(inst, at)
                continue
            if op.startswith("v_pk_") and _strip(op) in _PK_KIND and not _has_clamp(inst):
                self._bind_packed(inst, at)
                continue
            node = self._transfer(inst, at)
            if node is not None:
                self._bind(inst, node)
        return roots

    def _shuffle(self, inst, at):
        src = inst.operands[1] if len(inst.operands) > 1 else None
        child = self._deref(self._val(src, at)) if src is not None else OpaqueLeaf("dpp?")
        return _Shuffle(child, _dpp_offset(_dpp_control(inst)))

    def _lane_addr_offset(self, inst, at):
        """Butterfly XOR distance of a ``ds_bpermute``/``ds_permute``, decoded from its target-lane
        address (operand[1]). The common pattern is ``xor(C, (tid&mask)<<2)`` — a byte address, so the
        lane XOR distance is ``C>>2`` (an integer step: powers of two count-up/down like a shfl.bfly).
        Falls back to the canonical affine string when the pattern is not a clean XOR (still a stable
        order-key, but won't drive the count-up collapse)."""
        va = inst.operands[1] if len(inst.operands) > 1 else None
        if isinstance(va, RegisterOperand):
            d = self.du.last_writer(va.name, at)
            if d is not None and _strip(d.inst.opcode) in ("v_xor_b32", "s_xor_b32"):
                for o in d.inst.operands[1:]:
                    if isinstance(o, ImmediateOperand):
                        c = _parse_int(o.text)
                        if c is not None and c % 4 == 0:
                            return c >> 2  # byte XOR mask -> lane XOR distance
        aff = self.ev.of_operand(va, at) if va is not None else None
        return canon(aff) if aff is not None else "bperm"

    @staticmethod
    def _swizzle_ctrl(inst):
        for m in inst.modifiers:
            if m.startswith(("offset:", "swizzle:")):
                return m
        return "swizzle"

    # -- cross-warp LDS exchange (model the exchange from its address, per D114470722) ---------

    @staticmethod
    def _ds_offsets(inst):
        out = {}
        for m in inst.modifiers:
            for key in ("offset0:", "offset1:", "offset:"):
                if m.startswith(key):
                    try:
                        out[key[:-1]] = int(m[len(key):])
                    except ValueError:
                        pass
        return out

    @staticmethod
    def _shift_image(img, byte_off):
        if img is None or not byte_off:
            return img
        uniform, offs = img
        return uniform, frozenset(o + byte_off for o in offs)

    def _record_store(self, inst, at):
        """Record a ``ds_write`` as (slot image, stored value) in the current phase. ``ds_write2``
        writes two slots at ``offset0``/``offset1`` (in units of the data size)."""
        op = inst.opcode
        addr = inst.operands[0] if inst.operands else None
        base = thread_image(self.ev, self.ev.of_operand(addr, at)) if addr is not None else None
        offs = self._ds_offsets(inst)
        if op.startswith("ds_write2"):
            sz = 8 if "b64" in op else 4
            for okey, di in (("offset0", 1), ("offset1", 2)):
                if len(inst.operands) > di:
                    img = self._shift_image(base, offs.get(okey, 0) * sz)
                    self.smem_phase.append((img, self._scalar(self._val(inst.operands[di], at))))
        elif len(inst.operands) > 1:
            # single ds_write: the data may be a vector (b64/b128) -> one dword slot per element.
            data = inst.operands[1]
            byte0 = offs.get("offset", 0)
            elems = list(data.elements) if isinstance(data, VectorOperand) else [data]
            for k, e in enumerate(elems):
                img = self._shift_image(base, byte0 + k * 4)
                self.smem_phase.append((img, self._scalar(self._val(e, at))))

    def _match_store(self, img):
        """The stored value a load reads: address-matched against the last closed phase; falls back
        to an address-agnostic match only when every candidate store holds the same value. ``None``
        (fail closed) on an ambiguous or unwritten slot."""
        cands = self.smem_closed or self.smem_phase
        if not cands:
            return None
        if img is not None and all(si is not None for si, _ in cands):
            hits = [n for si, n in cands if si[0] == img[0] and (si[1] & img[1])]
            if hits:
                return hits[-1] if len({_coordfree_sig(n) for n in hits}) == 1 else None
            return None  # read a slot no store in the phase wrote -> fail closed
        if len({_coordfree_sig(n) for _, n in cands}) == 1:
            return cands[-1][1]
        return None

    def _match_or_fail(self, img, feeds):
        src = self._match_store(img)
        if src is None or isinstance(src, OpaqueLeaf):  # unresolved / opaque value -> no structure
            if feeds:  # only an unresolved read on an output path loses a reduction -> fail closed
                self._fail("ds-unresolved")
            return OpaqueLeaf("ds_read-unresolved")
        return SmemExchange(src)

    def _resolve_smem(self, inst, at):
        """Resolve a ``ds_read`` to ``[(dst_reg, SmemExchange)]``, one per slot read (``ds_read2``
        reads two)."""
        op = inst.opcode
        addr = inst.operands[1] if len(inst.operands) > 1 else None
        base = thread_image(self.ev, self.ev.of_operand(addr, at)) if addr is not None else None
        offs = self._ds_offsets(inst)
        dst = inst.operands[0] if inst.operands else None
        elems = list(dst.elements) if isinstance(dst, VectorOperand) else ([dst] if dst is not None else [])
        feeds = self._feeds_output(inst)
        out = []
        if op.startswith("ds_read2"):
            sz = 8 if "b64" in op else 4
            half = max(1, len(elems) // 2)
            for gi, okey in enumerate(("offset0", "offset1")):
                node = self._match_or_fail(self._shift_image(base, offs.get(okey, 0) * sz), feeds)
                for e in elems[gi * half:(gi + 1) * half]:
                    if isinstance(e, RegisterOperand):
                        out.append((e.name, node))
        else:
            node = self._match_or_fail(self._shift_image(base, offs.get("offset", 0)), feeds)
            for e in elems:
                if isinstance(e, RegisterOperand):
                    out.append((e.name, node))
        return out

    def _bind_load(self, inst, at):
        for reg, slot in _def_regs(inst):
            s = slot or 0
            self.regs[reg] = Leaf(leaf_coord(self.ev, self.du, inst, s), leaf_columns(self.colev, self.du, inst, s))

    def _bind_packed(self, inst, at):
        """Decompose a packed ``v_pk_*`` op into one scalar combine per lane (slot), each reading the
        matching half of every source register pair, and bind each dest sub-register separately."""
        base = _strip(inst.opcode)
        kind = _PK_KIND[base]
        width = ".f16" if base.endswith("f16") else ".f32"
        dst = inst.operands[0]
        delems = list(dst.elements) if isinstance(dst, VectorOperand) else [dst]
        nsrc = 3 if kind == "fma" else 2
        srcs = inst.operands[1:1 + nsrc]
        halfmap = _pk_half_map(inst, nsrc)  # [lane0 source-halves, lane1 source-halves] from op_sel
        negmap = _pk_neg_map(inst, nsrc)  # which source half is negated per lane (exact sign flip)
        for k, d in enumerate(delems):
            if not isinstance(d, RegisterOperand):
                continue
            halves = halfmap[k] if k < len(halfmap) else [k] * nsrc
            negs = negmap[k] if k < len(negmap) else [0] * nsrc
            vals = []
            for i in range(len(srcs)):
                v = self._pk_elem(srcs[i], halves[i], at)
                if negs[i]:  # a + (-b): exact negation -> a pure `neg` node (coord-blankable)
                    v = OpaqueOp("neg", (self._scalar(v), ))
                vals.append(v)
            if kind == "fma":
                self.regs[d.name] = FpOp("fma", (width, ), tuple(self._scalar(v) for v in vals), fused=True)
            else:  # butterfly-aware: a packed reduction is a per-lane shuffle-combine
                self.regs[d.name] = self._combine_nodes(kind, (width, ), vals[0], vals[1])

    def _bind_pk_mov(self, inst, at):
        """A packed move ``v_pk_mov`` relocates data (possibly repacking halves via op_sel) but does
        no arithmetic, so per-element it passes the source value through — transparent for the
        value-DAG (op_sel only reorders which physical half carries which value)."""
        dst = inst.operands[0]
        src = inst.operands[1] if len(inst.operands) > 1 else None
        delems = list(dst.elements) if isinstance(dst, VectorOperand) else [dst]
        selems = list(src.elements) if isinstance(src, VectorOperand) else ([src] if src is not None else [])
        for k, d in enumerate(delems):
            if not isinstance(d, RegisterOperand):
                continue
            s = selems[k] if k < len(selems) else (selems[-1] if selems else None)
            self.regs[d.name] = self._val(s, at) if s is not None else OpaqueLeaf("pkmov?")

    def _pk_elem(self, operand, half, at):
        """The RAW value of a source's ``half`` (0=lo / 1=hi element of a register pair, or a
        broadcast scalar); a ``_Shuffle`` marker is preserved so ``_combine_nodes`` sees a butterfly."""
        if isinstance(operand, VectorOperand):
            e = operand.elements[half] if half < len(operand.elements) else operand.elements[-1]
            return self._val(e, at)
        return self._val(operand, at)

    def _root_nodes(self, inst, at):
        data = _store_data(inst)
        if data is None:
            return []
        elems = data.elements if isinstance(data, VectorOperand) else (data, )
        return [self._scalar(self._val(r, at)) for r in elems if isinstance(r, RegisterOperand)]

    # -- transfer -------------------------------------------------------------

    def _transfer(self, inst, at):
        op = inst.opcode
        ops = inst.operands
        if id(inst) in self._loop_acc:  # loop-carried accumulator -> summarize as a LoopReduce
            return self._loop_reduce(inst, at)
        if is_mma(inst):
            return Mma("mma-out")  # fence intercepts GEMM before here; kept sound just in case
        if _strip(op) in ("v_mov_b32", "s_mov_b32") and len(ops) >= 2:  # _strip: opcode is v_mov_b32_e32
            return self._val(ops[1], at)
        if op.endswith("_dpp"):
            base = _dpp_base(op)
            if base in _FP_KIND or base in _FMA:
                return self._dpp_combine(inst, at, base)
            return None  # unreachable: cross-lane handled it
        base = _strip(op)
        if base in _FMAC and len(ops) >= 3:  # fma-accumulate: dst = src0*src1 + dst_old (dst is implicit acc)
            kids = (self._scalar(self._val(ops[1], at)), self._scalar(self._val(ops[2], at)),
                    self._scalar(self._val(ops[0], at)))
            return FpOp("fma", (_fp_width(base), ), kids, fused=True)
        if base in _MAXMIN3 and len(ops) >= 4:  # 3-way max/min = nested binary (assoc for max/min)
            kind = "max" if "max" in base else "min"
            w = (_fp_width(base), )
            a, b, c = (self._val(ops[i], at) for i in (1, 2, 3))
            return self._combine_nodes(kind, w, self._combine_nodes(kind, w, a, b), c)
        if base in _FMA and len(ops) >= 4:
            kids = tuple(self._scalar(self._val(o, at)) for o in ops[1:4])
            return FpOp("fma", (_fp_width(base), ), kids, fused=True)
        if base in _FP_KIND and len(ops) >= 3:
            return self._combine_nodes(_FP_KIND[base], (_fp_width(base), ),
                                       self._val(ops[1], at), self._val(ops[2], at))
        if base == "v_bfrev_b32" and len(ops) >= 2 and isinstance(ops[1], ImmediateOperand):
            # bit-reverse(0x00000001) = 0x80000000 = -0.0: how codegen materializes the sum
            # accumulator identity. Tag it so the collapse can strip `add(x, -0.0)` = x (see
            # `_is_add_neutral`); other bfrev uses stay opaque-pure.
            if _parse_int(ops[1].text) == 1:
                return OpaqueLeaf("fpconst:-0.0")
        return self._opaque_op(inst, at)

    def _loop_accumulators(self, header, latch):
        """Instruction ids in ``[header, latch]`` of a loop-carried fp accumulator (a combine whose
        destination is also one of its sources — ``acc = op(acc, chunk)`` / ``acc = a*b + acc``)."""
        out = []
        for i in range(header, min(latch + 1, len(self.flat))):
            inst = self.flat[i]
            b = _strip(inst.opcode)
            if (b in _FP_KIND or b in _FMA or b in _FMAC) and len(inst.operands) >= 3:
                dst = inst.operands[0]
                if isinstance(dst, RegisterOperand):
                    srcs = [o.name for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
                    if dst.name in srcs:
                        out.append(id(inst))
        return out

    def _loop_reduce(self, inst, at):
        """A loop-carried accumulation summarized WITHOUT unrolling: the fold op + the ONE iteration's
        contribution (``chunk``, the accumulator input removed) + the chunk-step key (BLOCK_N/BLOCK_K).
        A different chunk size -> different key -> sound split; the pre-loop seed is dropped."""
        base = _strip(inst.opcode)
        w = _fp_width(base)
        key = self._loop_acc[id(inst)]
        if base in _FMAC and len(inst.operands) >= 3:  # acc = src0*src1 + acc -> fold add, chunk = src0*src1
            a, b = (self._scalar(self._val(inst.operands[i], at)) for i in (1, 2))
            return LoopReduce("add" + w, FpOp("mul", (w, ), (a, b)), key)
        dst = inst.operands[0].name
        chunk_op = next((o for o in inst.operands[1:]
                         if not (isinstance(o, RegisterOperand) and o.name == dst)), None)
        chunk = self._scalar(self._val(chunk_op, at)) if chunk_op is not None else OpaqueLeaf("chunk?")
        op = "fma" if base in _FMA else _FP_KIND[base]
        return LoopReduce(op + w, chunk, key)

    def _combine_nodes(self, kind, mods, va, vb):
        if va is vb and kind in ("max", "min"):  # op(x, x) == x identity (a no-op wait/sched barrier);
            return va  # preserve the value (and any _Shuffle marker) instead of consuming it
        return self._combine2(kind, mods, va, vb)

    def _combine2(self, kind, mods, va, vb):
        """A binary fp combine. It is a butterfly ``ShflCombine`` only when it is a SELF combine —
        one operand is ``shuffle(x)`` and the other is that same ``x`` (object identity). This is the
        sound test: a cross-lane op (``ds_bpermute``) that shuffles a DIFFERENT value is not a
        self-butterfly, so it stays a plain ``FpOp`` and never wrongly fuses. A ``_Shuffle`` reaching
        the plain branch is an unconsumed partner -> ``_scalar`` fails it closed."""
        for x, y in ((va, vb), (vb, va)):
            if isinstance(y, _Shuffle) and (y.child is x or y.child is self._deref(x)):
                return ShflCombine(y.offset, kind, mods, self._deref(x))
        # A cross-lane shuffle of a DIFFERENT partial (a != b): the AMD wide reduction pipelines many
        # register-partials, combining `add(a, shuffle(b))` at a fixed offset. Model the shuffled
        # partial as a fan-in-1 cross-lane RELOCATE (SmemExchange) so the combine is faithful; the
        # extent/count-up collapse then recovers num_warps (the reduced element COUNT is invariant).
        # Fuzzer-gated (0 over-merge): dropping the offset is sound for a count-up result, and a
        # non-count-up reduction fails the collapse gate and keeps its per-config shape.
        def coerce(v):
            return SmemExchange(self._scalar(v.child)) if isinstance(v, _Shuffle) else self._scalar(v)

        return FpOp(kind, mods, (coerce(va), coerce(vb)))

    def _dpp_combine(self, inst, at, base):
        """A FUSED DPP combine ``vD = op(dpp(vsrc0), vsrc1)``. For a reduction it is a SELF combine;
        if the two sources are not the same value it is not a plain butterfly -> fail closed."""
        ops = inst.operands
        if len(ops) < 3:
            return self._opaque_op(inst, at)
        n0, n1 = self._val(ops[1], at), self._val(ops[2], at)
        if self._deref(n0) is not self._deref(n1):
            self._fail("dpp-nonself")
            return OpaqueLeaf("dpp-nonself:" + base)
        kind = "fma" if base in _FMA else _FP_KIND[base]
        return ShflCombine(_dpp_offset(_dpp_control(inst)), kind, (_fp_width(base), ), self._scalar(n1))

    def _opaque_op(self, inst, at):
        reg_ops = [o for o in inst.operands[1:] if isinstance(o, RegisterOperand)]
        others = [o for o in inst.operands[1:] if not isinstance(o, RegisterOperand)]
        tok = inst.opcode + "".join(inst.modifiers)
        if others:
            tok += "{" + ",".join(canon(self.ev.of_operand(o, at)) for o in others) + "}"
        return OpaqueOp(tok, tuple(self._scalar(self._val(o, at)) for o in reg_ops))

    # -- conservative floor ---------------------------------------------------

    def fingerprint(self):
        """The sound fail-closed key: launch geometry + the ORDERED DPP controls (inner_tree count-up
        vs unordered count-down) + shared-store widths + a HISTOGRAM of every fp-arithmetic mnemonic
        (scalar / packed / DPP) + loop chunk steps. Only ever over-splits."""
        dpp, stores, fpops = [], {}, {}
        for inst in self.func.body:
            op = inst.opcode
            if op.endswith("_dpp"):
                dpp.append(_dpp_control(inst))
            if op.startswith("ds_write"):
                stores[op] = stores.get(op, 0) + 1
            if _is_fp_arith(op):
                base = _dpp_base(op) if op.endswith("_dpp") else _strip(op)
                fpops[base] = fpops.get(base, 0) + 1
        st = ",".join(f"{k}x{v}" for k, v in sorted(stores.items()))
        fps = ",".join(f"{k}:{v}" for k, v in sorted(fpops.items()))
        return (f"fwd-incomplete|ntid={_ntid_str(self.func)}|dpp={','.join(dpp)}"
                f"|st={st}|fp={{{fps}}}|loops={loop_steps(self.func)}")


_MAX_DAG_NODES = 2_500  # above this the collapse cost (~O(n^2) on big unrolled AMD trees) is not worth
#                         it -> fall to the sound fingerprint. Keeps grading fast; big reductions lose
#                         recovery but stay sound. TODO: make the collapse near-linear to raise this.
_MAX_BODY = 2_000  # a huge unrolled reduction body is expensive to even reconstruct -> fingerprint first


def forward_descriptor(func):
    """Canonical descriptor string for one non-MMA entry."""
    interp = ForwardInterp(func)
    if len(func.body) > _MAX_BODY:  # too large to reconstruct cheaply -> sound fingerprint (no run)
        return interp.fingerprint()
    roots = interp.run()
    if not interp.faithful or not roots:
        return interp.fingerprint()
    if len(_forest_postorder(roots)) > _MAX_DAG_NODES:  # pathologically large DAG -> fail closed
        return interp.fingerprint()
    if len(roots) == 1:  # single output: collapse the balanced reduction (recovers num_warps)
        hashes = [tree_hash(collapse_balanced(roots[0]))]
    else:
        # multi-output: if every output is a clean elementwise fn of num_warps-invariant reductions,
        # its coord-free key is num_warps-invariant, and the SET of distinct keys (num_warps changes
        # the per-thread COUNT, not the set) is a recoverable descriptor. Fall back to the verbatim
        # layout-bearing hashes when any output is not safe to coord-blank. Gated by the fuzzer.
        collapsed = [collapse_balanced(r) for r in roots]
        keys = output_coordfree_keys(collapsed)
        if keys and all(k is not None for k in keys):
            hashes = sorted({hashlib.sha1(k.encode()).hexdigest()[:16] for k in keys})
        else:
            hashes = tree_hashes(roots)
    # A faithful straight-line tree hash captures the full reduction (extent = height, order = shape),
    # so it stands ALONE: NO ntid (would spuriously split the num_warps the collapse just proved
    # equal) and NO loops fence — chunking the SAME total reduction into a BLOCK_N/BLOCK_K loop is
    # bit-identical for inner_tree (fixed order) and, for unordered, the verbatim tree already carries
    # any chunk-driven shape change, so a loop-vs-unrolled pair (``loops=(2,)`` vs ``loops=()`` at the
    # same tree hash) is a spurious num_warps split. EXCEPTION: a real loop BACK-EDGE means the tree
    # walked the body once and does NOT capture the cross-iteration accumulation (persistent /
    # loop-carried), so the loops chunk-size fence stays load-bearing there (dropping it over-merged
    # sum_dim1_persistent). Empty hash set -> ntid + loops so unrelated trivial kernels do not merge.
    body = ",".join(sorted(h for h in hashes if h))
    if not body:
        return f"fwd|ntid={_ntid_str(func)}|loops={loop_step_sizes(func)}"
    if has_backedge(func):
        return f"fwd|{body}|loops={loop_step_sizes(func)}"
    return f"fwd|{body}"


def _mma_entry_descriptor(func, fence):
    """Descriptor for a matrix-core entry. For f16/bf16/f32 the tensor-core fold is tile/block_k/
    num_warps bit-invariant on gfx942, so the form-agnostic fence (dtype family + epilogue fp-kind
    presence) stands alone — dropping the K-extent count, loop chunk steps, and launch geometry that
    only over-split. fp8 keeps them (its accumulate cadence is bit-deciding). Gated by the fuzzer."""
    if "mma_fp8" in fence:
        return f"{fence}|mmacnt={mma_token_counts(func)}|loops={loop_steps(func)}|ntid={_ntid_str(func)}"
    return fence


def forward_module_descriptor(amdgcn_text):
    """The public checker: one canonical, hashable descriptor per AMDGCN module.

    Feed it ``ck.asm['amdgcn']``. Two autotune configs are bitwise-equivalent iff their descriptors
    are ``==``. Empty / no-entry module -> ``()``."""
    funcs = parse(amdgcn_text)
    parts = []
    for f in funcs:
        if not getattr(f, "is_entry", False):
            continue
        fence = mma_fence(f)
        parts.append(_mma_entry_descriptor(f, fence) if fence is not None else forward_descriptor(f))
    return tuple(sorted(parts))
