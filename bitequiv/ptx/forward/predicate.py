"""Predicate classification for the forward interpreter's sound floor.

A dropped predicate is sound iff it is thread-STRUCTURAL (its truth is a function of
tid / ctaid / ntid / kernel params). It is DATA-DEPENDENT — unsound to drop — iff its value derives
from a runtime memory load (``ld.global`` / ``ld.shared``), an MMA output, or an atomic. We classify
by def-use PROVENANCE: trace the predicate register back through its source operands; if any source
is produced by a data op the predicate is data-dependent. Special registers (``%tid`` / ``%laneid`` /
``%ctaid`` ...), kernel params, and pure integer arithmetic over them (even non-affine idioms like
``tid & 31``) have no data op in their provenance, so they are structural.

Conservative by construction (over-split, never over-merge): any predicate whose provenance reaches a
data op is data-dependent; only a purely param / thread-index-derived predicate is structural. Finer
structural kinds (warp-leader, first-N, loop induction variable) that would let the interpreter
RECOVER more freedom are a later extension; the floor needs only the structural-vs-data-dependent
split so the interpreter can fail closed on data-dependent control flow.
"""
from dataclasses import dataclass

from pyptx.ir.nodes import RegisterOperand, VectorOperand

from bitequiv.ptx.mma import _is_mma

DATA_DEPENDENT = "DATA_DEPENDENT"
STRUCTURAL = "STRUCTURAL"


@dataclass(frozen=True)
class PredInfo:
    """The classification of one predicate. ``is_structural`` is what the sound floor consumes."""

    kind: str

    @property
    def is_structural(self):
        return self.kind != DATA_DEPENDENT


def _is_data_source(inst):
    """True if ``inst`` produces a value from runtime DATA (a global/shared memory load, an MMA
    output, or an atomic) — the thing that makes a predicate over it unsound to drop."""
    op, mods = inst.opcode, inst.modifiers
    if op == "ld" and (".global" in mods or ".shared" in mods):
        return True
    if op.startswith("atom") or op == "red":
        return True
    return _is_mma(inst)


class PredicateDecoder:
    """Classify a predicate register as structural or data-dependent by def-use provenance."""

    def __init__(self, ev, du):
        self.ev = ev  # AffineEval — reserved for the finer structural kinds (later recovery work)
        self.du = du

    def decode(self, pred_reg, before_index):
        """``PredInfo`` for the predicate held in ``pred_reg`` at ``before_index`` (a program point in
        the same linearized stream the interpreter walks)."""
        dd = self._depends_on_data(pred_reg, before_index)
        return PredInfo(DATA_DEPENDENT if dd else STRUCTURAL)

    def _depends_on_data(self, reg, before_index):
        """True iff the value in ``reg`` traces back (through def-use) to a data source. Iterative
        with a seen-set so a shared/cyclic def graph terminates."""
        stack, seen = [(reg, before_index)], set()
        while stack:
            r, at = stack.pop()
            if (r, at) in seen:
                continue
            seen.add((r, at))
            d = self.du.last_writer(r, at)
            if d is None:
                continue  # kernel param / special register (%tid, %laneid, ...) -> structural leaf
            if _is_data_source(d.inst):
                return True
            for o in (d.inst.operands[1:] if d.inst.operands else []):  # source operands
                if isinstance(o, RegisterOperand):
                    stack.append((o.name, d.index))
                elif isinstance(o, VectorOperand):
                    stack.extend((e.name, d.index) for e in o.elements
                                 if isinstance(e, RegisterOperand))
        return False
