"""Predication model for AMDGCN — the AMD peer of ``bitequiv.ptx.forward.predicate``.

AMDGCN has NO per-instruction predicate token (unlike PTX ``@p``): masking is done through the
hardware ``EXEC`` register (a per-lane execute mask) and through ``v_cndmask_b32`` (a per-lane
select). The forward checker deliberately does NOT build an EXEC-mask control-flow model — instead:

  * divergent control (EXEC manipulation around a branch) is caught as unknown control flow
    (:func:`bitequiv.amdgcn.forward.cfg.has_unknown_control`) and fails closed;
  * ``v_cndmask_b32`` that computes a VALUE is kept as an opaque select node (sound: it only ever
    over-splits), and when it computes an ADDRESS the affine evaluator already picks the affine side
    (see :mod:`bitequiv.amdgcn.affine`).

So there is nothing to decode per-instruction here; this module documents the stance and exposes a
small helper for recognizing EXEC manipulation, used by future divergence-precision work.
"""
from __future__ import annotations

_EXEC_OPS = ("s_and_saveexec", "s_or_saveexec", "s_xor_saveexec", "s_andn2_saveexec", "s_mov_b64")


def touches_exec(inst):
    """True if the instruction reads/writes ``EXEC`` (a lane-mask manipulation). Not used by the
    minimal checker (divergence is caught in cfg); reserved for later divergence modeling."""
    if inst.opcode.startswith(_EXEC_OPS):
        return True
    return any(getattr(o, "name", "").startswith("exec") for o in inst.operands)
