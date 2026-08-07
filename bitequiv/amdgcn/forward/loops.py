"""Loop-chunk recognition for AMDGCN — the AMD peer of the loop machinery in
``bitequiv.ptx.forward.loops`` / ``ptx_reduction._loop_steps``.

Minimal, label-free version: a hardware loop's chunk STEP (BLOCK_N / BLOCK_K) shows up as a
self-increment ``v_add / s_add R, R, IMM`` (dst == a source, other source a non-zero immediate).
The sorted multiset of those constants is a sound fence: a different chunk size gives a different
fence, so configs with different BLOCK_N/BLOCK_K never wrongly merge. It over-includes (a parallel
loop's step counts too), which only over-splits — never over-merges.

Label-aware loop reconstruction (``LoopReduce`` recovery of the accumulation, which needs the loop
back-edge the parser currently drops) is a later precision upgrade.
"""
from __future__ import annotations

from bitequiv.amdgcn.core.affine_algebra import _parse_int
from bitequiv.amdgcn.forward.cfg import branch_target
from bitequiv.amdgcn.parser import ImmediateOperand, RegisterOperand


def find_loops(func):
    """Loops as ``(header_body_index, latch_body_index)`` from back-edges — a conditional/uncond
    branch whose target label sits at or before it. ``func.labels`` maps a label name to the body
    index it precedes (parser-provided)."""
    labels = getattr(func, "labels", {}) or {}
    out = []
    for i, inst in enumerate(func.body):
        if inst.opcode.startswith(("s_cbranch", "s_branch")):
            ti = labels.get(branch_target(inst))
            if ti is not None and ti <= i:
                out.append((ti, i))
    return out


def loop_increments(func, header, latch):
    """Sorted non-zero self-increment constants (``v_add/s_add R, R, IMM``) inside a loop body
    ``[header, latch]`` — the chunk step(s) (BLOCK_N / BLOCK_K). The ``LoopReduce`` key: a different
    chunk size gives a different key -> sound split."""
    steps = set()
    for i in range(header, min(latch + 1, len(func.body))):
        inst = func.body[i]
        if not (inst.opcode.startswith("v_add") or inst.opcode.startswith("s_add")):
            continue
        ops = inst.operands
        if len(ops) < 3 or not isinstance(ops[0], RegisterOperand):
            continue
        dst = ops[0].name
        imm, src_regs = None, []
        for o in ops[1:]:
            if isinstance(o, ImmediateOperand):
                v = _parse_int(o.text)
                if v is not None:
                    imm = v
            elif isinstance(o, RegisterOperand):
                src_regs.append(o.name)
        if imm not in (None, 0) and dst in src_regs:
            steps.add(imm)
    return tuple(sorted(steps))


def _self_increments(func):
    """Every non-zero self-increment constant (``v_add/s_add R, R, IMM``) in the entry body."""
    steps = []
    for inst in func.body:
        if not (inst.opcode.startswith("v_add") or inst.opcode.startswith("s_add")):
            continue
        ops = inst.operands
        if len(ops) < 3 or not isinstance(ops[0], RegisterOperand):
            continue
        dst = ops[0].name
        imm, src_regs = None, []
        for o in ops[1:]:
            if isinstance(o, ImmediateOperand):
                v = _parse_int(o.text)
                if v is not None:
                    imm = v
            elif isinstance(o, RegisterOperand):
                src_regs.append(o.name)
        if imm not in (None, 0) and dst in src_regs:
            steps.append(imm)
    return steps


def loop_steps(func):
    """Sorted multiset of self-increment constants (the chunk steps of any loop in the entry). Used
    on the FINGERPRINT floor, where the tree is not reconstructed, so the COUNT of increments is a
    legitimate (if coarse) extent signal that must be kept."""
    return tuple(sorted(_self_increments(func)))


def loop_step_sizes(func):
    """Sorted DISTINCT self-increment constants — the chunk SIZE set (BLOCK_N / BLOCK_K), with the
    COUNT dropped. On the FAITHFUL path the reconstructed tree already carries the reduction extent
    (height = log2 of the element count), so the count of unrolled increments is redundant AND
    spurious: fewer num_warps unroll the same loop more times, which multiplied one chunk step into
    many and split configs that are bit-identical (``loops=(2,)`` vs ``loops=(2,2,2,2,2,2,2,2,2)``).
    Keeping only the distinct sizes still splits a genuinely different chunk size."""
    return tuple(sorted(set(_self_increments(func))))
