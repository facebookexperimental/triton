"""A small structural parser for AMDGCN (gfx9xx CDNA) assembly text.

AMDGCN has no pure-Python parser equivalent to ``pyptx`` (searched — none exists), so this
is a hand-written line parser producing the instruction/operand node model the rest of the
engine consumes. It is deliberately close in shape to the ``pyptx`` node surface the PTX
engine expects (``Function`` with ``is_entry``/``body``, ``Instruction`` with
``opcode``/``modifiers``/``operands``, ``RegisterOperand``/``VectorOperand``/
``ImmediateOperand``) so ``linker``/``affine``/``leaves``/``builder`` mirror their PTX peers.

Grammar handled (per LLVM ``AMDGPUOperandSyntax``):
  * one instruction per line: ``mnemonic op0, op1, ... [modifier ...]``. The comma-separated
    operands come first; trailing space-separated tokens (``offen``, ``glc``, ``row_shr:8``,
    ``quad_perm:[2,3,0,1]``, ``bound_ctrl:1``) are modifiers.
  * register ranges ``v[2:5]`` / ``s[8:11]`` expand to their individual registers with a slot
    index (the AMD echo of a PTX ``v4`` vector dest), so def-use tracks sub-registers.
  * single registers ``v1``/``s0``/``vcc``/``exec`` (matching ``_SINGLE`` / ``_SPECIAL_REGS``).
    Everything else becomes an ``ImmediateOperand``: decimal / ``0x..`` hex / ``0f..``/``0d..``
    float literals, and non-register tokens such as ``off``/``null``/``vmcnt(0)`` (kept verbatim).
  * kernels are the ``.type NAME,@function`` symbols; body = the instructions from ``NAME:``
    to ``s_endpgm``. Debug (``.loc``/``.file``/``.cfi_*``), ``;``/``//`` comments, and other
    directives are skipped. ``.amdgpu_metadata`` YAML supplies ``.max_flat_workgroup_size``.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegisterOperand:
    """A single hardware register: ``v1``, ``s0``, ``vcc``, ``exec``. ``name`` is the token."""

    name: str


@dataclass(frozen=True)
class VectorOperand:
    """A register range ``v[2:5]`` / ``s[8:11]`` expanded to its element registers (with
    slot 0..N-1), the AMD echo of a PTX ``v2``/``v4`` vector operand."""

    elements: tuple  # tuple[RegisterOperand]
    text: str = ""


@dataclass(frozen=True)
class ImmediateOperand:
    """A literal: integer (``0x100``/``256``), float (``0f3f800000``), or an opaque token
    (``vmcnt(0)``, ``off``, ``null``) kept verbatim."""

    text: str


@dataclass
class Instruction:
    opcode: str
    modifiers: tuple
    operands: list
    predicate: object = None  # AMDGCN has no per-instruction predicate token (uses exec/vcc)
    raw: str = ""


@dataclass
class Function:
    name: str
    is_entry: bool
    body: list
    reqntid: dict = field(default_factory=dict)  # {'x': flat_workgroup_size} when known
    directives: tuple = ()


# ---- tokenizing -----------------------------------------------------------

_RANGE = re.compile(r"^(?P<pfx>[vsa])\[(?P<lo>\d+):(?P<hi>\d+)\]$")
_SINGLE = re.compile(r"^(?P<pfx>[vsa])(?P<num>\d+)$")
_SPECIAL_REGS = frozenset({"vcc", "vcc_lo", "vcc_hi", "exec", "exec_lo", "exec_hi", "scc", "m0"})


def _split_operands(args):
    """Split the operand region on top-level commas (not inside ``[...]`` / ``(...)``)."""
    out, depth, cur = [], 0, []
    for ch in args:
        if ch in "[(":
            depth += 1
            cur.append(ch)
        elif ch in "])":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _mk_operand(tok):
    m = _RANGE.match(tok)
    if m:
        pfx, lo, hi = m.group("pfx"), int(m.group("lo")), int(m.group("hi"))
        els = tuple(RegisterOperand(f"{pfx}{n}") for n in range(lo, hi + 1))
        return VectorOperand(els, text=tok)
    if _SINGLE.match(tok) or tok in _SPECIAL_REGS:
        return RegisterOperand(tok)
    return ImmediateOperand(tok)


def parse_instruction(line):
    """Parse one instruction line into an :class:`Instruction`, or ``None`` if not an
    instruction (label / directive / comment / blank).

    Operands are the top-level comma-separated tokens; only the LAST comma-chunk may carry
    trailing space-separated modifiers (``offen``, ``row_shr:8``, ``quad_perm:[..]``), since
    AMDGCN modifiers always follow the operand list."""
    # strip trailing comment
    s = line
    for c in (";", "//"):
        i = s.find(c)
        if i != -1:
            s = s[:i]
    s = s.strip()
    if not s or s.endswith(":") or s.startswith("."):
        return None
    parts = s.split(None, 1)
    opcode = parts[0]
    if not re.match(r"^[a-z][a-z0-9_]*$", opcode):
        return None  # not a mnemonic (stray token)
    args = parts[1] if len(parts) > 1 else ""
    chunks = _split_operands(args)
    operands, modifiers = [], []
    for idx, chunk in enumerate(chunks):
        if idx < len(chunks) - 1:
            operands.append(_mk_operand(chunk))  # a comma-separated operand
            continue
        toks = chunk.split()  # last chunk: <operand> <mod> <mod> ...
        for j, tok in enumerate(toks):
            (operands if j == 0 else modifiers).append(_mk_operand(tok) if j == 0 else tok)
    return Instruction(opcode=opcode, modifiers=tuple(modifiers), operands=operands, raw=s)


# ---- workgroup size from .amdgpu_metadata ---------------------------------

# `.amdgpu_metadata` YAML fields (matched against an already-lstripped line).
_MD_NAME = re.compile(r"^\.name:\s*(\S+)")
_MD_WG = re.compile(r"^\.max_flat_workgroup_size:\s*(\d+)")


def _normalize_symbol(name):
    r"""Normalize a symbol token so the metadata ``.name`` matches the ``.type NAME,@function``
    / label form: strip surrounding quotes (LLVM may emit ``.name: 'kernel'``) and a leading
    ``\01`` assembler mangling prefix. Applied to both sides (metadata name, .type name, and
    the body label) so the keys always agree — otherwise reqntid would silently fall back to
    ``{}``."""
    name = name.strip().strip("'\"")
    if name.startswith("\\01"):
        name = name[3:]
    return name


def _workgroup_sizes(text):
    r"""Map kernel name -> flat workgroup size from the ``.amdgpu_metadata`` YAML block.

    Within a kernel record ``.name`` and ``.max_flat_workgroup_size`` are adjacent but their
    order is not fixed (LLVM emits size before name), so pair whichever comes first with the
    next of the other. The pending name/size are RESET at every record boundary (a ``-`` YAML
    list item) so a record missing one field can never mis-pair with the next record's field.
    The captured name is normalized (quotes / ``\01`` prefix) to match the label key."""
    out = {}
    pend_name, pend_size = None, None
    in_meta = False
    for line in text.splitlines():
        if ".amdgpu_metadata" in line:
            in_meta = True
            continue
        if ".end_amdgpu_metadata" in line:
            break
        if not in_meta:
            continue
        rest = line.lstrip()
        if rest.startswith("-"):  # new kernel record -> drop any incomplete pending pair
            pend_name = pend_size = None
            rest = rest[1:].lstrip()  # a key may sit on the same line as the "- " marker
        if (m := _MD_NAME.match(rest)):
            pend_name = _normalize_symbol(m.group(1))
        elif (m := _MD_WG.match(rest)):
            pend_size = int(m.group(1))
        if pend_name is not None and pend_size is not None:  # both seen in THIS record -> commit
            out[pend_name] = pend_size
            pend_name = pend_size = None
    return out


# ---- module parsing -------------------------------------------------------

_KERNEL_TYPE = re.compile(r"^\s*\.type\s+(\S+),@function")


def _kernel_names(text):
    return [_normalize_symbol(m.group(1)) for line in text.splitlines() if (m := _KERNEL_TYPE.match(line))]


def _strip_comment(line):
    """Drop a trailing ``;`` / ``//`` comment and surrounding whitespace."""
    s = line
    for c in (";", "//"):
        i = s.find(c)
        if i != -1:
            s = s[:i]
    return s.strip()


def _label_of(line):
    """Kernel label name if the (comment-stripped) line is ``NAME:``, else ``None``.
    Normalized to match the ``.type`` / metadata name keys."""
    s = _strip_comment(line)
    return _normalize_symbol(s[:-1]) if s.endswith(":") and " " not in s[:-1] else None


def parse(text):
    """Parse an AMDGCN module into a list of entry :class:`Function` (one per kernel).

    Body extraction: for each ``NAME:`` label of a known kernel, collect instructions until
    ``s_endpgm`` (or the next kernel label). Robust to the debug/``.loc`` interleaving."""
    names = set(_kernel_names(text))
    wg = _workgroup_sizes(text)
    lines = text.splitlines()
    funcs = []
    i, n = 0, len(lines)
    while i < n:
        name = _label_of(lines[i])
        if name in names:
            body = []
            i += 1
            while i < n:
                lab = _label_of(lines[i])
                if lab in names:  # next kernel starts
                    i -= 1
                    break
                inst = parse_instruction(lines[i])
                if inst is not None:
                    body.append(inst)
                    if inst.opcode == "s_endpgm":
                        break
                i += 1
            reqntid = {"x": wg[name]} if name in wg else {}
            funcs.append(Function(name=name, is_entry=True, body=body, reqntid=reqntid))
        i += 1
    return funcs
