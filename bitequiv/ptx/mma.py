"""Shared MMA (tensor-core) fence for the bitwise-equivalence checker.

An MMA op (``wgmma`` / ``mma.sync`` / ``wmma`` / ``tcgen05.mma``) is a COLLECTIVE matmul whose
internal accumulation order we do not reconstruct. Instead we summarize an entry's MMA ops with a
tiling-invariant FENCE: a conjunction of facts that can only SPLIT equivalence classes, never merge
them. This lets the autotuner keep MMA configs that differ ONLY in a bit-irrelevant tiling choice
(BLOCK_M / BLOCK_N re-tiling, num_warps) in one class, while separating anything that changes the
bits (K split, operand/accumulator dtype, scale/accumulate flags).

Soundness (over-split, never over-merge):
- Each KEPT token piece is bit-relevant: K (the products summed per hardware pass — regrouping K
  changes the sum) and the operand + accumulator dtypes (rounding). tcgen05 keeps ``.kind::`` (dtype
  family) and ``.cta_group::`` (2-CTA changes the accumulation).
- Each DROPPED piece is tiling-free: the m/n tile dims (retiling the SAME dot products across a
  different BLOCK_M / BLOCK_N is bit-identical) and the issue/transpose mods
  (``.mma_async`` / ``.sync`` / ``.aligned`` / ``.row`` / ``.col``).
- The f32 epilogue fp ops ride as PRESENCE ``(has_fma, has_addmul)`` — NOT a count: the op count
  scales with the M/N tile (elements per thread) while tcgen05's mma count does not, so a count
  (even GCD-reduced) cannot cancel the tile scaling and over-splits equivalent re-tilings (measured:
  num_splits 1..8 x tilings -> 20 classes for a 4-class split-K kernel). Presence keeps
  enable_fp_fusion (fma single-rounded vs add+mul double-rounded) distinct; the bit-relevant K
  regrouping rides the token / caller's ``loops=`` (BLOCK_K) / ``splits=`` (num_splits) instead.
- fp8 (e4m3/e5m2/...): the ``max_num_imprecise_acc`` periodic-flush cadence is invisible even to
  presence, so we FALL BACK to the raw per-token count map + raw fp count (strictly more splitting).
- tcgen05 M/N/K live in a runtime descriptor, not the modifiers, so tcgen05 K rides ``loops=``
  (BLOCK_K) + ``splits=`` (num_splits) rather than the token — conservative.

This module is the single source of truth for both the forward interpreter and the backward
``_entry_signature`` (which currently uses the coarser ``_mma_guard``).
"""
import re

from bitequiv.ptx.linker import linearize

# The four tensor-core op families. wgmma/wmma/tcgen05 are families with non-matmul members
# (fences, commits, loads) that _is_mma filters out; bare `mma` (mma.sync) is always the matmul.
_MMA_OPCODES = frozenset({"wgmma", "mma", "wmma", "tcgen05"})

# Operand / accumulator element dtypes that appear as MMA modifier tokens. KEPT in the fence token
# (rounding is bit-relevant). fp8 additionally trips the conservative fallback.
_FP8_DTYPES = frozenset({".e4m3", ".e5m2", ".e3m2", ".e2m3", ".e2m1"})
_MMA_DTYPES = frozenset({".f64", ".f32", ".tf32", ".f16", ".bf16", ".s32", ".s8", ".u8", ".s4",
                         ".u4", ".b1"}) | _FP8_DTYPES

# The single tile+K modifier token, e.g. `.m64n128k16` (wgmma / mma.sync / wmma).
_TILE_RE = re.compile(r"^\.m(\d+)n(\d+)k(\d+)$")

# fp reduce/epilogue ops. Their COUNT scales with the M/N tile (elements per thread), so the fence
# records only their PRESENCE (has_fma, has_addmul), never the count. f32 family only (the accumulate
# / epilogue precision that matters).
_FP_EPI_OPS = frozenset({"add", "sub", "mul", "fma"})
_FP_EPI_WIDTHS = frozenset({".f32", ".f32x2"})


def _is_mma(inst):
    """True iff ``inst`` is a tensor-core MATMUL (not a fence / commit / load member of the family)."""
    op = inst.opcode
    if op == "mma":
        return True  # mma.sync — the bare opcode is the matmul
    if op == "wgmma":
        return any(m.startswith(".mma") for m in inst.modifiers)  # .mma_async (not .fence/.commit/.wait)
    if op == "wmma":
        return ".mma" in inst.modifiers  # not .load / .store
    if op == "tcgen05":
        return ".mma" in inst.modifiers  # not .commit / .ld / .st / .fence / .cp / .wait
    return False


def _k_of(mods):
    """The K tile dim (int) from the `.m<M>n<N>k<K>` token, or None if absent (e.g. tcgen05)."""
    for m in mods:
        mo = _TILE_RE.match(m)
        if mo:
            return int(mo.group(3))
    return None


# Operand dtype -> accumulation FAMILY (narrowest operand wins; the accumulator is the wider f32).
# f16 and bf16 map to ONE family "f16": on Blackwell both lower through tcgen05 `.kind::f16` and the
# same F=25-bit fixed-point accumulate, and a native tcgen05 GEMM does not even encode f16-vs-bf16 in
# the PTX (see `_mma_token`). tf32/f32 -> "tf32" (a tensor-core f32 input is first rounded to tf32).
# fp8 kinds are handled separately in `_mma_token` (kept conservative — fp8 is the one v2 != v5 case).
_FAMILY_PRIORITY = (({".f16", ".bf16"}, "f16"), ({".tf32"}, "tf32"), ({".f64"}, "f64"),
                    ({".s8", ".u8"}, "s8"), ({".s4", ".u4"}, "s4"), ({".b1"}, "b1"), ({".f32"}, "tf32"))


def _dtype_family(dtypes):
    """The accumulation family for a set of MMA operand dtypes. Narrowest operand wins because the
    accumulator is always the wider f32; e.g. ``{.f32, .bf16}`` (acc f32, operands bf16) -> "f16"."""
    for group, fam in _FAMILY_PRIORITY:
        if any(d in dtypes for d in group):
            return fam
    return "?"


def _fence_all_matmul(fence):
    """True iff every token in an :func:`_mma_fence` is the canonical non-fp8 ``matmul|`` form — i.e. a
    pure f16/bf16/tf32 tensor-core GEMM. The forward descriptor uses this to drop the bit-free
    ``loops=`` (BLOCK_K) for such GEMMs, while keeping it for fp8 (``max_num_imprecise_acc`` cadence)
    and the native-form cases."""
    return fence is not None and fence[0] == "mma" and all(t.startswith("matmul|") for t in fence[1])


def _mma_token(opcode, mods):
    """Form-agnostic, tiling-invariant matmul token. The v2 warp-level MMA (``mma.sync``) and the v5
    Blackwell MMA (``tcgen05.mma``) that compute the SAME matmul get the SAME token, so the autotuner
    can move freely between them. Keeps only the bit-relevant facts — the accumulation dtype FAMILY and
    the CTA group (2-CTA changes the accumulation) — and drops the FORM (mma vs tcgen05), the native
    m/n/k tile, and the issue/transpose mods (all bit-free re-tilings of the same dot products).

    Why v2 == v5 is sound for f16/bf16/tf32 (INFERRED, not an NVIDIA guarantee): for these types each
    product ``A*B`` is exact in f32 (f16/tf32 give <=22 product significand bits, bf16 <=16, f32 holds
    24 -> no rounding on the multiply), and both forms on the same Blackwell chip add the exact products
    in the same wide fixed-point accumulator (F=25 fractional bits) with a single normalize -> same
    bits. NVIDIA documents none of this; it is corroborated by two reverse-engineered bit-accurate
    tensor-core models (arXiv 2511.10909 map both HMMA and UTCHMMA to FDA(F=25); arXiv 2512.07004 find
    B200 == H100/H200 behavior, tensor cores NOT IEEE-754) and by our GB300 corpus (v2 and v5 land in
    one bit class for f16/bf16/tf32). fp8 is the EXCEPTION and is kept conservative below: fp8
    accumulation runs in limited precision with a periodic promote-to-f32 (Triton `max_num_imprecise_acc`
    + a fp8-specific block ordering) that need not match across the two lowerings, so fp8 v2 != v5.

    Note (dtype-relative): a native tcgen05 GEMM does not encode the specific operand dtype (``.kind::f16``
    covers f16 AND bf16; the inputs are untyped ``.b16``), so the family cannot tell f16 from bf16 for
    v5. The token is therefore dtype-RELATIVE: f16 and bf16 share the "f16" family. This is sound for
    how the checker is used (the autotuner compares configs of ONE kernel = one fixed dtype), but two
    DIFFERENT-dtype GEMMs would share a token."""
    if opcode == "tcgen05":
        kind = next((m[len(".kind::"):] for m in mods if m.startswith(".kind::")), "?")
        if kind == "f8f6f4":  # fp8: v2 != v5 -> keep the native tcgen05 form so it never merges into matmul|
            keep = sorted(m for m in mods if m.startswith(".kind::") or m.startswith(".cta_group::"))
            return "tcgen05|" + ",".join(keep)
        cta = next((m[len(".cta_group::"):] for m in mods if m.startswith(".cta_group::")), "1")
        return f"matmul|{kind}|cta{cta}"
    dtypes = [m for m in mods if m in _MMA_DTYPES]
    if any(d in _FP8_DTYPES for d in dtypes):  # native fp8 (wgmma / mma.sync): keep the conservative form
        k = _k_of(mods)
        return f"{opcode}|k{k}|{','.join(dtypes)}"
    cta = next((m[len(".cta_group::"):] for m in mods if m.startswith(".cta_group::")), "1")
    return f"matmul|{_dtype_family(dtypes)}|cta{cta}"


def _imm(operand):
    """The integer-immediate text of an operand (e.g. a wgmma scale/accumulate flag), or None for a
    register / vector / address / non-numeric operand."""
    txt = getattr(operand, "text", None) or getattr(operand, "name", None) or str(operand)
    txt = str(txt).strip()
    return txt if txt.lstrip("-").isdigit() else None


def _mma_flags(inst):
    """The set of immediate flag operands (scale-D / scale-A / scale-B / transpose imms). These are
    kernel-fixed (not autotuner knobs), so they never split configs of one kernel spuriously; kept
    as a belt so a config that DID differ in an accumulate/overwrite flag would still separate."""
    return {v for v in (_imm(o) for o in inst.operands) if v is not None}


def _fp_epi_counts(func):
    """(fma_count, addmul_count) of f32-family epilogue fp ops, counted SEPARATELY. enable_fp_fusion
    flips fma <-> mul+add in the epilogue, and the compiler can emit the SAME TOTAL count either way
    (measured on gemm_bias_relu_fp_fusion: 16 fma fused vs 16 add/mul unfused) — so a single lumped
    count collides and over-merges (fma is single-rounded, mul+add double-rounded -> different bits).
    Keeping fma apart splits fp_fusion on/off. Both COUNTS scale with the M/N tile (elements per
    thread), so the fence records only their PRESENCE, not the count (a count over-splits equivalent
    re-tilings — see :func:`_mma_fence`)."""
    fma, addmul = 0, 0
    for inst in linearize(func):
        if not (inst.modifiers and any(m in _FP_EPI_WIDTHS for m in inst.modifiers)):
            continue
        if inst.opcode == "fma":
            fma += 1
        elif inst.opcode in _FP_EPI_OPS:  # add / sub / mul
            addmul += 1
    return fma, addmul


def _mma_fence(func):
    """Tiling-invariant fence for an entry's MMA ops, or ``None`` if the entry has no MMA.

    Returns a hashable tuple. Two entries with equal fences differ only in a bit-IRRELEVANT tiling
    choice; any bit-relevant difference (K split, dtype, flag, enable_fp_fusion) yields a distinct
    fence. The f32 epilogue rides as PRESENCE ``(has_fma, has_addmul)`` — NOT a count: the op count
    scales with the M/N tile (elements per thread) while tcgen05's mma count does not, so a count
    (even GCD-reduced) cannot cancel the tile scaling and over-splits equivalent re-tilings. The
    bit-relevant K regrouping (BLOCK_K, num_splits) rides the token / caller's ``loops=`` /
    ``splits=`` instead of the op count. fp8 still falls back to the raw per-token count map + raw fp
    count (strictly more splitting) because its imprecise-acc flush cadence is invisible even to
    presence."""
    insts = [i for i in linearize(func) if _is_mma(i)]
    if not insts:
        return None
    tokens = tuple(sorted(_mma_token(i.opcode, i.modifiers) for i in insts))
    flags = set()
    for i in insts:
        flags |= _mma_flags(i)
    flags = frozenset(flags)
    fma, addmul = _fp_epi_counts(func)
    if any(any(d in t for d in _FP8_DTYPES) for t in tokens):
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return ("mma-fp8", tuple(sorted(counts.items())), flags, (fma, addmul))
    return ("mma", frozenset(tokens), flags, (int(fma > 0), int(addmul > 0)))
