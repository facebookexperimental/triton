"""AMDGCN matrix-core (MFMA / WMMA) recognition — the AMD peer of ``bitequiv.ptx.mma``.

On gfx942 (CDNA3) the only tensor-core path is ``v_mfma_*`` (plus sparse ``v_smfmac_*``);
there is no ``mma.sync`` vs ``tcgen05`` (v2 vs v5) split as on NVIDIA, so the v2==v5 unifier
is a no-op here — one matmul path, keyed by its dtype family. The MFMA mnemonic ENCODES the
tile shape (``v_mfma_f32_16x16x16_f16`` = M16 N16 K16, f16 operands / f32 acc), so a GEMM's
tiling is "how many MFMA instructions" (the count), and the instruction shape is fixed —
which makes a PRESENCE-keyed fence tiling-invariant.

Soundness stance (matches the NV fence): the K reduction inside a matrix instruction is not
FP-order-recoverable, so any config containing MFMA is decided by this fence, not by a
reconstructed tree. The fence keys on the SET of matmul mnemonics + the PRESENCE of epilogue
fp-op kinds (add/mul vs fma) — not their counts, which scale with the M/N tile. fp8 is the
exception: its accumulate/flush cadence is bit-deciding, so fp8 keeps the per-mnemonic count.
"""
from __future__ import annotations

from bitequiv.amdgcn.linker import linearize

_MMA_PREFIXES = ("v_mfma", "v_smfmac", "v_wmma")
_FP8_MARKERS = ("fp8", "_f8", "bf8", "e4m3", "e5m2")


def _base(opcode):
    """Strip the encoding-size / DPP suffix (``_e32`` / ``_e64`` / ``_dpp``)."""
    for suf in ("_e32", "_e64", "_dpp"):
        if opcode.endswith(suf):
            return opcode[:-len(suf)]
    return opcode


def _fp_kind(base):
    """The epilogue fp-op kind of a scalar/packed AMD combine, or ``None``.

    ``fma``/``fmac`` collapse to ``"fma"``; ``add``/``sub``/``mul`` stay themselves. The
    ``v_pk_`` (packed SIMD-2) prefix is normalized away first."""
    b = base.replace("v_pk_", "v_")
    for k in ("fma", "fmac", "add", "sub", "mul"):
        if b.startswith("v_" + k + "_"):
            return "fma" if k in ("fma", "fmac") else k
    return None


def is_mma(inst):
    return inst.opcode.startswith(_MMA_PREFIXES)


def mma_token(inst):
    """The full MFMA mnemonic (encodes MxNxK tile + operand/acc dtypes). Kept for the fp8 fence and
    the K-extent count."""
    return inst.opcode


def mma_family(inst):
    """The FORM-AGNOSTIC matmul token: the operand dtype family only (the trailing dtype of the MFMA
    mnemonic, e.g. ``v_mfma_f32_16x16x16_f16`` -> ``f16``, ``..._32x32x8_bf16`` -> ``bf16``). Drops
    the MxNxK tile so different BLOCK_M/N/K tilings of the same-dtype GEMM merge — sound on gfx942,
    where the empirical fuzzer shows the tensor-core K reduction is tile/block_k bit-invariant for
    f16/bf16/f32 (fp8 is handled separately, kept conservative)."""
    parts = inst.opcode.split("_")
    return parts[-1] if parts else inst.opcode


def _is_fp8(tokens):
    return any(any(m in t for m in _FP8_MARKERS) for t in tokens)


def mma_fence(func):
    """The sound matrix-core fence for an entry function, or ``None`` if it has no MFMA.

    Returns a canonical string. Non-fp8: the SET of matmul mnemonics + the PRESENCE set of
    epilogue fp-op kinds (tiling-invariant). fp8: the per-mnemonic COUNT map (cadence is
    bit-relevant) + the kinds. Two configs whose fences differ are never merged.
    """
    mma_tokens, mma_counts, families, fp_kinds = set(), {}, set(), set()
    for inst in linearize(func):
        if is_mma(inst):
            tok = mma_token(inst)
            mma_tokens.add(tok)
            mma_counts[tok] = mma_counts.get(tok, 0) + 1
            families.add(mma_family(inst))
        else:
            kind = _fp_kind(_base(inst.opcode))
            if kind is not None:
                fp_kinds.add(kind)
    if not mma_tokens:
        return None
    kinds = ",".join(sorted(fp_kinds))
    if _is_fp8(mma_tokens):  # fp8 accumulate cadence is bit-deciding -> keep the per-mnemonic count
        counts = ",".join(f"{t}:{mma_counts[t]}" for t in sorted(mma_counts))
        return "mma_fp8{" + counts + "}|kinds{" + kinds + "}"
    # f16/bf16/f32: tile-invariant on gfx942 -> key on the dtype FAMILY set (drop the MxNxK tile).
    return "mma{" + ",".join(sorted(families)) + "}|kinds{" + kinds + "}"


def mma_token_counts(func):
    """Sorted ``(token, count)`` over all MFMA in the function = the matmul REDUCTION EXTENT
    (K / head dim). Used only on the fail-closed side to restore the K extent a form-agnostic
    token would drop."""
    counts = {}
    for inst in linearize(func):
        if is_mma(inst):
            tok = mma_token(inst)
            counts[tok] = counts.get(tok, 0) + 1
    return tuple(sorted(counts.items()))
