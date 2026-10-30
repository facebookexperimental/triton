"""The naming discipline, enforced instead of audited by hand.

Anything that goes into a canonical artifact is bound by the discipline,
regardless of which module produced it -- module ownership is not the
criterion, participation in canonical output is.

This is legal hygiene infrastructure, the same class as the provenance and
hash-chain checks.  It constrains what the tree may contain, never what the
solver may decide, so it is not a solver guard.

The frozen pre-rename artifacts below are append-never: they are read only by
the code that wrote them and are enumerated once here rather than tracked by a
moving count, because a numeric baseline rots silently (63 quietly became 75)
while a whitelist cannot.
"""

import re
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]

# Assembled so the token is not itself a literal anywhere in the tree.
FORBIDDEN = "t" + "will"

# Frozen v6/v7/v8 outputs, superseded by the lit1 batch and never rewritten.
FROZEN_ARTIFACT_DIRS = ("ablations_v7",)
FROZEN_ARTIFACT_STEMS = (
    "bwd_lr4096_v7",
    "bwd_v7",
    "fwd_subtiled_v7",
    "fwd_v7",
    "fwd_subtiled_v8",
)
SCANNED_SUFFIXES = (".py", ".sh", ".json", ".md")


def _is_frozen(path: Path) -> bool:
    relative = path.relative_to(PKG_ROOT)
    parts = relative.parts
    if parts[0] in FROZEN_ARTIFACT_DIRS:
        return True
    if parts[0] != "solutions":
        return False
    return any(stem in part for stem in FROZEN_ARTIFACT_STEMS for part in parts[1:])


def _scanned_files():
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", "buck-out"}
    for path in PKG_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if skip.intersection(path.parts):
            continue
        yield path


def test_no_source_file_carries_the_paper_codename():
    offenders = []
    for path in _scanned_files():
        if path == Path(__file__) or _is_frozen(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if FORBIDDEN in text.lower():
            offenders.append(str(path.relative_to(PKG_ROOT)))
    assert offenders == [], (
        "the paper codename may appear only in frozen pre-rename artifacts; "
        f"found it in: {sorted(offenders)}"
    )


def test_every_schema_constant_uses_an_approved_prefix():
    """Schema strings are identities; the prefix is what carries the identity."""
    # "jos-" is the permitted alias: HANDOFF bars it from commit messages, not
    # from artifact identities. "oracle-" belongs to the ptxas oracle subsystem.
    # "prepared-" is the disclosed parallel regime: its artifacts must not be
    # mistakable for paper ones, so they carry their own identity rather than
    # a "paper-joint-" version bump.
    approved = (
        "paper-joint-",
        "emitter-",
        "curated-ddg-",
        "curation-manifest-",
        "jos-",
        "oracle-",
        "prepared-joint-",
    )
    pattern = re.compile(r'^([A-Z][A-Z0-9_]*SCHEMA[A-Z0-9_]*)\s*=\s*"([^"]+)"', re.M)
    seen = {}
    for path in _scanned_files():
        if path.suffix != ".py" or _is_frozen(path):
            continue
        for name, value in pattern.findall(path.read_text(encoding="utf-8")):
            seen[f"{path.relative_to(PKG_ROOT)}:{name}"] = value

    assert seen, "no schema constants were discovered -- the scan is broken"
    bad = {
        where: value
        for where, value in seen.items()
        if not value.startswith(approved)
    }
    assert bad == {}, f"schema constants outside the approved naming family: {bad}"


@pytest.mark.parametrize(
    "constant",
    ["AUTHORING_SCHEMA", "MAPPING_SCHEMA", "MEMORY_PLAN_SCHEMA", "SYNC_PLAN_SCHEMA"],
)
def test_skc_scaffold_schemas_were_renamed(constant):
    from skc import _schema

    assert getattr(_schema, constant).startswith("paper-joint-")
