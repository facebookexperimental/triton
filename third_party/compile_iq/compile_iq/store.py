"""ACF store: content-addressed by sha256(PTX), partitioned by GPU arch.

Layout:
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.acf          # the best ACF (opaque bytes)
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.acf.json     # provenance metadata

PINNED (hash-key design): the key is sha256(PTX) x arch today. PTX subsumes the
autotune config + dtype/alignment specialization, but NOT runtime shape (M,N,K).
If per-shape ACFs are wanted later, extend the key with shape/identity.
"""

import hashlib
import json
import os
import re
import sys

# Single debug knob for all of compile_iq, default OFF. Set TRITON_COMPILE_IQ_DEBUG=1
# to trace when events are collected and when ACF lookups hit/miss.
DEBUG_ENV = "TRITON_COMPILE_IQ_DEBUG"


def debug_enabled() -> bool:
    return bool(os.environ.get(DEBUG_ENV))


def dlog(component: str, msg: str) -> None:
    """Write a debug line to stderr iff the (default-False) debug knob is set."""
    if debug_enabled():
        sys.stderr.write(f"[compile_iq.{component}] {msg}\n")


_SECTION_RE = re.compile(r"^\s*\.section\s+(\S+)")
_LOC_FILE_RE = re.compile(r"^\s*\.(loc|file)(\s|$)")


def normalize_ptx(ptx: str) -> str:
    """Strip volatile debug / source-path info so the content hash reflects only the
    actual code -- not where the kernel source happened to live. Triton bakes the
    source file path into PTX debug info (`.file`, the `// path:line` tail of every
    `.loc`, and the `.debug_*` DWARF sections), so the SAME kernel hashes differently
    depending on whether it was compiled from the user's real file, from replay's
    `source.py` copy, or served from the Triton cache. Dropping these makes the key
    location-independent; the executable instructions are byte-identical without them,
    so collect / factory / consume all agree on one key.
    """
    out = []
    in_debug = False
    for line in ptx.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            in_debug = m.group(1).startswith(".debug")
            if in_debug:
                continue
        if in_debug:
            continue
        if _LOC_FILE_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def ptx_sha256(ptx: str) -> str:
    return hashlib.sha256(normalize_ptx(ptx).encode("utf-8")).hexdigest()


def store_root() -> str:
    return os.environ.get("COMPILE_IQ_STORE", os.path.expanduser("~/.compile_iq/store"))


def acf_path(ptx_sha: str, arch: str) -> str:
    return os.path.join(store_root(), arch, f"{ptx_sha}.acf")


def has_acf(ptx_sha: str, arch: str) -> bool:
    return os.path.exists(acf_path(ptx_sha, arch))


def read_acf(ptx_sha: str, arch: str) -> bytes | None:
    p = acf_path(ptx_sha, arch)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()


def read_meta(ptx_sha: str, arch: str) -> dict | None:
    """The provenance metadata (baseline_ms, best_ms, ...) for a stored ACF, or None if absent."""
    p = acf_path(ptx_sha, arch) + ".json"
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def write_acf(ptx_sha: str, arch: str, data: bytes, meta: dict | None = None) -> str:
    p = acf_path(ptx_sha, arch)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    if meta is not None:
        with open(p + ".json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    return p
