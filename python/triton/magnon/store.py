"""ACF store: content-addressed by sha256(PTX), partitioned by GPU arch, scoped by ptxas version.

Layout:
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.<ptxas>.acf        # the best ACF (opaque bytes), version-tagged
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.<ptxas>.acf.json   # provenance metadata
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.<ptxas>.acf.cubin  # optional assembled-cubin cache
    $COMPILE_IQ_STORE/<arch>/<ptx_sha256>.acf                # legacy untagged (no version given)

PINNED (hash-key design): kernel identity is sha256(PTX) x arch. PTX subsumes the autotune config +
dtype/alignment specialization, but NOT runtime shape (M,N,K). The ptxas version scopes the artifact
(an ACF is only valid for the ptxas that produced it), so different toolchains keep distinct ACFs for
the same kernel instead of colliding. If per-shape ACFs are wanted later, extend the key with shape.
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


def acf_path(ptx_sha: str, arch: str, ptxas_version: str = "") -> str:
    # Version-tagged ("<sha>.<ptxas>.acf") when a ptxas version is given: an ACF is bound to the ptxas
    # that produced it (a mismatched ptxas rejects it), so one kernel keeps a distinct ACF per ptxas and
    # different toolchains coexist. Falls back to legacy untagged "<sha>.acf" when no version is given.
    name = f"{ptx_sha}.{_ptxas_tag(ptxas_version)}.acf" if ptxas_version else f"{ptx_sha}.acf"
    return os.path.join(store_root(), arch, name)


def has_acf(ptx_sha: str, arch: str, ptxas_version: str = "") -> bool:
    return os.path.exists(acf_path(ptx_sha, arch, ptxas_version))


def read_acf(ptx_sha: str, arch: str, ptxas_version: str = "") -> bytes | None:
    p = acf_path(ptx_sha, arch, ptxas_version)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()


def read_meta(ptx_sha: str, arch: str, ptxas_version: str = "") -> dict | None:
    """The provenance metadata (baseline_ms, best_ms, ...) for a stored ACF, or None if absent."""
    p = acf_path(ptx_sha, arch, ptxas_version) + ".json"
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def write_acf(ptx_sha: str, arch: str, data: bytes, meta: dict | None = None, ptxas_version: str = "") -> str:
    p = acf_path(ptx_sha, arch, ptxas_version)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    if meta is not None:
        with open(p + ".json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    return p


# --- assembled ACF-cubin cache --------------------------------------------------------------------
# The cubin produced by `ptxas PTX --apply-controls=<acf>` is cached here so consume doesn't re-run
# ptxas on every process. It lives in the compile_iq ACF store ONLY -- never in Triton's compile
# cache (which stays plain), so the opaqueness invariant holds. It is the assembly that's reused
# cross-process; the plain-vs-ACF A/B and its win decision remain in-memory per process.
#
# Tagged by ptxas version: the .acf bytes are version-independent, but the assembled cubin is only
# valid for the ptxas that produced it -- serving a mismatched cubin would be a correctness bug.


def _ptxas_tag(ptxas_version: str) -> str:
    return re.sub(r"[^0-9A-Za-z._]", "", ptxas_version or "")[:32] or "unknown"


def acf_cubin_path(ptx_sha: str, arch: str, ptxas_version: str) -> str:
    return os.path.join(store_root(), arch, f"{ptx_sha}.{_ptxas_tag(ptxas_version)}.acf.cubin")


def read_acf_cubin(ptx_sha: str, arch: str, ptxas_version: str) -> bytes | None:
    p = acf_cubin_path(ptx_sha, arch, ptxas_version)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()


def write_acf_cubin(ptx_sha: str, arch: str, ptxas_version: str, data: bytes) -> str:
    p = acf_cubin_path(ptx_sha, arch, ptxas_version)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp.{os.getpid()}"  # write+rename so a concurrent reader never sees a partial cubin
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)
    return p
