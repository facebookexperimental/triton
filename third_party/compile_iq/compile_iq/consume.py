"""Stage 3 — consumption helper. Given the PTX about to be assembled, return the
ptxas args that apply a stored ACF (or [] on miss/error). Called by a gated hook in
the NVIDIA backend's make_cubin, so the ACF is applied during normal PTX->SASS
compilation, including each config compiled during autotuning.

Fail-open by construction: any miss/error returns [] -> ptxas runs unchanged.
"""

import tempfile

from . import store


def _dbg(msg):
    store.dlog("consume", msg)


def acf_args_for(ptx: str, arch: str | None) -> list[str]:
    """Return ['--apply-controls=<tmp.acf>'] if the store has an ACF for this PTX, else []."""
    try:
        if not arch:
            return []
        sha = store.ptx_sha256(ptx)
        data = store.read_acf(sha, arch)
        if not data:
            _dbg(f"MISS {sha[:16]} {arch}")
            return []
        tmp = tempfile.NamedTemporaryFile(suffix=".acf", delete=False)
        tmp.write(data)
        tmp.close()
        _dbg(f"HIT {sha[:16]} {arch} -> {tmp.name}")
        return [f"--apply-controls={tmp.name}"]
    except Exception as e:  # never break compilation
        _dbg(f"error (fail-open): {type(e).__name__}: {e}")
        return []
