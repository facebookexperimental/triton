"""Stage 3 — consumption helper. Given the PTX about to be assembled, return the ptxas args that
apply a stored ACF (or [] on miss/error). Called by a gated hook in the NVIDIA backend's make_cubin,
so the ACF is applied during normal PTX->SASS compilation. Fail-open: any miss/error returns [].
"""

import os

from . import store


def acf_args_for(ptx: str, arch: str | None, ptxas_version: str) -> list[str]:
    """['--apply-controls=<stored.acf>'] if the store has an ACF for this PTX and ptxas supports it
    (>=13.3), else []. Points ptxas straight at the stored ACF file -- no temp copy to leak."""
    try:
        from packaging.version import Version
        if not arch:
            return []
        # Minimum ptxas version that can apply our ACFs. Defaults to 13.3 (the version the
        # production CIQ search space `ptxas13.3.bin` targets). Override to match an ACF stored
        # from a different-versioned search space -- e.g. an EVO `cuda-13.0` tier needs ptxas 13.0,
        # since an ACF is rejected ("Invalid compiler controls file") by a mismatched ptxas.
        min_version = os.environ.get("COMPILE_IQ_PTXAS_MIN_VERSION", "13.3")
        if Version(ptxas_version) < Version(min_version):
            store.dlog(
                "consume", f"skip: ptxas {ptxas_version} < {min_version} -- cannot apply "
                "--apply-controls (set TRITON_PTXAS_BLACKWELL_PATH / COMPILE_IQ_PTXAS_MIN_VERSION)")
            return []
        sha = store.ptx_sha256(ptx)
        p = store.acf_path(sha, arch)
        if not os.path.exists(p):
            store.dlog("consume", f"MISS {sha[:16]} {arch}")
            return []
        store.dlog("consume", f"HIT {sha[:16]} {arch}")
        return [f"--apply-controls={p}"]
    except Exception as e:  # never break compilation
        store.dlog("consume", f"error (fail-open): {type(e).__name__}: {e}")
        return []
