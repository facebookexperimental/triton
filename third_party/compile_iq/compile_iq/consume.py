"""Stage 3 — consumption helper. Given the PTX about to be assembled, return the ptxas args that
apply a stored ACF (or [] on miss/error). Called by a gated hook in the NVIDIA backend's make_cubin,
so the ACF is applied during normal PTX->SASS compilation. Fail-open: any miss/error returns [].

TODO(compile_iq): the applied ACF is INVISIBLE to Triton's compile cache. The `--apply-controls`
flag is appended inside make_cubin from the FBTRITON_COMPILE_IQ_APPLY env + this store lookup; it is
NOT part of `opt.ptx_options`/backend_options, and the ACF *bytes* aren't hashed anywhere. The cache
key is `triton_key-src.hash-backend.hash-backend_options.hash-env_vars` (runtime/cache.py
get_cache_key), and FBTRITON_COMPILE_IQ_APPLY is not in get_cache_invalidating_env_vars(). So a
cached cubin is reused without re-running make_cubin -> the ACF is silently NOT applied, and a
re-tuned ACF for the same kernel does not bust the cache. Consequence: consume currently REQUIRES
TRITON_ALWAYS_COMPILE=1 to take effect. Fix: fold the ACF identity (e.g. its sha256, or the store
hit) into the compile cache key so a hit / changed ACF auto-triggers recompilation and
TRITON_ALWAYS_COMPILE is no longer needed.
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
        # TODO(compile_iq): this HIT does not invalidate Triton's compile cache (see module
        # docstring) -- relies on TRITON_ALWAYS_COMPILE=1. Fold `sha` into the cache key instead.
        return [f"--apply-controls={p}"]
    except Exception as e:  # never break compilation
        store.dlog("consume", f"error (fail-open): {type(e).__name__}: {e}")
        return []
