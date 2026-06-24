"""Stage 3 — consumption helper. Given the PTX about to be assembled, return the ptxas args that
apply a stored ACF (or [] on miss/error). Fail-open: any miss/error returns [].

Consumption is IN-MEMORY (gated by TRITON_COMPILE_IQ_APPLY), driven by the core compiler's
_maybe_apply_compile_iq -> backend apply_compile_iq_acf on BOTH the cache-hit and freshly-compiled
paths. The backend re-assembles the cubin from the (cached or fresh) PTX with --apply-controls and
stashes it as a *pending candidate* on the CompiledKernel -- it does NOT overwrite the live cubin.
The first real launch runs a plain-vs-ACF benchmark competition (CompiledKernel._compile_iq_resolve)
and keeps the winner, so consumption can never regress vs baseline (offline ACF wins are noisy and
don't always reproduce in-process). Triton's compile cache keeps its plain no-ACF cubin untouched;
the ACF cubin is in-memory only. So the ACF is opaque to the compile cache by construction -- a cache
hit still re-checks the ACF store -- TRITON_ALWAYS_COMPILE is not required, and an APPLY-off run
reloading the same cache entry can never pick up an ACF cubin.
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
