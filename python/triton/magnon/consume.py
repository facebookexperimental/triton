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

import dataclasses
import os

from . import store

# Total bytes of tensor storage the consume A/B may copy; above it the A/B is skipped (plain is kept).
AB_COPY_LIMIT_ENV = "TRITON_COMPILE_IQ_AB_COPY_LIMIT_MB"
_AB_COPY_LIMIT_MB_DEFAULT = 1024
_SCALAR_TYPES = (bool, int, float, str, type(None))


def scratch_args(args):
    """Copies of the launch args for the plain-vs-ACF A/B, or None when the A/B must be skipped.

    The A/B launches the kernel many times before the real launch. A kernel that modifies its own
    inputs (in-place update, atomics, split-K accumulation) would corrupt the caller's buffers, so
    the A/B runs on copies instead. Each tensor's whole storage is copied once and the view is
    rebuilt with the same offset, sizes and strides, so aliasing between args is preserved. Returns
    None for any arg it cannot redirect (an object that may carry a raw pointer) or when the copies
    would exceed TRITON_COMPILE_IQ_AB_COPY_LIMIT_MB; the caller then keeps plain."""
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    limit = int(os.environ.get(AB_COPY_LIMIT_ENV, _AB_COPY_LIMIT_MB_DEFAULT)) << 20
    storages = {}  # storage data_ptr -> copied storage, so aliased args share one copy

    def copy(t):
        s = t.untyped_storage()
        if s.data_ptr() not in storages:
            if sum(c.nbytes() for c in storages.values()) + s.nbytes() > limit:
                raise OverflowError
            storages[s.data_ptr()] = s.clone()
        return t.new_empty(0).set_(storages[s.data_ptr()], t.storage_offset(), t.size(), t.stride())

    out = []
    try:
        for a in args:
            if isinstance(a, torch.Tensor):
                out.append(copy(a))
            elif isinstance(a, TensorDescriptor) and isinstance(a.base, torch.Tensor):
                out.append(dataclasses.replace(a, base=copy(a.base)))
            elif isinstance(a, _SCALAR_TYPES):
                out.append(a)
            else:
                store.dlog("consume", f"A/B skipped: cannot copy arg of type {type(a).__name__}")
                return None
    except OverflowError:
        store.dlog("consume", f"A/B skipped: arg copies exceed {limit >> 20} MB ({AB_COPY_LIMIT_ENV})")
        return None
    return out


def acf_args_for(ptx: str, arch: str | None, ptxas_version: str) -> list[str]:
    """['--apply-controls=<stored.acf>'] if the store has an ACF for this PTX and ptxas supports it
    (>=13.3), else []. Points ptxas straight at the stored ACF file -- no temp copy to leak."""
    try:
        from packaging.version import Version
        if not arch:
            return []
        # Minimum ptxas version that can apply ACFs. Defaults to 13.3 (the GA version that supports
        # --apply-controls). An ACF must be applied with a ptxas matching the version it was minted
        # with, since a mismatched ptxas rejects it ("Invalid compiler controls file").
        min_version = os.environ.get("COMPILE_IQ_PTXAS_MIN_VERSION", "13.3")
        if Version(ptxas_version) < Version(min_version):
            store.dlog(
                "consume", f"skip: ptxas {ptxas_version} < {min_version} -- cannot apply "
                "--apply-controls (set TRITON_PTXAS_BLACKWELL_PATH / COMPILE_IQ_PTXAS_MIN_VERSION)")
            return []
        sha = store.ptx_sha256(ptx)
        p = store.acf_path(sha, arch, ptxas_version)  # version-tagged: match the ptxas that minted the ACF
        if not os.path.exists(p):
            store.dlog("consume", f"MISS {sha[:16]} {arch} ptxas={ptxas_version}")
            return []
        store.dlog("consume", f"HIT {sha[:16]} {arch} ptxas={ptxas_version}")
        return [f"--apply-controls={p}"]
    except Exception as e:  # never break compilation
        store.dlog("consume", f"error (fail-open): {type(e).__name__}: {e}")
        return []
