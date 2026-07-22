"""Pre-import HSTU autoWS config hook (no triton import).

Lets callers/tests set the HSTU autoWS config as a dict *before* importing
``triton_hstu_attention`` -- replacing the old ``HSTU_SELF_*`` environment
variables. ``triton_hstu_attention`` merges ``pop_overrides()`` into its
``HSTUAutoWSConfig`` at import (autoWS structural flags + the autotune tile config
are read at import / decoration, so they must be set before the kernel module is
imported). After import, use ``triton_hstu_attention.configure_autows()`` or the
``autows_cfg=`` argument on the entrypoints instead.

    import hstu_autows_config as C
    C.set_config(autows=True, dp=1, dq_reduce=True, dq_reuse=True, dq_iters=4,
                 warps=4, bwd_bm=128, bwd_bn=128, bwd_stages=2, pin=True)
    import triton_hstu_attention as A   # picks up the config
"""

_OVERRIDES: dict = {}


def set_config(**kwargs) -> None:
    """Set HSTU autoWS config overrides (field names of HSTUAutoWSConfig)."""
    _OVERRIDES.update(kwargs)


def pop_overrides() -> dict:
    """Return (a copy of) the pending overrides for triton_hstu_attention import."""
    return dict(_OVERRIDES)
