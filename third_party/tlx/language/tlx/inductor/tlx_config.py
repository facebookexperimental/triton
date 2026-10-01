"""TLX configuration sub-knobs.

The primary enable/mode knob (``tlx_mode`` / ``TORCHINDUCTOR_TLX_MODE``) is an
OSS-visible Inductor config option (``torch._inductor.config.triton.tlx_mode``).
The knobs below stay here in the FB Triton fork alongside the TLX templates.

This module must stay importable without torch: inductor's compile workers
import it for every Triton kernel just to read option names.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator


#: Spellings ``getenv_bool`` (triton._C, see ``python/src/ir.cc`` ``is_truthy``)
#: treats as true. Mirrored here so the knob and this gate cannot disagree
#: (e.g. ``TRITON_ENABLE_TORCH_TLX=true`` must enable both).
_TRUTHY_ENV_VALUES = frozenset({"1", "y", "on", "yes", "true"})


def _env_is_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUTHY_ENV_VALUES


def torch_tlx_enabled() -> bool:
    """Whether the torchTLX inductor integration may load.

    True when the ``TRITON_ENABLE_TORCH_TLX`` master switch (see
    ``triton.knobs.tlx``) is on, or when the resolved TLX mode is ``allow`` /
    ``force``. The mode is read from the environment and -- without importing
    anything -- from an already-imported ``torch._inductor.config``, so
    ``config.patch`` and JustKnob/build-default flows keep working while a
    fresh worker that has not imported torch decides on env vars alone.
    """
    if _env_is_truthy("TRITON_ENABLE_TORCH_TLX"):
        return True
    if os.environ.get("TORCHINDUCTOR_TLX_MODE") in ("allow", "force"):
        return True
    config_mod = sys.modules.get("torch._inductor.config")
    if config_mod is not None:
        try:
            if config_mod.triton.tlx_mode in ("allow", "force"):
                return True
        except AttributeError:
            pass
    return False


_TLX_ONLY_CUDA_OPTIONS_ENABLED: tuple[str, ...] = ("ctas_per_cga",)

# Inductor's compile workers need these option names for every Triton kernel.
# Keep them outside the template registry so that lookup does not import all
# of torch._inductor's GEMM templates into each worker. Empty unless the
# integration is enabled; a single list object, mutated in place and never
# rebound -- torch caches the reference via lru_cache.
tlx_only_cuda_options: list[str] = (
    list(_TLX_ONLY_CUDA_OPTIONS_ENABLED) if torch_tlx_enabled() else []
)


def ensure_tlx_options_populated() -> None:
    """Restore the enabled option names after an early disabled import.

    This module may be first imported while disabled (e.g. a compile of a
    non-TLX kernel); when the integration loads later in the same process,
    the registry calls this to reconcile the shared list in place -- never
    rebound, since torch caches the reference via lru_cache -- so existing
    holders observe the update.
    """
    if list(tlx_only_cuda_options) != list(_TLX_ONLY_CUDA_OPTIONS_ENABLED):
        tlx_only_cuda_options[:] = _TLX_ONLY_CUDA_OPTIONS_ENABLED


# Use shape-based heuristic rules to pick a single GEMM config.
# When False, falls back to iterating base configs from mm_configs.
use_heuristic_config: bool = (
    os.environ.get("TORCHINDUCTOR_TLX_USE_HEURISTIC_CONFIG", "1") == "1"
)

# Use TMA (asynchronous descriptor) stores for the GEMM epilogue.
# Matches upstream TLX behavior; the tl.store path is incompatible with
# NUM_CTAS=2 (MultiCTAReduction pass can't distribute direct stores across CTAs).
tma_epilogue_store: bool = (
    os.environ.get("TORCHINDUCTOR_TLX_TMA_EPILOGUE_STORE", "1") == "1"
)

# When True, yield both TMA_EPILOGUE_STORE=0 and TMA_EPILOGUE_STORE=1 variants
# so autotuning can pick the faster path. When False, always prefer TMA=1 if it
# fits in SMEM and fall back to TMA=0 otherwise (single variant, no autotuning).
autotune_tma_epilogue_store: bool = (
    os.environ.get("TORCHINDUCTOR_TLX_AUTOTUNE_TMA_EPILOGUE_STORE", "0") == "1"
)

# In "allow" mode, minimum speedup TLX must achieve over the best extern
# kernel (cublas) to be selected. Below this threshold, the extern kernel wins.
allow_min_speedup: float = float(
    os.environ.get("TORCHINDUCTOR_TLX_ALLOW_MIN_SPEEDUP", "1.0")
)


@contextmanager
def patch(**kwargs: object) -> Generator[None, None, None]:
    """Context manager to temporarily override TLX config values."""
    import sys

    _self = sys.modules[__name__]

    saved = {k: getattr(_self, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(_self, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(_self, k, v)
