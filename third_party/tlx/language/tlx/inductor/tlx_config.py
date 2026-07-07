"""TLX configuration sub-knobs.

The primary enable/mode knob (``tlx_mode`` / ``TORCHINDUCTOR_TLX_MODE``) is an
OSS-visible Inductor config option (``torch._inductor.config.triton.tlx_mode``).
The knobs below stay here in the FB Triton fork alongside the TLX templates.
"""

import os
from contextlib import contextmanager
from typing import Generator


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
