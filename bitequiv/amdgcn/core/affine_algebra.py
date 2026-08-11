# The symbolic integer algebra (Affine / Opaque lattice, canonical serialization, range/bit-mask
# helpers) is architecture-NEUTRAL, so the AMD checker REUSES the shared
# `bitequiv.core.affine_algebra` (extracted on the NV side, D114729647) instead of hard-copying it.
# The per-backend AffineEval / thread_image / _bit_basis stay ISA-specific in bitequiv/amdgcn/affine.py.
from bitequiv.core.affine_algebra import (  # noqa: F401
    _INF_TZ,
    _add_terms,
    _const,
    _ctz,
    _is_pow2,
    _parse_int,
    _rng_add,
    _rng_scale,
    _set_bits_mask,
    _symbol,
    Affine,
    canon,
    Opaque,
)
