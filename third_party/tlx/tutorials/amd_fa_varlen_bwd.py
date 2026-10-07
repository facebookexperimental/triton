"""Compatibility entry points for the promoted gfx950 variable-length backward kernels.

The implementation now lives in
``triton.tlx.ops.kernels.flash_attn.gfx950_varlen_bwd``. New callers should
import from that module; these names remain for existing tutorial users.
"""

from triton.tlx.ops.kernels.flash_attn.gfx950_varlen_bwd import (
    VarlenBackwardPlan,
    fa_varlen_backward,
    prepare_varlen_backward,
    validate_varlen_backward_plan,
)

__all__ = [
    "VarlenBackwardPlan",
    "fa_varlen_backward",
    "prepare_varlen_backward",
    "validate_varlen_backward_plan",
]
