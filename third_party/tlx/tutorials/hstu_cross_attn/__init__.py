# HSTU cross-attention kernels (fwd + bwd, incl. TRITON_AUTOWS variant), adapted
# from fbcode hammer for standalone OSS use. Kernels live in the triton repo so
# tritonbench can import them without buck/hammer.
from .triton_bw_cross_attention import (  # noqa: F401
    BwdVariant,
    set_bwd_variant,
    get_bwd_variant,
    triton_bw_hstu_mha,
    triton_bw_hstu_mha_wrapper,
)
