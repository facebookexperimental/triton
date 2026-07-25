# TorchTLX inductor integration.
#
# These modules import torch._inductor internals and are loaded lazily by
# PyTorch (via torch/_inductor/template_heuristics/tlx.py) -- never by triton
# itself. Do not import this subpackage from the tlx __init__ chain, or a
# triton-only consumer would pull in torch.
