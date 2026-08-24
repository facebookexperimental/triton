# TorchTLX inductor integration.
#
# These modules import torch._inductor internals and are loaded lazily by
# PyTorch (via torch/_inductor/template_heuristics/tlx.py) -- never by triton
# itself. Do not import this subpackage from the tlx __init__ chain, or a
# triton-only consumer would pull in torch.
#
# The hardware model (arch detection + on-chip memory limits) is NOT here: it
# lives in tlx/hw/, outside this subpackage, so the standalone tutorial kernels
# can share it without importing torch._inductor.
#
# Module layout:
#   registry.py   shape heuristics, Inductor heuristic classes, and the
#                 Inductor monkey-patch layer.
#   mm_templates.py / flex_attention_templates.py
#                 template objects plus the choice injection, each dispatching
#                 vendor-wise: append_tlx -> _append_tlx_{nvidia,amd} and
#                 append_tlx_flex -> _append_tlx_flex_{nvidia,amd}.
#   choices.py    TLXInductorChoices; fusion.py  force-fusion policy;
#   codegen.py    async-TMA store codegen; reduce_k.py  split-K reducer;
#   tlx_config.py env knobs.
