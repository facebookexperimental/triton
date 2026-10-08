"""
TLX Template Fusion Support

This module provides functions to control epilogue fusion behavior for TLX templates
during prototyping.

Usage:
    Set environment variable TORCHINDUCTOR_TLX_MODE=force to force
    TLX templates to fuse with epilogues regardless of benchmark results.
"""

import logging
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.scheduler import BaseSchedulerNode


fusion_log = logging.getLogger(__name__)


def _is_force_fusion_enabled() -> bool:
    """Check if force fusion is enabled via tlx_mode config."""
    from torch._inductor import config

    return config.triton.tlx_mode == "force"


def _is_tlx_choice(choice: Any) -> bool:
    """Check if a choice is a TLX template by name pattern."""
    if hasattr(choice, "name") and isinstance(choice.name, str):
        return "tlx_" in choice.name
    return False


def _uses_tma_epilogue_store(annotations: dict[str, Any]) -> bool:
    """Whether a choice or template buffer, by its KernelTemplateChoice
    annotation, is a TLX template storing its epilogue through TMA."""
    ktc = annotations.get("ktc")
    if ktc is None or not getattr(ktc.template, "name", "").startswith("tlx_"):
        return False
    return bool(ktc.params.to_kwargs().get("TMA_EPILOGUE_STORE", 0))


def tma_epilogue_store_fusion_unsupported(
    node1: "BaseSchedulerNode", node2: "BaseSchedulerNode"
) -> bool:
    """Whether node2 can't be an epilogue of a TLX template that may store it
    through TMA_EPILOGUE_STORE. That path stores exactly one buffer, the last
    epilogue result, through a row-major [M, N] TMA descriptor whose staging
    tile is budgeted at the template's dtype."""
    from torch._inductor import ir
    from torch._inductor.scheduler import OutputNode
    from torch._inductor.virtualized import V

    if not node1.is_template():
        return False
    template = node1.get_template_node()
    caller = getattr(template, "_render_caller", None)
    if caller is not None:
        candidates = [caller]
    elif isinstance(template, ir.MultiTemplateBuffer):
        candidates = template.choices
    else:
        candidates = [template]
    if not any(_uses_tma_epilogue_store(c.annotations) for c in candidates):
        return False

    fused = OrderedSet(node1.get_nodes()) | OrderedSet(node2.get_nodes())
    live = [
        buf
        for buf in (*node1.get_outputs(), *node2.get_outputs())
        if any(
            isinstance(use.node, OutputNode) or use.node not in fused
            for use in buf.users
        )
    ]
    if len(live) != 1 or live[0].get_name() == template.get_name():
        return True
    layout = live[0].node.get_layout()
    offset = (
        layout.view.get_layout().offset
        if isinstance(layout, ir.NonOwningLayout)
        else layout.offset
    )
    itemsize = layout.dtype.itemsize
    aligned = V.graph.sizevars.statically_known_multiple_of
    return (
        list(layout.size) != list(template.get_size())
        or layout.stride[-1] != 1
        or layout.dtype == torch.bool
        or not aligned(layout.stride[0] * itemsize, 16)
        or not aligned(offset * itemsize, 16)
        or itemsize > template.get_dtype().itemsize
    )


def _has_split_k(multi_node: "ir.MultiTemplateBuffer") -> bool:
    """Check if any TLX choice in the node uses SPLIT_K > 1.

    Since TritonTemplateCaller doesn't expose template kwargs directly,
    we check via the description string which includes SPLIT_K info,
    or infer from the heuristic config for the node's shape.
    """
    if not any(_is_tlx_choice(c) for c in multi_node.choices):
        return False
    try:
        from torch._inductor.utils import get_num_sms

        from . import tlx_config
        from .registry import get_heuristic_config

        layout = multi_node.get_layout()
        sizes = [int(s) for s in layout.size]
        if len(sizes) == 2:
            m, n = sizes
            # Infer K from input shapes
            inputs = multi_node.inputs
            if inputs and hasattr(inputs[0], "get_size"):
                k = int(inputs[0].get_size()[-1])
                if tlx_config.use_heuristic_config:
                    cfg = get_heuristic_config(m, n, k, get_num_sms())
                    if cfg and cfg.get("SPLIT_K", 1) > 1:
                        return True
    except Exception:
        pass
    return False


def should_force_fusion(multi_node: "ir.MultiTemplateBuffer") -> bool:
    """
    Check if fusion should be forced for a MultiTemplateBuffer.

    Returns True if TORCHINDUCTOR_TLX_MODE=force and the node
    contains TLX templates.  Never force fusion when SPLIT_K > 1
    because epilogue ops cannot be applied to partial results.
    """
    if not _is_force_fusion_enabled():
        return False
    if _has_split_k(multi_node):
        return False
    return any(_is_tlx_choice(c) for c in multi_node.choices)


def should_force_fusion_for_node(node: "BaseSchedulerNode") -> bool:
    """
    Check if fusion should be forced for a scheduler node.

    Returns True if TORCHINDUCTOR_TLX_MODE=force and the node
    is a TLX template (or any Triton template when we can't determine).
    """
    from torch._inductor import ir

    if not _is_force_fusion_enabled():
        return False
    if not node.is_template():
        return False
    template_node = node.get_template_node()
    if template_node is None:
        return False
    if isinstance(template_node, ir.MultiTemplateBuffer):
        if _has_split_k(template_node):
            return False
        return any(_is_tlx_choice(c) for c in template_node.choices)
    return False


def log_fusion_forced(
    ms_fused: float,
    ms1: float,
    ms2: float,
    path: str = "",
) -> None:
    """Log when fusion is forced despite benchmark results."""
    ms_sum = ms1 + ms2
    if ms_fused >= ms_sum:
        fusion_log.info(
            "TLX fusion forced%s despite slowdown: ms_fused=%.6f >= ms1+ms2=%.6f (%.2fx slower)",
            f" ({path})" if path else "",
            ms_fused,
            ms_sum,
            ms_fused / ms_sum,
        )
    else:
        fusion_log.info(
            "TLX fusion forced%s (also faster): ms_fused=%.6f < ms1+ms2=%.6f (%.2fx speedup)",
            f" ({path})" if path else "",
            ms_fused,
            ms_sum,
            ms_sum / ms_fused,
        )


def maybe_add_tlx_prefix(fused_name: str, src_code: str) -> str:
    """
    If the rendered kernel source uses TLX APIs, insert 'tlx' into
    the fused kernel name (e.g. 'fused_mm' -> 'fused_tlx_mm').
    """
    if "tlx." in src_code:
        return fused_name.replace("fused_", "fused_tlx_", 1)
    return fused_name
