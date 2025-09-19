import triton.language.core as tl

from . import types as tlx

# Blackwell-only

@tl.builtin
def alloc_clc_responses(
    num_responses: tl.constexpr,
    _semantic=None,
) -> tlx.clc_response:
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
        layout.vectorSize,
        layout.perPhase,
        layout.maxPhase,
        layout.order,
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
    )
    return tlx.clc_response(
        _semantic.builder.create_alloc_clc_responses(
            num_responses.value,
            layout_handle),
        num_responses,
        layout,
        _semantic)


@tl.builtin
def clc_issue(
    clc_response_addr: tlx.clc_response,
    barrier: tlx.mbarrier,
    _semantic=None,
):
    # Issue async `clusterlaunchcontrol.try_cancel` request for
    # CTA ID of available cluster
    return _semantic.builder.clc_issue(clc_response_addr.handle, barrier.handle)


@tl.builtin
def clc_query(
    clc_response_addr: tlx.clc_response,
    valid, cta_id_x, cta_id_y, cta_id_z,
    _semantic=None,
):
    # Extract CTA ID from CLC response
    # Return success/fail of PTX instruction
    return _semantic.builder.clc_query(
        clc_response_addr.handle,
        valid.handle,
        cta_id_x.handle,
        cta_id_y.handle,
        cta_id_z.handle,
    )
