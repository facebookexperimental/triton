import triton.language.extra.tlx as tlx
from .code_generator import visit_withAsyncTask, visit_withAsyncTasks, visit_withWarpPipelineStage

# Dispatch table
TLX_WITH_DISPATCH = {
    tlx.async_tasks: visit_withAsyncTasks,
    tlx.async_task: visit_withAsyncTask,
    tlx.warp_pipeline_stage: visit_withWarpPipelineStage,
}


def register_gluon_warp_pipeline():
    from triton.experimental.gluon.language.amd import warp_pipeline_stage as gluon_wps
    TLX_WITH_DISPATCH[gluon_wps] = visit_withWarpPipelineStage
