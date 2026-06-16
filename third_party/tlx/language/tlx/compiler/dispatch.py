import triton.language.extra.tlx as tlx

from .code_generator import visit_withAsyncTask, visit_withAsyncTasks, visit_withWarpPipelineStage

TLX_WITH_DISPATCH = {
    tlx.async_tasks: visit_withAsyncTasks,
    tlx.async_task: visit_withAsyncTask,
    tlx.warp_pipeline_stage: visit_withWarpPipelineStage,
}
