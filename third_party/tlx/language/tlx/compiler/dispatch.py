import ast
import triton.tlx.language as tlx
from triton.tlx.compiler import visit_withAsyncTask, visit_withAsyncTasks

# Dispatch table
TLX_WITH_DISPATCH = {
    tlx.async_tasks: visit_withAsyncTasks,
    tlx.async_task: visit_withAsyncTask,
}
