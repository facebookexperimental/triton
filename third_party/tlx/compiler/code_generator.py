# third_party/tlx/codegen/async.py

import ast
import triton.tlx.language as tlx  # Make sure async_task(s) are exposed via tlx.__init__.py


def _is_async_task(self, node) -> bool:
    if isinstance(node, ast.With):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call):
            withitemClass = self.visit(context.func)
            if withitemClass == tlx.async_task:
                return True
    return False


def _get_async_task(self, node):
    context = node.items[0].context_expr
    # Parse positional args (e.g., [0])
    args = [self.visit(arg) for arg in context.args]
    # Extract keyword arguments as (key, value AST nodes)
    kwargs = {kw.arg: self.visit(kw.value) for kw in context.keywords}
    with tlx.async_task(*args, _builder=self.builder, **kwargs) as task:
        return task


def visit_withAsyncTask(self, node):
    task = _get_async_task(self, node)
    # Visit the body of the `with` region
    self.visit_compound_statement(node.body)


def visit_withAsyncTasks(self, node):
    from triton.compiler.code_generator import enter_sub_region, _is_list_like
  # ⏳ defer import
    with enter_sub_region(self) as sr:
        stmts = node.body
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]

        # Count the number of sub tasks
        taskNumWarps = []
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            assert task.is_explict
            if not task.is_default:
                taskNumWarps.append(task.num_warps)

        # create tasks body block
        ws_op = self.builder.create_warp_specialize_op(taskNumWarps, len(stmts) - 1)
        index = 1
        has_default = False
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            if task.is_default:
                task_body = ws_op.get_default_region()
                has_default = True
            else:
                task_body = ws_op.get_partition_region(index-1)
                index += 1
            partitionBlock = self.builder.create_block_with_parent(task_body, [])
            self.builder.set_insertion_point_to_start(partitionBlock)
            self.visit(stmt)
            if task.is_default:
                self.builder.create_warp_yield_op()
            else:
                self.builder.create_warp_return_op()

        if not has_default:
            task_body = ws_op.get_region(0)
            partitionBlock = self.builder.create_block_with_parent(task_body, [])

        # Capture live-ins
        # liveins
        # wsOp->insertOperands(wsOp.getNumOperands(), capture);
        # replace_all_uses_with
        # replaceAllUsesInRegionWith
