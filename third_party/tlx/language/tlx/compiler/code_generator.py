# third_party/tlx/codegen/async.py

import ast
from typing import List
import triton.language.extra.tlx as tlx  # Make sure async_task(s) are exposed via tlx.__init__.py
from contextlib import contextmanager

# TLX allows users to specify the replicate number when definine
# a non-default partition region. We use a stack to keep track of
# replica_id of the region being compiled.
region_replica_id_stack: List[int] = []


@contextmanager
def tlx_enter_sub_region():
    global region_replica_id_stack
    replica_id_stack_backup = region_replica_id_stack.copy()
    try:
        yield
    finally:
        assert region_replica_id_stack == replica_id_stack_backup, "region_replica_id_stack is not restored"


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
    # Visit the body of the `with` region
    self.visit_compound_statement(node.body)


@tlx_enter_sub_region()
def visit_withAsyncTasks(self, node):
    from triton.compiler.code_generator import (enter_sub_region, _is_list_like, _is_constexpr)
    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()
        stmts = node.body
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]

        # dry visit async task body to count the number of sub tasks
        with tlx_enter_sub_region():
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)
            taskNumWarps = []
            taskNumRegs = []
            taskReplica = []

            global region_replica_id_stack
            region_replica_id_stack.append(-1)  # dummy placeholder

            num_default = 0
            for stmt in stmts:
                assert _is_async_task(self, stmt)
                task = _get_async_task(self, stmt)
                assert task.is_explict
                if task.is_default:
                    num_default += 1
                else:
                    assert task.replicate is not None, "Replicate must be non-None for non-default task"
                    taskReplica.append(task.replicate)
                    taskNumWarps.extend([task.num_warps] * task.replicate)
                    if task.num_regs:
                        taskNumRegs.extend([task.num_regs] * task.replicate)

            region_replica_id_stack.pop()  # revert adding dummy placeholder

        assert num_default == 1, "Default task must be one and only one"
        block.erase()

        assert len(taskNumRegs) in [0,
                                    len(taskNumWarps)], "Registers are set for either ALL or NONE of non-default tasks"

        # Create tasks body block
        self._set_insertion_point_and_loc(ip, last_loc)
        ws_op = self.builder.create_warp_specialize_op(taskNumWarps, taskNumRegs if len(taskNumRegs) > 0 else None,
                                                       sum(taskReplica))

        # dry visit async task body to calculate captures
        index = 0
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            assert task.is_explict
            if task.is_default:
                task_body = ws_op.get_default_region()
                block = self.builder.create_block_with_parent(task_body, [])
            else:
                # Only need to calculate captures for the first replica.
                region_replica_id_stack.append(0)
                task_body = ws_op.get_partition_region(index)
                index += task.replicate
                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                region_replica_id_stack.pop()

            block.erase()

        # Add captures
        captures = sorted(v for v in (liveins.keys() & self.used_vars) if not _is_constexpr(liveins[v]))
        for name in captures:
            val = liveins[name]
            ws_op.append_operand(val.handle)


        # real codegen
        index = 0
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            if task.is_default:
                task_body = ws_op.get_default_region()

                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)

                self.builder.create_warp_yield_op()
            else:
                for i in range(task.replicate):
                    region_replica_id_stack.append(i)

                    task_body = ws_op.get_partition_region(index)
                    index += 1

                    block = self.builder.create_block_with_parent(task_body, [])
                    self.builder.set_insertion_point_to_start(block)
                    with enter_sub_region(self):
                        self.visit(stmt)

                    for name in captures:
                        val = liveins[name]
                        arg = task_body.add_argument(val.handle.get_type())
                        block.replace_use_in_block_with(val.handle, arg)

                    self.builder.create_warp_return_op()
                    region_replica_id_stack.pop()
