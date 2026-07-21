# third_party/tlx/codegen/async.py

import ast
import threading
from contextlib import contextmanager
from typing import List

import triton.language.extra.tlx as tlx  # Make sure async_task(s) are exposed via tlx.__init__.py

# TLX allows users to specify the replicate number when defining
# a non-default partition region. We use a stack to keep track of
# replica_id of the region being compiled.
#
# Thread-local storage for TLX compiler state
# This allows parallel compilation of TLX templates without race conditions
_tlx_state = threading.local()


def _get_region_replica_id_stack() -> List[int]:
    """Get the thread-local region_replica_id_stack, initializing if needed."""
    if not hasattr(_tlx_state, 'region_replica_id_stack'):
        _tlx_state.region_replica_id_stack = []
    return _tlx_state.region_replica_id_stack


def _get_sub_region_has_exception() -> bool:
    """Get the thread-local sub_region_has_exception flag."""
    if not hasattr(_tlx_state, 'sub_region_has_exception'):
        _tlx_state.sub_region_has_exception = False
    return _tlx_state.sub_region_has_exception


def _set_sub_region_has_exception(value: bool) -> None:
    """Set the thread-local sub_region_has_exception flag."""
    _tlx_state.sub_region_has_exception = value


@contextmanager
def tlx_enter_sub_region():
    region_replica_id_stack = _get_region_replica_id_stack()
    replica_id_stack_backup = region_replica_id_stack.copy()
    try:
        _set_sub_region_has_exception(False)
        yield
    except Exception as e:
        _set_sub_region_has_exception(True)
        raise e
    finally:
        if not _get_sub_region_has_exception():
            current_stack = _get_region_replica_id_stack()
            assert current_stack == replica_id_stack_backup, "region_replica_id_stack is not restored"


def _is_async_task(self, node) -> bool:
    if isinstance(node, ast.With):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call):
            withitem_class = self.visit(context.func)
            if withitem_class == tlx.async_task:
                return True
    return False


def _resolve_async_task_stmts(self, stmts):
    """Resolve constexpr if-guards around async_task statements.

    Statements inside async_tasks() must be either:
      - `with tlx.async_task(...)` (passed through directly), or
      - `if CONSTEXPR:` guarding one or more `with tlx.async_task(...)`.

    For constexpr if-guards, the condition is evaluated at compile time and
    only the active branch's async_task statements are included.
    """
    from triton.language.core import _unwrap_if_constexpr

    resolved = []
    for stmt in stmts:
        if _is_async_task(self, stmt):
            resolved.append(stmt)
        elif isinstance(stmt, ast.If):
            cond = self.visit(stmt.test)
            cond = _unwrap_if_constexpr(cond)
            active_block = stmt.body if cond else stmt.orelse
            for inner_stmt in active_block:
                assert _is_async_task(self, inner_stmt), ("Statements inside a constexpr if-guard within async_tasks() "
                                                          "must be `with tlx.async_task(...)` blocks")
                resolved.append(inner_stmt)
        else:
            assert False, ("Statements inside async_tasks() must be `with tlx.async_task(...)` "
                           "blocks or constexpr if-guards around them")
    return resolved


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


def _validate_warp_group_start_ids(
    start_ids: List[int],
    num_warps: List[int],
    task_replicates: List[int],
    default_num_warps: int,
) -> None:
    """Validate that warp group start IDs are valid and non-overlapping across different tasks.

    Args:
        start_ids: List of warp group start IDs for each task (before replica expansion).
        num_warps: List of number of warps for each task (before replica expansion).
        task_replicates: List of replica counts for each task.
        default_num_warps: Number of warps used by the default region (starts at warp 0).

    Raises:
        AssertionError: If validation fails.
    """
    assert len(start_ids) == len(num_warps) == len(task_replicates), (
        f"start_ids length ({len(start_ids)}), num_warps length ({len(num_warps)}), "
        f"and task_replicates length ({len(task_replicates)}) must all match")

    # Check that all start IDs are non-negative
    for i, start_id in enumerate(start_ids):
        assert start_id >= 0, f"warp_group_start_id[{i}] = {start_id} must be non-negative"

    # Check for overlapping warp ranges between different tasks
    # Build list of (start, end) ranges for each task, considering replicas
    # Each task uses num_warps * replicate warps starting at start_id
    ranges = [(start_ids[i], start_ids[i] + num_warps[i] * task_replicates[i]) for i in range(len(start_ids))]

    # Default region uses warps [0, default_num_warps)
    default_range = (0, default_num_warps)

    # Check that no non-default task overlaps with the default region
    for i, (start_i, end_i) in enumerate(ranges):
        # Two ranges [a, b) and [c, d) overlap if a < d and c < b
        if start_i < default_range[1] and default_range[0] < end_i:
            assert False, (f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                           f"which overlaps with default region warps [{default_range[0]}, {default_range[1]})")

    # Check all pairs of non-default tasks for overlap
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            start_i, end_i = ranges[i]
            start_j, end_j = ranges[j]
            # Two ranges [a, b) and [c, d) overlap if a < d and c < b
            if start_i < end_j and start_j < end_i:
                assert False, (f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                               f"and task {j} uses warps [{start_j}, {end_j})")


@tlx_enter_sub_region()
def visit_withAsyncTasks(self, node):
    from triton.compiler.code_generator import enter_sub_region, _is_list_like, _is_constexpr
    from triton.language.core import _unwrap_if_constexpr

    # Mark the warp_specialize op so the Fixup pass can propagate async-task
    # policy to module attributes consumed by later lowering passes.
    ws_context = node.items[0].context_expr
    exclusive = False
    no_ending_cluster_sync = False
    mbarrier_try_wait_suspend_ns = None
    for kw in getattr(ws_context, "keywords", []):
        if kw.arg == "exclusive":
            exclusive = bool(_unwrap_if_constexpr(self.visit(kw.value)))
        elif kw.arg == "no_ending_cluster_sync":
            no_ending_cluster_sync = bool(_unwrap_if_constexpr(self.visit(kw.value)))
        elif kw.arg == "mbarrier_try_wait_suspend_ns":
            mbarrier_try_wait_suspend_ns = _unwrap_if_constexpr(self.visit(kw.value))
            if not isinstance(mbarrier_try_wait_suspend_ns, int) or mbarrier_try_wait_suspend_ns < 0:
                raise ValueError("mbarrier_try_wait_suspend_ns must be a non-negative integer")

    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()

        # Get thread-local region_replica_id_stack for this compilation
        region_replica_id_stack = _get_region_replica_id_stack()

        def _flatten_value_handles(val):
            handles = []
            # Prefer the generic flatten hook to support multi-result values (e.g. tensor descriptors)
            if hasattr(val, "_flatten_ir"):
                val._flatten_ir(handles)
            else:
                handles.append(val.handle)
            return handles

        stmts = node.body
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]

        # Resolve constexpr if-guards so that only async_task statements remain
        stmts = _resolve_async_task_stmts(self, stmts)

        # Check if only the default task remains after constexpr resolution.
        # If so, skip warp specialization entirely and emit the default task inline.
        has_non_default = False
        for stmt in stmts:
            task_check = _get_async_task(self, stmt)
            if not task_check.is_default:
                has_non_default = True
                break

        if not has_non_default:
            for stmt in stmts:
                self.visit(stmt)
            return

        # dry visit async task body to count the number of sub tasks
        with tlx_enter_sub_region():
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)
            task_num_warps = []
            task_num_regs = []
            task_replicas = []
            task_warp_group_start_ids = []

            # Per-task data for validation (before replica expansion)
            per_task_num_warps = []
            per_task_start_ids = []
            per_task_replicates = []

            region_replica_id_stack.append(-1)  # dummy placeholder

            num_default = 0
            for stmt in stmts:
                task = _get_async_task(self, stmt)
                assert task.is_explict
                assert task.replicate is not None, "Replicate must be non-None task"
                if task.is_default:
                    num_default += 1
                    if task.replicate > 1:
                        task_replicas.append(task.replicate - 1)
                        task_num_warps.extend([self.builder.options.num_warps] * (task.replicate - 1))
                        if task.num_regs:
                            task_num_regs.extend([task.num_regs] * (task.replicate - 1))
                        if task.warp_group_start_id is not None:
                            task_warp_group_start_ids.extend([task.warp_group_start_id] * (task.replicate - 1))
                else:
                    task_replicas.append(task.replicate)
                    task_num_warps.extend([task.num_warps] * task.replicate)
                    if task.num_regs:
                        task_num_regs.extend([task.num_regs] * task.replicate)
                    if task.warp_group_start_id is not None:
                        # Each replica gets its own start ID, incrementing by num_warps
                        for r in range(task.replicate):
                            task_warp_group_start_ids.append(task.warp_group_start_id + r * task.num_warps)
                        # Collect per-task data for validation
                        per_task_num_warps.append(task.num_warps)
                        per_task_start_ids.append(task.warp_group_start_id)
                        per_task_replicates.append(task.replicate)

            region_replica_id_stack.pop()  # revert adding dummy placeholder

        assert num_default == 1, "Default task must be one and only one"
        block.erase()

        assert len(task_num_regs) in [0, len(task_num_warps)
                                      ], ("Registers are set for either ALL or NONE of non-default tasks")
        assert len(task_warp_group_start_ids) in [
            0, len(task_num_warps)
        ], ("warp_group_start_id must be set for either ALL or NONE of non-default tasks")

        # Validate warp_group_start_ids
        if per_task_start_ids:
            _validate_warp_group_start_ids(
                per_task_start_ids,
                per_task_num_warps,
                per_task_replicates,
                self.builder.options.num_warps,
            )

        # Create tasks body block
        self._set_insertion_point_and_loc(ip, last_loc)
        ws_op = self.builder.create_warp_specialize_op(
            task_num_warps,
            task_num_regs if task_num_regs else None,
            sum(task_replicas),
            task_warp_group_start_ids if task_warp_group_start_ids else None,
        )
        if exclusive:
            ws_op.set_attr("tlx.exclusive", self.builder.get_unit_attr())
        if no_ending_cluster_sync:
            ws_op.set_attr("tlx.no_ending_cluster_sync", self.builder.get_unit_attr())
        if mbarrier_try_wait_suspend_ns is not None:
            ws_op.set_attr(
                "tlx.mbarrier_try_wait_suspend_ns",
                self.builder.get_int32_attr(mbarrier_try_wait_suspend_ns),
            )

        # dry visit async task body to calculate captures
        index = 0
        for stmt in stmts:
            task = _get_async_task(self, stmt)
            assert task.is_explict
            task_replicate = (task.replicate - 1) if task.is_default else task.replicate
            if task_replicate > 0:
                task_body = ws_op.get_partition_region(index)
                block = self.builder.create_block_with_parent(task_body, [])
                # Only need to calculate captures for the first replica.
                region_replica_id_stack.append(0)
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                region_replica_id_stack.pop()
                index += task_replicate
                block.erase()

        # Add captures to the partitions op (which owns explicitCaptures
        # after the upstream refactor in PR #9133).
        partition_op = ws_op.get_partition_op()
        captures = sorted(v for v in (liveins.keys() & self.used_vars) if not _is_constexpr(liveins[v]))
        for name in captures:
            val = liveins[name]
            if getattr(val, "__triton_aggregate__", False):
                for field in val.type.fields:
                    v = getattr(val, field[0])
                    for h in _flatten_value_handles(v):
                        partition_op.append_operand(h)
            else:
                for h in _flatten_value_handles(val):
                    partition_op.append_operand(h)

        # real codegen
        index = 0
        for stmt in stmts:
            task = _get_async_task(self, stmt)
            if task.is_default:
                region_replica_id_stack.append(0)
                task_body = ws_op.get_default_region()

                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)

                self.builder.create_warp_yield_op()
                region_replica_id_stack.pop()

            replicate_start = 1 if task.is_default else 0

            for i in range(replicate_start, task.replicate):
                region_replica_id_stack.append(i)

                task_body = ws_op.get_partition_region(index)
                index += 1

                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)

                for name in captures:
                    val = liveins[name]
                    if getattr(val, "__triton_aggregate__", False):
                        for field in val.type.fields:
                            v = getattr(val, field[0])
                            for h in _flatten_value_handles(v):
                                arg = task_body.add_argument(h.get_type())
                                block.replace_use_in_block_with(h, arg)
                    else:
                        for h in _flatten_value_handles(val):
                            arg = task_body.add_argument(h.get_type())
                            block.replace_use_in_block_with(h, arg)

                self.builder.create_warp_return_op()
                region_replica_id_stack.pop()


def visit_withWarpPipelineStage(self, node):
    from triton.language.core import _unwrap_if_constexpr

    # Reject on non-AMD targets early with a clear error.
    target_str = getattr(self.builder.options, 'target', '')
    if isinstance(target_str, str) and target_str and not target_str.startswith('hip'):
        raise ValueError("tlx.warp_pipeline_stage is only supported on AMD (HIP) targets")

    context = node.items[0].context_expr
    args = [self.visit(arg) for arg in context.args]
    kwargs = {kw.arg: self.visit(kw.value) for kw in context.keywords}

    label = _unwrap_if_constexpr(args[0]) if args else None
    if label is None:
        label = "cluster"
    priority = _unwrap_if_constexpr(kwargs.get("priority", None))
    if priority is None:
        priority = -1

    self.visit_compound_statement(node.body)
    self.builder.create_warp_pipeline_border(str(label), priority)
