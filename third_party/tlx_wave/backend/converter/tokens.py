"""Token and memory-effect graph for the TLX Wave converter."""

from dataclasses import dataclass, replace

from .diagnostics import fail

STAGE = "tokens"

_ASYNC_COPY_OPS = frozenset({"ttg.async_copy_global_to_local", "amdg.buffer_load_to_local"})
_TOKEN_CONTROL_OPS = frozenset({"ttg.async_commit_group", "ttg.async_wait"})
_TOKEN_OPS = _ASYNC_COPY_OPS | _TOKEN_CONTROL_OPS
_MEMORY_OPS = _ASYNC_COPY_OPS | frozenset({
    "amdg.buffer_load",
    "amdg.buffer_store",
    "tt.load",
    "tt.store",
    "ttg.local_load",
    "ttg.local_store",
})


@dataclass(frozen=True)
class TokenNode:
    node_id: int
    op_index: int
    op_name: str
    value_id: int | None
    input_token_ids: tuple[int, ...]
    source_address_value_id: int | None = None
    source_offset_value_id: int | None = None
    memdesc_value_id: int | None = None
    mask_value_id: int | None = None
    other_value_id: int | None = None
    wait_group: int | None = None
    committed_group_id: int | None = None
    waited_group_ids: tuple[int, ...] = ()
    retained_group_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenGroup:
    group_id: int
    commit_op_index: int
    token_value_id: int | None
    member_token_ids: tuple[int, ...]
    next_same_region_wait_op_index: int | None = None


@dataclass(frozen=True)
class LoopTokenCarry:
    loop_op_index: int
    init_source_value_id: int | None
    yield_source_value_id: int | None
    add_issue_dependency: bool
    issue_dependency_op_indices: tuple[int, ...] = ()
    readiness_carry: bool = False


@dataclass(frozen=True)
class IfTokenCarry:
    if_op_index: int
    then_source_value_id: int | None
    else_source_value_id: int | None


@dataclass(frozen=True)
class MemoryEffect:
    effect_id: int
    op_index: int
    op_name: str
    kind: str
    address_space: str
    address_value_id: int | None = None
    offset_value_id: int | None = None
    value_value_id: int | None = None
    mask_value_id: int | None = None
    token_node_id: int | None = None
    cache_modifier: str | None = None
    volatile: bool = False
    ordering: str | None = None
    sync_scope: str | None = None
    alias_class: str = "unknown"
    depends_on_effect_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenProgram:
    nodes: tuple[TokenNode, ...]
    groups: tuple[TokenGroup, ...]
    memory_effects: tuple[MemoryEffect, ...]
    node_ids_by_value_id: dict[int, int]
    users_by_value_id: dict[int, tuple[int, ...]]
    loop_token_carries_by_op: dict[int, tuple[LoopTokenCarry, ...]]
    if_token_carries_by_op: dict[int, tuple[IfTokenCarry, ...]]
    async_protocol_dependency_value_ids_by_op: dict[int, tuple[int, ...]]

    def node_for_value(self, value_id):
        node_id = self.node_ids_by_value_id.get(value_id)
        return None if node_id is None else self.nodes[node_id]

    def users_for_value(self, value_id):
        return tuple(self.nodes[node_id] for node_id in self.users_by_value_id.get(value_id, ()))


def build_token_program(source_program, type_layout_program):
    del type_layout_program
    nodes = []
    groups = []
    memory_effects = []
    node_ids_by_value = {}
    users_by_value = {}
    open_async_tokens = []
    committed_groups = []
    dependency_frontier = _DependencyFrontier()

    for op in source_program.ops:
        token_node_id = None
        if _needs_token_node(source_program, op):
            node, group, open_async_tokens, committed_groups = _build_token_node(
                source_program,
                op,
                len(nodes),
                len(groups),
                tuple(open_async_tokens),
                tuple(committed_groups),
            )
            nodes.append(node)
            token_node_id = node.node_id
            if node.value_id is not None:
                node_ids_by_value[node.value_id] = node.node_id
            if group is not None:
                groups.append(group)
            for input_token_id in node.input_token_ids:
                _append_user(users_by_value, input_token_id, node.node_id)
            if group is not None:
                for member_token_id in group.member_token_ids:
                    _append_user(users_by_value, member_token_id, node.node_id)
            if op.name == "ttg.async_wait":
                for group_id in (*node.waited_group_ids, *node.retained_group_ids):
                    waited_group = groups[group_id]
                    if waited_group.token_value_id is not None:
                        _append_user(
                            users_by_value,
                            waited_group.token_value_id,
                            node.node_id,
                        )
                    for member_token_id in waited_group.member_token_ids:
                        _append_user(users_by_value, member_token_id, node.node_id)

        if op.name in _MEMORY_OPS:
            memory_effects.extend(
                _memory_effects_for_op(
                    source_program,
                    op,
                    token_node_id,
                    len(memory_effects),
                    dependency_frontier,
                ))

    nodes = tuple(nodes)
    next_wait_by_commit = _next_same_region_wait_by_commit(source_program)
    groups = tuple(
        replace(
            group,
            next_same_region_wait_op_index=next_wait_by_commit.get(
                int(group.commit_op_index)
            ),
        )
        for group in groups
    )
    loop_token_carries_by_op = _loop_token_carries_by_op(
        source_program,
        nodes,
        groups,
    )
    (
        async_protocol_dependency_value_ids_by_op,
        readiness_loop_carries_by_op,
        readiness_if_carries_by_op,
    ) = _async_protocol_dependencies_by_op(source_program)
    loop_token_carries_by_op = _merge_loop_token_carries(
        loop_token_carries_by_op,
        readiness_loop_carries_by_op,
    )
    if_token_carries_by_op = _merge_if_token_carries(
        _if_token_carries_by_op(
            source_program,
            nodes,
            groups,
            loop_token_carries_by_op,
        ),
        readiness_if_carries_by_op,
    )
    return TokenProgram(
        nodes,
        groups,
        tuple(memory_effects),
        node_ids_by_value,
        {value_id: tuple(node_ids)
         for value_id, node_ids in users_by_value.items()},
        loop_token_carries_by_op,
        if_token_carries_by_op,
        async_protocol_dependency_value_ids_by_op,
    )


def _async_protocol_dependencies_by_op(source_program):
    """Thread explicit waits through their dominated local-memory operations.

    The wait result is the structural source-level proof that earlier async
    copies are ready.  Record it as an explicit operand of every dominated DS
    operation.  When the next explicit wait is reached, also record the prior
    wait as a release dependency if that epoch contained DS operations.  This
    gives the bridge the complete wait -> DS -> next-wait token chain without
    consulting DMA destinations or LDS alias state.

    The load's AMD readiness annotation separately decides whether the emitter
    may ignore ordinary LDS access state; domination alone must not turn a
    synchronous local-store/load path into a relaxed DMA consumer.

    Async DMA issue is not a consumer of this relation and must never acquire
    a destination-based or inferred dependency here.
    """
    dependencies = {}
    loop_carries = {}
    if_carries = {}

    def visit_region(region_id, inherited_wait_value_ids=()):
        wait_value_ids = tuple(int(value_id) for value_id in inherited_wait_value_ids)
        used_wait_value_ids = set()
        pending_release_value_ids = set()
        region = source_program.regions[int(region_id)]
        for op_index in region.op_indices:
            op = source_program.ops[int(op_index)]
            if op.name == "ttg.async_wait":
                release_value_ids = tuple(
                    value_id
                    for value_id in wait_value_ids
                    if value_id in pending_release_value_ids
                )
                if release_value_ids:
                    dependencies[int(op.index)] = release_value_ids
                    pending_release_value_ids.difference_update(release_value_ids)
                wait_result = _first_token_result(source_program, op)
                wait_value_ids = (() if wait_result is None else (int(wait_result), ))
            elif wait_value_ids and op.name in {"ttg.local_load", "ttg.local_store"}:
                dependencies[int(op.index)] = wait_value_ids
                used_wait_value_ids.update(wait_value_ids)
                pending_release_value_ids.update(wait_value_ids)
            if op.name == "scf.for" and len(op.region_ids) == 1:
                (
                    body_wait_value_ids,
                    body_used_wait_value_ids,
                    body_pending_release_value_ids,
                ) = visit_region(
                    op.region_ids[0],
                    wait_value_ids,
                )
                # A DS operation before the body's wait consumes the preheader
                # wait on the first iteration and the body wait later.
                # Represent that recurrence with a hidden token iter_arg; it
                # is a DS readiness edge and never a DMA issue dependency.
                if (
                    len(wait_value_ids) == 1
                    and len(body_wait_value_ids) == 1
                    and wait_value_ids != body_wait_value_ids
                    and wait_value_ids[0] in body_used_wait_value_ids
                ):
                    loop_carries.setdefault(int(op.index), []).append(
                        LoopTokenCarry(
                            loop_op_index=int(op.index),
                            init_source_value_id=int(wait_value_ids[0]),
                            yield_source_value_id=int(body_wait_value_ids[0]),
                            add_issue_dependency=False,
                            issue_dependency_op_indices=(),
                            readiness_carry=True,
                        )
                    )
                used_wait_value_ids.update(body_used_wait_value_ids)
                pending_release_value_ids.update(
                    value_id
                    for value_id in body_pending_release_value_ids
                    if value_id in wait_value_ids
                )
                continue
            if op.name == "scf.if" and len(op.region_ids) == 2:
                branch_results = tuple(
                    visit_region(child_region_id, wait_value_ids)
                    for child_region_id in op.region_ids
                )
                for (
                    _,
                    branch_used_wait_value_ids,
                    _,
                ) in branch_results:
                    used_wait_value_ids.update(branch_used_wait_value_ids)
                then_wait_value_ids, else_wait_value_ids = (
                    branch_wait_value_ids
                    for branch_wait_value_ids, _, _ in branch_results
                )
                if then_wait_value_ids == else_wait_value_ids:
                    wait_value_ids = then_wait_value_ids
                    for _, _, branch_pending_release_value_ids in branch_results:
                        pending_release_value_ids.update(
                            value_id
                            for value_id in branch_pending_release_value_ids
                            if value_id in wait_value_ids
                        )
                    continue
                then_source_value_id = (
                    None
                    if not then_wait_value_ids
                    else int(then_wait_value_ids[0])
                )
                else_source_value_id = (
                    None
                    if not else_wait_value_ids
                    else int(else_wait_value_ids[0])
                )
                if then_source_value_id is None and else_source_value_id is None:
                    wait_value_ids = ()
                    continue
                # The scf.if result is the path-sensitive readiness proof.
                # _convert_if supplies a neutral token for a path with no
                # wait, then rewrites either source wait to the merged result
                # for following DS consumers.
                if_carries.setdefault(int(op.index), []).append(
                    IfTokenCarry(
                        if_op_index=int(op.index),
                        then_source_value_id=then_source_value_id,
                        else_source_value_id=else_source_value_id,
                    )
                )
                wait_value_ids = (
                    then_source_value_id
                    if then_source_value_id is not None
                    else else_source_value_id,
                )
                if any(
                    branch_pending_release_value_ids
                    for _, _, branch_pending_release_value_ids in branch_results
                ):
                    pending_release_value_ids.update(wait_value_ids)
                continue
            for child_region_id in op.region_ids:
                (
                    _,
                    child_used_wait_value_ids,
                    child_pending_release_value_ids,
                ) = visit_region(
                    child_region_id,
                    wait_value_ids,
                )
                used_wait_value_ids.update(child_used_wait_value_ids)
                pending_release_value_ids.update(
                    value_id
                    for value_id in child_pending_release_value_ids
                    if value_id in wait_value_ids
                )
        return (
            wait_value_ids,
            frozenset(used_wait_value_ids),
            frozenset(pending_release_value_ids),
        )

    visit_region(source_program.top_region_id)
    return (
        dependencies,
        {
            op_index: tuple(carries)
            for op_index, carries in loop_carries.items()
        },
        {
            op_index: tuple(carries)
            for op_index, carries in if_carries.items()
        },
    )


def _merge_loop_token_carries(*carry_maps):
    merged = {}
    for carry_map in carry_maps:
        for op_index, carries in carry_map.items():
            existing = list(merged.get(int(op_index), ()))
            for carry in carries:
                if carry not in existing:
                    existing.append(carry)
            merged[int(op_index)] = tuple(existing)
    return merged


def _merge_if_token_carries(*carry_maps):
    merged = {}
    for carry_map in carry_maps:
        for op_index, carries in carry_map.items():
            existing = list(merged.get(int(op_index), ()))
            for carry in carries:
                if carry not in existing:
                    existing.append(carry)
            merged[int(op_index)] = tuple(existing)
    return merged


def _next_same_region_wait_by_commit(source_program):
    """Map each commit to the next explicit wait in its structured region.

    A wait in a parent or child region cannot directly consume an SSA-only
    dependency captured while emitting the commit's region.  Recording this
    relation in token analysis lets later stages coalesce compatible memory
    ordering into a real explicit wait without rediscovering source users.
    """
    result = {}
    for region in source_program.regions:
        next_wait_op_index = None
        for op_index in reversed(region.op_indices):
            op = source_program.ops[int(op_index)]
            if op.name == "ttg.async_wait":
                next_wait_op_index = int(op.index)
            elif op.name == "ttg.async_commit_group":
                result[int(op.index)] = next_wait_op_index
    return result


def _if_token_carries_by_op(
    source_program,
    nodes,
    groups,
    loop_token_carries_by_op,
):
    groups_by_id = {group.group_id: group for group in groups}
    carries_by_op = {}
    for op in source_program.ops:
        if op.name != "scf.if" or len(op.region_ids) != 2:
            continue
        branch_op_indices = tuple(
            _region_op_indices_recursive(source_program, region_id)
            for region_id in op.region_ids
        )
        all_branch_op_indices = frozenset().union(*branch_op_indices)
        groups_by_branch = tuple(
            tuple(sorted(
                (group for group in groups if group.commit_op_index in op_indices),
                key=lambda group: group.commit_op_index,
            ))
            for op_indices in branch_op_indices
        )
        if not any(groups_by_branch):
            continue
        externally_used_token_ids = {
            group.token_value_id
            for node in nodes
            if (node.op_name == "ttg.async_wait"
                and node.op_index not in all_branch_op_indices
                and node.op_index > op.index)
            for group_id in node.waited_group_ids
            for group in (groups_by_id[group_id], )
            if group.token_value_id is not None
        }
        externally_used_token_ids.update(
            source_value_id
            for loop_op_index, loop_carries in loop_token_carries_by_op.items()
            if loop_op_index not in all_branch_op_indices
            for carry in loop_carries
            for source_value_id in (
                carry.init_source_value_id,
                carry.yield_source_value_id,
            )
            if source_value_id is not None
        )
        carries = []
        for slot in range(max(len(branch) for branch in groups_by_branch)):
            then_group = groups_by_branch[0][slot] if slot < len(groups_by_branch[0]) else None
            else_group = groups_by_branch[1][slot] if slot < len(groups_by_branch[1]) else None
            slot_token_ids = {
                group.token_value_id
                for group in (then_group, else_group)
                if group is not None and group.token_value_id is not None
            }
            if slot_token_ids.isdisjoint(externally_used_token_ids):
                continue
            then_token_id = None if then_group is None else then_group.token_value_id
            else_token_id = None if else_group is None else else_group.token_value_id
            if then_token_id is None and else_token_id is None:
                continue
            carries.append(IfTokenCarry(
                if_op_index=op.index,
                then_source_value_id=then_token_id,
                else_source_value_id=else_token_id,
            ))
        if carries:
            carries_by_op[op.index] = tuple(carries)
    return carries_by_op


def _loop_token_carries_by_op(source_program, nodes, groups):
    groups_by_id = {group.group_id: group for group in groups}
    carries_by_op = {}
    for op in source_program.ops:
        if op.name != "scf.for" or len(op.region_ids) != 1:
            continue
        body_op_indices = _region_op_indices_recursive(source_program, op.region_ids[0])
        carries = _loop_token_carries_for_body(
            op,
            body_op_indices,
            nodes,
            groups,
            groups_by_id,
        )
        if carries:
            carries_by_op[op.index] = carries
    return carries_by_op


def _loop_token_carries_for_body(
    op,
    body_op_indices,
    nodes,
    groups,
    groups_by_id,
):
    initial_queue = _committed_queue_before_op(nodes, groups, op.index)
    end_queue = _loop_body_end_queue(initial_queue, nodes, groups, body_op_indices)
    queue_carries = _loop_async_queue_carries(
        op,
        initial_queue,
        end_queue,
        body_op_indices,
        nodes,
        groups_by_id,
    )
    externally_waited_body_tokens = _externally_waited_body_tokens(
        nodes,
        groups_by_id,
        body_op_indices,
    )
    yielded_by_queue = {
        carry.yield_source_value_id
        for carry in queue_carries
        if carry.yield_source_value_id is not None
    }
    final_body_carries = []
    for body_token in externally_waited_body_tokens:
        if body_token in yielded_by_queue:
            continue
        final_body_carries.append(LoopTokenCarry(
            op.index,
            None,
            body_token,
            False,
            (),
        ))
    return (*queue_carries, *final_body_carries)


def _committed_queue_before_op(nodes, groups, op_index):
    groups_by_commit = {group.commit_op_index: group for group in groups}
    nodes_by_op = {node.op_index: node for node in nodes}
    committed_queue = []
    for token_op_index in sorted(set(groups_by_commit) | set(nodes_by_op)):
        if token_op_index >= op_index:
            break
        group = groups_by_commit.get(token_op_index)
        if group is not None:
            committed_queue.append(group)
        node = nodes_by_op.get(token_op_index)
        if node is not None and node.op_name == "ttg.async_wait" and not node.input_token_ids:
            committed_queue = _remove_waited_groups(committed_queue, node.waited_group_ids)
    return tuple(committed_queue)


def _loop_body_end_queue(initial_queue, nodes, groups, body_op_indices):
    groups_by_commit = {
        group.commit_op_index: group
        for group in groups
        if group.commit_op_index in body_op_indices
    }
    nodes_by_op = {
        node.op_index: node
        for node in nodes
        if node.op_index in body_op_indices
    }
    committed_queue = list(initial_queue)
    for token_op_index in sorted(set(groups_by_commit) | set(nodes_by_op)):
        group = groups_by_commit.get(token_op_index)
        if group is not None:
            committed_queue.append(group)
        node = nodes_by_op.get(token_op_index)
        if node is not None and node.op_name == "ttg.async_wait" and not node.input_token_ids:
            committed_queue = _remove_waited_groups(committed_queue, node.waited_group_ids)
    return tuple(committed_queue)


def _remove_waited_groups(committed_queue, waited_group_ids):
    if not waited_group_ids:
        return list(committed_queue)
    waited = set(waited_group_ids)
    return [group for group in committed_queue if group.group_id not in waited]


def _loop_async_queue_carries(
    op,
    initial_queue,
    end_queue,
    body_op_indices,
    nodes,
    groups_by_id,
):
    initial_slots_by_token = {
        group.token_value_id: index
        for index, group in enumerate(initial_queue)
        if group.token_value_id is not None
    }
    needed_slots = set()
    for token_id in _loop_waited_external_tokens(
            op,
            body_op_indices,
            nodes,
            groups_by_id,
            set(initial_slots_by_token),
    ):
        needed_slots.add(initial_slots_by_token[token_id])
    worklist = list(sorted(needed_slots))
    while worklist:
        slot = worklist.pop()
        if slot >= len(end_queue):
            # The source wait drained this queue slot and the body did not
            # replenish it.  Carry a neutral token on subsequent iterations;
            # the first iteration still receives the real initial group.
            continue
        yield_token_id = end_queue[slot].token_value_id
        next_slot = initial_slots_by_token.get(yield_token_id)
        if next_slot is not None and next_slot not in needed_slots:
            needed_slots.add(next_slot)
            worklist.append(next_slot)
    carries = []
    for slot in sorted(needed_slots):
        init_token_id = initial_queue[slot].token_value_id
        yield_token_id = (None if slot >= len(end_queue) else end_queue[slot].token_value_id)
        if init_token_id is None:
            continue
        # Loop-carried async tokens are used by async_wait/final-wait
        # bookkeeping.  Matching the AMD LLVM lowering, they must not become
        # hard issue dependencies on the next direct-to-LDS copy.
        carries.append(
            LoopTokenCarry(
                loop_op_index=op.index,
                init_source_value_id=init_token_id,
                yield_source_value_id=yield_token_id,
                add_issue_dependency=False,
                issue_dependency_op_indices=(),
            ))
    return tuple(carries)


def _loop_waited_external_tokens(op, body_op_indices, nodes, groups_by_id, initial_token_ids):
    waited_tokens = []
    for node in sorted(nodes, key=lambda node: node.op_index):
        if node.op_name != "ttg.async_wait":
            continue
        if node.op_index < op.index:
            continue
        if node.op_index in body_op_indices:
            include_wait = True
        else:
            include_wait = node.op_index > op.index
        if not include_wait:
            continue
        for group_id in node.waited_group_ids:
            group = groups_by_id[group_id]
            token_id = group.token_value_id
            if group.commit_op_index in body_op_indices or token_id not in initial_token_ids:
                continue
            waited_tokens.append(token_id)
    return _dedupe_preserving_order(waited_tokens)


def _externally_waited_body_tokens(nodes, groups_by_id, body_op_indices):
    body_tokens = []
    for node in sorted(nodes, key=lambda node: node.op_index):
        if node.op_index in body_op_indices or node.op_name != "ttg.async_wait":
            continue
        for group_id in node.waited_group_ids:
            group = groups_by_id[group_id]
            if group.commit_op_index not in body_op_indices or group.token_value_id is None:
                continue
            body_tokens.append(group.token_value_id)
    return _dedupe_preserving_order(body_tokens)


def _region_op_indices_recursive(source_program, region_id):
    result = []
    for op_index in source_program.regions[region_id].op_indices:
        result.append(op_index)
        for child_region_id in source_program.ops[op_index].region_ids:
            result.extend(_region_op_indices_recursive(source_program, child_region_id))
    return frozenset(result)


def _dedupe_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _needs_token_node(source_program, op):
    return (op.name in _TOKEN_OPS or _first_token_result(source_program, op) is not None
            or bool(_input_token_ids(source_program, op)))


def _build_token_node(
    source_program,
    op,
    node_id,
    next_group_id,
    open_async_tokens,
    committed_groups,
):
    if op.name in {"ttg.async_commit_group", "ttg.async_wait"}:
        _require_token_operands(source_program, op)

    value_id = _first_token_result(source_program, op)
    input_token_ids = _input_token_ids(source_program, op)
    token_fields = _token_fields(source_program, op)
    group = None
    committed_group_id = None
    waited_group_ids = ()
    retained_group_ids = ()
    next_open_async_tokens = list(open_async_tokens)
    next_committed_groups = list(committed_groups)

    if op.name in _ASYNC_COPY_OPS:
        if value_id is None:
            fail(
                "TLXW_TOKEN_MISSING_RESULT",
                STAGE,
                f"{op.name} must produce an async token",
                source_op_index=op.index,
            )
        next_open_async_tokens.append(value_id)

    if op.name == "ttg.async_commit_group":
        if input_token_ids:
            member_token_ids = input_token_ids
            committed = set(member_token_ids)
            next_open_async_tokens = [token_id for token_id in next_open_async_tokens if token_id not in committed]
        else:
            member_token_ids = tuple(next_open_async_tokens)
            next_open_async_tokens = []
        committed_group_id = next_group_id
        group = TokenGroup(
            committed_group_id,
            op.index,
            value_id,
            tuple(member_token_ids),
        )
        next_committed_groups.append(group)

    if op.name == "ttg.async_wait":
        wait_group = token_fields["wait_group"]
        if wait_group is not None and wait_group < 0:
            fail(
                "TLXW_TOKEN_MALFORMED_WAIT",
                STAGE,
                f"ttg.async_wait requires a nonnegative wait group, got {wait_group}",
                source_op_index=op.index,
            )
        if not input_token_ids:
            waited_group_ids = _waited_group_ids(next_committed_groups, wait_group)
            waited = set(waited_group_ids)
            retained_group_ids = tuple(
                group.group_id
                for group in next_committed_groups
                if group.group_id not in waited
            )
            if waited_group_ids:
                next_committed_groups = [group for group in next_committed_groups if group.group_id not in waited]

    node = TokenNode(
        node_id,
        op.index,
        op.name,
        value_id,
        tuple(input_token_ids),
        committed_group_id=committed_group_id,
        waited_group_ids=tuple(waited_group_ids),
        retained_group_ids=tuple(retained_group_ids),
        **token_fields,
    )
    return node, group, next_open_async_tokens, next_committed_groups


def _token_fields(source_program, op):
    if op.name == "ttg.async_copy_global_to_local":
        return _global_async_copy_fields(op)
    if op.name == "amdg.buffer_load_to_local":
        return _buffer_async_copy_fields(op)
    if op.name == "ttg.async_wait":
        return {
            "source_address_value_id": None,
            "source_offset_value_id": None,
            "memdesc_value_id": None,
            "mask_value_id": None,
            "other_value_id": None,
            "wait_group": _int_attr(op, "num"),
        }
    del source_program
    return {
        "source_address_value_id": None,
        "source_offset_value_id": None,
        "memdesc_value_id": None,
        "mask_value_id": None,
        "other_value_id": None,
        "wait_group": None,
    }


def _global_async_copy_fields(op):
    segments = _operand_segments(op, 4, (1, 1, 1 if len(op.operands) > 2 else 0, 0))
    _require_operand_count(op, segments)
    mask_index = int(segments[0]) + int(segments[1])
    other_index = mask_index + int(segments[2])
    return {
        "source_address_value_id": _operand_or_none(op, 0),
        "source_offset_value_id": None,
        "memdesc_value_id": _operand_or_none(op, 1),
        "mask_value_id": (_operand_or_none(op, mask_index) if int(segments[2]) else None),
        "other_value_id": (_operand_or_none(op, other_index) if int(segments[3]) else None),
        "wait_group": None,
    }


def _buffer_async_copy_fields(op):
    segments = _operand_segments(
        op,
        6,
        (1, 1, 1, 1 if len(op.operands) > 3 else 0, 0, 0),
    )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "source_address_value_id": _operand_or_none(op, base_index),
        "source_offset_value_id": _operand_or_none(op, offset_index),
        "memdesc_value_id": _operand_or_none(op, 0),
        "mask_value_id": (_operand_or_none(op, mask_index) if int(segments[3]) else None),
        "other_value_id": (_operand_or_none(op, other_index) if int(segments[4]) else None),
        "wait_group": None,
    }


class _DependencyFrontier:

    def __init__(self):
        self._last_writes_by_domain = {}
        self._reads_since_write_by_domain = {}

    def dependencies_for(
        self,
        *,
        kind,
        address_space,
        volatile=False,
        ordering=None,
        sync_scope=None,
    ):
        domains = _alias_domains_for_query(address_space, self._known_domains())
        if _effect_is_barrier_like(volatile, ordering, sync_scope):
            return _dedupe_effect_ids(effect_id for domain in domains for effect_id in (
                *self._last_writes_by_domain.get(domain, ()),
                *self._reads_since_write_by_domain.get(domain, ()),
            ))
        if kind == "read":
            return _dedupe_effect_ids(effect_id for domain in domains
                                      for effect_id in self._last_writes_by_domain.get(domain, ()))
        if kind == "write":
            return _dedupe_effect_ids(effect_id for domain in domains for effect_id in (
                *self._last_writes_by_domain.get(domain, ()),
                *self._reads_since_write_by_domain.get(domain, ()),
            ))
        return ()

    def record(self, effect):
        domain = _alias_domain(effect.address_space)
        if _effect_is_barrier_like(
                effect.volatile,
                effect.ordering,
                effect.sync_scope,
        ) or effect.kind == "write":
            if domain == "unknown":
                self._last_writes_by_domain.clear()
                self._reads_since_write_by_domain.clear()
            self._last_writes_by_domain[domain] = (effect.effect_id, )
            self._reads_since_write_by_domain[domain] = ()
            return
        if effect.kind == "read":
            self._reads_since_write_by_domain[domain] = (
                *self._reads_since_write_by_domain.get(domain, ()),
                effect.effect_id,
            )

    def _known_domains(self):
        return frozenset((
            *self._last_writes_by_domain.keys(),
            *self._reads_since_write_by_domain.keys(),
        ))


def _memory_effects_for_op(
    source_program,
    op,
    token_node_id,
    next_effect_id,
    dependency_frontier,
):
    fields = _token_fields(source_program, op) if op.name in _ASYNC_COPY_OPS else {}
    if op.name == "ttg.async_copy_global_to_local":
        return _effect_pair(
            source_program,
            op,
            token_node_id,
            fields["source_address_value_id"],
            None,
            fields["memdesc_value_id"],
            fields["mask_value_id"],
            next_effect_id,
            dependency_frontier,
            read_space="global",
        )
    if op.name == "amdg.buffer_load_to_local":
        return _effect_pair(
            source_program,
            op,
            token_node_id,
            fields["source_address_value_id"],
            fields["source_offset_value_id"],
            fields["memdesc_value_id"],
            fields["mask_value_id"],
            next_effect_id,
            dependency_frontier,
            read_space="buffer",
        )
    if op.name == "tt.load":
        mask_value_id = _operand_or_none(op, 1) if len(op.operands) > 1 else None
        return (_memory_effect(
            source_program,
            op,
            "read",
            _pointer_address_space(source_program, _operand_or_none(op, 0)),
            _operand_or_none(op, 0),
            None,
            None,
            mask_value_id,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "tt.store":
        mask_value_id = _operand_or_none(op, 2) if len(op.operands) > 2 else None
        return (_memory_effect(
            source_program,
            op,
            "write",
            _pointer_address_space(source_program, _operand_or_none(op, 0)),
            _operand_or_none(op, 0),
            None,
            _operand_or_none(op, 1),
            mask_value_id,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "ttg.local_load":
        return (_memory_effect(
            source_program,
            op,
            "read",
            "local",
            _operand_or_none(op, 0),
            None,
            _operand_or_none(op, 1) if len(op.operands) > 1 else None,
            None,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "ttg.local_store":
        return (_memory_effect(
            source_program,
            op,
            "write",
            "local",
            _operand_or_none(op, 1),
            None,
            _operand_or_none(op, 0),
            None,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "amdg.buffer_load":
        fields = _buffer_load_fields(op)
        return (_memory_effect(
            source_program,
            op,
            "read",
            "buffer",
            fields["base_value_id"],
            fields["offset_value_id"],
            None,
            fields["mask_value_id"],
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "amdg.buffer_store":
        fields = _buffer_store_fields(op)
        return (_memory_effect(
            source_program,
            op,
            "write",
            "buffer",
            fields["base_value_id"],
            fields["offset_value_id"],
            fields["value_value_id"],
            fields["mask_value_id"],
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    return ()


def _effect_pair(
    source_program,
    op,
    token_node_id,
    source_address_value_id,
    source_offset_value_id,
    memdesc_value_id,
    mask_value_id,
    next_effect_id,
    dependency_frontier,
    *,
    read_space,
):
    read = _memory_effect(
        source_program,
        op,
        "read",
        read_space,
        source_address_value_id,
        source_offset_value_id,
        None,
        mask_value_id,
        token_node_id,
        next_effect_id,
        dependency_frontier,
    )
    write = _memory_effect(
        source_program,
        op,
        "write",
        "local",
        memdesc_value_id,
        None,
        None,
        mask_value_id,
        token_node_id,
        next_effect_id + 1,
        dependency_frontier,
        explicit_dependency_ids=(read.effect_id, ),
    )
    return read, write


def _memory_effect(
        source_program,
        op,
        kind,
        address_space,
        address_value_id,
        offset_value_id,
        value_value_id,
        mask_value_id,
        token_node_id,
        effect_id,
        dependency_frontier,
        explicit_dependency_ids=(),
):
    del source_program
    volatile = bool(op.attrs.get("volatile", False))
    ordering = _attr_or_none(op, "ordering")
    sync_scope = _attr_or_none(op, "syncscope")
    effect = MemoryEffect(
        effect_id,
        op.index,
        op.name,
        kind,
        address_space,
        address_value_id,
        offset_value_id,
        value_value_id,
        mask_value_id,
        token_node_id,
        _cache_modifier(op),
        volatile,
        ordering,
        sync_scope,
        "unknown",
        _dedupe_effect_ids((
            *dependency_frontier.dependencies_for(
                kind=kind,
                address_space=address_space,
                volatile=volatile,
                ordering=ordering,
                sync_scope=sync_scope,
            ),
            *explicit_dependency_ids,
        )),
    )
    dependency_frontier.record(effect)
    return effect


def _alias_domain(address_space):
    if address_space in {"global", "buffer"}:
        return "global"
    if address_space == "local":
        return "local"
    return "unknown"


def _alias_domains_for_query(address_space, known_domains):
    domain = _alias_domain(address_space)
    if domain == "unknown":
        return tuple(sorted(known_domains | {"global", "local", "unknown"}))
    return (domain, "unknown")


def _effect_is_barrier_like(volatile, ordering, sync_scope):
    return bool(volatile or ordering or sync_scope)


def _dedupe_effect_ids(effect_ids):
    result = []
    seen = set()
    for effect_id in effect_ids:
        effect_id = int(effect_id)
        if effect_id in seen:
            continue
        seen.add(effect_id)
        result.append(effect_id)
    return tuple(sorted(result))


def _buffer_load_fields(op):
    segments = _operand_segments(op, 5, None)
    _require_operand_count(op, segments)
    if segments[0] != 1 or segments[1] != 1:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load requires base pointer and offsets operands",
            source_op_index=op.index,
        )
    if segments[2] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load supports at most one stride operand",
            source_op_index=op.index,
        )
    if segments[3] not in (0, 1) or segments[4] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load supports at most one mask and one other operand",
            source_op_index=op.index,
        )
    offset_index = int(segments[0])
    stride_index = offset_index + int(segments[1])
    mask_index = stride_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "base_value_id": _operand_or_none(op, 0),
        "offset_value_id": _operand_or_none(op, offset_index),
        "stride_value_id": _operand_or_none(op, stride_index) if segments[2] else None,
        "mask_value_id": _operand_or_none(op, mask_index) if segments[3] else None,
        "other_value_id": _operand_or_none(op, other_index) if segments[4] else None,
    }


def _waited_group_ids(committed_groups, keep_count):
    keep_count = max(0, int(keep_count or 0))
    wait_count = max(0, len(committed_groups) - keep_count)
    return tuple(group.group_id for group in committed_groups[:wait_count])


def _first_token_result(source_program, op):
    for value_id in op.results:
        if _value_is_token(source_program, value_id):
            return value_id
    return None


def _input_token_ids(source_program, op):
    return tuple(value_id for value_id in op.operands if _value_is_token(source_program, value_id))


def _require_token_operands(source_program, op):
    for value_id in op.operands:
        if not _value_is_token(source_program, value_id):
            fail(
                "TLXW_TOKEN_NON_TOKEN_DEPENDENCY",
                STAGE,
                f"{op.name} operand {value_id} is not a token",
                source_op_index=op.index,
                source_value_id=value_id,
            )


def _value_is_token(source_program, value_id):
    value = source_program.values.get(value_id)
    return value is not None and value.type.kind == "token"


def _operand_segments(op, expected_len, default):
    segments = op.attrs.get("operandSegmentSizes")
    if segments is None:
        if default is None:
            fail(
                "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
                STAGE,
                f"{op.name} expected operandSegmentSizes",
                source_op_index=op.index,
            )
        segments = default
    segments = tuple(int(segment) for segment in segments)
    if len(segments) != expected_len:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} expected {expected_len} operand segments, got {segments}",
            source_op_index=op.index,
        )
    if any(segment < 0 for segment in segments):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments must be nonnegative, got {segments}",
            source_op_index=op.index,
        )
    return segments


def _buffer_store_fields(op):
    segments = _operand_segments(op, 5, None)
    _require_operand_count(op, segments)
    if segments[0] != 1 or segments[1] != 1 or segments[2] != 1:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_store requires value, base pointer, and offsets operands",
            source_op_index=op.index,
        )
    if segments[3] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_store supports at most one boundary-check operand",
            source_op_index=op.index,
        )
    if segments[4] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_store supports at most one mask operand",
            source_op_index=op.index,
        )
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2]) + int(segments[3])
    boundary_index = offset_index + int(segments[2])
    return {
        "value_value_id": _operand_or_none(op, 0),
        "base_value_id": _operand_or_none(op, base_index),
        "offset_value_id": _operand_or_none(op, offset_index),
        "boundary_check_value_id": _operand_or_none(op, boundary_index) if segments[3] else None,
        "mask_value_id": _operand_or_none(op, mask_index) if segments[4] else None,
    }


def _require_operand_count(op, segments):
    if sum(segments) != len(op.operands):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments {segments} do not match "
            f"{len(op.operands)} operands",
            source_op_index=op.index,
        )


def _operand_or_none(op, index):
    return op.operands[index] if index is not None and index < len(op.operands) else None


def _pointer_address_space(source_program, value_id):
    value = source_program.values.get(value_id)
    if value is None:
        return "unknown"
    address_space = value.type.address_space
    if address_space in {3, "3"}:
        return "local"
    if address_space in {1, "1"}:
        return "global"
    return "unknown"


def _append_user(users_by_value, value_id, node_id):
    users = users_by_value.setdefault(value_id, [])
    if node_id not in users:
        users.append(node_id)


def _int_attr(op, name):
    value = op.attrs.get(name)
    return None if value is None else int(value)


def _attr_or_none(op, name):
    value = op.attrs.get(name)
    return None if value is None else str(value)


def _cache_modifier(op):
    value = op.attrs.get("cacheModifier")
    if value is None:
        value = op.attrs.get("cache")
    return None if value is None else str(value)
