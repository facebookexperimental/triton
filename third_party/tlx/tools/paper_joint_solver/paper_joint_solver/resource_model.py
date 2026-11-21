"""Typed value and storage footprints for the paper's memory model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


_SHAPED_TYPE_RE = re.compile(r"^(tensor|!ttg\.memdesc)<((?:[0-9]+x)+)(.*)$")
_SCALAR_TYPE_RE = re.compile(
    r"^(bf16|f16|f32|f64|i1|i8|i16|i32|i64|index)$"
)
_ELEMENT_TYPE_RE = re.compile(
    r"^(bf16|f16|f32|f64|i1|i8|i16|i32|i64|index)(?=[,>])"
)
_DTYPE_BITS = {
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "index": 64,
}
_ZERO_SIZED_TYPES = ("!ttg.async.token", "!tt.tensordesc<")


@dataclass(frozen=True)
class StorageObject:
    id: str
    kind: str
    amount: int
    members: frozenset[int]


@dataclass
class _MutableStorage:
    kind: str
    amount: int
    members: set[int]


def shaped_type_info(type_str: str) -> tuple[list[int], int] | None:
    match = _SHAPED_TYPE_RE.match(type_str)
    if not match:
        return None
    shape = [
        int(dim) for dim in match.group(2).removesuffix("x").split("x")
    ]
    element_type = match.group(3)
    element = _ELEMENT_TYPE_RE.match(element_type)
    if element:
        bits = _DTYPE_BITS[element.group(1)]
    elif element_type.startswith("!tt.ptr<"):
        bits = 64
    else:
        raise ValueError(f"unsupported shaped element type: {type_str}")
    return shape, bits


def type_bytes(type_str: str) -> int:
    shaped = shaped_type_info(type_str)
    if shaped:
        shape, bits = shaped
        elements = 1
        for dim in shape:
            elements *= dim
        return (elements * bits + 7) // 8
    scalar = _SCALAR_TYPE_RE.match(type_str)
    if scalar:
        return (_DTYPE_BITS[scalar.group(1)] + 7) // 8
    if type_str.startswith("!tt.ptr<"):
        return 8
    if type_str == _ZERO_SIZED_TYPES[0] or type_str.startswith(
        _ZERO_SIZED_TYPES[1]
    ):
        return 0
    raise ValueError(f"unsupported result type: {type_str}")


def register_words(result_types: Iterable[str]) -> int:
    total_bytes = 0
    for type_str in result_types:
        if type_str.startswith("!ttg.memdesc<"):
            continue
        total_bytes += type_bytes(type_str)
    return (total_bytes + 3) // 4


def storage_amount(
    result_types: Iterable[str], kind: str, tmem_column_bytes: int
) -> int:
    if kind == "smem":
        marker = "#ttg.shared_memory"
        return sum(type_bytes(t) for t in result_types if marker in t)
    if kind == "tmem":
        marker = "#ttng.tensor_memory"
        total_bytes = sum(type_bytes(t) for t in result_types if marker in t)
        return (total_bytes + tmem_column_bytes - 1) // tmem_column_bytes
    raise ValueError(f"unknown storage kind: {kind}")


def _op_ref(operand: object) -> str | None:
    if isinstance(operand, dict):
        value = operand.get("op")
        return value if isinstance(value, str) else None
    return None


def derive_resources(
    nodes: Mapping[int, object],
    edges: Iterable[object],
    ops: Mapping[str, dict],
    *,
    tmem_column_bytes: int,
) -> tuple[dict[int, int], dict[str, StorageObject]]:
    """Derive register values and alias-aware SMEM/TMEM storage objects."""
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for edge in edges:
        successors.setdefault(edge.src, []).append(edge.dst)
        predecessors.setdefault(edge.dst, []).append(edge.src)

    def result_types(node_id: int) -> list[str]:
        return ops.get(nodes[node_id].op_ref, {}).get("result_types", [])

    def view_closure(seeds: Iterable[int], *, include_local_alloc: bool) -> set[int]:
        members = set(seeds)
        work = list(seeds)
        while work:
            source = work.pop()
            for target in successors.get(source, ()):
                kind = nodes[target].op_kind
                is_view = kind == "ttg.memdesc_trans"
                if include_local_alloc:
                    is_view = is_view or kind == "ttg.local_alloc"
                if is_view and target not in members:
                    members.add(target)
                    work.append(target)
        return members

    mutable: dict[str, _MutableStorage] = {}

    def add_storage(
        storage_id: str, kind: str, amount: int, members: set[int]
    ) -> bool:
        if amount <= 0:
            return False
        current = mutable.setdefault(
            storage_id, _MutableStorage(kind=kind, amount=amount, members=set())
        )
        if current.kind != kind:
            raise ValueError(f"storage {storage_id} changes kind")
        current.amount = max(current.amount, amount)
        current.members.update(members)
        return True

    claimed: set[int] = set()
    tma_with_allocation: set[int] = set()
    for node in nodes.values():
        if node.op_kind != "ttg.local_alloc":
            continue
        amount = storage_amount(result_types(node.id), "smem", tmem_column_bytes)
        if amount <= 0:
            raise ValueError(f"local_alloc node {node.id} has no SMEM footprint")
        members = view_closure([node.id], include_local_alloc=False)

        # A local_alloc is the allocation boundary. Upstream descriptor loads
        # and memdesc views name this object, but two local_alloc users of one
        # descriptor load remain two distinct SMEM allocations.
        work = list(predecessors.get(node.id, ()))
        seen: set[int] = set()
        while work:
            source = work.pop()
            if source in seen:
                continue
            seen.add(source)
            source_node = nodes[source]
            if (
                source_node.pipeline == "TMA"
                and source_node.op_kind == "tt.descriptor_load"
            ):
                members.add(source)
                tma_with_allocation.add(source)
            elif source_node.op_kind == "ttg.memdesc_trans":
                members.add(source)
                work.extend(predecessors.get(source, ()))
        if add_storage(f"smem:{node.id}", "smem", amount, members):
            claimed.update(members)

    for node in nodes.values():
        if (
            node.id in tma_with_allocation
            or node.pipeline != "TMA"
            or node.op_kind != "tt.descriptor_load"
        ):
            continue
        amount = sum(type_bytes(t) for t in result_types(node.id))
        members = view_closure([node.id], include_local_alloc=False)
        if add_storage(f"smem:{node.id}", "smem", amount, members):
            claimed.update(members)

    node_by_op_ref = {node.op_ref: node.id for node in nodes.values() if node.op_ref}
    for node in nodes.values():
        if node.op_kind != "ttng.tmem_alloc":
            continue
        amount = storage_amount(result_types(node.id), "tmem", tmem_column_bytes)
        if amount <= 0:
            raise ValueError(f"tmem_alloc node {node.id} has no TMEM footprint")
        members = view_closure([node.id], include_local_alloc=False)
        if add_storage(f"tmem:{node.op_ref}", "tmem", amount, members):
            claimed.update(members)

    for node in nodes.values():
        if "mma" not in node.op_kind:
            continue
        entry = ops.get(node.op_ref, {})
        operands = entry.get("operands", [])
        accumulator = _op_ref(operands[2]) if len(operands) > 2 else None
        if accumulator is None:
            continue
        accumulator_types = ops.get(accumulator, {}).get("result_types", [])
        amount = storage_amount(accumulator_types, "tmem", tmem_column_bytes)
        members = {node.id}
        owner = node_by_op_ref.get(accumulator)
        if owner is not None:
            members.add(owner)
        if add_storage(f"tmem:{accumulator}", "tmem", amount, members):
            claimed.update(members)

    for node in nodes.values():
        if node.op_kind != "ttng.tmem_store":
            continue
        operands = ops.get(node.op_ref, {}).get("operands", [])
        target = _op_ref(operands[0]) if operands else None
        if target is None:
            continue
        target_types = ops.get(target, {}).get("result_types", [])
        amount = storage_amount(target_types, "tmem", tmem_column_bytes)
        if add_storage(f"tmem:{target}", "tmem", amount, {node.id}):
            claimed.add(node.id)

    unclaimed_handles = [
        node.id
        for node in nodes.values()
        if node.id not in claimed
        and any(
            result_type.startswith("!ttg.memdesc<")
            for result_type in result_types(node.id)
        )
    ]
    if unclaimed_handles:
        raise ValueError(
            "unrecognized memory-handle transformation at nodes "
            f"{sorted(unclaimed_handles)}"
        )

    registers = {
        node.id: 0 if node.id in claimed else register_words(result_types(node.id))
        for node in nodes.values()
    }
    objects = {
        storage_id: StorageObject(
            id=storage_id,
            kind=value.kind,
            amount=value.amount,
            members=frozenset(value.members),
        )
        for storage_id, value in mutable.items()
    }
    return registers, objects
