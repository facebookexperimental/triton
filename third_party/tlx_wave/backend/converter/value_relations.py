"""Exact value algebra for symbolic global-memory offsets."""

from dataclasses import dataclass, replace

from ..wave_bridge_tools import load_wave_dsl
from . import target_ir
from .diagnostics import fail

_MEMORY_OFFSET_OPERAND = {
    "buffer_load": 1,
    "buffer_load_to_local": 2,
    "buffer_store": 2,
}

STAGE = "value_relations"


@dataclass(frozen=True)
class _Point:
    block: object
    item: object
    slot: object


@dataclass(frozen=True)
class _Relation:
    expr: object
    bindings: tuple[tuple[str, int], ...] = ()
    constraints: tuple[object, ...] = ()


class _Analysis:

    def __init__(self, target_program):
        self.program = target_program
        self.dsl = load_wave_dsl()
        self.definition_by_result = {int(result): op for op in target_program.ops for result in op.results}
        self.block = self.dsl.sym("block")
        self.item = self.dsl.sym("item")
        self.slot = self.dsl.sym("slot")
        self._memo = {}
        self._active = set()
        self._explicit_uniform = {}
        self._explicit_uniform_active = set()
        self._assumption_by_id = {
            int(assumption.assumption_id): assumption
            for assumption in target_program.assumptions
        }

    def memory_relation(self, op):
        offset_index = _MEMORY_OFFSET_OPERAND.get(op.kind)
        if offset_index is None or len(op.operands) <= offset_index:
            return None
        attrs = target_ir.attrs_dict(op)
        offset_id = int(op.operands[offset_index])
        domain = self._domain(offset_id)
        if domain is None:
            return None
        point, facts = domain
        relation = self._evaluate(offset_id, point, facts)
        if relation is None:
            return None
        element_bytes = int(attrs.get("element_byte_width", 0))
        if element_bytes <= 0:
            return None
        # amdg.buffer_* forms its byte voffset in i32.  The descriptor consumes
        # that bit pattern as an unsigned 32-bit offset, so preserve the source
        # operation's fixed-width algebra before converting bytes to bits.
        byte_offset = self.dsl.mod(
            relation.expr * element_bytes,
            1 << 32,
        )
        bit_offset = byte_offset * 8
        integer = self.dsl.ixs_eq(bit_offset, self.dsl.floor(bit_offset))
        relation_facts = (*facts, *relation.constraints)
        proofs, normalized = self.dsl.ixs_check((integer, bit_offset), relation_facts)
        if proofs[0] is not True:
            return None
        bit_offset = normalized[1]
        return _Relation(bit_offset, relation.bindings)

    def _domain(self, target_value_id):
        value = self.program.values[int(target_value_id)]
        if value.layout_map_id is None:
            # A scalar splat is an exact uniform packet even when a hand-built
            # target program has no layout map.  Admit only that explicit SSA
            # carrier: a missing layout alone is never evidence that a
            # distributed value is uniform.
            if not self._is_explicit_uniform(target_value_id):
                return None
            return _Point(self.block, self.item, self.slot), ()
        layout = self.program.layouts[int(value.layout_map_id)]
        if layout.linear_layout is None:
            return None
        block_count = _linear_input_extent(layout.linear_layout, "block")
        warp_count = _linear_input_extent(layout.linear_layout, "warp")
        lane_width = int(value.type.lane_width or layout.lane_width or 0)
        slot_count = int(value.type.component_count)
        if min(block_count, warp_count, lane_width, slot_count) <= 0:
            return None
        point = _Point(
            self.block,
            self.item,
            self.slot,
        )
        facts = (
            *_range_facts(self.dsl, self.block, 0, block_count - 1),
            *_range_facts(
                self.dsl,
                self.item,
                0,
                lane_width * warp_count - 1,
            ),
            *_range_facts(self.dsl, self.slot, 0, slot_count - 1),
        )
        return point, facts

    def _is_explicit_uniform(self, target_value_id):
        target_value_id = int(target_value_id)
        if target_value_id in self._explicit_uniform:
            return self._explicit_uniform[target_value_id]
        if target_value_id in self._explicit_uniform_active:
            return False
        self._explicit_uniform_active.add(target_value_id)
        try:
            value = self.program.values[target_value_id]
            if value.type.representation == "scalar":
                result = True
            else:
                op = self.definition_by_result.get(target_value_id)
                result = (op is not None and op.kind == "splat" and len(op.operands) == 1
                          and self._is_explicit_uniform(op.operands[0]))
        finally:
            self._explicit_uniform_active.remove(target_value_id)
        self._explicit_uniform[target_value_id] = result
        return result

    def _evaluate(self, target_value_id, point, facts):
        target_value_id = int(target_value_id)
        value = self.program.values[target_value_id]
        op = self.definition_by_result.get(target_value_id)
        if op is not None and op.kind == "assume":
            return self._evaluate_assume(
                target_value_id,
                op,
                point,
                facts,
            )
        if value.type.representation == "scalar":
            return self._scalar(target_value_id)
        if value.type.element_type not in {"i1", "i8", "i16", "i32", "i64", "index"}:
            return None
        key = self._relation_key(target_value_id, point)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return None
        self._active.add(key)
        try:
            relation = self._evaluate_definition(
                target_value_id,
                point,
                facts,
            )
            self._memo[key] = relation
            return relation
        finally:
            self._active.remove(key)

    def _evaluate_definition(self, target_value_id, point, facts):
        op = self.definition_by_result.get(int(target_value_id))
        if op is None:
            return None
        attrs = target_ir.attrs_dict(op)
        if op.kind == "constant":
            literal = attrs.get("value")
            if type(literal) is bool:
                literal = int(literal)
            if type(literal) is int:
                return _Relation(self.dsl.ixs_int(literal))
            return None
        if op.kind == "make_range":
            return _Relation(self._substitute_point(
                self.dsl.ixs_deserialize(attrs["relation"]),
                point,
            ))
        if op.kind == "splat" and len(op.operands) == 1:
            return self._evaluate(int(op.operands[0]), point, facts)
        if op.kind == "binary" and len(op.operands) == 2:
            operands = tuple(self._evaluate(int(operand), point, facts) for operand in op.operands)
            if any(operand is None for operand in operands):
                return None
            return self._binary(op, attrs, operands[0], operands[1], facts)
        if op.kind == "type_convert" and len(op.operands) == 1:
            source = self._evaluate(int(op.operands[0]), point, facts)
            if source is None or attrs.get("mode") != "index_cast":
                return None
            source_bits = _element_bits(self.program.values[int(op.operands[0])].type.element_type)
            result_bits = _element_bits(self.program.values[int(target_value_id)].type.element_type)
            if source_bits is None or result_bits is None:
                return None
            expr = (_signed_fixed(self.dsl, source.expr, result_bits) if result_bits < source_bits else source.expr)
            return self._normalized(
                expr,
                source.bindings,
                facts,
                source.constraints,
            )
        if op.kind in {"broadcast", "expand_dims", "layout_convert"}:
            return self._pullback(
                int(op.operands[0]),
                attrs["relation"],
                point,
                facts,
            )
        if op.kind == "split":
            result_index = tuple(int(value) for value in op.results).index(int(target_value_id))
            blob = attrs["relations"][result_index]
            return self._pullback(
                int(op.operands[0]),
                blob,
                point,
                facts,
            )
        if op.kind == "join":
            return self._pullback_join(
                op,
                attrs["relation"],
                point,
                facts,
            )
        if op.kind == "select" and len(op.operands) == 3:
            constant_condition = self._constant_bool(int(op.operands[0]))
            if constant_condition is not None:
                return self._evaluate(
                    int(op.operands[1 if constant_condition else 2]),
                    point,
                    facts,
                )
            condition = self._evaluate(int(op.operands[0]), point, facts)
            true_value = self._evaluate(int(op.operands[1]), point, facts)
            false_value = self._evaluate(int(op.operands[2]), point, facts)
            if condition is None or true_value is None or false_value is None:
                return None
            bindings = _merge_bindings(condition, true_value, false_value)
            if bindings is None:
                return None
            return self._normalized(
                condition.expr * true_value.expr + (1 - condition.expr) * false_value.expr,
                bindings,
                facts,
                (
                    *condition.constraints,
                    *true_value.constraints,
                    *false_value.constraints,
                ),
            )
        if op.kind == "if" and len(op.region_ids) == 2:
            try:
                result_index = tuple(int(value) for value in op.results).index(int(target_value_id))
                yields = tuple(
                    int(self.program.regions[int(region_id)].yield_value_ids[result_index])
                    for region_id in op.region_ids)
            except (ValueError, IndexError):
                return None
            return self._equivalent_relations(
                tuple(self._evaluate(value_id, point, facts) for value_id in yields),
                facts,
            )
        if op.kind == "for_loop" and len(op.region_ids) == 1:
            source_results = int(attrs.get("source_result_count", 0))
            try:
                result_index = tuple(int(value) for value in op.results[:source_results]).index(int(target_value_id))
                region = self.program.regions[int(op.region_ids[0])]
                init_id = int(op.operands[3 + result_index])
                block_arg_id = int(region.block_arg_ids[1 + result_index])
                yield_id = int(region.yield_value_ids[result_index])
            except (ValueError, IndexError):
                return None
            initial = self._evaluate(init_id, point, facts)
            if initial is None:
                return None
            block_key = self._relation_key(block_arg_id, point)
            previous = self._memo.get(block_key)
            self._memo[block_key] = initial
            try:
                yielded = self._evaluate(yield_id, point, facts)
            finally:
                if previous is None:
                    self._memo.pop(block_key, None)
                else:
                    self._memo[block_key] = previous
            return self._equivalent_relations((initial, yielded), facts)
        return None

    def _scalar(self, target_value_id):
        op = self.definition_by_result.get(int(target_value_id))
        if op is not None and op.kind == "constant":
            literal = target_ir.attrs_dict(op).get("value")
            if type(literal) is bool:
                literal = int(literal)
            if type(literal) is int:
                return _Relation(self.dsl.ixs_int(literal))
        value = self.program.values[int(target_value_id)]
        element_type = value.type.element_type
        if element_type != "index" and _element_bits(element_type) is None:
            return None
        name = f"t{target_value_id}"
        return _Relation(
            self.dsl.sym(name),
            ((name, int(target_value_id)), ),
        )

    def _binary(self, op, attrs, lhs, rhs, facts):
        bindings = _merge_bindings(lhs, rhs)
        if bindings is None:
            return None
        proof_facts = (
            *facts,
            *lhs.constraints,
            *rhs.constraints,
        )
        operation = attrs.get("operation")
        width = attrs.get("source_width")
        try:
            width = int(width)
        except (TypeError, ValueError):
            return None
        if width <= 0 or width > 64:
            return None
        if operation == "addi":
            mathematical = lhs.expr + rhs.expr
        elif operation == "subi":
            mathematical = lhs.expr - rhs.expr
        elif operation == "muli":
            mathematical = lhs.expr * rhs.expr
        elif operation == "andi":
            mathematical = lhs.expr & rhs.expr
        elif operation == "ori":
            mathematical = lhs.expr | rhs.expr
        elif operation == "xori":
            mathematical = self.dsl.xor(lhs.expr, rhs.expr)
        elif operation in {"divui", "remui"}:
            if width >= 63:
                return None
            unsigned_lhs = _unsigned_fixed(self.dsl, lhs.expr, width)
            unsigned_rhs = _unsigned_fixed(self.dsl, rhs.expr, width)
            proofs, _ = self.dsl.ixs_check(
                (unsigned_rhs > 0, ),
                proof_facts,
            )
            if proofs[0] is not True:
                return None
            quotient = self.dsl.trunc(unsigned_lhs / unsigned_rhs)
            mathematical = (quotient if operation == "divui" else self.dsl.mod(unsigned_lhs, unsigned_rhs))
        elif operation in {"divsi", "remsi"}:
            preconditions = [(rhs.expr < 0) | (rhs.expr > 0)]
            if operation == "divsi":
                minimum = -(1 << (width - 1))
                preconditions.append((lhs.expr < minimum) | (lhs.expr > minimum) | (rhs.expr < -1) | (rhs.expr > -1))
            proofs, _ = self.dsl.ixs_check(
                tuple(preconditions),
                proof_facts,
            )
            if any(proof is not True for proof in proofs):
                return None
            quotient = self.dsl.trunc(lhs.expr / rhs.expr)
            mathematical = (quotient if operation == "divsi" else lhs.expr - rhs.expr * quotient)
        else:
            return None
        # The target contract makes address-chain overflow impossible, so
        # additive and multiplicative offsets are their mathematical values.
        if operation in {"divui", "remui"}:
            mathematical = _signed_fixed(self.dsl, mathematical, width)
        return self._normalized(
            mathematical,
            bindings,
            facts,
            (*lhs.constraints, *rhs.constraints),
        )

    def _evaluate_assume(self, target_value_id, op, point, facts):
        try:
            result_index = tuple(int(result) for result in op.results).index(int(target_value_id))
            operand_id = int(op.operands[result_index])
        except (ValueError, IndexError):
            return None
        source = self._evaluate(operand_id, point, facts)
        if source is None:
            return None
        leaf_names = tuple(name for name, binding_target_id in source.bindings if int(binding_target_id) == operand_id)
        constraints = self._assume_constraints(
            op,
            target_value_id,
            source.expr,
        )
        bindings = tuple((
            name,
            int(target_value_id
                ) if len(leaf_names) == 1 and int(binding_target_id) == operand_id else int(binding_target_id),
        ) for name, binding_target_id in source.bindings)
        return self._normalized(
            source.expr,
            bindings,
            facts,
            (*source.constraints, *constraints),
        )

    def _assume_constraints(self, op, result_id, expression):
        projected = []
        for fact_id, target_id in zip(
                op.fact_ids,
                op.fact_target_ids,
                strict=True,
        ):
            if int(target_id) != int(result_id):
                continue
            fact = self._assumption_by_id.get(int(fact_id))
            if fact is None:
                fail(
                    "TLXW_RELATION_UNKNOWN_FACT",
                    STAGE,
                    f"assume references missing fact {fact_id}",
                    target_op_id=op.target_op_id,
                    fact_id=fact_id,
                )
            if int(result_id) not in fact.subject_target_ids:
                fail(
                    "TLXW_RELATION_FACT_TARGET",
                    STAGE,
                    f"assumption {fact_id} does not apply to target value {result_id}",
                    target_op_id=op.target_op_id,
                    target_value_id=result_id,
                    fact_id=fact_id,
                )
            if fact.kind == "range":
                if fact.lower is not None:
                    projected.append(expression >= int(fact.lower))
                if fact.upper is not None:
                    projected.append(expression <= int(fact.upper))
            elif fact.kind == "divisible":
                projected.append(self.dsl.ixs_eq(
                    self.dsl.mod(expression, int(fact.divisor)),
                    0,
                ))
        return tuple(projected)

    def _pullback(self, source_id, blob, point, facts):
        source_point = self._source_point(source_id, blob, point)
        if source_point is None:
            return None
        return self._evaluate(source_id, source_point, facts)

    def _pullback_join(self, op, blob, point, facts):
        packet_slots = sum(int(self.program.values[int(operand)].type.component_count) for operand in op.operands)
        source_point = self._source_point_with_slots(
            packet_slots,
            int(op.operands[0]),
            blob,
            point,
        )
        if source_point is None:
            return None
        start = 0
        for operand in op.operands:
            count = int(self.program.values[int(operand)].type.component_count)
            inside = (source_point.slot >= start) & (source_point.slot < start + count)
            if self._proves(inside, facts):
                local = _Point(
                    source_point.block,
                    source_point.item,
                    source_point.slot - start,
                )
                return self._evaluate(int(operand), local, facts)
            start += count
        return None

    def _source_point(self, source_id, blob, point):
        return self._source_point_with_slots(
            int(self.program.values[int(source_id)].type.component_count),
            int(source_id),
            blob,
            point,
        )

    def _source_point_with_slots(self, source_slots, source_id, blob, point):
        if not blob or source_slots <= 0:
            return None
        value = self.program.values[int(source_id)]
        if value.layout_map_id is None:
            return None
        layout = self.program.layouts[int(value.layout_map_id)]
        warp_count = _linear_input_extent(layout.linear_layout, "warp")
        lane_width = int(value.type.lane_width or layout.lane_width or 0)
        source_items = lane_width * warp_count
        if source_items <= 0:
            return None
        packed = self._substitute_point(
            self.dsl.ixs_deserialize(blob),
            point,
        )
        return _Point(
            self.dsl.floor(packed / (source_slots * source_items)),
            self.dsl.mod(
                self.dsl.floor(packed / source_slots),
                source_items,
            ),
            self.dsl.mod(packed, source_slots),
        )

    def _equivalent_relations(self, relations, facts):
        if not relations or any(relation is None for relation in relations):
            return None
        bindings = _merge_bindings(*relations)
        if bindings is None:
            return None
        first = relations[0]
        common_constraints = tuple(constraint for constraint in first.constraints if all(
            any(self.dsl.ixs_serialize(constraint) == self.dsl.ixs_serialize(other)
                for other in relation.constraints)
            for relation in relations[1:]))
        goals = tuple(self.dsl.ixs_eq(first.expr, relation.expr) for relation in relations[1:])
        if goals:
            proofs, _ = self.dsl.ixs_check(
                goals,
                (*facts, *common_constraints),
            )
            if any(proof is not True for proof in proofs):
                return None
        return self._normalized(
            first.expr,
            bindings,
            facts,
            common_constraints,
        )

    def _normalized(self, expr, bindings, facts, constraints=()):
        constraints = tuple(constraints)
        _, normalized = self.dsl.ixs_check(
            (expr, ),
            (*facts, *constraints),
        )
        return _Relation(normalized[0], tuple(bindings), constraints)

    def _proves(self, predicate, facts):
        proofs, _ = self.dsl.ixs_check((predicate, ), facts)
        return proofs[0] is True

    def _substitute_point(self, expr, point):
        fresh = _Point(
            self.dsl.sym("__wave_query_block"),
            self.dsl.sym("__wave_query_item"),
            self.dsl.sym("__wave_query_slot"),
        )
        rename_facts = tuple(
            self.dsl.ixs_eq(source, replacement) for source, replacement in zip(
                (self.block, self.item, self.slot),
                (fresh.block, fresh.item, fresh.slot),
            ))
        _, renamed = self.dsl.ixs_check((expr, ), rename_facts)
        specialize_facts = tuple(
            self.dsl.ixs_eq(source, replacement) for source, replacement in zip(
                (fresh.block, fresh.item, fresh.slot),
                (point.block, point.item, point.slot),
            ))
        _, specialized = self.dsl.ixs_check(renamed, specialize_facts)
        return specialized[0]

    def _relation_key(self, target_value_id, point):
        return (
            int(target_value_id),
            *(self.dsl.ixs_serialize(expr) for expr in (
                point.block,
                point.item,
                point.slot,
            )),
        )

    def _constant_bool(self, target_value_id):
        op = self.definition_by_result.get(int(target_value_id))
        if op is None or op.kind != "constant":
            return None
        literal = target_ir.attrs_dict(op).get("value")
        return literal if type(literal) is bool else None


def attach_symbolic_memory_relations(target_program):
    """Attach one exact offset map to every global-memory packet operation."""
    analysis = _Analysis(target_program)
    ops = []
    changed = False
    for op in target_program.ops:
        relation = analysis.memory_relation(op)
        if relation is None:
            if op.kind in _MEMORY_OFFSET_OPERAND:
                offset_index = _MEMORY_OFFSET_OPERAND[op.kind]
                offset_id = (int(op.operands[offset_index]) if len(op.operands) > offset_index else None)
                fail(
                    "TLXW_RELATION_UNREPRESENTABLE_OFFSET",
                    STAGE,
                    "global-memory offset SSA chain has no exact symbolic relation",
                    source_op_index=op.source_op_index,
                    source_value_id=offset_id,
                    target_op_id=op.target_op_id,
                    target_value_id=offset_id,
                )
            ops.append(op)
            continue
        attrs = target_ir.attrs_dict(op)
        attrs.update({
            "bit_offset_relation": tuple(int(byte) for byte in analysis.dsl.ixs_serialize(relation.expr)),
            "index_binding_count": len(relation.bindings),
            "index_binding_names": tuple(name for name, _target_id in relation.bindings),
        })
        dependency_count = int(attrs.get("barrier_order_dependency_count", 0))
        if op.kind == "buffer_load_to_local":
            dependency_count += int(attrs.get("issue_dependency_count", 0))
        split = len(op.operands) - dependency_count
        binding_ids = tuple(target_id for _name, target_id in relation.bindings)
        ops.append(
            replace(
                op,
                operands=(*op.operands[:split], *binding_ids, *op.operands[split:]),
                attrs=target_ir._attrs_tuple(attrs, op.target_op_id),
            ))
        changed = True
    if not changed:
        return target_program
    return replace(target_program, ops=tuple(ops))


def _merge_bindings(*relations):
    merged = {}
    for relation in relations:
        for name, target_id in relation.bindings:
            previous = merged.setdefault(str(name), int(target_id))
            if previous != int(target_id):
                return None
    return tuple(sorted(merged.items()))


def _range_facts(dsl, expr, lower, upper):
    return (
        dsl.ixs_eq(expr, dsl.floor(expr)),
        expr >= int(lower),
        expr <= int(upper),
    )


def _linear_input_extent(linear, name):
    if linear is None:
        return 0
    for dim_name, bases in linear.bases:
        if str(dim_name) == str(name):
            return 1 << len(bases)
    return 1


def _element_bits(element_type):
    if element_type == "index":
        return 64
    if not isinstance(element_type, str) or not element_type.startswith("i"):
        return None
    try:
        width = int(element_type[1:])
    except ValueError:
        return None
    return width if width > 0 else None


def _unsigned_fixed(dsl, expr, width):
    if width <= 0 or width >= 63:
        raise ValueError("fixed-width unsigned relation requires 1..62 bits")
    return dsl.mod(expr, 1 << width)


def _signed_fixed(dsl, expr, width):
    if width <= 0 or width >= 63:
        raise ValueError("fixed-width signed relation requires 1..62 bits")
    bias = 1 << (width - 1)
    return dsl.mod(expr + bias, 1 << width) - bias
