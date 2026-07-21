"""MLIR import stage for the from-scratch TLX Wave converter."""

import re
from dataclasses import replace

from .diagnostics import fail
from .source_ir import (
    KernelInfo,
    SourceOp,
    SourceProgram,
    SourceRegion,
    SourceType,
    SourceValue,
    TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZATION_ATTR,
    TLX_WAVE_ENABLE_SPLIT_BARRIERS_ATTR,
)

STAGE = "import"


def import_source_program(
    mod,
    kernel_name=None,
    *,
    compiler_membar_barriers=(),
):
    compiler_membar_barriers = frozenset(compiler_membar_barriers)
    kernel_name = _resolve_kernel_name(mod, kernel_name)
    fn = mod.get_function(kernel_name)
    if fn is None:
        fail(
            "TLXW_IMPORT_KERNEL_NOT_FOUND",
            STAGE,
            f"cannot find tt.func @{kernel_name}",
        )
    func_op = _kernel_func_op(mod, kernel_name)
    if func_op is None:
        fail(
            "TLXW_IMPORT_KERNEL_NOT_FOUND",
            STAGE,
            f"cannot find tt.func op @{kernel_name}",
        )

    arg_attrs = _function_arg_attrs(func_op)
    values = {}
    for index in range(fn.get_num_args()):
        value = fn.args(index)
        value_id = _value_id(value)
        source_type = _source_type(value.get_type())
        source_type = replace(
            source_type,
            pointer_range=_arg_int_attr(
                arg_attrs[index] if index < len(arg_attrs) else None,
                "tt.pointer_range",
            ),
            divisibility=_arg_int_attr(
                arg_attrs[index] if index < len(arg_attrs) else None,
                "tt.divisibility",
            ),
        )
        _require_supported_source_type(source_type, value_id)
        values[value_id] = SourceValue(
            value_id,
            source_type,
            producer_name=f"arg{index}",
            argument_index=index,
        )

    ops = []
    regions = []
    top_region_id = _collect_region(
        fn.get_region(0),
        ops,
        regions,
        values,
        parent_op_index=None,
        region_index=None,
        compiler_membar_barriers=compiler_membar_barriers,
    )

    kernel = KernelInfo(
        kernel_name,
        _module_str_attr(mod, "ttg.target"),
        _module_int_attr(mod, "ttg.num-ctas"),
        _module_int_attr(mod, "ttg.num-warps"),
        _module_int_attr(mod, "ttg.threads-per-warp"),
        func_op.get_bool_attr("noinline"),
        tuple(_value_id(fn.args(index)) for index in range(fn.get_num_args())),
        bool(func_op.get_bool_attr(TLX_WAVE_ENABLE_SPLIT_BARRIERS_ATTR)),
        bool(
            func_op.get_bool_attr(
                TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZATION_ATTR
            )
        ),
    )
    return SourceProgram(
        kernel,
        tuple(ops),
        values,
        tuple(regions),
        top_region_id,
    )


def _resolve_kernel_name(mod, kernel_name):
    if kernel_name is not None:
        return kernel_name
    funcs = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_visibility") == "public":
            funcs.append(op)
        return True

    mod.walk(visit)
    if len(funcs) != 1:
        names = (", ".join(func.get_str_attr("sym_name") or "<unnamed>" for func in funcs) or "none")
        fail(
            "TLXW_IMPORT_KERNEL_COUNT",
            STAGE,
            "expected exactly one public tt.func kernel, "
            f"found {len(funcs)} ({names})",
        )
    return funcs[0].get_str_attr("sym_name")


def _kernel_func_op(mod, kernel_name):
    found = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_name") == kernel_name:
            found.append(op)
        return True

    mod.walk(visit)
    return found[0] if found else None


def _collect_region(
    region,
    ops,
    regions,
    values,
    *,
    parent_op_index,
    region_index,
    compiler_membar_barriers,
):
    region_id = len(regions)
    regions.append(None)
    op_indices = []
    block_arg_ids = []
    for block_index in range(region.size()):
        block = region.get_block(block_index)
        if block_index != 0:
            fail(
                "TLXW_IMPORT_MULTI_BLOCK_REGION",
                STAGE,
                "multi-block regions are not supported yet",
                source_op_index=parent_op_index,
            )
        for arg_index in range(block.get_num_arguments()):
            arg = block.get_argument(arg_index)
            value_id = _value_id(arg)
            block_arg_ids.append(value_id)
            if value_id in values:
                continue
            source_type = _source_type(arg.get_type())
            _require_supported_source_type(source_type, value_id)
            values[value_id] = SourceValue(
                value_id,
                source_type,
                producer_name="block_argument",
                region_id=region_id,
                region_arg_index=arg_index,
            )
        for block_op_index in range(block.get_num_operations()):
            op = block.get_operation(block_op_index)
            op_index = len(ops)
            ops.append(None)
            op_indices.append(op_index)
            child_region_ids = tuple(
                _collect_region(
                    op.get_region(child_index),
                    ops,
                    regions,
                    values,
                    parent_op_index=op_index,
                    region_index=child_index,
                    compiler_membar_barriers=compiler_membar_barriers,
                ) for child_index in range(op.get_num_regions()))
            attrs = _source_attrs(op)
            if op in compiler_membar_barriers:
                attrs["tlx.compiler_membar_barrier"] = True
            source_op = SourceOp(
                op_index,
                op.get_name(),
                tuple(_value_id(op.get_operand(i)) for i in range(op.get_num_operands())),
                tuple(_value_id(op.get_result(i)) for i in range(op.get_num_results())),
                attrs,
                child_region_ids,
                region_id,
                parent_op_index,
                region_index,
            )
            ops[op_index] = source_op
            for result_index in range(op.get_num_results()):
                result = op.get_result(result_index)
                value_id = _value_id(result)
                source_type = _source_type(result.get_type())
                _require_supported_source_type(source_type, value_id)
                values[value_id] = SourceValue(
                    value_id,
                    source_type,
                    owner_op_index=op_index,
                    producer_name=op.get_name(),
                )
    regions[region_id] = SourceRegion(
        region_id,
        tuple(op_indices),
        tuple(block_arg_ids),
        parent_op_index,
        region_index,
    )
    return region_id


def _source_attrs(op):
    attrs = dict(op.get_attrs())
    for name in ("axis", "end", "num", "predicate", "start"):
        value = op.get_int_attr(name)
        if value is not None:
            attrs[name] = int(value)
    for name in ("offsets", "operandSegmentSizes", "order"):
        values = op.get_int_array_attr(name)
        if values is not None:
            attrs[name] = tuple(int(value) for value in values)
    dimension = attrs.get("dimension")
    if dimension is not None:
        match = re.fullmatch(r"#gpu<dim ([xyz])>", str(dimension))
        if match is not None:
            attrs["axis"] = {"x": 0, "y": 1, "z": 2}[match.group(1)]
    return attrs


def _function_arg_attrs(func_op):
    attrs = dict(func_op.get_attrs()).get("arg_attrs")
    return () if attrs is None else tuple(attrs)


def _arg_int_attr(attrs, name):
    if attrs is None:
        return None
    if isinstance(attrs, str):
        match = re.search(rf"{re.escape(name)}\s*=\s*([+-]?\d+)\s*:", attrs)
        return None if match is None else int(match.group(1))
    attr = dict(attrs).get(name)
    if attr is None:
        return None
    try:
        return int(attr)
    except TypeError:
        return int(str(attr))


def _source_type(type_obj):
    element_type = _type_method(type_obj, "get_element_type")
    pointee_type = _type_method(type_obj, "get_pointee_type")
    encoding_attr = _type_method(type_obj, "get_encoding")
    if pointee_type is None and element_type is not None:
        pointee_type = _type_method(element_type, "get_pointee_type")
    address_space = _type_method(type_obj, "get_address_space")
    if address_space is None and element_type is not None:
        address_space = _type_method(element_type, "get_address_space")
    element_byte_width = _scalar_byte_width(element_type)
    if element_byte_width is None:
        element_byte_width = _scalar_byte_width(pointee_type)
    if element_byte_width is None and _is_scalar_type(type_obj):
        element_byte_width = _scalar_byte_width(type_obj)
    return SourceType(
        _type_str(type_obj),
        _source_type_kind(type_obj),
        _tuple_or_empty(_type_method(type_obj, "get_shape")),
        _type_str(element_type) if element_type is not None else None,
        element_byte_width,
        _type_str(pointee_type) if pointee_type is not None else None,
        _attr_str(encoding_attr),
        encoding_attr,
        _attr_str(_type_method(type_obj, "get_memory_space")),
        _type_method(type_obj, "get_mutable_memory"),
        _tuple_or_empty(_type_method(type_obj, "get_alloc_shape")),
        address_space,
    )


def _source_type_kind(type_obj):
    if _type_predicate(type_obj, "is_memdesc"):
        return "memdesc"
    if _type_predicate(type_obj, "is_ranked_tensor"):
        return "tensor"
    if _type_predicate(type_obj, "is_ptr"):
        return "pointer"
    if _type_predicate(type_obj, "is_async_token"):
        return "token"
    if _is_scalar_type(type_obj):
        return "scalar"
    return "other"


def _require_supported_source_type(source_type, value_id):
    if source_type.kind != "other":
        return
    fail(
        "TLXW_IMPORT_UNSUPPORTED_TYPE",
        STAGE,
        f"unsupported source type {source_type.raw}",
        source_value_id=value_id,
    )


def _is_scalar_type(type_obj):
    return (_type_predicate(type_obj, "is_index") or _type_predicate(type_obj, "is_fp16")
            or _type_predicate(type_obj, "is_bf16") or _type_predicate(type_obj, "is_fp32")
            or _type_predicate(type_obj, "is_fp64")
            or any(_type_is_integer_width(type_obj, width) for width in (1, 8, 16, 32, 64)))


def _scalar_byte_width(type_obj):
    if type_obj is None:
        return None
    if _type_is_integer_width(type_obj, 1) or _type_is_integer_width(type_obj, 8):
        return 1
    if (_type_is_integer_width(type_obj, 16) or _type_predicate(type_obj, "is_fp16")
            or _type_predicate(type_obj, "is_bf16")):
        return 2
    if _type_is_integer_width(type_obj, 32) or _type_predicate(type_obj, "is_fp32"):
        return 4
    if (_type_is_integer_width(type_obj, 64) or _type_predicate(type_obj, "is_fp64")
            or _type_predicate(type_obj, "is_index")):
        return 8
    return None


def _type_is_integer_width(type_obj, width):
    fn = getattr(type_obj, "is_integer", None)
    if fn is None:
        return False
    return bool(fn(width))


def _type_predicate(type_obj, method):
    return bool(_type_method(type_obj, method, False))


def _type_method(type_obj, method, default=None):
    fn = getattr(type_obj, method, None)
    if fn is None:
        return default
    return fn()


def _tuple_or_empty(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _type_str(type_obj):
    return str(type_obj)


def _attr_str(attr):
    return None if attr is None else str(attr)


def _module_str_attr(mod, name):
    return mod.get_operation().get_str_attr(name)


def _module_int_attr(mod, name):
    return mod.get_operation().get_int_attr(name)


def _value_id(value):
    return int(value.id())
