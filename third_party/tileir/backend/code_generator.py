import ast
import inspect
import re
from typing import Dict, Optional
import warnings

import triton
import triton.knobs as knobs
import triton.language as language
from triton.language import constexpr
from triton._utils import (
    find_paths_if,
    get_iterable_path,
)
from triton.language.core import _unwrap_if_constexpr, base_value, base_type

from triton.compiler.code_generator import (
    _is_list_like,
    _is_constexpr,
    _is_triton_tensor,
    _unwrap_if_constexpr,
    ASTFunction,
    CodeGenerator,
    enter_sub_region,
    flatten_values_to_ir,
    unflatten_ir_values,
)
from triton.compiler.errors import CompilationError
from triton.runtime.jit import (
    get_jit_fn_file_line,
    get_full_name,
    JITFunction,
    JITCallable,
)

from triton.backends.tileir.conf import TileIREnvConf
from .fb.language import tileir_tensor_descriptor_type


def mangle_fn(name, arg_tys, caller_context):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_args = '_'.join([tileir_mangle_ty(ty) for ty in arg_tys])
    mangled_args = mangled_args.replace("'", '_sq_')
    # [ and ] are not allowed in LLVM identifiers
    mangled_args = mangled_args.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_args}'
    if caller_context is not None:
        ret += caller_context.mangle()
    return ret


def tileir_mangle_ty(ty):
    return ty.mangle()


def tileir_mangle_fn(name, arg_tys, constants):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_arg_names = "_".join([tileir_mangle_ty(ty) for ty in arg_tys])
    mangled_constants = "_".join([f"{i}c{repr(constants[i])}" for i in sorted(constants)])
    mangled_constants = mangled_constants.replace(".", "_d_")
    mangled_constants = mangled_constants.replace("'", "_sq_")
    # [ and ] are not allowed in LLVM identifiers
    mangled_constants = mangled_constants.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_arg_names}__{mangled_constants}'
    return ret



# TODO: FIXME HACK: META INTEGRATION CODE GENERATOR.
# TileIRCodeGenerator, str_to_ty, and ast_to_ttir provide the Meta-specific
# code generation path for the TileIR backend. These override the default
# Triton code generator to handle TileIR-specific types (e.g. tensordesc)
# and plug into the ast_to_ttir property on TileIROptions.

class TileIRCodeGenerator(CodeGenerator):

    def __init__(
        self,
        context,
        prototype,
        gscope,
        function_name,
        jit_fn: JITFunction,
        options,
        codegen_fns,
        module_map,
        is_gluon=False,
        module=None,
        is_kernel=False,
        function_types: Optional[Dict] = None,
        noinline=False,
        file_name: Optional[str] = None,
        begin_line=0,
    ):
        super().__init__(
            context=context,
            prototype=prototype,
            gscope=gscope,
            function_name=function_name,
            jit_fn=jit_fn,
            options=options,
            codegen_fns=codegen_fns,
            module_map=module_map,
            is_gluon=is_gluon,
            module=module,
            is_kernel=is_kernel,
            function_types=function_types,
            noinline=noinline,
            file_name=file_name,
            begin_line=begin_line,
        )

    def get_used_vars(self, stmt):
        used_vars = dict()
        for node in ast.walk(stmt):
            if isinstance(node, ast.FunctionDef):
                continue
            if isinstance(node, ast.With):
                continue
            if isinstance(node, ast.AugAssign):
                used_vars[node.target.id] = node.target
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars[node.id] = node
        return used_vars

    def call_JitFunction(self, fn: JITFunction, args, kwargs):
        args = inspect.getcallargs(fn.fn, *args, **kwargs)
        args = [args[name] for name in fn.arg_names]
        for i, arg in enumerate(args):
            if isinstance(arg, (language.dtype, float, int, bool)):
                args[i] = language.core.constexpr(arg)
        args_cst = find_paths_if(args, lambda _, x: _is_constexpr(x))
        args_cst = {path: get_iterable_path(args, path) for path in args_cst}
        args_path = find_paths_if(args, lambda _, x: not _is_constexpr(x))
        args_val = [get_iterable_path(args, path) for path in args_path]
        # mangle
        fn_name = tileir_mangle_fn(
            get_full_name(fn), [arg.type for arg in args_val], args_cst
        )
        # generate function def if necessary
        if not self.module.has_function(fn_name):
            # If the callee is not set, we use the same debug setting as the caller
            file_name, begin_line = get_jit_fn_file_line(fn)
            arg_types = [
                language.core.constexpr if arg is None or isinstance(arg,
                                                                     (bool, int, language.core.dtype)) else arg.type
                for arg in args
            ]
            prototype = ASTFunction([], arg_types, args_cst, dict())
            # TileIR backend does not support noinline mode currently
            if fn.noinline:
                import warnings

                warnings.warn(
                    "Current backend does not support noinline mode, noinline will be turn off.",
                    RuntimeWarning,
                )
                fn.noinline = False
            generator = TileIRCodeGenerator(
                self.context,
                prototype,
                fn.get_capture_scope(),
                module=self.module,
                jit_fn=fn,
                function_name=fn_name,
                function_types=self.function_ret_types,
                noinline=fn.noinline,
                file_name=file_name,
                begin_line=begin_line,
                options=self.builder.options,
                codegen_fns=self.builder.codegen_fns,
                module_map=self.builder.module_map,
                is_gluon=False,
            )
            try:
                generator.visit(fn.parse())
            except Exception as e:
                # Wrap the error in the callee with the location of the call.
                if knobs.compilation.front_end_debugging:
                    raise
                raise CompilationError(self.jit_fn.src, self.cur_node, None) from e

            callee_ret_type = generator.ret_type
            self.function_ret_types[fn_name] = callee_ret_type
        else:
            callee_ret_type = self.function_ret_types[fn_name]
        symbol = self.module.get_function(fn_name)
        args_val = flatten_values_to_ir(args_val)
        call_op = self.builder.call(symbol, args_val)
        if callee_ret_type == language.void:
            return None
        handles = [call_op.get_result(i) for i in range(call_op.get_num_results())]
        return next(unflatten_ir_values(handles, [callee_ret_type]))


def str_to_ty(name, c):
    from builtins import tuple
    from triton.language.core import (
        pointer_type,
        tuple_type,
        block_type,
        int32,
        int64,
    )

    # Ensure we recurse properly to this implementation.
    if isinstance(name, tuple):
        fields = type(name).__dict__.get("_fields", None)
        return tuple_type([str_to_ty(x, c) for x in name], fields)

    if name[0] == "*":
        name = name[1:]
        const = False
        if name[0] == "k":
            name = name[1:]
            const = True
        ty = str_to_ty(name, c)
        return pointer_type(element_ty=ty, const=const)

    if name.startswith("tensordesc"):
        inner = name.split("<")[1].rstrip(">")
        dtype, rest = inner.split("[", maxsplit=1)
        block_shape, rest = rest.split("]", maxsplit=1)
        block_shape = [int(s.strip()) for s in block_shape.rstrip("]").split(",")]
        dtype = str_to_ty(dtype, None)
        ndim = len(block_shape)
        shape_type = tuple_type([int32] * ndim)
        stride_type = tuple_type(([int64] * ndim))
        block = block_type(dtype, block_shape)
        return tileir_tensor_descriptor_type(block, shape_type, stride_type)

    # Fall back to language's default for non-tensor descriptor types.
    return language.str_to_ty(name, c)


def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    arg_types = [None] * len(fn.arg_names)
    const_iter = iter(src.constants.items())
    kc, vc = next(const_iter, (None, None))

    for i, (ks, v) in enumerate(src.signature.items()):
        idx = fn.arg_names.index(ks)
        cexpr = None
        if kc is not None and kc[0] == i:
            cexpr = vc
            kc, vc = next(const_iter, (None, None))
        arg_types[idx] = str_to_ty(v, cexpr)

    prototype = ASTFunction([], arg_types, src.constants, src.attrs)
    file_name, begin_line = get_jit_fn_file_line(fn)
    # query function representation
    from collections import namedtuple
    leaves = filter(lambda v: len(v) == 1, src.constants)
    constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
    signature = src.signature

    tileir_additional_suffix = ""
    proxy = namedtuple("SpecializationProxy", ["constants", "signature",])(constants, signature)
    generator = TileIRCodeGenerator(
        context,
        prototype,
        gscope=fn.get_capture_scope(),
        function_name=fn.repr(proxy) + tileir_additional_suffix,
        jit_fn=fn,
        is_kernel=True,
        file_name=file_name,
        begin_line=begin_line,
        options=options,
        codegen_fns=codegen_fns,
        module_map=module_map,
        is_gluon=False,
    )
    generator.visit(fn.parse())

    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    ret.name = generator.function_name
    return ret
