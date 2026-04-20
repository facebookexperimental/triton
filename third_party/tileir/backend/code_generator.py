import ast
import inspect
import re
from typing import Dict, Optional
import warnings

import triton
import triton.knobs as knobs
import triton.language as language
from triton.language import constexpr, str_to_ty
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
    mangled_constants = "_".join(
        [f"{i}c{repr(constants[i])}" for i in sorted(constants)]
    )
    mangled_constants = mangled_constants.replace(".", "_d_")
    mangled_constants = mangled_constants.replace("'", "_sq_")
    # [ and ] are not allowed in LLVM identifiers
    mangled_constants = mangled_constants.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_arg_names}__{mangled_constants}'
    return ret

