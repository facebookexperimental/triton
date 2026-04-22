from __future__ import annotations, division
import ast
import copy
import hashlib
import inspect
import itertools
import os
import threading
import re
import textwrap
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Generic, Iterable, Optional, TypeVar, overload, Dict, Any, Tuple

from triton.backends import BaseBackend
from types import ModuleType
from .. import knobs
from .driver import driver
from . import _async_compile
from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, is_namedtuple
from .cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars, native_specialize_impl

TRITON_MODULE = "triton.language"
GLUON_MODULE = "triton.experimental.gluon.language"

# Structured cache entry for the Layer 1 identity-based fast path.
# A namedtuple is used so guard code can use readable .field access while
# the hot launch path uses fast [index] access (same cost as plain tuple).
_LastCall = namedtuple("_LastCall", [
    "device",
    "args",
    "kernel",
    "bound_vals",
    "launch_fn",
    "function",
    "packed_metadata",
    "coop",
    "cluster",
    "pdl",
    "no_scratch",
    "instrumentation_mode",
])

T = TypeVar("T")


class _LRUDict:
    """Thread-safe bounded LRU cache backed by OrderedDict.

    Used for kernel_cache and _run_cache to prevent unbounded CUDA memory
    growth when kernels are compiled for many distinct shapes.

    Each public method is guarded by a lock so that compound operations
    (e.g. move_to_end + __getitem__) are atomic even when multiple threads
    invoke a JITFunction concurrently.
    """

    __slots__ = ("_maxsize", "_data", "_lock", "_on_evict")

    def __init__(self, maxsize: int = 256, on_evict=None):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._on_evict = on_evict

    def get(self, key):
        with self._lock:
            try:
                self._data.move_to_end(key)
                return self._data[key]
            except KeyError:
                return None

    def put(self, key, value):
        if self._maxsize <= 0:
            return
        with self._lock:
            try:
                self._data.move_to_end(key)
            except KeyError:
                if len(self._data) >= self._maxsize:
                    evicted_key, evicted_val = self._data.popitem(last=False)
                    if self._on_evict is not None:
                        self._on_evict(evicted_key, evicted_val)
            self._data[key] = value

    def clear(self):
        with self._lock:
            if self._on_evict is not None:
                for k, v in self._data.items():
                    self._on_evict(k, v)
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def values(self):
        with self._lock:
            return list(self._data.values())

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __deepcopy__(self, memo):
        with self._lock:
            new = _LRUDict(self._maxsize)
            new._data = OrderedDict(self._data)
            return new

    def __copy__(self):
        with self._lock:
            new = _LRUDict(self._maxsize)
            new._data = OrderedDict(self._data)
            return new


class _DeferredModuleUnloader:
    """Deferred queue for cuModuleUnload calls.

    CUDA modules cannot be unloaded during CUDA graph capture (it would
    invalidate CUfunction handles referenced by the graph).  Instead,
    evicted module handles are queued here and flushed at the next kernel
    launch when the stream is NOT in capture mode.
    """

    __slots__ = ("_queue", "_lock")

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()

    def queue_module(self, module_handle):
        if module_handle is not None and module_handle != 0:
            with self._lock:
                self._queue.append(module_handle)

    def flush(self, stream):
        if not self._queue:
            return
        try:
            if driver.active.utils.is_stream_capturing(stream):
                return
        except Exception:
            return
        with self._lock:
            handles = list(self._queue)
            self._queue.clear()
        for handle in handles:
            try:
                driver.active.utils.unload_binary(handle)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Dependencies Finder
# -----------------------------------------------------------------------------


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.

    This visitor also keeps track of the global variables touched by the
    JITFunction.  When we launch the kernel, we check that these have the same
    values as they did when we ran this visitor.  If not, we raise an error (or
    otherwise we could recompile).
    """

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        # This function's __globals__ dict.
        self.globals = globals
        self.nonlocals = nonlocals

        # Python builtins that can be accessed from Triton kernels.
        self.supported_python_builtins = {
            'float',
            'getattr',
            'int',
            'isinstance',
            'len',
            'list',
            'max',
            'min',
            'print',
            'range',
        }
        self.supported_modules = {
            GLUON_MODULE,
            TRITON_MODULE,
            "copy",
            "math",
        }

        # used_global_vals tells us which global variables are used by this
        # function and all those it transitively calls, plus the values of those
        # variables when each function was initially run.  (That is, if A calls
        # C, and B calls C, then the values for C in used_global_vals will be
        # from the first time C was run, either by A or B.)
        #
        # Each function may have a different __globals__ dict, so the global
        # variable `foo` may actually have a different value in the different
        # functions.  Thus this map is actually
        #  (var_name, id(__globals__)) -> (var_value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.visiting_arg_default_value = False

    @property
    def ret(self):
        return self.hasher.hexdigest()

    def _is_triton_builtin(self, node, func):
        if inspect.isbuiltin(node.func):
            return True
        module = getattr(func, "__module__", "")
        return module.startswith(TRITON_MODULE)

    def _update_hash(self, func):
        assert isinstance(func, JITCallable)
        # Merge our used_global_vals with those of the called function,
        # after checking that all overlapping values are consistent.
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} when compiling {self.name}, but inner kernel {func.__name__} has conflicting value {v2} from when it was first compiled.  This is not allowed."
                )
        self.used_global_vals.update(func.used_global_vals)
        # update hash
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

    def record_reference(self, val, var_dict=None, name=None):
        from ..language.core import constexpr
        # Only keep track of "interesting" global variables, that non-evil users
        # might change.  Don't consider functions, modules, builtins, etc.  This
        # helps keep the list of vars we have to check small.
        if val is None or type(val) is ModuleType:
            return

        if getattr(val, "__triton_aggregate__", False):
            for attr in val.hash_attrs:
                self.record_reference(attr)
            return

        if getattr(val, "__triton_builtin__", False):
            return

        # Stubs that aren't real functions
        if getattr(val, "__module__", "") == "triton.language.extra.libdevice":
            return

        if isinstance(val, JITCallable):
            self._update_hash(val)
            return

        if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
            raise RuntimeError(f"Unsupported function referenced: {val}")

        # Python default arguments are resolved only once, when the
        # function is defined.  So if you do `foo(a=A)` and the value of
        # A changes, foo will still use the old value of A.
        # It would be pretty evil if someone did `import x` and then
        # `x = blah`.
        if self.visiting_arg_default_value:
            return

        if var_dict is not None:
            self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
        return

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id

        if node.id in self.local_names:
            # The global name is hidden by the local name.
            return None

        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)
        if node.id in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, node.id)
        return val

    def visit_Tuple(self, node):
        # We need to explicitly return the tuple values so that visit_Assign can
        # access them in the case of `a, b = ...`.
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr)
        self.record_reference(ret)
        return ret

    def visit_FunctionDef(self, node):
        # Save the local name, which may hide the global name.
        self.local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)

    def visit_arguments(self, node):
        # The purpose of this function is to visit everything in `arguments`
        # just like `generic_visit`, except when we're visiting default values
        # (i.e. the `foo` part of `def fn(x = foo)`), we set
        # self.visiting_arg_default_value = True.  This allows visit_Name to be
        # aware that we're inside function default values, which have special
        # semantics.

        # According to the AST docs, the arguments node has the following structure.
        #
        # arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #              expr* kw_defaults, arg? kwarg, expr* defaults)
        def visit_defaults(defaults):
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(node.posonlyargs, node.args, [node.vararg] if node.vararg else [], node.kwonlyargs):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        # Target is either a single string, or a list of strings (if the assn
        # target is a tuple).
        target = self.visit(node)
        if isinstance(target, list):
            self.local_names |= set(target)
        else:
            self.local_names.add(target)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            # TODO(jlebar): I don't actually know how to hit this.  You don't
            # get it from `a, b = ...` -- in that case, node.targets is a single
            # Tuple, and in fact we *do* need to handle that case if we want
            # existing code to work.
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_For(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's fine.
        self.generic_visit(node)


# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


def _normalize_ty(ty) -> str:
    import triton.language.core as core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return _normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)


class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num: int, param: inspect.Parameter, do_not_specialize: bool,
                 do_not_specialize_on_alignment: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self) -> str:
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def annotation_type(self) -> str:
        a = self.annotation
        if a.startswith("*k"):
            a = a[2:]
        elif a.startswith("*"):
            a = a[1:]
        if a in set(type_canonicalisation_dict.values()):
            return self.annotation
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        if self.is_constexpr:
            return False
        return "const" in self.annotation or self.annotation.startswith("*k")

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


def mangle_type(arg, specialize=False):
    is_const = False
    align = True
    return native_specialize_impl(BaseBackend, arg, is_const, specialize, align)[0]


class KernelInterface(Generic[T]):
    run: T

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        raise NotImplementedError("run not implemented")

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
        # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))


def serialize_specialization_data(name, signature, constants, attrs, options, key):
    constants = {
        key: str(value) if value.__class__.__name__ == "dtype" else
        {"constexpr": value.value} if value.__class__.__name__ == "constexpr" else value
        for key, value in constants.items()
    }

    import json
    obj = {
        'name': name, 'signature': signature, 'constant_keys': [list(x) for x in constants.keys()], 'constant_vals':
        list(constants.values()), 'attrs_keys': [list(x) for x in attrs.keys()], 'attrs_vals': list(attrs.values()),
        'options': options.__dict__, 'key': key
    }
    serialized_obj = json.dumps(obj)
    return serialized_obj


def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    arg = lambda x: x[0] if x[1].default is inspect.Parameter.empty else f"{x[0]}=default_{x[0]}"
    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    specialize_impl = native_specialize_impl
    func_namespace["specialize_impl"] = specialize_impl
    func_namespace["backend"] = backend
    func_namespace["JITCallable"] = JITCallable

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"


class JITCallable:

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        # function source code (without decorators)
        src = textwrap.dedent("".join(self.raw_src))
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src
        self.hash = None

        # Map of global variables used by the function and any functions it
        # transitively calls, plus their values.  The values are collected when
        # the function is first compiled.  Then every time we run the function,
        # we check that the values of the globals match what's expected,
        # otherwise we raise an error.
        #
        # Different functions can have different __globals__ maps, so the map
        # key is actually (var name, id(__globals__)), and the map value is
        # (value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        return self.__globals__ | nonlocals

    @property
    def cache_key(self) -> str:
        # TODO : hash should be attribute of `self`
        with self._hash_lock:
            if self.hash is not None:
                return self.hash
            # Set a placeholder hash to break recursion in case the function
            # transitively calls itself. The full hash is set after.
            self.hash = f"recursion:{self._fn_name}"
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(name=self._fn_name, globals=self.__globals__, nonlocals=nonlocals,
                                                     src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            from triton.language.core import constexpr
            self.hash += str([(name, val)
                              for (name, _), (val, _) in self.used_global_vals.items()
                              if isinstance(val, constexpr)])
            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

    def __hash__(self):
        return hash(self.cache_key)

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    @property
    def type(self):
        from triton.language.core import constexpr_type
        return constexpr_type(self)

    def _unsafe_update_src(self, new_src):
        """
        The only method allowed to modify src.
        Bypasses the __setattr__ restriction by calling super().__setattr__ directly.

        Note that it is the callers responsibility to make sure any triton functions that call this function have the `.hash` value reset to None.
        """
        self.hash = None
        self._src = new_src

    def _set_src(self):
        raise AttributeError("Cannot set attribute 'src' directly. "
                             "Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
                             "instead.")

    def _get_src(self):
        return self._src

    src = property(fget=_get_src, fset=_set_src)


@dataclass
class JitFunctionInfo:
    module: ModuleType
    name: str
    jit_function: JITFunction


def compute_cache_key(kernel_key_cache, specialization, options):
    # TODO: Handle runtime knob swapping. This is currently too slow on the Python
    # critial path.
    # The original change was for testing, but we can invalidate caches explicitly if
    # tests break.
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


def convert_to_tuple_if_list(item):
    # If the incoming item is a list, recursively iterate through it to convert all lists therein into tuples
    if not isinstance(item, list):
        return item

    # The value must be a list at this point
    for i, nested_value in enumerate(item):
        item[i] = convert_to_tuple_if_list(nested_value)

    return tuple(item)


class _DeviceCaches(defaultdict):
    """A defaultdict that also invalidates the Layer 1 fast-path cache
    (``_last_call``) whenever the in-memory kernel cache is cleared.
    Without this, ``device_caches.clear()`` would wipe Layer 2 but
    leave a stale Layer 1 entry, causing the fast path to return a
    kernel that is no longer in the device cache."""

    def __init__(self, jit_function=None, default_factory=None):
        super().__init__(default_factory)
        self._jit_function = jit_function

    def clear(self):
        super().clear()
        if self._jit_function is not None:
            self._jit_function.clear_fast_path_caches()

    def __reduce__(self):
        # Return as a plain defaultdict for pickling/deepcopy.
        # The _jit_function back-reference is not meaningful in a copy.
        return (defaultdict, (self.default_factory, ), None, None, iter(self.items()))

    def __deepcopy__(self, memo):
        # Deepcopy as a plain defaultdict — the _jit_function
        # back-reference should not be copied.
        result = defaultdict(self.default_factory)
        memo[id(self)] = result
        for k, v in self.items():
            result[copy.deepcopy(k, memo)] = copy.deepcopy(v, memo)
        return result


class JITFunction(JITCallable, KernelInterface[T]):

    def is_gluon(self):
        return False

    def _call_hook(
        self,
        hook,
        key,
        signature,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ) -> bool | None:
        if not hook:
            return None

        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        # Build repr string, only including optional params when they're set
        repr_parts = [
            f"num_warps={options.num_warps}",
            f"num_ctas={options.num_ctas}",
            f"num_stages={options.num_stages}",
        ]
        # Use getattr to safely access backend-specific attributes
        minRegAutoWS = getattr(options, 'minRegAutoWS', None)
        maxRegAutoWS = getattr(options, 'maxRegAutoWS', None)
        pingpongAutoWS = getattr(options, 'pingpongAutoWS', None)
        if minRegAutoWS is not None:
            repr_parts.append(f"minRegAutoWS={minRegAutoWS}")
        if maxRegAutoWS is not None:
            repr_parts.append(f"maxRegAutoWS={maxRegAutoWS}")
        if pingpongAutoWS is not None:
            repr_parts.append(f"pingpongAutoWS={pingpongAutoWS}")
        repr_parts.extend([
            f"enable_fp_fusion={options.enable_fp_fusion}",
            f"launch_cooperative_grid={options.launch_cooperative_grid}",
        ])
        repr = f"{name}[{', '.join(repr_parts)}]({arg_reprs})"
        full_name = get_full_name(self.fn)

        specialization_data = serialize_specialization_data(full_name, signature, constants, configs[0], options, key)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'minRegAutoWS': getattr(options, 'minRegAutoWS', None),
            'maxRegAutoWS': getattr(options, 'maxRegAutoWS', None),
            'pingpongAutoWS': getattr(options, 'pingpongAutoWS', None),
            'enable_fp_fusion': options.enable_fp_fusion,
            'launch_cooperative_grid': options.launch_cooperative_grid,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
            'is_warmup': is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=is_warmup,
            already_compiled=False,
        )

    def add_pre_run_hook(self, hook):
        '''
        Add a hook that will be executed prior to the execution of run
        function with args and kwargs passed into the kernel
        '''
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def _compute_fast_key(self, args, kwargs, device):
        """Compute a minimal tuple that uniquely determines the compiled kernel.

        Returns None if the args contain types that can't be fast-path'd,
        or if the total number of args + kwargs doesn't match the param count.
        """
        if len(args) + len(kwargs) != len(self.params):
            return None
        parts = [device, knobs.compilation.instrumentation_mode]
        for i, arg in enumerate(args):
            kp = self.params[i]
            if kp.is_constexpr:
                parts.append(arg)
            elif arg is None:
                parts.append(None)
            elif type(arg) is bool:
                parts.append(bool)
            elif type(arg) is int:
                parts.append(arg)
            elif type(arg) is float:
                parts.append(float)
            elif hasattr(arg, 'data_ptr'):
                parts.append((arg.dtype, arg.data_ptr() % 16 == 0))
            elif hasattr(arg, 'tma_desc_cpu_ptr'):
                parts.append('tma')
            else:
                return None
        param_by_name = {p.name: p for p in self.params}
        for k, v in sorted(kwargs.items()):
            kp = param_by_name.get(k)
            if kp is not None and kp.is_constexpr:
                parts.append((k, v))
            elif v is None:
                parts.append((k, None))
            elif type(v) is bool:
                parts.append((k, bool))
            elif type(v) is int:
                parts.append((k, v))
            elif type(v) is float:
                parts.append((k, float))
            elif hasattr(v, 'data_ptr'):
                parts.append((k, v.dtype, v.data_ptr() % 16 == 0))
            elif hasattr(v, 'tma_desc_cpu_ptr'):
                parts.append((k, 'tma'))
            else:
                return None
        return tuple(parts)

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from ..compiler import CompiledKernel, compile, ASTSource, make_backend
        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)

        def _on_kernel_evict(_key, compiled_kernel):
            # Invalidate fast-path caches that may hold stale CUfunction
            # handles from the evicted module. Without this, Layer 1/2
            # would try to launch with an unloaded CUfunction → CUDA error.
            self.clear_fast_path_caches()
            if hasattr(compiled_kernel, 'module') and compiled_kernel.module is not None:
                self._deferred_unloader.queue_module(compiled_kernel.module)

        return _LRUDict(knobs.runtime.kernel_cache_size, on_evict=_on_kernel_evict), {}, target, backend, binder

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        # options
        options = backend.parse_options(kwargs)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs

    def _fast_launch(self, kernel, grid, stream, bound_vals, launch_fn, function, packed_metadata, coop, cluster, pdl,
                     no_scratch):
        """Resolve grid and dispatch kernel using cached launch properties.

        Shared between Layer 1 and Layer 2 fast paths to avoid code
        divergence — any change to the launch sequence only needs to
        happen in one place.
        """
        assert grid is not None
        if callable(grid):
            grid = grid(dict(zip(self.arg_names, bound_vals)))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        launch_enter = knobs.runtime.launch_enter_hook
        launch_exit = knobs.runtime.launch_exit_hook
        launch_metadata = None
        if launch_enter is not None:
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_vals)
        if no_scratch:
            launch_fn(grid_0, grid_1, grid_2, stream, function, coop, cluster, pdl, None, None, packed_metadata,
                      launch_metadata, launch_enter, launch_exit, *bound_vals)
        else:
            kernel.run(grid_0, grid_1, grid_2, stream, function, packed_metadata, launch_metadata, launch_enter,
                       launch_exit, *bound_vals)

    def clear_fast_path_caches(self):
        """Invalidate Layer 1 (identity) and Layer 2 (signature) fast-path caches.

        Call this after mutating any JITCallable that was previously passed as
        an argument to this kernel (e.g. via ``_unsafe_update_src``).
        """
        self._last_call = None
        self._last_kwargs = {}
        self._run_cache.clear()

    def run(self, *args, grid, warmup, **kwargs):
        # Fast path: if cache is populated, try identity/signature check
        # (separate method to keep run() bytecode compact).
        if not warmup and self._last_call is not None and not self.pre_run_hooks and not knobs.compilation.always_compile:
            result = self._try_fast_path(args, grid, kwargs)
            if result is not None:
                return result

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["sanitize_overflow"] = kwargs.get("sanitize_overflow",
                                                 False) or knobs.runtime.sanitize_overflow or kwargs["debug"]
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        self._deferred_unloader.flush(stream)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key)

        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)

            if os.environ.get("TRITON_DUMP_TLX_BENCHMARK"):
                try:
                    from triton.tools.tlx_benchmark_gen import capture_kernel_args
                    capture_kernel_args(bound_args, signature, constexprs, self.params)
                except Exception:
                    pass

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

            if os.environ.get("TRITON_DUMP_TLX_BENCHMARK"):
                try:
                    from triton.tools.tlx_benchmark_gen import capture_grid
                    capture_grid((grid_0, grid_1, grid_2))
                except Exception:
                    pass

            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())

            # Populate fast-path cache on first launch only.  Subsequent calls
            # hit the fast path (Layer 1/2) which updates the cache itself.
            # Populating on every slow-path call is too expensive for kernels
            # with many parameters (~5-7 µs for 50 kwargs) and measurably
            # regresses short-running kernels.
            if self._last_call is None and not self.pre_run_hooks and not knobs.compilation.always_compile:
                self._last_call = self._make_launch_cache(device, args, kernel, tuple(bound_args.values()))
                self._last_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ('debug', 'sanitize_overflow', 'instrumentation_mode')
                }
                fast_key = self._compute_fast_key(args, self._last_kwargs, device)
                if fast_key is not None:
                    self._run_cache.put(fast_key, self._last_call[2:])

        return kernel

    def _try_fast_path(self, args, grid, kwargs):
        """Attempt Layer 1 (identity) and Layer 2 (signature) fast paths.

        Returns the kernel on hit, or None on miss so run() falls through
        to the slow path.
        """
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        self._deferred_unloader.flush(stream)

        # Layer 1: Identity check — same Python objects as last call?
        last = self._last_call
        if last is not None and last.device is device and last.instrumentation_mode == knobs.compilation.instrumentation_mode:
            last_args = last.args
            if len(args) == len(last_args):
                identical = True
                for i in range(len(args)):
                    if args[i] is not last_args[i]:
                        identical = False
                        break
                if identical:
                    last_kw = self._last_kwargs
                    if len(kwargs) != len(last_kw):
                        identical = False
                    else:
                        for k, v in kwargs.items():
                            if k not in last_kw or v is not last_kw[k]:
                                identical = False
                                break
                if identical:
                    kernel = last.kernel
                    if self.used_global_vals:
                        not_present = object()
                        for (name, _), (val, globals_dict) in self.used_global_vals.items():
                            if globals_dict.get(name, not_present) != val:
                                kernel = None
                                break
                    if kernel is not None:
                        bound_vals = last.bound_vals
                        self._fast_launch(kernel, grid, stream, bound_vals, *last[4:11])
                        return kernel

        # Layer 2: Signature-based dict lookup — same specialization?
        fast_key = self._compute_fast_key(args, kwargs, device)
        if fast_key is not None:
            cached = self._run_cache.get(fast_key)
            if cached is not None:
                kernel = cached[0]
                if self.used_global_vals:
                    not_present = object()
                    for (name, _), (val, globals_dict) in self.used_global_vals.items():
                        if globals_dict.get(name, not_present) != val:
                            kernel = None
                            break
                if kernel is not None:
                    bound_vals = args
                    if kwargs:
                        param_names = [p.name for p in self.params]
                        bound_dict = dict(zip(param_names, args))
                        # Fill in defaults for params not covered by positional args,
                        # in case non-param kwargs (e.g. num_warps) inflated the
                        # length check in _compute_fast_key.
                        for p in self.params[len(args):]:
                            if p.has_default:
                                bound_dict[p.name] = p.default
                        bound_dict.update(kwargs)
                        bound_vals = tuple(bound_dict[n] for n in param_names)
                    self._fast_launch(kernel, grid, stream, bound_vals, *cached[2:9])
                    self._last_call = self._make_launch_cache(device, args, kernel, bound_vals)
                    self._last_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k not in ('debug', 'sanitize_overflow', 'instrumentation_mode')
                    }
                    return kernel

        return None

    @staticmethod
    def _make_launch_cache(device, args, kernel, bound_vals):
        """Build a _LastCall namedtuple caching everything needed for fast-path launch.

        Returns a 12-element _LastCall -- see the layout comment in run()
        above ``last = self._last_call``.  Layer 2 stores ``tuple[2:]``
        (without device/args) in ``_run_cache``.
        """
        launcher = kernel.run  # CudaLauncher or HIPLauncher instance (resolves property once)
        # HIPLauncher lacks launch_cluster, launch_pdl, and global_scratch_size.
        # Use getattr for these CUDA-specific attributes.  When they are absent
        # the launcher.launch() C function has a different call signature, so
        # no_scratch must be False to route through kernel.run() (__call__).
        has_direct_launch = hasattr(launcher, "launch_cluster")
        return _LastCall(
            device=device,
            args=args,
            kernel=kernel,
            bound_vals=bound_vals,
            launch_fn=launcher.launch,
            function=kernel.function,
            packed_metadata=kernel.packed_metadata,
            coop=launcher.launch_cooperative_grid,
            cluster=getattr(launcher, "launch_cluster", False),
            pdl=getattr(launcher, "launch_pdl", False),
            no_scratch=(has_direct_launch and launcher.global_scratch_size == 0 and launcher.profile_scratch_size == 0),
            instrumentation_mode=knobs.compilation.instrumentation_mode,
        )

    def repr(self, _):
        return self._fn_name if self._repr is None else self._repr(_)

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []
        do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

        super().__init__(fn)
        self.module = fn.__module__
        self.version = version
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
        self._repr = repr
        self.launch_metadata = launch_metadata

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = i in do_not_specialize or param.name in do_not_specialize
            dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
            self.params.append(KernelParam(i, param, dns, dns_oa))

        # cache of just-in-time compiled kernels
        self.device_caches = _DeviceCaches(self, self.create_binder)

        # Deferred queue for cuModuleUnload — flushed at kernel launch
        # when stream is not in CUDA graph capture mode.
        self._deferred_unloader = _DeferredModuleUnloader()

        # Last-call cache for identity-based fast path (Layer 1).
        # _LastCall namedtuple built by _make_launch_cache(); see run() for layout.
        self._last_call = None
        self._last_kwargs = {}
        # Signature-based fast-path cache (Layer 2).
        # Maps (device, fast_arg_signature) -> compiled kernel.
        # Bounded LRU to prevent unbounded CUDA memory growth with varying shapes.
        self._run_cache = _LRUDict(knobs.runtime.kernel_cache_size)

        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

    def preload(self, specialization_data):
        import json
        import triton.language as tl
        device = driver.active.get_current_device()
        deserialized_obj = json.loads(specialization_data)
        if deserialized_obj['name'] != self._fn_name:
            raise RuntimeError(
                f"Specialization data is for {deserialized_obj['name']} but trying to preload for {self._fn_name}")
        constant_keys = map(tuple, deserialized_obj['constant_keys'])
        constant_vals = deserialized_obj['constant_vals']
        constexprs = {
            key:
            tl.dtype(value) if tl.dtype.is_dtype(value) else
            tl.constexpr(value['constexpr']) if isinstance(value, dict) and 'constexpr' in value else value
            for key, value in zip(constant_keys, constant_vals)
        }
        attrs_keys = map(tuple, deserialized_obj['attrs_keys'])
        attrs_vals = deserialized_obj['attrs_vals']
        attrs = dict(zip(attrs_keys, attrs_vals))
        # JSON serializes tuples as lists, so they need to be converted back;
        # This can be done unconditionally, since lists are not accepted in Triton kernel signatures.
        signature = {key: convert_to_tuple_if_list(value) for key, value in deserialized_obj['signature'].items()}
        options = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in deserialized_obj['options'].items()
        }
        key = deserialized_obj['key']
        _, _, _, backend, _ = self.device_caches[device]
        options = backend.parse_options(options)
        return self._do_compile(
            key,
            signature,
            device,
            constexprs,
            options,
            attrs,
            warmup=True,
        )

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        kernel_cache, _, target, backend, _ = self.device_caches[device]

        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs], warmup):
            return None
        src = self.ASTSource(self, signature, constexprs, attrs)

        async_mode = _async_compile.active_mode.get()
        if async_mode is not None:

            env_vars = get_cache_invalidating_env_vars()
            cache_key = get_cache_key(src, backend, options, env_vars)

            def async_compile():
                return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

            def finalize_compile(kernel):
                kernel_cache.put(key, kernel)
                self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options,
                                [attrs], warmup)

            kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
        else:
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache.put(key, kernel)
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                            warmup)
        return kernel

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__qualname__})"


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> JITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> KernelInterface[T]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


# -----------------------------------------------------------------------------
# Utilities for mocking tensors
# -----------------------------------------------------------------------------


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == "dtype" and arg.__module__ == "torch":
            return MockTensor(arg)
        return arg

    def __init__(self, dtype, shape=None):
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        return 0  # optimistically assumes 32 bit pointer range


class TensorWrapper:

    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.data = base.data
        self.device = base.device
        self.shape = self.base.shape

    def data_ptr(self):
        return self.base.data_ptr()

    def stride(self, *args):
        return self.base.stride(*args)

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"

    def element_size(self):
        return self.base.element_size()

    def cpu(self):
        return TensorWrapper(self.base.cpu(), self.dtype)

    def copy_(self, other):
        self.base.copy_(other.base)

    def clone(self):
        return TensorWrapper(self.base.clone(), self.dtype)

    def to(self, device):
        return TensorWrapper(self.base.to(device), self.dtype)

    def new_empty(self, sizes):
        return TensorWrapper(self.base.new_empty(sizes), self.dtype)


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif hasattr(tensor, "data_ptr"):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")


def get_jit_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.starting_line_number
    # Match the following pattern:
    # @triton.autotune(...) <- foo.__code__.co_firstlineno
    # @triton.heuristics(...)
    # @triton.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


class BoundConstexprFunction(JITCallable):

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    @property
    def cache_key(self):
        return self.__func__.cache_key

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


class ConstexprFunction(JITCallable):

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Create a bound function to support constexpr_function methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    def __call__(self, *args, _semantic=None, **kwargs):
        from triton.language.core import _unwrap_if_constexpr, constexpr
        # de-constexpr arguments and discard the _semantic keyword argument:
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # call the raw Python function f:
        res = self.fn(*args, **kwargs)

        if _semantic is None:
            # Not called by triton code generator, e.g. in host code, another constexpr function, or even an aggreate's __init__ function
            return res

        # convert result back to a Triton constexpr:
        if knobs.runtime.interpret:
            return res  # No constexpr in interpreter
        return constexpr(res)


def constexpr_function(fn):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)
