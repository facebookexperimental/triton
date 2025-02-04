import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import NonGPUDriver
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature):

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr", ):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i in signature) if signature else ''

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} _arg{i}" for i, ty in signature.items() if ty != "constexpr")
    arg_decls = ', ' + arg_decls if arg_decls else ''

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")
    internal_args_list = ', ' + ', '.join(internal_args_list) if internal_args_list else ''

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    params = [f"_arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params = ', '.join(params) + ', ' if params else ''
    kernel_ptr_t = ', '.join([f"{ty_to_cpp(ty)}" for ty in signature.values() if ty != "constexpr"] + ["uint32_t"] * 6)

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_ptr_t});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static void _launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr{arg_decls}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({params}0, 0, 0, 1, 1, 1);
      return;
  }}

  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({params}x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

  // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({params}x, y, z, gridX, gridY, gridZ);
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &py_obj_stream, &pKrnl,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {newline.join(ptr_decls)}
  _launch(gridX, gridY, gridZ, num_threads, kernel_ptr{internal_args_list});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args):
        self.launch(*args)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(NonGPUDriver):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_target(self):
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        # We use warp_size = 1 to emulate CPU-like behavior.
        warp_size = 1
        return GPUTarget("cpu", cpu_arch, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    def clear_cache(self, cache):
        cache.zero_()
