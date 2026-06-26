#include <Python.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace py = pybind11;

using DTypePtrKey = std::pair<Py_hash_t, bool>;
using DTypeKey = Py_hash_t;

struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey &k) const {
    return std::hash<Py_hash_t>()(k.first) ^ (std::hash<bool>()(k.second) << 1);
  }
};

using DtypePtr2Str =
    std::unordered_map<DTypePtrKey, PyObject *, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, PyObject *>;

using TypeHandler = std::pair<py::object, py::object> (*)(PyObject *,
                                                          PyObject *, bool,
                                                          bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject *, TypeHandler>;

static std::pair<py::object, py::object>
specialize_arg(PyObject *backend, PyObject *arg, bool is_const,
               bool specialize_value, bool align);

static bool init_called = false;

static PyObject *constexpr_cls = nullptr;
static PyObject *jit_callable_cls = nullptr;
static PyObject *tensor_descriptor_cls = nullptr;
static PyObject *nvidia_tensor_descriptor_cls = nullptr;
static PyObject *nvidia_tensor_descriptor_im2col_cls = nullptr;
static PyObject *amd_tensor_descriptor_cls = nullptr;
static PyObject *canonicalize_dtype_fn = nullptr;
static PyObject *canonicalize_ptr_dtype_fn = nullptr;
static PyObject *torch_tensor_cls = nullptr;

// Fast tensor access API — registered at runtime by _torch_bridge extension.
// When available, provides ~10x faster dtype/data_ptr extraction by bypassing
// Python attribute lookups and accessing THPVariable struct fields directly.
struct TritonTensorAccessAPI {
  int8_t (*get_scalar_type)(PyObject *);
  uint64_t (*get_data_ptr)(PyObject *);
  // Extract TensorDescriptor fields (base.data_ptr, shape, strides) in one
  // shot. Returns ndim on success, -1 on failure.
  int (*extract_tensordesc)(PyObject *td_obj, uint64_t *out_data_ptr,
                            int64_t *out_shape, int64_t *out_strides,
                            int max_ndim);
};
static TritonTensorAccessAPI *g_tensor_api = nullptr;

// ScalarType → fc type_code mapping (indexed by c10::ScalarType int8_t value)
static uint8_t scalar_type_to_fc_code[64];
static bool scalar_type_map_initialized[64] = {};

static PyObject *i32_str = nullptr;
static PyObject *i64_str = nullptr;
static PyObject *u64_str = nullptr;
static PyObject *fp32_str = nullptr;
static PyObject *u1_str = nullptr;
static PyObject *D_str = nullptr;
static PyObject *constexpr_str = nullptr;
static PyObject *empty_str = nullptr;
static PyObject *nvTmaDesc_str = nullptr;

static PyObject *base_attr = nullptr;
static PyObject *data_ptr_attr = nullptr;
static PyObject *dtype_attr = nullptr;
static PyObject *cache_key_attr = nullptr;
static PyObject *_fields_attr = nullptr;
static PyObject *block_shape_attr = nullptr;
static PyObject *shape_attr = nullptr;
static PyObject *layout_attr = nullptr;
static PyObject *has_native_tensor_spec_attr = nullptr;
static PyObject *get_tensor_spec_attr = nullptr;
static PyObject *align_kwarg = nullptr;
static PyObject *tma_desc_cpu_ptr_attr = nullptr;

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TypeHandlerCache type_handler_cache;

// Wrappers to make steal and borrow slightly simpler. We use raw CPython API
// with py::object to handle decref, as using the pybind11 APIs adds exception
// handling overhead which is quite significant here.
py::object from_new_ref(py::handle val) {
  return py::reinterpret_steal<py::object>(val);
}
py::object from_borrowed_ref(py::handle val) {
  return py::reinterpret_borrow<py::object>(val);
}

PyObject *intern_from_string(const char *str) {
  PyObject *obj = PyUnicode_InternFromString(str);
  if (!obj)
    throw py::error_already_set();
  return obj;
}

PyObject *import_from(const char *module_name, const char *var_name) {
  py::object var = py::module_::import(module_name).attr(var_name);
  return var.release().ptr();
}

void init_interned_strings() {
  i32_str = intern_from_string("i32");
  i64_str = intern_from_string("i64");
  u64_str = intern_from_string("u64");
  fp32_str = intern_from_string("fp32");
  u1_str = intern_from_string("u1");
  D_str = intern_from_string("D");
  constexpr_str = intern_from_string("constexpr");
  empty_str = intern_from_string("");
  nvTmaDesc_str = intern_from_string("nvTmaDesc");

  base_attr = intern_from_string("base");
  data_ptr_attr = intern_from_string("data_ptr");
  dtype_attr = intern_from_string("dtype");
  cache_key_attr = intern_from_string("cache_key");
  _fields_attr = intern_from_string("_fields");
  block_shape_attr = intern_from_string("block_shape");
  shape_attr = intern_from_string("shape");
  layout_attr = intern_from_string("layout");
  has_native_tensor_spec_attr =
      intern_from_string("supports_native_tensor_specialization");
  get_tensor_spec_attr = intern_from_string("get_tensor_specialization");

  align_kwarg = py::make_tuple("align").release().ptr();
  tma_desc_cpu_ptr_attr = intern_from_string("tma_desc_cpu_ptr");
}

void init_type_handler_cache();

bool init_globals() noexcept try {
  // Import releavant symbols
  jit_callable_cls = import_from("triton.runtime.jit", "JITCallable");
  tensor_descriptor_cls =
      import_from("triton.tools.tensor_descriptor", "TensorDescriptor");
  nvidia_tensor_descriptor_cls = import_from(
      "triton.experimental.gluon.nvidia.hopper", "TensorDescriptor");
  nvidia_tensor_descriptor_im2col_cls = import_from(
      "triton.experimental.gluon.nvidia.hopper", "TensorDescriptorIm2Col");
  amd_tensor_descriptor_cls =
      import_from("triton.experimental.gluon.amd.gfx1250", "TensorDescriptor");

  auto m_canonicalize = py::module_::import("triton._utils");
  canonicalize_dtype_fn = import_from("triton._utils", "canonicalize_dtype");
  canonicalize_ptr_dtype_fn =
      import_from("triton._utils", "canonicalize_ptr_dtype");
  constexpr_cls = import_from("triton.language", "constexpr");

  try {
    torch_tensor_cls = import_from("torch", "Tensor");
  } catch (py::error_already_set &) {
  }

  init_interned_strings();
  init_type_handler_cache();

  init_called = true;
  return true;
} catch (py::error_already_set &e) {
  e.restore();
  return false;
}

std::pair<py::object, py::object> specialize_tensordesc(PyObject *arg,
                                                        bool has_layout) {
  auto base = from_new_ref(PyObject_GetAttr(arg, base_attr));
  if (!base)
    return {};

  auto dtype = from_new_ref(PyObject_GetAttr(base.ptr(), dtype_attr));
  if (!dtype)
    return {};

  PyObject *type_str;
  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};
  DTypeKey dsk{dtype_hash};
  auto it = dtype2str.find(dsk);
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    auto res = from_new_ref(PyObject_CallFunctionObjArgs(canonicalize_dtype_fn,
                                                         dtype.ptr(), nullptr));
    if (!res)
      return {};
    dtype2str[dsk] = res.ptr();
    type_str = res.release().ptr();
  }

  std::string desc_cstr;
  desc_cstr.reserve(128);

  // Determine im2col by class type (Gluon only).
  bool is_im2col = false;
  if (has_layout && nvidia_tensor_descriptor_im2col_cls) {
    int is_inst = PyObject_IsInstance(arg, nvidia_tensor_descriptor_im2col_cls);
    if (is_inst < 0)
      return {};
    is_im2col = is_inst == 1;
  }

  desc_cstr = is_im2col ? "tensordesc_im2col<" : "tensordesc<";
  auto dtype_str = from_new_ref(PyObject_Str(type_str));
  if (!dtype_str)
    return {};

  const char *dtype_cstr = PyUnicode_AsUTF8(dtype_str.ptr());
  if (!dtype_cstr)
    return {};
  desc_cstr += dtype_cstr;

  auto block_shape_obj = from_new_ref(PyObject_GetAttr(arg, block_shape_attr));
  if (!block_shape_obj)
    return {};
  auto block_shape_list = from_new_ref(PySequence_List(block_shape_obj.ptr()));
  if (!block_shape_list)
    return {};
  auto block_shape_str = from_new_ref(PyObject_Str(block_shape_list.ptr()));
  if (!block_shape_str)
    return {};
  const char *block_shape_cstr = PyUnicode_AsUTF8(block_shape_str.ptr());
  if (!block_shape_cstr)
    return {};
  desc_cstr += block_shape_cstr;

  // For im2col mode, append input tensor rank after block_shape
  // Format: tensordesc_im2col<dtype[block_shape],input_rank=N,layout>
  // This allows the driver to know the N-dimensional shape/strides to pass
  if (is_im2col) {
    auto tensor_shape_obj = from_new_ref(PyObject_GetAttr(arg, shape_attr));
    if (!tensor_shape_obj)
      return {};
    Py_ssize_t tensor_rank = PySequence_Size(tensor_shape_obj.ptr());
    if (tensor_rank < 0)
      return {};
    desc_cstr += ",input_rank=";
    desc_cstr += std::to_string(tensor_rank);
  }

  if (has_layout) {
    auto layout_obj = from_new_ref(PyObject_GetAttr(arg, layout_attr));
    if (!layout_obj)
      return {};
    auto layout_repr = from_new_ref(PyObject_Repr(layout_obj.ptr()));
    if (!layout_repr)
      return {};
    desc_cstr += ",";
    const char *layout_cstr = PyUnicode_AsUTF8(layout_repr.ptr());
    if (!layout_cstr)
      return {};
    desc_cstr += layout_cstr;
  }

  desc_cstr += ">";
  auto type_str_result = from_new_ref(PyUnicode_FromString(desc_cstr.c_str()));
  if (!type_str_result)
    return {};

  return {std::move(type_str_result), py::none()};
}

std::pair<py::object, py::object> handle_long_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
  if (PyErr_Occurred()) {
    return {};
  }

  if (specialize_value && (val == 1)) {
    return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
  }

  py::handle type_str;
  py::handle key_obj;
  if (overflow == 0) {
    type_str = (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str;
    if (specialize_value) {
      key_obj = (align && ((val & 15) == 0)) ? D_str : empty_str;
    }
  } else {
    unsigned long long val_64 = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
      // this runs into an edge-case where the Python reference
      // returns i64 as type and alignment of the value despite
      // not being representable as such which at kernel launch later
      // will throw an OverflowError nevertheless, here we throw
      // OverflowError immediately
      PyErr_SetString(PyExc_OverflowError,
                      "integer to be specialized too large to represent");
      return {};
    }
    type_str = u64_str;
    if (specialize_value) {
      key_obj = (align && ((val_64 & 15) == 0)) ? D_str : empty_str;
    }
  }
  if (!key_obj) {
    return {from_borrowed_ref(type_str), py::none()};
  }
  return {from_borrowed_ref(type_str), from_borrowed_ref(key_obj)};
}

std::pair<py::object, py::object> handle_tensor(PyObject *backend,
                                                PyObject *arg, bool is_const,
                                                bool specialize_value,
                                                bool align) {
  // handle type_str specialization of a tensor
  auto dtype = from_new_ref(PyObject_GetAttr(arg, dtype_attr));
  if (!dtype)
    return {};

  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};

  DTypePtrKey dsk{dtype_hash, is_const};
  auto it = dtype_ptr2str.find(dsk);

  py::handle type_str;
  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    auto canon_res =
        PyObject_CallFunctionObjArgs(canonicalize_ptr_dtype_fn, dtype.ptr(),
                                     is_const ? Py_True : Py_False, nullptr);
    if (!canon_res)
      return {};
    dtype_ptr2str[dsk] = canon_res;
    type_str = canon_res;
  }

  // handle alignment specialization of a tensor
  if (!specialize_value) {
    return {from_borrowed_ref(type_str), py::none()};
  }

  bool native_impl_available = false;
  auto native_spec_obj =
      from_new_ref(PyObject_GetAttr(backend, has_native_tensor_spec_attr));
  if (native_spec_obj) {
    native_impl_available = PyObject_IsTrue(native_spec_obj.ptr());
  } else {
    PyErr_Clear();
    // on error we fall back to native_impl_available = false gracefully
  }

  py::object key;
  if (native_impl_available) {
    auto data_ptr_result =
        from_new_ref(PyObject_CallMethodNoArgs(arg, data_ptr_attr));
    if (!data_ptr_result)
      return {};

    auto data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result.ptr());
    if (PyErr_Occurred())
      return {};

    auto key_obj = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;
    key = from_borrowed_ref(key_obj);
  } else {
    PyObject *args[3] = {backend, arg, align ? Py_True : Py_False};
    PyObject *kwnames = align_kwarg;
    key = from_new_ref(
        PyObject_VectorcallMethod(get_tensor_spec_attr, args, 2, kwnames));
    if (!key)
      return {};
  }

  return {from_borrowed_ref(type_str), std::move(key)};
}

std::pair<py::object, py::object> handle_bool_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  return {from_borrowed_ref(u1_str), py::none()};
}

std::pair<py::object, py::object>
handle_float_type(PyObject *backend, PyObject *arg, bool is_const,
                  bool specialize_value, bool align) {
  return {from_borrowed_ref(fp32_str), py::none()};
}

std::pair<py::object, py::object>
handle_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                         bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

std::pair<py::object, py::object>
handle_gluon_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                               bool specialize_value, bool align) {
  return specialize_tensordesc(arg, true);
}

std::pair<py::object, py::object>
handle_constexpr_type(PyObject *backend, PyObject *arg, bool is_const,
                      bool specialize_value, bool align) {
  return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
}

std::pair<py::object, py::object>
handle_jit_callable(PyObject *backend, PyObject *arg, bool is_const,
                    bool specialize_value, bool align) {
  auto cache_key = from_new_ref(PyObject_GetAttr(arg, cache_key_attr));
  if (!cache_key)
    return {};
  return {from_borrowed_ref(constexpr_str), std::move(cache_key)};
}

std::pair<py::object, py::object> handle_tuple(PyObject *backend, PyObject *arg,
                                               bool is_const,
                                               bool specialize_value,
                                               bool align) {
  Py_ssize_t size = PyTuple_GET_SIZE(arg);
  if (size == 0) {
    // return tuple of empty tuples as in python reference
    return {from_borrowed_ref(arg), from_borrowed_ref(arg)};
  }

  bool is_namedtuple = PyObject_HasAttr(arg, _fields_attr);
  auto tuple_type = Py_TYPE(arg);

  // Create tuples directly instead of lists
  auto tys_tuple = from_new_ref(PyTuple_New(size));
  if (!tys_tuple)
    return {};

  auto keys_tuple = from_new_ref(PyTuple_New(size));
  if (!keys_tuple)
    return {};

  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *item = PyTuple_GET_ITEM(arg, i); // Borrowed reference
    // python reference calls specialize recursively with default arguments set
    // currently this is is_const=False, specialize_value=True, align=True
    auto [type, key] = specialize_arg(backend, item, false, true, true);
    if (!type || !key)
      return {};
    // Steals reference
    PyTuple_SET_ITEM(tys_tuple.ptr(), i, type.release().ptr());
    PyTuple_SET_ITEM(keys_tuple.ptr(), i, key.release().ptr());
  }

  if (is_namedtuple) {
    tys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, tys_tuple.ptr()));
    if (!tys_tuple)
      return {};
    keys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, keys_tuple.ptr()));
    if (!keys_tuple)
      return {};
  }

  return {std::move(tys_tuple), std::move(keys_tuple)};
}

// initialize type handler which returns specialize impelemntations based on
// type(arg)
void init_type_handler_cache() {
  // Python Types (int, bool, float, tuple)
  type_handler_cache[&PyLong_Type] = handle_long_type;
  type_handler_cache[&PyBool_Type] = handle_bool_type;
  type_handler_cache[&PyFloat_Type] = handle_float_type;
  type_handler_cache[&PyTuple_Type] = handle_tuple;

  // torch.Tensor
  if (torch_tensor_cls && PyType_Check(torch_tensor_cls)) {
    type_handler_cache[(PyTypeObject *)torch_tensor_cls] = handle_tensor;
  }
  // TensorDescriptor
  if (tensor_descriptor_cls && PyType_Check(tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)tensor_descriptor_cls] =
        handle_tensor_descriptor;
  }
  // GluonTensorDescriptor
  if (nvidia_tensor_descriptor_cls &&
      PyType_Check(nvidia_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)nvidia_tensor_descriptor_cls] =
        handle_gluon_tensor_descriptor;
  }
  if (nvidia_tensor_descriptor_im2col_cls &&
      PyType_Check(nvidia_tensor_descriptor_im2col_cls)) {
    type_handler_cache[(PyTypeObject *)nvidia_tensor_descriptor_im2col_cls] =
        handle_gluon_tensor_descriptor;
  }
  if (amd_tensor_descriptor_cls && PyType_Check(amd_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)amd_tensor_descriptor_cls] =
        handle_gluon_tensor_descriptor;
  }
  // constexpr
  if (constexpr_cls && PyType_Check(constexpr_cls)) {
    type_handler_cache[(PyTypeObject *)constexpr_cls] = handle_constexpr_type;
  }
  // JITCallable
  if (jit_callable_cls && PyType_Check(jit_callable_cls)) {
    type_handler_cache[(PyTypeObject *)jit_callable_cls] = handle_jit_callable;
  }
}

// specialization logic without passing of objects from Python (to be called in
// specialize_impl only)
std::pair<py::object, py::object> specialize_arg(PyObject *backend,
                                                 PyObject *arg, bool is_const,
                                                 bool specialize_value,
                                                 bool align) {
  // fast-path for default types
  PyTypeObject *arg_type = Py_TYPE(arg);
  auto it = type_handler_cache.find(arg_type);
  if (it != type_handler_cache.end()) {
    return it->second(backend, arg, is_const, specialize_value, align);
  }

  // separate handling of None
  if (Py_IsNone(arg)) {
    return {from_borrowed_ref(constexpr_str), py::none()};
  }

  // handling of sublcasses of tuples
  if (PyTuple_Check(arg)) {
    return handle_tuple(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking full inheritance
  if (PyObject_IsInstance(arg, constexpr_cls)) {
    return handle_constexpr_type(backend, arg, is_const, specialize_value,
                                 align);
  }

  if (PyObject_IsInstance(arg, tensor_descriptor_cls)) {
    return handle_tensor_descriptor(backend, arg, is_const, specialize_value,
                                    align);
  }

  if (PyObject_IsInstance(arg, nvidia_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const,
                                          specialize_value, align);
  }

  if (PyObject_IsInstance(arg, amd_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const,
                                          specialize_value, align);
  }

  if (PyObject_IsInstance(arg, jit_callable_cls)) {
    return handle_jit_callable(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking attributes directly
  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  // Handle TMA descriptors (objects with tma_desc_cpu_ptr attribute)
  if (PyObject_HasAttr(arg, tma_desc_cpu_ptr_attr)) {
    return {from_borrowed_ref(nvTmaDesc_str), py::none()};
  }

  // fallback for default types
  if (PyLong_Check(arg)) {
    return handle_long_type(backend, arg, is_const, specialize_value, align);
  }
  if (PyFloat_Check(arg)) {
    return handle_float_type(backend, arg, is_const, specialize_value, align);
  }

  return {};
}

// main entry-point from Python implementing specialization logic natively
PyObject *specialize_impl(PyObject *self, PyObject *const *args,
                          Py_ssize_t nargs) {
  if (!init_called) {
    if (!init_globals()) {
      return nullptr;
    }
  }

  if (nargs != 5) {
    PyErr_SetString(PyExc_TypeError,
                    "native_specialize_impl expected 5 arguments");
    return nullptr;
  }

  PyObject *backend = args[0];
  PyObject *arg = args[1];
  int is_const = PyObject_IsTrue(args[2]);
  int specialize_value = PyObject_IsTrue(args[3]);
  int align = PyObject_IsTrue(args[4]);

  if (is_const == -1 || specialize_value == -1 || align == -1) {
    PyErr_SetString(PyExc_TypeError, "native_specialize_impl expected boolean "
                                     "arguments for args2, args3, args4");
    return nullptr;
  }

  auto [type, key] =
      specialize_arg(backend, arg, is_const, specialize_value, align);

  // check if specialization failed
  if (!type || !key) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError, "failed to specialize argument of type: %s",
                   Py_TYPE(arg)->tp_name);
    }
    return nullptr;
  }

  return PyTuple_Pack(2, type.ptr(), key.ptr());
}

// ============================================================================
// Fast Dispatch Cache
// ============================================================================
// C-level specialization cache that replaces the Python binder + cache_key +
// dict.get with a single function call on cache hit. No tensor lifetime
// extension, no identity caching. Fully sound.
//
// Thread safety: This cache relies on the GIL for thread safety. The hash table
// operations (insert, resize, lookup) have no internal synchronization. If
// CPython's free-threaded mode (PEP 703 / nogil) is ever adopted, a lock or
// atomic operations must be added here.

static constexpr int FC_MAX_ARGS = 64;

enum ArgTypeCode : uint8_t {
  TC_I32 = 0,
  TC_I64 = 1,
  TC_U64 = 2,
  TC_FP32 = 3,
  TC_U1 = 4,
  TC_CONSTEXPR = 5,
  TC_NVTMADESC = 6,
  TC_TENSORDESC = 7, // Host-side TensorDescriptor, keyed by dtype+block_shape
  TC_PTR_BASE = 32,
  TC_PTR_CONST_BASE = 64,
  TC_UNSUPPORTED = 255,
};

struct ArgKeySlot {
  uint8_t type_code;
  uint8_t align_bit;
  Py_hash_t constexpr_hash;
};

struct FCCacheKey {
  ArgKeySlot slots[FC_MAX_ARGS];
  uint16_t n_args;
  uint64_t options_hash;

  bool operator==(const FCCacheKey &other) const {
    if (n_args != other.n_args || options_hash != other.options_hash)
      return false;
    return memcmp(slots, other.slots, n_args * sizeof(ArgKeySlot)) == 0;
  }
};

struct FCCacheKeyHash {
  size_t operator()(const FCCacheKey &k) const {
    size_t hash = 14695981039346656037ULL;
    const uint8_t *data = reinterpret_cast<const uint8_t *>(k.slots);
    size_t len = k.n_args * sizeof(ArgKeySlot);
    for (size_t i = 0; i < len; i++) {
      hash ^= data[i];
      hash *= 1099511628211ULL;
    }
    hash ^= k.options_hash;
    hash *= 1099511628211ULL;
    return hash;
  }
};

struct ParamMeta {
  uint8_t is_constexpr : 1;
  uint8_t do_not_specialize : 1;
  uint8_t do_not_specialize_on_alignment : 1;
  uint8_t has_annotation : 1;
  uint8_t
      is_ptr : 1; // Set if annotation starts with '*' (pointer/tensor param)
  uint8_t is_tensordesc : 1; // Set if annotation starts with 'tensordesc'
  uint8_t annotation_type_code;
};

struct FCEntry {
  FCCacheKey key;
  PyObject *kernel;     // Strong ref to CompiledKernel
  PyObject *dispatcher; // Strong ref or NULL
  PyObject **constexpr_vals;
  int *constexpr_positions;
  int n_constexpr;
  int *dispatch_arg_indices; // Indices into call_args for dispatcher (or NULL)
  int n_dispatch_args;       // Length of dispatch_arg_indices (0 = legacy path)
  bool occupied;
};

struct FastCache {
  ParamMeta param_meta[FC_MAX_ARGS];
  int n_params;
  FCEntry *table;
  size_t capacity;
  size_t count;

  FastCache() : n_params(0), table(nullptr), capacity(0), count(0) {
    memset(param_meta, 0, sizeof(param_meta));
  }

  ~FastCache() {
    if (table) {
      for (size_t i = 0; i < capacity; i++) {
        if (table[i].occupied) {
          Py_XDECREF(table[i].kernel);
          Py_XDECREF(table[i].dispatcher);
          if (table[i].constexpr_vals) {
            for (int j = 0; j < table[i].n_constexpr; j++)
              Py_XDECREF(table[i].constexpr_vals[j]);
            free(table[i].constexpr_vals);
          }
          free(table[i].constexpr_positions);
          free(table[i].dispatch_arg_indices);
        }
      }
      free(table);
    }
  }

  void init_table(size_t cap) {
    capacity = cap;
    table = (FCEntry *)calloc(capacity, sizeof(FCEntry));
  }

  void resize() {
    size_t new_cap = capacity * 2;
    FCEntry *new_table = (FCEntry *)calloc(new_cap, sizeof(FCEntry));
    FCCacheKeyHash hasher;
    for (size_t i = 0; i < capacity; i++) {
      if (table[i].occupied) {
        size_t idx = hasher(table[i].key) % new_cap;
        while (new_table[idx].occupied)
          idx = (idx + 1) % new_cap;
        new_table[idx] = table[i];
      }
    }
    free(table);
    table = new_table;
    capacity = new_cap;
  }

  FCEntry *lookup(const FCCacheKey &key, PyObject *const *args) {
    if (!table || count == 0)
      return nullptr;
    FCCacheKeyHash hasher;
    size_t idx = hasher(key) % capacity;
    size_t probes = 0;
    while (probes < capacity) {
      if (!table[idx].occupied)
        return nullptr;
      if (table[idx].key == key) {
        // Verify constexpr/tensordesc equality (hash collision guard)
        for (int i = 0; i < table[idx].n_constexpr; i++) {
          int pos = table[idx].constexpr_positions[i];
          int eq;
          if (key.slots[pos].type_code == TC_TENSORDESC) {
            // Compare stored (dtype, block_shape) proxy against current arg
            PyObject *base = PyObject_GetAttrString(args[pos], "base");
            PyObject *dtype =
                base ? PyObject_GetAttrString(base, "dtype") : nullptr;
            Py_XDECREF(base);
            PyObject *block_shape =
                PyObject_GetAttrString(args[pos], "block_shape");
            PyObject *bs_tuple =
                block_shape ? PySequence_Tuple(block_shape) : nullptr;
            Py_XDECREF(block_shape);
            if (dtype && bs_tuple) {
              PyObject *cmp_key = PyTuple_Pack(2, dtype, bs_tuple);
              eq = cmp_key ? PyObject_RichCompareBool(
                                 table[idx].constexpr_vals[i], cmp_key, Py_EQ)
                           : -1;
              Py_XDECREF(cmp_key);
            } else {
              eq = -1;
            }
            Py_XDECREF(dtype);
            Py_XDECREF(bs_tuple);
          } else {
            eq = PyObject_RichCompareBool(table[idx].constexpr_vals[i],
                                          args[pos], Py_EQ);
          }
          if (eq <= 0) {
            if (eq == -1)
              PyErr_Clear();
            return nullptr;
          }
        }
        return &table[idx];
      }
      idx = (idx + 1) % capacity;
      probes++;
    }
    return nullptr;
  }

  void insert(const FCCacheKey &key, PyObject *kernel, PyObject *dispatcher,
              PyObject *const *args, int n_args) {
    if (!table)
      init_table(16);
    if (count * 4 >= capacity * 3)
      resize();

    int n_ce = 0;
    for (int i = 0; i < n_args && i < n_params; i++) {
      if (param_meta[i].is_constexpr ||
          key.slots[i].type_code == TC_CONSTEXPR ||
          key.slots[i].type_code == TC_TENSORDESC)
        n_ce++;
    }
    int *positions = n_ce ? (int *)malloc(n_ce * sizeof(int)) : nullptr;
    PyObject **vals =
        n_ce ? (PyObject **)malloc(n_ce * sizeof(PyObject *)) : nullptr;
    if (n_ce && (!positions || !vals)) {
      free(positions);
      free(vals);
      return; // OOM — skip insertion
    }
    int ci = 0;
    for (int i = 0; i < n_args && i < n_params; i++) {
      if (param_meta[i].is_constexpr ||
          key.slots[i].type_code == TC_CONSTEXPR) {
        positions[ci] = i;
        vals[ci] = args[i];
        Py_INCREF(args[i]);
        ci++;
      } else if (key.slots[i].type_code == TC_TENSORDESC) {
        // Store (dtype, block_shape) tuple as comparison key for equality
        // verification. TensorDescriptor.__eq__ compares base tensors
        // element-wise (raises on bool conversion), so we use a proxy tuple.
        positions[ci] = i;
        PyObject *base = PyObject_GetAttrString(args[i], "base");
        PyObject *dtype =
            base ? PyObject_GetAttrString(base, "dtype") : nullptr;
        Py_XDECREF(base);
        PyObject *block_shape = PyObject_GetAttrString(args[i], "block_shape");
        PyObject *bs_tuple =
            block_shape ? PySequence_Tuple(block_shape) : nullptr;
        Py_XDECREF(block_shape);
        if (dtype && bs_tuple) {
          PyObject *cmp_key = PyTuple_Pack(2, dtype, bs_tuple);
          Py_DECREF(dtype);
          Py_DECREF(bs_tuple);
          if (!cmp_key) {
            // Cleanup already-stored vals and bail
            for (int k = 0; k < ci; k++)
              Py_DECREF(vals[k]);
            free(vals);
            free(positions);
            return;
          }
          vals[ci] = cmp_key; // new ref from PyTuple_Pack
        } else {
          Py_XDECREF(dtype);
          Py_XDECREF(bs_tuple);
          // Can't build comparison key — skip insertion entirely
          for (int k = 0; k < ci; k++)
            Py_DECREF(vals[k]);
          free(vals);
          free(positions);
          return;
        }
        ci++;
      }
    }

    FCCacheKeyHash hasher;
    size_t idx = hasher(key) % capacity;
    while (table[idx].occupied)
      idx = (idx + 1) % capacity;

    table[idx].key = key;
    table[idx].kernel = kernel;
    Py_INCREF(kernel);
    table[idx].dispatcher = dispatcher;
    Py_XINCREF(dispatcher);
    table[idx].constexpr_vals = vals;
    table[idx].constexpr_positions = positions;
    table[idx].n_constexpr = n_ce;
    table[idx].dispatch_arg_indices = nullptr;
    table[idx].n_dispatch_args = 0;
    table[idx].occupied = true;
    count++;
  }

  void set_dispatch_indices(size_t idx, int *indices, int n) {
    table[idx].dispatch_arg_indices = indices;
    table[idx].n_dispatch_args = n;
  }
};

// Interned attribute strings for fast cache
static PyObject *fc_cache_capsule_attr = nullptr;

// Dtype hash → type_code (populated on first encounter)
static std::unordered_map<Py_hash_t, uint8_t> fc_dtype_to_code;
static uint8_t fc_next_dtype_code = 0;

static uint8_t fc_get_tensor_type_code(PyObject *arg, bool is_const) {
  // Fast path: use torch_bridge direct struct access (no Python calls)
  if (g_tensor_api) {
    int8_t st = g_tensor_api->get_scalar_type(arg);
    if (st < 0 || st >= 64)
      goto slow_path;
    uint8_t code;
    if (scalar_type_map_initialized[st]) {
      code = scalar_type_to_fc_code[st];
    } else {
      // Check if slow path already assigned a code for this dtype hash.
      // This can happen when a Tensor subclass (fails THPVariable_CheckExact)
      // was seen first via the slow path, and now a regular tensor with the
      // same dtype arrives via the fast path. Reuse the existing code to
      // avoid orphaning cache entries keyed by the slow-path code.
      PyObject *dtype_obj = PyObject_GetAttr(arg, dtype_attr);
      if (!dtype_obj) {
        PyErr_Clear();
        return TC_UNSUPPORTED;
      }
      Py_hash_t h = PyObject_Hash(dtype_obj);
      Py_DECREF(dtype_obj);
      if (h == -1) {
        PyErr_Clear();
        return TC_UNSUPPORTED;
      }
      auto it = fc_dtype_to_code.find(h);
      if (it != fc_dtype_to_code.end()) {
        // Reuse existing code from slow path
        code = it->second;
        scalar_type_to_fc_code[st] = code;
        scalar_type_map_initialized[st] = true;
        return is_const ? (TC_PTR_CONST_BASE + code) : (TC_PTR_BASE + code);
      }
      // No existing mapping — allocate a fresh code
      code = fc_next_dtype_code++;
      if (code > 30)
        return TC_UNSUPPORTED;
      scalar_type_to_fc_code[st] = code;
      scalar_type_map_initialized[st] = true;
      fc_dtype_to_code[h] = code;
    }
    return is_const ? (TC_PTR_CONST_BASE + code) : (TC_PTR_BASE + code);
  }
slow_path:
  PyObject *dtype_obj = PyObject_GetAttr(arg, dtype_attr);
  if (!dtype_obj) {
    PyErr_Clear();
    return TC_UNSUPPORTED;
  }
  Py_hash_t h = PyObject_Hash(dtype_obj);
  Py_DECREF(dtype_obj);
  if (h == -1) {
    PyErr_Clear();
    return TC_UNSUPPORTED;
  }

  auto it = fc_dtype_to_code.find(h);
  uint8_t code;
  if (it != fc_dtype_to_code.end()) {
    code = it->second;
  } else {
    // Thread-safety: no race here — all callers hold the GIL (CPython C-API),
    // so fc_next_dtype_code++ and fc_dtype_to_code[] writes are serialized.
    code = fc_next_dtype_code++;
    if (code > 30)
      return TC_UNSUPPORTED;
    fc_dtype_to_code[h] = code;
  }
  return is_const ? (TC_PTR_CONST_BASE + code) : (TC_PTR_BASE + code);
}

static int fc_get_tensor_alignment(PyObject *arg) {
  // Fast path: direct struct access via torch_bridge
  if (g_tensor_api) {
    uint64_t ptr = g_tensor_api->get_data_ptr(arg);
    if (ptr != 0)
      return (ptr & 15) == 0 ? 1 : 0;
    // ptr==0: either not a torch tensor or zero-size tensor — fall through
  }
  PyObject *ptr_obj = PyObject_CallMethodNoArgs(arg, data_ptr_attr);
  if (!ptr_obj)
    return -1;
  unsigned long long ptr = PyLong_AsUnsignedLongLong(ptr_obj);
  Py_DECREF(ptr_obj);
  if (PyErr_Occurred())
    return -1;
  return (ptr & 15) == 0 ? 1 : 0;
}

static void fc_capsule_destructor(PyObject *capsule) {
  FastCache *c = (FastCache *)PyCapsule_GetPointer(capsule, "FastCache");
  if (c)
    delete c;
}

static FastCache *fc_get_or_create(PyObject *jit_fn, PyObject *params_list,
                                   int n_params) {
  if (!fc_cache_capsule_attr) {
    fc_cache_capsule_attr = PyUnicode_InternFromString("_fc_cache");
    if (!fc_cache_capsule_attr)
      return nullptr;
  }
  PyObject *capsule = PyObject_GetAttr(jit_fn, fc_cache_capsule_attr);
  if (capsule && PyCapsule_IsValid(capsule, "FastCache")) {
    FastCache *c = (FastCache *)PyCapsule_GetPointer(capsule, "FastCache");
    Py_DECREF(capsule);
    return c;
  }
  PyErr_Clear();
  if (capsule)
    Py_DECREF(capsule);

  FastCache *cache = new FastCache();
  cache->n_params = n_params;

  for (int i = 0; i < n_params && i < FC_MAX_ARGS; i++) {
    PyObject *param = PyList_GetItem(params_list, i);
    if (!param) {
      PyErr_Clear();
      break;
    }

    PyObject *val;
    val = PyObject_GetAttrString(param, "is_constexpr");
    if (val) {
      cache->param_meta[i].is_constexpr = PyObject_IsTrue(val);
      Py_DECREF(val);
    } else
      PyErr_Clear();

    val = PyObject_GetAttrString(param, "do_not_specialize");
    if (val) {
      cache->param_meta[i].do_not_specialize = PyObject_IsTrue(val);
      Py_DECREF(val);
    } else
      PyErr_Clear();

    val = PyObject_GetAttrString(param, "do_not_specialize_on_alignment");
    if (val) {
      cache->param_meta[i].do_not_specialize_on_alignment =
          PyObject_IsTrue(val);
      Py_DECREF(val);
    } else
      PyErr_Clear();

    val = PyObject_GetAttrString(param, "annotation_type");
    if (val && val != Py_None) {
      const char *s = PyUnicode_AsUTF8(val);
      if (s) {
        cache->param_meta[i].has_annotation = 1;
        if (!strcmp(s, "i32"))
          cache->param_meta[i].annotation_type_code = TC_I32;
        else if (!strcmp(s, "i64"))
          cache->param_meta[i].annotation_type_code = TC_I64;
        else if (!strcmp(s, "u64"))
          cache->param_meta[i].annotation_type_code = TC_U64;
        else if (!strcmp(s, "fp32"))
          cache->param_meta[i].annotation_type_code = TC_FP32;
        else if (!strcmp(s, "u1"))
          cache->param_meta[i].annotation_type_code = TC_U1;
        else
          cache->param_meta[i].has_annotation = 0;
      }
    }
    if (val)
      Py_DECREF(val);
    else
      PyErr_Clear();

    // Check if this is a pointer param (annotation starts with '*')
    // or a tensordesc param (annotation starts with 'tensordesc')
    val = PyObject_GetAttrString(param, "annotation");
    if (val && val != Py_None) {
      const char *s = PyUnicode_AsUTF8(val);
      if (s && s[0] == '*') {
        cache->param_meta[i].is_ptr = 1;
      } else if (s) {
        char lower[11] = {};
        for (int j = 0; j < 10 && s[j]; j++)
          lower[j] = (s[j] >= 'A' && s[j] <= 'Z') ? s[j] + 32 : s[j];
        if (strncmp(lower, "tensordesc", 10) == 0)
          cache->param_meta[i].is_tensordesc = 1;
      }
    }
    if (val)
      Py_DECREF(val);
    else
      PyErr_Clear();
  }

  capsule = PyCapsule_New(cache, "FastCache", fc_capsule_destructor);
  if (!capsule) {
    delete cache;
    return nullptr;
  }
  PyObject_SetAttr(jit_fn, fc_cache_capsule_attr, capsule);
  Py_DECREF(capsule);
  return cache;
}

// Hash a TensorDescriptor for the fast cache key.
// Combines hash(arg.base.dtype) with hash(tuple(arg.block_shape)) to match
// the specialization semantics of the Python path (specialize_tensordesc).
static Py_hash_t fc_hash_tensordesc(PyObject *arg) {
  PyObject *base = PyObject_GetAttrString(arg, "base");
  if (!base)
    return -1;
  PyObject *dtype = PyObject_GetAttrString(base, "dtype");
  Py_DECREF(base);
  if (!dtype)
    return -1;
  Py_hash_t h1 = PyObject_Hash(dtype);
  Py_DECREF(dtype);
  if (h1 == -1)
    return -1;

  PyObject *block_shape = PyObject_GetAttrString(arg, "block_shape");
  if (!block_shape)
    return -1;
  PyObject *tup = PySequence_Tuple(block_shape);
  Py_DECREF(block_shape);
  if (!tup)
    return -1;
  Py_hash_t h2 = PyObject_Hash(tup);
  Py_DECREF(tup);
  if (h2 == -1)
    return -1;

  // Mix hashes to reduce collisions
  Py_hash_t combined = h1 ^ (h2 * (Py_hash_t)0x9e3779b97f4a7c15ULL);
  return combined == -1 ? -2 : combined; // -1 is reserved for error
}

// Build cache key from args. Returns false if unsupported arg type encountered.
static bool fc_build_key(FCCacheKey &key, FastCache *cache,
                         PyObject *const *call_args, int n_args,
                         uint64_t opts_hash) {
  key.n_args = (uint16_t)n_args;
  key.options_hash = opts_hash;
  memset(key.slots, 0, n_args * sizeof(ArgKeySlot));

  for (int i = 0; i < n_args; i++) {
    PyObject *arg = call_args[i];
    if (!arg)
      return false;
    ParamMeta &meta = cache->param_meta[i];

    if (meta.is_constexpr) {
      key.slots[i].type_code = TC_CONSTEXPR;
      Py_hash_t h = PyObject_Hash(arg);
      if (h == -1) {
        PyErr_Clear();
        return false;
      }
      key.slots[i].constexpr_hash = h;
    } else if (PyBool_Check(arg)) {
      key.slots[i].type_code = TC_U1;
      key.slots[i].align_bit = 255;
    } else if (PyLong_Check(arg)) {
      bool spec = !meta.do_not_specialize;
      bool align = !meta.do_not_specialize_on_alignment;
      int overflow;
      long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
      }
      if (spec && val == 1) {
        key.slots[i].type_code = TC_CONSTEXPR;
        key.slots[i].constexpr_hash = 1;
      } else if (overflow == 0) {
        key.slots[i].type_code =
            (val >= INT32_MIN && val <= INT32_MAX) ? TC_I32 : TC_I64;
        key.slots[i].align_bit =
            spec ? ((align && (val & 15) == 0) ? 1 : 0) : 255;
      } else {
        key.slots[i].type_code = TC_U64;
        if (spec) {
          unsigned long long uval = PyLong_AsUnsignedLongLong(arg);
          if (PyErr_Occurred()) {
            PyErr_Clear();
            return false;
          }
          key.slots[i].align_bit = (align && (uval & 15) == 0) ? 1 : 0;
        } else {
          key.slots[i].align_bit = 255;
        }
      }
    } else if (PyFloat_Check(arg)) {
      key.slots[i].type_code =
          meta.has_annotation ? meta.annotation_type_code : TC_FP32;
      key.slots[i].align_bit = 255;
    } else if (Py_IsNone(arg)) {
      key.slots[i].type_code = TC_CONSTEXPR;
      key.slots[i].constexpr_hash = PyObject_Hash(Py_None);
    } else if (meta.is_ptr) {
      // Tensor/pointer-like (determined at cache init from annotation)
      uint8_t tc = fc_get_tensor_type_code(arg, false);
      if (tc == TC_UNSUPPORTED)
        return false;
      key.slots[i].type_code = tc;
      bool spec = !meta.do_not_specialize;
      bool align_flag = !meta.do_not_specialize_on_alignment;
      if (spec && align_flag) {
        int a = fc_get_tensor_alignment(arg);
        if (a < 0) {
          PyErr_Clear();
          return false;
        }
        key.slots[i].align_bit = (uint8_t)a;
      } else if (spec) {
        key.slots[i].align_bit = 0;
      } else {
        key.slots[i].align_bit = 255;
      }
    } else if (meta.is_tensordesc) {
      // TensorDescriptor — key by dtype + block_shape (matches Python path).
      // Uses TC_TENSORDESC (not TC_CONSTEXPR) to avoid constexpr equality
      // check, which would fail because TensorDescriptor.__eq__ compares
      // base tensors element-wise (raises on bool conversion).
      key.slots[i].type_code = TC_TENSORDESC;
      Py_hash_t h = fc_hash_tensordesc(arg);
      if (h == -1) {
        PyErr_Clear();
        return false;
      }
      key.slots[i].constexpr_hash = h;
    } else {
      // No metadata match — try detecting tensor-like objects.
      // With _torch_bridge loaded, this is ~10x cheaper than Python attr
      // lookups (direct THPVariable struct access). Without the bridge,
      // falls back to PyObject_GetAttr which is ~80-150ns per call.
      uint8_t tc = fc_get_tensor_type_code(arg, false);
      if (tc != TC_UNSUPPORTED) {
        meta.is_ptr = 1; // Cache for future calls → direct is_ptr branch
        key.slots[i].type_code = tc;
        bool spec = !meta.do_not_specialize;
        bool align_flag = !meta.do_not_specialize_on_alignment;
        if (spec && align_flag) {
          int a = fc_get_tensor_alignment(arg);
          if (a < 0) {
            PyErr_Clear();
            return false;
          }
          key.slots[i].align_bit = (uint8_t)a;
        } else if (spec) {
          key.slots[i].align_bit = 0;
        } else {
          key.slots[i].align_bit = 255;
        }
      } else {
        // fc_get_tensor_type_code failed — try detecting TensorDescriptor.
        // TensorDescriptor has .base.dtype (not .dtype directly), so the
        // tensor probe above fails. Detect by checking for .base + .block_shape
        // and handle like meta.is_tensordesc to avoid false cache hits from
        // zeroed slots that can't distinguish different TensorDescriptor types.
        Py_hash_t h = fc_hash_tensordesc(arg);
        if (h != -1) {
          // It's a TensorDescriptor — cache this for future calls
          meta.is_tensordesc = 1;
          key.slots[i].type_code = TC_TENSORDESC;
          key.slots[i].constexpr_hash = h;
        } else {
          // Not a TensorDescriptor either — truly unknown type.
          // Return false to force cache miss (falling back to Python slow
          // path) rather than leaving a zeroed slot that would produce
          // false cache hits for different arg types at this position.
          PyErr_Clear();
          return false;
        }
      }
    }
  }
  return true;
}

// native_fast_dispatch(jit_fn, args_tuple, params_list, options_hash,
// grid_tuple, stream) On cache hit with dispatcher: calls dispatcher(grid_0,
// grid_1, grid_2, stream, *args)
//   and returns kernel.
// On cache hit without dispatcher: returns kernel (Python handles
// launch_metadata path). On miss: returns None.
PyObject *native_fast_dispatch(PyObject *self, PyObject *const *args,
                               Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "native_fast_dispatch expects 6 arguments");
    return nullptr;
  }

  PyObject *jit_fn = args[0];
  PyObject *call_args_tuple = args[1];
  PyObject *params_list = args[2];
  PyObject *options_hash_obj = args[3];
  PyObject *grid_tuple = args[4];
  PyObject *stream_obj = args[5];

  if (!PyTuple_Check(call_args_tuple))
    Py_RETURN_NONE;
  Py_ssize_t n = PyTuple_GET_SIZE(call_args_tuple);
  if (n > FC_MAX_ARGS)
    Py_RETURN_NONE;

  uint64_t opts_hash = PyLong_AsUnsignedLongLong(options_hash_obj);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    Py_RETURN_NONE;
  }

  int np = (int)PyList_Size(params_list);
  FastCache *cache = fc_get_or_create(jit_fn, params_list, np);
  if (!cache) {
    PyErr_Clear();
    Py_RETURN_NONE;
  }

  // If cache is empty, skip key building — nothing to match against.
  // Python will handle this call and insert later.
  if (!cache->table || cache->count == 0)
    Py_RETURN_NONE;

  // Use a thread_local key to avoid 1KB+ stack allocation per call
  FCCacheKey key;
  PyObject *const *ca = &PyTuple_GET_ITEM(call_args_tuple, 0);
  if (!fc_build_key(key, cache, ca, (int)n, opts_hash))
    Py_RETURN_NONE;

  // Lookup: returns entry on hit, nullptr on miss
  FCEntry *entry = cache->lookup(key, ca);
  if (!entry)
    Py_RETURN_NONE;

  PyObject *kernel = entry->kernel;
  PyObject *dispatcher = entry->dispatcher;

  // Cache hit with dispatcher: dispatch entirely in C (no Python overhead)
  if (dispatcher) {
    if (!PyTuple_Check(grid_tuple))
      Py_RETURN_NONE; // Non-tuple grid (e.g. kernel[1]) — let Python handle it
    Py_ssize_t grid_n = PyTuple_GET_SIZE(grid_tuple);
    // Determine kernel args count and build vectorcall args
    int n_kernel_args = 0;
    if (entry->n_dispatch_args > 0) {
      n_kernel_args = entry->n_dispatch_args;
    } else {
      for (Py_ssize_t i = 0; i < n && i < cache->n_params; i++) {
        if (!cache->param_meta[i].is_constexpr)
          n_kernel_args++;
      }
    }
    Py_ssize_t vc_nargs = 3 + 1 + n_kernel_args;
    PyObject **vc_args = (PyObject **)alloca(vc_nargs * sizeof(PyObject *));
    static PyObject *one = PyLong_FromLong(1);
    vc_args[0] = grid_n > 0 ? PyTuple_GET_ITEM(grid_tuple, 0) : one;
    vc_args[1] = grid_n > 1 ? PyTuple_GET_ITEM(grid_tuple, 1) : one;
    vc_args[2] = grid_n > 2 ? PyTuple_GET_ITEM(grid_tuple, 2) : one;
    vc_args[3] = stream_obj;
    if (entry->n_dispatch_args > 0) {
      // Use stored dispatch_arg_indices to select only the args the dispatcher
      // expects
      for (int j = 0; j < entry->n_dispatch_args; j++)
        vc_args[4 + j] = ca[entry->dispatch_arg_indices[j]];
    } else {
      // Legacy path: pass all non-constexpr args
      int ki = 0;
      for (Py_ssize_t i = 0; i < n && i < cache->n_params; i++) {
        if (!cache->param_meta[i].is_constexpr)
          vc_args[4 + ki++] = ca[i];
      }
    }
    // Guard: clear stale exceptions before calling the dispatcher to prevent
    // SystemError ("returned a result with an exception set") in td_get_ptr.
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return nullptr;
    }
    PyObject *result =
        PyObject_Vectorcall(dispatcher, vc_args, vc_nargs, nullptr);
    if (!result) {
      // Dispatcher vectorcall failed — propagate the error rather than
      // silently falling back (the caller would not re-launch the kernel).
      return nullptr;
    }
    Py_DECREF(result);
    Py_INCREF(kernel);
    return kernel;
  }

  // Cache hit but no dispatcher — return kernel for Python to launch.
  Py_INCREF(kernel);
  return kernel;
}

// native_fast_dispatch_insert(jit_fn, args_tuple, params_list, options_hash,
// kernel, dispatcher, dispatch_indices)
PyObject *native_fast_dispatch_insert(PyObject *self, PyObject *const *args,
                                      Py_ssize_t nargs) {
  if (nargs != 7) {
    PyErr_SetString(PyExc_TypeError,
                    "native_fast_dispatch_insert expects 7 arguments");
    return nullptr;
  }

  PyObject *jit_fn = args[0];
  PyObject *call_args_tuple = args[1];
  PyObject *params_list = args[2];
  PyObject *options_hash_obj = args[3];
  PyObject *kernel = args[4];
  PyObject *dispatcher = args[5];
  PyObject *dispatch_indices = args[6];

  if (!PyTuple_Check(call_args_tuple))
    Py_RETURN_NONE;
  Py_ssize_t n = PyTuple_GET_SIZE(call_args_tuple);
  if (n > FC_MAX_ARGS)
    Py_RETURN_NONE;

  uint64_t opts_hash = PyLong_AsUnsignedLongLong(options_hash_obj);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    Py_RETURN_NONE;
  }

  int np = (int)PyList_Size(params_list);
  FastCache *cache = fc_get_or_create(jit_fn, params_list, np);
  if (!cache) {
    PyErr_Clear();
    Py_RETURN_NONE;
  }

  FCCacheKey key;
  PyObject *const *ca = &PyTuple_GET_ITEM(call_args_tuple, 0);
  if (!fc_build_key(key, cache, ca, (int)n, opts_hash))
    Py_RETURN_NONE;

  PyObject *disp = (dispatcher == Py_None) ? nullptr : dispatcher;
  cache->insert(key, kernel, disp, ca, (int)n);

  // Store dispatch_arg_indices if provided
  if (dispatch_indices != Py_None && PyTuple_Check(dispatch_indices)) {
    Py_ssize_t n_indices = PyTuple_GET_SIZE(dispatch_indices);
    if (n_indices > 0) {
      int *indices = (int *)malloc(n_indices * sizeof(int));
      if (indices) {
        for (Py_ssize_t i = 0; i < n_indices; i++) {
          indices[i] =
              (int)PyLong_AsLong(PyTuple_GET_ITEM(dispatch_indices, i));
        }
        if (!PyErr_Occurred() && cache->table) {
          // Find the just-inserted entry
          FCCacheKeyHash hasher;
          size_t idx = hasher(key) % cache->capacity;
          bool found = false;
          while (cache->table[idx].occupied) {
            if (cache->table[idx].key == key) {
              cache->set_dispatch_indices(idx, indices, (int)n_indices);
              found = true;
              break;
            }
            idx = (idx + 1) % cache->capacity;
          }
          if (!found)
            free(indices);
        } else {
          PyErr_Clear();
          free(indices);
        }
      }
    }
  }

  Py_RETURN_NONE;
}

/* =========================================================================
 * _JITCacheProxy: C-level proxy returned by __getitem__ when c_cache=True.
 * Eliminates Python preamble overhead by doing cache lookup + dispatch
 * entirely in C via vectorcall.
 * ========================================================================= */
typedef struct {
  PyObject_HEAD vectorcallfunc vectorcall;
  PyObject *jit_fn;      // JITFunction (for cache access)
  PyObject *params_list; // self.params
  PyObject *run_partial; // functools.partial(self.run, grid=grid, warmup=False)
  PyObject *grid_py[3];  // pre-extracted grid PyLong objects
  PyObject *stream_getter;     // driver.active.get_current_stream
  PyObject *device_getter;     // driver.active.get_current_device
  PyObject *param_name_to_idx; // dict: param_name → positional index
  uint64_t options_hash;
  int n_params;
} JITCacheProxy;

static PyObject *JITCacheProxy_vectorcall(PyObject *callable,
                                          PyObject *const *args, size_t nargsf,
                                          PyObject *kwnames);
static void JITCacheProxy_dealloc(PyObject *o);

static PyObject *JITCacheProxy_vectorcall(PyObject *callable,
                                          PyObject *const *args, size_t nargsf,
                                          PyObject *kwnames) {
  JITCacheProxy *self = (JITCacheProxy *)callable;
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PyObject **merged_args = nullptr;
  PyObject *const *effective_args = args;
  int effective_nargs = (int)nargs;

  // When kwargs are present, merge them into positional args in C.
  // This mirrors the Python-side logic in jit.py run() c_cache path.
  if (kwnames && PyTuple_GET_SIZE(kwnames) > 0) {
    if (!self->param_name_to_idx) {
      goto fallback;
    }
    Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
    int total = self->n_params;
    // Allocate merged array on stack (max ~64 params for typical kernels)
    merged_args = (PyObject **)alloca(total * sizeof(PyObject *));
    // Copy positional args
    for (int i = 0; i < total; i++)
      merged_args[i] = (i < (int)nargs) ? (PyObject *)args[i] : Py_None;
    // Merge kwargs by name lookup
    for (Py_ssize_t ki = 0; ki < nkw; ki++) {
      PyObject *name = PyTuple_GET_ITEM(kwnames, ki);
      PyObject *idx_obj = PyDict_GetItem(self->param_name_to_idx, name);
      if (idx_obj) {
        int idx = (int)PyLong_AsLong(idx_obj);
        if (idx >= 0 && idx < total)
          merged_args[idx] = (PyObject *)args[nargs + ki];
      }
      // kwargs not in param_name_to_idx are "options" — affect hash only.
      // For now, treat them as cache-miss (different options_hash) and
      // fallback.
      else
        goto fallback;
    }
    effective_args = merged_args;
    effective_nargs = total;
  } else if (nargs > self->n_params) {
    goto fallback;
  } else if (nargs < self->n_params) {
    // Pad missing trailing args (constexpr params not passed positionally)
    // with Py_None to match the FastCache insertion key.
    int total = self->n_params;
    merged_args = (PyObject **)alloca(total * sizeof(PyObject *));
    for (int i = 0; i < (int)nargs; i++)
      merged_args[i] = (PyObject *)args[i];
    for (int i = (int)nargs; i < total; i++)
      merged_args[i] = Py_None;
    effective_args = merged_args;
    effective_nargs = total;
  }

  {
    FastCache *cache =
        fc_get_or_create(self->jit_fn, self->params_list, self->n_params);
    if (!cache || !cache->table || cache->count == 0)
      goto fallback;

    FCCacheKey key;
    if (!fc_build_key(key, cache, effective_args, effective_nargs,
                      self->options_hash))
      goto fallback;

    FCEntry *entry = cache->lookup(key, effective_args);
    if (!entry)
      goto fallback;

    PyObject *kernel = entry->kernel;
    PyObject *dispatcher = entry->dispatcher;

    if (!dispatcher)
      goto fallback;

    // Get stream
    PyObject *dev = PyObject_CallNoArgs(self->device_getter);
    if (!dev) {
      PyErr_Clear();
      goto fallback;
    }
    PyObject *stream_obj = PyObject_CallOneArg(self->stream_getter, dev);
    Py_DECREF(dev);
    if (!stream_obj) {
      PyErr_Clear();
      goto fallback;
    }

    // Build dispatcher vectorcall args: grid0, grid1, grid2, stream,
    // *kernel_args
    int n_kernel_args = 0;
    if (entry->n_dispatch_args > 0) {
      n_kernel_args = entry->n_dispatch_args;
    } else {
      for (int i = 0; i < effective_nargs && i < cache->n_params; i++) {
        if (!cache->param_meta[i].is_constexpr)
          n_kernel_args++;
      }
    }
    Py_ssize_t vc_nargs = 3 + 1 + n_kernel_args;
    PyObject **vc_args = (PyObject **)alloca(vc_nargs * sizeof(PyObject *));
    vc_args[0] = self->grid_py[0];
    vc_args[1] = self->grid_py[1];
    vc_args[2] = self->grid_py[2];
    vc_args[3] = stream_obj;
    if (entry->n_dispatch_args > 0) {
      for (int j = 0; j < entry->n_dispatch_args; j++)
        vc_args[4 + j] = effective_args[entry->dispatch_arg_indices[j]];
    } else {
      int ki = 0;
      for (int i = 0; i < effective_nargs && i < cache->n_params; i++) {
        if (!cache->param_meta[i].is_constexpr)
          vc_args[4 + ki++] = effective_args[i];
      }
    }
    // Guard: if any prior C-API call leaked a stale exception (e.g., from
    // fc_build_key probing arg types), clear it before calling the dispatcher.
    // Without this, td_get_ptr→data_ptr() may trigger CPython's SystemError:
    // "returned a result with an exception set".
    if (PyErr_Occurred()) {
      PyErr_Clear();
      Py_DECREF(stream_obj);
      goto fallback;
    }
    PyObject *result =
        PyObject_Vectorcall(dispatcher, vc_args, vc_nargs, nullptr);
    Py_DECREF(stream_obj);
    if (!result) {
      // Propagate error — dispatcher may have partially launched.
      // Do NOT fallback (would risk double-launch).
      return nullptr;
    }
    Py_DECREF(result);
    Py_INCREF(kernel);
    return kernel;
  }

fallback:
  // Fall through to Python: self.run(*args, grid=grid, warmup=False, **kwargs)
  // Use Vectorcall to preserve any keyword arguments from kwnames.
  return PyObject_Vectorcall(self->run_partial, args, nargsf, kwnames);
}

static void JITCacheProxy_dealloc(PyObject *o) {
  PyObject_GC_UnTrack(o);
  JITCacheProxy *self = (JITCacheProxy *)o;
  Py_XDECREF(self->jit_fn);
  Py_XDECREF(self->params_list);
  Py_XDECREF(self->run_partial);
  Py_XDECREF(self->grid_py[0]);
  Py_XDECREF(self->grid_py[1]);
  Py_XDECREF(self->grid_py[2]);
  Py_XDECREF(self->stream_getter);
  Py_XDECREF(self->device_getter);
  Py_XDECREF(self->param_name_to_idx);
  Py_TYPE(o)->tp_free(o);
}

static int JITCacheProxy_traverse(PyObject *o, visitproc visit, void *arg) {
  JITCacheProxy *self = (JITCacheProxy *)o;
  Py_VISIT(self->jit_fn);
  Py_VISIT(self->params_list);
  Py_VISIT(self->run_partial);
  Py_VISIT(self->grid_py[0]);
  Py_VISIT(self->grid_py[1]);
  Py_VISIT(self->grid_py[2]);
  Py_VISIT(self->stream_getter);
  Py_VISIT(self->device_getter);
  Py_VISIT(self->param_name_to_idx);
  return 0;
}

static int JITCacheProxy_clear(PyObject *o) {
  JITCacheProxy *self = (JITCacheProxy *)o;
  Py_CLEAR(self->jit_fn);
  Py_CLEAR(self->params_list);
  Py_CLEAR(self->run_partial);
  Py_CLEAR(self->param_name_to_idx);
  Py_CLEAR(self->grid_py[0]);
  Py_CLEAR(self->grid_py[1]);
  Py_CLEAR(self->grid_py[2]);
  Py_CLEAR(self->stream_getter);
  Py_CLEAR(self->device_getter);
  return 0;
}

static PyTypeObject JITCacheProxyType = {PyVarObject_HEAD_INIT(NULL, 0)};

static void _init_jit_cache_proxy_type() {
  JITCacheProxyType.tp_name = "triton._C.libtriton._JITCacheProxy";
  JITCacheProxyType.tp_basicsize = sizeof(JITCacheProxy);
  JITCacheProxyType.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_HAVE_GC;
  JITCacheProxyType.tp_vectorcall_offset = offsetof(JITCacheProxy, vectorcall);
  JITCacheProxyType.tp_call = PyVectorcall_Call;
  JITCacheProxyType.tp_dealloc = JITCacheProxy_dealloc;
  JITCacheProxyType.tp_traverse = JITCacheProxy_traverse;
  JITCacheProxyType.tp_clear = JITCacheProxy_clear;
}

// native_create_jit_proxy(jit_fn, grid_tuple, params_list, options_hash,
// stream_getter, device_getter[, extra_kwargs])
PyObject *native_create_jit_proxy(PyObject *self_unused, PyObject *const *args,
                                  Py_ssize_t nargs) {
  if (nargs < 6 || nargs > 7) {
    PyErr_SetString(PyExc_TypeError,
                    "native_create_jit_proxy expects 6 or 7 arguments");
    return nullptr;
  }
  PyObject *jit_fn = args[0];
  PyObject *grid_tuple = args[1];
  PyObject *params_list = args[2];
  PyObject *options_hash_obj = args[3];
  PyObject *stream_getter = args[4];
  PyObject *device_getter = args[5];
  PyObject *extra_kwargs =
      (nargs >= 7 && args[6] != Py_None) ? args[6] : nullptr;

  uint64_t opts_hash = PyLong_AsUnsignedLongLong(options_hash_obj);
  if (PyErr_Occurred())
    return nullptr;

  int n_params = (int)PyList_Size(params_list);

  // Build run_partial = functools.partial(jit_fn.run, grid=grid_tuple,
  // warmup=False)
  static PyObject *partial_fn = nullptr;
  if (!partial_fn) {
    PyObject *mod = PyImport_ImportModule("functools");
    if (!mod)
      return nullptr;
    partial_fn = PyObject_GetAttrString(mod, "partial");
    Py_DECREF(mod);
    if (!partial_fn)
      return nullptr;
  }
  static PyObject *run_str = nullptr, *grid_str = nullptr,
                  *warmup_str = nullptr, *skip_fc_str = nullptr;
  if (!run_str)
    run_str = PyUnicode_InternFromString("run");
  if (!grid_str)
    grid_str = PyUnicode_InternFromString("grid");
  if (!warmup_str)
    warmup_str = PyUnicode_InternFromString("warmup");
  if (!skip_fc_str)
    skip_fc_str = PyUnicode_InternFromString("_skip_fc");

  PyObject *run_method = PyObject_GetAttr(jit_fn, run_str);
  if (!run_method)
    return nullptr;
  PyObject *kw = PyDict_New();
  if (!kw) {
    Py_DECREF(run_method);
    return nullptr;
  }
  if (PyDict_SetItem(kw, grid_str, grid_tuple) < 0 ||
      PyDict_SetItem(kw, warmup_str, Py_False) < 0 ||
      PyDict_SetItem(kw, skip_fc_str, Py_True) < 0) {
    Py_DECREF(kw);
    Py_DECREF(run_method);
    return nullptr;
  }
  // Merge extra_kwargs (e.g., ctas_per_cga) into the partial's kwargs so that
  // the fallback path passes them to run() for correct compilation options.
  if (extra_kwargs && PyDict_Check(extra_kwargs)) {
    if (PyDict_Merge(kw, extra_kwargs, 1) < 0) {
      Py_DECREF(kw);
      Py_DECREF(run_method);
      return nullptr;
    }
  }
  PyObject *pack = PyTuple_Pack(1, run_method);
  Py_DECREF(run_method);
  if (!pack) {
    Py_DECREF(kw);
    return nullptr;
  }
  PyObject *run_partial = PyObject_Call(partial_fn, pack, kw);
  Py_DECREF(pack);
  Py_DECREF(kw);
  if (!run_partial)
    return nullptr;

  // Extract grid values
  Py_ssize_t gs = PyTuple_Check(grid_tuple) ? PyTuple_GET_SIZE(grid_tuple) : 0;
  static PyObject *one_obj = nullptr;
  if (!one_obj)
    one_obj = PyLong_FromLong(1);

  JITCacheProxy *proxy =
      (JITCacheProxy *)PyObject_GC_New(JITCacheProxy, &JITCacheProxyType);
  if (!proxy) {
    Py_DECREF(run_partial);
    return nullptr;
  }
  proxy->vectorcall = JITCacheProxy_vectorcall;
  proxy->jit_fn = jit_fn;
  Py_INCREF(jit_fn);
  proxy->params_list = params_list;
  Py_INCREF(params_list);
  proxy->run_partial = run_partial;
  proxy->grid_py[0] = (gs > 0) ? PyTuple_GET_ITEM(grid_tuple, 0) : one_obj;
  Py_INCREF(proxy->grid_py[0]);
  proxy->grid_py[1] = (gs > 1) ? PyTuple_GET_ITEM(grid_tuple, 1) : one_obj;
  Py_INCREF(proxy->grid_py[1]);
  proxy->grid_py[2] = (gs > 2) ? PyTuple_GET_ITEM(grid_tuple, 2) : one_obj;
  Py_INCREF(proxy->grid_py[2]);
  proxy->stream_getter = stream_getter;
  Py_INCREF(stream_getter);
  proxy->device_getter = device_getter;
  Py_INCREF(device_getter);
  // Get _param_name_to_idx from jit_fn for kwargs→positional merging
  static PyObject *pnti_str = nullptr;
  if (!pnti_str)
    pnti_str = PyUnicode_InternFromString("_param_name_to_idx");
  PyObject *pnti = PyObject_GetAttr(jit_fn, pnti_str);
  proxy->param_name_to_idx = pnti; // may be NULL if attr missing
  if (!pnti)
    PyErr_Clear();
  proxy->options_hash = opts_hash;
  proxy->n_params = n_params;
  PyObject_GC_Track((PyObject *)proxy);
  return (PyObject *)proxy;
}

/* =========================================================================
 * _AutotuneCacheProxy: C-level proxy for autotuned kernels.
 * Eliminates ~12us of Python overhead in Autotuner.run() by doing key
 * extraction, config lookup, arg merging, and dispatch entirely in C.
 * ========================================================================= */

static constexpr int AT_MAX_KEY_FIELDS = 16;
static constexpr int AT_MAX_CONSTEXPRS = 32;

// One entry in the autotune dispatch table: maps an autotuner key to a
// pre-built dispatch state (constexpr values, options_hash, cached grid).
struct ATEntry {
  PyObject **key_vals; // stored key values for equality check
  int n_key_vals;
  uint64_t key_hash;
  bool occupied;

  // Dispatch state for this config
  PyObject **constexpr_vals; // config constexpr values to merge into full_args
  int *constexpr_positions;  // which slots in full_args they go into
  int n_constexprs;
  uint64_t options_hash; // fc_options_hash for this winning config
  PyObject *pre_hook;    // config pre_hook callable (NULL or Py_None = no hook)
};

static void at_entry_release(ATEntry &e) {
  for (int j = 0; j < e.n_key_vals; j++)
    Py_XDECREF(e.key_vals[j]);
  free(e.key_vals);
  for (int j = 0; j < e.n_constexprs; j++)
    Py_XDECREF(e.constexpr_vals[j]);
  free(e.constexpr_vals);
  free(e.constexpr_positions);
  Py_XDECREF(e.pre_hook);
}

static void at_table_resize(std::vector<ATEntry> &table,
                            std::vector<std::vector<ATEntry>> &retired) {
  size_t new_cap = table.size() * 2;
  std::vector<ATEntry> new_table(new_cap);
  for (size_t i = 0; i < table.size(); i++) {
    if (table[i].occupied) {
      size_t idx = table[i].key_hash % new_cap;
      while (new_table[idx].occupied)
        idx = (idx + 1) % new_cap;
      new_table[idx] = table[i];
    }
  }
  // Retain the old buffer instead of letting move-assignment free it: a
  // concurrent lock-free lookup may still be reading entries from it after
  // releasing the GIL inside PyObject_RichCompareBool. Entries are
  // shallow-copied into new_table (they share the same key_vals/constexpr
  // allocations, which stay owned by the live table), so the retired buffer
  // is kept purely as valid backing memory and never releases entry contents.
  retired.push_back(std::move(table));
  table = std::move(new_table);
}

// Lookup is lock-free. A concurrent insert may resize the table while this
// thread has released the GIL inside PyObject_RichCompareBool. Because resize
// retains (never frees) the old buffer for the proxy's lifetime, reads here can
// never touch freed memory; the worst case is a stale cap/idx producing a
// false-negative that falls back to Python — always correct.
static ATEntry *at_table_lookup(std::vector<ATEntry> &table, size_t count,
                                uint64_t hash, PyObject *const *key_vals,
                                int n_key_vals) {
  if (table.empty() || count == 0)
    return nullptr;
  size_t cap = table.size();
  size_t idx = hash % cap;
  size_t probes = 0;
  while (probes < cap) {
    if (!table[idx].occupied)
      return nullptr;
    if (table[idx].key_hash == hash && table[idx].n_key_vals == n_key_vals) {
      bool eq = true;
      for (int i = 0; i < n_key_vals; i++) {
        int cmp = PyObject_RichCompareBool(table[idx].key_vals[i], key_vals[i],
                                           Py_EQ);
        if (cmp <= 0) {
          if (cmp == -1)
            PyErr_Clear();
          eq = false;
          break;
        }
      }
      if (eq)
        return &table[idx];
    }
    idx = (idx + 1) % cap;
    probes++;
  }
  return nullptr;
}

static void at_table_insert(std::vector<ATEntry> &table, size_t &count,
                            std::vector<std::vector<ATEntry>> &retired,
                            uint64_t hash, PyObject *const *key_vals,
                            int n_key_vals, PyObject *const *constexpr_vals,
                            int *constexpr_positions, int n_constexprs,
                            uint64_t options_hash,
                            PyObject *pre_hook = nullptr) {
  if (table.empty())
    table.resize(16);
  if (count * 4 >= table.size() * 3)
    at_table_resize(table, retired);

  PyObject **kv = (PyObject **)malloc(n_key_vals * sizeof(PyObject *));
  PyObject **cv = n_constexprs
                      ? (PyObject **)malloc(n_constexprs * sizeof(PyObject *))
                      : nullptr;
  int *cp = n_constexprs ? (int *)malloc(n_constexprs * sizeof(int)) : nullptr;
  // Bail out if any allocation failed (the entry is simply not cached, so the
  // Python fallback handles this key). Guard kv only when n_key_vals > 0 since
  // malloc(0) may legitimately return nullptr.
  if ((n_key_vals && !kv) || (n_constexprs && (!cv || !cp))) {
    free(kv);
    free(cv);
    free(cp);
    return;
  }
  for (int i = 0; i < n_key_vals; i++) {
    kv[i] = key_vals[i];
    Py_INCREF(kv[i]);
  }
  for (int i = 0; i < n_constexprs; i++) {
    cv[i] = constexpr_vals[i];
    Py_INCREF(cv[i]);
    cp[i] = constexpr_positions[i];
  }

  size_t cap = table.size();
  size_t idx = hash % cap;
  while (table[idx].occupied)
    idx = (idx + 1) % cap;

  table[idx].key_vals = kv;
  table[idx].n_key_vals = n_key_vals;
  table[idx].key_hash = hash;
  table[idx].constexpr_vals = cv;
  table[idx].constexpr_positions = cp;
  table[idx].n_constexprs = n_constexprs;
  table[idx].options_hash = options_hash;
  table[idx].pre_hook = pre_hook;
  if (pre_hook && pre_hook != Py_None)
    Py_INCREF(pre_hook);
  table[idx].occupied = true;
  count++;
}

typedef struct {
  PyObject_HEAD vectorcallfunc vectorcall;
  PyObject *jit_fn;       // inner JITFunction
  PyObject *params_list;  // JIT params (for fc_build_key)
  PyObject *fallback_run; // Python Autotuner.run() bound method
  PyObject *stream_getter;
  PyObject *device_getter;
  PyObject *grid_fn;           // callable grid, or NULL for static
  PyObject *grid_static;       // static grid tuple, or NULL for callable
  PyObject *param_name_to_idx; // dict for kwargs→positional merging (or NULL)
  int *key_indices;            // positions in args for autotuner key
  int n_key_indices;
  int *dtype_indices; // positions of tensor args (for dtype in key)
  int n_dtype_indices;
  int n_params;                  // total params of inner JITFunction
  std::vector<ATEntry> at_table; // autotune dispatch table
  size_t at_count;               // number of occupied entries
  // Old table buffers retained (never freed) across resizes so that a
  // concurrent lock-free lookup that released the GIL mid-compare can never
  // read freed memory. Released only when the proxy is destroyed.
  std::vector<std::vector<ATEntry>> at_retired;
} AutotuneCacheProxy;

static PyObject *AutotuneCacheProxy_vectorcall(PyObject *callable,
                                               PyObject *const *args,
                                               size_t nargsf,
                                               PyObject *kwnames);
static void AutotuneCacheProxy_dealloc(PyObject *o);
static int AutotuneCacheProxy_traverse(PyObject *o, visitproc visit, void *arg);
static int AutotuneCacheProxy_clear(PyObject *o);

static PyObject *AutotuneCacheProxy_vectorcall(PyObject *callable,
                                               PyObject *const *args,
                                               size_t nargsf,
                                               PyObject *kwnames) {
  AutotuneCacheProxy *self = (AutotuneCacheProxy *)callable;
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PyObject *e_pre_hook = nullptr; // owned ref, cleaned up at fallback

  // Handle kwargs: merge into positional array using param_name_to_idx
  PyObject *const *effective_args = args;
  Py_ssize_t effective_nargs = nargs;
  PyObject *merged_buf_storage[FC_MAX_ARGS];
  if (kwnames && PyTuple_GET_SIZE(kwnames) > 0) {
    if (!self->param_name_to_idx) {
      goto fallback;
    }
    Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
    // Size the merged buffer on n_params (like JITCacheProxy) and fully
    // initialize [0, total) before placing kwargs. This avoids leaving gaps
    // when a kwarg targets a high-index param: every slot that effective_args
    // can later read is guaranteed initialized.
    Py_ssize_t total = self->n_params;
    if (total > FC_MAX_ARGS) {
      goto fallback;
    }
    for (Py_ssize_t i = 0; i < total; i++)
      merged_buf_storage[i] = (i < nargs) ? (PyObject *)args[i] : Py_None;
    // Place kwargs at correct positions
    for (Py_ssize_t ki = 0; ki < nkw; ki++) {
      PyObject *name = PyTuple_GET_ITEM(kwnames, ki);
      PyObject *idx_obj =
          PyDict_GetItem(self->param_name_to_idx, name); // borrowed
      if (!idx_obj) {
        goto fallback; // unknown kwarg
      }
      Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
      if (idx < 0 || idx >= total)
        goto fallback;
      merged_buf_storage[idx] = (PyObject *)args[nargs + ki];
    }
    effective_args = merged_buf_storage;
    effective_nargs = total;
  }

  {
    // 1. Build autotuner key: extract key values + dtype codes
    PyObject *key_buf[AT_MAX_KEY_FIELDS + FC_MAX_ARGS]; // stack alloc
    int n_key = 0;
    PyObject
        *dtype_refs[AT_MAX_KEY_FIELDS]; // owned refs to DECREF after lookup
    int n_dtype_refs = 0;

    // Extract key field values from args by pre-computed indices
    for (int i = 0; i < self->n_key_indices; i++) {
      int idx = self->key_indices[i];
      if (idx >= (int)effective_nargs)
        goto fallback;
      key_buf[n_key++] = effective_args[idx];
    }
    // Extract dtype from tensor args
    for (int i = 0; i < self->n_dtype_indices; i++) {
      int idx = self->dtype_indices[i];
      if (idx >= (int)effective_nargs) {
        for (int d = 0; d < n_dtype_refs; d++)
          Py_DECREF(dtype_refs[d]);
        goto fallback;
      }
      PyObject *arg = effective_args[idx];
      // Fast path: check if it's a tensor (has dtype attribute)
      if (g_tensor_api) {
        int8_t st = g_tensor_api->get_scalar_type(arg);
        if (st >= 0) {
          static PyObject *dtype_str = nullptr;
          if (!dtype_str)
            dtype_str = PyUnicode_InternFromString("dtype");
          PyObject *dtype = PyObject_GetAttr(arg, dtype_str);
          if (!dtype) {
            PyErr_Clear();
            for (int d = 0; d < n_dtype_refs; d++)
              Py_DECREF(dtype_refs[d]);
            goto fallback;
          }
          key_buf[n_key++] = dtype;
          dtype_refs[n_dtype_refs++] = dtype; // defer DECREF
          continue;
        }
      }
      // Slow path: try getting dtype attribute
      static PyObject *dtype_str2 = nullptr;
      if (!dtype_str2)
        dtype_str2 = PyUnicode_InternFromString("dtype");
      if (PyObject_HasAttr(arg, dtype_str2)) {
        PyObject *dtype = PyObject_GetAttr(arg, dtype_str2);
        if (!dtype) {
          PyErr_Clear();
          for (int d = 0; d < n_dtype_refs; d++)
            Py_DECREF(dtype_refs[d]);
          goto fallback;
        }
        key_buf[n_key++] = dtype;
        dtype_refs[n_dtype_refs++] = dtype; // defer DECREF
      }
    }

    // 2. Hash the key
    uint64_t hash = 14695981039346656037ULL; // FNV-1a offset
    for (int i = 0; i < n_key; i++) {
      Py_hash_t h = PyObject_Hash(key_buf[i]);
      if (h == -1) {
        PyErr_Clear();
        for (int d = 0; d < n_dtype_refs; d++)
          Py_DECREF(dtype_refs[d]);
        goto fallback;
      }
      hash ^= (uint64_t)h;
      hash *= 1099511628211ULL;
    }

    // 3. Lookup in autotune table
    ATEntry *entry =
        at_table_lookup(self->at_table, self->at_count, hash, key_buf, n_key);
    // Release dtype refs now that lookup is complete
    for (int d = 0; d < n_dtype_refs; d++)
      Py_DECREF(dtype_refs[d]);
    if (!entry) {
      goto fallback;
    }

    // Copy entry fields locally — entry pointer may be invalidated if another
    // thread triggers at_table_insert (which resizes the vector) during any
    // Python call below that releases the GIL.
    PyObject **e_constexpr_vals = entry->constexpr_vals;
    int *e_constexpr_positions = entry->constexpr_positions;
    int e_n_constexprs = entry->n_constexprs;
    uint64_t e_options_hash = entry->options_hash;
    e_pre_hook = entry->pre_hook;
    if (e_pre_hook)
      Py_INCREF(e_pre_hook);

    // 4. Build full_args: user args + constexpr values from config
    PyObject *full_args[FC_MAX_ARGS];
    int full_nargs = self->n_params;
    // Copy user args
    for (int i = 0; i < full_nargs; i++)
      full_args[i] =
          (i < (int)effective_nargs) ? (PyObject *)effective_args[i] : Py_None;
    // Insert constexpr values from the winning config
    for (int i = 0; i < e_n_constexprs; i++)
      full_args[e_constexpr_positions[i]] = e_constexpr_vals[i];

    // 4b. Call pre_hook before grid/fc_build_key — it mutates TensorDescriptors
    // (e.g. block_shape) which affects the FC key hash. Must run before
    // fc_build_key. If the C dispatch fails after this point, the Python
    // fallback will call pre_hook again, but pre_hooks are idempotent.
    if (e_pre_hook) {
      static PyObject *arg_names_str2 = nullptr;
      if (!arg_names_str2)
        arg_names_str2 = PyUnicode_InternFromString("arg_names");
      PyObject *arg_names = PyObject_GetAttr(self->jit_fn, arg_names_str2);
      if (!arg_names) {
        PyErr_Clear();
        goto fallback;
      }
      PyObject *nargs_dict = PyDict_New();
      if (!nargs_dict) {
        Py_DECREF(arg_names);
        goto fallback;
      }
      Py_ssize_t names_len = PyList_GET_SIZE(arg_names);
      for (Py_ssize_t i = 0; i < names_len && i < full_nargs; i++) {
        PyObject *name = PyList_GET_ITEM(arg_names, i);
        PyDict_SetItem(nargs_dict, name, full_args[i]);
      }
      Py_DECREF(arg_names);
      PyObject *hook_result = PyObject_CallOneArg(e_pre_hook, nargs_dict);
      Py_DECREF(nargs_dict);
      if (!hook_result) {
        PyErr_Clear();
        goto fallback;
      }
      Py_DECREF(hook_result);
    }

    // 5. Evaluate grid
    PyObject *grid_tuple = nullptr;
    bool grid_needs_decref = false;
    if (self->grid_static) {
      grid_tuple = self->grid_static;
    } else if (self->grid_fn) {
      // Callable grid must be re-evaluated every call — it may depend on
      // non-key args (e.g. n_elements) that change between calls.
      static PyObject *arg_names_str = nullptr;
      if (!arg_names_str)
        arg_names_str = PyUnicode_InternFromString("arg_names");
      PyObject *arg_names = PyObject_GetAttr(self->jit_fn, arg_names_str);
      if (!arg_names) {
        PyErr_Clear();
        goto fallback;
      }
      PyObject *meta = PyDict_New();
      if (!meta) {
        Py_DECREF(arg_names);
        goto fallback;
      }
      Py_ssize_t names_len = PyList_GET_SIZE(arg_names);
      for (Py_ssize_t i = 0; i < names_len && i < full_nargs; i++) {
        PyObject *name = PyList_GET_ITEM(arg_names, i);
        PyDict_SetItem(meta, name, full_args[i]);
      }
      Py_DECREF(arg_names);
      grid_tuple = PyObject_CallOneArg(self->grid_fn, meta);
      Py_DECREF(meta);
      if (!grid_tuple)
        goto fallback;
      grid_needs_decref = true;
    } else {
      goto fallback;
    }

    // 6. Dispatch via native_fast_dispatch (reuse existing JIT C cache)
    FastCache *cache =
        fc_get_or_create(self->jit_fn, self->params_list, self->n_params);
    if (!cache)
      goto grid_cleanup;

    FCCacheKey fc_key;
    if (!fc_build_key(fc_key, cache, full_args, full_nargs, e_options_hash))
      goto grid_cleanup;

    {
      FCEntry *fc_entry = cache->lookup(fc_key, full_args);
      if (!fc_entry || !fc_entry->dispatcher)
        goto grid_cleanup;

      // Get stream
      PyObject *dev = PyObject_CallNoArgs(self->device_getter);
      if (!dev) {
        PyErr_Clear();
        goto grid_cleanup;
      }
      PyObject *stream_obj = PyObject_CallOneArg(self->stream_getter, dev);
      Py_DECREF(dev);
      if (!stream_obj) {
        PyErr_Clear();
        goto grid_cleanup;
      }

      // Extract grid values
      Py_ssize_t gs =
          PyTuple_Check(grid_tuple) ? PyTuple_GET_SIZE(grid_tuple) : 0;
      static PyObject *one_obj = nullptr;
      if (!one_obj)
        one_obj = PyLong_FromLong(1);
      PyObject *g0 = (gs > 0) ? PyTuple_GET_ITEM(grid_tuple, 0) : one_obj;
      PyObject *g1 = (gs > 1) ? PyTuple_GET_ITEM(grid_tuple, 1) : one_obj;
      PyObject *g2 = (gs > 2) ? PyTuple_GET_ITEM(grid_tuple, 2) : one_obj;

      // Build dispatcher args: grid0, grid1, grid2, stream, *kernel_args
      // Use dispatch_arg_indices when available (handles None pointer args
      // correctly by only passing args the dispatcher actually needs).
      int n_kernel_args = 0;
      if (fc_entry->n_dispatch_args > 0) {
        n_kernel_args = fc_entry->n_dispatch_args;
      } else {
        for (int i = 0; i < full_nargs && i < cache->n_params; i++) {
          if (!cache->param_meta[i].is_constexpr)
            n_kernel_args++;
        }
      }
      Py_ssize_t vc_nargs = 3 + 1 + n_kernel_args;
      PyObject **vc_args = (PyObject **)alloca(vc_nargs * sizeof(PyObject *));
      vc_args[0] = g0;
      vc_args[1] = g1;
      vc_args[2] = g2;
      vc_args[3] = stream_obj;
      if (fc_entry->n_dispatch_args > 0) {
        // Use stored dispatch_arg_indices to select only the args the
        // dispatcher expects (skips None pointers, TMA shadow slots, etc.)
        for (int j = 0; j < fc_entry->n_dispatch_args; j++)
          vc_args[4 + j] = full_args[fc_entry->dispatch_arg_indices[j]];
      } else {
        // Legacy path: pass all non-constexpr args
        int ki = 0;
        for (int i = 0; i < full_nargs && i < cache->n_params; i++) {
          if (!cache->param_meta[i].is_constexpr)
            vc_args[4 + ki++] = full_args[i];
        }
      }
      // Guard: clear stale exceptions before calling the dispatcher to prevent
      // SystemError ("returned a result with an exception set") in td_get_ptr.
      if (PyErr_Occurred()) {
        PyErr_Clear();
        Py_DECREF(stream_obj);
        goto grid_cleanup;
      }
      PyObject *result =
          PyObject_Vectorcall(fc_entry->dispatcher, vc_args, vc_nargs, nullptr);
      Py_DECREF(stream_obj);
      if (grid_needs_decref)
        Py_DECREF(grid_tuple);
      if (!result) {
        Py_XDECREF(e_pre_hook);
        return nullptr;
      }
      Py_DECREF(result);
      Py_XDECREF(e_pre_hook);
      Py_INCREF(fc_entry->kernel);
      return fc_entry->kernel;
    }

  grid_cleanup:
    if (grid_needs_decref)
      Py_DECREF(grid_tuple);
  }

fallback:
  Py_XDECREF(e_pre_hook);
  return PyObject_Vectorcall(self->fallback_run, args, nargsf, kwnames);
}

static void AutotuneCacheProxy_dealloc(PyObject *o) {
  PyObject_GC_UnTrack(o);
  AutotuneCacheProxy *self = (AutotuneCacheProxy *)o;
  Py_XDECREF(self->jit_fn);
  Py_XDECREF(self->params_list);
  Py_XDECREF(self->fallback_run);
  Py_XDECREF(self->stream_getter);
  Py_XDECREF(self->device_getter);
  Py_XDECREF(self->grid_fn);
  Py_XDECREF(self->grid_static);
  Py_XDECREF(self->param_name_to_idx);
  free(self->key_indices);
  free(self->dtype_indices);
  for (size_t i = 0; i < self->at_table.size(); i++) {
    if (self->at_table[i].occupied)
      at_entry_release(self->at_table[i]);
  }
  self->at_table.~vector();
  // Retired buffers hold shallow copies that share allocations with the live
  // table (already released above), so destroy them as raw memory only.
  self->at_retired.~vector();
  Py_TYPE(o)->tp_free(o);
}

static int AutotuneCacheProxy_traverse(PyObject *o, visitproc visit,
                                       void *arg) {
  AutotuneCacheProxy *self = (AutotuneCacheProxy *)o;
  Py_VISIT(self->jit_fn);
  Py_VISIT(self->params_list);
  Py_VISIT(self->fallback_run);
  Py_VISIT(self->stream_getter);
  Py_VISIT(self->device_getter);
  Py_VISIT(self->grid_fn);
  Py_VISIT(self->grid_static);
  Py_VISIT(self->param_name_to_idx);
  for (size_t i = 0; i < self->at_table.size(); i++) {
    if (self->at_table[i].occupied) {
      ATEntry *entry = &self->at_table[i];
      for (int j = 0; j < entry->n_key_vals; j++)
        Py_VISIT(entry->key_vals[j]);
      for (int j = 0; j < entry->n_constexprs; j++)
        Py_VISIT(entry->constexpr_vals[j]);
      Py_VISIT(entry->pre_hook);
    }
  }
  return 0;
}

static int AutotuneCacheProxy_clear(PyObject *o) {
  AutotuneCacheProxy *self = (AutotuneCacheProxy *)o;
  Py_CLEAR(self->jit_fn);
  Py_CLEAR(self->params_list);
  Py_CLEAR(self->fallback_run);
  Py_CLEAR(self->stream_getter);
  Py_CLEAR(self->device_getter);
  Py_CLEAR(self->grid_fn);
  Py_CLEAR(self->grid_static);
  Py_CLEAR(self->param_name_to_idx);
  for (size_t i = 0; i < self->at_table.size(); i++) {
    if (self->at_table[i].occupied) {
      ATEntry *entry = &self->at_table[i];
      for (int j = 0; j < entry->n_key_vals; j++)
        Py_CLEAR(entry->key_vals[j]);
      for (int j = 0; j < entry->n_constexprs; j++)
        Py_CLEAR(entry->constexpr_vals[j]);
      Py_CLEAR(entry->pre_hook);
    }
  }
  return 0;
}

static PyTypeObject AutotuneCacheProxyType = {PyVarObject_HEAD_INIT(NULL, 0)};

static void _init_autotune_cache_proxy_type() {
  AutotuneCacheProxyType.tp_name = "triton._C.libtriton._AutotuneCacheProxy";
  AutotuneCacheProxyType.tp_basicsize = sizeof(AutotuneCacheProxy);
  AutotuneCacheProxyType.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_HAVE_GC;
  AutotuneCacheProxyType.tp_vectorcall_offset =
      offsetof(AutotuneCacheProxy, vectorcall);
  AutotuneCacheProxyType.tp_call = PyVectorcall_Call;
  AutotuneCacheProxyType.tp_dealloc = AutotuneCacheProxy_dealloc;
  AutotuneCacheProxyType.tp_traverse = AutotuneCacheProxy_traverse;
  AutotuneCacheProxyType.tp_clear = AutotuneCacheProxy_clear;
}

// native_create_autotune_proxy(jit_fn, key_indices_list, dtype_indices_list,
//                              params_list, n_params, stream_getter,
//                              device_getter, fallback_run)
PyObject *native_create_autotune_proxy(PyObject *self_unused,
                                       PyObject *const *args,
                                       Py_ssize_t nargs) {
  if (nargs != 8) {
    PyErr_SetString(PyExc_TypeError,
                    "native_create_autotune_proxy expects 8 arguments");
    return nullptr;
  }
  PyObject *jit_fn = args[0];
  PyObject *key_indices_list = args[1];
  PyObject *dtype_indices_list = args[2];
  PyObject *params_list = args[3];
  PyObject *n_params_obj = args[4];
  PyObject *stream_getter = args[5];
  PyObject *device_getter = args[6];
  PyObject *fallback_run = args[7];

  int n_params = (int)PyLong_AsLong(n_params_obj);
  if (PyErr_Occurred())
    return nullptr;

  // Extract key_indices
  Py_ssize_t n_keys = PyList_GET_SIZE(key_indices_list);
  int *key_indices = (int *)malloc(n_keys * sizeof(int));
  if (n_keys && !key_indices) {
    PyErr_NoMemory();
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n_keys; i++)
    key_indices[i] = (int)PyLong_AsLong(PyList_GET_ITEM(key_indices_list, i));

  // Extract dtype_indices
  Py_ssize_t n_dtypes = PyList_GET_SIZE(dtype_indices_list);
  int *dtype_indices = (int *)malloc(n_dtypes * sizeof(int));
  if (n_dtypes && !dtype_indices) {
    free(key_indices);
    PyErr_NoMemory();
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n_dtypes; i++)
    dtype_indices[i] =
        (int)PyLong_AsLong(PyList_GET_ITEM(dtype_indices_list, i));

  if (PyErr_Occurred()) {
    free(key_indices);
    free(dtype_indices);
    return nullptr;
  }

  AutotuneCacheProxy *proxy = (AutotuneCacheProxy *)PyObject_GC_New(
      AutotuneCacheProxy, &AutotuneCacheProxyType);
  if (!proxy) {
    free(key_indices);
    free(dtype_indices);
    return nullptr;
  }

  proxy->vectorcall = AutotuneCacheProxy_vectorcall;
  proxy->jit_fn = jit_fn;
  Py_INCREF(jit_fn);
  proxy->params_list = params_list;
  Py_INCREF(params_list);
  proxy->fallback_run = fallback_run;
  Py_INCREF(fallback_run);
  proxy->stream_getter = stream_getter;
  Py_INCREF(stream_getter);
  proxy->device_getter = device_getter;
  Py_INCREF(device_getter);
  proxy->grid_fn = nullptr;
  proxy->grid_static = nullptr;
  // Get _param_name_to_idx from jit_fn for kwargs→positional merging
  static PyObject *pnti_str = nullptr;
  if (!pnti_str)
    pnti_str = PyUnicode_InternFromString("_param_name_to_idx");
  PyObject *pnti = PyObject_GetAttr(jit_fn, pnti_str);
  proxy->param_name_to_idx = pnti; // may be NULL if attr missing
  if (!pnti)
    PyErr_Clear();
  proxy->key_indices = key_indices;
  proxy->n_key_indices = (int)n_keys;
  proxy->dtype_indices = dtype_indices;
  proxy->n_dtype_indices = (int)n_dtypes;
  proxy->n_params = n_params;
  new (&proxy->at_table) std::vector<ATEntry>();
  proxy->at_count = 0;
  new (&proxy->at_retired) std::vector<std::vector<ATEntry>>();

  PyObject_GC_Track((PyObject *)proxy);
  return (PyObject *)proxy;
}

// native_autotune_proxy_insert(proxy, key_vals_list, constexpr_vals_list,
//                              constexpr_positions_list, options_hash,
//                              pre_hook)
PyObject *native_autotune_proxy_insert(PyObject *self_unused,
                                       PyObject *const *args,
                                       Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "native_autotune_proxy_insert expects 6 arguments");
    return nullptr;
  }
  PyObject *proxy_obj = args[0];
  PyObject *key_vals_list = args[1];
  PyObject *constexpr_vals_list = args[2];
  PyObject *constexpr_positions_list = args[3];
  PyObject *options_hash_obj = args[4];
  PyObject *pre_hook_obj = args[5];

  if (Py_TYPE(proxy_obj) != &AutotuneCacheProxyType) {
    PyErr_SetString(PyExc_TypeError,
                    "First argument must be AutotuneCacheProxy");
    return nullptr;
  }
  AutotuneCacheProxy *proxy = (AutotuneCacheProxy *)proxy_obj;

  uint64_t options_hash = PyLong_AsUnsignedLongLong(options_hash_obj);
  if (PyErr_Occurred())
    return nullptr;

  Py_ssize_t n_key_vals = PyList_GET_SIZE(key_vals_list);
  Py_ssize_t n_ce = PyList_GET_SIZE(constexpr_vals_list);

  if (n_key_vals > AT_MAX_KEY_FIELDS || n_ce > AT_MAX_CONSTEXPRS) {
    Py_RETURN_NONE; // too large, skip C cache silently
  }

  // Build key hash
  uint64_t hash = 14695981039346656037ULL;
  PyObject *key_vals_arr[AT_MAX_KEY_FIELDS];
  for (Py_ssize_t i = 0; i < n_key_vals; i++) {
    key_vals_arr[i] = PyList_GET_ITEM(key_vals_list, i);
    Py_hash_t h = PyObject_Hash(key_vals_arr[i]);
    if (h == -1) {
      PyErr_Clear();
      Py_RETURN_NONE;
    }
    hash ^= (uint64_t)h;
    hash *= 1099511628211ULL;
  }

  // Build constexpr arrays
  PyObject *ce_vals[AT_MAX_CONSTEXPRS];
  int ce_positions[AT_MAX_CONSTEXPRS];
  for (Py_ssize_t i = 0; i < n_ce; i++) {
    ce_vals[i] = PyList_GET_ITEM(constexpr_vals_list, i);
    ce_positions[i] =
        (int)PyLong_AsLong(PyList_GET_ITEM(constexpr_positions_list, i));
  }
  if (PyErr_Occurred())
    Py_RETURN_NONE;

  at_table_insert(proxy->at_table, proxy->at_count, proxy->at_retired, hash,
                  key_vals_arr, (int)n_key_vals, ce_vals, ce_positions,
                  (int)n_ce, options_hash,
                  (pre_hook_obj != Py_None) ? pre_hook_obj : nullptr);
  Py_RETURN_NONE;
}

// native_autotune_proxy_set_grid(proxy, grid)
// Sets the grid on the proxy. If grid is callable, stores as grid_fn.
// If grid is a tuple/int, stores as grid_static.
PyObject *native_autotune_proxy_set_grid(PyObject *self_unused,
                                         PyObject *const *args,
                                         Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(PyExc_TypeError,
                    "native_autotune_proxy_set_grid expects 2 arguments");
    return nullptr;
  }
  PyObject *proxy_obj = args[0];
  PyObject *grid = args[1];

  if (Py_TYPE(proxy_obj) != &AutotuneCacheProxyType) {
    PyErr_SetString(PyExc_TypeError,
                    "First argument must be AutotuneCacheProxy");
    return nullptr;
  }
  AutotuneCacheProxy *proxy = (AutotuneCacheProxy *)proxy_obj;

  if (PyCallable_Check(grid) && !PyTuple_Check(grid)) {
    Py_XDECREF(proxy->grid_fn);
    Py_XDECREF(proxy->grid_static);
    proxy->grid_fn = grid;
    Py_INCREF(grid);
    proxy->grid_static = nullptr;
  } else {
    Py_XDECREF(proxy->grid_fn);
    Py_XDECREF(proxy->grid_static);
    proxy->grid_fn = nullptr;
    proxy->grid_static = nullptr;
    // Normalize to tuple
    PyObject *grid_tuple = grid;
    if (!PyTuple_Check(grid)) {
      grid_tuple = PyTuple_Pack(1, grid);
      if (!grid_tuple)
        return nullptr;
    } else {
      Py_INCREF(grid_tuple);
    }
    proxy->grid_static = grid_tuple;
  }
  Py_RETURN_NONE;
}

bool visit_make_tensordesc_args(PyObject *arg, PyObject *sig,
                                PyObject *relevant_paths,
                                PyObject *tensordesc_meta,
                                bool has_tensordesc_meta, PyObject *base_args,
                                PyObject *make_tensordesc_arg,
                                Py_ssize_t *tensordesc_idx, PyObject *result) {
  assert(PyTuple_Check(sig));
  auto arg_fast =
      from_new_ref(PySequence_Fast(arg, "Expected iterable args node"));
  if (!arg_fast)
    return false;

  Py_ssize_t arg_len = PySequence_Fast_GET_SIZE(arg_fast.ptr());
  Py_ssize_t sig_len = PyTuple_GET_SIZE(sig);
  assert(sig_len == arg_len || !"Invalid signature");
  Py_ssize_t len = arg_len;

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *a = PySequence_Fast_GET_ITEM(arg_fast.ptr(), i);
    PyObject *s = PyTuple_GET_ITEM(sig, i);

    if (PyUnicode_CheckExact(s)) {
      Py_ssize_t size;
      const char *type_cstr = PyUnicode_AsUTF8AndSize(s, &size);
      if (!type_cstr)
        return false;

      // if not s.startswith("tensordesc")
      std::string_view tensordesc = "tensordesc";
      std::string_view type_str(type_cstr, size);
      if (type_str.substr(0, tensordesc.length()) != tensordesc) {
        if (PyList_Append(result, a) < 0)
          return false;
        continue;
      }

      PyObject *meta = Py_None;
      if (has_tensordesc_meta) {
        // Borrowed reference
        meta = PyList_GetItem(tensordesc_meta, *tensordesc_idx);
        if (!meta)
          return false;
      }

      PyObject *vector_args[] = {a, meta, base_args};
      auto desc_args = from_new_ref(
          PyObject_Vectorcall(make_tensordesc_arg, vector_args, 3, nullptr));
      if (!desc_args)
        return false;
      // list.extend(desc_args)
      if (PyList_SetSlice(result, PY_SSIZE_T_MAX, PY_SSIZE_T_MAX,
                          desc_args.ptr()) < 0)
        return false;

      *tensordesc_idx += 1;
      continue;
    }

    auto key = from_new_ref(PyLong_FromSsize_t(i));
    if (!key)
      return false;

    // Borrowed ref
    PyObject *inner_relevant_paths =
        PyDict_GetItemWithError(relevant_paths, key.ptr());
    if (PyErr_Occurred())
      return false;

    if (!inner_relevant_paths) {
      // Short-circuit if tuple doesn't contain any tensordesc args
      if (PyList_Append(result, a) < 0)
        return false;
      continue;
    }

    // Recurse into tuple
    auto inner_res = from_new_ref(PyList_New(0));
    if (!inner_res)
      return false;
    if (!visit_make_tensordesc_args(
            a, s, inner_relevant_paths, tensordesc_meta, has_tensordesc_meta,
            base_args, make_tensordesc_arg, tensordesc_idx, inner_res.ptr()))
      return false;

    auto inner_tuple = from_new_ref(PyList_AsTuple(inner_res.ptr()));
    if (!inner_tuple)
      return false;
    if (PyList_Append(result, inner_tuple.ptr()) < 0)
      return false;
  }
  return true;
}

PyObject *make_tensordesc_args(PyObject *self, PyObject *const *args,
                               Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "make_tensordesc_args expected 6 arguments");
    return nullptr;
  }

  PyObject *kernel_args = args[0];
  PyObject *signature = args[1];
  PyObject *relevant_paths = args[2];
  PyObject *tensordesc_meta = args[3];
  PyObject *base_args = args[4];
  PyObject *make_tensordesc_arg = args[5];

  if (!PyList_CheckExact(tensordesc_meta)) {
    PyErr_SetString(PyExc_TypeError, "Expected tensordesc_meta to be a list");
    return nullptr;
  }
  bool has_tensordesc_meta = PyList_GET_SIZE(tensordesc_meta) > 0;

  auto result = from_new_ref(PyList_New(0));
  if (!result)
    return nullptr;

  Py_ssize_t tensordesc_idx = 0;
  if (!visit_make_tensordesc_args(kernel_args, signature, relevant_paths,
                                  tensordesc_meta, has_tensordesc_meta,
                                  base_args, make_tensordesc_arg,
                                  &tensordesc_idx, result.ptr()))
    return nullptr;

  if (has_tensordesc_meta) {
    Py_ssize_t meta_len = PySequence_Size(tensordesc_meta);
    if (meta_len < 0)
      return nullptr;

    if (tensordesc_idx != meta_len) {
      PyErr_SetString(PyExc_ValueError,
                      "make_tensordesc_args: tensordesc_idx != meta_len");
      return nullptr;
    }
  }

  return result.release().ptr();
}

static PyMethodDef module_methods[] = {
    {"native_specialize_impl", (PyCFunction)specialize_impl, METH_FASTCALL,
     nullptr},
    {"make_tensordesc_args", (PyCFunction)make_tensordesc_args, METH_FASTCALL,
     "Helper to translate tensordesc args"},
    {"native_fast_dispatch", (PyCFunction)native_fast_dispatch, METH_FASTCALL,
     nullptr},
    {"native_fast_dispatch_insert", (PyCFunction)native_fast_dispatch_insert,
     METH_FASTCALL, nullptr},
    {"native_create_jit_proxy", (PyCFunction)native_create_jit_proxy,
     METH_FASTCALL, nullptr},
    {"native_create_autotune_proxy", (PyCFunction)native_create_autotune_proxy,
     METH_FASTCALL, nullptr},
    {"native_autotune_proxy_insert", (PyCFunction)native_autotune_proxy_insert,
     METH_FASTCALL, nullptr},
    {"native_autotune_proxy_set_grid",
     (PyCFunction)native_autotune_proxy_set_grid, METH_FASTCALL, nullptr},
    {"register_tensor_access_api",
     (PyCFunction) + [](PyObject *self, PyObject *arg) -> PyObject * {
       (void)self;
       if (!PyCapsule_IsValid(arg, "triton_tensor_access_api")) {
         PyErr_SetString(
             PyExc_TypeError,
             "Expected a PyCapsule with name 'triton_tensor_access_api'");
         return nullptr;
       }
       g_tensor_api = (TritonTensorAccessAPI *)PyCapsule_GetPointer(
           arg, "triton_tensor_access_api");
       Py_RETURN_NONE;
     },
     METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr} // sentinel
};

} // anonymous namespace

void init_native_specialize(pybind11::module &m) {
  // Initialize JITCacheProxy type
  _init_jit_cache_proxy_type();
  if (PyType_Ready(&JITCacheProxyType) < 0)
    return;
  // Initialize AutotuneCacheProxy type
  _init_autotune_cache_proxy_type();
  if (PyType_Ready(&AutotuneCacheProxyType) < 0)
    return;
  // add functions to module
  PyModule_AddFunctions(m.ptr(), module_methods);
}
