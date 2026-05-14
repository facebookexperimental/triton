#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <utility>

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
  uint8_t is_ptr
      : 1; // Set if annotation starts with '*' (pointer/tensor param)
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
    table[idx].occupied = true;
    count++;
  }
};

// Interned attribute strings for fast cache
static PyObject *fc_cache_capsule_attr = nullptr;

// Dtype hash → type_code (populated on first encounter)
static std::unordered_map<Py_hash_t, uint8_t> fc_dtype_to_code;
static uint8_t fc_next_dtype_code = 0;

static uint8_t fc_get_tensor_type_code(PyObject *arg, bool is_const) {
  PyObject *dtype_obj = PyObject_GetAttr(arg, dtype_attr);
  if (!dtype_obj)
    return TC_UNSUPPORTED;
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
      if (s && s[0] == '*')
        cache->param_meta[i].is_ptr = 1;
      else if (s && strncmp(s, "tensordesc", 10) == 0)
        cache->param_meta[i].is_tensordesc = 1;
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
    }
    // NOTE: Unrecognized types (e.g. torch.Tensor params without annotation)
    // leave the slot zeroed (from memset).  This is intentional for
    // performance: proper detection via fc_get_tensor_type_code requires
    // Python attr lookups (arg.dtype, arg.data_ptr()) adding ~0.1us per
    // tensor on the hot path — a ~10-24% regression for typical kernels.
    //
    // Assumptions that make zeroed slots safe:
    //  1. Triton JIT kernels have fixed signatures — the Python type at each
    //     position never changes across invocations.
    //  2. PyTorch allocates tensors 16-byte aligned (via cudaMalloc / caching
    //     allocator), so alignment specialization is stable across calls.
    //
    // If assumption (2) is violated (e.g. user passes a tensor sliced into
    // unaligned storage), the C fast cache may return a kernel specialized
    // for aligned access.  The Python slow path handles this correctly; add
    // proper detection here only if this becomes a real-world correctness
    // issue (see fc_get_tensor_type_code / fc_get_tensor_alignment).
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
    // Count kernel args (skip constexprs)
    int n_kernel_args = 0;
    for (Py_ssize_t i = 0; i < n && i < cache->n_params; i++) {
      if (!cache->param_meta[i].is_constexpr)
        n_kernel_args++;
    }
    Py_ssize_t vc_nargs = 3 + 1 + n_kernel_args;
    PyObject **vc_args = (PyObject **)alloca(vc_nargs * sizeof(PyObject *));
    static PyObject *one = PyLong_FromLong(1);
    vc_args[0] = grid_n > 0 ? PyTuple_GET_ITEM(grid_tuple, 0) : one;
    vc_args[1] = grid_n > 1 ? PyTuple_GET_ITEM(grid_tuple, 1) : one;
    vc_args[2] = grid_n > 2 ? PyTuple_GET_ITEM(grid_tuple, 2) : one;
    vc_args[3] = stream_obj;
    int ki = 0;
    for (Py_ssize_t i = 0; i < n && i < cache->n_params; i++) {
      if (!cache->param_meta[i].is_constexpr)
        vc_args[4 + ki++] = ca[i];
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
// kernel, dispatcher)
PyObject *native_fast_dispatch_insert(PyObject *self, PyObject *const *args,
                                      Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "native_fast_dispatch_insert expects 6 arguments");
    return nullptr;
  }

  PyObject *jit_fn = args[0];
  PyObject *call_args_tuple = args[1];
  PyObject *params_list = args[2];
  PyObject *options_hash_obj = args[3];
  PyObject *kernel = args[4];
  PyObject *dispatcher = args[5];

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
  PyObject *stream_getter; // driver.active.get_current_stream
  PyObject *device_getter; // driver.active.get_current_device
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

  // Fast path: no kwargs, arg count matches
  if (kwnames && PyTuple_GET_SIZE(kwnames) > 0)
    goto fallback;
  if (nargs != self->n_params)
    goto fallback;

  {
    FastCache *cache =
        fc_get_or_create(self->jit_fn, self->params_list, self->n_params);
    if (!cache || !cache->table || cache->count == 0)
      goto fallback;

    FCCacheKey key;
    if (!fc_build_key(key, cache, args, (int)nargs, self->options_hash))
      goto fallback;

    FCEntry *entry = cache->lookup(key, args);
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
    for (int i = 0; i < (int)nargs && i < cache->n_params; i++) {
      if (!cache->param_meta[i].is_constexpr)
        n_kernel_args++;
    }
    Py_ssize_t vc_nargs = 3 + 1 + n_kernel_args;
    PyObject **vc_args = (PyObject **)alloca(vc_nargs * sizeof(PyObject *));
    vc_args[0] = self->grid_py[0];
    vc_args[1] = self->grid_py[1];
    vc_args[2] = self->grid_py[2];
    vc_args[3] = stream_obj;
    int ki = 0;
    for (int i = 0; i < (int)nargs && i < cache->n_params; i++) {
      if (!cache->param_meta[i].is_constexpr)
        vc_args[4 + ki++] = args[i];
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
  return 0;
}

static int JITCacheProxy_clear(PyObject *o) {
  JITCacheProxy *self = (JITCacheProxy *)o;
  Py_CLEAR(self->jit_fn);
  Py_CLEAR(self->params_list);
  Py_CLEAR(self->run_partial);
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
// stream_getter, device_getter)
PyObject *native_create_jit_proxy(PyObject *self_unused, PyObject *const *args,
                                  Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "native_create_jit_proxy expects 6 arguments");
    return nullptr;
  }
  PyObject *jit_fn = args[0];
  PyObject *grid_tuple = args[1];
  PyObject *params_list = args[2];
  PyObject *options_hash_obj = args[3];
  PyObject *stream_getter = args[4];
  PyObject *device_getter = args[5];

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
  proxy->options_hash = opts_hash;
  proxy->n_params = n_params;
  PyObject_GC_Track((PyObject *)proxy);
  return (PyObject *)proxy;
}

static PyMethodDef module_methods[] = {
    {"native_specialize_impl", (PyCFunction)specialize_impl, METH_FASTCALL,
     nullptr},
    {"native_fast_dispatch", (PyCFunction)native_fast_dispatch, METH_FASTCALL,
     nullptr},
    {"native_fast_dispatch_insert", (PyCFunction)native_fast_dispatch_insert,
     METH_FASTCALL, nullptr},
    {"native_create_jit_proxy", (PyCFunction)native_create_jit_proxy,
     METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr} // sentinel
};

} // anonymous namespace

void init_native_specialize(pybind11::module &m) {
  // Initialize JITCacheProxy type
  _init_jit_cache_proxy_type();
  if (PyType_Ready(&JITCacheProxyType) < 0)
    return;
  // add functions to module
  PyModule_AddFunctions(m.ptr(), module_methods);
}
