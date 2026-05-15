// _torch_bridge.cpp — Fast tensor accessor bridge for Triton's C fast cache.
//
// This extension links against torch_python to access THPVariable internals
// directly, avoiding Python C-API overhead (PyObject_GetAttr,
// CallMethodNoArgs). It exports a PyCapsule containing function pointers that
// specialize.cc and driver.c consume.

#include <Python.h>
#include <c10/core/ScalarType.h>
#include <stdint.h>
#include <torch/csrc/autograd/python_variable.h>

// ============================================================================
// API struct — shared between this bridge and specialize.cc / driver.c
// ============================================================================

struct TritonTensorAccessAPI {
  // Returns c10::ScalarType as int8_t. Returns -1 if not a valid torch tensor.
  int8_t (*get_scalar_type)(PyObject *obj);
  // Returns data_ptr as uint64_t. Returns 0 if not a valid torch tensor.
  uint64_t (*get_data_ptr)(PyObject *obj);
  // Extract TensorDescriptor fields in one shot (avoids
  // PyObject_GetAttrString). td_obj: a Python dataclass TensorDescriptor
  // instance. out_data_ptr: receives base tensor's data_ptr. out_shape:
  // receives shape values (up to max_ndim). out_strides: receives stride values
  // (up to max_ndim). Returns ndim on success, -1 on failure (not a valid
  // TensorDescriptor).
  int (*extract_tensordesc)(PyObject *td_obj, uint64_t *out_data_ptr,
                            int64_t *out_shape, int64_t *out_strides,
                            int max_ndim);
};

// ============================================================================
// Fast accessor implementations
// ============================================================================

static int8_t fast_get_scalar_type(PyObject *obj) {
  // Use Check (not CheckExact) to support tensor subclasses like
  // nn.Parameter and DTensor. THPVariable_Unpack is safe for all
  // subclasses — they share the same at::Tensor memory layout.
  if (!THPVariable_Check(obj))
    return -1;
  const auto &tensor = THPVariable_Unpack(obj);
  return static_cast<int8_t>(tensor.scalar_type());
}

static uint64_t fast_get_data_ptr(PyObject *obj) {
  if (!THPVariable_CheckExact(obj))
    return 0;
  const auto &tensor = THPVariable_Unpack(obj);
  return reinterpret_cast<uint64_t>(tensor.data_ptr());
}

// Interned attribute name strings for fast dict lookup
static PyObject *s_base = nullptr;
static PyObject *s_shape = nullptr;
static PyObject *s_strides = nullptr;

static int fast_extract_tensordesc(PyObject *td_obj, uint64_t *out_data_ptr,
                                   int64_t *out_shape, int64_t *out_strides,
                                   int max_ndim) {
  // TensorDescriptor is a @dataclass — fields live in __dict__
  PyObject *dict = PyObject_GenericGetDict(td_obj, nullptr);
  if (!dict)
    return -1;

  // Get .base (the PyTorch tensor)
  PyObject *base = PyDict_GetItem(dict, s_base); // borrowed ref
  if (!base || !THPVariable_CheckExact(base)) {
    Py_DECREF(dict);
    return -1;
  }
  const auto &tensor = THPVariable_Unpack(base);
  *out_data_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());

  // Get .shape (Python list of ints)
  PyObject *shape_list = PyDict_GetItem(dict, s_shape); // borrowed ref
  if (!shape_list || !PyList_Check(shape_list)) {
    Py_DECREF(dict);
    return -1;
  }
  Py_ssize_t ndim = PyList_GET_SIZE(shape_list);
  if (ndim > max_ndim) {
    Py_DECREF(dict);
    return -1;
  }
  for (Py_ssize_t i = 0; i < ndim; i++) {
    out_shape[i] = PyLong_AsLongLong(PyList_GET_ITEM(shape_list, i));
  }
  if (PyErr_Occurred()) {
    Py_DECREF(dict);
    return -1;
  }

  // Get .strides (Python list of ints)
  PyObject *strides_list = PyDict_GetItem(dict, s_strides); // borrowed ref
  if (!strides_list || !PyList_Check(strides_list)) {
    Py_DECREF(dict);
    return -1;
  }
  for (Py_ssize_t i = 0; i < ndim; i++) {
    out_strides[i] = PyLong_AsLongLong(PyList_GET_ITEM(strides_list, i));
  }
  if (PyErr_Occurred()) {
    Py_DECREF(dict);
    return -1;
  }

  Py_DECREF(dict);
  return static_cast<int>(ndim);
}

// Singleton API instance
static TritonTensorAccessAPI g_api = {
    fast_get_scalar_type,
    fast_get_data_ptr,
    fast_extract_tensordesc,
};

// ============================================================================
// Python-callable: returns PyCapsule wrapping the API struct
// ============================================================================

static PyObject *get_tensor_access_capsule(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
  return PyCapsule_New(&g_api, "triton_tensor_access_api", nullptr);
}

// ============================================================================
// Module definition
// ============================================================================

static PyMethodDef methods[] = {
    {"get_tensor_access_capsule", get_tensor_access_capsule, METH_NOARGS,
     "Returns a PyCapsule containing fast tensor accessor function pointers."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_torch_bridge",
    "Fast tensor access bridge for Triton (links torch_python)",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__torch_bridge(void) {
  // Intern attribute name strings
  s_base = PyUnicode_InternFromString("base");
  s_shape = PyUnicode_InternFromString("shape");
  s_strides = PyUnicode_InternFromString("strides");
  if (!s_base || !s_shape || !s_strides) {
    Py_XDECREF(s_base);
    Py_XDECREF(s_shape);
    Py_XDECREF(s_strides);
    return nullptr;
  }
  return PyModule_Create(&module_def);
}
