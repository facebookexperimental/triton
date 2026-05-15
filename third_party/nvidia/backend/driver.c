#include "cuda.h"
#include <dlfcn.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  _Alignas(alignof(CUtensorMap)) CUtensorMap tensorMap;
} PyCUtensorMapObject;

typedef enum { ARG_CONSTEXPR = 0, ARG_KERNEL = 1, ARG_TUPLE = 2 } ArgType;

// Annotation struct to know how the argument should be handled.
typedef struct {
  PyObject_HEAD;
  PyObject *nested_tuple; // Can be a List of PyKernelArgObjects or None
  ArgType type;
} PyKernelArgObject;

// Deallocator
static void PyKernelArg_dealloc(PyKernelArgObject *self) {
  Py_XDECREF(self->nested_tuple);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Constructor
static int PyKernelArg_init(PyKernelArgObject *self, PyObject *args,
                            PyObject *kwds) {
  static char *kwlist[] = {"nested_tuple", "type", NULL};
  PyObject *tup = NULL;
  int type_val = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &tup,
                                   &type_val)) {
    return -1;
  }
  Py_XINCREF(tup);
  self->nested_tuple = tup;
  self->type = (ArgType)type_val;
  return 0;
}

static void PyKernelArg_free(void *ptr) { free(ptr); }

static PyTypeObject PyKernelArgType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "triton.backends.nvidia.PyKernelArg",
    .tp_basicsize = sizeof(PyKernelArgObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Kernel Argument Metadata",
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyKernelArg_init,
    .tp_dealloc = (destructor)PyKernelArg_dealloc,
};

// Raises a Python exception and returns false if code is not CUDA_SUCCESS.
static bool gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return true;

  const char *prefix = "Triton Error [CUDA]: ";
  const char *str;
  cuGetErrorString(code, &str);
  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return;                                                                  \
  } while (0)

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      goto cleanup;                                                            \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        goto cleanup;                                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  CUdevice device;
  cuDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_num_regs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA_CHECK_AND_RETURN_NULL(
      cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);

cleanup:
  return NULL;
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }
  CUfunction fun;
  CUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  int32_t n_max_threads = 0;
  // create driver handles
  CUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(pctx));
  }

  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuModuleLoadData(&mod, data));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
      &n_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
  // set dynamic shared memory if necessary
  int shared_optin;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
      &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > 49152 && shared_optin > 49152) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
        &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

typedef CUresult (*cuTensorMapEncodeIm2col_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);

#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");      \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libcuda.so.1"); \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }

defineGetFunctionHandle(getCuOccupancyMaxActiveClustersHandle,
                        cuOccupancyMaxActiveClusters);

defineGetFunctionHandle(getCuTensorMapEncodeTiledHandle,
                        cuTensorMapEncodeTiled);

defineGetFunctionHandle(getCuTensorMapEncodeIm2colHandle,
                        cuTensorMapEncodeIm2col);

defineGetFunctionHandle(getLaunchKernelExHandle, cuLaunchKernelEx);

static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
  int clusterDim = -1, maxActiveClusters = -1;
  int shared = 0;
  CUfunction func;

  if (!PyArg_ParseTuple(args, "Kii", &func, &shared, &clusterDim)) {
    return NULL;
  }

  // Let each SM have one block
  int maxActiveBlocks = 1;
  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
  Py_END_ALLOW_THREADS;

  CUlaunchAttribute launchAttr[1];
  launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttr[0].value.clusterDim.x = clusterDim;
  launchAttr[0].value.clusterDim.y = 1;
  launchAttr[0].value.clusterDim.z = 1;
  CUlaunchConfig config;
  config.gridDimX = clusterDim * maxActiveBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared;
  config.hStream = 0;
  config.numAttrs = 1;
  config.attrs = launchAttr;

  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuOccupancyMaxActiveClusters,
                                      getCuOccupancyMaxActiveClustersHandle);

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  Py_END_ALLOW_THREADS;
  return PyLong_FromLong(maxActiveClusters);

cleanup:
  return NULL;
}

static PyObject *setPrintfFifoSize(PyObject *self, PyObject *args) {
  long size;
  if (!PyArg_ParseTuple(args, "l", &size)) {
    return NULL;
  }
  if (size < 0) {
    PyErr_SetString(PyExc_ValueError, "fifo size must be non-negative");
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS;

  // Ensure we have an active context.
  CUcontext ctx = NULL;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(ctx));
  }

  // We can't set the fifo size after running a kernel that calls printf.  This
  // is true even if the set() call is a nop and the new size is the same as the
  // old size.
  //
  // This is unfriendly, so check if the old size matches the new size, and skip
  // the set() call if so.
  size_t oldSize = 0;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuCtxGetLimit(&oldSize, CU_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != size) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, size));
  }

  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

static PyObject *PyCUtensorMap_alloc(PyTypeObject *type, Py_ssize_t n_items) {
  PyCUtensorMapObject *self = NULL;
  void *mem = NULL;
  size_t size = type->tp_basicsize;

  if (posix_memalign(&mem, 128, size) != 0) {
    PyErr_NoMemory();
    return NULL;
  }

  self = (PyCUtensorMapObject *)mem;
  PyObject_INIT(self, type);
  return (PyObject *)self;
}

static void PyCUtensorMap_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

static void PyCUtensorMap_free(void *ptr) { free(ptr); }

// clang-format off
static PyTypeObject PyCUtensorMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.nvidia.PyCUtensorMap",
    .tp_basicsize = sizeof(PyCUtensorMapObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "<PyCUtensorMap object>",
    .tp_new = PyType_GenericNew,
    .tp_alloc = PyCUtensorMap_alloc,
    .tp_dealloc = (destructor)PyCUtensorMap_dealloc,
    .tp_free = PyCUtensorMap_free,
};
// clang-format on

static PyObject *fillTMADescriptorTiled(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  int swizzle;
  int elemSize;
  int elemType;
  PyObject *blockSize;
  PyObject *shape;
  PyObject *strides;
  int padding;

  if (!PyArg_ParseTuple(args, "KiiiOOOi", &global_address, &swizzle, &elemSize,
                        &elemType, &blockSize, &shape, &strides, &padding)) {
    return NULL;
  }

  PyCUtensorMapObject *desc = (PyCUtensorMapObject *)PyObject_CallObject(
      (PyObject *)&PyCUtensorMapType, NULL);
  if (!desc) {
    return NULL;
  }

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;

  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(blockSizeFast);

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      goto cleanup;
    }
    blockSizeInt[rank - i - 1] = PyLong_AsLongLong(item);
  }

  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(shapeFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    shapeInt[rank - i - 1] = PyLong_AsLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i + 1 < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    stridesLL[rank - i - 2] = elemSize * PyLong_AsLongLong(item);
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);
  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;

  CUtensorMapFloatOOBfill fill =
      (padding == 1) ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                     : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  uint32_t elementStrides[5] = {1, 1, 1, 1, 1};
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUresult res = cuTensorMapEncodeTiled(
      &desc->tensorMap, elemType, rank, (void *)global_address, shapeInt,
      stridesLL, blockSizeInt, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
  if (res != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(res, &str);
    char err[4096] = {0};
    size_t off = 0;
    off += snprintf(
        err + off, sizeof(err) - off,
        "Triton Error [CUDA]: Failed to create tensor map descriptor: %s\n",
        str ? str : "Unknown error");
    off += snprintf(err + off, sizeof(err) - off,
                    "elemType=%d rank=%d global_address=0x%llx elemSize=%d "
                    "swizzle=%d padding=%d\n",
                    elemType, rank, (unsigned long long)global_address,
                    elemSize, swizzle, padding);
    off += snprintf(err + off, sizeof(err) - off, "shape=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%llu%s",
                   (unsigned long long)shapeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "strides=[");
    for (int i = 0; i + 1 < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%llu%s",
                      (unsigned long long)stridesLL[i],
                      (i + 2 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "blockSize=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%u%s",
                      (unsigned)blockSizeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] elementStrides=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%u%s",
                      (unsigned)elementStrides[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] fill=%d\n", (int)fill);
    PyErr_SetString(PyExc_RuntimeError, err);

    goto cleanup;
  }

  // Follow the CUTLASS change for the driver version check
  // https://github.com/NVIDIA/cutlass/commit/b7ecaa605dd70326900433695e11ebfec407edd2#diff-1dfcaf77b33258ff3175540718d9caff1cd471215f741ba42943ef00770e6d04
  int driver_version = 0;
  CUresult driver_version_result = cuDriverGetVersion(&driver_version);
  assert(driver_version_result == CUDA_SUCCESS);

  if (driver_version <= 13010) {
    int max_byte_index = 0;
    for (int i = 0; i < rank; ++i) {
      int bytes_stride = i == 0 ? elemSize : stridesLL[i - 1];
      max_byte_index += (shapeInt[i] - 1) * bytes_stride;
    }
    if (max_byte_index + 1 < 128 * 1024) {
      uint64_t *desc_u64 = (uint64_t *)&desc->tensorMap;
      desc_u64[1] &= ~(1llu << 21);
    }
  }

  return (PyObject *)desc;

cleanup:
  Py_XDECREF(blockSizeFast);
  Py_XDECREF(shapeFast);
  Py_XDECREF(stridesFast);
  Py_XDECREF(desc);
  return NULL;
}

static PyObject *fillTMADescriptorIm2col(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  int swizzle;
  int elemSize;
  int elemType;
  PyObject *blockSize;
  PyObject *shape;
  PyObject *strides;
  int padding;
  PyObject *pixelBoxLower;
  PyObject *pixelBoxUpper;
  PyObject *elementStrides;

  if (!PyArg_ParseTuple(args, "KiiiOOOiOOO", &global_address, &swizzle,
                        &elemSize, &elemType, &blockSize, &shape, &strides,
                        &padding, &pixelBoxLower, &pixelBoxUpper,
                        &elementStrides)) {
    return NULL;
  }

  PyCUtensorMapObject *desc = (PyCUtensorMapObject *)PyObject_CallObject(
      (PyObject *)&PyCUtensorMapType, NULL);
  if (!desc) {
    return NULL;
  }

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;
  PyObject *pixelBoxLowerFast = NULL;
  PyObject *pixelBoxUpperFast = NULL;
  PyObject *elementStridesFast = NULL;

  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];
  int pixelBoxLowerInt[5] = {0};
  int pixelBoxUpperInt[5] = {0};
  uint32_t elementStridesInt[5] = {1, 1, 1, 1, 1}; // Default to all 1s

  // For im2col mode, shape determines the tensor rank, not blockSize
  // blockSize is typically 2D [pixelsPerColumn, channelsPerPixel]
  // while shape can be 4D or 5D (e.g., NHWC or NDHWC)
  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(shapeFast);

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    shapeInt[rank - i - 1] = PyLong_AsLong(item);
  }

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int blockRank = PySequence_Fast_GET_SIZE(blockSizeFast);
  if (blockRank != 2) {
    PyErr_SetString(PyExc_RuntimeError,
                    "blockSize must have exactly 2 dimensions for im2col");
    goto cleanup;
  }

  for (int i = 0; i < blockRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      goto cleanup;
    }
    blockSizeInt[blockRank - i - 1] = PyLong_AsLongLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_Format(PyExc_RuntimeError,
                 "Rank mismatch for strides in fillTMADescriptorIm2col: shape "
                 "has rank %d but strides has %zd elements. "
                 "Expected strides to have %d elements.",
                 rank, PySequence_Fast_GET_SIZE(stridesFast), rank);
    goto cleanup;
  }
  for (int i = 0; i + 1 < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "strides must be an int");
      goto cleanup;
    }
    stridesLL[rank - i - 2] = elemSize * PyLong_AsLongLong(item);
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);

  // Parse pixel box lower corner
  pixelBoxLowerFast =
      PySequence_Fast(pixelBoxLower, "pixelBoxLower must be a sequence");
  if (!pixelBoxLowerFast)
    goto cleanup;

  int spatialRank = PySequence_Fast_GET_SIZE(pixelBoxLowerFast);
  if (spatialRank > 5) {
    PyErr_SetString(PyExc_RuntimeError, "Pixel box rank too large (max 5)");
    goto cleanup;
  }

  for (int i = 0; i < spatialRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(pixelBoxLowerFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "pixelBoxLower elements must be int");
      goto cleanup;
    }
    pixelBoxLowerInt[spatialRank - i - 1] = PyLong_AsLong(item);
  }

  // Parse pixel box upper corner
  pixelBoxUpperFast =
      PySequence_Fast(pixelBoxUpper, "pixelBoxUpper must be a sequence");
  if (!pixelBoxUpperFast)
    goto cleanup;

  if (spatialRank != PySequence_Fast_GET_SIZE(pixelBoxUpperFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Pixel box corner rank mismatch");
    goto cleanup;
  }

  for (int i = 0; i < spatialRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(pixelBoxUpperFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "pixelBoxUpper elements must be int");
      goto cleanup;
    }
    pixelBoxUpperInt[spatialRank - i - 1] = PyLong_AsLong(item);
  }

  // Parse element strides
  elementStridesFast =
      PySequence_Fast(elementStrides, "elementStrides must be a sequence");
  if (!elementStridesFast)
    goto cleanup;

  int elementStridesLen = PySequence_Fast_GET_SIZE(elementStridesFast);
  if (elementStridesLen != rank) {
    PyErr_SetString(PyExc_RuntimeError,
                    "elementStrides length must match tensor rank");
    goto cleanup;
  }

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(elementStridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "elementStrides elements must be int");
      goto cleanup;
    }
    elementStridesInt[rank - i - 1] = PyLong_AsLong(item);
  }

  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;
  Py_DECREF(pixelBoxLowerFast);
  pixelBoxLowerFast = NULL;
  Py_DECREF(pixelBoxUpperFast);
  pixelBoxUpperFast = NULL;
  Py_DECREF(elementStridesFast);
  elementStridesFast = NULL;

  CUtensorMapFloatOOBfill fill =
      (padding == 1) ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                     : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  static cuTensorMapEncodeIm2col_t cuTensorMapEncodeIm2col = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeIm2col,
                                      getCuTensorMapEncodeIm2colHandle);

  int channelsPerPixel = blockSizeInt[0];
  int pixelsPerColumn = blockSizeInt[1];

  CUresult res = cuTensorMapEncodeIm2col(
      &desc->tensorMap, elemType, rank, (void *)global_address, shapeInt,
      stridesLL, pixelBoxLowerInt, pixelBoxUpperInt, channelsPerPixel,
      pixelsPerColumn, elementStridesInt, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);

  if (res != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(res, &str);
    char err[4096] = {0};
    size_t off = 0;
    off += snprintf(err + off, sizeof(err) - off,
                    "Triton Error [CUDA]: Failed to create im2col tensor map "
                    "descriptor: %s\n",
                    str ? str : "Unknown error");
    off +=
        snprintf(err + off, sizeof(err) - off,
                 "elemType=%d rank=%d global_address=0x%llx elemSize=%d "
                 "swizzle=%d padding=%d channelsPerPixel=%d "
                 "pixelsPerColumn=%d\n",
                 elemType, rank, (unsigned long long)global_address, elemSize,
                 swizzle, padding, channelsPerPixel, pixelsPerColumn);
    off += snprintf(err + off, sizeof(err) - off, "shape=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%llu%s",
                   (unsigned long long)shapeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "strides=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%llu%s",
                      (unsigned long long)stridesLL[i],
                      (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "blockSize=[");
    for (int i = 0; i < blockRank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%u%s",
                   (unsigned)blockSizeInt[i], (i + 1 < blockRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "pixelBoxLower=[");
    for (int i = 0; i < spatialRank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%d%s", pixelBoxLowerInt[i],
                      (i + 1 < spatialRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] pixelBoxUpper=[");
    for (int i = 0; i < spatialRank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%d%s", pixelBoxUpperInt[i],
                      (i + 1 < spatialRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "elementStrides=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%u%s",
                   (unsigned)elementStridesInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    PyErr_SetString(PyExc_RuntimeError, err);

    goto cleanup;
  }

  return (PyObject *)desc;

cleanup:
  Py_XDECREF(blockSizeFast);
  Py_XDECREF(shapeFast);
  Py_XDECREF(stridesFast);
  Py_XDECREF(pixelBoxLowerFast);
  Py_XDECREF(pixelBoxUpperFast);
  Py_XDECREF(elementStridesFast);
  Py_XDECREF(desc);
  return NULL;
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill1DTMADescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dim;
  uint32_t tensorDim;
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKiiK", &global_address, &dim, &tensorDim,
                        &elementSize, &desc_address)) {
    return NULL;
  }
  uint64_t dims[1] = {dim};
  uint64_t globalStrides[1] = {dim * elementSize};
  uint32_t boxDim[1] = {tensorDim};
  uint32_t elementStrides[1] = {1};
  CUtensorMapDataType type;
  switch (elementSize) {
  case 1:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    break;
  case 2:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    break;
  case 4:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
    return NULL;
  }
  assert((elementSize * tensorDim) >= 32 && "block size too small.");
  int rank = 1;
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK_AND_RETURN_NULL(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, type, rank, (void *)global_address, dims,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  Py_INCREF(Py_None);
  return Py_None;
cleanup:
  return NULL;
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill2DTMADescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dims[2];
  uint32_t tensorDims[2];
  int elementSize;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKKiiiK", &global_address, &dims[1], &dims[0],
                        &tensorDims[1], &tensorDims[0], &elementSize,
                        &desc_address)) {
    return NULL;
  }
  uint64_t globalStrides[2] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize};
  uint32_t elementStrides[2] = {1, 1};
  CUtensorMapDataType type;
  switch (elementSize) {
  case 1:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    break;
  case 2:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    break;
  case 4:
    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  int rank = 2;
  // Swizzling should be picked in codegen but since we need to set it on the
  // descriptor we rely on a convention between this function and codegen.
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  uint32_t contigDimSizeInByte = elementSize * tensorDims[0];
  if (tensorDims[1] < 8 || contigDimSizeInByte < 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (contigDimSizeInByte >= 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if (contigDimSizeInByte >= 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else {
    assert(contigDimSizeInByte >= 32);
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  }
  // The bounding box inner dimension must be less than or equal to the swizzle
  // size.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // We clamp the block size and the codegen will emit multiple copy operations.
  if (contigDimSizeInByte > 128) {
    tensorDims[0] = 128 / elementSize;
  }
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK_AND_RETURN_NULL(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, type, rank, (void *)global_address, dims,
      globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  Py_INCREF(Py_None);
  return Py_None;
cleanup:
  return NULL;
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill1DTMADescriptorType(PyObject *self, PyObject *args) {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
  unsigned long long global_address;
  uint64_t dim;
  uint32_t tensorDim;
  int dataType;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKiiK", &global_address, &dim, &tensorDim,
                        &dataType, &desc_address)) {
    return NULL;
  }
  uint64_t dims[1] = {dim};
  uint32_t boxDim[1] = {tensorDim};
  uint32_t elementStrides[1] = {1};
  CUtensorMapDataType type;
  uint32_t elementSize = 0;
  switch (dataType) {
  case 0:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    elementSize = 2;
    break;
  case 1:
    type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    elementSize = 2;
    break;
  case 2:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    elementSize = 4;
    break;
  case 3:
    type = CU_TENSOR_MAP_DATA_TYPE_INT32;
    elementSize = 4;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "dataType must be 0, 1, 2, or 3");
    return NULL;
  }
  uint64_t globalStrides[1] = {dim * elementSize};
  assert((elementSize * tensorDim) >= 32 && "block size too small.");
  int rank = 1;
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK_AND_RETURN_NULL(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, type, rank, (void *)global_address, dims,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  Py_INCREF(Py_None);
#endif
  return Py_None;
cleanup:
  return NULL;
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill2DTMADescriptorType(PyObject *self, PyObject *args) {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
  unsigned long long global_address;
  uint64_t dims[2];
  uint32_t tensorDims[2];
  int dataType;
  unsigned long long desc_address;
  if (!PyArg_ParseTuple(args, "KKKiiiK", &global_address, &dims[1], &dims[0],
                        &tensorDims[1], &tensorDims[0], &dataType,
                        &desc_address)) {
    return NULL;
  }
  int elementSize = 0;
  CUtensorMapDataType type;
  switch (dataType) {
  case 0:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    elementSize = 2;
    break;
  case 1:
    type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    elementSize = 2;
    break;
  case 2:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    elementSize = 4;
    break;
  case 3:
    type = CU_TENSOR_MAP_DATA_TYPE_INT32;
    elementSize = 4;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "dataType must be 0, 1, 2, or 3");
    return NULL;
  }
  uint64_t globalStrides[2] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize};
  uint32_t elementStrides[2] = {1, 1};
  int rank = 2;
  // Swizzling should be picked in codegen but since we need to set it on the
  // descriptor we rely on a convention between this function and codegen.
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  uint32_t contigDimSizeInByte = elementSize * tensorDims[0];
  if (contigDimSizeInByte >= 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if (contigDimSizeInByte >= 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (contigDimSizeInByte >= 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else {
    assert(false && "block size too small.");
  }
  // The bounding box inner dimension must be less than or equal to the swizzle
  // size.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // We clamp the block size and the codegen will emit multiple copy operations.
  if (contigDimSizeInByte > 128) {
    tensorDims[0] = 128 / elementSize;
  }
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK_AND_RETURN_NULL(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, type, rank, (void *)global_address, dims,
      globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  Py_INCREF(Py_None);
#endif
  return Py_None;
cleanup:
  return NULL;
}

static void ensureCudaContext() {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

// File-level handle for cuLaunchKernelEx (shared by _launch and _TritonDispatcher)
static cuLaunchKernelEx_t g_cuLaunchKernelExHandle = NULL;
static inline cuLaunchKernelEx_t ensureLaunchHandle(void) {
  if (g_cuLaunchKernelExHandle == NULL)
    g_cuLaunchKernelExHandle = getLaunchKernelExHandle();
  return g_cuLaunchKernelExHandle;
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps,
                    int num_ctas, int launch_cooperative_grid, int launch_pdl,
                    int preferredClusterDimX, int preferredClusterDimY,
                    int preferredClusterDimZ, int shared_memory,
                    CUstream stream, CUfunction function, void **params) {
  if (gridX * gridY * gridZ > 0) {
    // 5 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[5];
    cuLaunchKernelEx_t cuLaunchKernelExHandle = ensureLaunchHandle();
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {
      CUlaunchAttribute pdlAttr = {
          .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
          .value = 1};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }

    if (launch_cooperative_grid != 0) {
      CUlaunchAttribute coopAttr = {.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                                    .value = 1};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }

    if (num_ctas != 1 || preferredClusterDimX > 0) {
      // Only set CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION for Triton's num_ctas
      // path. For ctas_per_cga path (num_ctas == 1), PTX's .reqnctapercluster
      // handles it.
      if (num_ctas > 1) {
        CUlaunchAttribute clusterAttr = {};
        clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        clusterAttr.value.clusterDim.x = num_ctas;
        clusterAttr.value.clusterDim.y = 1;
        clusterAttr.value.clusterDim.z = 1;
        launchAttr[num_attrs] = clusterAttr;
        ++num_attrs;
      }

      CUlaunchAttribute clusterSchedulingAttr = {};
      clusterSchedulingAttr.id =
          CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference =
          CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }

#if CUDA_VERSION >= 12080
    if (preferredClusterDimX > 0) {
      CUlaunchAttribute preferredClusterAttr = {};
      preferredClusterAttr.id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
      preferredClusterAttr.value.preferredClusterDim.x = preferredClusterDimX;
      preferredClusterAttr.value.preferredClusterDim.y = preferredClusterDimY;
      preferredClusterAttr.value.preferredClusterDim.z = preferredClusterDimZ;
      launchAttr[num_attrs] = preferredClusterAttr;
      ++num_attrs;
    }
#endif

    // num_ctas == 16 is non-portable. Does work for H100 and B200 tho
    config.numAttrs = num_attrs;
    if (num_ctas == 16) {
      CUDA_CHECK(cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }
}

static PyObject *data_ptr_str = NULL;
static PyObject *td_get_str = NULL;  /* interned "get" for allocator.get() */

// Extract a CUDA device pointer from a pointer-like PyObject obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(void *ptr, PyObject *obj) {
  CUdeviceptr *dev_ptr = ptr;
  if (obj == Py_None) {
    *dev_ptr = (CUdeviceptr)0; // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return true;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(
        PyExc_TypeError,
        "Pointer argument must be either uint64 or have data_ptr method");
    return false;
  }
  if (!PyLong_Check(ret)) {
    PyErr_SetString(PyExc_TypeError,
                    "data_ptr method of Pointer object must return 64-bit int");
    return false;
  }
  *dev_ptr = PyLong_AsUnsignedLongLong(ret);
  Py_DECREF(ret);
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  CUresult status = cuPointerGetAttribute(
      dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument cannot be accessed from Triton "
                 "(cpu tensor?)");
    return false;
  }
  return gpuAssert(status, __FILE__, __LINE__);
}

bool extractI8(void *ptr, PyObject *obj) {
  *((int8_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI16(void *ptr, PyObject *obj) {
  *((int16_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI64(void *ptr, PyObject *obj) {
  *((int64_t *)ptr) = PyLong_AsLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU8(void *ptr, PyObject *obj) {
  *((uint8_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU16(void *ptr, PyObject *obj) {
  *((uint16_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU32(void *ptr, PyObject *obj) {
  *((uint32_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU64(void *ptr, PyObject *obj) {
  *((uint64_t *)ptr) = PyLong_AsUnsignedLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractFP16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  uint16_t result;
  // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 &&            \
    !defined(PYPY_VERSION)
  _PyFloat_Pack2(temp_double, (unsigned char *)&result, 1);
#else
  PyFloat_Pack2(temp_double, (char *)&result, 1);
#endif
  *((uint16_t *)ptr) = result;
  return PyErr_Occurred() == NULL;
}

bool extractBF16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  uint32_t u32 = *(uint32_t *)&f32;
  *((uint16_t *)ptr) = (u32 >> 16);
  return PyErr_Occurred() == NULL;
}

bool extractFP32(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  *((uint32_t *)ptr) = *(uint32_t *)&f32;
  return PyErr_Occurred() == NULL;
}

bool extractFP64(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  *((uint64_t *)ptr) = *(uint64_t *)&temp_double;
  return PyErr_Occurred() == NULL;
}

// Extract a CUtensorMap descriptor from a python object, and store it to the
// memory location pointed by ptr. Supports both PyCUtensorMap objects (from
// fill_tma_descriptor_tiled) and duck-typed wrappers with tma_desc_cpu_ptr()
// (e.g., KernelParamWrapper from fast_moe/fbgemm).
static PyObject *tma_desc_cpu_ptr_str = NULL;

bool extractTmaDesc(void *ptr, PyObject *obj) {
  if (sizeof(CUtensorMap *) != 8) {
    PyErr_SetString(PyExc_SystemError,
                    "getTmaDesc() requires 64-bit compilation");
    return false;
  }

  if (Py_TYPE(obj) == &PyCUtensorMapType) {
    // Fast path: native PyCUtensorMap object
    *((CUtensorMap *)ptr) = ((PyCUtensorMapObject *)obj)->tensorMap;
  } else {
    // Duck-typing fallback: try tma_desc_cpu_ptr() method
    if (!tma_desc_cpu_ptr_str) {
      tma_desc_cpu_ptr_str = PyUnicode_InternFromString("tma_desc_cpu_ptr");
      if (!tma_desc_cpu_ptr_str)
        return false;
    }
    PyObject *host_ptr_obj =
        PyObject_CallMethodNoArgs(obj, tma_desc_cpu_ptr_str);
    if (!host_ptr_obj) {
      // Only replace the error if the method doesn't exist (AttributeError).
      // If the method exists but raised, propagate the real exception.
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "object must be of type PyCUtensorMap or have "
                     "tma_desc_cpu_ptr() method, got %s",
                     Py_TYPE(obj)->tp_name);
      }
      return false;
    }
    uintptr_t host_ptr = (uintptr_t)PyLong_AsUnsignedLongLong(host_ptr_obj);
    Py_DECREF(host_ptr_obj);
    if (PyErr_Occurred())
      return false;
    if (host_ptr == 0) {
      PyErr_SetString(PyExc_ValueError, "tma_desc_cpu_ptr() returned NULL");
      return false;
    }
    memcpy(ptr, (const void *)host_ptr, sizeof(CUtensorMap));
  }

  // Depending on the cuda version, alignof(CUtensorMap) may be 64 or 128.
  size_t alignment = alignof(CUtensorMap);
  uintptr_t remainder = (uintptr_t)ptr & (alignment - 1);
  if (remainder != 0) {
    PyErr_Format(
        PyExc_ValueError,
        "CUtensorMap must be aligned to %ld, but got (&map) mod %ld = %ld",
        alignment, alignment, remainder);
    return false;
  }
  return true;
}

typedef bool (*ExtractorFunc)(void *ptr, PyObject *obj);

#define MAX_NAMES_PER_EXTRACTOR 2

typedef struct {
  ExtractorFunc extract;
  size_t size;
  size_t alignment;
  const char *name[MAX_NAMES_PER_EXTRACTOR];
} Extractor;

typedef enum {
  EXTRACTOR_UNKOWN_INDEX = 0,
  // pointers
  EXTRACTOR_POINTER_INDEX = 1,
  // ints
  EXTRACTOR_INT8_INDEX = 2,
  EXTRACTOR_INT16_INDEX = 3,
  EXTRACTOR_INT32_INDEX = 4,
  EXTRACTOR_INT64_INDEX = 5,
  // uints
  EXTRACTOR_UINT8_INDEX = 6,
  EXTRACTOR_UINT16_INDEX = 7,
  EXTRACTOR_UINT32_INDEX = 8,
  EXTRACTOR_UINT64_INDEX = 9,
  // floats
  EXTRACTOR_FP16_INDEX = 10,
  EXTRACTOR_BF16_INDEX = 11,
  EXTRACTOR_FP32_INDEX = 12,
  EXTRACTOR_FP64_INDEX = 13,
  // custom
  EXTRACTOR_NVTMADESC_INDEX = 14,
  // last entry to have a count
  EXTRACTOR_TYPE_COUNT
} ExtractorTypeIndex;

Extractor extraction_map[EXTRACTOR_TYPE_COUNT] = {
    [EXTRACTOR_UNKOWN_INDEX] =
        (Extractor){.extract = NULL, .size = 0, .name = NULL},
    [EXTRACTOR_POINTER_INDEX] = (Extractor){.extract = extractPointer,
                                            .size = sizeof(CUdeviceptr),
                                            .name = NULL},
    [EXTRACTOR_INT8_INDEX] = (Extractor){.extract = extractI8,
                                         .size = sizeof(int8_t),
                                         .name = {"i8"}},
    [EXTRACTOR_INT16_INDEX] = (Extractor){.extract = extractI16,
                                          .size = sizeof(int16_t),
                                          .name = {"i16"}},
    [EXTRACTOR_INT32_INDEX] = (Extractor){.extract = extractI32,
                                          .size = sizeof(int32_t),
                                          .name = {"i1", "i32"}},
    [EXTRACTOR_INT64_INDEX] = (Extractor){.extract = extractI64,
                                          .size = sizeof(int64_t),
                                          .name = {"i64"}},
    [EXTRACTOR_UINT8_INDEX] = (Extractor){.extract = extractU8,
                                          .size = sizeof(uint8_t),
                                          .name = {"u8"}},
    [EXTRACTOR_UINT16_INDEX] = (Extractor){.extract = extractU16,
                                           .size = sizeof(uint16_t),
                                           .name = {"u16"}},
    [EXTRACTOR_UINT32_INDEX] = (Extractor){.extract = extractU32,
                                           .size = sizeof(uint32_t),
                                           .name = {"u1", "u32"}},
    [EXTRACTOR_UINT64_INDEX] = (Extractor){.extract = extractU64,
                                           .size = sizeof(uint64_t),
                                           .name = {"u64"}},
    [EXTRACTOR_FP16_INDEX] = (Extractor){.extract = extractFP16,
                                         .size = sizeof(uint16_t),
                                         .name = {"fp16"}},
    [EXTRACTOR_BF16_INDEX] = (Extractor){.extract = extractBF16,
                                         .size = sizeof(uint16_t),
                                         .name = {"bf16"}},
    [EXTRACTOR_FP32_INDEX] = (Extractor){.extract = extractFP32,
                                         .size = sizeof(uint32_t),
                                         .name = {"fp32", "f32"}},
    [EXTRACTOR_FP64_INDEX] = (Extractor){.extract = extractFP64,
                                         .size = sizeof(uint64_t),
                                         .name = {"fp64"}},
    [EXTRACTOR_NVTMADESC_INDEX] = (Extractor){.extract = extractTmaDesc,
                                              .size = sizeof(CUtensorMap),
                                              .alignment = alignof(CUtensorMap),
                                              .name = {"nvTmaDesc"}},
};

Extractor getExtractor(uint8_t index) {
  if (index >= EXTRACTOR_TYPE_COUNT) {
    return extraction_map[EXTRACTOR_UNKOWN_INDEX];
  }
  return extraction_map[index];
}

bool isMatch(const char *type_bytes, ExtractorTypeIndex idx) {
  Extractor extractor = extraction_map[idx];
  for (int j = 0; j < MAX_NAMES_PER_EXTRACTOR; j++) {
    if (extractor.name[j] != NULL &&
        strcmp(type_bytes, extractor.name[j]) == 0) {
      return true;
    }
  }
  return false;
}

ExtractorTypeIndex getExtractorIndex(PyObject *type) {
  Py_ssize_t type_len = 0;
  const char *type_bytes = PyUnicode_AsUTF8AndSize(type, &type_len);
  if (!type_bytes) {
    return EXTRACTOR_UNKOWN_INDEX;
  }
  if (type_len < 2) {
    PyErr_Format(PyExc_RuntimeError, "Unexpected data type: %R", type);
    return EXTRACTOR_UNKOWN_INDEX;
  }
  // Examples: '*fp32', 'fp32', 'i8', etc.
  if (type_bytes[0] == '*') {
    return EXTRACTOR_POINTER_INDEX;
  }
  for (ExtractorTypeIndex i = EXTRACTOR_INT8_INDEX; i < EXTRACTOR_TYPE_COUNT;
       i++) {
    if (isMatch(type_bytes, i)) {
      return i;
    }
  }

  PyErr_Format(PyExc_RuntimeError, "Unknown data type: %R", type);
  return EXTRACTOR_UNKOWN_INDEX;
}

// Takes in a list of types (ex: ['*fp32', 'u8', 'nvTmaDesc']) and returns
// a bytes array that represent extractors for quick argument extraction
// when launching.
static PyObject *buildSignatureMetadata(PyObject *self, PyObject *args) {
  PyObject *signature = NULL;
  if (!PyArg_ParseTuple(args, "O", &signature)) {
    return NULL;
  }
  PyObject *fast_signature = PySequence_Fast(
      signature, "Expected kernel_arg_types to be a sequence or iterable");
  if (!fast_signature) {
    return NULL;
  }
  Py_ssize_t signature_size = PySequence_Fast_GET_SIZE(fast_signature);
  PyObject **signature_items = PySequence_Fast_ITEMS(fast_signature);

  // Create return bytes object.
  PyObject *ret_bytes = PyBytes_FromStringAndSize(NULL, signature_size);
  if (ret_bytes == NULL) {
    Py_XDECREF(fast_signature);
    return NULL;
  }
  char *buffer = PyBytes_AS_STRING(ret_bytes);
  for (Py_ssize_t i = 0; i < signature_size; ++i) {
    ExtractorTypeIndex extractor_idx = getExtractorIndex(signature_items[i]);
    if (extractor_idx == EXTRACTOR_UNKOWN_INDEX) {
      goto cleanup;
    }
    buffer[i] = (uint8_t)extractor_idx;
  }

  Py_XDECREF(fast_signature);
  return ret_bytes;

cleanup:
  Py_XDECREF(fast_signature);
  Py_XDECREF(ret_bytes);
  return NULL;
}

bool extractArgs(PyObject **final_list, int *list_idx, PyObject *kernel_args,
                 PyObject *arg_annotations) {
  // Extract arg annotations
  PyObject *fast_annotations = PySequence_Fast(
      arg_annotations, "Expected arg_annotations to be a sequence or iterable");
  if (!fast_annotations) {
    goto cleanup;
  }
  Py_ssize_t num_annotations = PySequence_Fast_GET_SIZE(fast_annotations);
  PyObject **annotations = PySequence_Fast_ITEMS(fast_annotations);

  PyObject *fast_args = PySequence_Fast(
      kernel_args, "Expected kernel_args to be a sequence or iterable");
  if (!fast_args) {
    goto cleanup;
  }
  PyObject **args = PySequence_Fast_ITEMS(fast_args);

  int arg_idx = 0;
  for (int i = 0; i < num_annotations; ++i) {
    PyKernelArgObject *annotation = (PyKernelArgObject *)annotations[i];
    switch (annotation->type) {
    case ARG_KERNEL:
      final_list[(*list_idx)++] = args[arg_idx++];
      break;
    case ARG_TUPLE:
      if (!extractArgs(final_list, list_idx, args[arg_idx++],
                       annotation->nested_tuple)) {
        goto cleanup;
      }
      break;
    case ARG_CONSTEXPR:
      arg_idx++;
      break;
    }
  }
  Py_DECREF(fast_annotations);
  Py_DECREF(fast_args);
  return true;

cleanup:
  Py_XDECREF(fast_annotations);
  Py_XDECREF(fast_args);
  return false;
}

bool launchHook(PyObject *hook, PyObject *metadata) {
  if (hook != Py_None) {
    PyObject *ret = PyObject_CallOneArg(hook, metadata);
    if (!ret) {
      return false;
    }
    Py_DECREF(ret);
  }
  return true;
}

static PyObject *launchKernel(PyObject *self, PyObject *args) {
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before
  // calls to cuPointerGetAttributes
  ensureCudaContext();

  // Parse the arguments.
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  int num_warps, num_ctas, shared_memory, preferredClusterDimX,
      preferredClusterDimY, preferredClusterDimZ;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  PyObject *arg_annotations = NULL;
  Py_buffer signature;
  PyObject *kernel_args = NULL;
  if (!PyArg_ParseTuple(
          args, "iiiKKpp(iiiiii)OOOOOOy*O", &gridX, &gridY, &gridZ, &_stream,
          &_function, &launch_cooperative_grid, &launch_pdl, &num_warps,
          &num_ctas, &shared_memory, &preferredClusterDimX,
          &preferredClusterDimY, &preferredClusterDimZ, &launch_metadata,
          &launch_enter_hook, &launch_exit_hook, &global_scratch_obj,
          &profile_scratch_obj, &arg_annotations, &signature, &kernel_args)) {
    return NULL;
  }

  // launch entry hook.
  if (!launchHook(launch_enter_hook, launch_metadata)) {
    goto cleanup;
  }

  uint8_t *extractor_data = (uint8_t *)signature.buf;
  Py_ssize_t num_args = signature.len;

  // Extract kernel parameters - flatten tuples & remove constexpr.
  PyObject **args_data = (PyObject **)alloca(num_args * sizeof(PyObject *));
  if (args_data == NULL) {
    goto cleanup;
  }
  int list_idx = 0;
  if (!extractArgs(args_data, &list_idx, kernel_args, arg_annotations)) {
    goto cleanup;
  }

  // Number of parameters passed to kernel. + 2 for global & profile scratch.
  int num_params = num_args + 2;
  void **params = (void **)alloca(num_params * sizeof(void *));
  int params_idx = 0;
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  for (Py_ssize_t i = 0; i < num_args; ++i) {
    // Get extractor that will send back a struct with
    // * size for allocation
    // * function to call to put the parameter in params buffer
    Extractor extractor = getExtractor(extractor_data[i]);
    if (extractor.extract == NULL) {
      goto cleanup;
    }

    size_t alignment = extractor.alignment;
    if (alignment != 0) {
      // Allocate enough space on the stack to guarantee an aligned block.
      size_t size_with_alignment = extractor.size + alignment - 1;
      void *storage_ptr = alloca(size_with_alignment);
      void *aligned_ptr = (void *)((((uintptr_t)storage_ptr) + alignment - 1) &
                                   ~(alignment - 1));
      if (aligned_ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to align parameter storage");
        goto cleanup;
      }
      params[params_idx] = aligned_ptr;
    } else {
      params[params_idx] = alloca(extractor.size);
    }

    PyObject *current_arg = args_data[i];
    if (!extractor.extract(params[params_idx++], current_arg)) {
      goto cleanup;
    }
  }
  // Add scratch objects.
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], global_scratch_obj)) {
    goto cleanup;
  }
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], profile_scratch_obj)) {
    goto cleanup;
  }

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid,
          launch_pdl, preferredClusterDimX, preferredClusterDimY,
          preferredClusterDimZ, shared_memory, (CUstream)_stream,
          (CUfunction)_function, params);
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    goto cleanup;
  }

  if (!launchHook(launch_exit_hook, launch_metadata)) {
    goto cleanup;
  }
  PyBuffer_Release(&signature);
  Py_RETURN_NONE;

cleanup:
  PyBuffer_Release(&signature);
  return NULL;
}

/* =========================================================================
 * _TritonDispatcher: Meta-specific vectorcall-based kernel dispatcher.
 *
 * Replaces the Python dispatch path with a single C vectorcall.
 * Pre-binds CUfunction, launch attributes, and arg type layout.
 * Calling convention: dispatcher(grid_x, grid_y, grid_z, stream, *kernel_args)
 * ========================================================================= */

#define TD_MAX_KERNEL_ARGS 64
#define TD_FIXED_ARGS 4  /* grid_x, grid_y, grid_z, stream */

typedef union {
    CUdeviceptr ptr;
    int8_t   i8;
    int16_t  i16;
    int32_t  i32;
    int64_t  i64;
    uint8_t  u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float    f32;
    double   f64;
} TDArgSlot;

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    CUfunction function;
    unsigned grid_mult;     /* num_ctas — grid_x multiplied by this */
    unsigned block_dim_x;   /* 32 * num_warps */
    unsigned shared_mem;
    CUlaunchAttribute launch_attrs[5];
    unsigned num_launch_attrs;
    int arg_types[TD_MAX_KERNEL_ARGS];  /* ExtractorTypeIndex values */
    int num_args;
    int total_params;
    TDArgSlot arg_storage[TD_MAX_KERNEL_ARGS];
    void *kernel_params[TD_MAX_KERNEL_ARGS];
    int has_global_scratch;
    int has_profile_scratch;
    /* Scratch allocation support */
    unsigned global_scratch_size;
    unsigned global_scratch_align;
    unsigned profile_scratch_size;
    unsigned profile_scratch_align;
    PyObject *allocator;          /* _allocation._allocator (ContextVar) */
    PyObject *profile_allocator;  /* _allocation._profile_allocator (wrapper) */
} TritonDispatcher;

/* Forward declarations */
static PyObject *TritonDispatcher_vectorcall(
    PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static void TritonDispatcher_dealloc(PyObject *self);

/* Fast pointer extraction (no cuPointerGetAttribute validation — hot path) */
static inline CUdeviceptr td_get_ptr(PyObject *obj) {
    if (PyLong_Check(obj))
        return (CUdeviceptr)PyLong_AsUnsignedLongLong(obj);
    if (obj == Py_None)
        return 0;
    PyObject *r = PyObject_CallMethodNoArgs(obj, data_ptr_str);
    if (!r) return 0;
    CUdeviceptr p = (CUdeviceptr)PyLong_AsUnsignedLongLong(r);
    Py_DECREF(r);
    return p;
}

/* Fast fp16/bf16 packing (equivalent to extractFP16/BF16 but returns value) */
static inline uint16_t td_pack_fp16(double v) {
    uint16_t result;
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && \
    !defined(PYPY_VERSION)
    _PyFloat_Pack2(v, (unsigned char *)&result, 1);
#else
    PyFloat_Pack2(v, (char *)&result, 1);
#endif
    return result;
}
static inline uint16_t td_pack_bf16(double v) {
    float f = (float)v;
    uint32_t b; memcpy(&b, &f, 4);
    return (uint16_t)(b >> 16);
}

/* Arg conversion using ExtractorTypeIndex codes */
static inline int td_convert_args(TritonDispatcher *self, PyObject *const *kargs) {
    for (int i = 0; i < self->num_args; i++) {
        PyObject *a = kargs[i];
        TDArgSlot *s = &self->arg_storage[i];
        switch (self->arg_types[i]) {
        case EXTRACTOR_POINTER_INDEX:
            s->ptr = td_get_ptr(a);
            break;
        case EXTRACTOR_INT8_INDEX:  s->i8  = (int8_t)PyLong_AsLong(a); break;
        case EXTRACTOR_INT16_INDEX: s->i16 = (int16_t)PyLong_AsLong(a); break;
        case EXTRACTOR_INT32_INDEX: s->i32 = (int32_t)PyLong_AsLong(a); break;
        case EXTRACTOR_INT64_INDEX: s->i64 = (int64_t)PyLong_AsLongLong(a); break;
        case EXTRACTOR_UINT8_INDEX:  s->u8  = (uint8_t)PyLong_AsUnsignedLong(a); break;
        case EXTRACTOR_UINT16_INDEX: s->u16 = (uint16_t)PyLong_AsUnsignedLong(a); break;
        case EXTRACTOR_UINT32_INDEX: s->u32 = (uint32_t)PyLong_AsUnsignedLong(a); break;
        case EXTRACTOR_UINT64_INDEX: s->u64 = (uint64_t)PyLong_AsUnsignedLongLong(a); break;
        case EXTRACTOR_FP16_INDEX: s->u16 = td_pack_fp16(PyFloat_AsDouble(a)); break;
        case EXTRACTOR_BF16_INDEX: s->u16 = td_pack_bf16(PyFloat_AsDouble(a)); break;
        case EXTRACTOR_FP32_INDEX: { float f = (float)PyFloat_AsDouble(a); memcpy(&s->u32, &f, 4); break; }
        case EXTRACTOR_FP64_INDEX: s->f64 = PyFloat_AsDouble(a); break;
        default:
            PyErr_Format(PyExc_TypeError, "Unknown type code %d for arg %d", self->arg_types[i], i);
            return -1;
        }
    }
    if (PyErr_Occurred()) return -1;
    return 0;
}

/* Relaunch with pre-built attrs (cuLaunchKernelEx wrapper) */
static inline CUresult td_relaunch(
    TritonDispatcher *d, unsigned gx, unsigned gy, unsigned gz, CUstream stream)
{
    if (gx * gy * gz == 0) return CUDA_SUCCESS;
    CUlaunchConfig cfg;
    cfg.gridDimX = gx * d->grid_mult; cfg.gridDimY = gy; cfg.gridDimZ = gz;
    cfg.blockDimX = d->block_dim_x; cfg.blockDimY = 1; cfg.blockDimZ = 1;
    cfg.sharedMemBytes = d->shared_mem; cfg.hStream = stream;
    cfg.attrs = d->num_launch_attrs > 0 ? d->launch_attrs : NULL;
    cfg.numAttrs = d->num_launch_attrs;
    return ensureLaunchHandle()(&cfg, d->function, d->kernel_params, NULL);
}

/* ---- Constructor ---- */
static PyObject *TritonDispatcher_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    unsigned long long func_ptr;
    int num_warps, num_ctas, shared_mem;
    int launch_pdl, launch_coop, launch_cluster;
    PyObject *arg_type_codes;
    int has_global_scratch, has_profile_scratch;
    unsigned global_scratch_size = 0, global_scratch_align = 1;
    unsigned profile_scratch_size = 0, profile_scratch_align = 1;
    PyObject *allocator_obj = NULL, *profile_allocator_obj = NULL;

    static char *kwlist[] = {
        "function", "num_warps", "num_ctas", "shared_mem",
        "launch_pdl", "launch_cooperative_grid", "launch_cluster",
        "arg_type_codes", "has_global_scratch", "has_profile_scratch",
        "global_scratch_size", "global_scratch_align",
        "profile_scratch_size", "profile_scratch_align",
        "allocator", "profile_allocator", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "KiiiiiiOpp|IIIIOO", kwlist,
            &func_ptr, &num_warps, &num_ctas, &shared_mem,
            &launch_pdl, &launch_coop, &launch_cluster,
            &arg_type_codes, &has_global_scratch, &has_profile_scratch,
            &global_scratch_size, &global_scratch_align,
            &profile_scratch_size, &profile_scratch_align,
            &allocator_obj, &profile_allocator_obj))
        return NULL;

    TritonDispatcher *self = (TritonDispatcher *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->vectorcall = TritonDispatcher_vectorcall;
    self->function = (CUfunction)(uintptr_t)func_ptr;
    self->grid_mult = (unsigned)num_ctas;
    self->block_dim_x = 32 * (unsigned)num_warps;
    self->shared_mem = (unsigned)shared_mem;
    self->has_global_scratch = has_global_scratch;
    self->has_profile_scratch = has_profile_scratch;
    self->global_scratch_size = global_scratch_size;
    self->global_scratch_align = global_scratch_align;
    self->profile_scratch_size = profile_scratch_size;
    self->profile_scratch_align = profile_scratch_align;
    self->allocator = allocator_obj;
    Py_XINCREF(self->allocator);
    self->profile_allocator = profile_allocator_obj;
    Py_XINCREF(self->profile_allocator);

    memset(self->launch_attrs, 0, sizeof(self->launch_attrs));
    memset(self->arg_storage, 0, sizeof(self->arg_storage));
    memset(self->kernel_params, 0, sizeof(self->kernel_params));

    /* Parse arg types (ExtractorTypeIndex values from buildSignatureMetadata) */
    Py_ssize_t n = PyTuple_Size(arg_type_codes);
    if (n > TD_MAX_KERNEL_ARGS - 2) {
        PyErr_SetString(PyExc_ValueError, "Too many kernel args");
        Py_DECREF(self);
        return NULL;
    }
    self->num_args = (int)n;
    for (Py_ssize_t i = 0; i < n; i++)
        self->arg_types[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(arg_type_codes, i));
    if (PyErr_Occurred()) {
        Py_DECREF(self);
        return NULL;
    }

    /* Build kernel_params pointers */
    int pidx = 0;
    for (int i = 0; i < self->num_args; i++) {
        self->kernel_params[pidx] = &self->arg_storage[pidx];
        pidx++;
    }
    self->kernel_params[pidx] = &self->arg_storage[pidx]; pidx++;  /* global_scratch */
    self->kernel_params[pidx] = &self->arg_storage[pidx]; pidx++;  /* profile_scratch */
    self->total_params = pidx;

    /* Pre-build launch attributes */
    unsigned na = 0;
    if (launch_pdl) {
        self->launch_attrs[na].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        self->launch_attrs[na].value.programmaticStreamSerializationAllowed = 1;
        na++;
    }
    if (launch_coop) {
        self->launch_attrs[na].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        self->launch_attrs[na].value.cooperative = 1;
        na++;
    }
    if (launch_cluster || num_ctas > 1) {
        if (num_ctas > 1) {
            self->launch_attrs[na].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            self->launch_attrs[na].value.clusterDim.x = num_ctas;
            self->launch_attrs[na].value.clusterDim.y = 1;
            self->launch_attrs[na].value.clusterDim.z = 1;
            na++;
        }
        self->launch_attrs[na].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        self->launch_attrs[na].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
        na++;
    }
    self->num_launch_attrs = na;

    return (PyObject *)self;
}

static void TritonDispatcher_dealloc(PyObject *o) {
    TritonDispatcher *self = (TritonDispatcher *)o;
    Py_XDECREF(self->allocator);
    Py_XDECREF(self->profile_allocator);
    Py_TYPE(o)->tp_free(o);
}

/* ==== THE HOT PATH ==== */
static PyObject *TritonDispatcher_vectorcall(
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    TritonDispatcher *self = (TritonDispatcher *)callable;
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs < TD_FIXED_ARGS + self->num_args) {
        PyErr_Format(PyExc_TypeError,
            "_TritonDispatcher: expected %d args, got %zd",
            TD_FIXED_ARGS + self->num_args, nargs);
        return NULL;
    }

    long gx_l = PyLong_AsLong(args[0]);
    long gy_l = PyLong_AsLong(args[1]);
    long gz_l = PyLong_AsLong(args[2]);
    if (PyErr_Occurred()) return NULL;

    unsigned gx = (unsigned)gx_l;
    unsigned gy = (unsigned)gy_l;
    unsigned gz = (unsigned)gz_l;

    if (gx * gy * gz == 0)
        Py_RETURN_NONE;

    CUstream stream = (CUstream)(uintptr_t)PyLong_AsUnsignedLongLong(args[3]);

    /* Convert kernel args.
     * No cleanup needed on failure: scratch/profile buffers are allocated below,
     * and no other resources have been acquired at this point. */
    if (td_convert_args(self, args + TD_FIXED_ARGS) < 0)
        return NULL;

    /* Scratch allocation: call Python allocator if scratch is needed.
     * alloc_size = grid_x * grid_y * grid_z * num_ctas * scratch_size */
    PyObject *scratch_buf = NULL, *profile_buf = NULL;
    if (self->global_scratch_size > 0 && self->allocator) {
        unsigned long long alloc_size =
            (unsigned long long)gx * gy * gz * self->grid_mult * self->global_scratch_size;
        PyObject *alloc_fn = PyObject_CallMethodNoArgs(self->allocator, td_get_str);
        if (!alloc_fn) return NULL;
        scratch_buf = PyObject_CallFunction(alloc_fn, "KIK",
            alloc_size, (unsigned)self->global_scratch_align, (unsigned long long)(uintptr_t)stream);
        Py_DECREF(alloc_fn);
        if (!scratch_buf) return NULL;
        PyObject *ptr_obj = PyObject_CallMethodNoArgs(scratch_buf, data_ptr_str);
        if (!ptr_obj) { Py_DECREF(scratch_buf); return NULL; }
        self->arg_storage[self->num_args].ptr = (CUdeviceptr)PyLong_AsUnsignedLongLong(ptr_obj);
        Py_DECREF(ptr_obj);
    } else {
        self->arg_storage[self->num_args].ptr = 0;
    }

    if (self->profile_scratch_size > 0 && self->profile_allocator) {
        unsigned long long alloc_size =
            (unsigned long long)gx * gy * gz * self->grid_mult * self->profile_scratch_size;
        PyObject *alloc_fn = PyObject_CallMethodNoArgs(self->profile_allocator, td_get_str);
        if (!alloc_fn) { Py_XDECREF(scratch_buf); return NULL; }
        profile_buf = PyObject_CallFunction(alloc_fn, "KIK",
            alloc_size, (unsigned)self->profile_scratch_align, (unsigned long long)(uintptr_t)stream);
        Py_DECREF(alloc_fn);
        if (!profile_buf) { Py_XDECREF(scratch_buf); return NULL; }
        PyObject *ptr_obj = PyObject_CallMethodNoArgs(profile_buf, data_ptr_str);
        if (!ptr_obj) { Py_DECREF(profile_buf); Py_XDECREF(scratch_buf); return NULL; }
        self->arg_storage[self->num_args + 1].ptr = (CUdeviceptr)PyLong_AsUnsignedLongLong(ptr_obj);
        Py_DECREF(ptr_obj);
    } else {
        self->arg_storage[self->num_args + 1].ptr = 0;
    }

    /* Launch using pre-built attrs.
     * Thread safety: arg_storage is per-instance and the GIL is held up to
     * Py_BEGIN_ALLOW_THREADS. cuLaunchKernelEx copies kernel_params at call
     * entry (documented CUDA driver behavior), so releasing the GIL after
     * the call begins is safe — another thread cannot corrupt params mid-copy. */
    CUresult err;
    Py_BEGIN_ALLOW_THREADS
    err = td_relaunch(self, gx, gy, gz, stream);
    Py_END_ALLOW_THREADS

    /* Release scratch buffers after launch (kernel params already copied) */
    Py_XDECREF(scratch_buf);
    Py_XDECREF(profile_buf);

    if (err != CUDA_SUCCESS) {
        const char *s = NULL;
        cuGetErrorString(err, &s);
        PyErr_Format(PyExc_RuntimeError,
            "Triton Error [CUDA]: cuLaunchKernelEx failed: %s (%d)",
            s ? s : "unknown", (int)err);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyTypeObject TritonDispatcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.nvidia._TritonDispatcher",
    .tp_basicsize = sizeof(TritonDispatcher),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(TritonDispatcher, vectorcall),
    .tp_call = PyVectorcall_Call,
    .tp_new = TritonDispatcher_new,
    .tp_dealloc = TritonDispatcher_dealloc,
    .tp_doc = "Full C dispatcher for Triton JIT kernel launch (vectorcall).",
};

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    TritonDispatcher *dispatcher;
    unsigned grid[3];
    PyObject *get_stream_fn;
    PyObject *get_device_fn;
    int num_args;
    PyObject *slow_path_fn;
} TritonJITRunner;

static PyObject *JITRunner_vectorcall(
    PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static void JITRunner_dealloc(PyObject *self);

static PyObject *JITRunner_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject *dispatcher_obj, *grid_tuple, *get_stream_fn, *get_device_fn, *slow_path_fn;
    int num_kernel_args;

    static char *kwlist[] = {
        "dispatcher", "grid", "get_stream_fn", "get_device_fn", "slow_path_fn", "num_kernel_args", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOi", kwlist,
            &dispatcher_obj, &grid_tuple, &get_stream_fn, &get_device_fn, &slow_path_fn, &num_kernel_args))
        return NULL;

    TritonJITRunner *self = (TritonJITRunner *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->vectorcall = JITRunner_vectorcall;
    self->dispatcher = (TritonDispatcher *)dispatcher_obj;
    Py_INCREF(dispatcher_obj);

    Py_ssize_t gs = PyTuple_Size(grid_tuple);
    self->grid[0] = (gs > 0) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 0)) : 1;
    self->grid[1] = (gs > 1) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 1)) : 1;
    self->grid[2] = (gs > 2) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 2)) : 1;

    self->get_stream_fn = get_stream_fn; Py_INCREF(get_stream_fn);
    self->get_device_fn = get_device_fn; Py_INCREF(get_device_fn);
    self->slow_path_fn = slow_path_fn; Py_INCREF(slow_path_fn);
    self->num_args = num_kernel_args;

    return (PyObject *)self;
}

static void JITRunner_dealloc(PyObject *o) {
    TritonJITRunner *self = (TritonJITRunner *)o;
    Py_XDECREF((PyObject *)self->dispatcher);
    Py_XDECREF(self->get_stream_fn);
    Py_XDECREF(self->get_device_fn);
    Py_XDECREF(self->slow_path_fn);
    Py_TYPE(o)->tp_free(o);
}

static PyObject *JITRunner_vectorcall(
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    TritonJITRunner *self = (TritonJITRunner *)callable;
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (kwnames != NULL && PyTuple_GET_SIZE(kwnames) > 0) {
        return PyObject_Vectorcall(self->slow_path_fn, args, nargsf, kwnames);
    }
    if (nargs < self->num_args) {
        PyErr_Format(PyExc_TypeError,
            "_TritonJITRunner: expected %d args, got %zd", self->num_args, nargs);
        return NULL;
    }

    /* The caller (activate_fast_dispatch) ensures the kernel is already
     * compiled for these arg types. We just extract + launch. */
    PyObject *device = PyObject_CallNoArgs(self->get_device_fn);
    if (!device) return NULL;
    PyObject *stream_obj = PyObject_CallOneArg(self->get_stream_fn, device);
    Py_DECREF(device);
    if (!stream_obj) return NULL;
    uint64_t stream = PyLong_AsUnsignedLongLong(stream_obj);
    Py_DECREF(stream_obj);

    /* Build dispatcher args: [grid_x, grid_y, grid_z, stream, *kernel_args] */
    Py_ssize_t disp_nargs = TD_FIXED_ARGS + self->num_args;
    PyObject **disp_args = (PyObject **)alloca(disp_nargs * sizeof(PyObject *));
    PyObject *gx = PyLong_FromUnsignedLong(self->grid[0]);
    PyObject *gy = PyLong_FromUnsignedLong(self->grid[1]);
    PyObject *gz = PyLong_FromUnsignedLong(self->grid[2]);
    PyObject *st = PyLong_FromUnsignedLongLong(stream);
    if (!gx || !gy || !gz || !st) {
        Py_XDECREF(gx); Py_XDECREF(gy); Py_XDECREF(gz); Py_XDECREF(st);
        return NULL;
    }
    disp_args[0] = gx; disp_args[1] = gy; disp_args[2] = gz; disp_args[3] = st;
    for (int i = 0; i < self->num_args; i++)
        disp_args[TD_FIXED_ARGS + i] = (PyObject *)args[i];

    PyObject *result = TritonDispatcher_vectorcall(
        (PyObject *)self->dispatcher, (PyObject *const *)disp_args, disp_nargs, NULL);
    Py_DECREF(gx); Py_DECREF(gy); Py_DECREF(gz); Py_DECREF(st);
    return result;
}

static PyTypeObject TritonJITRunnerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.nvidia._TritonJITRunner",
    .tp_basicsize = sizeof(TritonJITRunner),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(TritonJITRunner, vectorcall),
    .tp_call = PyVectorcall_Call,
    .tp_new = JITRunner_new,
    .tp_dealloc = JITRunner_dealloc,
};

/* =========================================================================
 * _ProxyRunner: returned by fast_subscript (__getitem__ in C).
 * Pre-binds grid + stream getter + dispatcher. Always extracts args fresh
 * ========================================================================= */
typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    TritonDispatcher *dispatcher;
    unsigned grid[3];
    PyObject *get_stream_fn;
    PyObject *get_device_fn;
    PyObject *kernel;
    int num_args;
} ProxyRunner;

static PyObject *ProxyRunner_vectorcall(
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames);
static void ProxyRunner_dealloc(PyObject *o);

static PyObject *ProxyRunner_vectorcall(
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    ProxyRunner *self = (ProxyRunner *)callable;
    TritonDispatcher *d = self->dispatcher;
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs < self->num_args) {
        PyErr_Format(PyExc_TypeError,
            "_ProxyRunner: expected >= %d args, got %zd", self->num_args, nargs);
        return NULL;
    }

    /* Always extract args fresh */
    if (td_convert_args(d, args) < 0) return NULL;

    /* Get current stream */
    PyObject *dev = PyObject_CallNoArgs(self->get_device_fn);
    if (!dev) return NULL;
    PyObject *st = PyObject_CallOneArg(self->get_stream_fn, dev);
    Py_DECREF(dev);
    if (!st) return NULL;
    CUstream stream = (CUstream)(uintptr_t)PyLong_AsUnsignedLongLong(st);
    Py_DECREF(st);

    /* Launch.
     * Thread safety: arg_storage is per-dispatcher-instance and td_convert_args
     * runs while holding the GIL. cuLaunchKernelEx copies kernel_params at call
     * entry (documented CUDA driver behavior), so releasing the GIL after the
     * call begins is safe. Sharing the same dispatcher across Python threads is
     * NOT supported — each ProxyRunner holds a dedicated dispatcher reference. */
    CUresult err;
    Py_BEGIN_ALLOW_THREADS
    err = td_relaunch(d, self->grid[0], self->grid[1], self->grid[2], stream);
    Py_END_ALLOW_THREADS
    if (err != CUDA_SUCCESS) {
        const char *s = NULL; cuGetErrorString(err, &s);
        PyErr_Format(PyExc_RuntimeError, "cuLaunchKernelEx: %s (%d)", s?s:"?", (int)err);
        return NULL;
    }
    Py_INCREF(self->kernel);
    return self->kernel;
}

static void ProxyRunner_dealloc(PyObject *o) {
    ProxyRunner *self = (ProxyRunner *)o;
    Py_XDECREF((PyObject *)self->dispatcher);
    Py_XDECREF(self->get_stream_fn);
    Py_XDECREF(self->get_device_fn);
    Py_XDECREF(self->kernel);
    Py_TYPE(o)->tp_free(o);
}

static PyTypeObject ProxyRunnerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.nvidia._ProxyRunner",
    .tp_basicsize = sizeof(ProxyRunner),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(ProxyRunner, vectorcall),
    .tp_call = PyVectorcall_Call,
    .tp_dealloc = ProxyRunner_dealloc,
};

/* =========================================================================
 * fast_subscript: C __getitem__ slot for the dynamic heap type.
 * Creates _ProxyRunner per grid, caching in _runner_cache dict.
 * ========================================================================= */
static PyObject *fast_subscript(PyObject *self, PyObject *grid) {
    static PyObject *_rc_key = NULL;
    if (!_rc_key) _rc_key = PyUnicode_InternFromString("_runner_cache");

    /* Normalize grid to tuple for consistent cache key */
    PyObject *grid_tuple;
    if (!PyTuple_Check(grid)) {
        grid_tuple = PyTuple_Pack(1, grid);
        if (!grid_tuple) return NULL;
    } else {
        grid_tuple = grid;
        Py_INCREF(grid_tuple);
    }

    PyObject **dictptr = _PyObject_GetDictPtr(self);
    if (!dictptr || !*dictptr) { Py_DECREF(grid_tuple); goto fallback; }
    PyObject *dict = *dictptr;

    PyObject *cache = PyDict_GetItem(dict, _rc_key);
    if (!cache) { Py_DECREF(grid_tuple); goto fallback; }

    PyObject *runner = PyDict_GetItem(cache, grid_tuple);
    if (runner) { Py_DECREF(grid_tuple); Py_INCREF(runner); return runner; }

    /* Create new ProxyRunner */
    {
        static PyObject *_disp_key = NULL, *_nka_key = NULL, *_gsf_key = NULL, *_gdf_key = NULL, *_kern_key = NULL;
        if (!_disp_key) {
            _disp_key = PyUnicode_InternFromString("_fast_dispatcher");
            _nka_key = PyUnicode_InternFromString("_fast_num_args");
            _gsf_key = PyUnicode_InternFromString("_fast_get_stream");
            _gdf_key = PyUnicode_InternFromString("_fast_get_device");
            _kern_key = PyUnicode_InternFromString("_fast_kernel");
        }

        PyObject *disp = PyDict_GetItem(dict, _disp_key);
        if (!disp) { Py_DECREF(grid_tuple); goto fallback; }
        PyObject *kern = PyDict_GetItem(dict, _kern_key);
        if (!kern) kern = Py_None;

        PyObject *nka_obj = PyDict_GetItem(dict, _nka_key);
        int nka = nka_obj ? (int)PyLong_AsLong(nka_obj) : 0;
        PyObject *gsf = PyDict_GetItem(dict, _gsf_key);
        PyObject *gdf = PyDict_GetItem(dict, _gdf_key);
        if (!gsf || !gdf) { Py_DECREF(grid_tuple); goto fallback; }

        ProxyRunner *pr = PyObject_New(ProxyRunner, &ProxyRunnerType);
        if (!pr) { Py_DECREF(grid_tuple); return NULL; }
        pr->vectorcall = ProxyRunner_vectorcall;
        /* Multiple ProxyRunner instances may reference the same TritonDispatcher.
         * This is safe because: (1) td_convert_args + cuLaunchKernelEx both run
         * while holding the GIL, and cuLaunchKernelEx copies kernel_params at
         * call entry before we release the GIL via Py_BEGIN_ALLOW_THREADS.
         * (2) Each ProxyRunner_vectorcall completes the full sequence
         * (convert → launch → GIL release) atomically from Python's perspective.
         * Direct multi-threaded use of kernel._dispatcher without the GIL is
         * unsupported (same restriction as all CPython C extension objects). */
        pr->dispatcher = (TritonDispatcher *)disp; Py_INCREF(disp);
        Py_ssize_t gs = PyTuple_Size(grid_tuple);
        pr->grid[0] = (gs > 0) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 0)) : 1;
        pr->grid[1] = (gs > 1) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 1)) : 1;
        pr->grid[2] = (gs > 2) ? (unsigned)PyLong_AsLong(PyTuple_GET_ITEM(grid_tuple, 2)) : 1;

        pr->get_stream_fn = gsf; Py_INCREF(gsf);
        pr->get_device_fn = gdf; Py_INCREF(gdf);
        pr->kernel = kern; Py_INCREF(kern);
        pr->num_args = nka;

        if (PyDict_SetItem(cache, grid_tuple, (PyObject *)pr) < 0) {
            Py_DECREF(grid_tuple);
            Py_DECREF(pr);
            return NULL;
        }
        Py_DECREF(grid_tuple);
        return (PyObject *)pr;
    }

fallback:;
    /* Build functools.partial(self.run, grid=grid, warmup=False) */
    static PyObject *_run_str = NULL, *_grid_str = NULL, *_warmup_str = NULL;
    if (!_run_str) _run_str = PyUnicode_InternFromString("run");
    if (!_grid_str) _grid_str = PyUnicode_InternFromString("grid");
    if (!_warmup_str) _warmup_str = PyUnicode_InternFromString("warmup");
    PyObject *run = PyObject_GetAttr(self, _run_str);
    if (!run) return NULL;
    PyObject *kw = PyDict_New();
    if (!kw) { Py_DECREF(run); return NULL; }
    if (PyDict_SetItem(kw, _grid_str, grid) < 0 ||
        PyDict_SetItem(kw, _warmup_str, Py_False) < 0) {
        Py_DECREF(run); Py_DECREF(kw); return NULL;
    }
    PyObject *partial_mod = PyImport_ImportModule("functools");
    if (!partial_mod) { Py_DECREF(run); Py_DECREF(kw); return NULL; }
    PyObject *partial_fn = PyObject_GetAttrString(partial_mod, "partial");
    Py_DECREF(partial_mod);
    PyObject *pack = PyTuple_Pack(1, run);
    if (!pack) { Py_DECREF(partial_fn); Py_DECREF(run); Py_DECREF(kw); return NULL; }
    PyObject *result = PyObject_Call(partial_fn, pack, kw);
    Py_DECREF(pack); Py_DECREF(partial_fn); Py_DECREF(run); Py_DECREF(kw);
    return result;
}

/* create_fast_jit_type(base_type) — create heap type with mp_subscript = fast_subscript */
static PyObject *py_create_fast_jit_type(PyObject *module, PyObject *base_type) {
    if (!PyType_Check(base_type)) {
        PyErr_SetString(PyExc_TypeError, "Expected a type");
        return NULL;
    }
    static PyType_Slot slots[] = {
        {Py_mp_subscript, fast_subscript},
        {0, NULL},
    };
    PyType_Spec spec = {
        .name = "triton.backends.nvidia._FastJITFunction",
        .basicsize = 0,
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE,
        .slots = slots,
    };
    PyObject *bases = PyTuple_Pack(1, base_type);
    if (!bases) return NULL;
    PyObject *new_type = PyType_FromSpecWithBases(&spec, bases);
    Py_DECREF(bases);
    return new_type;
}

/* ========================================================================= */

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cubin into CUDA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"cuOccupancyMaxActiveClusters", occupancyMaxActiveClusters, METH_VARARGS,
     "Python interface for cuOccupancyMaxActiveClusters function"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Python interface for cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, x), which "
     "controls how many bytes can be streamed from kernels before data starts "
     "being dropped.  This inherits all the limitations of this call; in "
     "particular it's an error to change this value after launching any kernel "
     "that calls printf()."},
    {"fill_tma_descriptor_tiled", fillTMADescriptorTiled, METH_VARARGS,
     "Create TMA descriptor for tiled mode"},
    {"fill_tma_descriptor_im2col", fillTMADescriptorIm2col, METH_VARARGS,
     "Create TMA descriptor for im2col mode"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     "Calling it with a signature list (ex: ['*fp32', 'u8', 'nvTmaDesc']), "
     "will return metadata to be passed into 'launch' for quicker "
     "argument parsing."},
    {"launch", launchKernel, METH_VARARGS, "launches cuda kernel"},
    {"fill_1d_tma_descriptor", fill1DTMADescriptor, METH_VARARGS, "doc"},
    {"fill_2d_tma_descriptor", fill2DTMADescriptor, METH_VARARGS, "doc"},
    {"fill_1d_tma_descriptor_type", fill1DTMADescriptorType, METH_VARARGS,
     "doc"},
    {"fill_2d_tma_descriptor_type", fill2DTMADescriptorType, METH_VARARGS,
     "doc"},
    {"create_fast_jit_type", py_create_fast_jit_type, METH_O,
     "Create a heap type inheriting from JITFunction with C mp_subscript"},

    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cuda_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cuda_utils(void) {
  if (PyType_Ready(&PyCUtensorMapType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&PyKernelArgType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&TritonDispatcherType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&TritonJITRunnerType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&ProxyRunnerType) < 0) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if (data_ptr_str == NULL) {
    return NULL;
  }
  td_get_str = PyUnicode_InternFromString("get");
  if (td_get_str == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  Py_INCREF(&PyCUtensorMapType);
  PyModule_AddObject(m, "PyCUtensorMap", (PyObject *)&PyCUtensorMapType);

  Py_INCREF(&PyKernelArgType);
  PyModule_AddObject(m, "PyKernelArg", (PyObject *)&PyKernelArgType);
  PyModule_AddIntConstant(m, "ARG_CONSTEXPR", ARG_CONSTEXPR);
  PyModule_AddIntConstant(m, "ARG_KERNEL", ARG_KERNEL);
  PyModule_AddIntConstant(m, "ARG_TUPLE", ARG_TUPLE);

  Py_INCREF(&TritonDispatcherType);
  PyModule_AddObject(m, "_TritonDispatcher", (PyObject *)&TritonDispatcherType);
  Py_INCREF(&TritonJITRunnerType);
  PyModule_AddObject(m, "_TritonJITRunner", (PyObject *)&TritonJITRunnerType);
  Py_INCREF(&ProxyRunnerType);
  PyModule_AddObject(m, "_ProxyRunner", (PyObject *)&ProxyRunnerType);

  return m;
}
