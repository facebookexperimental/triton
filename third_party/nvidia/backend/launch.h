/*
 * nvidia/backend/launch.h — Minimal runtime header for Triton standalone
 * launchers.
 *
 * This header provides everything a compiler-generated launcher needs to call
 * cuLaunchKernelEx.  It has NO dependency on Python.h — the generated launcher
 * is a plain C function callable from C, C++, or via ctypes/cffi.
 *
 * Consumers: compiler-generated launcher sources (asm["launcher_src"]),
 *            TritonCC, AOT-T, JIT (variadic launcher), custom integrations.
 *
 * ALL consumers call the same function: triton_launch_kernel().
 * TMA construction, params[] layout, and launch attribute setup all happen
 * inside this one function.  Changes to TMA promotion only need to update here.
 */

#ifndef TRITON_RUNTIME_LAUNCH_H
#define TRITON_RUNTIME_LAUNCH_H

#include <cuda.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Error handling
 * ------------------------------------------------------------------------- */

/**
 * Check a CUresult and return it if non-zero.
 * Use inside functions that return CUresult.
 */
#define TRITON_CUDA_CHECK(expr)                                                \
  do {                                                                         \
    CUresult _triton_err = (expr);                                             \
    if (_triton_err != CUDA_SUCCESS) {                                         \
      return _triton_err;                                                      \
    }                                                                          \
  } while (0)

/**
 * Check a CUresult, print an error message and return it if non-zero.
 * Use for debugging / verbose error reporting.
 */
#define TRITON_CUDA_CHECK_LOG(expr)                                            \
  do {                                                                         \
    CUresult _triton_err = (expr);                                             \
    if (_triton_err != CUDA_SUCCESS) {                                         \
      const char *_triton_err_str = NULL;                                      \
      cuGetErrorString(_triton_err, &_triton_err_str);                         \
      fprintf(stderr, "Triton Error [CUDA] at %s:%d: %s\n", __FILE__,          \
              __LINE__, _triton_err_str ? _triton_err_str : "unknown");        \
      return _triton_err;                                                      \
    }                                                                          \
  } while (0)

/* -------------------------------------------------------------------------
 * Lazy-loaded cuLaunchKernelEx
 * ------------------------------------------------------------------------- */

typedef CUresult (*triton_cuLaunchKernelEx_fn)(const CUlaunchConfig *config,
                                               CUfunction f,
                                               void **kernelParams,
                                               void **extra);

static triton_cuLaunchKernelEx_fn g_triton_launch_fn = NULL;

/**
 * Initialize cuLaunchKernelEx at program startup.
 * Runs automatically before main() via __attribute__((constructor)).
 * Thread-safe by virtue of running before any threads are created.
 *
 * Note: dlopen handle is intentionally not closed — libcuda.so.1 must remain
 * loaded for the process lifetime since cuLaunchKernelEx is called on every
 * kernel launch.
 */
__attribute__((constructor)) static void triton_init_launch_kernel_ex(void) {
  void *lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!lib)
    return; /* g_triton_launch_fn remains NULL */

  g_triton_launch_fn =
      (triton_cuLaunchKernelEx_fn)dlsym(lib, "cuLaunchKernelEx");
}

static inline triton_cuLaunchKernelEx_fn triton_get_launch_kernel_ex(void) {
  return g_triton_launch_fn;
}

/* -------------------------------------------------------------------------
 * Launch descriptor (data-driven launch)
 * ------------------------------------------------------------------------- */

/**
 * Maximum number of launch attributes a Triton launcher may set.
 * Currently: PDL, cooperative, cluster dim, cluster scheduling, preferred
 * cluster dim.
 */
#define TRITON_MAX_LAUNCH_ATTRS 5
#define TRITON_MAX_TMA_DESCS 8
#define TRITON_MAX_TMA_DIMS 5
#define TRITON_MAX_PARAMS 256

/**
 * ABI version of triton_kernel_launch_desc_t.  Bump this whenever the struct
 * layout below changes in a backward-incompatible way.  Producers stamp this
 * into desc->abi_version and triton_launch_kernel() rejects mismatches at
 * runtime, so a stale ctypes mirror (launch_desc.py) or generated launcher is
 * caught instead of silently corrupting kernel args.
 */
#define TRITON_LAUNCH_DESC_ABI_VERSION 2

/* static_assert spelling differs between C11 and C++; provide one name. */
#if defined(__cplusplus)
#define TRITON_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define TRITON_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/**
 * TMA recipe: describes how to construct one CUtensorMap from user args.
 *
 * All offsets refer to byte positions within the args_buf passed to
 * triton_launch_kernel(). The launcher reads ptr/shape/stride values
 * from args_buf at these offsets and calls cuTensorMapEncodeTiled().
 *
 * Compile-time constants (block_shape, swizzle, elem_type, etc.) are baked
 * directly into the struct by the compiler.
 */
typedef struct {
  int ndim;
  uint32_t block_shape[TRITON_MAX_TMA_DIMS];
  int swizzle;
  int elem_type;
  int elem_size;
  int fp4_padded;
  int fill_mode;
  int ptr_offset;
  int shape_offsets[TRITON_MAX_TMA_DIMS];
  int stride_offsets[TRITON_MAX_TMA_DIMS];
  int desc_param_idx;
} triton_tma_recipe_t;

/**
 * Parameter descriptor: where in args_buf each kernel param lives.
 * 'size' is currently unused by triton_launch_kernel() (which only reads
 * offset and is_tma) but reserved for future use (e.g. debug validation).
 */
typedef struct {
  int offset;
  int size;
  int is_tma;
} triton_param_desc_t;

/**
 * Per-kernel launch descriptor: everything needed to launch a kernel.
 * Generated by the compiler as a static const, passed to
 * triton_launch_kernel().
 */
typedef struct {
  int abi_version; /* must equal TRITON_LAUNCH_DESC_ABI_VERSION */
  int num_warps;
  int num_ctas;
  unsigned shared_mem;
  int launch_pdl;
  int launch_cooperative_grid;
  int launch_cluster;
  /* Explicit multi-dimensional cluster (the ctas_per_cga path): when num_ctas
   * == 1 and any entry is > 1, these become
   * CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION. All entries <= 1 means "no explicit
   * cluster" -- the 1-D num_ctas path runs instead. (num_ctas takes priority,
   * so the two are never both applied.) */
  int cluster_dims[3];
  int preferred_cluster_dims[3];

  int num_params;
  triton_param_desc_t params[TRITON_MAX_PARAMS];

  int num_tma_recipes;
  triton_tma_recipe_t tma_recipes[TRITON_MAX_TMA_DESCS];
} triton_kernel_launch_desc_t;

/*
 * Layout guards.  All fields are 4-byte (int / unsigned / int32 arrays /
 * structs of ints) so the structs are tightly packed with no padding.  These
 * asserts catch accidental layout drift between this header and the Python
 * ctypes mirror in third_party/nvidia/backend/launch_desc.py (which assumes the
 * same layout).
 */
TRITON_STATIC_ASSERT(sizeof(triton_param_desc_t) == 3 * sizeof(int),
                     "triton_param_desc_t layout drift");
TRITON_STATIC_ASSERT(sizeof(triton_tma_recipe_t) == 23 * sizeof(int),
                     "triton_tma_recipe_t layout drift");
TRITON_STATIC_ASSERT(sizeof(triton_kernel_launch_desc_t) ==
                         14 * sizeof(int) +
                             TRITON_MAX_PARAMS * sizeof(triton_param_desc_t) +
                             sizeof(int) +
                             TRITON_MAX_TMA_DESCS * sizeof(triton_tma_recipe_t),
                     "triton_kernel_launch_desc_t layout drift");

/* -------------------------------------------------------------------------
 * Lazy-loaded cuTensorMapEncodeTiled (for TMA)
 * ------------------------------------------------------------------------- */

typedef CUresult (*triton_cuTensorMapEncodeTiled_fn)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

static triton_cuTensorMapEncodeTiled_fn g_triton_tma_encode_fn = NULL;

/* Shared libcuda handle for constructor-time symbol resolution. Intentionally
 * never dlclose'd: the library stays mapped for the process lifetime (same
 * pattern as triton_init_launch_kernel_ex above). */
static void *g_triton_libcuda = NULL;

/* Resolved CUDA driver version (e.g. 12080 for 12.8) for the small-tensor
 * CUtensorMap workaround below. <= 0 if unavailable. Resolved by the
 * constructor below, so triton_get_driver_version() is a plain read. */
typedef CUresult (*triton_cuDriverGetVersion_fn)(int *);
static int g_triton_driver_version = -1;

/**
 * Initialize cuTensorMapEncodeTiled + cuDriverGetVersion at program startup.
 * Thread-safe by running before main().
 */
__attribute__((constructor)) static void triton_init_cuda_symbols(void) {
  g_triton_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!g_triton_libcuda)
    return;
  g_triton_tma_encode_fn = (triton_cuTensorMapEncodeTiled_fn)dlsym(
      g_triton_libcuda, "cuTensorMapEncodeTiled");
  triton_cuDriverGetVersion_fn ver_fn = (triton_cuDriverGetVersion_fn)dlsym(
      g_triton_libcuda, "cuDriverGetVersion");
  int v = 0;
  if (ver_fn && ver_fn(&v) == CUDA_SUCCESS)
    g_triton_driver_version = v;
}

static inline triton_cuTensorMapEncodeTiled_fn triton_get_tma_encode(void) {
  return g_triton_tma_encode_fn;
}

static inline int triton_get_driver_version(void) {
  return g_triton_driver_version;
}

/**
 * Construct a single CUtensorMap from a TMA recipe and user args.
 *
 * Mirrors the proven JIT TMA encoder (driver.c td_extract_tensordesc /
 * fill_tma_descriptor_tiled): shape/stride are read from args_buf in Triton
 * row-major order, then reversed to column-major for cuTensorMapEncodeTiled;
 * the outermost global stride is derived; L2 promotion is 128B; and the
 * small-tensor CUtensorMap bit workaround is applied on driver <= 13010.
 */
static inline CUresult
triton_construct_tma_desc(CUtensorMap *desc, const triton_tma_recipe_t *recipe,
                          const void *args_buf) {

  triton_cuTensorMapEncodeTiled_fn encode_fn = triton_get_tma_encode();
  if (!encode_fn)
    return CUDA_ERROR_NOT_FOUND;

  int rank = recipe->ndim;
  CUdeviceptr base_ptr;
  memcpy(&base_ptr, (const char *)args_buf + recipe->ptr_offset,
         sizeof(base_ptr));

  /* Read per-dim shape and stride from args_buf in Triton (row-major) order. */
  int64_t shp[TRITON_MAX_TMA_DIMS] = {0};
  int64_t strd[TRITON_MAX_TMA_DIMS] = {0};
  for (int j = 0; j < rank; j++) {
    memcpy(&shp[j], (const char *)args_buf + recipe->shape_offsets[j],
           sizeof(int64_t));
    if (recipe->stride_offsets[j] >= 0)
      memcpy(&strd[j], (const char *)args_buf + recipe->stride_offsets[j],
             sizeof(int64_t));
    else
      strd[j] = 0;
  }
  if (recipe->fp4_padded && rank > 0)
    shp[rank - 1] *= 2;

  /* Reverse row-major -> column-major for cuTensorMapEncodeTiled. */
  cuuint64_t global_dim[TRITON_MAX_TMA_DIMS] = {0};
  cuuint64_t global_strides[TRITON_MAX_TMA_DIMS] = {0};
  cuuint32_t box_dim[TRITON_MAX_TMA_DIMS] = {0};
  cuuint32_t elem_strides[TRITON_MAX_TMA_DIMS] = {1, 1, 1, 1, 1};
  for (int j = 0; j < rank; j++) {
    box_dim[rank - 1 - j] = recipe->block_shape[j];
    global_dim[rank - 1 - j] = (cuuint64_t)shp[j];
  }
  for (int j = 0; j + 1 < rank; j++)
    global_strides[rank - 2 - j] =
        (cuuint64_t)((int64_t)recipe->elem_size * strd[j]);
  if (rank > 0)
    global_strides[rank - 1] =
        global_dim[rank - 1] *
        (rank == 1 ? (cuuint64_t)recipe->elem_size : global_strides[rank - 2]);

  CUtensorMapSwizzle swizzle_mode = (CUtensorMapSwizzle)recipe->swizzle;
  CUtensorMapFloatOOBfill fill =
      recipe->fill_mode ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                        : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  CUresult r =
      encode_fn(desc, (CUtensorMapDataType)recipe->elem_type, (cuuint32_t)rank,
                (void *)(uintptr_t)base_ptr, global_dim, global_strides,
                box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle_mode, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
  if (r != CUDA_SUCCESS)
    return r;

  /* Small-tensor CUtensorMap workaround for driver <= 13010 (mirrors the JIT
   * dispatcher path): clear bit 21 of word 1 when the tensor fits in 128 KiB.
   * Match the proven path's behavior: apply the workaround when the driver
   * version is unknown (drv == -1) or known to be <= 13010. */
  int drv = triton_get_driver_version();
  if (drv <= 13010) {
    int64_t max_byte_index = 0;
    for (int j = 0; j < rank; j++) {
      int64_t bytes_stride = (j == 0) ? (int64_t)recipe->elem_size
                                      : (int64_t)global_strides[j - 1];
      max_byte_index += ((int64_t)global_dim[j] - 1) * bytes_stride;
    }
    if (max_byte_index + 1 < 128 * 1024) {
      uint64_t *desc_u64 = (uint64_t *)desc;
      desc_u64[1] &= ~(1ull << 21);
    }
  }
  return CUDA_SUCCESS;
}

/**
 * Optional per-recipe TMA cache.
 *
 * Callers that relaunch the same kernel repeatedly (e.g. the JIT dispatcher)
 * can pass one cache entry per TMA recipe plus stable CUtensorMap storage to
 * triton_launch_kernel_cached(); the launcher then skips cuTensorMapEncodeTiled
 * when a tensordesc's base ptr / shape / strides / fill mode are unchanged
 * since the last launch. One entry per recipe; zero-initialize before first
 * use.
 *
 * fill_mode is part of the key because callers (e.g. the dispatcher) may flip a
 * recipe's padding mode between launches; a stale descriptor must not be
 * reused.
 */
typedef struct {
  int valid;
  uint64_t ptr;
  int64_t shape[TRITON_MAX_TMA_DIMS];
  int64_t stride[TRITON_MAX_TMA_DIMS];
  int fill_mode;
} triton_tma_cache_entry_t;

/**
 * Like triton_construct_tma_desc, but skips re-encoding when the inputs match
 * the cache. *desc must be stable storage that persists across calls (so a
 * cached descriptor can be reused). If cache is NULL, always encodes.
 */
static inline CUresult triton_construct_tma_desc_cached(
    CUtensorMap *desc, const triton_tma_recipe_t *recipe, const void *args_buf,
    triton_tma_cache_entry_t *cache) {
  if (!cache)
    return triton_construct_tma_desc(desc, recipe, args_buf);

  uint64_t cur_ptr = 0;
  memcpy(&cur_ptr, (const char *)args_buf + recipe->ptr_offset,
         sizeof(cur_ptr));
  int64_t cur_shape[TRITON_MAX_TMA_DIMS] = {0};
  int64_t cur_stride[TRITON_MAX_TMA_DIMS] = {0};
  for (int d = 0; d < recipe->ndim; d++) {
    memcpy(&cur_shape[d], (const char *)args_buf + recipe->shape_offsets[d],
           sizeof(int64_t));
    if (recipe->stride_offsets[d] >= 0)
      memcpy(&cur_stride[d], (const char *)args_buf + recipe->stride_offsets[d],
             sizeof(int64_t));
    else
      cur_stride[d] = -1; /* sentinel: contiguous dim (stride_offsets < 0).
                           * Safe: stride_offsets is compile-time-static per
                           * recipe, and real strides are always >= 0. */
  }

  if (cache->valid && cache->ptr == cur_ptr &&
      cache->fill_mode == recipe->fill_mode) {
    int same = 1;
    for (int d = 0; d < recipe->ndim; d++) {
      if (cache->shape[d] != cur_shape[d] ||
          cache->stride[d] != cur_stride[d]) {
        same = 0;
        break;
      }
    }
    if (same)
      return CUDA_SUCCESS; /* reuse the previously-encoded *desc */
  }

  CUresult r = triton_construct_tma_desc(desc, recipe, args_buf);
  if (r == CUDA_SUCCESS) {
    cache->valid = 1;
    cache->ptr = cur_ptr;
    cache->fill_mode = recipe->fill_mode;
    for (int d = 0; d < recipe->ndim; d++) {
      cache->shape[d] = cur_shape[d];
      cache->stride[d] = cur_stride[d];
    }
  }
  return r;
}

/* -------------------------------------------------------------------------
 * triton_launch_kernel — THE one function all consumers call.
 * ------------------------------------------------------------------------- */

/**
 * Shared launch core (NVIDIA/CUDA only): constructs TMA descriptors, builds the
 * params[] array and launch attributes, then issues cuLaunchKernelEx.
 *
 * This is the internal implementation. Consumers (JIT variadic launcher,
 * TritonCC, AOT-T) do not call it directly; they go through one of the two thin
 * wrappers below:
 *   - triton_launch_kernel()        stateless: TMA descriptors are rebuilt on
 *                                   the stack every call (no cache).
 *   - triton_launch_kernel_cached() reuses caller-owned TMA descriptor storage
 *                                   and an optional per-recipe cache.
 *
 * @param grid       Grid dimensions [x, y, z]
 * @param stream     CUDA stream
 * @param function   CUDA function handle
 * @param args_buf   Flat buffer containing user args at known offsets
 * @param desc       Per-kernel static launch descriptor (compiler-generated)
 * @param tma_descs  Storage for >= desc->num_tma_recipes CUtensorMap, filled
 *                   here from desc->tma_recipes + args_buf and pointed at by
 * the kernel's TMA params. Must be 64-byte aligned; unused when
 *                   desc->num_tma_recipes == 0.
 * @param tma_cache  Optional per-recipe cache (>= desc->num_tma_recipes
 * entries, zero-initialized before first use) so a repeated launch can skip
 * re-encoding an unchanged CUtensorMap. NULL disables caching (descriptors are
 * rebuilt every call).
 * @return           CUDA_SUCCESS or error code
 */
static inline CUresult triton_launch_kernel_impl(
    const uint32_t grid[3], CUstream stream, CUfunction function,
    void *args_buf, const triton_kernel_launch_desc_t *desc,
    CUtensorMap *tma_descs, triton_tma_cache_entry_t *tma_cache) {

  if (!function)
    return CUDA_ERROR_INVALID_HANDLE;
  if (!desc)
    return CUDA_ERROR_INVALID_VALUE;
  /* Reject a descriptor built against a different ABI (e.g. a stale ctypes
   * mirror or a generated launcher compiled against an older launch.h). */
  if (desc->abi_version != TRITON_LAUNCH_DESC_ABI_VERSION)
    return CUDA_ERROR_INVALID_VALUE;

  /* Empty grid: nothing to launch. Matches the legacy JIT _launch skip and
   * avoids cuLaunchKernelEx rejecting a zero-sized grid. */
  if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0)
    return CUDA_SUCCESS;

  triton_cuLaunchKernelEx_fn launch_fn = triton_get_launch_kernel_ex();
  if (!launch_fn)
    return CUDA_ERROR_NOT_FOUND;

  /* --- TMA: construct descriptors from recipes (cached if tma_cache != NULL)
   */
  if (desc->num_tma_recipes > 0) {
    if (desc->num_tma_recipes > TRITON_MAX_TMA_DESCS) {
      fprintf(stderr,
              "[triton] ERROR: num_tma_recipes (%d) exceeds "
              "TRITON_MAX_TMA_DESCS (%d)\n",
              desc->num_tma_recipes, TRITON_MAX_TMA_DESCS);
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (!triton_get_tma_encode()) {
      fprintf(stderr,
              "[triton] ERROR: kernel has %d TMA recipes but "
              "cuTensorMapEncodeTiled is unavailable (driver too old?)\n",
              desc->num_tma_recipes);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }
  for (int i = 0; i < desc->num_tma_recipes; i++) {
    TRITON_CUDA_CHECK(triton_construct_tma_desc_cached(
        &tma_descs[i], &desc->tma_recipes[i], args_buf,
        tma_cache ? &tma_cache[i] : NULL));
  }

  /* --- Build kernel params[] array --- */
  if (desc->num_params > TRITON_MAX_PARAMS)
    return CUDA_ERROR_INVALID_VALUE;
  void *params[TRITON_MAX_PARAMS];
  int tma_idx = 0;
  for (int i = 0; i < desc->num_params; i++) {
    if (desc->params[i].is_tma) {
      if (tma_idx >= desc->num_tma_recipes)
        return CUDA_ERROR_INVALID_VALUE;
      params[i] = (void *)&tma_descs[tma_idx++];
    } else {
      params[i] = (void *)((char *)args_buf + desc->params[i].offset);
    }
  }

  /* --- Build launch attributes --- */
  CUlaunchAttribute attrs[TRITON_MAX_LAUNCH_ATTRS];
  unsigned num_attrs = 0;

  if (desc->launch_pdl) {
    attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attrs[num_attrs].value.programmaticStreamSerializationAllowed = 1;
    num_attrs++;
  }

  if (desc->launch_cooperative_grid) {
    attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[num_attrs].value.cooperative = 1;
    num_attrs++;
  }

  int has_multidim_cluster =
      (desc->cluster_dims[0] > 1 || desc->cluster_dims[1] > 1 ||
       desc->cluster_dims[2] > 1);
  if (desc->launch_cluster || desc->num_ctas > 1 || has_multidim_cluster) {
    if (desc->num_ctas > 1) {
      /* 1-D cluster: num_ctas CTAs along x. */
      attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attrs[num_attrs].value.clusterDim.x = (unsigned)desc->num_ctas;
      attrs[num_attrs].value.clusterDim.y = 1;
      attrs[num_attrs].value.clusterDim.z = 1;
      num_attrs++;
    } else if (has_multidim_cluster) {
      /* Explicit multi-dimensional cluster (ctas_per_cga). */
      attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attrs[num_attrs].value.clusterDim.x = (unsigned)desc->cluster_dims[0];
      attrs[num_attrs].value.clusterDim.y = (unsigned)desc->cluster_dims[1];
      attrs[num_attrs].value.clusterDim.z = (unsigned)desc->cluster_dims[2];
      num_attrs++;
    }
    attrs[num_attrs].id =
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attrs[num_attrs].value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    num_attrs++;
  }

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
  if (desc->preferred_cluster_dims[0] > 0) {
    attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
    attrs[num_attrs].value.clusterDim.x =
        (unsigned)desc->preferred_cluster_dims[0];
    attrs[num_attrs].value.clusterDim.y =
        (unsigned)desc->preferred_cluster_dims[1];
    attrs[num_attrs].value.clusterDim.z =
        (unsigned)desc->preferred_cluster_dims[2];
    num_attrs++;
  }
#endif

  /* --- Launch --- */
  CUlaunchConfig config;
  config.gridDimX = grid[0] * (uint32_t)desc->num_ctas;
  config.gridDimY = grid[1];
  config.gridDimZ = grid[2];
  config.blockDimX = 32 * (uint32_t)desc->num_warps;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = desc->shared_mem;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;

  return launch_fn(&config, function, params, NULL);
}

static inline CUresult
triton_launch_kernel(const uint32_t grid[3], CUstream stream,
                     CUfunction function, void *args_buf,
                     const triton_kernel_launch_desc_t *desc) {
  /* Stateless path: TMA descriptors built on the stack each call, no cache. */
  __attribute__((aligned(64))) CUtensorMap tma_descs[TRITON_MAX_TMA_DESCS];
  return triton_launch_kernel_impl(grid, stream, function, args_buf, desc,
                                   tma_descs, NULL);
}

/**
 * Cached variant for callers that relaunch the same kernel repeatedly.
 *
 * @param tma_descs  Caller-owned stable storage of >= desc->num_tma_recipes
 *                   CUtensorMap (persisted across calls so cached descriptors
 *                   can be reused); should be 64-byte aligned.
 * @param tma_cache  Caller-owned array of >= desc->num_tma_recipes cache
 *                   entries, zero-initialized before first use. May be NULL.
 */
static inline CUresult triton_launch_kernel_cached(
    const uint32_t grid[3], CUstream stream, CUfunction function,
    void *args_buf, const triton_kernel_launch_desc_t *desc,
    CUtensorMap *tma_descs, triton_tma_cache_entry_t *tma_cache) {
  return triton_launch_kernel_impl(grid, stream, function, args_buf, desc,
                                   tma_descs, tma_cache);
}

/* -------------------------------------------------------------------------
 * Hook support (optional)
 * ------------------------------------------------------------------------- */

typedef void (*triton_launch_hook_fn)(void *metadata);

/**
 * Per-translation-unit hook function pointers.  Set by the runtime before
 * first launch.  If NULL (default), hooks are skipped.
 *
 * These are intentionally `static` (per-TU) because each generated launcher
 * is compiled into its own .so and loaded independently.  For multi-TU
 * scenarios, the runtime should call triton_set_launch_hooks() on each
 * loaded launcher .so individually.
 */
static triton_launch_hook_fn triton_launch_enter_hook = NULL;
static triton_launch_hook_fn triton_launch_exit_hook = NULL;

static inline void triton_set_launch_hooks(triton_launch_hook_fn enter,
                                           triton_launch_hook_fn exit_hook) {
  triton_launch_enter_hook = enter;
  triton_launch_exit_hook = exit_hook;
}

#ifdef __cplusplus
}
#endif

#endif /* TRITON_RUNTIME_LAUNCH_H */
