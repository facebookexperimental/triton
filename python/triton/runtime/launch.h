/*
 * triton/runtime/launch.h — Minimal runtime header for Triton standalone
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
 *
 * This (and g_triton_launch_fn) are `static`, so every translation unit that
 * includes this header gets its own copy + its own constructor. When many
 * generated launchers are linked into one binary (e.g. an AOT-T .so with many
 * kernels, or a TritonCC bundle) each TU independently dlopen's libcuda at
 * startup, bumping its (never-released) refcount. That is harmless — the
 * library stays mapped regardless — just redundant; we accept it to keep
 * launch.h header-only (no companion .c to compile/link per consumer).
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

/**
 * Initialize cuTensorMapEncodeTiled at program startup (same pattern as
 * triton_init_launch_kernel_ex). Thread-safe by running before main().
 */
__attribute__((constructor)) static void triton_init_tma_encode(void) {
  void *lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!lib)
    return;
  g_triton_tma_encode_fn =
      (triton_cuTensorMapEncodeTiled_fn)dlsym(lib, "cuTensorMapEncodeTiled");
}

static inline triton_cuTensorMapEncodeTiled_fn triton_get_tma_encode(void) {
  return g_triton_tma_encode_fn;
}

/**
 * Construct a single CUtensorMap from a TMA recipe and user args.
 *
 * STUB: only the shared-core skeleton (recipe struct + this launcher call site)
 * lands in this diff. The encoder body must match the proven JIT encoder
 * byte-for-byte (row-major -> column-major dim reversal, derived outermost
 * stride, L2_128B promotion, small-tensor driver workaround); that correct
 * implementation, with a byte-compare unit test, lands in a later diff. No
 * consumer emits TMA recipes until then (num_tma_recipes == 0), so this path is
 * never exercised in the meantime.
 */
static inline CUresult
triton_construct_tma_desc(CUtensorMap *desc, const triton_tma_recipe_t *recipe,
                          const void *args_buf) {
  (void)desc;
  (void)recipe;
  (void)args_buf;
  return CUDA_ERROR_NOT_SUPPORTED;
}

/* -------------------------------------------------------------------------
 * triton_launch_kernel — THE one function all consumers call.
 * ------------------------------------------------------------------------- */

/**
 * Launch a Triton kernel.  This is the ONLY launch entry point.
 *
 * All consumers (JIT variadic launcher, TritonCC, AOT-T) call this function.
 * It handles: TMA construction, params[] layout, launch attrs,
 * cuLaunchKernelEx.
 *
 * @param grid       Grid dimensions [x, y, z]
 * @param stream     CUDA stream
 * @param function   CUDA function handle
 * @param args_buf   Flat buffer containing user args at known offsets
 * @param desc       Per-kernel static launch descriptor (compiler-generated)
 * @return           CUDA_SUCCESS or error code
 */
static inline CUresult
triton_launch_kernel(const uint32_t grid[3], CUstream stream,
                     CUfunction function, void *args_buf,
                     const triton_kernel_launch_desc_t *desc) {

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

  /* --- TMA: construct descriptors from recipes --- */
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
  __attribute__((aligned(64))) CUtensorMap tma_descs[TRITON_MAX_TMA_DESCS];
  for (int i = 0; i < desc->num_tma_recipes; i++) {
    TRITON_CUDA_CHECK(triton_construct_tma_desc(
        &tma_descs[i], &desc->tma_recipes[i], args_buf));
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
