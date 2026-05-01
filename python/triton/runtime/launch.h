/*
 * triton/runtime/launch.h — Minimal runtime header for Triton standalone
 * launchers.
 *
 * This header provides everything a compiler-generated launcher needs to call
 * cuLaunchKernelEx.  It has NO dependency on Python.h — the generated launcher
 * is a plain C function callable from C, C++, or via ctypes/cffi.
 *
 * Consumers: compiler-generated launcher sources (asm["launcher_src"]),
 *            TritonCC, AOT-T, custom integrations.
 */

#ifndef TRITON_RUNTIME_LAUNCH_H
#define TRITON_RUNTIME_LAUNCH_H

#include <cuda.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

/**
 * Get cuLaunchKernelEx function pointer (loaded at startup).
 * Thread-safe — initialization happens before main().
 * Returns NULL if libcuda.so.1 is not available.
 */
static inline triton_cuLaunchKernelEx_fn triton_get_launch_kernel_ex(void) {
  return g_triton_launch_fn;
}

/* -------------------------------------------------------------------------
 * Launch attribute helpers
 * ------------------------------------------------------------------------- */

/**
 * Maximum number of launch attributes a Triton launcher may set.
 * Currently: PDL, cooperative, cluster dim, cluster scheduling, preferred
 * cluster dim.
 */
#define TRITON_MAX_LAUNCH_ATTRS 5

/**
 * Build the CUlaunchAttribute array and return the number of attributes set.
 *
 * All parameters are compile-time constants baked into the generated launcher.
 * This function is meant to be called from generated code.
 */
static inline unsigned triton_build_launch_attrs(
    CUlaunchAttribute attrs[TRITON_MAX_LAUNCH_ATTRS], int launch_pdl,
    int launch_cooperative_grid, int num_ctas, int launch_cluster,
    int preferred_cluster_dim_x, int preferred_cluster_dim_y,
    int preferred_cluster_dim_z) {
  unsigned n = 0;

  if (launch_pdl) {
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attrs[n].value.programmaticStreamSerializationAllowed = 1;
    n++;
  }

  if (launch_cooperative_grid) {
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[n].value.cooperative = 1;
    n++;
  }

  if (launch_cluster || num_ctas > 1) {
    /* Triton clusters are always 1-D (num_ctas along x); multi-dimensional
     * clusters use the ctas_per_cga / PTX .reqnctapercluster path where
     * num_ctas == 1 and no runtime CLUSTER_DIMENSION attr is needed. */
    if (num_ctas > 1) {
      attrs[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attrs[n].value.clusterDim.x = (unsigned)num_ctas;
      attrs[n].value.clusterDim.y = 1;
      attrs[n].value.clusterDim.z = 1;
      n++;
    }
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attrs[n].value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    n++;
  }

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
  if (preferred_cluster_dim_x > 0) {
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
    attrs[n].value.clusterDim.x = (unsigned)preferred_cluster_dim_x;
    attrs[n].value.clusterDim.y = (unsigned)preferred_cluster_dim_y;
    attrs[n].value.clusterDim.z = (unsigned)preferred_cluster_dim_z;
    n++;
  }
#else
  (void)preferred_cluster_dim_x;
  (void)preferred_cluster_dim_y;
  (void)preferred_cluster_dim_z;
#endif

  return n;
}

/**
 * Build and execute a CUlaunchConfig.  Consolidates the common launch pattern.
 *
 * @param grid          Grid dimensions [x, y, z]
 * @param num_warps     Warps per block (compile-time constant)
 * @param num_ctas      CTAs per cluster (compile-time constant)
 * @param shared_mem    Dynamic shared memory in bytes (compile-time constant)
 * @param stream        CUDA stream
 * @param function      CUDA function handle
 * @param params        Kernel parameter array (void*[])
 * @param attrs         Pre-built launch attributes
 * @param num_attrs     Number of launch attributes
 * @return              CUDA_SUCCESS or error code
 */
static inline CUresult
triton_launch_kernel(const uint32_t grid[3], int num_warps, int num_ctas,
                     unsigned shared_mem, CUstream stream, CUfunction function,
                     void **params, CUlaunchAttribute *attrs,
                     unsigned num_attrs) {

  triton_cuLaunchKernelEx_fn launch_fn = triton_get_launch_kernel_ex();
  if (!launch_fn)
    return CUDA_ERROR_NOT_FOUND;

  CUlaunchConfig config;
  config.gridDimX = grid[0] * (uint32_t)num_ctas;
  config.gridDimY = grid[1];
  config.gridDimZ = grid[2];
  config.blockDimX = 32 * (uint32_t)num_warps;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared_mem;
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
