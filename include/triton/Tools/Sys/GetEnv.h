#ifndef TRITON_TOOLS_SYS_GETENV_H
#define TRITON_TOOLS_SYS_GETENV_H

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>

namespace mlir::triton {

inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    "AMDGCN_ENABLE_DUMP",
    "AMDGCN_USE_BUFFER_ATOMICS",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_MMA_V5",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING",
    "LLVM_PASS_PLUGIN_PATH",
    "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_DUMP_PATH",
    "MLIR_ENABLE_TIMING",
    "MLIR_DISABLE_MULTITHREADING",
    "TRITON_ALLOW_NPOT",
    "TRITON_DEFAULT_FP_FUSION",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_DISABLE_SLPVECTORIZATION",
    "TRITON_DISABLE_POST_MISCHED",
    "TRITON_DUMP_MIR",
    "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_ENABLE_TTGIR_SCHED",
    "TRITON_TTGIR_SCHED_APPLY",
    "TRITON_TTGIR_SCHED_BARRIER_STRIDE",
    "TRITON_TTGIR_SCHED_SLICE_LOADS",
    "TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS",
    "TRITON_ENABLE_AMD_MODULO",
    "TRITON_AMD_MODULO_SERIALIZE",
    "TRITON_AMD_MODULO_MAX_STAGE",
    "TRITON_AMD_EARLY_LOWER",
    "TRITON_HIP_USE_ASYNC_COPY",
    "TRITON_HIP_USE_BLOCK_PINGPONG",
    "TRITON_HIP_USE_IN_THREAD_TRANSPOSE",
    "TRITON_LLVM_DEBUG_ONLY",
    "TRITON_ENABLE_ASAN",
    "TRITON_OVERRIDE_ARCH",
    "TRITON_USE_META_WS",
    "TRITON_DISABLE_WSBARRIER_REORDER",
    "TRITON_WS_SMEM_PLAN_SEARCH",
    "TRITON_WS_MEM_PLAN_TOPK",
    "TRITON_WS_MEM_PLAN_PICK",
    "TRITON_WS_MEM_PLAN_TOPK_DUMP",
    "TRITON_WS_STAGING_COPIES",
    "USE_IR_LOC",
    "NVPTX_ENABLE_DUMP",
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION",
    "TRITON_F32_DEFAULT",
    "TRITON_PREFER_TMEM_16x256_LAYOUT",
    "TRITON_DUMP_TTGIR_TO_TLX",
    "TRITON_DUMP_TLX_BENCHMARK",
    "TRITON_ENABLE_EXPERIMENTAL_CONSAN",
    "TRITON_PLUGIN_PATHS",
    "TRITON_STRICT_REDUCTION_ORDERING",
    "TRITON_USE_MODULO_SCHEDULE",
    "TRITON_USE_LIST_SCHEDULE",
    "TRITON_LIST_SCHEDULE_TOPK",
    "TRITON_LIST_SCHEDULE_TOPK_DUMP",
    "TRITON_LIST_SCHEDULE_BEAM",
    "TRITON_LIST_SCHEDULE_PICK",
    "TRITON_MODULO_DUMP_SCHEDULE",
    "TRITON_MODULO_SMEM_BUDGET_KB",
    "TRITON_MODULO_EXHAUSTIVE_PARTITION",
    "TRITON_MODULO_CLUSTER_GREEDY",
    "TRITON_MODULO_STAGE_SEPARATION",
    "TRITON_MODULO_EPILOGUE_SUBTILE",
    "TRITON_MODULO_DUMP_TOPN",
    "TRITON_MODULO_DEBUG_RECMII",
    "TRITON_MODULO_REG_DEFAULT",
    "TRITON_MODULO_CORES_PENALTY",
    "TRITON_MODULO_JOINT_SOLVER_CMD",
    "TRITON_MODULO_JOINT_SOLVER_TIMEOUT_S",
    "TRITON_MODULO_JOINT_SOLVER_NORMALIZE",
    "TRITON_MODULO_ASYNC_FLIGHT_BLOCKING",
    "TRITON_MODULO_CAPACITY_TERM",
    "TRITON_MODULO_RECMII_RERANK",
    "TRITON_MODULO_RECMII_K",
    "TRITON_MODULO_STREAMING_VL",
    "TRITON_MODULO_REG_BUDGET",
    "TRITON_MODULO_SCHED_SHIFT",
    "TRITON_DATA_PARTITION_N",
    "TRITON_MODULO_SELECT_VARIANT",
    "TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT",
    "TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY",
    "TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY",
    // clang-format on
};

inline const std::set<std::string> CACHE_NEUTRAL_ENV_VARS = {
    // clang-format off
    "TRITON_REPRODUCER_PATH",
    "TRITON_ENABLE_PYTHON_STACKTRACE",
    "TRITON_TLX_DUMP_DIR",
    // clang-format on
};

namespace tools {

inline void assertIsRecognized(const std::string &env) {
  bool is_invalidating = CACHE_INVALIDATING_ENV_VARS.find(env.c_str()) !=
                         CACHE_INVALIDATING_ENV_VARS.end();
  bool is_neutral =
      CACHE_NEUTRAL_ENV_VARS.find(env.c_str()) != CACHE_NEUTRAL_ENV_VARS.end();
  std::string errmsg = env + "is not recognized. "
                             "Please add it to triton/Tools/Sys/GetEnv.h";
  assert((is_invalidating || is_neutral) && errmsg.c_str());
}

static std::mutex getenv_mutex;

inline std::string getStrEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  assertIsRecognized(env);
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

// return value of a cache-invalidating boolean environment variable
inline bool getBoolEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  assertIsRecognized(env);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

inline std::optional<bool> isEnvValueBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "on" || str == "true" || str == "1")
    return true;
  if (str == "off" || str == "false" || str == "0")
    return false;
  return std::nullopt;
}
} // namespace tools
} // namespace mlir::triton

#endif
