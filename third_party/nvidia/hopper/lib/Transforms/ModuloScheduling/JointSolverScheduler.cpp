// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// See JointSolverScheduler.h. Problem/solution schema is versioned
// "joint-solver-0.1" and documented in
// python/triton/tools/modulo_joint_solver.py.

#include "JointSolverScheduler.h"

#include "ExhaustiveScheduler.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

#define DEBUG_TYPE "modulo-scheduling-joint-solver"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::triton::gpu {

/// Reservation duration — must match runRauIMS/ExhaustiveScheduler exactly:
/// warp-issue selfLatency slots (NOT the engine occupancy that feeds ResMII;
/// minII already encodes that floor).
static int nodeDuration(const DDGNode &node) {
  if (node.pipeline == HWPipeline::NONE)
    return 1;
  return std::max(node.selfLatency, 1);
}

/// True feasibility upper bound for the II sweep: at II = critical path +
/// total serial work, every op fits in one stage back-to-back, so a schedule
/// always exists. This replaces the heuristic slack window (guard 2) — a
/// complete search proves each II infeasible instead of failing to pack it.
static int serialUpperBound(const DataDependenceGraph &ddg, int minII) {
  auto heights = ddg.computeCriticalPathHeights();
  int criticalPath = 0;
  for (auto &[_, h] : heights)
    criticalPath = std::max(criticalPath, h);
  int64_t serialWork = 0;
  for (const auto &node : ddg.getNodes())
    serialWork += nodeDuration(node);
  int64_t bound = criticalPath + serialWork;
  return static_cast<int>(std::max<int64_t>(minII, bound));
}

static std::string buildProblemJSON(const DataDependenceGraph &ddg, int minII,
                                    int maxII, int smemBudget, int tmemColLimit,
                                    const ModuloScheduleResult *incumbent) {
  // Streaming classification (Twill §5.3): a variable-latency op with no
  // incoming data dependence (TMA input loads) runs ahead of the pipeline
  // behind its ring buffer, so in steady state its consumers do not wait its
  // latency — the solver models its outgoing edges as latency 0 when
  // TRITON_MODULO_STREAMING_VL=1. Ring depth stays a solver decision (the
  // objective already rewards depth against the SMEM budget).
  llvm::DenseSet<unsigned> hasIncoming;
  for (const auto &edge : ddg.getEdges())
    if (edge.distance == 0)
      hasIncoming.insert(edge.dstIdx);
  llvm::json::Array nodes;
  for (const auto &node : ddg.getNodes()) {
    bool streaming =
        node.pipeline == HWPipeline::TMA && !hasIncoming.contains(node.idx);
    nodes.push_back(llvm::json::Object{
        {"id", static_cast<int64_t>(node.idx)},
        {"pipeline", getPipelineName(node.pipeline)},
        {"duration", nodeDuration(node)},
        {"streaming", streaming},
    });
  }
  llvm::json::Array edges;
  for (const auto &edge : ddg.getEdges()) {
    edges.push_back(llvm::json::Object{
        {"src", static_cast<int64_t>(edge.srcIdx)},
        {"dst", static_cast<int64_t>(edge.dstIdx)},
        {"latency", edge.latency},
        {"distance", static_cast<int64_t>(edge.distance)},
    });
  }
  llvm::json::Array buffers;
  for (const auto &buf : extractSchedBuffers(ddg)) {
    llvm::json::Array consumers;
    for (unsigned c : buf.consumerNodes)
      consumers.push_back(static_cast<int64_t>(c));
    buffers.push_back(llvm::json::Object{
        {"alloc_node", static_cast<int64_t>(buf.allocNodeIdx)},
        {"kind", buf.isTmem ? "tmem" : "smem"},
        {"size_bytes", buf.sizeBytes},
        {"tmem_cols", buf.tmemCols},
        {"consumers", std::move(consumers)},
    });
  }

  double timeLimitS = 20.0;
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_TIMEOUT_S");
      !env.empty())
    timeLimitS = std::max(1.0, std::atof(env.c_str()));
  int64_t normalizeU = 300;
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_NORMALIZE");
      !env.empty())
    normalizeU = std::atoll(env.c_str());

  bool streamingVL =
      tools::getBoolEnv("TRITON_MODULO_STREAMING_VL"); // default off

  llvm::json::Object root{
      {"version", "joint-solver-0.1"},
      {"min_ii", minII},
      {"max_ii", maxII},
      {"smem_budget", smemBudget},
      {"tmem_col_limit", tmemColLimit},
      {"time_limit_s", timeLimitS},
      {"normalize_u", normalizeU},
      {"streaming_vl", streamingVL},
      {"nodes", std::move(nodes)},
      {"edges", std::move(edges)},
      {"buffers", std::move(buffers)},
  };
  if (incumbent && incumbent->II > 0) {
    llvm::json::Object cyc;
    for (const auto &[idx, c] : incumbent->nodeToCycle)
      cyc[std::to_string(idx)] = c;
    root["incumbent"] =
        llvm::json::Object{{"ii", incumbent->II}, {"cycles", std::move(cyc)}};
  }
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(std::move(root));
  return out;
}

/// Re-verify the subprocess's schedule against the same constraints the
/// in-process schedulers enforce: dependences and exclusive modular
/// reservation. A schedule that fails here is discarded (fall back to the
/// heuristics) — the Python solver is advisory, never trusted.
/// Under TRITON_MODULO_STREAMING_VL the solver legitimately places a
/// streaming producer's consumers inside its raw latency (the ring absorbs
/// it), so verification uses the same effective latency-0 rule for those
/// edges — otherwise every streaming schedule would be rejected here.
static bool verifySolution(const DataDependenceGraph &ddg,
                           const ModuloScheduleResult &res) {
  if (res.II <= 0)
    return false;
  llvm::DenseSet<unsigned> streaming;
  if (tools::getBoolEnv("TRITON_MODULO_STREAMING_VL")) {
    llvm::DenseSet<unsigned> hasIncoming;
    for (const auto &edge : ddg.getEdges())
      if (edge.distance == 0)
        hasIncoming.insert(edge.dstIdx);
    for (const auto &node : ddg.getNodes())
      if (node.pipeline == HWPipeline::TMA && !hasIncoming.contains(node.idx))
        streaming.insert(node.idx);
  }
  for (const auto &edge : ddg.getEdges()) {
    auto s = res.nodeToCycle.find(edge.srcIdx);
    auto d = res.nodeToCycle.find(edge.dstIdx);
    if (s == res.nodeToCycle.end() || d == res.nodeToCycle.end())
      return false;
    int lat = streaming.contains(edge.srcIdx) ? 0 : edge.latency;
    if (d->second <
        s->second + lat - static_cast<int>(edge.distance) * res.II) {
      LLVM_DEBUG(DBGS() << "verify: dependence violated N" << edge.srcIdx
                        << " -> N" << edge.dstIdx << "\n");
      return false;
    }
  }
  ModuloReservationTable table(res.II);
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline == HWPipeline::NONE)
      continue;
    int dur = nodeDuration(node);
    int cycle = res.nodeToCycle.lookup(node.idx);
    if (cycle < 0 || dur > res.II)
      return false;
    if (!table.isIntervalFree(cycle, node.pipeline, dur)) {
      LLVM_DEBUG(DBGS() << "verify: reservation conflict at N" << node.idx
                        << "\n");
      return false;
    }
    table.reserve(cycle, node.pipeline, node.idx, dur);
  }
  return true;
}

FailureOr<std::string> runJointSolverSubprocess(llvm::StringRef problemJson) {
  llvm::SmallString<128> inPath, outPath;
  if (llvm::sys::fs::createTemporaryFile("modulo-joint-solver-in", "json",
                                         inPath))
    return failure();
  if (llvm::sys::fs::createTemporaryFile("modulo-joint-solver-out", "json",
                                         outPath)) {
    llvm::sys::fs::remove(inPath);
    return failure();
  }
  llvm::scope_exit cleanup([&] {
    llvm::sys::fs::remove(inPath);
    llvm::sys::fs::remove(outPath);
  });

  {
    std::error_code ec;
    llvm::raw_fd_ostream os(inPath, ec);
    if (ec)
      return failure();
    os << problemJson;
  }

  // Same subprocess pattern as LLMSchedulePass. The default command form
  // works wherever the invoking environment can import triton (the normal
  // JIT path); standalone triton-opt runs can point
  // TRITON_MODULO_JOINT_SOLVER_CMD at an explicit interpreter + script.
  std::string cmdPrefix = "python3 -m triton.tools.modulo_joint_solver";
  if (auto env = tools::getStrEnv("TRITON_MODULO_JOINT_SOLVER_CMD");
      !env.empty())
    cmdPrefix = env;
  std::string cmd = cmdPrefix + " " + std::string(inPath) + " " +
                    std::string(outPath) + " 2>/dev/null";
  int ret = std::system(cmd.c_str());
  if (ret != 0)
    LLVM_DEBUG(DBGS() << "solver exited with " << ret << "\n");

  auto bufOrErr = llvm::MemoryBuffer::getFile(outPath);
  if (!bufOrErr)
    return failure();
  return (*bufOrErr)->getBuffer().str();
}

FailureOr<ModuloScheduleResult>
runJointSolverSchedule(const DataDependenceGraph &ddg, int minII,
                       const ModuloScheduleResult *incumbent, int smemBudget,
                       int tmemColLimit) {
  if (minII <= 0 || ddg.getNumNodes() == 0)
    return failure();
  int maxII = serialUpperBound(ddg, minII);
  LLVM_DEBUG(DBGS() << "minII=" << minII << " maxII(serial bound)=" << maxII
                    << " nodes=" << ddg.getNumNodes() << "\n");

  auto rawOut = runJointSolverSubprocess(
      buildProblemJSON(ddg, minII, maxII, smemBudget, tmemColLimit, incumbent));
  if (failed(rawOut))
    return failure();
  auto parsed = llvm::json::parse(*rawOut);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *obj = parsed->getAsObject();
  if (!obj)
    return failure();
  auto status = obj->getString("status");
  if (!status || *status != "ok") {
    LLVM_DEBUG({
      auto msg = obj->getString("message");
      DBGS() << "solver status: " << (status ? *status : "<none>") << " "
             << (msg ? *msg : "") << "\n";
    });
    return failure();
  }

  ModuloScheduleResult result;
  auto ii = obj->getInteger("ii");
  auto *cycles = obj->getObject("cycles");
  if (!ii || !cycles)
    return failure();
  result.II = static_cast<int>(*ii);
  for (const auto &kv : *cycles) {
    unsigned idx = 0;
    if (llvm::StringRef(kv.first).getAsInteger(10, idx))
      return failure();
    auto cyc = kv.second.getAsInteger();
    if (!cyc)
      return failure();
    result.nodeToCycle[idx] = static_cast<int>(*cyc);
  }
  if (result.nodeToCycle.size() != ddg.getNumNodes())
    return failure();

  // TRITON_MODULO_SCHED_SHIFT=k (debug): rigidly translate the solution by
  // +k cycles before verification. A modulo schedule is model-equivalent
  // under translation (dependences and modular reservations are invariant),
  // but the stage split (cycle / II) is NOT — this knob deterministically
  // samples the emitter-facing stage structures that solver nondeterminism
  // otherwise draws at random, for hunting shape-dependent emitter bugs
  // (case4 flake, 2026-07-10).
  if (auto env = tools::getStrEnv("TRITON_MODULO_SCHED_SHIFT"); !env.empty()) {
    int shift = std::atoi(env.c_str());
    if (shift > 0) {
      LLVM_DEBUG(DBGS() << "shifting schedule by +" << shift << " cycles\n");
      for (auto &kv : result.nodeToCycle)
        kv.second += shift;
    }
  }

  if (!verifySolution(ddg, result)) {
    LLVM_DEBUG(DBGS() << "solution failed re-verification — discarding\n");
    return failure();
  }
  LLVM_DEBUG(DBGS() << "SUCCESS at II=" << result.II << "\n");
  return result;
}

} // namespace mlir::triton::gpu
