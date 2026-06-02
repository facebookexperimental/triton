// LLM-based modulo scheduling pass.
//
// Builds the DDG and latency table in C++ (reusing the existing modulo
// scheduling infrastructure), formats them as compact text, and calls
// the Claude API directly via curl to produce the schedule.

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "nvidia/hopper/include/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>

#define DEBUG_TYPE "nvgpu-llm-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

// ============================================================================
// System prompt (embedded)
// ============================================================================

static constexpr llvm::StringLiteral kSystemPrompt =
    R"(You are a GPU kernel scheduling expert. Given a Data Dependence Graph (DDG) with latency information, produce a modulo schedule.

The input provides:
- MinII (already computed from ResMII and RecMII)
- Nodes with op name, pipeline (MEM/TC/CUDA/SFU/NONE), latency, and selfLatency
- Edges with source, destination, latency, and loop-carried distance

Your task: assign each non-NONE node a cycle, then derive stages and clusters.

Rules:
1. Use II = MinII as the initiation interval
2. Place each node at an absolute cycle such that:
   - cycle[dst] >= cycle[src] + edge_latency - edge_distance * II
   - No two ops on the same pipeline share the same modulo slot (cycle % II)
   - selfLatency=1 means each op uses 1 modulo slot; selfLatency=0 means no slot
3. stage = cycle / II
4. Within each stage, sort by cycle and assign dense cluster IDs (lowest cycle = 0)
5. prologue_latency = cycle of first TC node in stage 0
6. Buffer count = lifetime / II + 1, where lifetime = max_consumer_end - producer_cycle

Output format (ONLY this, no explanation):

modulo.schedule @loop0 {
  ii = <II>, max_stage = <S>, prologue_latency = <L>, trip_count = <N>

  modulo.stage @s0 {
    <op_name>  {pipe: <PIPE>, cycle: <C>, cluster: <K>, latency: <L>, selfLatency: <SL>}
  }
  modulo.stage @s1 {
    ...
  }
})";

// ============================================================================
// Format DDG as text for the LLM
// ============================================================================

static std::string getOpShortName(Operation *op) {
  return op->getName().getStringRef().str();
}

static std::string formatDDG(const ttg::DataDependenceGraph &ddg,
                             scf::ForOp loop) {
  std::string result;
  llvm::raw_string_ostream os(result);

  int tripCount = -1;
  if (auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIntOp>()) {
    if (auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>()) {
      if (auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>()) {
        int64_t stepVal = step.value();
        if (stepVal > 0)
          tripCount = (ub.value() - lb.value() + stepVal - 1) / stepVal;
      }
    }
  }

  os << "DDG for loop (trip_count=" << tripCount << "):\n";
  os << "  ResMII=" << ddg.computeResMII() << ", RecMII=" << ddg.computeRecMII()
     << ", MinII=" << ddg.computeMinII() << "\n\n";

  os << "Nodes:\n";
  for (const auto &node : ddg.getNodes()) {
    os << "  N" << node.idx << ": " << getOpShortName(node.op)
       << "  pipe=" << ttg::getPipelineName(node.pipeline)
       << "  lat=" << node.latency << "  selfLat=" << node.selfLatency;
    if (node.minWarps > 1)
      os << "  minWarps=" << node.minWarps;
    if (node.isSuperNode)
      os << "  [super: innerII=" << node.innerII
         << " prologueLat=" << node.prologueLatency << "]";
    os << "\n";
  }

  os << "\nEdges:\n";
  for (const auto &edge : ddg.getEdges()) {
    os << "  N" << edge.srcIdx << " -> N" << edge.dstIdx
       << "  lat=" << edge.latency << "  dist=" << edge.distance << "\n";
  }

  return result;
}

// ============================================================================
// Call local claude CLI
// ============================================================================

static std::string callClaude(const std::string &ddgText) {
  // Combine system prompt + DDG into one system-prompt file.
  // Use a short -p argument to trigger non-interactive mode.
  SmallString<128> sysPath;
  if (llvm::sys::fs::createTemporaryFile("llm-sys", "txt", sysPath))
    return "";
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(sysPath, ec);
    if (ec)
      return "";
    os << kSystemPrompt << "\n\n--- DDG INPUT ---\n\n" << ddgText;
  }

  SmallString<128> outPath;
  if (llvm::sys::fs::createTemporaryFile("llm-out", "txt", outPath)) {
    llvm::sys::fs::remove(sysPath);
    return "";
  }

  const char *modelEnv = std::getenv("TRITON_LLM_MODEL");
  std::string model = modelEnv ? modelEnv : "sonnet";

  std::string sysPathStr(sysPath.begin(), sysPath.end());
  std::string outPathStr(outPath.begin(), outPath.end());

  // Pass system prompt + DDG via --system-prompt-file.
  // Use a short -p argument as the user prompt.
  std::string cmd = "claude --bare"
                    " --system-prompt-file " +
                    sysPathStr +
                    " -p 'Produce the modulo.schedule for the DDG above.'"
                    " --output-format text"
                    " --model " +
                    model +
                    " --max-turns 1"
                    " > " +
                    outPathStr + " 2>/dev/null";

  LDBG("Calling local claude CLI (model=" << model << ")...");
  LDBG("Command: " << cmd);
  int ret = std::system(cmd.c_str());

  llvm::sys::fs::remove(sysPath);

  if (ret != 0) {
    LDBG("claude CLI failed with exit code " << ret);
    llvm::sys::fs::remove(outPath);
    return "";
  }

  std::ifstream outFile(outPathStr);
  std::stringstream buf;
  if (outFile.is_open())
    buf << outFile.rdbuf();
  llvm::sys::fs::remove(outPath);

  return buf.str();
}

// ============================================================================
// Parse LLM response
// ============================================================================

struct MMAAnnotation {
  int stage;
  int order;
};

struct ParsedSchedule {
  int ii = -1;
  int maxStage = -1;
  int numStages = -1;
  SmallVector<MMAAnnotation> mmaAnnotations;
};

static ParsedSchedule parseSchedule(const std::string &output) {
  ParsedSchedule schedule;

  std::regex headerRe(R"(ii\s*=\s*(\d+),\s*max_stage\s*=\s*(\d+))");
  std::smatch match;
  if (std::regex_search(output, match, headerRe)) {
    schedule.ii = std::stoi(match[1]);
    schedule.maxStage = std::stoi(match[2]);
  }

  // Parse MMA node cycle/cluster from modulo.schedule format.
  std::regex mmaRe(R"(tc_gen5_mma[^\}]*cycle:\s*(\d+),\s*cluster:\s*(\d+))");
  auto end = std::sregex_iterator();
  for (auto it = std::sregex_iterator(output.begin(), output.end(), mmaRe);
       it != end; ++it) {
    MMAAnnotation ann;
    int cycle = std::stoi((*it)[1]);
    ann.order = std::stoi((*it)[2]);
    ann.stage = schedule.ii > 0 ? cycle / schedule.ii : 0;
    schedule.mmaAnnotations.push_back(ann);
  }

  // Fallback: "MMA N<id>: stage=<S> order=<O>" format.
  if (schedule.mmaAnnotations.empty()) {
    std::regex altRe(
        R"(MMA\s+N\d+:\s*stage\s*=\s*(\d+)\s*,?\s*order\s*=\s*(\d+))");
    for (auto it = std::sregex_iterator(output.begin(), output.end(), altRe);
         it != end; ++it) {
      MMAAnnotation ann;
      ann.stage = std::stoi((*it)[1]);
      ann.order = std::stoi((*it)[2]);
      schedule.mmaAnnotations.push_back(ann);
    }
  }

  // Derive num_stages from buffer counts or max_stage.
  std::regex bufRe(R"(modulo\.alloc\s+SMEM\s+\[(\d+)\s+x)");
  int maxBuf = 0;
  for (auto it = std::sregex_iterator(output.begin(), output.end(), bufRe);
       it != end; ++it)
    maxBuf = std::max(maxBuf, std::stoi((*it)[1]));

  schedule.numStages = maxBuf > 0 ? maxBuf : (schedule.maxStage + 1);
  return schedule;
}

// ============================================================================
// Apply schedule to IR
// ============================================================================

static void applySchedule(scf::ForOp loop, const ParsedSchedule &schedule) {
  auto ctx = loop.getContext();
  int mmaIdx = 0;

  loop->walk([&](Operation *op) {
    if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op)) {
      if (mmaIdx < (int)schedule.mmaAnnotations.size()) {
        const auto &ann = schedule.mmaAnnotations[mmaIdx];
        std::string json = "{\"stage\": \"" + std::to_string(ann.stage) +
                           "\", \"order\": \"" + std::to_string(ann.order) +
                           "\"}";
        op->setAttr("tt.autows", StringAttr::get(ctx, json));
        LDBG("Set tt.autows on MMA #" << mmaIdx << ": " << json);
        ++mmaIdx;
      }
    }
  });

  if (schedule.numStages > 0) {
    loop->setAttr("tt.num_stages", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                    schedule.numStages));
  }
}

// ============================================================================
// Pass
// ============================================================================

struct LLMSchedulePass
    : public PassWrapper<LLMSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLMSchedulePass)

  LLMSchedulePass() = default;
  LLMSchedulePass(const LLMSchedulePass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "nvgpu-llm-schedule"; }

  StringRef getDescription() const override {
    return "LLM-based modulo scheduling for warp specialization";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    SmallVector<scf::ForOp> loops;
    moduleOp.walk([&](scf::ForOp loop) {
      bool has = false;
      loop->walk([&](Operation *op) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp,
                ttng::AsyncTMACopyGlobalToLocalOp, ttng::TCGen5MMAOp,
                ttng::TCGen5MMAScaledOp>(op))
          has = true;
      });
      if (has)
        loops.push_back(loop);
    });

    if (loops.empty()) {
      LDBG("No schedulable loops — skipping");
      return;
    }

    for (auto loop : loops) {
      auto ddg = ttg::DataDependenceGraph::build(loop, model);
      if (ddg.getNumNodes() == 0)
        continue;

      LDBG("Built DDG: " << ddg.getNumNodes() << " nodes, "
                         << ddg.getEdges().size() << " edges, "
                         << "MinII=" << ddg.computeMinII());

      std::string ddgText = formatDDG(ddg, loop);
      std::string output = callClaude(ddgText);

      if (output.empty()) {
        LDBG("LLM returned empty output");
        continue;
      }

      LDBG("LLM response length: " << output.size());
      LDBG("LLM schedule graph:\n" << output);

      ParsedSchedule schedule = parseSchedule(output);
      if (schedule.ii < 0) {
        LDBG("Failed to parse schedule");
        continue;
      }

      LDBG("Parsed: II=" << schedule.ii << " maxStage=" << schedule.maxStage
                         << " numStages=" << schedule.numStages
                         << " MMAs=" << schedule.mmaAnnotations.size());

      applySchedule(loop, schedule);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVGPULLMSchedule() {
  return std::make_unique<LLMSchedulePass>();
}

void mlir::registerNVGPULLMSchedule() { PassRegistration<LLMSchedulePass>(); }
