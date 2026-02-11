#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define DEBUG_TYPE "tritongpu-partition-scheduling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace {

/// Check if DOT file dumping is enabled via environment variable.
static bool shouldDumpDotFile() {
  static bool enabled = std::getenv("TRITON_DUMP_PARTITION_DOT") != nullptr;
  return enabled;
}

//===----------------------------------------------------------------------===//
// Op Categories and Scheduling Template Infrastructure
//===----------------------------------------------------------------------===//
//
// This section defines the categorization framework for partition scheduling.
// The goal is to categorize ops first, then apply templated scheduling rules.
// Currently this is used for analysis/logging only - the actual scheduling
// logic is unchanged.

/// Categories of operations for partition scheduling.
enum class OpCategory {
  Load,          // TMA loads
  LocalAlloc,    // Buffer allocations
  MMA,           // MMA operations
  MemDescView,   // Memory descriptor views
  EpilogueStore, // Descriptor stores
  TMAReduction,  // TMA reduction operations
  DataPartition, // Ops exclusive to one MMA's slice
  Shared,        // Ops shared across MMAs
  Correction,    // Cross-iteration MMA users
  Default        // Everything else
};

/// Get a string representation of an OpCategory.
static llvm::StringRef toString(OpCategory category) {
  switch (category) {
  case OpCategory::Load:
    return "Load";
  case OpCategory::LocalAlloc:
    return "LocalAlloc";
  case OpCategory::MMA:
    return "MMA";
  case OpCategory::MemDescView:
    return "MemDescView";
  case OpCategory::EpilogueStore:
    return "EpilogueStore";
  case OpCategory::TMAReduction:
    return "TMAReduction";
  case OpCategory::Correction:
    return "Correction";
  case OpCategory::DataPartition:
    return "DataPartition";
  case OpCategory::Shared:
    return "Shared";
  case OpCategory::Default:
    return "Default";
  }
  llvm_unreachable("Unknown OpCategory");
}

//===----------------------------------------------------------------------===//
// Data Partition Detection
//===----------------------------------------------------------------------===//

/// Collect backward slice for an MMA operation.
static SetVector<Operation *>
collectMMABackwardSlice(scf::ForOp loop, ttng::MMAv5OpInterface mmaOp) {
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.inclusive = false;
  options.filter = [&](Operation *op) {
    return loop->isAncestor(op) && !isa<ttng::MMAv5OpInterface>(op);
  };
  for (Value operand : mmaOp->getOperands()) {
    (void)getBackwardSlice(operand, &slice, options);
  }
  return slice;
}

/// Get color for an operation category.
static StringRef getOpColorForCategory(OpCategory cat) {
  switch (cat) {
  case OpCategory::MMA:
    return "lightgreen";
  case OpCategory::Load:
    return "lightblue";
  case OpCategory::EpilogueStore:
    return "lightyellow";
  case OpCategory::LocalAlloc:
  case OpCategory::MemDescView:
    return "lightgray";
  case OpCategory::DataPartition:
    return "lightcyan";
  case OpCategory::Correction:
    return "lightpink";
  case OpCategory::TMAReduction:
    return "pink";
  case OpCategory::Shared:
  case OpCategory::Default:
    return "white";
  }
  return "white";
}

/// Get color for an operation based on its type.
static StringRef getOpColor(Operation *op) {
  // Assign color based on operation type
  if (isa<ttng::TCGen5MMAOp>(op))
    return "lightgreen";
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return "lightblue";
  if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return "lightyellow";
  if (isa<LocalAllocOp>(op))
    return "lightgray";
  if (isa<DescriptorStoreOp>(op))
    return "orange";
  if (isa<ttng::AsyncTMAReduceOp>(op))
    return "pink";
  return "white";
}

//===----------------------------------------------------------------------===//
// DOT File Dumping for Partition Visualization
//===----------------------------------------------------------------------===//

/// Get the loop depth of an operation.
static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp>(parent))
      depth++;
    parent = parent->getParentOp();
  }
  return depth;
}

/// Check if an operation is a "key" operation for visualization.
static bool isKeyOperation(Operation *op) {
  // TMA loads
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return true;
  // MMAs
  if (isa<ttng::TCGen5MMAOp>(op))
    return true;
  // TMA stores
  if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return true;
  // MemDesc allocs
  if (isa<LocalAllocOp>(op))
    return true;
  // Descriptor stores
  if (isa<DescriptorStoreOp>(op))
    return true;
  // Descriptor reduces (TMA atomic reductions)
  if (isa<DescriptorReduceOp>(op))
    return true;
  // Dot operations and local memory ops
  if (isa<DotOp, LocalLoadOp, LocalStoreOp>(op))
    return true;
  return false;
}

/// Get a short label for an operation.
static std::string getOpLabel(Operation *op) {
  std::string label;
  llvm::raw_string_ostream os(label);

  // Get operation name without dialect prefix for brevity
  StringRef opName = op->getName().getStringRef();
  size_t dotPos = opName.rfind('.');
  if (dotPos != StringRef::npos)
    opName = opName.substr(dotPos + 1);

  os << opName;

  // Add tensor type info for loads
  if (auto tmaLoad = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    if (auto descType = tmaLoad.getDesc().getType()) {
      os << "\\n";
      // Print element type if available
      if (auto ptrType = dyn_cast<PointerType>(descType)) {
        if (auto tensorType =
                dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
          os << "[";
          for (unsigned i = 0; i < tensorType.getRank(); i++) {
            if (i > 0)
              os << "x";
            os << tensorType.getDimSize(i);
          }
          os << "]";
        }
      }
    }
  }

  // Add shape info for MMA
  if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    os << "\\n[MxNxK]";
  }

  return label;
}

/// Get a one-line pretty representation of an operation for debug printing.
/// Format: "op_name <shape> (depth=N)"
static std::string prettyOp(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  // Op name (short form without dialect prefix)
  StringRef opName = op->getName().getStringRef();
  size_t dotPos = opName.rfind('.');
  if (dotPos != StringRef::npos)
    os << opName.substr(dotPos + 1);
  else
    os << opName;

  // Result type info (shape + element type for tensors/memdescs)
  if (op->getNumResults() > 0) {
    SmallVector<std::string> typeStrs;
    for (Value r : op->getResults()) {
      Type ty = r.getType();
      std::string ts;
      llvm::raw_string_ostream tos(ts);
      if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
        tos << "<";
        for (unsigned d = 0; d < tensorTy.getRank(); d++) {
          if (d > 0)
            tos << "x";
          tos << tensorTy.getDimSize(d);
        }
        tos << "x" << tensorTy.getElementType() << ">";
      } else if (auto memDescTy = dyn_cast<MemDescType>(ty)) {
        tos << "<memdesc ";
        for (unsigned d = 0; d < memDescTy.getRank(); d++) {
          if (d > 0)
            tos << "x";
          tos << memDescTy.getShape()[d];
        }
        tos << ">";
      }
      if (!ts.empty())
        typeStrs.push_back(std::move(ts));
    }
    if (!typeStrs.empty()) {
      os << " ";
      for (unsigned i = 0; i < typeStrs.size(); i++) {
        if (i > 0)
          os << ", ";
        os << typeStrs[i];
      }
    }
  }

  os << " (depth=" << getLoopDepth(op) << ")";
  return result;
}

/// Get color for an operation based on its type.
//===----------------------------------------------------------------------===//
// Scheduling Templates (Unified Approach)
//===----------------------------------------------------------------------===//
//
// Abstract partition types based on operation semantics:
// - gemm: gen5 mma operations (core compute)
// - correction: cross-iteration correction ops (online softmax)
// - epilogue: epilogue store operations (descriptor_store)
// - load: TMA load operations (descriptor_load, descriptor_gather)
// - reduction: TMA reduction operations (descriptor_reduce)
// - computation[N]: per-data-partition computation tensor ops
//
// Templates control how abstract partitions map to physical partitions.

/// Abstract partition type for semantic categorization.
enum class AbstractPartition {
  Gemm,        // gen5 mma operations
  Correction,  // cross-iteration correction ops
  Epilogue,    // epilogue store operations
  Load,        // TMA load operations
  Reduction,   // TMA reduction operations
  Computation, // computation tensor ops (per data partition)
  Default      // fallback for uncategorized ops
};

/// Template options for controlling partition placement.
struct TemplateOptions {
  /// If true, merge epilogue into computation partition.
  /// If false, epilogue gets its own partition.
  bool mergeEpilogueIntoComputation = false;

  /// If true, merge reduction into computation partition.
  /// If false, reduction gets its own partition.
  bool mergeReductionIntoComputation = false;

  /// Whether correction ops exist. If false, correction partition is skipped
  /// and correction requests fall back to the default partition.
  bool hasCorrection = false;

  /// Whether TMA reduction ops exist. If false, reduction partition is skipped
  /// and reduction requests fall back to the default partition.
  bool hasReduction = false;

  /// Number of data partitions (for parallel computation chains).
  unsigned numDataPartitions = 1;
};

/// Base class for scheduling templates.
class SchedulingTemplate {
public:
  virtual ~SchedulingTemplate() = default;

  /// Create the partitions for this template in the schedule.
  virtual void createPartitions(WarpSchedule &schedule) = 0;

  /// Get the partition for a given abstract partition type and data partition
  /// ID.
  virtual Partition *getPartition(AbstractPartition absPart,
                                  unsigned dpId = 0) = 0;

  /// Get the epilogue partition (for scheduling epilogue stores).
  virtual Partition *getEpiloguePartition() = 0;

  /// Get the template name for debugging.
  virtual StringRef getName() const = 0;

  /// Get template options.
  virtual const TemplateOptions &getOptions() const = 0;

  /// Get the abstract partition name for a physical partition (for debugging).
  virtual std::string getPartitionName(Partition *partition) const = 0;
};

/// Unified Flash Attention template using general partition rules.
/// Works for both forward and backward passes.
class UnifiedFATemplate : public SchedulingTemplate {
public:
  explicit UnifiedFATemplate(TemplateOptions opts) : options(std::move(opts)) {}

  void createPartitions(WarpSchedule &schedule) override {
    // Match FAForwardTemplate ordering: default, gemm(stage1), load,
    // computation[0..N], epilogue(last).
    defaultPartition = schedule.addPartition(0);
    gemmPartition = schedule.addPartition(1); // Stage 1 for MMA
    loadPartition = schedule.addPartition(0);

    // Create per-data-partition computation partitions
    for (unsigned i = 0; i < options.numDataPartitions; ++i)
      computationPartitions.push_back(schedule.addPartition(0));

    // Correction partition: only create if correction ops exist,
    // otherwise fall back to default to avoid wasting a warp group.
    if (options.hasCorrection)
      correctionPartition = schedule.addPartition(0);
    else
      correctionPartition = defaultPartition;

    // Reduction partition: only create if TMA reduction ops exist.
    if (options.hasReduction) {
      if (options.mergeReductionIntoComputation &&
          !computationPartitions.empty())
        reductionPartition = computationPartitions.back();
      else
        reductionPartition = schedule.addPartition(0);
    } else {
      reductionPartition = defaultPartition;
    }

    // Epilogue partition must be last
    if (options.mergeEpilogueIntoComputation && !computationPartitions.empty())
      epiloguePartition = computationPartitions.back();
    else
      epiloguePartition = schedule.addPartition(0); // Must be last
  }

  Partition *getPartition(AbstractPartition absPart,
                          unsigned dpId = 0) override {
    switch (absPart) {
    case AbstractPartition::Gemm:
      return gemmPartition;
    case AbstractPartition::Load:
      return loadPartition;
    case AbstractPartition::Correction:
      return correctionPartition;
    case AbstractPartition::Epilogue:
      return epiloguePartition;
    case AbstractPartition::Reduction:
      return reductionPartition;
    case AbstractPartition::Computation:
      return dpId < computationPartitions.size() ? computationPartitions[dpId]
                                                 : defaultPartition;
    case AbstractPartition::Default:
      return defaultPartition;
    }
    llvm_unreachable("Unknown abstract partition");
  }

  Partition *getEpiloguePartition() override { return epiloguePartition; }
  StringRef getName() const override { return "UnifiedFA"; }
  const TemplateOptions &getOptions() const override { return options; }

  std::string getPartitionName(Partition *partition) const override {
    if (partition == defaultPartition)
      return "default";
    if (partition == gemmPartition)
      return "gemm";
    if (partition == loadPartition)
      return "load";
    if (partition == correctionPartition && options.hasCorrection)
      return "correction";
    if (partition == reductionPartition && options.hasReduction)
      return "reduction";
    if (partition == epiloguePartition)
      return "epilogue";
    for (unsigned i = 0; i < computationPartitions.size(); ++i) {
      if (partition == computationPartitions[i]) {
        if (computationPartitions.size() == 1)
          return "computation";
        return "computation[" + std::to_string(i) + "]";
      }
    }
    return "dynamic";
  }

private:
  TemplateOptions options;
  Partition *gemmPartition = nullptr;
  Partition *loadPartition = nullptr;
  Partition *correctionPartition = nullptr;
  Partition *reductionPartition = nullptr;
  Partition *epiloguePartition = nullptr;
  Partition *defaultPartition = nullptr;
  SmallVector<Partition *> computationPartitions;
};

/// Simple GEMM template (no data partitioning).
class GEMMTemplate : public SchedulingTemplate {
public:
  void createPartitions(WarpSchedule &schedule) override {
    defaultPartition = schedule.addPartition(0);
    gemmPartition = schedule.addPartition(1);
    loadPartition = schedule.addPartition(0);
    epiloguePartition = schedule.addPartition(0);
  }

  Partition *getPartition(AbstractPartition absPart,
                          unsigned dpId = 0) override {
    switch (absPart) {
    case AbstractPartition::Gemm:
      return gemmPartition;
    case AbstractPartition::Load:
      return loadPartition;
    case AbstractPartition::Epilogue:
      return epiloguePartition;
    default:
      return defaultPartition;
    }
  }

  Partition *getEpiloguePartition() override { return epiloguePartition; }
  StringRef getName() const override { return "GEMM"; }
  const TemplateOptions &getOptions() const override {
    static TemplateOptions defaultOpts;
    return defaultOpts;
  }

  std::string getPartitionName(Partition *partition) const override {
    if (partition == defaultPartition)
      return "default";
    if (partition == gemmPartition)
      return "gemm";
    if (partition == loadPartition)
      return "load";
    if (partition == epiloguePartition)
      return "epilogue";
    return "dynamic";
  }

private:
  Partition *defaultPartition = nullptr;
  Partition *gemmPartition = nullptr;
  Partition *loadPartition = nullptr;
  Partition *epiloguePartition = nullptr;
};

/// Map from old OpCategory to new AbstractPartition.
static AbstractPartition toAbstractPartition(OpCategory cat) {
  switch (cat) {
  case OpCategory::MMA:
  case OpCategory::MemDescView:
    return AbstractPartition::Gemm;
  case OpCategory::Load:
  case OpCategory::LocalAlloc:
    return AbstractPartition::Load;
  case OpCategory::EpilogueStore:
    return AbstractPartition::Epilogue;
  case OpCategory::Correction:
    return AbstractPartition::Correction;
  case OpCategory::TMAReduction:
    return AbstractPartition::Reduction;
  case OpCategory::DataPartition:
    return AbstractPartition::Computation;
  case OpCategory::Shared:
  case OpCategory::Default:
    return AbstractPartition::Default;
  }
  return AbstractPartition::Default;
}

//===----------------------------------------------------------------------===//
// OpCategorizer - Categorizes operations for scheduling (Analysis Only)
//===----------------------------------------------------------------------===//
//
// This class implements Phase 1 of the two-phase scheduling approach:
// 1. Categorize all ops based on their role
// 2. Apply scheduling template based on categories (future)
//
// Currently this is used for analysis/logging only - the actual scheduling
// logic in getInitialSchedule() is unchanged.

/// Information about a categorized operation.
struct CategorizedOp {
  Operation *op;
  OpCategory category;
  unsigned dataPartitionId = 0;
  Operation *parentMMA = nullptr;
};

/// Categorizes operations in a loop for partition scheduling.
class OpCategorizer {
public:
  OpCategorizer(scf::ForOp mainLoop, ArrayRef<ttng::MMAv5OpInterface> mmaOps)
      : mainLoop(mainLoop), mmas(mmaOps.begin(), mmaOps.end()) {
    // Collect all loops (nested + main)
    for (auto nestedLoop : mainLoop.getOps<scf::ForOp>())
      loops.push_back(nestedLoop);
    loops.push_back(mainLoop);
  }

  /// Categorize all operations in the loop.
  void categorize() {
    collectMMABackwardSlices();
    categorizeLoads();
    categorizeMMAs();
    categorizeEpilogueStores();
    categorizeTMAReductions();
    categorizeDataPartitionOps();
    categorizeCorrectionOps();
  }

  /// Get the category for an operation.
  OpCategory getCategory(Operation *op) const {
    auto it = opCategories.find(op);
    return it != opCategories.end() ? it->second.category : OpCategory::Default;
  }

  /// Get full categorization info for an operation.
  const CategorizedOp *getCategorizedOp(Operation *op) const {
    auto it = opCategories.find(op);
    return it != opCategories.end() ? &it->second : nullptr;
  }

  /// Get all categorized operations.
  const DenseMap<Operation *, CategorizedOp> &getAllOps() const {
    return opCategories;
  }

  /// Get operations in a specific category.
  SmallVector<CategorizedOp> getOpsInCategory(OpCategory cat) const {
    SmallVector<CategorizedOp> result;
    for (auto &[op, catOp] : opCategories) {
      if (catOp.category == cat)
        result.push_back(catOp);
    }
    return result;
  }

  /// Get the detected data partition factor.
  unsigned getDataPartitionFactor() const { return dataPartitionFactor; }

  /// Get all MMAs.
  ArrayRef<ttng::MMAv5OpInterface> getMMAs() const { return mmas; }

  /// Get all loops (nested + main).
  ArrayRef<scf::ForOp> getLoops() const { return loops; }

  /// Get loads and their associated allocs.
  ArrayRef<Operation *> getLoadsAndAllocs() const { return loadsAndAllocs; }

  /// Get the data partition ID for a specific MMA operation.
  /// Returns -1 if the MMA is not part of data partitioning.
  int getDataPartitionIdForMMA(Operation *mma) const {
    if (dataPartitionFactor <= 1)
      return -1;

    // Iterate through mmaToSlice to find the partition ID
    unsigned partitionId = 0;
    for (auto &[mmaOp, slice] : mmaToSlice) {
      if (mmaOp == mma)
        return partitionId;
      partitionId++;
    }
    return -1;
  }

  /// Print categorization summary (counts only) for debugging.
  void printSummary() const {
    llvm::errs() << "[OpCategorizer] Summary:\n";
    llvm::errs() << "  Loops: " << loops.size() << "\n";
    llvm::errs() << "  MMAs: " << mmas.size() << "\n";
    llvm::errs() << "  Data partition factor: " << dataPartitionFactor << "\n";

    // Count ops per category
    DenseMap<OpCategory, unsigned> counts;
    for (auto &[op, catOp] : opCategories) {
      counts[catOp.category]++;
    }
    for (auto &[cat, count] : counts) {
      llvm::errs() << "  " << toString(cat) << ": " << count << "\n";
    }
  }

  /// Pretty-print all categorized ops grouped by category.
  void printCategorizedOps(llvm::raw_ostream &os) const {
    os << "=== OpCategorizer Results ===\n";
    os << "  Loops: " << loops.size() << ", MMAs: " << mmas.size()
       << ", dpFactor: " << dataPartitionFactor << "\n";

    // Group ops by category in deterministic order
    constexpr OpCategory categoryOrder[] = {
        OpCategory::MMA,           OpCategory::Load,
        OpCategory::LocalAlloc,    OpCategory::MemDescView,
        OpCategory::EpilogueStore, OpCategory::TMAReduction,
        OpCategory::Correction,    OpCategory::DataPartition,
        OpCategory::Shared,        OpCategory::Default};

    for (OpCategory cat : categoryOrder) {
      SmallVector<const CategorizedOp *> ops;
      for (auto &[op, catOp] : opCategories) {
        if (catOp.category == cat)
          ops.push_back(&catOp);
      }
      if (ops.empty())
        continue;

      os << "  [" << toString(cat) << "] (" << ops.size() << " ops):\n";
      for (const auto *catOp : ops) {
        os << "    " << prettyOp(catOp->op);
        if (catOp->dataPartitionId > 0 ||
            catOp->category == OpCategory::DataPartition)
          os << " [dp=" << catOp->dataPartitionId << "]";
        os << "\n";
      }
    }
  }

private:
  void collectMMABackwardSlices() {
    // Only process innermost loop's MMAs for data partitioning
    scf::ForOp innermostLoop = loops.empty() ? mainLoop : loops[0];

    SmallVector<ttng::MMAv5OpInterface> loopMmas;
    for (auto mmaOp : mmas) {
      if (mmaOp->getParentOp() == innermostLoop.getOperation())
        loopMmas.push_back(mmaOp);
    }

    if (loopMmas.size() < 2) {
      dataPartitionFactor = 1;
      return;
    }

    // Collect backward slice for each MMA
    for (auto mmaOp : loopMmas) {
      mmaToSlice[mmaOp.getOperation()] =
          collectMMABackwardSlice(innermostLoop, mmaOp);
    }

    // Find shared ops (appear in multiple slices)
    DenseMap<Operation *, unsigned> opCount;
    for (auto &[mma, slice] : mmaToSlice) {
      for (Operation *op : slice)
        opCount[op]++;
    }
    for (auto &[op, count] : opCount) {
      if (count > 1)
        sharedOps.insert(op);
    }

    // Group dependent MMAs using union-find.
    // MMA B depends on MMA A if A's result feeds (directly or via iter args
    // and intermediate ops) into B's operands.
    // Strategy: For each MMA, collect its forward user set (excluding other
    // MMAs). If that forward set overlaps with another MMA's backward slice,
    // they are dependent.
    unsigned n = loopMmas.size();
    SmallVector<unsigned> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<unsigned(unsigned)> find = [&](unsigned x) -> unsigned {
      return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    auto unite = [&](unsigned a, unsigned b) { parent[find(a)] = find(b); };

    // Build forward reachability from each MMA result (through iter args too)
    for (unsigned i = 0; i < n; ++i) {
      Operation *mmaOp = loopMmas[i].getOperation();
      // Collect all ops reachable from this MMA's results
      DenseSet<Operation *> forwardSet;
      SmallVector<Value> worklist;
      for (Value result : mmaOp->getResults())
        worklist.push_back(result);

      // Also follow cross-iteration paths: MMA result → yield → iter arg
      for (OpOperand &use : mmaOp->getUses()) {
        if (use.getOwner() == innermostLoop.getBody()->getTerminator()) {
          worklist.push_back(
              innermostLoop.getRegionIterArg(use.getOperandNumber()));
        }
      }

      while (!worklist.empty()) {
        Value val = worklist.pop_back_val();
        for (Operation *user : val.getUsers()) {
          if (!innermostLoop->isAncestor(user))
            continue;
          if (isa<ttng::MMAv5OpInterface>(user))
            continue; // Don't traverse through other MMAs
          if (!forwardSet.insert(user).second)
            continue; // Already visited
          for (Value result : user->getResults())
            worklist.push_back(result);
        }
      }

      // Check if any other MMA's backward slice overlaps with this forward set
      for (unsigned j = 0; j < n; ++j) {
        if (i == j || find(i) == find(j))
          continue;
        auto &slice = mmaToSlice[loopMmas[j].getOperation()];
        for (Operation *op : slice) {
          if (forwardSet.contains(op)) {
            unite(i, j);
            break;
          }
        }
      }
    }

    // Count distinct groups that have exclusive (non-shared) ops
    DenseSet<unsigned> groupsWithExclusiveOps;
    for (unsigned i = 0; i < n; ++i) {
      auto &slice = mmaToSlice[loopMmas[i].getOperation()];
      for (Operation *op : slice) {
        if (!sharedOps.contains(op) && !isa<arith::ConstantOp>(op)) {
          groupsWithExclusiveOps.insert(find(i));
          break;
        }
      }
    }

    dataPartitionFactor =
        groupsWithExclusiveOps.size() > 1 ? groupsWithExclusiveOps.size() : 1;

    llvm::errs() << "[data-partition] " << n << " MMAs → "
                 << groupsWithExclusiveOps.size()
                 << " independent groups (dpFactor=" << dataPartitionFactor
                 << ")\n";
  }

  void categorizeLoads() {
    for (auto loop : loops) {
      for (Operation &op : loop.getOps()) {
        if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
          continue;

        addCategorizedOp(&op, OpCategory::Load);
        loadsAndAllocs.push_back(&op);

        // Categorize associated allocs
        SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
        for (Operation *user : op.getUsers()) {
          if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
            if (sharedEnc == alloc.getType().getEncoding()) {
              addCategorizedOp(alloc, OpCategory::LocalAlloc);
              loadsAndAllocs.push_back(alloc);
            }
          } else if (isa<ttng::TMEMAllocOp>(user)) {
            addCategorizedOp(user, OpCategory::LocalAlloc);
            loadsAndAllocs.push_back(user);
          }
        }
      }
    }
  }

  void categorizeMMAs() {
    for (auto mmaOp : mmas) {
      addCategorizedOp(mmaOp, OpCategory::MMA, 0, mmaOp);

      // Categorize memory descriptor views feeding into MMA
      SmallVector<Operation *> worklist;
      for (Value operand : mmaOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
      }
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          continue;
        if (opCategories.contains(op))
          continue;
        addCategorizedOp(op, OpCategory::MemDescView, 0, mmaOp);
        if (Operation *defOp = op->getOperand(0).getDefiningOp())
          worklist.push_back(defOp);
      }
    }
  }

  void categorizeEpilogueStores() {
    for (auto loop : loops) {
      for (auto storeOp : loop.getOps<DescriptorStoreOp>()) {
        addCategorizedOp(storeOp, OpCategory::EpilogueStore);
      }
    }
  }

  void categorizeDataPartitionOps() {
    if (dataPartitionFactor <= 1)
      return;

    // Map exclusive ops to their MMA's partition ID
    unsigned partitionId = 0;
    for (auto &[mma, slice] : mmaToSlice) {
      for (Operation *op : slice) {
        if (!sharedOps.contains(op) && !opCategories.contains(op) &&
            !isa<arith::ConstantOp>(op)) {
          addCategorizedOp(op, OpCategory::DataPartition, partitionId, mma);
        }
      }
      partitionId++;
    }

    // Mark shared ops
    for (Operation *op : sharedOps) {
      if (!opCategories.contains(op)) {
        addCategorizedOp(op, OpCategory::Shared);
      }
    }
  }

  void categorizeCorrectionOps() {
    for (auto mmaOp : mmas) {
      scf::ForOp loop = mmaOp->getParentOfType<scf::ForOp>();
      for (OpOperand &use : mmaOp->getUses()) {
        if (use.getOwner() != loop.getBody()->getTerminator())
          continue;
        // MMA result is yielded - find users in next iteration
        for (OpOperand &iterUse :
             loop.getRegionIterArg(use.getOperandNumber()).getUses()) {
          Operation *user = iterUse.getOwner();
          if (!opCategories.contains(user)) {
            addCategorizedOp(user, OpCategory::Correction);
          }
        }
        break;
      }
    }
  }

  /// Categorize TMA reduction operations (descriptor_reduce and
  /// async_tma_reduce).
  void categorizeTMAReductions() {
    auto isReductionOp = [](Operation *op) {
      return isa<DescriptorReduceOp, ttng::AsyncTMAReduceOp>(op);
    };
    for (scf::ForOp loop : loops) {
      loop.walk([&](Operation *op) {
        if (isReductionOp(op))
          addCategorizedOp(op, OpCategory::TMAReduction);
      });
    }
    // Also check the main loop if not in loops
    if (loops.empty()) {
      mainLoop.walk([&](Operation *op) {
        if (isReductionOp(op))
          addCategorizedOp(op, OpCategory::TMAReduction);
      });
    }
  }

  void addCategorizedOp(Operation *op, OpCategory cat,
                        unsigned dataPartitionId = 0,
                        Operation *parentMMA = nullptr) {
    opCategories[op] = CategorizedOp{op, cat, dataPartitionId, parentMMA};
  }

  scf::ForOp mainLoop;
  SmallVector<scf::ForOp> loops;
  SmallVector<ttng::MMAv5OpInterface> mmas;
  DenseMap<Operation *, CategorizedOp> opCategories;
  DenseMap<Operation *, SetVector<Operation *>> mmaToSlice;
  DenseSet<Operation *> sharedOps;
  SmallVector<Operation *> loadsAndAllocs;
  unsigned dataPartitionFactor = 1;
  bool isBackwardPass = false;
};

/// Dump partitions to a DOT file for visualization.
/// Uses OpCategorizer to get category and data partition information.
static void dumpPartitionsToDot(const WarpSchedule &schedule, scf::ForOp loop,
                                const OpCategorizer *categorizer,
                                StringRef filename = "partitions.dot") {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file " << filename << ": " << EC.message()
                 << "\n";
    return;
  }

  file << "digraph Partitions {\n";
  file << "  rankdir=TB;\n";
  file << "  node [shape=box, style=filled];\n";
  file << "  compound=true;\n\n";

  // Collect all ops in program order per partition
  DenseMap<const Partition *, SmallVector<Operation *>> partitionOps;

  // Walk the loop body in program order
  loop.getBody()->walk([&](Operation *op) {
    if (const Partition *p =
            schedule.getPartition(const_cast<Operation *>(op))) {
      partitionOps[p].push_back(op);
    }
  });

  // Map from operation to unique node ID
  DenseMap<Operation *, std::string> opToNodeId;
  unsigned nodeCounter = 0;

  // Create clusters for each partition
  for (const Partition &partition : schedule.getPartitions()) {
    file << "  subgraph cluster_" << partition.getIndex() << " {\n";
    file << "    label=\"Partition " << partition.getIndex() << " (stage "
         << partition.getStage() << ")\";\n";
    file << "    style=filled;\n";
    file << "    color=lightgrey;\n";
    file << "    fillcolor=\"#f0f0f0\";\n\n";

    // Get ops for this partition in program order
    auto it = partitionOps.find(&partition);
    if (it == partitionOps.end()) {
      file << "    empty_" << partition.getIndex()
           << " [label=\"(empty)\", fillcolor=white];\n";
      file << "  }\n\n";
      continue;
    }

    SmallVector<Operation *> keyOps;
    for (Operation *op : it->second) {
      if (isKeyOperation(op)) {
        keyOps.push_back(op);
      }
    }

    // Create nodes for key operations
    for (Operation *op : keyOps) {
      std::string nodeId = "op_" + std::to_string(nodeCounter++);
      opToNodeId[op] = nodeId;

      unsigned depth = getLoopDepth(op);
      std::string label = getOpLabel(op);
      StringRef color = getOpColor(op);

      // Get category and data partition ID from OpCategorizer if available
      std::string categoryInfo;
      if (categorizer) {
        if (const auto *catOp = categorizer->getCategorizedOp(op)) {
          categoryInfo = toString(catOp->category).str();
          // Add data partition ID for DataPartition category
          if (catOp->category == OpCategory::DataPartition) {
            categoryInfo += " dp=" + std::to_string(catOp->dataPartitionId);
          }
        }
      }

      // Format: "opname\n[category dp=N] (depth=N)" or "opname\n(depth=N)" if
      // no categorizer
      file << "    " << nodeId << " [label=\"" << label;
      if (!categoryInfo.empty()) {
        file << "\\n[" << categoryInfo << "]";
      }
      file << "\\n(depth=" << depth << ")\", fillcolor=" << color << "];\n";
    }

    // If no key ops, show placeholder
    if (keyOps.empty()) {
      file << "    other_" << partition.getIndex() << " [label=\"("
           << it->second.size() << " ops)\", fillcolor=white];\n";
    }

    file << "  }\n\n";
  }

  // Add legend
  file << "\n  // Legend\n";
  file << "  subgraph cluster_legend {\n";
  file << "    label=\"Legend\";\n";
  file << "    style=filled;\n";
  file << "    fillcolor=white;\n";
  file << "    leg_load [label=\"TMA Load\", fillcolor=lightblue];\n";
  file << "    leg_mma [label=\"MMA\", fillcolor=lightgreen];\n";
  file << "    leg_store [label=\"TMA Store\", fillcolor=lightyellow];\n";
  file << "    leg_descstore [label=\"Desc Store\", fillcolor=orange];\n";
  file << "    leg_alloc [label=\"Local Alloc\", fillcolor=lightgray];\n";
  file << "    leg_dp [label=\"DataPartition (dp=N)\", fillcolor=lightcyan];\n";
  file << "  }\n";

  file << "}\n";

  llvm::errs() << "Dumped partition graph to " << filename << "\n";
}

/// Pretty-print partition assignments showing key ops in each partition.
/// This gives a human-readable view of the scheduling result.
static void printPartitionAssignments(const WarpSchedule &schedule,
                                      scf::ForOp loop,
                                      const OpCategorizer *categorizer,
                                      const SchedulingTemplate *tmpl,
                                      llvm::raw_ostream &os) {
  os << "=== Partition Assignments ===\n";
  os << "  Partitions: "
     << std::distance(schedule.getPartitions().begin(),
                      schedule.getPartitions().end())
     << "\n";

  for (const Partition &partition : schedule.getPartitions()) {
    SmallVector<Operation *> keyOps;
    unsigned otherCount = 0;

    for (Operation *op : partition.getOps()) {
      if (isKeyOperation(op))
        keyOps.push_back(op);
      else
        otherCount++;
    }

    os << "  P" << partition.getIndex() << " (stage " << partition.getStage()
       << ")";
    // Print abstract partition name from the template
    if (tmpl) {
      std::string name =
          tmpl->getPartitionName(const_cast<Partition *>(&partition));
      os << " [" << name << "]";
    }
    if (keyOps.empty() && otherCount == 0) {
      os << ": (empty)\n";
      continue;
    }
    os << ":\n";

    for (Operation *op : keyOps) {
      os << "    " << prettyOp(op);
      if (categorizer) {
        auto it = categorizer->getAllOps().find(op);
        if (it != categorizer->getAllOps().end())
          os << " [" << toString(it->second.category) << "]";
      }
      os << "\n";
    }
    if (otherCount > 0)
      os << "    (" << otherCount << " other ops)\n";
  }
}

/// Select the appropriate scheduling template based on the categorized ops.
/// Uses unified template approach - no forward/backward distinction needed.
/// The UnifiedFATemplate creates partitions based on abstract operation roles:
/// - gemm: gen5 mma operations
/// - load: TMA load operations
/// - correction: cross-iteration correction ops
/// - computation[N]: per-data-partition tensor ops
/// - epilogue: descriptor store operations
/// - reduction: TMA reduction operations
/// Template options control merging behavior (e.g., epilogue into computation).
static std::unique_ptr<SchedulingTemplate>
selectTemplate(const OpCategorizer &categorizer) {
  unsigned dpFactor = categorizer.getDataPartitionFactor();
  bool hasCorrection =
      !categorizer.getOpsInCategory(OpCategory::Correction).empty();

  auto epilogueStores = categorizer.getOpsInCategory(OpCategory::EpilogueStore);
  auto mmas = categorizer.getMMAs();

  // Debug output for template selection
  llvm::errs() << "[selectTemplate] dpFactor=" << dpFactor
               << ", hasCorrection=" << hasCorrection
               << ", epilogueStores=" << epilogueStores.size()
               << ", mmas=" << mmas.size() << "\n";

  // Use UnifiedFA for any pattern with multiple MMAs (FA fwd, FA bwd, etc.)
  // or with correction ops. Fall back to GEMM only for simple single-MMA cases.
  if (hasCorrection || mmas.size() > 1 || dpFactor > 1) {
    bool hasReduction =
        !categorizer.getOpsInCategory(OpCategory::TMAReduction).empty();

    TemplateOptions opts;
    opts.numDataPartitions = dpFactor;
    opts.hasCorrection = hasCorrection;
    opts.hasReduction = hasReduction;
    opts.mergeEpilogueIntoComputation = true;
    opts.mergeReductionIntoComputation = true;
    llvm::errs()
        << "[tritongpu-partition-scheduling] Selected template: UnifiedFA"
        << " (dpFactor=" << dpFactor << ")\n";
    return std::make_unique<UnifiedFATemplate>(opts);
  }

  llvm::errs() << "[tritongpu-partition-scheduling] Selected template: GEMM\n";
  return std::make_unique<GEMMTemplate>();
}

} // namespace

//===----------------------------------------------------------------------===//
// assignPartitions
//===----------------------------------------------------------------------===//

// Find the last operation in the loop body that defined this value, with a
// maximum of distance 1.
static Operation *findDefOpInLoop(scf::ForOp loop, Value value,
                                  int distance = 0) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getParentBlock() != loop.getBody())
      return {};
    // Don't look back more than distance 1.
    if (distance == 1)
      return {};
    return findDefOpInLoop(
        loop, loop.getYieldedValues()[arg.getArgNumber() - 1], distance + 1);
  }
  Operation *defOp = value.getDefiningOp();
  if (!loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
    return {};
  return defOp;
}

// For `op`, invoke `callback` on all the definitions of its inputs from within
// `loop`, which might not be in the same iteration.
static void iterateDefs(scf::ForOp loop, Operation *op,
                        function_ref<void(OpResult)> callback) {
  visitNestedOperands(op, [&](OpOperand &operand) {
    Value value = operand.get();
    if (value.getParentBlock() != loop.getBody())
      return;
    auto arg = dyn_cast<BlockArgument>(value);
    if (arg == loop.getInductionVar())
      return;
    auto [def, distance] = getDefinitionAndDistance(loop, operand.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def);
  });
}

// For `op`, invoke `callback` on all its transitive users within `loop`, which
// may be in a future iteration.
static void iterateUsers(scf::ForOp loop, Operation *op,
                         function_ref<void(Operation *)> callback) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (!isa<scf::YieldOp>(owner)) {
      callback(owner);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(&use);
  }
}

// Check if any of the inputs to `op` are reachable from a non-null partition.
static bool hasDefPartition(scf::ForOp loop, Operation *op,
                            WarpSchedule &schedule) {
  SmallVector<Operation *> worklist{op};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    Partition *p = schedule.getPartition(op);
    if (p && p != schedule.getRootPartition())
      return true;
    iterateDefs(loop, op,
                [&](OpResult def) { worklist.push_back(def.getDefiningOp()); });
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Data Partition Detection
// Recursively schedule the dependencies of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleDependencies(scf::ForOp loop, WarpSchedule &schedule,
                                 Partition *partition, Operation *op) {
  SmallVector<Value> deps;
  for (Value value : getNestedOperands(op)) {
    if (isa<RankedTensorType, MemDescType>(value.getType()))
      deps.push_back(value);
  }

  while (!deps.empty()) {
    Value dep = deps.pop_back_val();

    if (auto arg = dyn_cast<BlockArgument>(dep)) {
      if (arg.getOwner() == loop.getBody() && arg != loop.getInductionVar())
        deps.push_back(loop.getYieldedValues()[arg.getArgNumber() - 1]);
      continue;
    }

    Operation *defOp =
        loop.getBody()->findAncestorOpInBlock(*dep.getDefiningOp());
    if (!defOp || !hasDefPartition(loop, defOp, schedule) ||
        !schedule.trySchedule(partition, defOp))
      continue;
    llvm::append_range(deps, getNestedOperands(defOp));
  }
}

// Recursively schedule the users of an operation, stopping when
// encountering an operation that is already assigned.
// If \p partition is null, a new partition will be created if needed.
static Partition *scheduleUsers(scf::ForOp loop, WarpSchedule &schedule,
                                Partition *partition, Operation *op) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = loop.getBody()->findAncestorOpInBlock(*use->getOwner());

    if (user == loop.getBody()->getTerminator()) {
      for (OpOperand &use :
           loop.getRegionIterArg(use->getOperandNumber()).getUses())
        uses.push_back(&use);
      continue;
    }

    if (schedule.isScheduled(user))
      continue;
    if (!partition)
      partition = schedule.addPartition(/* stage is unused */ 0);
    schedule.trySchedule(partition, user);
    for (OpOperand &use : user->getUses())
      uses.push_back(&use);
  }
  return partition;
}

// Schedule post-loop operations (operations outside and after the loop) into
// the epilogue partition. This recursively schedules operations that consume
// loop results and their transitive users.
static void schedulePostLoopOps(scf::ForOp loop, WarpSchedule &schedule,
                                Partition *epiloguePartition) {
  SmallVector<OpOperand *> uses;

  // Collect all uses of the loop's results.
  for (OpResult result : loop.getResults()) {
    for (OpOperand &use : result.getUses())
      uses.push_back(&use);
  }

  // Recursively schedule all post-loop users.
  DenseSet<Operation *> visited;
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = use->getOwner();

    // Skip if already visited or scheduled.
    if (!visited.insert(user).second || schedule.isScheduled(user))
      continue;

    // Only schedule operations that are outside the loop.
    if (loop->isAncestor(user))
      continue;

    // Schedule this post-loop operation to the epilogue partition.
    schedule.trySchedule(epiloguePartition, user);

    // Add all users of this operation to process transitively.
    for (OpResult result : user->getResults())
      for (OpOperand &nextUse : result.getUses())
        uses.push_back(&nextUse);
  }
}

// Given a partitioning scheme, determine an initial schedule by performing a
// first-order partition assignment to the operations in the scheme and its
// users and/or dependencies. This sets up the initial partitioning of the ops.
static std::optional<WarpSchedule> getInitialSchedule(scf::ForOp mainLoop) {
  // Check for an existing schedule.
  if (FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(mainLoop);
      succeeded(scheduleOr))
    return {std::move(*scheduleOr)};

  // Collect all loops (nested + main)
  SmallVector<scf::ForOp> loops{mainLoop.getOps<scf::ForOp>()};
  loops.push_back(mainLoop);

  // Collect all MMAs
  SmallVector<ttng::MMAv5OpInterface> mmas;
  for (auto loop : loops) {
    for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>())
      mmas.push_back(mmaOp);
  }

  //===--------------------------------------------------------------------===//
  // Phase 1: Categorize all operations using OpCategorizer
  //===--------------------------------------------------------------------===//
  OpCategorizer categorizer(mainLoop, mmas);
  categorizer.categorize();

  // Print categorized ops if DOT dumping is enabled
  if (shouldDumpDotFile()) {
    categorizer.printCategorizedOps(llvm::errs());
  }

  unsigned dataPartitionFactor = categorizer.getDataPartitionFactor();
  llvm::errs() << "[tritongpu-partition-scheduling] Using template-based "
                  "scheduling with data partition factor: "
               << dataPartitionFactor << "\n";

  //===--------------------------------------------------------------------===//
  // Phase 2: Select and create template
  //===--------------------------------------------------------------------===//
  std::unique_ptr<SchedulingTemplate> tmpl = selectTemplate(categorizer);
  llvm::errs() << "[tritongpu-partition-scheduling] Selected template: "
               << tmpl->getName() << "\n";

  WarpSchedule schedule;
  tmpl->createPartitions(schedule);

  // Get partition references from template using AbstractPartition
  Partition *defaultPartition = tmpl->getPartition(AbstractPartition::Default);
  Partition *mmaPartition = tmpl->getPartition(AbstractPartition::Gemm);
  Partition *loadPartition = tmpl->getPartition(AbstractPartition::Load);
  Partition *epiloguePartition =
      tmpl->getPartition(AbstractPartition::Epilogue);
  Partition *correctionPartition =
      tmpl->getPartition(AbstractPartition::Correction);
  Partition *reductionPartition =
      tmpl->getPartition(AbstractPartition::Reduction);

  // Use default partition as fallback for correction/reduction if not set
  if (!correctionPartition)
    correctionPartition = defaultPartition;
  if (!reductionPartition)
    reductionPartition = defaultPartition;

  //===--------------------------------------------------------------------===//
  // Phase 3: Schedule ops using template-based partition assignment
  //===--------------------------------------------------------------------===//

  // Schedule loads and their associated allocs
  SmallVector<Operation *> loadsAndAllocs;
  for (auto loop : loops) {
    for (Operation &op : loop.getOps()) {
      if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
        continue;
      schedule.trySchedule(loadPartition, &op);
      loadsAndAllocs.push_back(&op);

      // Local alloc users of the load with matching encoding
      SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
      for (Operation *user : op.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
          if (sharedEnc == alloc.getType().getEncoding()) {
            schedule.trySchedule(loadPartition, alloc);
            loadsAndAllocs.push_back(alloc);
          }
        } else if (isa<ttng::TMEMAllocOp>(user)) {
          schedule.trySchedule(loadPartition, user);
          loadsAndAllocs.push_back(user);
        }
      }
    }
  }

  // Schedule epilogue stores
  for (auto loop : loops)
    for (DescriptorStoreOp op : loop.getOps<DescriptorStoreOp>())
      schedule.trySchedule(epiloguePartition, op);

  // Schedule MMAs and their associated stores
  for (auto loop : loops) {
    for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
      schedule.trySchedule(mmaPartition, mmaOp);

      // If the store is unrelated to the use of the MMA, place in MMA partition
      auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(
          findDefOpInLoop(loop, mmaOp.getAccDep()));
      if (!ttng::hasAccReadModifyWrite(mmaOp, loop) && storeOp &&
          loop.isDefinedOutsideOfLoop(storeOp.getSrc()))
        schedule.trySchedule(mmaPartition, storeOp);
    }
  }

  // Schedule memory descriptor views feeding into MMAs
  for (auto loop : loops) {
    for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
      SmallVector<Operation *> operandViews;
      for (Value operand : mmaOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          operandViews.push_back(defOp);
      }
      while (!operandViews.empty()) {
        Operation *op = operandViews.pop_back_val();
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          continue;

        // Duplicate the op if necessary to ensure MMA partition is only user
        if (!llvm::all_of(op->getUsers(), [&](Operation *user) {
              return schedule.getPartition(user) == mmaPartition;
            })) {
          Operation *newOp = OpBuilder(op).clone(*op);
          op->replaceUsesWithIf(newOp->getResults(), [&](OpOperand &use) {
            return schedule.getPartition(use.getOwner()) == mmaPartition;
          });
          op = newOp;
        }

        schedule.trySchedule(mmaPartition, op);
        if (Operation *defOp = op->getOperand(0).getDefiningOp())
          operandViews.push_back(defOp);
      }
    }
  }

  // If there are no loads or MMAs, don't warp specialize.
  if (loadsAndAllocs.empty() && mmas.empty())
    return std::nullopt;

  //===--------------------------------------------------------------------===//
  // Phase 4: Propagate users using template-based partition assignment
  //===--------------------------------------------------------------------===//

  // Load users go to default partition
  for (Operation *loadOrAlloc : loadsAndAllocs)
    scheduleUsers(loadOrAlloc->getParentOfType<scf::ForOp>(), schedule,
                  defaultPartition, loadOrAlloc);

  // Correction ops (cross-iteration MMA users) go to correction partition
  // Use direct MMA use-chain walking (same as original code) rather than
  // relying on the categorizer, because the categorizer may skip ops that
  // were already categorized under a different category.
  for (auto mmaOp : mmas) {
    for (OpOperand &use : mmaOp->getUses()) {
      auto loop = mmaOp->getParentOfType<scf::ForOp>();
      if (use.getOwner() != loop.getBody()->getTerminator())
        continue;
      for (OpOperand &use :
           loop.getRegionIterArg(use.getOperandNumber()).getUses()) {
        schedule.trySchedule(correctionPartition, use.getOwner());
        scheduleUsers(loop, schedule, correctionPartition, use.getOwner());
      }
      break;
    }
  }

  // TMA reduction ops go to reduction partition
  auto tmaReductionOps = categorizer.getOpsInCategory(OpCategory::TMAReduction);
  for (const auto &catOp : tmaReductionOps) {
    schedule.trySchedule(reductionPartition, catOp.op);
  }

  //===--------------------------------------------------------------------===//
  // Phase 5: Create per-MMA partitions for data partitioning
  //===--------------------------------------------------------------------===//
  // For each MMA in the innermost loop, create a separate partition for its
  // users. This enables data partitioning where different MMA groups have
  // their own computation chains.
  //
  // When dataPartitionFactor > 1, the OpCategorizer has already identified
  // which operations belong to which data partition. We use this information
  // to schedule backward slice ops to the correct partitions, then schedule
  // MMA users as before.

  DenseMap<Operation *, Partition *> mmaToPartition;
  SmallVector<Operation *> inFirstLoop;

  // Use template's data partition partitions if available
  unsigned dpId = 0;
  for (auto mmaOp : llvm::reverse(mmas)) {
    if (mmaOp->getParentOfType<scf::ForOp>() == loops[0]) {
      Partition *dataPart = nullptr;
      if (dataPartitionFactor > 1 && dpId < dataPartitionFactor) {
        // Use template's pre-allocated data partition (Computation partition)
        dataPart = tmpl->getPartition(AbstractPartition::Computation, dpId);
        dpId++;
      }
      // Schedule users, creating a new partition if dataPart is null
      auto part = scheduleUsers(mmaOp->getParentOfType<scf::ForOp>(), schedule,
                                dataPart, mmaOp);
      mmaToPartition[mmaOp.getOperation()] = part;
      inFirstLoop.push_back(mmaOp.getOperation());
    }
  }

  // For causal attention with 3 loops, match MMAs in second loop to first loop
  unsigned Idx = 0;
  for (auto mmaOp : llvm::reverse(mmas)) {
    if (loops.size() == 3 && mmaOp->getParentOfType<scf::ForOp>() == loops[1]) {
      auto *part = mmaToPartition[inFirstLoop[Idx]];
      scheduleUsers(mmaOp->getParentOfType<scf::ForOp>(), schedule, part,
                    mmaOp);
      ++Idx;
    }
  }

  // Print partition assignments if DOT dumping is enabled
  if (shouldDumpDotFile()) {
    printPartitionAssignments(schedule, loops.back(), &categorizer, tmpl.get(),
                              llvm::errs());
  }

  // Dump partitions to DOT file if enabled
  if (shouldDumpDotFile()) {
    dumpPartitionsToDot(schedule, loops.back(), &categorizer,
                        "/tmp/partition_scheduling.dot");
  }

  return schedule;
}

namespace {
// This data structure represents a cluster of operations that have not been
// assigned to a stage. Operations form a cluster when:
//
// - they are adjacent in the SSA use def graph
// - they are not already assigned to a partition
// - at least one of their inputs is reachable from a definition partition
//
struct OpCluster {
  // These are the operations in the cluster.
  SetVector<Operation *> ops;
  // The definition partitions are the partitions from which inputs of the
  // operation are reachable. When the cluster is fully formed, the defining op
  // in the loop of any input to any operation in the cluster is either in the
  // root partition or one of these partitions.
  SetVector<Partition *> defPartitions;
  // The sink partitions which consume the outputs of operations in this
  // cluster. When the cluster is fully formed, all uses in the loop of outputs
  // of any operation in the cluster belong to one of these partitions.
  SetVector<Partition *> sinkPartitions;
};

// Owning class for a bunch of clusters. This class manages the lifetimes of the
// clusters and has some helper functions.
struct OpClusters : public llvm::MapVector<Operation *, OpCluster *> {
  using MapVector::MapVector;

  // Create a new cluster that contains only the given operation, a return a
  // cluster that already contains the operation.
  OpCluster *getOrCreate(Operation *op) {
    OpCluster *&cluster = (*this)[op];
    if (!cluster) {
      cluster = clusters.emplace_back(new OpCluster).get();
      cluster->ops.insert(op);
    }
    return cluster;
  }
  // Merge two clusters by merging their sets and clearing the other cluster,
  // marking it as dead.
  void merge(OpCluster *dst, OpCluster *src) {
    dst->ops.insert_range(src->ops);
    dst->defPartitions.insert_range(src->defPartitions);
    dst->sinkPartitions.insert_range(src->sinkPartitions);
    for (Operation *op : src->ops)
      (*this)[op] = dst;
    src->ops.clear();
    src->defPartitions.clear();
    src->sinkPartitions.clear();
  }

  SmallVector<std::unique_ptr<OpCluster>> clusters;
};
} // namespace

// Operations that require partition assignment are those reachable from an
// operation in a partition. This function propagates partitions by first
// forming contiguous clusters from the unassigned operations and then deciding
// what to do with the operations in that cluster.
void propagatePartitions(scf::ForOp loop, WarpSchedule &schedule) {
  OpClusters opClusters;

  for (Partition &partition : schedule.getPartitions()) {
    // For each partition, check if any of their inputs are reachable from
    // another partition and spawn a single cluster at that operation.
    auto defCallback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (!schedule.isScheduled(defOp) &&
          hasDefPartition(loop, defOp, schedule)) {
        // Add the current partition as a sink to the cluster.
        opClusters.getOrCreate(defOp)->sinkPartitions.insert(&partition);
      }
    };
    schedule.iterateDefs(loop, &partition, defCallback);

    // For each partition, place users of its outputs in a cluster if it is not
    // already assigned to a partition.
    auto useCallback = [&](OpResult result, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (!schedule.isScheduled(user)) {
        // Add the current partition as a def to the cluster.
        opClusters.getOrCreate(user)->defPartitions.insert(&partition);
      }
    };
    schedule.iterateUses(loop, &partition, useCallback);
  }

  // Now we have a pile of single-operation clusters directly adjacent to the
  // operations in a partition. Grow the clusters by adding adjacent operations
  // clusters and merging clusters when possible.
  SmallVector<Operation *> worklist =
      llvm::to_vector(llvm::make_first_range(opClusters));
  while (!worklist.empty()) {
    // Grab an op off the worklist. We know it has a cluster already.
    Operation *op = worklist.pop_back_val();
    OpCluster *cluster = opClusters.find(op)->second;
    // Look at the definitions directly feeding into this operation.
    iterateDefs(loop, op, [&](OpResult def) {
      Operation *defOp = def.getDefiningOp();
      if (schedule.isScheduled(defOp)) {
        // The input originates from an operation already assigned to a
        // partition. Add this as a def partition.
        cluster->defPartitions.insert(schedule.getPartition(defOp));
      } else {
        // If the input is not reachable from a partition, ignore it.
        if (!hasDefPartition(loop, defOp, schedule))
          return;
        // This operation is not assigned to a partition.
        OpCluster *&defCluster = opClusters[defOp];
        if (!defCluster) {
          // This operation has not yet been added to a cluster. Add it to the
          // current cluster and recurse on it.
          defCluster = cluster;
          cluster->ops.insert(defOp);
          worklist.push_back(defOp);
        } else if (defCluster != cluster) {
          // This operation is part of another cluster. Merge the two clusters
          // together and continue.
          opClusters.merge(cluster, defCluster);
        }
      }
    });
    // Check the users of the operation.
    iterateUsers(loop, op, [&](Operation *user) {
      if (schedule.isScheduled(user)) {
        // If the user is already assigned to a partition, add that partition as
        // one of the sink partitions.
        Partition *userPartition = schedule.getPartition(user);
        cluster->sinkPartitions.insert(userPartition);
        return;
      }
      // If the user does not already have a cluster, add it to the current
      // cluster. We don't have to handle merging here because when the user
      // visits the current op, it will trigger the merge.
      OpCluster *&userCluster = opClusters[user];
      if (userCluster)
        return;
      userCluster = cluster;
      cluster->ops.insert(user);
      worklist.push_back(user);
    });
  }

  // We have clustered unassigned ops in the liveouts of ops in assigned
  // partitions and in the critical paths between ops in different partitions.
  // Ops that are next to each other are placed in the same cluster. Now the
  // task is to figure out how to assign partitions to the ops in each cluster
  // based on the def and sink partitions, which is very non-trivial.
  for (OpCluster &cluster : llvm::make_pointee_range(opClusters.clusters)) {
    // Skip dead clusters.
    if (cluster.ops.empty())
      continue;
    assert(!cluster.defPartitions.empty());
    assert(llvm::all_of(
        cluster.ops, [&](Operation *op) { return !schedule.isScheduled(op); }));

    // If there are multiple def or sink partitions, don't know what to do.
    // Assign the whole cluster to its own partition.
    if (cluster.defPartitions.size() > 1 || cluster.sinkPartitions.size() > 1) {
      Partition *newPartition = schedule.addPartition(0);
      for (Operation *op : cluster.ops)
        schedule.insert(newPartition, op);
      continue;
    }

    // If there is no sink partition, this means there is a backedge somewhere,
    // for now assign the cluster to the def partition.
    Partition *defPartition = cluster.defPartitions.front();
    if (cluster.sinkPartitions.empty()) {
      for (Operation *op : cluster.ops)
        schedule.insert(defPartition, op);
      continue;
    }

    // Find the critical path between the def partition and sink partition.
    Partition *sinkPartition = cluster.sinkPartitions.front();
    SetVector<Operation *> critPath;
    DenseSet<Operation *> opsInCluster(cluster.ops.begin(), cluster.ops.end());
    auto callback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (opsInCluster.contains(defOp))
        critPath.insert(defOp);
    };
    schedule.iterateDefs(loop, sinkPartition, callback);
    for (unsigned i = 0; i < critPath.size(); ++i) {
      Operation *op = critPath[i];
      iterateDefs(loop, op, [&](OpResult def) {
        Operation *defOp = def.getDefiningOp();
        if (opsInCluster.contains(defOp))
          critPath.insert(defOp);
      });
    }

    // If all ops are on the critical path, assign them to the def partition.
    if (critPath.size() == cluster.ops.size()) {
      for (Operation *op : cluster.ops)
        schedule.insert(defPartition, op);
      continue;
    }

    // Some ops are on the critical path, and there is also a backedge.
    // Rematerialize the critical path ops into the sink partition. Leave the
    // rest in the def partition and rely on DCE to remove them.
    critPath = topologicalSort(critPath);
    DenseSet<Operation *> sinkOps(sinkPartition->getOps().begin(),
                                  sinkPartition->getOps().end());
    for (Operation *op : llvm::reverse(critPath)) {
      OpBuilder b(op);
      Operation *clone = b.clone(*op);
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &use) {
        return sinkOps.contains(use.getOwner());
      });
      sinkOps.insert(clone);
      schedule.insert(sinkPartition, clone);
    }
    for (Operation *op : cluster.ops)
      schedule.insert(defPartition, op);
  }
}

// Rematerialize chains of broadcasts where the user is in a different partition
// than the broadcast to reduce the amount of data that needs to be transferred.
void rematerializeBroadcasts(WarpSchedule &schedule, OpOperand *use) {
  static_assert(
      std::is_base_of_v<OpTrait::OneResult<BroadcastOp>, BroadcastOp> &&
      std::is_base_of_v<OpTrait::OneResult<ExpandDimsOp>, ExpandDimsOp> &&
      std::is_base_of_v<OpTrait::OneResult<ConvertLayoutOp>, ConvertLayoutOp>);

  Operation *defOp = use->get().getDefiningOp();
  while (isa_and_nonnull<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(defOp)) {
    Operation *clone = OpBuilder(defOp).clone(*defOp);
    Partition *userPartition = schedule.getPartition(use->getOwner());
    assert(userPartition && "user not scheduled");
    schedule.insert(userPartition, clone);
    use->set(clone->getResult(0));

    defOp = clone->getOperand(0).getDefiningOp();
    use = &clone->getOpOperand(0);
  }
}

/// Walk over \p loop and clone Broadcast/ExpandDims/ConvertLayout ops into each
/// partition that they have users in. This reduces the amount of data that
/// needs to be transferred through memory.
void optimizeSchedule(scf::ForOp loop, WarpSchedule &schedule) {
  // Walk everything in reverse so that operations are visited before their
  // operands.
  loop.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (!isa<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(op))
      return;

    Partition *partition = schedule.getPartition(op);
    if (!partition)
      return;

    // Record all the other partitions in which we have users.
    llvm::SmallDenseSet<Partition *, 2> userPartitions;
    for (OpOperand &use : op->getUses()) {
      Partition *userPartition = schedule.getPartition(use.getOwner());
      if (!userPartition || userPartition == partition)
        continue;
      userPartitions.insert(userPartition);
    }

    for (auto *userPartition : userPartitions) {
      // Clone the instruction into each user partition.
      Operation *clone = OpBuilder(op).clone(*op);
      schedule.insert(userPartition, clone);
      // Replace all users in that partition with the clone.
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &otherUse) {
        return schedule.getPartition(otherUse.getOwner()) == userPartition;
      });
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionScheduling
    : public triton::gpu::impl::TritonGPUPartitionSchedulingBase<
          PartitionScheduling> {
  using TritonGPUPartitionSchedulingBase::TritonGPUPartitionSchedulingBase;

  void runOnOperation() override;
};
} // namespace

void PartitionScheduling::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (auto [idx, loop] : llvm::enumerate(loops)) {
    if (std::optional<WarpSchedule> schedule = getInitialSchedule(loop)) {
      propagatePartitions(loop, *schedule);

      // Schedule post-loop operations into the epilogue partition after
      // propagatePartitions completes. The epilogue partition is the last
      // partition created (see getInitialSchedule).
      if (schedule->getNumPartitions() > 0) {
        Partition *epiloguePartition =
            schedule->getPartition(schedule->getNumPartitions() - 1);
        schedulePostLoopOps(loop, *schedule, epiloguePartition);
      }

      optimizeSchedule(loop, *schedule);
      schedule->serialize(loop);
      loop->setAttr(
          kWarpSpecializeTagAttrName,
          IntegerAttr::get(IntegerType::get(loop.getContext(), 32), idx));
      // Clean Broadcast/ExpandDims/ConvertLayout that were left with no users
      // after optimizeSchedule. We wait until after the schedule is serialized
      // to avoid invalidating pointers stored in the schedule.
      loop.walk<WalkOrder::PostOrder, ReverseIterator>([](Operation *op) {
        // By default, the walk is in postorder so it is safe to delete ops
        // while we walk.
        if (isa<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(op))
          if (op->use_empty())
            op->erase();
      });
    }
  }
}
