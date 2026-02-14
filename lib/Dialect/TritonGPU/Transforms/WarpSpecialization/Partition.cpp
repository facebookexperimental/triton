#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <sstream>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// DOT File Dumping for Partition Visualization
//===----------------------------------------------------------------------===//

/// Check if DOT file dumping is enabled via environment variable.
static bool shouldDumpDotFile() {
  if (const char *env = std::getenv("TRITON_DUMP_PARTITION_DOT")) {
    return std::string(env) != "0";
  }
  return false;
}

/// Get operation_id attribute value, or -1 if not present.
static int getOperationId(Operation *op) {
  if (!op)
    return -1;
  if (auto opIdAttr = op->getAttrOfType<IntegerAttr>("operation_id")) {
    return opIdAttr.getInt();
  }
  return -1;
}

/// Get named location string from an operation, or empty string if not present.
/// Supports NameLoc, FusedLoc, FileLineColLoc, and CallSiteLoc.
static std::string getNamedLoc(Operation *op) {
  if (!op)
    return "";
  Location loc = op->getLoc();

  // Try to get NameLoc (e.g., loc("myName"))
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    return nameLoc.getName().str();
  }
  // Try FusedLoc which may contain a NameLoc or FileLineColLoc
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto nameLoc = dyn_cast<NameLoc>(subLoc)) {
        return nameLoc.getName().str();
      }
    }
    // If no NameLoc found, try to get FileLineColLoc
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(subLoc)) {
        return "L_" + std::to_string(fileLoc.getLine());
      }
    }
  }
  // Try FileLineColLoc directly (e.g., "file.py":42:0)
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    return "L_" + std::to_string(fileLoc.getLine());
  }
  // Try CallSiteLoc - extract location from callee
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    // Get the callee location (where the function is defined)
    Location calleeLoc = callSiteLoc.getCallee();
    if (auto fileLoc = dyn_cast<FileLineColLoc>(calleeLoc)) {
      return "L_" + std::to_string(fileLoc.getLine());
    }
    if (auto nameLoc = dyn_cast<NameLoc>(calleeLoc)) {
      return nameLoc.getName().str();
    }
  }
  return "";
}

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
  return label;
}

/// Operation category for visualization (semantic role in scheduling).
enum class OpCategory {
  MMA,     // tc_gen5_mma operations (core compute)
  Load,    // TMA loads (descriptor_load, async_tma_copy)
  Store,   // TMA stores (descriptor_store, async_tma_copy_local_to_global)
  Alloc,   // Buffer allocations (local_alloc, tmem_alloc)
  MemDesc, // Memory descriptor views (memdesc_trans, memdesc_subview)
  DataPartition, // Ops exclusive to one MMA's data slice
  Correction,    // Cross-iteration MMA users (e.g., online softmax correction)
  Other          // Other key operations
};

/// Get the category of an operation.
static OpCategory getOpCategory(Operation *op) {
  StringRef opName = op->getName().getStringRef();

  if (opName.contains("tc_gen5_mma"))
    return OpCategory::MMA;
  if (opName.contains("descriptor_load") ||
      opName.contains("descriptor_gather") || opName.contains("local_load") ||
      opName.contains("tmem_load"))
    return OpCategory::Load;
  if (opName.contains("descriptor_store") ||
      opName.contains("descriptor_reduce") || opName.contains("local_store") ||
      opName.contains("tmem_store"))
    return OpCategory::Store;
  if (opName.contains("local_alloc") || opName.contains("tmem_alloc"))
    return OpCategory::Alloc;
  if (opName.contains("memdesc_trans") || opName.contains("memdesc_subview"))
    return OpCategory::MemDesc;
  return OpCategory::Other;
}

/// Get color for an operation based on its category.
static StringRef getOpColor(OpCategory cat) {
  switch (cat) {
  case OpCategory::MMA:
    return "lightgreen";
  case OpCategory::Load:
    return "lightblue";
  case OpCategory::Store:
    return "lightyellow";
  case OpCategory::Alloc:
    return "lightgray";
  case OpCategory::MemDesc:
    return "orange";
  case OpCategory::DataPartition:
    return "lightcyan";
  case OpCategory::Correction:
    return "lightpink";
  case OpCategory::Other:
    return "white";
  }
  return "white";
}

/// Get category name for display.
static StringRef getCategoryName(OpCategory cat) {
  switch (cat) {
  case OpCategory::MMA:
    return "MMA";
  case OpCategory::Load:
    return "Load";
  case OpCategory::Store:
    return "Store";
  case OpCategory::Alloc:
    return "Alloc";
  case OpCategory::MemDesc:
    return "MemDesc";
  case OpCategory::DataPartition:
    return "DataPartition";
  case OpCategory::Correction:
    return "Correction";
  case OpCategory::Other:
    return "Other";
  }
  return "Other";
}

/// Get color for a semantic category cluster.
static StringRef getSemanticCategoryColor(OpCategory cat) {
  switch (cat) {
  case OpCategory::DataPartition:
    return "#e6f3ff"; // Light blue background
  case OpCategory::Correction:
    return "#fff3e6"; // Light orange background
  default:
    return "#f0f0f0"; // Default gray
  }
}

/// Semantic category for higher-level operation grouping.
enum class SemanticCategory {
  DataPartition, // Ops in the main data processing path for one MMA slice
  Correction,    // Cross-iteration correction ops (e.g., softmax normalization)
  Epilogue,      // Post-loop epilogue operations (final stores)
  Prologue       // Pre-computation setup operations
};

/// Get semantic category name for display.
static StringRef getSemanticCategoryName(SemanticCategory cat) {
  switch (cat) {
  case SemanticCategory::DataPartition:
    return "DataPartition";
  case SemanticCategory::Correction:
    return "Correction";
  case SemanticCategory::Epilogue:
    return "Epilogue";
  case SemanticCategory::Prologue:
    return "Prologue";
  }
  return "Unknown";
}

/// Get color for semantic category cluster.
static StringRef getSemanticClusterColor(SemanticCategory cat) {
  switch (cat) {
  case SemanticCategory::DataPartition:
    return "#e6f3ff"; // Light blue
  case SemanticCategory::Correction:
    return "#fff0f0"; // Light red/pink
  case SemanticCategory::Epilogue:
    return "#f0fff0"; // Light green
  case SemanticCategory::Prologue:
    return "#fff3e6"; // Light orange
  }
  return "#f0f0f0";
}

/// Check if an operation is likely a correction operation.
/// Correction ops are users of MMA results that are yielded across iterations.
/// This includes operations like softmax normalization, running max/sum
/// updates.
static bool isCorrectionOp(Operation *op, scf::ForOp loop,
                           ArrayRef<Operation *> mmaOps) {
  // Check if this op uses any iter arg that comes from an MMA result
  for (Operation *mma : mmaOps) {
    for (OpOperand &use : mma->getUses()) {
      // Check if MMA result is yielded
      if (use.getOwner() != loop.getBody()->getTerminator())
        continue;

      // Get the corresponding iter arg for next iteration
      unsigned yieldIdx = use.getOperandNumber();
      if (yieldIdx >= loop.getNumRegionIterArgs())
        continue;

      BlockArgument iterArg = loop.getRegionIterArg(yieldIdx);

      // Check if our op uses this iter arg (directly or indirectly)
      for (OpOperand &iterUse : iterArg.getUses()) {
        if (iterUse.getOwner() == op)
          return true;
      }
    }
  }

  // Also check for typical correction op names as fallback
  StringRef opName = op->getName().getStringRef();
  if (opName.contains("maxnumf") || opName.contains("maximumf") ||
      opName.contains("mulf") || opName.contains("subf") ||
      opName.contains("addf"))
    return false; // These are common ops, need more context

  return false;
}

/// Check if an operation is in the epilogue (stores outside main loop).
static bool isEpilogueOp(Operation *op, scf::ForOp loop) {
  // Check if the operation is a store-type op outside the loop
  if (!loop->isAncestor(op)) {
    StringRef opName = op->getName().getStringRef();
    return opName.contains("store");
  }
  return false;
}

/// Check if an operation is a "key" operation for visualization.
/// Key ops are: loads, stores, tc_gen5_mma, and computation ops on tensors.
static bool isKeyOperation(Operation *op) {
  StringRef opName = op->getName().getStringRef();

  // TC Gen5 MMA operations (core compute)
  if (opName.contains("tc_gen5_mma"))
    return true;

  // Descriptor loads (TMA loads on tensors)
  if (opName.contains("descriptor_load") ||
      opName.contains("descriptor_gather"))
    return true;

  // Descriptor stores (TMA stores on tensors)
  if (opName.contains("descriptor_store") ||
      opName.contains("descriptor_reduce"))
    return true;

  // TMEM operations (tensor memory)
  if (opName.contains("tmem_store") || opName.contains("tmem_load") ||
      opName.contains("tmem_alloc"))
    return true;

  // Local memory operations on tensors
  if (opName.contains("local_alloc") || opName.contains("local_load") ||
      opName.contains("local_store"))
    return true;

  // Memory descriptor operations
  if (opName.contains("memdesc_trans") || opName.contains("memdesc_subview"))
    return true;

  // Computation operations on tensors (arith dialect ops that produce tensor
  // results) These are key for understanding data flow in flash attention
  if (opName.contains("arith.") || opName.contains("math.")) {
    // Check if any result is a tensor type
    for (Value result : op->getResults()) {
      if (isa<RankedTensorType>(result.getType()))
        return true;
    }
  }

  // Triton-specific computation ops on tensors
  if (opName.contains("tt.") && !opName.contains("tt.ptr") &&
      !opName.contains("tt.splat") && !opName.contains("tt.broadcast") &&
      !opName.contains("tt.make_range") && !opName.contains("tt.addptr")) {
    for (Value result : op->getResults()) {
      if (isa<RankedTensorType>(result.getType()))
        return true;
    }
  }

  return false;
}

/// Dump a WarpSchedule to a DOT file for visualization.
/// Shows both partitions (as clusters) and semantic categories (as nested
/// clusters).
static void dumpScheduleToDot(const WarpSchedule &schedule, scf::ForOp loop,
                              StringRef filename) {
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

  // First pass: assign operation IDs in program order across all operations
  // Walk the entire function (or parent of loop) to assign consistent IDs
  // This matches how MemoryPlanner assigns operation_id
  DenseMap<Operation *, unsigned> opToId;
  unsigned opId = 0;

  // Walk the parent function/region to get all operations in order
  if (Operation *parent = loop->getParentOp()) {
    parent->walk([&](Operation *op) {
      if (isKeyOperation(op)) {
        opToId[op] = opId++;
      }
    });
  } else {
    // Fallback: just walk the loop body
    loop.getBody()->walk([&](Operation *op) {
      if (isKeyOperation(op)) {
        opToId[op] = opId++;
      }
    });
  }

  // Collect MMA operations for determining semantic categories
  SmallVector<Operation *> mmaOps;
  loop.getBody()->walk([&](Operation *op) {
    if (op->getName().getStringRef().contains("tc_gen5_mma")) {
      mmaOps.push_back(op);
    }
  });

  // Map each operation to its associated MMA (if any) for DataPartition
  // detection
  DenseMap<Operation *, Operation *> opToMMA;
  DenseMap<Operation *, SmallPtrSet<Operation *, 4>> mmaBackwardSlice;

  // For each MMA, collect operations in its backward slice
  for (Operation *mma : mmaOps) {
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    (void)getBackwardSlice(mma, &slice, options);
    for (Operation *op : slice) {
      mmaBackwardSlice[op].insert(mma);
    }
  }

  /// Determine semantic category for each operation
  auto getSemanticCategory = [&](Operation *op) -> SemanticCategory {
    // Check if it's a correction op (cross-iteration, like softmax)
    if (isCorrectionOp(op, loop, mmaOps))
      return SemanticCategory::Correction;

    // Check if it's exclusive to one MMA's backward slice (DataPartition)
    auto it = mmaBackwardSlice.find(op);
    if (it != mmaBackwardSlice.end() && it->second.size() == 1)
      return SemanticCategory::DataPartition;

    // Default to DataPartition for key ops
    return SemanticCategory::DataPartition;
  };

  // Map from MMA to its data partition ID
  DenseMap<Operation *, unsigned> mmaToDataPartitionId;
  unsigned dpId = 0;
  for (Operation *mma : mmaOps) {
    mmaToDataPartitionId[mma] = dpId++;
  }

  // Get data partition ID for an operation (returns -1 if not in a data
  // partition)
  auto getDataPartitionId = [&](Operation *op) -> int {
    auto it = mmaBackwardSlice.find(op);
    if (it == mmaBackwardSlice.end() || it->second.size() != 1)
      return -1;
    Operation *mma = *it->second.begin();
    auto mmaIt = mmaToDataPartitionId.find(mma);
    return mmaIt != mmaToDataPartitionId.end() ? static_cast<int>(mmaIt->second)
                                               : -1;
  };

  // Map from operation to unique node ID for DOT
  DenseMap<Operation *, std::string> opToNodeId;
  unsigned nodeCounter = 0;
  unsigned categoryClusterCounter = 0;

  // Create clusters for each partition (skip empty ones)
  for (const Partition &partition : schedule.getPartitions()) {
    ArrayRef<Operation *> ops = partition.getOps();

    // Collect key operations and group by semantic category
    DenseMap<SemanticCategory, SmallVector<Operation *>> categoryOps;
    for (Operation *op : ops) {
      if (isKeyOperation(op)) {
        SemanticCategory semCat = getSemanticCategory(op);
        categoryOps[semCat].push_back(op);
      }
    }

    // Skip empty partitions (no key ops)
    bool hasKeyOps = false;
    for (auto &[cat, catOps] : categoryOps) {
      if (!catOps.empty()) {
        hasKeyOps = true;
        break;
      }
    }
    if (!hasKeyOps)
      continue;

    file << "  subgraph cluster_" << partition.getIndex() << " {\n";
    file << "    label=\"Partition " << partition.getIndex() << " (stage "
         << partition.getStage() << ")\";\n";
    file << "    style=filled;\n";
    file << "    color=darkgrey;\n";
    file << "    fillcolor=\"#e8e8e8\";\n\n";

    // Create nested clusters for each semantic category
    for (auto &[semCat, catOps] : categoryOps) {
      if (catOps.empty())
        continue;

      file << "    subgraph cluster_cat_" << categoryClusterCounter++ << " {\n";
      file << "      label=\"" << getSemanticCategoryName(semCat) << "\";\n";
      file << "      style=filled;\n";
      file << "      color=gray;\n";
      file << "      fillcolor=\"" << getSemanticClusterColor(semCat)
           << "\";\n\n";

      // Create nodes for key operations in this category
      std::string prevNodeId;
      for (Operation *op : catOps) {
        std::string nodeId = "op_" + std::to_string(nodeCounter++);
        opToNodeId[op] = nodeId;

        unsigned depth = getLoopDepth(op);
        std::string label = getOpLabel(op);
        OpCategory cat = getOpCategory(op);
        StringRef color = getOpColor(cat);
        StringRef catName = getCategoryName(cat);

        // Use operation_id attribute if available, otherwise use our assigned
        // ID
        int opIdAttr = getOperationId(op);
        unsigned id =
            opIdAttr >= 0 ? static_cast<unsigned>(opIdAttr) : opToId.lookup(op);

        // Get NamedLoc for more readable names
        std::string locName = getNamedLoc(op);

        // Get data partition ID for operations exclusive to one MMA's slice
        int dpIdForOp = getDataPartitionId(op);

        // Format: "[id] locName: opname\n[category] (depth=N)"
        // or "[id] opname\n[category] (depth=N)" if no NamedLoc
        // For operations in a data partition, add "(dp=N)" to show data
        // partition ID
        file << "      " << nodeId << " [label=\"[" << id << "] ";
        if (!locName.empty()) {
          file << locName << ":\\n";
        }
        file << label << "\\n[" << catName;
        if (dpIdForOp >= 0) {
          file << " dp=" << dpIdForOp;
        }
        file << "] (depth=" << depth << ")\", fillcolor=" << color << "];\n";
      }

      file << "    }\n\n";
    }

    file << "  }\n\n";
  }

  file << "}\n";

  llvm::errs() << "Dumped partition graph to " << filename << "\n";
}

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

Partition *WarpSchedule::addPartition(unsigned stage) {
  partitions.push_back(std::make_unique<Partition>(partitions.size(), stage));
  return partitions.back().get();
}

Partition *WarpSchedule::getPartition(Operation *op) {
  return opToPartition.lookup(op);
}
const Partition *WarpSchedule::getPartition(Operation *op) const {
  return opToPartition.lookup(op);
}

Partition *WarpSchedule::getPartition(unsigned idx) {
  return partitions[idx].get();
}
const Partition *WarpSchedule::getPartition(unsigned idx) const {
  return partitions[idx].get();
}

void WarpSchedule::insert(Partition *partition, Operation *op) {
  partition->ops.push_back(op);
  opToPartition[op] = partition;
}

bool WarpSchedule::isScheduled(Operation *op) const {
  const Partition *partition = getPartition(op);
  return partition && partition != getRootPartition();
}

bool WarpSchedule::trySchedule(Partition *partition, Operation *op) {
  if (isScheduled(op))
    return false;
  insert(partition, op);
  return true;
}

FailureOr<WarpSchedule> WarpSchedule::deserialize(scf::ForOp loop) {
  auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
  if (!stages)
    return failure();

  auto tag = loop->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
  if (!tag)
    return failure();

  WarpSchedule result;
  result.tag = tag.getInt();
  for (auto [idx, attr] : llvm::enumerate(stages)) {
    auto stage = dyn_cast<IntegerAttr>(attr);
    if (!stage || stage.getInt() < 0) {
      return mlir::emitError(loop.getLoc(), "partition stages attribute '")
             << kPartitionStagesAttrName << "' has invalid element " << attr;
    }

    result.partitions.push_back(
        std::make_unique<Partition>(idx, stage.getInt()));
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    Partition *partition = result.getRootPartition();
    if (auto attr = op.getAttrOfType<IntegerAttr>(kPartitionAttrName)) {
      int64_t idx = attr.getInt();
      if (idx < 0 || idx >= result.partitions.size())
        return mlir::emitError(op.getLoc(), "invalid partition index ") << idx;
      partition = result.partitions[idx].get();
    }
    result.insert(partition, &op);
  }

  // Dump partitions to DOT file if enabled
  if (shouldDumpDotFile()) {
    dumpScheduleToDot(result, loop, "/tmp/partitions.dot");
  }

  return result;
}

void WarpSchedule::serialize(scf::ForOp loop) const {
  SmallVector<Attribute> stages;
  Builder b(loop.getContext());

  // Dump partitions to DOT file if enabled
  if (shouldDumpDotFile()) {
    dumpScheduleToDot(*this, loop, "/tmp/partitions.dot");
  }

  // Serialize partition attributes for operations inside the loop.
  loop.walk([&](Operation *op) {
    if (Partition *partition = opToPartition.lookup(op)) {
      if (partition == getRootPartition())
        return;
      op->setAttr(kPartitionAttrName,
                  b.getI32IntegerAttr(partition->getIndex()));
    }
  });

  // Serialize partition attributes for post-loop operations -
  // operations scheduled in partitions but outside the loop body.
  // Walk the parent op to find post-loop operations that are still in the IR
  // fix: some operations in opToPartition are erased by inserAref pass in NVWS.
  if (Operation *parent = loop->getParentOp()) {
    parent->walk([&](Operation *op) {
      // Skip operations inside the loop (already handled above)
      if (loop->isAncestor(op))
        return;
      Partition *partition = opToPartition.lookup(op);
      if (!partition || partition == getRootPartition())
        return;
      op->setAttr(kPartitionAttrName,
                  b.getI32IntegerAttr(partition->getIndex()));
    });
  }

  for (Partition &partition : getPartitions())
    stages.push_back(b.getI32IntegerAttr(partition.getStage()));
  loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
}

LogicalResult WarpSchedule::verify(scf::ForOp loop) const {
  // The root partition is only allowed to transitively depend on itself.
  bool failed = false;
  iterateInputs(loop, getRootPartition(), [&](OpOperand &input) {
    auto [def, distance] = getDefiningOpAndDistance(loop, input.get());
    // Ignore values defined outside the loop.
    if (!def || def->getParentOp() != loop)
      return;
    const Partition *defPartition = opToPartition.at(def);
    if (defPartition == getRootPartition())
      return;
    InFlightDiagnostic diag = mlir::emitWarning(input.getOwner()->getLoc());
    diag << "operation in the root partition depends on a value that "
            "originates from a non-root partition through operand #"
         << input.getOperandNumber();
    diag.attachNote(def->getLoc())
        << "operand defined here in partition #" << defPartition->getIndex()
        << " at distance " << distance;
    failed = true;
  });
  if (failed)
    return failure();

  return success();
}

void WarpSchedule::eraseFrom(scf::ForOp loop) {
  loop.walk([&](Operation *op) { op->removeAttr(kPartitionAttrName); });
  loop->removeAttr(kPartitionStagesAttrName);
}

void WarpSchedule::iterateInputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpOperand &)> callback) const {
  for (Operation *op : partition->getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      if (value.getParentBlock() != loop.getBody())
        return;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        assert(arg.getOwner() == loop.getBody());
        // Ignore the induction variable.
        if (arg == loop.getInductionVar())
          return;
        // This value originates from a previous iteration.
        assert(llvm::is_contained(loop.getRegionIterArgs(), arg));
        callback(operand);
      } else if (getPartition(value.getDefiningOp()) != partition) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void WarpSchedule::iterateOutputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  for (Operation *op : partition->getOps()) {
    for (OpOperand &use : op->getUses()) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());

      // Handle post-loop operations.
      if (!owner) {
        // The user is outside the loop, so it's a post-loop operation.
        // Use the operation directly.
        owner = use.getOwner();
        if (getPartition(owner) != partition) {
          callback(owner, use);
        }
        continue;
      }

      if (isa<scf::YieldOp>(owner)) {
        // This value is used in a subsequent iteration.
        callback(owner, use);
      } else if (getPartition(owner) != partition) {
        // This value is used in a different partition in the same iteration.
        callback(owner, use);
      }
    }
  }
}

void WarpSchedule::iterateDefs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, partition, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void WarpSchedule::iterateUses(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, partition, [&](Operation *owner, OpOperand &use) {
    uses.emplace_back(cast<OpResult>(use.get()), &use, 0);
  });
  while (!uses.empty()) {
    auto [output, use, distance] = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());

    // Handle post-loop operations.
    if (!owner) {
      // The user is outside the loop, so it's a post-loop operation.
      callback(output, *use, distance);
      continue;
    }

    if (!isa<scf::YieldOp>(owner)) {
      callback(output, *use, distance);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(output, &use, distance + 1);
  }
}

void WarpSchedule::dump() const {
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    llvm::errs() << "=== PARTITION #" << i << " ===\n";
    for (Operation *op : partition.getOps()) {
      op->print(llvm::errs(), OpPrintingFlags().skipRegions());
      llvm::errs() << "\n";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "=== ROOT PARTITION ===\n";
  for (Operation *op : getRootPartition()->getOps()) {
    op->print(llvm::errs(), OpPrintingFlags().skipRegions());
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}
