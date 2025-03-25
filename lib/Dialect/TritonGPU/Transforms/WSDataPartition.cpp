#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define DEBUG_TYPE "tritongpu-warp-spec-data-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static bool oneVecCoversTheOther(SmallVector<AsyncTaskId> &one,
                                 SmallVector<AsyncTaskId> &other) {
  // Every element of other appears in one.
  for (AsyncTaskId t : other) {
    // If t doesn't appear in one, return false.
    bool found = false;
    for (AsyncTaskId t2 : one) {
      if (t2 == t) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

// Make sure the def chain contains the right taskId.
void fixTaskId(triton::FuncOp &funcOp) {
  bool changed = false;
  do {
    changed = false;
    funcOp.walk([&](Operation *op) {
      auto asyncTaskIds = getAsyncTaskIds(op);
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // Do not update loads.
        if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(defOp))
          continue;
        auto defTaskIds = getAsyncTaskIds(defOp);
        // Make sure defTaskIds cover asyncTaskIds. Call addAsyncTaskIds if
        // necessary.
        if (!oneVecCoversTheOther(defTaskIds, asyncTaskIds)) {
          // Skip control flow ops.
          if (isa<scf::YieldOp, scf::ForOp, scf::IfOp>(op))
            continue;
          // Const ops with same value but different task ids can be folded.
          if (defOp->getDialect()->getNamespace() == "arith") {
            LLVM_DEBUG({
              LDBG("backward fixing taskId for");
              defOp->dump();
            });
            addAsyncTaskIds(defOp, asyncTaskIds);
            changed = true;
            LLVM_DEBUG({
              LDBG("resulting");
              defOp->dump();
            });
          }
        }
        if (operand.hasOneUse() &&
            !oneVecCoversTheOther(asyncTaskIds, defTaskIds)) {
          // YieldOp may lose task attribute during MLIR canonicalization.
          if (isa<scf::YieldOp, scf::IfOp>(op)) {
            LLVM_DEBUG({
              LDBG("forward fixing taskId for");
              defOp->dump();
            });
            addAsyncTaskIds(op, defTaskIds);
            changed = true;
            LLVM_DEBUG({
              LDBG("resulting");
              defOp->dump();
            });
          }
        }
      }
    });
  } while (changed);
}

struct DataPartitionScheme {
  unsigned numPartitions = 0;
  // ops to be partitioned.
  SetVector<Operation *> ops;
  // Which dimension to partition. For dot, dim 0 means along M dimension, 1
  // means along N dimension.
  DenseMap<Operation *, unsigned> opPartitionDims;
  // For dot, which operand to partition along opPartitionDims.
  DenseMap<Operation *, unsigned> dotPartitionOperand;
  // Ops that are rematerialized through both dimensions.
  DenseMap<Operation *, SetVector<unsigned>> rematerializedOps;
  // Ops should not be partitioned due to rematerialization.
  DenseSet<Operation *> opsToSkip;

  // op with noOpPartitionDim will be duplicated instead of partitioned.
  // Use -2 to avoid conflict with Empty/Tombstone value.
  static const unsigned noOpPartitionDim = ~0U - 2;

  void append(DataPartitionScheme &other) {
    for (auto op : other.ops)
      ops.insert(op);
    for (auto op : other.opPartitionDims)
      opPartitionDims.insert(op);
    for (auto op : other.dotPartitionOperand)
      dotPartitionOperand.insert(op);
    for (auto &op : other.rematerializedOps)
      rematerializedOps.insert(op);
    for (auto op : other.opsToSkip)
      opsToSkip.insert(op);
  }

  bool partitionIsCompatible() { return true; }

  bool isValidPartitionDim(unsigned dim) const {
    return dim < numPartitions || dim == DataPartitionScheme::noOpPartitionDim;
  }

  unsigned flipPartitionDim(unsigned dim) const {
    if (dim == DataPartitionScheme::noOpPartitionDim)
      return dim;
    return numPartitions - 1 - dim;
  }

  bool isPartitioned(Operation *op) const {
    return opPartitionDims.contains(op) || rematerializedOps.contains(op);
  }

  bool isSkipped(Operation *op) const { return opsToSkip.contains(op); }

  void undoPartition(Operation *op) {
    if (opPartitionDims.contains(op)) {
      opPartitionDims.erase(op);
      ops.remove(op);
      opsToSkip.insert(op);
    }
  }

  void dump() const {
    LDBG("=================== DataPartitionScheme ====================");
    LDBG(" numPartitions " << numPartitions);
    LDBG(" ops to partition:");
    for (auto &op : ops) {
      std::string operand;
      if (dotPartitionOperand.contains(op)) {
        operand = "operand " + std::to_string(dotPartitionOperand.at(op));
      }
      assert(opPartitionDims.contains(op) && "missing partition dim");
      LDBG(" dim " << opPartitionDims.at(op) << " " << operand);
      op->dump();
    }
    LDBG("\n");
    if (!rematerializedOps.empty()) {
      LDBG(" ops to rematerialize\n");
      for (auto &op : rematerializedOps) {
        op.first->dump();
        LDBG(" along dim ");
        for (auto &dim : op.second) {
          LDBG(dim << " ");
        }
      }
      LDBG("\n");
    }

    if (!opsToSkip.empty()) {
      LDBG(" ops to skip\n");
      for (auto &op : opsToSkip)
        op->dump();
      LDBG("\n");
    }

    LDBG("===========================================================");
  };
};

static SmallVector<int64_t> getShape(Type type) {
  if (auto descType = dyn_cast<MemDescType>(type))
    return {descType.getShape().begin(), descType.getShape().end()};
  else if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return {tensorType.getShape().begin(), tensorType.getShape().end()};
  else if (auto ptrType = dyn_cast<PointerType>(type))
    return getShape(ptrType.getPointeeType());
  return {};
}

static SmallVector<int64_t> getShape(Value v) { return getShape(v.getType()); }

static bool needToSlice(Value v, unsigned dim, int size) {
  if (dim == DataPartitionScheme::noOpPartitionDim)
    return true;
  auto shape = getShape(v);
  return shape.size() > dim && shape[dim] > size;
}

static bool isControlFlowOp(Operation *op) {
  return isa<ReturnOp, FuncOp, scf::YieldOp, scf::ForOp, scf::IfOp>(op);
}

// Duplicate the op for different partition dims.
static bool rematerializeOp(Operation *op, DataPartitionScheme &partitionScheme,
                            unsigned currentDim) {
  // Bail out if op is already rematerialized.
  if (partitionScheme.rematerializedOps.contains(op)) {
    partitionScheme.rematerializedOps[op].insert(currentDim);
    return true;
  }

  if (isa<LocalAllocOp, arith::ConstantOp>(op)) {
    // assert op has a conflicting partition dim.
    auto existingDim = partitionScheme.opPartitionDims[op];
    assert(existingDim != currentDim && "op has no conflicting partition dim");
    partitionScheme.rematerializedOps[op].insert(existingDim);
    partitionScheme.rematerializedOps[op].insert(currentDim);
    // Undo the partition of the dependency ops in the backward slice.
    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    getBackwardSlice(op, &slice);
    for (auto depOp : slice)
      partitionScheme.undoPartition(depOp);
    return true;
  }
  return false;
}

bool getBackwardSliceToPartition(Value v, DataPartitionScheme &partitionScheme,
                                 unsigned currentDim) {
  assert(partitionScheme.isValidPartitionDim(currentDim) && "invalid dim");
  if (!needToSlice(v, currentDim, partitionScheme.numPartitions))
    return true;
  if (auto op = v.getDefiningOp()) {
    // Check dim compatibility
    if (!partitionScheme.ops.insert(op)) {
      if (!isControlFlowOp(op) &&
          partitionScheme.opPartitionDims[op] != currentDim) {
        // Duplicate the op if possible.
        if (!rematerializeOp(op, partitionScheme, currentDim)) {
          LLVM_DEBUG({
            LDBG("incompatible partitioning during backwards:");
            LDBG("dim " << currentDim);
            op->dump();
          });
          return false;
        }
      }
      return true;
    }
    partitionScheme.opPartitionDims[op] = currentDim;

    // Flip dim when op is trans
    if (isa<TransOp>(op))
      currentDim = partitionScheme.flipPartitionDim(currentDim);

    // Recusively process operands backwards.
    if (op->hasTrait<OpTrait::Elementwise>() ||
        isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
            BroadcastOp, ExpandDimsOp, MakeRangeOp, SplatOp, ConvertLayoutOp,
            triton::gpu::LocalAllocOp, LoadOp, TransOp, AtomicRMWOp,
            triton::AddPtrOp, ExperimentalDescriptorLoadOp>(op)) {
      for (Value operand : op->getOperands())
        if (!getBackwardSliceToPartition(operand, partitionScheme, currentDim))
          return false;
    } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
      if (!getBackwardSliceToPartition(currentDim == 0 ? dotOp.getA()
                                                       : dotOp.getB(),
                                       partitionScheme, currentDim))
        return false;
      if (!getBackwardSliceToPartition(dotOp.getC(), partitionScheme,
                                       currentDim))
        return false;
      partitionScheme.dotPartitionOperand[dotOp] = currentDim == 0 ? 0 : 1;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // track yield value
      // find result index of v
      unsigned resultIndex = 0;
      for (int i = 0; i < op->getNumResults(); ++i) {
        if (op->getResult(i) == v) {
          resultIndex = i;
          break;
        }
      }
      partitionScheme.ops.insert(ifOp.thenYield());
      partitionScheme.opPartitionDims[ifOp.thenYield()] = currentDim;
      partitionScheme.ops.insert(ifOp.elseYield());
      partitionScheme.opPartitionDims[ifOp.elseYield()] = currentDim;
      auto thenYieldArg = ifOp.thenYield().getOperand(resultIndex);
      auto elseYieldArg = ifOp.elseYield().getOperand(resultIndex);
      if (getBackwardSliceToPartition(thenYieldArg, partitionScheme,
                                      currentDim))
        return false;
      if (!getBackwardSliceToPartition(elseYieldArg, partitionScheme,
                                       currentDim))
        return false;
    } else {
      llvm_unreachable("Unexpected op");
    }
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
      // track initial value
      auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
      if (!getBackwardSliceToPartition(initArg, partitionScheme, currentDim))
        return false;
      // track yield value
      auto yieldArg = forOp.getYieldedValues()[bbArg.getArgNumber() - 1];
      if (!getBackwardSliceToPartition(yieldArg, partitionScheme, currentDim))
        return false;
    }
  }

  return true;
};

// Return false if the partition is not possible.
bool getForwardSliceToPartition(Value v, DataPartitionScheme &partitionScheme,
                                unsigned currentDim, DenseSet<Value> &seen) {
  assert(partitionScheme.isValidPartitionDim(currentDim) && "invalid dim");
  if (!seen.insert(v).second)
    return true;
  if (!needToSlice(v, currentDim, partitionScheme.numPartitions))
    return true;

  // Recusively process operands forwards.
  for (Operation *depOp : v.getUsers()) {
    // Flip dim when op is trans
    if (isa<TransOp>(depOp))
      currentDim = partitionScheme.flipPartitionDim(currentDim);

    // Check dim compatibility
    if (!partitionScheme.ops.insert(depOp)) {
      if (!isControlFlowOp(depOp) &&
          partitionScheme.opPartitionDims[depOp] != currentDim) {
        LLVM_DEBUG({
          LDBG("incompatible partitioning during forwards:");
          depOp->dump();
        });
        return false;
      }
      // YieldOp can be partitioned multiple times, one for each of its
      // operands.
      if (!isa<scf::YieldOp>(depOp))
        continue;
    }

    partitionScheme.opPartitionDims[depOp] = currentDim;

    auto onlyUsedByAtomicStore = [](Value v) {
      SetVector<Operation *> forwardSlice;
      getForwardSlice(v, &forwardSlice);
      AtomicRMWOp atomicStore;
      for (auto op : forwardSlice) {
        if (atomicStore = dyn_cast<AtomicRMWOp>(op)) {
          break;
        }
      }

      if (!atomicStore)
        return false;

      // Check all ops in fowardSlice are only connected to atomicStore
      SmallVector<Operation *> queue = {atomicStore};
      forwardSlice.remove(atomicStore);
      while (!queue.empty()) {
        auto op = queue.back();
        queue.pop_back();
        for (Value operand : op->getOperands()) {
          if (auto defOp = operand.getDefiningOp()) {
            if (forwardSlice.contains(defOp)) {
              forwardSlice.remove(defOp);
              queue.push_back(defOp);
            }
          }
        }
      }

      return forwardSlice.empty();
    };

    if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(depOp)) {
      if ((currentDim == 0 && v == dotOp.getB()) ||
          (currentDim == 1 && v == dotOp.getA())) {
        // It is fine to continue the partition if the dot output is immediately
        // stored out via an atomic add, as the dot computes a partial result.
        if (onlyUsedByAtomicStore(dotOp.getD())) {
          partitionScheme.dotPartitionOperand[dotOp] =
              v == dotOp.getA() ? 0 : 1;
          // Duplicate the users of the dot output since the shape of the output
          // will not be changed
          currentDim = DataPartitionScheme::noOpPartitionDim;
        } else {
          LLVM_DEBUG({
            auto opnd = (v == dotOp.getA()) ? "A" : "B";
            LDBG("skip partitioning along K of " << opnd << " of dot\n");
            dotOp.dump();
          });
          return false;
        }
      } else {
        partitionScheme.dotPartitionOperand[dotOp] = currentDim == 0 ? 0 : 1;
      }
    }

    for (Value result : depOp->getResults())
      if (!getForwardSliceToPartition(result, partitionScheme, currentDim,
                                      seen))
        return false;

    if (auto yieldOp = dyn_cast<scf::YieldOp>(depOp)) {
      auto parentOp = yieldOp->getParentOp();
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == v) {
          partitionScheme.ops.insert(parentOp);
          if (!getForwardSliceToPartition(
                  parentOp->getResult(operand.getOperandNumber()),
                  partitionScheme, currentDim, seen))
            return false;
          ;
        }
      }
    }
  }

  return true;
}; // namespace gpu

// Compute a closure of all ops originated from
// or being dependent on by the root op.
bool getSliceToPartition(Value root, DataPartitionScheme &partitionScheme,
                         unsigned currentDim) {
  if (!getBackwardSliceToPartition(root, partitionScheme, currentDim))
    return false;
  DataPartitionScheme forwardPartitionScheme = partitionScheme;
  DenseSet<Value> seen;
  bool forwardSuccess = getForwardSliceToPartition(root, forwardPartitionScheme,
                                                   currentDim, seen);
  // Merge the two partition schemes
  partitionScheme.append(forwardPartitionScheme);
  if (!forwardSuccess)
    return false;

  for (auto op : forwardPartitionScheme.ops) {
    // skip ops that have noOpPartitionDim
    currentDim = partitionScheme.opPartitionDims[op];
    if (currentDim == DataPartitionScheme::noOpPartitionDim)
      continue;
    if (op->hasTrait<OpTrait::Elementwise>() ||
        isa<tt::StoreOp, ExperimentalDescriptorStoreOp, AtomicRMWOp>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (!getBackwardSliceToPartition(operand.get(), partitionScheme,
                                         currentDim))
          return false;
      }
    } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
      unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
      if (!getBackwardSliceToPartition(dotOp.getOperand(opndIndx),
                                       partitionScheme, currentDim))
        return false;
      if (currentDim == 0 && opndIndx == 0 ||
          currentDim == 1 && opndIndx == 1) {
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(dotOp.getC(), partitionScheme,
                                         currentDim))
          return false;
      } else {
        // slice the other operand
        unsigned otherOpndIndx = 1 - opndIndx;
        if (!getBackwardSliceToPartition(dotOp.getOperand(otherOpndIndx),
                                         partitionScheme, 1 - currentDim))
          return false;
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(dotOp.getC(), partitionScheme,
                                         DataPartitionScheme::noOpPartitionDim))
          return false;
      }
    }
  }

  return true;
}

bool computePartitionScheme(triton::FuncOp &funcOp,
                            DataPartitionScheme &partitionScheme) {
  // Use dot to drive the partition
  SetVector<nvidia_gpu::WarpGroupDotOp> dots;

  // check all dot ops that have more than one async task id
  funcOp.walk([&](Operation *op) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    if (asyncTaskIds.size() > 1) {
      if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
        dots.insert(dotWaitOp);
      }
    }
  });

  // Checking if all dots can be partitioned in the same way
  int numWarps =
      TritonGPUDialect::getNumWarps(funcOp->getParentOfType<ModuleOp>());
  for (auto dotOp : dots) {
    if (partitionScheme.isPartitioned(dotOp) ||
        partitionScheme.isSkipped(dotOp)) {
      continue;
    }

    // partition along M first, otherwise along N
    RankedTensorType dotType = dotOp.getType();
    LLVM_DEBUG({
      LDBG("Computing partition scheme for");
      dotOp.dump();
    });

    auto shapePerCTA = getShapePerCTA(dotType);
    if (shapePerCTA.size() != 2) {
      LDBG("partition not possible: shapePerCTA " << shapePerCTA.size());
      return false;
    }
    auto CTALayout = getCTALayout(dotType.getEncoding());
    auto asyncTaskIds = getAsyncTaskIds(dotOp);
    int sliceSizeM = shapePerCTA[0] / asyncTaskIds.size();
    int sliceSizeN = shapePerCTA[1] / asyncTaskIds.size();
    SmallVector<unsigned, 2> partitionDim, partitionSize;

    if (sliceSizeM >= 64) {
      partitionDim.push_back(0);
      partitionSize.push_back(sliceSizeM);
    }

    if (sliceSizeN >= 256) {
      partitionDim.push_back(1);
      partitionSize.push_back(sliceSizeN);
    }

    if (partitionDim.empty()) {
      LDBG("Partition not available: " << sliceSizeM << " " << sliceSizeN);
      return false;
    }

    if (partitionScheme.numPartitions == 0) {
      partitionScheme.numPartitions = asyncTaskIds.size();
    } else {
      if (partitionScheme.numPartitions != asyncTaskIds.size()) {
        LDBG("partition not possible, in conflict with previous partition\n");
        return false;
      }
    }

    bool success = false;
    for (int i = 0; i < partitionDim.size(); ++i) {
      // Partition the slice closure
      auto trialPartitionScheme = partitionScheme;
      LLVM_DEBUG(
          { LDBG("Trying partition along " << partitionDim[i] << " \n"); });

      if (getSliceToPartition(dotOp.getD(), trialPartitionScheme,
                              partitionDim[i])) {
        success = true;
        partitionScheme = trialPartitionScheme;
      }

      LLVM_DEBUG({
        LDBG(" Trial slice:\n");
        trialPartitionScheme.dump();
        LDBG("\n");
      });

      if (success)
        break;
    }

    if (!success) {
      LDBG("partition not possible\n");
      return false;
    }
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG(" Final slice:\n");
    partitionScheme.dump();
    LDBG("\n");
  });

  return !partitionScheme.ops.empty();
}

// For each op to be rematerialized, create a new op and replace its user with
// the new op.
void rewriteRematerializedOps(triton::FuncOp &funcOp,
                              DataPartitionScheme &partitionScheme) {
  if (partitionScheme.rematerializedOps.empty())
    return;

  OpBuilderWithAsyncTaskIds builder(funcOp.getContext());

  // For each rematerialized op, create a new op and replace its user with it.
  for (auto opDim : partitionScheme.rematerializedOps) {
    auto oldOp = opDim.first;
    builder.setInsertionPoint(oldOp);
    builder.setAsyncTaskIdsFromOp(oldOp);

    // Skip the first dim which will be using the original op.
    for (unsigned i = 1; i < opDim.second.size(); i++) {
      unsigned dim = opDim.second[i];
      LLVM_DEBUG({
        LDBG("rewriting op along dim " << dim << ":");
        oldOp->dump();
      });

      Operation *newOp = nullptr;
      if (auto allocOp = dyn_cast<LocalAllocOp>(oldOp)) {
        // create a memdesc view
        auto memdescType = allocOp.getType();
        SmallVector<int64_t> shape = getShape(memdescType);
        int sliceSize = shape[dim] / partitionScheme.numPartitions;
        shape[dim] = sliceSize;
        Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
            allocOp.getLoc(), 0, 32);
        auto slicedMemdescType = MemDescType::get(
            shape, memdescType.getElementType(), memdescType.getEncoding(),
            memdescType.getMemorySpace(), memdescType.getMutableMemory());
        SmallVector<Value> offsets(shape.size(), zero);
        auto viewOp = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
            allocOp.getLoc(), slicedMemdescType, allocOp.getResult(), offsets);
        newOp = viewOp;
      } else if (isa<arith::ConstantOp>(oldOp)) {
        newOp = builder.clone(*oldOp);
      } else {
        llvm_unreachable("Unexpected op");
      }

      LLVM_DEBUG({
        LDBG("new op:");
        newOp->dump();
      });

      setAsyncTaskIds(newOp, getAsyncTaskIds(oldOp));
      partitionScheme.ops.insert(newOp);
      partitionScheme.opPartitionDims[newOp] = dim;

      // replace the users that have same partition dim with the op.
      auto dimMatches = [&](OpOperand &operand) {
        auto user = operand.getOwner();
        assert(partitionScheme.opPartitionDims.contains(user) &&
               "user not partitioned");
        unsigned userDim = partitionScheme.opPartitionDims[user];
        if (isa<TransOp>(user)) {
          // flip userDim for trans
          userDim = partitionScheme.flipPartitionDim(userDim);
        } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(user)) {
          // infer userDim for dot
          assert(partitionScheme.dotPartitionOperand.contains(user) &&
                 "no operand info");
          unsigned opndIndx = partitionScheme.dotPartitionOperand[user];
          if (userDim == 0 && opndIndx == 1 || userDim == 1 && opndIndx == 0)
            userDim = DataPartitionScheme::noOpPartitionDim;
        }

        if (userDim != dim)
          return false;
        LLVM_DEBUG({
          LDBG("replacing user with dim " << userDim << ":");
          user->dump();
        });
        return true;
      };

      oldOp->getResult(0).replaceUsesWithIf(newOp->getResult(0), dimMatches);
    }
  }
}

Operation *sliceOp(Value v, int offset, OpBuilderWithAsyncTaskIds &builder,
                   IRMapping &mappings, IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme);

Operation *sliceOp(Operation *op, int offset,
                   OpBuilderWithAsyncTaskIds &builder, IRMapping &mappings,
                   IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme) {
  if (!partitionScheme.ops.contains(op))
    return op;
  if (mappings.contains(op))
    return mappings.lookupOrNull(op);
  if (reverseMappings.contains(op))
    return op;

  unsigned dim = partitionScheme.opPartitionDims[op];
  unsigned numOfPartitions = partitionScheme.numPartitions;

  LLVM_DEBUG({
    LDBG("slicing along dim " << dim << ":");
    op->dump();
  });

  auto asyncTaskIds = getAsyncTaskIds(op);
  SmallVector<mlir::AsyncTaskId, 3> sliceTaskIds;
  if (asyncTaskIds.size() == numOfPartitions) {
    // We are slicing the op for consumer only
    sliceTaskIds.push_back(asyncTaskIds[offset]);
  } else if (asyncTaskIds.size() == 1) {
    // We are slicing the op for producer only
    sliceTaskIds.push_back(asyncTaskIds.front());
  } else if (asyncTaskIds.size() > numOfPartitions) {
    // We are slicing the op for both producer and consumer
    sliceTaskIds.push_back(asyncTaskIds.front());
    sliceTaskIds.push_back(asyncTaskIds[offset + 1]);
  } else {
    llvm_unreachable("Unexpected asyncTaskIds.size()");
  }

  builder.setAsynTaskIdsFromArray(sliceTaskIds);
  auto cloneAndSetResultType = [&](Operation *op) {
    builder.setInsertionPoint(op);
    auto newOp = builder.clone(*op, mappings);
    setAsyncTaskIds(newOp, sliceTaskIds);
    mappings.map(op, newOp);
    reverseMappings.map(newOp, op);
    // set result shape
    if (!op->getResults().empty()) {
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      bool needRetype = true;
      if (dim == DataPartitionScheme::noOpPartitionDim) {
        // Just duplicate the op for noOpPartitionDim
        needRetype = false;
      } else if (isa<nvidia_gpu::WarpGroupDotOp>(op)) {
        assert(partitionScheme.dotPartitionOperand.contains(op) &&
               "no operand info");
        unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
        if (dim == 0 && opndIndx == 1 || dim == 1 && opndIndx == 0) {
          needRetype = false;
        }
      }

      if (needRetype) {
        if (auto type = dyn_cast<MemDescType>(v.getType())) {
          SmallVector<int64_t> shape{type.getShape().begin(),
                                     type.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newType =
              MemDescType::get(shape, type.getElementType(), type.getEncoding(),
                               type.getMemorySpace(), type.getMutableMemory());
          newV.setType(newType);
        } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
          SmallVector<int64_t> shape{type.getShape().begin(),
                                     type.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newType = RankedTensorType::get(shape, type.getElementType(),
                                               type.getEncoding());
          newV.setType(newType);
        }
      }
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
    return newOp;
  };

  // slice operands first
  Operation *newOp;
  if ((dim == DataPartitionScheme::noOpPartitionDim) ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ConvertLayoutOp, BroadcastOp, SplatOp, ExpandDimsOp, FpToFpOp,
          AtomicRMWOp, LocalAllocOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, builder, mappings, reverseMappings,
              partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    builder.setInsertionPoint(op);
    auto valAttr = cast<DenseElementsAttr>(constOp.getValueAttr());
    auto valType = cast<ShapedType>(valAttr.getType());
    SmallVector<int64_t> shape{valType.getShape().begin(),
                               valType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newValType = valType.clone(shape);
    auto newValAttr = valAttr.resizeSplat(newValType);
    newOp = builder.createWithAsyncTaskIds<arith::ConstantOp>(op->getLoc(),
                                                              newValAttr);
    // Do not drop original task id as constant folding may lose one constant.
    setAsyncTaskIds(newOp, getAsyncTaskIds(op));
    auto v = op->getResult(0);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (auto makeRangeOp = dyn_cast<MakeRangeOp>(op)) {
    builder.setInsertionPoint(op);
    int newRangeStart = makeRangeOp.getStart();
    int newRangeEnd = makeRangeOp.getEnd();
    int sliceSize = (newRangeEnd - newRangeStart) / numOfPartitions;
    newRangeStart += offset * sliceSize;
    newRangeEnd = newRangeStart + sliceSize;
    auto v = op->getResult(0);
    auto type = cast<RankedTensorType>(v.getType());
    auto newType = RankedTensorType::get({sliceSize}, builder.getI32Type(),
                                         type.getEncoding());
    newOp = builder.createWithAsyncTaskIds<MakeRangeOp>(
        op->getLoc(), newType, newRangeStart, newRangeEnd);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (isa<StoreOp, LoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, builder, mappings, reverseMappings,
              partitionScheme);
    // TODO: slice store base ptr
    newOp = cloneAndSetResultType(op);
  } else if (isa<ExperimentalDescriptorLoadOp, ExperimentalDescriptorStoreOp>(
                 op)) {
    SmallVector<int64_t> shape;
    Value coordVal;
    if (auto loadOp = dyn_cast<ExperimentalDescriptorLoadOp>(op)) {
      coordVal = loadOp.getIndices()[dim];
      shape = getShape(loadOp.getResult());
    } else if (auto storeOp = dyn_cast<ExperimentalDescriptorStoreOp>(op)) {
      coordVal = storeOp.getIndices()[dim];
      shape = getShape(storeOp.getSrc());
    }
    auto newCoordVal = coordVal;
    if (offset) {
      builder.setInsertionPointAfter(coordVal.getDefiningOp());
      Value offsetVal = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          op->getLoc(), offset * shape[dim] / numOfPartitions, 32);
      newCoordVal = builder.createWithAsyncTaskIds<arith::AddIOp>(
          op->getLoc(), coordVal, offsetVal);
      mappings.map(coordVal, newCoordVal);
      reverseMappings.map(newCoordVal, coordVal);
    }

    newOp = cloneAndSetResultType(op);
    if (isa<ExperimentalDescriptorLoadOp>(op)) {
      // map load result
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
  } else if (auto transOp = dyn_cast<TransOp>(op)) {
    sliceOp(transOp.getSrc(), offset, builder, mappings, reverseMappings,
            partitionScheme);
    builder.setInsertionPoint(op);
    auto v = transOp.getResult();
    auto type = dyn_cast<MemDescType>(v.getType());
    SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newType =
        MemDescType::get(shape, type.getElementType(), type.getEncoding(),
                         type.getMemorySpace(), type.getMutableMemory());
    builder.setInsertionPoint(op);
    newOp = builder.clone(*op, mappings);
    setAsyncTaskIds(newOp, sliceTaskIds);
    auto newV = newOp->getResult(0);
    newV.setType(newType);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
    assert(partitionScheme.dotPartitionOperand.contains(op) &&
           "no operand info");
    unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
    LDBG("slicing operand " << opndIndx << "\n");
    sliceOp(dotOp.getOperand(opndIndx), offset, builder, mappings,
            reverseMappings, partitionScheme);
    if (dim == 0 && opndIndx == 1 || dim == 1 && opndIndx == 0) {
      // slice the other operand
      unsigned otherOpndIndx = 1 - opndIndx;
      LDBG("slicing operand " << otherOpndIndx << "\n");
      sliceOp(dotOp.getOperand(otherOpndIndx), offset, builder, mappings,
              reverseMappings, partitionScheme);
    }
    // Hanlde accumulator
    sliceOp(dotOp.getC(), offset, builder, mappings, reverseMappings,
            partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // Add new loop arguments
    SmallVector<Value> newLoopArgs;
    for (auto initArg : forOp.getInitArgs())
      newLoopArgs.push_back(initArg);
    DenseMap<int, int> newArgIdices;
    for (unsigned i = 0; i < forOp.getInitArgs().size(); i++) {
      auto initArg = forOp.getInitArgs()[i];
      Value newInitArg;
      auto newInitArgOp = sliceOp(initArg, offset, builder, mappings,
                                  reverseMappings, partitionScheme);
      if (auto bbArg = dyn_cast<BlockArgument>(initArg)) {
        // find the corresponding new block argument
        Block *parentBlock = bbArg.getOwner();
        unsigned argIndex = parentBlock->getNumArguments();
        for (unsigned i = 0; i < parentBlock->getNumArguments(); ++i) {
          if (parentBlock->getArgument(i) == bbArg) {
            argIndex = i;
            break;
          }
        }
        assert(argIndex < parentBlock->getNumArguments() &&
               "new init argment not found");
        Region *parentRegion = parentBlock->getParent();
        Region &newParentRegion =
            newInitArgOp->getRegion(parentRegion->getRegionNumber());
        newInitArg = parentRegion->getArgument(argIndex);
      } else {
        newInitArg = mappings.lookupOrNull(initArg);
      }

      if (newInitArg) {
        assert(newInitArg != initArg && "value not sliced");
        newLoopArgs.append({newInitArg});
        forOp.getBody()->insertArgument(forOp.getBody()->getNumArguments(),
                                        newInitArg.getType(), forOp.getLoc());
        newArgIdices[i] = newLoopArgs.size() - 1;
      }
    }

    // Create newForOp and take the region of forOp
    builder.setInsertionPoint(op);
    auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newLoopArgs);
    assert(newForOp.getRegionIterArgs().size() ==
           newForOp.getInitArgs().size());
    newForOp->setAttrs(forOp->getAttrs());
    partitionScheme.ops.insert(newForOp);
    newOp = newForOp;

    // Replace forOp with newForOp
    newForOp.getRegion().takeBody(forOp.getRegion());
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    op->setAttr("to_be_removed", builder.getUnitAttr());

    // Map new loop arguments
    for (auto argIndex : newArgIdices) {
      Value v = newForOp.getResult(argIndex.first);
      Value newV = newForOp.getResult(argIndex.second);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);

      auto regionArg = newForOp.getRegionIterArg(argIndex.first);
      auto newRegionArg = newForOp.getRegionIterArg(argIndex.second);
      mappings.map(regionArg, newRegionArg);
      reverseMappings.map(newRegionArg, regionArg);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Slice the yield op and update if results
    auto thenYieldOp = ifOp.thenYield();
    auto elseYieldOp = ifOp.elseYield();
    auto newThenYieldOp = sliceOp(thenYieldOp, offset, builder, mappings,
                                  reverseMappings, partitionScheme);
    sliceOp(elseYieldOp, offset, builder, mappings, reverseMappings,
            partitionScheme);
    assert(newThenYieldOp->getNumOperands() > ifOp->getNumResults() &&
           "no need to slice if op");
    // Clone ifOp with updated results but re-use the original regions.
    builder.setInsertionPoint(op);
    SmallVector<Type, 4> newResultTypes;
    for (auto thenResult : thenYieldOp.getResults()) {
      newResultTypes.push_back(thenResult.getType());
    }
    auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                             ifOp.getCondition());
    // Move the original regions to the cloned operation.
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    newOp = newIfOp;
    newIfOp->setAttrs(ifOp->getAttrs());
    partitionScheme.ops.insert(newIfOp);
    ifOp->setAttr("to_be_removed", builder.getUnitAttr());

    // Replace ifOp with newIfOp
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      ifOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));

    // Map if results based on the mapping for yield
    for (auto &v : thenYieldOp->getOpOperands()) {
      auto newV = mappings.lookupOrNull(v.get());
      if (newV) {
        int operandIndex = v.getOperandNumber();
        // find the corresponding operand index of newV in newYieldOp
        int newOperandIndex = -1;
        for (int i = 0; i < newThenYieldOp->getNumOperands(); ++i) {
          if (newThenYieldOp->getOperand(i) == newV) {
            newOperandIndex = i;
            break;
          }
        }
        assert(newOperandIndex >= 0 && "newV not found in newYieldOp");
        auto newResult = newIfOp.getResult(operandIndex);
        auto newSlicedResult = newIfOp.getResult(newOperandIndex);
        mappings.map(newResult, newSlicedResult);
        reverseMappings.map(newSlicedResult, newResult);
      }
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    int num = yieldOp.getNumOperands();
    for (int i = 0; i < num; i++) {
      auto operand = yieldOp.getOperand(i);
      sliceOp(operand, offset, builder, mappings, reverseMappings,
              partitionScheme);
      if (auto newV = mappings.lookupOrNull(operand))
        yieldOp->insertOperands(op->getNumOperands(), newV);
    }
    newOp = op;
  } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    assert(reduceOp.getAxis() != dim &&
           "reduce should not happen on the partitioned dimension");
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, builder, mappings, reverseMappings,
              partitionScheme);
    newOp = cloneAndSetResultType(op);
    // recursively set async task ids for child ops
    newOp->walk(
        [&](Operation *childOp) { setAsyncTaskIds(childOp, sliceTaskIds); });
  } else {
    llvm_unreachable("unsupported op type");
  }

  LLVM_DEBUG({
    LDBG("resulting");
    newOp->dump();
  });
  mappings.map(op, newOp);
  reverseMappings.map(newOp, op);
  return newOp;
}

Operation *sliceOp(Value v, int offset, OpBuilderWithAsyncTaskIds &builder,
                   IRMapping &mappings, IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme) {
  if (auto op = v.getDefiningOp()) {
    return sliceOp(op, offset, builder, mappings, reverseMappings,
                   partitionScheme);
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    return sliceOp(bbAargOwner, offset, builder, mappings, reverseMappings,
                   partitionScheme);
  }
}

void partitionTasks(triton::FuncOp &funcOp, int numConsumerGroups) {

  // op -> (partition dim, num of partitions)
  DataPartitionScheme partitionScheme;
  if (!computePartitionScheme(funcOp, partitionScheme)) {
    if (numConsumerGroups > 1)
      llvm::errs() << "computePartitionScheme failed when requested\n";
    return;
  }

  // Rewrite the rematerialized ops.
  LDBG("Rewriting rematerialized Ops");
  rewriteRematerializedOps(funcOp, partitionScheme);
  LLVM_DEBUG({
    LDBG("After rewriting rematerialized Ops:");
    funcOp.dump();
    LDBG("\n");
    LDBG(" Final parition scheme:\n");
    partitionScheme.dump();
  });

  // Slice the ops.
  for (int i = 0; i < partitionScheme.numPartitions; i++) {
    OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
    IRMapping mappings, reverseMappings;

    LDBG("partitioning op for task " << i + 1 << ":\n");

    // TODO: compute a topological order for partitionScheme.ops and
    // slice in that order.
    int numOps = partitionScheme.ops.size();
    for (int j = 0; j < numOps; j++) {
      auto op = partitionScheme.ops[j];
      sliceOp(op, i, builder, mappings, reverseMappings, partitionScheme);
    }

    // clean up
    LLVM_DEBUG({
      LDBG("prior to clean up:");
      funcOp.dump();
    });
    SmallVector<Operation *> opsToDelete;
    for (auto op : partitionScheme.ops) {
      if (op->hasAttr("to_be_removed"))
        opsToDelete.push_back(op);
    }
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }
  }

  LLVM_DEBUG({
    LDBG("prior to final opsToDelete:");
    funcOp.dump();
  });

  SmallVector<Operation *> opsToDelete;
  for (auto op : partitionScheme.ops) {
    if (isa<scf::YieldOp>(op))
      continue;
    bool notUsed = true;
    for (auto result : op->getResults()) {
      if (!result.getUsers().empty()) {
        notUsed = false;
        break;
      }
    }
    if (notUsed)
      opsToDelete.push_back(op);
  }

  LLVM_DEBUG({
    LDBG("opsToDelete:\n");
    for (auto op : opsToDelete) {
      LDBG("op: ");
      op->dump();
    }
    LDBG("\n");
  });
  for (auto op : opsToDelete) {
    partitionScheme.ops.remove(op);
    op->erase();
  }
  LLVM_DEBUG({
    LDBG("prior to clean up:");
    funcOp.dump();
  });

  // delete block arguments
  RewritePatternSet cleanUpPatterns(funcOp.getContext());
  populateForOpDeadArgumentElimination(cleanUpPatterns);
  scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns, funcOp.getContext());
  scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns, funcOp.getContext());
  if (applyPatternsAndFoldGreedily(funcOp, std::move(cleanUpPatterns))
          .failed()) {
    llvm_unreachable("failed to clean up");
    // signalPassFailure();
  }

  // Make sure original ops are not used
  LLVM_DEBUG({
    LDBG("after partition");
    funcOp.dump();
    LDBG("\n");
  });
  fixTaskId(funcOp);
}

#define GEN_PASS_DEF_TRITONGPUWSDATAPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUWSDataPartitionPass
    : public impl::TritonGPUWSDataPartitionBase<TritonGPUWSDataPartitionPass> {
public:
  using impl::TritonGPUWSDataPartitionBase<
      TritonGPUWSDataPartitionPass>::TritonGPUWSDataPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numConsumerGroups == 0)
      return;
    partitionTasks(funcOp, numConsumerGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
