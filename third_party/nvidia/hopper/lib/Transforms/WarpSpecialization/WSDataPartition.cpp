#include "Utility.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <optional>

using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-data-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static const char *kDataPartitionAttrName = "tt.data_partition_factor";

static bool containsAll(const SmallVector<WSPartitionId> &superset,
                        const SmallVector<WSPartitionId> &subset) {
  for (WSPartitionId id : subset) {
    if (!llvm::is_contained(superset, id))
      return false;
  }
  return true;
}

static bool isControlFlowOp(Operation *op) {
  return isa<ReturnOp, FuncOp, scf::YieldOp, scf::ConditionOp, scf::ForOp,
             scf::IfOp, scf::WhileOp>(op);
}

// Ensure all ops in the def-use chain carry the correct partition IDs.
static void fixTaskId(triton::FuncOp &funcOp) {
  bool changed = false;
  do {
    changed = false;
    funcOp.walk([&](Operation *op) {
      auto partitionIds = getWSPartitionIds(op);
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // Do not update loads.
        if (isa<LoadOp, DescriptorLoadOp>(defOp))
          continue;
        auto defTaskIds = getWSPartitionIds(defOp);
        // Backward propagation: ensure def covers op's partition IDs.
        if (!containsAll(defTaskIds, partitionIds)) {
          // Skip control flow ops.
          if (isa<scf::YieldOp, scf::ConditionOp, scf::ForOp, scf::IfOp,
                  scf::WhileOp>(op))
            continue;
          // Only propagate backward to arithmetic ops (e.g. constants).
          // Const ops with same value but different partition ids can be
          // folded.
          if (defOp->getDialect()->getNamespace() == "arith") {
            LLVM_DEBUG({
              LDBG("backward fixing taskId for");
              defOp->dump();
            });
            addWSPartitionIds(defOp, partitionIds);
            changed = true;
            LLVM_DEBUG({
              LDBG("resulting");
              defOp->dump();
            });
          }
        }

        // Forward propagation: ensure op covers def's partition IDs
        if (operand.hasOneUse() && !containsAll(partitionIds, defTaskIds)) {
          // YieldOp/ConditionOp may lose task attributes during MLIR
          // canonicalization.
          if (isa<scf::YieldOp, scf::ConditionOp, scf::IfOp, scf::WhileOp>(
                  op)) {
            LLVM_DEBUG({
              LDBG("forward fixing taskId for");
              defOp->dump();
            });
            addWSPartitionIds(op, defTaskIds);
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
  bool skipPartitioning = false;
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
  // Function arguments (TensorDescType) that need their block type sliced.
  // Maps argument index -> partition dimension (in descriptor space).
  DenseMap<unsigned, unsigned> funcArgPartitionDims;

  // op with noOpPartitionDim will be duplicated instead of partitioned.
  // Use -2 to avoid conflict with Empty/Tombstone value.
  static const unsigned noOpPartitionDim = ~0U - 2;

  void append(DataPartitionScheme &other) {
    skipPartitioning |= other.skipPartitioning;
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
    for (auto &[argIndex, dim] : other.funcArgPartitionDims) {
      auto it = funcArgPartitionDims.find(argIndex);
      assert((it == funcArgPartitionDims.end() || it->second == dim) &&
             "funcArgPartitionDims conflict during append");
      funcArgPartitionDims[argIndex] = dim;
    }
  }

  bool partitionIsCompatible() {
    for (Operation *op : ops) {
      auto it = opPartitionDims.find(op);
      assert(it != opPartitionDims.end() && "missing partition dim");
      unsigned dim = it->second;
      if (dim != 0)
        continue;

      for (Value result : op->getResults()) {
        auto type = dyn_cast<MemDescType>(result.getType());
        if (!type)
          continue;

        auto tmem =
            dyn_cast<ttng::TensorMemoryEncodingAttr>(type.getEncoding());
        if (!tmem)
          continue;

        ArrayRef<int64_t> shape = type.getShape().take_back(2);
        int64_t slicedM = shape[0] / numPartitions;
        int64_t minM =
            tmem.getBlockM() * tmem.getCGALayout().getCTASplitNum()[0];
        if (slicedM >= minM)
          continue;

        op->emitWarning()
            << "skipping M-dimension data partitioning because slicing "
               "TMEM result from "
            << shape[0] << " to " << slicedM
            << " rows would require updating tensor memory encoding blockM="
            << tmem.getBlockM();
        skipPartitioning = true;
        return false;
      }
    }

    return true;
  }

  bool isValidPartitionDim(unsigned dim) const {
    return dim < DataPartitionScheme::noOpPartitionDim ||
           dim == DataPartitionScheme::noOpPartitionDim;
  }

  unsigned flipPartitionDim(unsigned dim, const ArrayRef<int32_t> &order,
                            bool forward) const {
    if (dim == DataPartitionScheme::noOpPartitionDim)
      return dim;
    return forward ? order[dim] : llvm::find(order, dim) - order.begin();
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

    if (!funcArgPartitionDims.empty()) {
      LDBG(" func arg partition dims:");
      for (auto &[argIndex, dim] : funcArgPartitionDims) {
        LDBG("  arg " << argIndex << " -> dim " << dim);
      }
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
  else if (auto tensorDescType = dyn_cast<TensorDescType>(type))
    return {tensorDescType.getShape().begin(), tensorDescType.getShape().end()};
  else if (auto ptrType = dyn_cast<PointerType>(type))
    return getShape(ptrType.getPointeeType());
  return {};
}

static SmallVector<int64_t> getShape(Value v) { return getShape(v.getType()); }

static bool needToSlice(Value v, unsigned dim, int size) {
  if (dim == DataPartitionScheme::noOpPartitionDim)
    return true;
  if (isa<AsyncTokenType>(v.getType()))
    return true;
  auto shape = getShape(v);
  return shape.size() > dim && shape[dim] >= size;
}

static bool isDotOrMMAv5Op(Operation *op) {
  return isa<nvidia_gpu::WarpGroupDotOp, ttng::MMAv5OpInterface>(op);
}

static Value getDotOperandA(Operation *op) {
  if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op))
    return dotOp.getA();
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op))
    return mmaOp.getA();
  llvm_unreachable("expected dot or MMAv5 op");
}

static Value getDotOperandB(Operation *op) {
  if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op))
    return dotOp.getB();
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op))
    return mmaOp.getB();
  llvm_unreachable("expected dot or MMAv5 op");
}

static Value getDotAccumulatorInput(Operation *op) {
  if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op))
    return dotOp.getC();
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op))
    return mmaOp.getAccumulator();
  llvm_unreachable("expected dot or MMAv5 op");
}

static Value getDotPartitionRoot(Operation *op) {
  if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op))
    return dotOp.getD();
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op))
    return mmaOp.getAccumulator();
  llvm_unreachable("expected dot or MMAv5 op");
}

static Value getMMAv5ScaleOperand(Operation *op, unsigned operandIdx) {
  auto scaledMmaOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op);
  if (!scaledMmaOp)
    return {};
  assert((operandIdx == 0 || operandIdx == 1) && "unexpected MMA operand");
  return operandIdx == 0 ? scaledMmaOp.getAScale() : scaledMmaOp.getBScale();
}

static std::optional<unsigned>
getTmemCopySrcPartitionDim(ttng::TMEMCopyOp copyOp, unsigned dstDim) {
  if (dstDim == DataPartitionScheme::noOpPartitionDim)
    return dstDim;

  auto dstTy = copyOp.getDst().getType();
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(dstTy.getEncoding()))
    return dstDim;

  // Scale TMEM copy sources encode logical rows as packed 32x128b chunks.
  // The common forms put repRows in dim 0; the 5D TMA form uses dim 1.
  if (dstDim != 0)
    return std::nullopt;
  auto srcShape = copyOp.getSrc().getType().getShape();
  if (srcShape.size() == 5 && srcShape[0] == 1 && srcShape[3] == 2 &&
      srcShape[4] == 256)
    return 1;
  return 0;
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
    (void)getBackwardSlice(op, &slice);
    for (auto depOp : slice)
      partitionScheme.undoPartition(depOp);
    return true;
  }
  return false;
}

// Given shape1 and shape2, where shape1 value is the unsqueezed
// shape and shape2 is the squeezed shape, determine a mapping from
// an origDim to the other dim. When unsqueeze=True we are mapping
// from shape2 to shape1, but when unsqueeze=False we are mapping
// from shape1 to shape2.
static unsigned remappedSqueezedDim(SmallVector<int64_t> &shape1,
                                    SmallVector<int64_t> &shape2,
                                    unsigned origDim, bool unsqueeze) {
  if (shape1.size() == shape2.size()) {
    return origDim;
  }
  // Total is currDim + offset when unsqueeze = False
  // and currDim when unsqueeze = True
  unsigned total = 0;
  unsigned currDim = 0;
  unsigned offset = 0;
  while (total <= origDim) {
    if (shape1[currDim + offset] == shape2[currDim]) {
      currDim++;
      total++;
    } else {
      assert(shape1[currDim + offset] == 1);
      offset++;
      if (!unsqueeze) {
        total++;
      }
    }
  }
  if (unsqueeze) {
    return origDim + offset;
  } else {
    return origDim - offset;
  }
}

struct FlatInterval {
  int64_t begin;
  int64_t end;

  bool operator==(const FlatInterval &other) const {
    return begin == other.begin && end == other.end;
  }
};

static std::optional<int64_t> product(ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0)
      return std::nullopt;
    result *= value;
  }
  return result;
}

static SmallVector<FlatInterval> getPartitionIntervals(ArrayRef<int64_t> shape,
                                                       unsigned dim,
                                                       unsigned numPartitions,
                                                       unsigned offset) {
  SmallVector<FlatInterval> intervals;
  if (dim >= shape.size() || shape[dim] < numPartitions ||
      shape[dim] % numPartitions != 0)
    return intervals;

  auto maybeOuterCount = product(shape.take_front(dim));
  auto maybeStride = product(shape.drop_front(dim + 1));
  if (!maybeOuterCount || !maybeStride)
    return intervals;

  int64_t outerCount = *maybeOuterCount;
  int64_t stride = *maybeStride;
  int64_t chunk = shape[dim] / numPartitions;
  int64_t intervalLen = chunk * stride;
  int64_t period = shape[dim] * stride;
  for (int64_t outer = 0; outer < outerCount; ++outer) {
    int64_t begin = outer * period + offset * intervalLen;
    intervals.push_back({begin, begin + intervalLen});
  }

  SmallVector<FlatInterval> merged;
  for (FlatInterval interval : intervals) {
    if (!merged.empty() && merged.back().end == interval.begin) {
      merged.back().end = interval.end;
      continue;
    }
    merged.push_back(interval);
  }
  return merged;
}

static bool partitionIntervalsMatch(ArrayRef<int64_t> lhsShape, unsigned lhsDim,
                                    ArrayRef<int64_t> rhsShape, unsigned rhsDim,
                                    unsigned numPartitions) {
  for (unsigned offset = 0; offset < numPartitions; ++offset) {
    auto lhsIntervals =
        getPartitionIntervals(lhsShape, lhsDim, numPartitions, offset);
    auto rhsIntervals =
        getPartitionIntervals(rhsShape, rhsDim, numPartitions, offset);
    if (lhsIntervals.empty() || rhsIntervals.empty() ||
        lhsIntervals != rhsIntervals)
      return false;
  }
  return true;
}

static std::optional<unsigned>
remapReshapePartitionDim(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                         unsigned dim, unsigned numPartitions, bool forward) {
  auto maybeSrcElements = product(srcShape);
  auto maybeDstElements = product(dstShape);
  if (!maybeSrcElements || !maybeDstElements ||
      *maybeSrcElements != *maybeDstElements)
    return std::nullopt;

  ArrayRef<int64_t> fromShape = forward ? srcShape : dstShape;
  ArrayRef<int64_t> toShape = forward ? dstShape : srcShape;
  if (dim >= fromShape.size())
    return std::nullopt;

  for (auto [candidateDim, ignored] : llvm::enumerate(toShape)) {
    (void)ignored;
    if (partitionIntervalsMatch(fromShape, dim, toShape, candidateDim,
                                numPartitions))
      return candidateDim;
  }
  return std::nullopt;
}

static bool shapedResultsCanRepresentDim(Operation *op, unsigned dim) {
  if (dim == DataPartitionScheme::noOpPartitionDim)
    return true;
  for (Value result : op->getResults()) {
    Type type = result.getType();
    if (!isa<MemDescType, RankedTensorType, TensorDescType>(type))
      continue;
    if (getShape(type).size() <= dim)
      return false;
  }
  return true;
}

static bool getBackwardSliceToPartition(Value v,
                                        DataPartitionScheme &partitionScheme,
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
    if (auto transOp = dyn_cast<TransOp>(op)) {
      currentDim = partitionScheme.flipPartitionDim(currentDim,
                                                    transOp.getOrder(), false);
    } else if (auto memDescTransOp = dyn_cast<MemDescTransOp>(op)) {
      currentDim = partitionScheme.flipPartitionDim(
          currentDim, memDescTransOp.getOrder(), false);
    }

    if (auto expandDimsOp = dyn_cast<ExpandDimsOp>(op)) {
      // currentDim is the dim after expansion.
      assert(expandDimsOp.getAxis() != currentDim &&
             "expanded dim always has shape 1");
      // Parition along currentDim - 1 for ExpandDimsOp.
      if (expandDimsOp.getAxis() < currentDim)
        currentDim--;
    }

    // Recusively process operands backwards.
    if (auto loadOp = dyn_cast<DescriptorLoadOp>(op)) {
      auto outputShape = getShape(loadOp.getResult());
      auto inputShape = getShape(loadOp.getDesc());
      unsigned remappedDim =
          remappedSqueezedDim(inputShape, outputShape, currentDim, true);
      if (!getBackwardSliceToPartition(loadOp.getDesc(), partitionScheme,
                                       remappedDim)) {
        return false;
      }
    } else if (auto tmemAllocOp = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      for (Value operand : op->getOperands()) {
        if (!getBackwardSliceToPartition(operand, partitionScheme,
                                         currentDim)) {
          return false;
        }
      }
      for (Operation *user : tmemAllocOp.getResult().getUsers()) {
        auto copyOp = dyn_cast<ttng::TMEMCopyOp>(user);
        if (!copyOp || copyOp.getDst() != tmemAllocOp.getResult())
          continue;
        if (!partitionScheme.ops.insert(copyOp)) {
          if (!isControlFlowOp(copyOp) &&
              partitionScheme.opPartitionDims[copyOp] != currentDim) {
            return false;
          }
          continue;
        }
        partitionScheme.opPartitionDims[copyOp] = currentDim;
        auto srcDim = getTmemCopySrcPartitionDim(copyOp, currentDim);
        if (!srcDim)
          return false;
        if (!getBackwardSliceToPartition(copyOp.getSrc(), partitionScheme,
                                         *srcDim))
          return false;
      }
    } else if (op->hasTrait<OpTrait::Elementwise>() ||
               isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                   arith::ExtFOp, BroadcastOp, ExpandDimsOp, MakeRangeOp,
                   SplatOp, ConvertLayoutOp, triton::gpu::LocalAllocOp, LoadOp,
                   triton::gpu::LocalLoadOp, TransOp, MemDescTransOp,
                   AtomicRMWOp, triton::AddPtrOp, nvidia_gpu::TMEMLoadOp,
                   nvidia_gpu::TMEMStoreOp, FpToFpOp, SplitOp, JoinOp,
                   MapElementwiseOp>(op)) {
      for (Value operand : op->getOperands())
        if (!getBackwardSliceToPartition(operand, partitionScheme,
                                         currentDim)) {
          return false;
        }
    } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
      if (!getBackwardSliceToPartition(currentDim == 0 ? Value(dotOp.getA())
                                                       : dotOp.getB(),
                                       partitionScheme, currentDim))
        return false;
      if (!getBackwardSliceToPartition(dotOp.getC(), partitionScheme,
                                       currentDim))
        return false;
      partitionScheme.dotPartitionOperand[dotOp] = currentDim == 0 ? 0 : 1;
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      unsigned operandIdx = currentDim == 0 ? 0 : 1;
      Value operand = operandIdx == 0 ? mmaOp.getA() : mmaOp.getB();
      if (!getBackwardSliceToPartition(operand, partitionScheme, currentDim)) {
        return false;
      }
      if (!getBackwardSliceToPartition(mmaOp.getAccumulator(), partitionScheme,
                                       currentDim)) {
        return false;
      }
      if (Value scale = getMMAv5ScaleOperand(op, operandIdx)) {
        // TODO: Add correctness coverage for this scale slicing path. PTX scale
        // layouts are logical (M-or-N, K/scale_vec), so output M/N
        // partitioning slices the selected scale operand along logical dim 0.
        if (!getBackwardSliceToPartition(scale, partitionScheme,
                                         /*currentDim=*/0))
          return false;
      }
      partitionScheme.dotPartitionOperand[mmaOp] = operandIdx;
    } else if (auto tmemCopyOp = dyn_cast<ttng::TMEMCopyOp>(op)) {
      auto srcDim = getTmemCopySrcPartitionDim(tmemCopyOp, currentDim);
      if (!srcDim)
        return false;
      if (!getBackwardSliceToPartition(tmemCopyOp.getSrc(), partitionScheme,
                                       *srcDim))
        return false;
      if (!getBackwardSliceToPartition(tmemCopyOp.getDst(), partitionScheme,
                                       currentDim))
        return false;
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      auto srcShape = getShape(reshapeOp.getSrc());
      auto dstShape = getShape(reshapeOp.getResult());
      auto srcDim = remapReshapePartitionDim(srcShape, dstShape, currentDim,
                                             partitionScheme.numPartitions,
                                             /*forward=*/false);
      if (!srcDim)
        return false;
      if (!getBackwardSliceToPartition(reshapeOp.getSrc(), partitionScheme,
                                       *srcDim))
        return false;
    } else if (auto reshapeOp = dyn_cast<MemDescReshapeOp>(op)) {
      auto srcShape = getShape(reshapeOp.getSrc());
      auto dstShape = getShape(reshapeOp.getResult());
      auto srcDim = remapReshapePartitionDim(srcShape, dstShape, currentDim,
                                             partitionScheme.numPartitions,
                                             /*forward=*/false);
      if (!srcDim)
        return true;
      if (!getBackwardSliceToPartition(reshapeOp.getSrc(), partitionScheme,
                                       *srcDim))
        return false;
    } else if (isa<ttng::ReinterpretTensorDescOp, MakeTensorDescOp>(op)) {
      return true;
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
      if (!getBackwardSliceToPartition(thenYieldArg, partitionScheme,
                                       currentDim)) {
        return false;
      }
      if (!getBackwardSliceToPartition(elseYieldArg, partitionScheme,
                                       currentDim)) {
        return false;
      }
    } else {
      llvm_unreachable("Unexpected op");
    }
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(bbAargOwner)) {
      if (auto whileOp = dyn_cast<scf::WhileOp>(bbAargOwner);
          whileOp && bbArg.getOwner() == whileOp.getAfterBody()) {
        // scf.while "after" arg j is fed from the scf.condition forwarded
        // operand j (which also feeds result j).
        Value fwd = whileOp.getConditionOp().getArgs()[bbArg.getArgNumber()];
        if (!getBackwardSliceToPartition(fwd, partitionScheme, currentDim)) {
          return false;
        }
      } else {
        // scf.for body iter arg, or scf.while "before" arg: track the init
        // value and the value yielded back for the next iteration. The slot
        // abstraction hides the induction-variable offset (scf.for) and the
        // two-region structure (scf.while). A scf.for induction variable is not
        // in getRegionIterArgs() and is skipped.
        auto iterArgs = loop.getRegionIterArgs();
        auto it = llvm::find(iterArgs, bbArg);
        if (it != iterArgs.end()) {
          auto slot =
              getLoopCarriedSlot(loop, std::distance(iterArgs.begin(), it));
          if (slot.init && !getBackwardSliceToPartition(
                               slot.init->get(), partitionScheme, currentDim)) {
            return false;
          }
          if (slot.yielded &&
              !getBackwardSliceToPartition(slot.yielded->get(), partitionScheme,
                                           currentDim)) {
            return false;
          }
        }
      }
    } else if (isa<triton::FuncOp>(bbAargOwner)) {
      if (isa<TensorDescType>(bbArg.getType())) {
        unsigned argIndex = bbArg.getArgNumber();
        auto it = partitionScheme.funcArgPartitionDims.find(argIndex);
        if (it != partitionScheme.funcArgPartitionDims.end()) {
          // Same arg reached again; must agree on dimension.
          if (it->second != currentDim) {
            return false;
          }
        } else {
          partitionScheme.funcArgPartitionDims[argIndex] = currentDim;
        }
      }
    }
  }

  return true;
};

// Return false if the partition is not possible.
static bool getForwardSliceToPartition(Value v,
                                       DataPartitionScheme &partitionScheme,
                                       unsigned currentDim,
                                       DenseSet<Value> &seen) {
  assert(partitionScheme.isValidPartitionDim(currentDim) && "invalid dim");
  auto op = v.getDefiningOp();
  if (op) {
    if (auto expandDimsOp = dyn_cast<ExpandDimsOp>(op)) {
      if (currentDim != DataPartitionScheme::noOpPartitionDim &&
          currentDim >= expandDimsOp.getAxis()) {
        currentDim += 1;
        // Update the result for expand dims
        partitionScheme.opPartitionDims[op] = currentDim;
      }
    }
  }
  if (!seen.insert(v).second)
    return true;
  if (!needToSlice(v, currentDim, partitionScheme.numPartitions))
    return true;

  // Recusively process operands forwards.
  unsigned originalDim = currentDim;
  for (Operation *depOp : v.getUsers()) {
    currentDim = originalDim;
    // Flip dim when op is trans
    if (auto transOp = dyn_cast<TransOp>(depOp)) {
      currentDim = partitionScheme.flipPartitionDim(currentDim,
                                                    transOp.getOrder(), true);
    } else if (auto memDescTransOp = dyn_cast<MemDescTransOp>(depOp)) {
      currentDim = partitionScheme.flipPartitionDim(
          currentDim, memDescTransOp.getOrder(), true);
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(depOp)) {
      auto srcShape = getShape(reshapeOp.getSrc());
      auto dstShape = getShape(reshapeOp.getResult());
      auto dstDim = remapReshapePartitionDim(srcShape, dstShape, currentDim,
                                             partitionScheme.numPartitions,
                                             /*forward=*/true);
      if (!dstDim)
        return false;
      currentDim = *dstDim;
    } else if (auto reshapeOp = dyn_cast<MemDescReshapeOp>(depOp)) {
      auto srcShape = getShape(reshapeOp.getSrc());
      auto dstShape = getShape(reshapeOp.getResult());
      auto dstDim = remapReshapePartitionDim(srcShape, dstShape, currentDim,
                                             partitionScheme.numPartitions,
                                             /*forward=*/true);
      if (!dstDim)
        return false;
      currentDim = *dstDim;
    }

    if (!shapedResultsCanRepresentDim(depOp, currentDim))
      return false;

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
      Operation *atomicStore = nullptr;
      for (auto op : forwardSlice) {
        if (isa<AtomicRMWOp, DescriptorReduceOp>(op)) {
          atomicStore = op;
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

    if (isDotOrMMAv5Op(depOp)) {
      Value opndA = getDotOperandA(depOp);
      Value opndB = getDotOperandB(depOp);
      if ((currentDim == 0 && v == opndB) || (currentDim == 1 && v == opndA)) {
        // It is fine to continue the partition if the dot output is immediately
        // stored out via an atomic add, as the dot computes a partial result.
        if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(depOp);
            dotOp && onlyUsedByAtomicStore(dotOp.getD())) {
          partitionScheme.dotPartitionOperand[depOp] = v == opndA ? 0 : 1;
          // Duplicate the users of the dot output since the shape of the output
          // will not be changed
          currentDim = DataPartitionScheme::noOpPartitionDim;
        } else {
          LLVM_DEBUG({
            auto opnd = (v == opndA) ? "A" : "B";
            LDBG("skip partitioning along K of " << opnd << " of dot\n");
            depOp->dump();
          });
          return false;
        }
      } else if (v == opndA || v == getMMAv5ScaleOperand(depOp, 0)) {
        partitionScheme.dotPartitionOperand[depOp] = 0;
      } else if (v == opndB || v == getMMAv5ScaleOperand(depOp, 1)) {
        partitionScheme.dotPartitionOperand[depOp] = 1;
      } else {
        partitionScheme.dotPartitionOperand[depOp] = currentDim == 0 ? 0 : 1;
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
          unsigned idx = operand.getOperandNumber();
          // For scf.for / scf.if, yield operand k feeds result k. For an
          // scf.while "after" yield, operand k instead feeds "before" arg k
          // (the loop-carry back edge); results come from scf.condition.
          Value fwd;
          if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
            fwd = whileOp.getBeforeArguments()[idx];
          else
            fwd = parentOp->getResult(idx);
          if (!getForwardSliceToPartition(fwd, partitionScheme, currentDim,
                                          seen))
            return false;
        }
      }
    } else if (auto condOp = dyn_cast<scf::ConditionOp>(depOp)) {
      // scf.while "before" terminator: a forwarded operand feeds both the
      // matching result and the matching "after" arg.
      auto whileOp = cast<scf::WhileOp>(condOp->getParentOp());
      partitionScheme.ops.insert(whileOp);
      for (OpOperand &operand : condOp.getArgsMutable()) {
        if (operand.get() == v) {
          unsigned argIdx = operand.getOperandNumber() - 1;
          if (!getForwardSliceToPartition(whileOp->getResult(argIdx),
                                          partitionScheme, currentDim, seen))
            return false;
          if (!getForwardSliceToPartition(whileOp.getAfterArguments()[argIdx],
                                          partitionScheme, currentDim, seen))
            return false;
        }
      }
    }
  }

  return true;
};

// Compute a closure of all ops originated from
// or being dependent on by the root op.
static bool getSliceToPartition(Value root,
                                DataPartitionScheme &partitionScheme,
                                unsigned currentDim) {
  if (!getBackwardSliceToPartition(root, partitionScheme, currentDim)) {
    return false;
  }
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
    if (auto descStoreOp = dyn_cast<DescriptorStoreOp>(op)) {
      auto inputShape = getShape(descStoreOp.getSrc());
      auto outputShape = getShape(descStoreOp.getDesc());
      unsigned remappedDim =
          remappedSqueezedDim(outputShape, inputShape, currentDim, false);
      if (!getBackwardSliceToPartition(descStoreOp.getDesc(), partitionScheme,
                                       currentDim))
        return false;
      if (!getBackwardSliceToPartition(descStoreOp.getSrc(), partitionScheme,
                                       remappedDim))
        return false;
    } else if (auto tmemStoreOp = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
      if (!getBackwardSliceToPartition(tmemStoreOp.getSrc(), partitionScheme,
                                       currentDim))
        return false;
    } else if (op->hasTrait<OpTrait::Elementwise>() ||
               isa<StoreOp, AtomicRMWOp, MapElementwiseOp>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (!getBackwardSliceToPartition(operand.get(), partitionScheme,
                                         currentDim))
          return false;
      }
    } else if (isDotOrMMAv5Op(op)) {
      unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
      if (!getBackwardSliceToPartition(op->getOperand(opndIndx),
                                       partitionScheme, currentDim))
        return false;
      Value accumulator = getDotAccumulatorInput(op);

      if (currentDim == 0 && opndIndx == 0 ||
          currentDim == 1 && opndIndx == 1) {
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(accumulator, partitionScheme,
                                         currentDim))
          return false;
        if (Value scale = getMMAv5ScaleOperand(op, opndIndx)) {
          if (!getBackwardSliceToPartition(scale, partitionScheme,
                                           /*currentDim=*/0))
            return false;
        }
      } else {
        // slice the other operand
        unsigned otherOpndIndx = 1 - opndIndx;
        if (!getBackwardSliceToPartition(op->getOperand(otherOpndIndx),
                                         partitionScheme, 1 - currentDim))
          return false;
        if (Value scale = getMMAv5ScaleOperand(op, otherOpndIndx)) {
          if (!getBackwardSliceToPartition(scale, partitionScheme,
                                           /*currentDim=*/0))
            return false;
        }
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(accumulator, partitionScheme,
                                         DataPartitionScheme::noOpPartitionDim))
          return false;
      }
    }
  }

  return true;
}

static bool computePartitionScheme(triton::FuncOp &funcOp,
                                   DataPartitionScheme &partitionScheme) {
  // Use dot to drive the partition
  SetVector<Operation *> dots;
  SetVector<Operation *> fallbackDots;

  // Prefer dots that span multiple partition IDs, but preserve explicit data
  // partitioning for IR that does not have multi-task dot annotations.
  funcOp.walk([&](Operation *op) {
    if (!isDotOrMMAv5Op(op))
      return;
    if (getWSPartitionIds(op).size() > 1)
      dots.insert(op);
    else
      fallbackDots.insert(op);
  });

  if (dots.empty())
    dots = fallbackDots;

  if (dots.empty())
    return true;

  // Checking if all dots can be partitioned in the same way
  for (auto op : dots) {
    if (partitionScheme.isPartitioned(op) || partitionScheme.isSkipped(op)) {
      continue;
    }

    // partition along M first, otherwise along N
    Value accumulator = getDotPartitionRoot(op);

    auto dotType = accumulator.getType();
    LLVM_DEBUG({
      LDBG("Computing partition scheme for");
      op->dump();
      LDBG("\n");
    });

    auto partitionIds = getWSPartitionIds(op);
    if (partitionScheme.numPartitions == 0) {
      partitionScheme.numPartitions = partitionIds.size();
    }

    auto shapePerCTA = getShapePerCTA(dotType);
    if (shapePerCTA.size() != 2) {
      LDBG("partition not possible: shapePerCTA " << shapePerCTA.size());
      return false;
    }
    int sliceSizeM = shapePerCTA[0] / partitionScheme.numPartitions;
    int sliceSizeN = shapePerCTA[1] / partitionScheme.numPartitions;
    SmallVector<unsigned, 2> partitionDim, partitionSize;

    if (sliceSizeM >= 64) {
      partitionDim.push_back(0);
      partitionSize.push_back(sliceSizeM);
    }

    if (sliceSizeN >= 128) {
      partitionDim.push_back(1);
      partitionSize.push_back(sliceSizeN);
    }

    if (partitionDim.empty()) {
      LDBG("Partition not available: " << sliceSizeM << " " << sliceSizeN);
      return false;
    }

    bool success = false;
    for (int i = 0; i < partitionDim.size(); ++i) {
      // Partition the slice closure
      auto trialPartitionScheme = partitionScheme;
      LLVM_DEBUG(
          { LDBG("Trying partition along " << partitionDim[i] << " \n"); });

      bool compatible = getSliceToPartition(accumulator, trialPartitionScheme,
                                            partitionDim[i]) &&
                        trialPartitionScheme.partitionIsCompatible();
      if (trialPartitionScheme.skipPartitioning) {
        unsigned numPartitions = partitionScheme.numPartitions;
        partitionScheme = DataPartitionScheme();
        partitionScheme.numPartitions = numPartitions;
        partitionScheme.skipPartitioning = true;
        return true;
      }
      if (compatible) {
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
static void rewriteRematerializedOps(triton::FuncOp &funcOp,
                                     DataPartitionScheme &partitionScheme) {
  if (partitionScheme.rematerializedOps.empty())
    return;

  OpBuilderWithPartitionIds builder(funcOp.getContext());

  // For each rematerialized op, create a new op and replace its user with it.
  for (auto opDim : partitionScheme.rematerializedOps) {
    auto oldOp = opDim.first;
    builder.setInsertionPoint(oldOp);
    builder.setPartitionIdsFromOp(oldOp);

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
        auto slicedMemdescType = MemDescType::get(
            shape, memdescType.getElementType(), memdescType.getEncoding(),
            memdescType.getMemorySpace(), memdescType.getMutableMemory());
        SmallVector<int32_t> offsets(shape.size(), 0);
        auto viewOp = builder.createWithPartitionIds<MemDescSubsliceOp>(
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

      setWSPartitionIds(newOp, getWSPartitionIds(oldOp));
      partitionScheme.ops.insert(newOp);
      partitionScheme.opPartitionDims[newOp] = dim;

      // replace the users that have same partition dim with the op.
      auto dimMatches = [&](OpOperand &operand) {
        auto user = operand.getOwner();
        assert(partitionScheme.opPartitionDims.contains(user) &&
               "user not partitioned");
        unsigned userDim = partitionScheme.opPartitionDims[user];
        if (auto transOp = dyn_cast<TransOp>(user)) {
          userDim = partitionScheme.flipPartitionDim(userDim,
                                                     transOp.getOrder(), true);
        } else if (auto memDescTransOp = dyn_cast<MemDescTransOp>(user)) {
          userDim = partitionScheme.flipPartitionDim(
              userDim, memDescTransOp.getOrder(), true);
        } else if (isDotOrMMAv5Op(user)) {
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

static Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                          IRMapping &reverseMappings,
                          DataPartitionScheme &partitionScheme);

static Operation *sliceOp(Operation *op, int offset, IRMapping &mappings,
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

  auto partitionIds = getWSPartitionIds(op);
  SmallVector<mlir::WSPartitionId, 3> sliceTaskIds;
  if (partitionIds.size() == numOfPartitions) {
    // We are slicing the op for consumer only
    sliceTaskIds.push_back(partitionIds[offset]);
  } else if (partitionIds.size() == 1) {
    // We are slicing the op for producer only
    sliceTaskIds.push_back(partitionIds.front());
  } else if (partitionIds.size() > numOfPartitions) {
    // We are slicing the op for both producer and consumer
    sliceTaskIds.push_back(partitionIds.front());
    sliceTaskIds.push_back(partitionIds[offset + 1]);
  } else {
    assert(partitionIds.empty() && "Unexpected partitionIds.size()");
  }

  OpBuilderWithPartitionIds builder(op->getContext());
  builder.setPartitionIdsFromArray(sliceTaskIds);
  auto cloneAndSetResultType = [&](Operation *op) {
    builder.setInsertionPoint(op);
    auto newOp = builder.clone(*op, mappings);
    setWSPartitionIds(newOp, sliceTaskIds);
    if (numOfPartitions > 1 && isa<LocalAllocOp, ttng::TMEMAllocOp>(newOp)) {
      newOp->setLoc(appendToNameLoc(
          newOp->getLoc(), "_" + std::to_string(offset), op->getContext()));
    }
    mappings.map(op, newOp);
    reverseMappings.map(newOp, op);
    // set result shape for all results
    for (auto [v, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      bool needRetype = true;
      if (dim == DataPartitionScheme::noOpPartitionDim) {
        // Just duplicate the op for noOpPartitionDim
        needRetype = false;
      } else if (isDotOrMMAv5Op(op)) {
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
          // change encoding for ttng.tensor_memory_encoding to match gen5.
          if (auto tmem = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(
                  type.getEncoding())) {
            Attribute accEncoding =
                triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
                    builder.getContext(), tmem.getBlockM(),
                    dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
                    tmem.getColStride(), tmem.getCGALayout(), tmem.getTwoCTAs(),
                    tmem.getCtaMode());
            auto newType = MemDescType::get(shape, type.getElementType(),
                                            accEncoding, type.getMemorySpace(),
                                            type.getMutableMemory());
            newV.setType(newType);
          } else {
            auto newType = MemDescType::get(
                shape, type.getElementType(), type.getEncoding(),
                type.getMemorySpace(), type.getMutableMemory());
            newV.setType(newType);
          }
        } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
          SmallVector<int64_t> shape{type.getShape().begin(),
                                     type.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newType = RankedTensorType::get(shape, type.getElementType(),
                                               type.getEncoding());
          newV.setType(newType);
        } else if (auto type = dyn_cast<TensorDescType>(v.getType())) {
          SmallVector<int64_t> shape(type.getShape());
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newType = TensorDescType::get(shape, type.getElementType(),
                                             type.getSharedLayout());
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
          AtomicRMWOp, LocalAllocOp, LocalLoadOp, SplitOp, JoinOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto tmemLdOp = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    auto srcTy = mappings.lookupOrNull(tmemLdOp.getSrc()).getType();
    auto type = cast<MemDescType>(srcTy);

    RankedTensorType oldRetType = tmemLdOp.getType();
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = mlir::triton::gpu::lookupNumWarps(op);
    builder.setInsertionPoint(op);
    // The source op is already sliced at this point, so srcTy, type, tmem is
    // sliced. We use getTmemCompatibleLayout to get a block layout that is for
    // the sliced tmem here.
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(type, numWarps);

    // oldRetType is the desired output, we slice it and convert from the
    // compatible layout to the sliced desired output.
    SmallVector<int64_t> shape{oldRetType.getShape().begin(),
                               oldRetType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newAccType = RankedTensorType::get(shape, oldRetType.getElementType(),
                                            newDistributedEncoding);
    triton::nvidia_gpu::TMEMLoadOp ld;

    // Create token
    if (auto token = tmemLdOp.getDep()) {
      ld = builder.createWithPartitionIds<triton::nvidia_gpu::TMEMLoadOp>(
          op->getLoc(), newAccType, token.getType(),
          mappings.lookupOrNull(tmemLdOp.getSrc()),
          mappings.lookupOrNull(token));
    } else {
      ld = builder.createWithPartitionIds<triton::nvidia_gpu::TMEMLoadOp>(
          op->getLoc(), newAccType, mappings.lookupOrNull(tmemLdOp.getSrc()));
    }

    // The TMEMLoad result has the TMEM-compatible layout (which may be
    // LinearEncodingAttr). Convert it to the sliced version of the original
    // layout so downstream ops (like tt.reduce) see the expected encoding.
    auto slicedOrigType = RankedTensorType::get(
        shape, oldRetType.getElementType(), oldRetType.getEncoding());
    Value loadResult = ld.getResult();
    if (loadResult.getType() != slicedOrigType) {
      auto cvtOp = builder.createWithPartitionIds<ConvertLayoutOp>(
          op->getLoc(), slicedOrigType, loadResult);
      mappings.map(tmemLdOp.getResult(), cvtOp->getResult(0));
    } else {
      mappings.map(tmemLdOp.getResult(), loadResult);
    }
    // Map the token result
    if (ld.getToken())
      mappings.map(tmemLdOp.getToken(), ld.getToken());
    for (auto [v, newV] : llvm::zip(op->getResults(), ld.getResults())) {
      reverseMappings.map(newV, v);
    }
    newOp = ld;
  } else if (auto tmemStOp = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
    sliceOp(tmemStOp.getDst(), offset, mappings, reverseMappings,
            partitionScheme);
    sliceOp(tmemStOp.getDep(), offset, mappings, reverseMappings,
            partitionScheme);

    // Slice retype the source operand with a tmem compatible layout.
    auto dstTy = mappings.lookupOrNull(tmemStOp.getDst()).getType();
    auto type = cast<MemDescType>(dstTy);
    auto tmem = cast<nvidia_gpu::TensorMemoryEncodingAttr>(type.getEncoding());

    RankedTensorType oldSrcType = tmemStOp.getSrc().getType();
    auto retShapePerCTA = getShapePerCTA(oldSrcType);
    int numWarps = mlir::triton::gpu::lookupNumWarps(op);
    builder.setInsertionPoint(op);
    // The source op is already sliced at this point, so srcTy, type, tmem is
    // sliced. We use getTmemCompatibleLayout to get a block layout that is
    // for the sliced tmem here.
    auto compatibleLayouts =
        nvidia_gpu::getTmemCompatibleLayouts(op, oldSrcType, type);
    assert(!compatibleLayouts.empty() && "No TMEM-compatible layout found");
    auto newDistributedEncoding = compatibleLayouts.front();
    // oldRetType is the desired output, we slice it and convert from the
    // compatible layout to the sliced desired output.
    SmallVector<int64_t> shape{oldSrcType.getShape().begin(),
                               oldSrcType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newSrcType = RankedTensorType::get(shape, oldSrcType.getElementType(),
                                            newDistributedEncoding);
    sliceOp(tmemStOp.getSrc(), offset, mappings, reverseMappings,
            partitionScheme);

    // Convert the source operand to a tmem compatible layout via
    // ConvertLayoutOp instead of mutating the type in-place (which would break
    // ops like arith.constant whose value attribute must match the result
    // type).
    auto newSrc = mappings.lookupOrNull(tmemStOp.getSrc());
    assert(newSrc && "TMEMStoreOp src not found in mappings; was it "
                     "backward-sliced in getSliceToPartition?");
    // The TMEM store needs its source in a TMEM-compatible layout, but that
    // requirement is local to this store: the source value may be shared with
    // other consumers (e.g. an arith.constant feeding both the store and a
    // downstream elementwise chain) that expect the original sliced encoding.
    // Convert only for this store and restore the shared mapping afterwards, so
    // remapping tmemStOp.getSrc() below does not force the TMEM-compatible
    // layout onto every other user of the value.
    Value prevSrcMapping = newSrc;
    if (newSrc.getType() != newSrcType) {
      auto cvtOp =
          ConvertLayoutOp::create(builder, op->getLoc(), newSrcType, newSrc);
      mappings.map(tmemStOp.getSrc(), cvtOp->getResult(0));
    }
    newOp = cloneAndSetResultType(op);
    mappings.map(tmemStOp.getSrc(), prevSrcMapping);
  } else if (auto tmemCopyOp = dyn_cast<nvidia_gpu::TMEMCopyOp>(op)) {
    sliceOp(tmemCopyOp.getDst(), offset, mappings, reverseMappings,
            partitionScheme);
    sliceOp(tmemCopyOp.getSrc(), offset, mappings, reverseMappings,
            partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto tmemAllocOp = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    // Check for src.
    if (tmemAllocOp.getSrc()) {
      // src is blocked layout. apply convert layout on src
      auto srcTy = cast<RankedTensorType>(
          mappings.lookupOrNull(tmemAllocOp.getSrc()).getType());

      // convert from srcTy to a compatible blocked layout.
      auto retShapePerCTA = getShapePerCTA(srcTy);
      int numWarps = mlir::triton::gpu::lookupNumWarps(op);
      builder.setInsertionPoint(op);

      // TODO: This should probably be written as memdesc_subslice
      // calculate new tmem type.
      auto retType = cast<MemDescType>(tmemAllocOp.getType());
      SmallVector<int64_t> shape{retType.getShape().begin(),
                                 retType.getShape().end()};
      int sliceSize = shape[dim] / numOfPartitions;
      shape[dim] = sliceSize;
      Attribute accEncoding = retType.getEncoding();
      if (auto tmem = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(
              retType.getEncoding())) {
        accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
            builder.getContext(), tmem.getBlockM(),
            dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
            tmem.getColStride(), tmem.getCGALayout(), tmem.getTwoCTAs(),
            tmem.getCtaMode());
      }
      auto newType = MemDescType::get(shape, retType.getElementType(),
                                      accEncoding, retType.getMemorySpace(),
                                      retType.getMutableMemory());

      auto newDistributedEncoding =
          nvidia_gpu::getDefaultLayoutForTmemLdSt(newType, numWarps);
      auto newAccType = RankedTensorType::get(
          srcTy.getShape(), srcTy.getElementType(), newDistributedEncoding);
      auto cvtOp = builder.createWithPartitionIds<ConvertLayoutOp>(
          op->getLoc(), newAccType,
          mappings.lookupOrNull(tmemAllocOp.getSrc()));

      Operation *alloc;
      // replace tmemAllocOp with alloc, where the src is cvtOp.
      // Create token
      if (auto token = tmemAllocOp.getToken()) {
        auto newAllocOp =
            builder.createWithPartitionIds<triton::nvidia_gpu::TMEMAllocOp>(
                op->getLoc(), newType, token.getType(), cvtOp);
        auto newToken = newAllocOp.getToken();
        mappings.map(token, newToken);
        reverseMappings.map(newToken, token);
        alloc = newAllocOp;
      } else {
        alloc = builder.createWithPartitionIds<triton::nvidia_gpu::TMEMAllocOp>(
            op->getLoc(), newType, cvtOp);
      }
      auto v = tmemAllocOp->getResult(0);
      auto newV = alloc->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
      newOp = alloc;
    } else
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
    newOp = builder.createWithPartitionIds<arith::ConstantOp>(op->getLoc(),
                                                              newValAttr);
    // Do not drop original partition id as constant folding may lose one
    // constant.
    setWSPartitionIds(newOp, getWSPartitionIds(op));
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
    newOp = builder.createWithPartitionIds<MakeRangeOp>(
        op->getLoc(), newType, newRangeStart, newRangeEnd);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (isa<StoreOp, LoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    // TODO: slice store base ptr
    newOp = cloneAndSetResultType(op);
  } else if (isa<DescriptorLoadOp, DescriptorStoreOp>(op)) {
    SmallVector<int64_t> shape;
    Value coordVal;
    if (auto loadOp = dyn_cast<DescriptorLoadOp>(op)) {
      sliceOp(loadOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = loadOp.getIndices()[dim];
      shape = getShape(loadOp.getResult());
    } else if (auto storeOp = dyn_cast<DescriptorStoreOp>(op)) {
      sliceOp(storeOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = storeOp.getIndices()[dim];
      shape = getShape(storeOp.getSrc());
    }
    auto newCoordVal = coordVal;
    if (offset) {
      if (auto *defOp = coordVal.getDefiningOp())
        builder.setInsertionPointAfter(defOp);
      else
        builder.setInsertionPoint(op);
      Value offsetVal = builder.createWithPartitionIds<arith::ConstantIntOp>(
          op->getLoc(), offset * shape[dim] / numOfPartitions, 32);
      newCoordVal = builder.createWithPartitionIds<arith::AddIOp>(
          op->getLoc(), coordVal, offsetVal);
      mappings.map(coordVal, newCoordVal);
      reverseMappings.map(newCoordVal, coordVal);
    }

    newOp = cloneAndSetResultType(op);
    if (isa<DescriptorLoadOp>(op)) {
      // map load result
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
  } else if (auto tensorDescOp = dyn_cast<MakeTensorDescOp>(op)) {
    newOp = cloneAndSetResultType(op);
  } else if (auto tensorDescOp = dyn_cast<ttng::ReinterpretTensorDescOp>(op)) {
    newOp = cloneAndSetResultType(op);
  } else if (isa<TransOp, MemDescTransOp>(op)) {
    sliceOp(op->getOperand(0), offset, mappings, reverseMappings,
            partitionScheme);
    builder.setInsertionPoint(op);
    auto v = op->getResult(0);
    SmallVector<int64_t> shape = getShape(v.getType());
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    Type newType;
    if (auto descType = dyn_cast<MemDescType>(v.getType())) {
      newType = MemDescType::get(
          shape, descType.getElementType(), descType.getEncoding(),
          descType.getMemorySpace(), descType.getMutableMemory());
    } else if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      newType = RankedTensorType::get(shape, tensorType.getElementType(),
                                      tensorType.getEncoding());
    } else {
      llvm_unreachable("unsupported type");
    }
    builder.setInsertionPoint(op);
    newOp = builder.clone(*op, mappings);
    setWSPartitionIds(newOp, sliceTaskIds);
    auto newV = newOp->getResult(0);
    newV.setType(newType);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
    auto srcShape = getShape(reshapeOp.getSrc());
    auto dstShape = getShape(reshapeOp.getResult());
    auto srcDim = remapReshapePartitionDim(srcShape, dstShape, dim,
                                           partitionScheme.numPartitions,
                                           /*forward=*/false);
    if (!srcDim)
      llvm_unreachable("partitioned tensor reshape has no source dimension");
    sliceOp(reshapeOp.getSrc(), offset, mappings, reverseMappings,
            partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto reshapeOp = dyn_cast<MemDescReshapeOp>(op)) {
    auto srcShape = getShape(reshapeOp.getSrc());
    auto dstShape = getShape(reshapeOp.getResult());
    auto srcDim = remapReshapePartitionDim(srcShape, dstShape, dim,
                                           partitionScheme.numPartitions,
                                           /*forward=*/false);
    if (srcDim) {
      sliceOp(reshapeOp.getSrc(), offset, mappings, reverseMappings,
              partitionScheme);
      newOp = cloneAndSetResultType(op);
    } else {
      builder.setInsertionPoint(op);
      auto clonedReshape = builder.clone(*op, mappings);
      setWSPartitionIds(clonedReshape, sliceTaskIds);
      mappings.map(op, clonedReshape);
      reverseMappings.map(clonedReshape, op);

      auto result = op->getResult(0);
      auto type = cast<MemDescType>(result.getType());
      SmallVector<int64_t> shape{type.getShape().begin(),
                                 type.getShape().end()};
      assert(dim < shape.size() &&
             "memdesc_reshape partition dim out of bounds");
      int sliceSize = shape[dim] / numOfPartitions;
      shape[dim] = sliceSize;
      auto slicedType =
          MemDescType::get(shape, type.getElementType(), type.getEncoding(),
                           type.getMemorySpace(), type.getMutableMemory());
      SmallVector<int32_t> offsets(shape.size(), 0);
      offsets[dim] = offset * sliceSize;
      auto subsliceOp = builder.createWithPartitionIds<MemDescSubsliceOp>(
          op->getLoc(), slicedType, clonedReshape->getResult(0), offsets);
      mappings.map(result, subsliceOp.getResult());
      reverseMappings.map(subsliceOp.getResult(), result);
      newOp = subsliceOp;
    }
  } else if (isDotOrMMAv5Op(op)) {
    assert(partitionScheme.dotPartitionOperand.contains(op) &&
           "no operand info");
    unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
    LDBG("slicing operand " << opndIndx << "\n");
    sliceOp(op->getOperand(opndIndx), offset, mappings, reverseMappings,
            partitionScheme);
    if (Value scale = getMMAv5ScaleOperand(op, opndIndx)) {
      LDBG("slicing scale operand " << opndIndx << "\n");
      sliceOp(scale, offset, mappings, reverseMappings, partitionScheme);
    }
    if (dim == 0 && opndIndx == 1 || dim == 1 && opndIndx == 0) {
      // slice the other operand
      unsigned otherOpndIndx = 1 - opndIndx;
      LDBG("slicing operand " << otherOpndIndx << "\n");
      sliceOp(op->getOperand(otherOpndIndx), offset, mappings, reverseMappings,
              partitionScheme);
      if (Value scale = getMMAv5ScaleOperand(op, otherOpndIndx)) {
        LDBG("slicing scale operand " << otherOpndIndx << "\n");
        sliceOp(scale, offset, mappings, reverseMappings, partitionScheme);
      }
    }
    // Handle accumulator
    Value accumulator = getDotAccumulatorInput(op);
    LDBG("slicing accumulator\n");
    sliceOp(accumulator, offset, mappings, reverseMappings, partitionScheme);

    // Handle token
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      if (auto token = mmaOp.getAccDep()) {
        LDBG("slicing token \n");
        sliceOp(token, offset, mappings, reverseMappings, partitionScheme);
      }
    }

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
      auto newInitArgOp =
          sliceOp(initArg, offset, mappings, reverseMappings, partitionScheme);
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
    auto newForOp = builder.createWithPartitionIds<scf::ForOp>(
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
    auto newThenYieldOp = sliceOp(thenYieldOp, offset, mappings,
                                  reverseMappings, partitionScheme);
    sliceOp(elseYieldOp, offset, mappings, reverseMappings, partitionScheme);
    assert(newThenYieldOp->getNumOperands() > ifOp->getNumResults() &&
           "no need to slice if op");
    // Clone ifOp with updated results but re-use the original regions.
    builder.setInsertionPoint(op);
    SmallVector<Type, 4> newResultTypes;
    for (auto thenResult : thenYieldOp.getResults()) {
      newResultTypes.push_back(thenResult.getType());
    }
    auto newIfOp = scf::IfOp::create(builder, ifOp.getLoc(), newResultTypes,
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
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    SmallVector<Value> newLoopArgs(whileOp.getInits().begin(),
                                   whileOp.getInits().end());
    struct NewWhileSlot {
      unsigned slotIdx;
      BlockArgument oldBeforeArg;
      BlockArgument oldAfterArg;
      OpResult oldResult;
    };
    SmallVector<NewWhileSlot> newSlots;

    for (auto [slotIdx, initArg] : llvm::enumerate(whileOp.getInits())) {
      auto newInitArgOp =
          sliceOp(initArg, offset, mappings, reverseMappings, partitionScheme);
      Value newInitArg = mappings.lookupOrNull(initArg);
      if (!newInitArg && isa<BlockArgument>(initArg)) {
        auto bbArg = cast<BlockArgument>(initArg);
        Block *parentBlock = bbArg.getOwner();
        unsigned argIndex = bbArg.getArgNumber();
        Region *parentRegion = parentBlock->getParent();
        Region &newParentRegion =
            newInitArgOp->getRegion(parentRegion->getRegionNumber());
        newInitArg = newParentRegion.front().getArgument(argIndex);
      }
      if (!newInitArg)
        continue;

      assert(newInitArg != initArg && "value not sliced");
      newLoopArgs.push_back(newInitArg);
      auto slot = getLoopCarriedSlot(
          cast<LoopLikeOpInterface>(whileOp.getOperation()), slotIdx);
      newSlots.push_back({static_cast<unsigned>(slotIdx), slot.iterArg,
                          slot.afterArg, slot.result});
    }

    builder.setInsertionPoint(op);
    SmallVector<Type> resultTypes(whileOp->getResultTypes());
    for (Value newArg :
         llvm::drop_begin(newLoopArgs, whileOp.getInits().size()))
      resultTypes.push_back(newArg.getType());
    auto newWhileOp = builder.createWithPartitionIds<scf::WhileOp>(
        whileOp.getLoc(), resultTypes, newLoopArgs);
    newWhileOp->setAttrs(whileOp->getAttrs());
    partitionScheme.ops.insert(newWhileOp);
    newOp = newWhileOp;

    SmallVector<Type> beforeArgTypes;
    for (Value arg : newLoopArgs)
      beforeArgTypes.push_back(arg.getType());
    SmallVector<Location> beforeLocs(beforeArgTypes.size(), whileOp.getLoc());
    SmallVector<Location> afterLocs(resultTypes.size(), whileOp.getLoc());
    if (newWhileOp.getBefore().empty())
      newWhileOp.getBefore().emplaceBlock();
    if (newWhileOp.getAfter().empty())
      newWhileOp.getAfter().emplaceBlock();
    newWhileOp.getBeforeBody()->addArguments(beforeArgTypes, beforeLocs);
    newWhileOp.getAfterBody()->addArguments(resultTypes, afterLocs);
    newWhileOp.getBeforeBody()->getOperations().splice(
        newWhileOp.getBeforeBody()->getOperations().begin(),
        whileOp.getBeforeBody()->getOperations());
    newWhileOp.getAfterBody()->getOperations().splice(
        newWhileOp.getAfterBody()->getOperations().begin(),
        whileOp.getAfterBody()->getOperations());

    for (auto [oldArg, newArg] :
         llvm::zip(whileOp.getBeforeArguments(),
                   newWhileOp.getBeforeArguments().take_front(
                       whileOp.getBeforeArguments().size())))
      oldArg.replaceAllUsesWith(newArg);
    for (auto [oldArg, newArg] :
         llvm::zip(whileOp.getAfterArguments(),
                   newWhileOp.getAfterArguments().take_front(
                       whileOp.getAfterArguments().size())))
      oldArg.replaceAllUsesWith(newArg);

    for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
      whileOp.getResult(i).replaceAllUsesWith(newWhileOp.getResult(i));
    whileOp->setAttr("to_be_removed", builder.getUnitAttr());

    unsigned appendedIdx = 0;
    unsigned oldInitCount = whileOp.getInits().size();
    unsigned oldResultCount = whileOp.getNumResults();
    for (const NewWhileSlot &slot : newSlots) {
      Value newBeforeArg =
          newWhileOp.getBeforeArguments()[oldInitCount + appendedIdx];
      newWhileOp.getConditionOp().getArgsMutable().append(newBeforeArg);
      mappings.map(slot.oldBeforeArg, newBeforeArg);
      reverseMappings.map(newBeforeArg, slot.oldBeforeArg);

      if (slot.oldAfterArg) {
        Value newAfterArg =
            newWhileOp.getAfterArguments()[oldResultCount + appendedIdx];
        mappings.map(slot.oldAfterArg, newAfterArg);
        mappings.map(
            newWhileOp.getAfterArguments()[slot.oldAfterArg.getArgNumber()],
            newAfterArg);
        reverseMappings.map(newAfterArg, slot.oldAfterArg);
      }
      if (slot.oldResult) {
        Value newResult = newWhileOp.getResult(oldResultCount + appendedIdx);
        mappings.map(slot.oldResult, newResult);
        mappings.map(newWhileOp.getResult(slot.oldResult.getResultNumber()),
                     newResult);
        reverseMappings.map(newResult, slot.oldResult);
      }
      ++appendedIdx;
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    // For ForOp yields, only append sliced yield operands for positions where
    // the parent ForOp actually added a new init arg. The ForOp slicing records
    // new args via mappings on ForOp results. If a yield value was mapped
    // (sliced inside the loop) but the corresponding ForOp init arg was NOT
    // mapped (not sliced outside the loop), appending would create a
    // type/ordering mismatch between init args and yield operands.
    auto parentForOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    auto parentWhileOp = dyn_cast<scf::WhileOp>(yieldOp->getParentOp());
    int num = yieldOp.getNumOperands();
    for (int i = 0; i < num; i++) {
      auto operand = yieldOp.getOperand(i);
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
      auto newV = mappings.lookupOrNull(operand);
      if (parentWhileOp) {
        // scf.while: the while slicing appends a loop-carried arg for every
        // sliced init, and the after-yield operand k feeds before-arg k 1:1,
        // so a sliced yield operand always has a matching appended before-arg;
        // append unconditionally when the operand was sliced. (Slicing an
        // after-region-only value whose init was not sliced would break the
        // 1:1 alignment and would need a guard analogous to the for path.)
        if (newV)
          yieldOp->insertOperands(yieldOp->getNumOperands(), newV);
        continue;
      }
      if (!parentForOp) {
        if (newV)
          yieldOp->insertOperands(yieldOp->getNumOperands(), newV);
        continue;
      }
      // scf.for: the ForOp slicing appended one new result/iter-arg for every
      // sliced init arg (its result i is now mapped). Keep results and yield
      // operands 1:1 so the loop stays well-formed (numResults ==
      // numYieldOperands): append exactly one operand per new iter-arg. Prefer
      // the sliced yield value; if this slot's yield operand was NOT itself
      // sliced (e.g. a token/predicate carried across a short loop), carry the
      // appended partition-copy region iter-arg through instead. Without this,
      // a sliced init with an unsliced yield leaves the ForOp with more results
      // than yield operands, which later crashes ForOpDeadArgElimination.
      if (mappings.lookupOrNull(parentForOp.getResult(i))) {
        Value yieldVal =
            newV ? newV
                 : mappings.lookupOrNull(parentForOp.getRegionIterArg(i));
        assert(yieldVal && "no yield value for sliced ForOp iter-arg");
        yieldOp->insertOperands(yieldOp->getNumOperands(), yieldVal);
      }
    }
    newOp = op;
  } else if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
    int num = condOp.getArgs().size();
    for (int i = 0; i < num; i++) {
      auto operand = condOp.getArgs()[i];
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
      if (auto newV = mappings.lookupOrNull(operand))
        condOp.getArgsMutable().append(newV);
    }
    newOp = op;
  } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    assert(reduceOp.getAxis() != dim &&
           "reduce should not happen on the partitioned dimension");
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
    // recursively set partition ids for child ops
    newOp->walk(
        [&](Operation *childOp) { setWSPartitionIds(childOp, sliceTaskIds); });
  } else if (isa<MapElementwiseOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
    // recursively set partition ids for child ops
    newOp->walk(
        [&](Operation *childOp) { setWSPartitionIds(childOp, sliceTaskIds); });
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

static Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                          IRMapping &reverseMappings,
                          DataPartitionScheme &partitionScheme) {
  if (auto op = v.getDefiningOp()) {
    return sliceOp(op, offset, mappings, reverseMappings, partitionScheme);
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    if (isa<triton::FuncOp>(bbAargOwner)) {
      // Host-side TMA func arg: type updated in post-processing.
      return bbAargOwner;
    }
    return sliceOp(bbAargOwner, offset, mappings, reverseMappings,
                   partitionScheme);
  }
}

static bool doDeepCleanup(triton::FuncOp &funcOp,
                          DataPartitionScheme &partitionScheme) {
  SmallVector<Operation *> opsToDelete;
  DenseSet<Operation *> opsCanBeTriviallyDead;

  do {
    opsToDelete.clear();
    opsCanBeTriviallyDead.clear();

    // Identify root ops that are not used so to be deleted.
    funcOp.walk([&](Operation *op) {
      if (isa<scf::YieldOp, scf::ConditionOp>(op))
        return;
      if (!partitionScheme.ops.contains(op))
        return;

      // Ignore the side effect of ops that are already sliced. The
      // resulting ops preserve the side effect.
      if (!isMemoryEffectFree(op))
        opsCanBeTriviallyDead.insert(op);

      // Don't delete region ops directly. After slicing, the only
      // region ops remaining in the partition scheme are the final sliced
      // versions (originals were erased via "to_be_removed"). These contain
      // the partitioned ops and must be preserved. Let the canonicalization
      // patterns handle dead argument elimination instead.
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op))
        return;

      bool notUsed = true;
      for (auto result : op->getResults()) {
        if (!result.getUsers().empty()) {
          notUsed = false;
          break;
        }
      }
      if (notUsed)
        opsToDelete.push_back(op);
    });

    LLVM_DEBUG({
      LDBG("opsToDelete:\n");
      for (auto op : opsToDelete) {
        LDBG("op: ");
        op->dump();
      }
      LDBG("\n");
    });

    if (opsToDelete.empty())
      return true;

    // Delete root ops.
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }

    LLVM_DEBUG({
      LDBG("prior to loop arg deletion:");
      funcOp.dump();
    });

    // delete block arguments
    RewritePatternSet cleanUpPatterns(funcOp.getContext());
    populateForOpDeadArgumentElimination(cleanUpPatterns,
                                         opsCanBeTriviallyDead);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns,
                                            funcOp.getContext());
    scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns,
                                           funcOp.getContext());
    scf::WhileOp::getCanonicalizationPatterns(cleanUpPatterns,
                                              funcOp.getContext());
    if (applyPatternsGreedily(funcOp, std::move(cleanUpPatterns)).failed()) {
      return false;
    }
  } while (!opsToDelete.empty());
  return true;
}

/// Check if a value is effectively a splat constant by tracing through
/// element-preserving ops (convert_layout, truncf, extf, split). Returns the
/// splat element Attribute in the target value's element type, or nullopt.
static std::optional<Attribute> getEffectiveSplatAttr(Value v) {
  // Direct constant.
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    auto valAttr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
    if (valAttr && valAttr.isSplat())
      return valAttr.getSplatValue<Attribute>();
    return std::nullopt;
  }
  // convert_layout preserves values and element type.
  if (auto convertOp = v.getDefiningOp<ConvertLayoutOp>())
    return getEffectiveSplatAttr(convertOp.getSrc());
  // truncf preserves splatness; convert the element value.
  if (auto truncOp = v.getDefiningOp<arith::TruncFOp>()) {
    auto srcAttr = getEffectiveSplatAttr(truncOp.getIn());
    if (!srcAttr)
      return std::nullopt;
    auto srcFloat = dyn_cast<FloatAttr>(*srcAttr);
    if (!srcFloat)
      return std::nullopt;
    auto dstElemType = cast<FloatType>(
        cast<RankedTensorType>(truncOp.getType()).getElementType());
    bool losesInfo;
    APFloat trunced = srcFloat.getValue();
    trunced.convert(dstElemType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &losesInfo);
    return FloatAttr::get(dstElemType, trunced);
  }
  // extf preserves splatness; convert the element value.
  if (auto extOp = v.getDefiningOp<arith::ExtFOp>()) {
    auto srcAttr = getEffectiveSplatAttr(extOp.getIn());
    if (!srcAttr)
      return std::nullopt;
    auto srcFloat = dyn_cast<FloatAttr>(*srcAttr);
    if (!srcFloat)
      return std::nullopt;
    auto dstElemType = cast<FloatType>(
        cast<RankedTensorType>(extOp.getType()).getElementType());
    bool losesInfo;
    APFloat extended = srcFloat.getValue();
    extended.convert(dstElemType.getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);
    return FloatAttr::get(dstElemType, extended);
  }
  // split preserves values and element type.
  if (auto splitOp = v.getDefiningOp<SplitOp>())
    return getEffectiveSplatAttr(splitOp.getSrc());
  // reshape preserves splatness and element type.
  if (auto reshapeOp = v.getDefiningOp<ReshapeOp>())
    return getEffectiveSplatAttr(reshapeOp.getSrc());
  // trans/permute preserves splatness and element type.
  if (auto transOp = v.getDefiningOp<TransOp>())
    return getEffectiveSplatAttr(transOp.getSrc());
  return std::nullopt;
}

/// Reorder load ops within each basic block so that loads are sorted by the
/// position of their earliest use in the same block. This ensures that after
/// data partitioning, loads are placed closer to their first consumer.
///
/// For GEMM, where A is partitioned into A0, A1 and B is shared, this produces
/// the order: A0, A1, B (matching the use pattern Mma(A0, B), Mma(A1, B)).
///
/// TODO: We may be able to reorder other operations, but this is only
/// implemented for loads for now.
static void reorderLoadsToFirstUse(triton::FuncOp &funcOp) {
  funcOp.walk([](Block *block) {
    // Collect load ops in block order.
    SmallVector<Operation *> loads;
    for (auto &op : block->getOperations()) {
      if (isa<DescriptorLoadOp, LoadOp>(&op))
        loads.push_back(&op);
    }

    if (loads.size() <= 1)
      return;

    // Build position map for all ops in the block.
    DenseMap<Operation *, unsigned> opPositions;
    unsigned pos = 0;
    for (auto &op : block->getOperations())
      opPositions[&op] = pos++;

    // For each load, find the position of its earliest use in the same block.
    auto getFirstUsePosition = [&](Operation *loadOp) -> unsigned {
      unsigned earliest = UINT_MAX;
      for (auto result : loadOp->getResults()) {
        for (auto *user : result.getUsers()) {
          if (user->getBlock() == block) {
            earliest = std::min(earliest, opPositions[user]);
          }
        }
      }
      return earliest;
    };

    // Compute first-use positions and stable sort.
    SmallVector<std::pair<Operation *, unsigned>> loadWithUse;
    for (auto *load : loads)
      loadWithUse.push_back({load, getFirstUsePosition(load)});

    llvm::stable_sort(loadWithUse, [](const auto &a, const auto &b) {
      return a.second < b.second;
    });

    // Reorder loads in sorted order. Each load is placed after the previous
    // sorted load, but never before any of its own operands (to preserve SSA
    // dominance).
    for (size_t i = 1; i < loadWithUse.size(); i++) {
      auto *prevLoad = loadWithUse[i - 1].first;
      auto *curLoad = loadWithUse[i].first;

      // Target position: right after the previous load in sorted order.
      Operation *target = prevLoad->getNextNode();

      // Check that all operands of curLoad dominate the target position.
      bool canMove = true;
      for (Value operand : curLoad->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == block && !defOp->isBeforeInBlock(target)) {
            canMove = false;
            break;
          }
        }
      }

      if (canMove && curLoad != target)
        curLoad->moveBefore(target);
    }
  });
}

bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups) {
  DataPartitionScheme partitionScheme;
  partitionScheme.numPartitions = numConsumerGroups;
  if (!computePartitionScheme(funcOp, partitionScheme)) {
    if (numConsumerGroups > 1) {
      LDBG("computePartitionScheme failed when requested");
      return false;
    }
    return true;
  }
  if (partitionScheme.ops.empty())
    return true;

  // Bail out if a TensorDescType func arg is used as a ForOp init arg.
  // This case requires extra handling to update ForOp iter arg types
  // consistently, deferred to a follow-up.
  for (auto &[argIndex, dim] : partitionScheme.funcArgPartitionDims) {
    auto bbArg = funcOp.getArgument(argIndex);
    for (Operation *user : bbArg.getUsers()) {
      if (auto loop = dyn_cast<LoopLikeOpInterface>(user)) {
        for (Value initArg : loop.getInits()) {
          if (initArg == bbArg) {
            LDBG("TensorDescType func arg " << argIndex
                                            << " used as loop init arg; "
                                               "not supported yet");
            return false;
          }
        }
      }
    }
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
    IRMapping mappings, reverseMappings;
    LDBG("partitioning op for task " << i + 1 << ":\n");
    int numOps = partitionScheme.ops.size();
    for (int j = 0; j < numOps; j++) {
      auto op = partitionScheme.ops[j];
      sliceOp(op, i, mappings, reverseMappings, partitionScheme);
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
    LDBG("prior to final cleanup:");
    funcOp.dump();
  });

  // Make sure original ops are not used
  if (!doDeepCleanup(funcOp, partitionScheme)) {
    LDBG("final cleanup failed");
    return false;
  }

  // Make sure original ops are not used
  LLVM_DEBUG({
    LDBG("after partition");
    funcOp.dump();
    LDBG("\n");
  });

  fixTaskId(funcOp);

  // Handle unpartitioned descriptor_store ops that reference func args we're
  // about to modify. This can happen when there are multiple store paths and
  // only one of them includes the dot. For example, with FLATTEN=True the
  // persistent GEMM kernel creates an if condition when k_tiles==0 that
  // is just a store.
  for (auto &[argIndex, dim] : partitionScheme.funcArgPartitionDims) {
    auto &entryBlock = funcOp.getBlocks().front();
    auto bbArg = entryBlock.getArgument(argIndex);
    auto descType = cast<TensorDescType>(bbArg.getType());
    auto blockType = descType.getBlockType();
    int64_t slicedSize =
        blockType.getShape()[dim] / partitionScheme.numPartitions;

    SmallVector<DescriptorStoreOp> unpartitionedStores;
    for (Operation *user : bbArg.getUsers()) {
      if (auto descStoreOp = dyn_cast<DescriptorStoreOp>(user)) {
        if (!partitionScheme.isPartitioned(descStoreOp)) {
          // Skip stores whose source is already the sliced size — these
          // were created by the partition pass itself.
          auto srcType = cast<RankedTensorType>(descStoreOp.getSrc().getType());
          if (srcType.getShape()[dim] == slicedSize)
            continue;
          unpartitionedStores.push_back(descStoreOp);
        }
      }
    }

    for (auto descStoreOp : unpartitionedStores) {
      OpBuilder builder(descStoreOp);
      Value src = descStoreOp.getSrc();
      auto srcType = cast<RankedTensorType>(src.getType());
      SmallVector<int64_t> srcShape(srcType.getShape());

      // Compute the sliced source type.
      SmallVector<int64_t> slicedShape(srcShape);
      slicedShape[dim] = slicedSize;

      // Create sliced source values — one per partition.
      SmallVector<Value> slicedSrcs;
      if (auto splatAttr = getEffectiveSplatAttr(src)) {
        // Splat constants: create a new splat with the sliced shape.
        auto slicedValType = RankedTensorType::get(
            slicedShape, srcType.getElementType(), srcType.getEncoding());
        auto slicedValAttr = DenseElementsAttr::get(slicedValType, *splatAttr);
        Value splatConst = arith::ConstantOp::create(
            builder, descStoreOp.getLoc(), slicedValAttr);
        for (int i = 0; i < partitionScheme.numPartitions; i++)
          slicedSrcs.push_back(splatConst);
      } else if (partitionScheme.numPartitions == 2) {
        // Non-splat source with 2 partitions: use reshape + trans + split.
        //
        // For a source tensor<S0 x S1 x ... x f16> partitioned along dim:
        //   1. Reshape: replace S[dim] with [2, S[dim]/2]
        //      e.g. tensor<256x128> → tensor<2x128x128> (dim=0)
        //   2. Trans: move the size-2 dimension to the last position
        //      e.g. tensor<2x128x128> → tensor<128x128x2>
        //   3. Split: split along the last dimension (size 2)
        //      e.g. tensor<128x128x2> → tensor<128x128>, tensor<128x128>
        auto loc = descStoreOp.getLoc();

        // Build the reshaped shape: insert [2, S[dim]/2] at position dim.
        SmallVector<int64_t> reshapedShape;
        for (size_t d = 0; d < srcShape.size(); d++) {
          if (d == (size_t)dim) {
            reshapedShape.push_back(2);
            reshapedShape.push_back(srcShape[d] / 2);
          } else {
            reshapedShape.push_back(srcShape[d]);
          }
        }

        auto reshapeOp = ReshapeOp::create(builder, loc, reshapedShape, src,
                                           /*allowReorder=*/false);

        // Build trans order: move dim (the size-2 position) to last.
        int rank = reshapedShape.size();
        SmallVector<int32_t> transOrder;
        for (int d = 0; d < rank; d++) {
          if (d != dim)
            transOrder.push_back(d);
        }
        transOrder.push_back(dim);

        auto transOp =
            TransOp::create(builder, loc, reshapeOp.getResult(), transOrder);

        auto splitOp = SplitOp::create(builder, loc, transOp.getResult());
        slicedSrcs.push_back(splitOp.getOutLHS());
        slicedSrcs.push_back(splitOp.getOutRHS());
      } else {
        LDBG("Cannot slice non-splat source of unpartitioned descriptor_store");
        return false;
      }

      // Create numPartitions replacement stores with adjusted coordinates.
      for (int i = 0; i < partitionScheme.numPartitions; i++) {
        SmallVector<Value> indices(descStoreOp.getIndices());
        if (i > 0) {
          Value offset = arith::ConstantIntOp::create(
              builder, descStoreOp.getLoc(), i * slicedSize, 32);
          indices[dim] = arith::AddIOp::create(builder, descStoreOp.getLoc(),
                                               indices[dim], offset);
        }
        DescriptorStoreOp::create(builder, descStoreOp.getLoc(),
                                  descStoreOp.getDesc(), slicedSrcs[i],
                                  indices);
      }

      descStoreOp.erase();
    }
  }

  // Handle unpartitioned descriptor_load ops similarly. After updating the
  // func arg type, any remaining full-sized load would have a type mismatch.
  // Replace each with numPartitions sliced loads + join + trans + reshape to
  // reconstruct the original full-sized tensor for downstream users.
  for (auto &[argIndex, dim] : partitionScheme.funcArgPartitionDims) {
    auto &entryBlock = funcOp.getBlocks().front();
    auto bbArg = entryBlock.getArgument(argIndex);
    auto descType = cast<TensorDescType>(bbArg.getType());
    auto blockType = descType.getBlockType();
    int64_t slicedSize =
        blockType.getShape()[dim] / partitionScheme.numPartitions;

    SmallVector<DescriptorLoadOp> unpartitionedLoads;
    for (Operation *user : bbArg.getUsers()) {
      if (auto descLoadOp = dyn_cast<DescriptorLoadOp>(user)) {
        if (!partitionScheme.isPartitioned(descLoadOp)) {
          auto resultType =
              cast<RankedTensorType>(descLoadOp.getResult().getType());
          if (resultType.getShape()[dim] == slicedSize)
            continue;
          unpartitionedLoads.push_back(descLoadOp);
        }
      }
    }

    for (auto descLoadOp : unpartitionedLoads) {
      if (partitionScheme.numPartitions != 2) {
        LDBG("Cannot reconstruct non-splat unpartitioned descriptor_load "
             "with numPartitions != 2");
        return false;
      }
      OpBuilder builder(descLoadOp);
      auto loc = descLoadOp.getLoc();
      auto resultType =
          cast<RankedTensorType>(descLoadOp.getResult().getType());
      SmallVector<int64_t> resultShape(resultType.getShape());

      // Compute the sliced result type.
      SmallVector<int64_t> slicedShape(resultShape);
      slicedShape[dim] = slicedSize;
      auto slicedResultType = RankedTensorType::get(
          slicedShape, resultType.getElementType(), resultType.getEncoding());

      // Create sliced loads.
      SmallVector<Value> slicedLoads;
      for (int i = 0; i < partitionScheme.numPartitions; i++) {
        SmallVector<Value> indices(descLoadOp.getIndices());
        if (i > 0) {
          Value offset =
              arith::ConstantIntOp::create(builder, loc, i * slicedSize, 32);
          indices[dim] =
              arith::AddIOp::create(builder, loc, indices[dim], offset);
        }
        auto slicedLoad = DescriptorLoadOp::create(
            builder, loc, slicedResultType, descLoadOp.getDesc(), indices);
        slicedLoads.push_back(slicedLoad.getResult());
      }

      // Reconstruct the full tensor: join + trans + reshape.
      // join: tensor<S0x...x(S[dim]/2)x...> x2 →
      // tensor<S0x...x(S[dim]/2)x...x2>
      auto joinOp =
          JoinOp::create(builder, loc, slicedLoads[0], slicedLoads[1]);

      // trans: move the last dim (size 2) to position dim.
      int rank = resultShape.size() + 1; // after join, rank increased by 1
      SmallVector<int32_t> transOrder;
      for (int d = 0; d < rank; d++) {
        if (d == dim)
          transOrder.push_back(rank - 1); // insert the size-2 dim here
        if (d < rank - 1)
          transOrder.push_back(d);
      }

      auto transOp =
          TransOp::create(builder, loc, joinOp.getResult(), transOrder);

      // reshape: merge the partition dim back.
      // e.g. tensor<2x128x128> → tensor<256x128>
      auto reshapeOp =
          ReshapeOp::create(builder, loc, resultShape, transOp.getResult(),
                            /*allowReorder=*/false);

      // TODO: Patch with open PR?
      // The reshape may produce a different encoding (e.g. #linear) than
      // the original descriptor_load result (#blocked).  Insert a
      // convert_layout to restore the original encoding so that
      // downstream elementwise users (arith.extf, etc.) remain valid.
      Value replacement = reshapeOp.getResult();
      auto reshapeType = cast<RankedTensorType>(replacement.getType());
      if (reshapeType.getEncoding() != resultType.getEncoding()) {
        auto cvt =
            ConvertLayoutOp::create(builder, loc, resultType, replacement);
        replacement = cvt.getResult();
      }
      descLoadOp.getResult().replaceAllUsesWith(replacement);
      descLoadOp.erase();
    }
  }

  // Update function argument types for host-side TMA descriptors.
  if (!partitionScheme.funcArgPartitionDims.empty()) {
    auto &entryBlock = funcOp.getBlocks().front();
    for (auto &[argIndex, dim] : partitionScheme.funcArgPartitionDims) {
      auto bbArg = entryBlock.getArgument(argIndex);
      auto descType = cast<TensorDescType>(bbArg.getType());
      SmallVector<int64_t> shape(descType.getShape());
      shape[dim] /= partitionScheme.numPartitions;
      bbArg.setType(TensorDescType::get(shape, descType.getElementType(),
                                        descType.getSharedLayout()));
    }
    // Update FuncOp signature to match.
    SmallVector<Type> argTys(entryBlock.getArgumentTypes());
    funcOp.setFunctionType(FunctionType::get(
        funcOp.getContext(), argTys, funcOp.getFunctionType().getResults()));
  }

  // Reorder loads so they are closer to their first use. After data
  // partitioning, duplicated loads may end up far from their consumers.
  reorderLoadsToFirstUse(funcOp);

  return true;
}

#define GEN_PASS_DEF_NVGPUWSDATAPARTITION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWSDataPartitionPass
    : public impl::NVGPUWSDataPartitionBase<NVGPUWSDataPartitionPass> {
public:
  using impl::NVGPUWSDataPartitionBase<
      NVGPUWSDataPartitionPass>::NVGPUWSDataPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    std::optional<uint32_t> dataPartitonFactor;
    SmallVector<LoopLikeOpInterface> loops;
    funcOp->walk([&](LoopLikeOpInterface loop) {
      if (loop->hasAttr(mlir::triton::kWarpSpecializeAttrName))
        loops.push_back(loop);
      if (auto factor =
              loop->getAttrOfType<IntegerAttr>(kDataPartitionAttrName)) {
        assert((!dataPartitonFactor || factor.getInt() == dataPartitonFactor) &&
               "data partition factor mismatch");
        dataPartitonFactor = factor.getInt();
      }
    });
    if (!dataPartitonFactor && numWarpGroups <= 2)
      return;

    if (!dataPartitonFactor)
      dataPartitonFactor = numWarpGroups - 1;
    if (dataPartitonFactor < 2)
      return;
    if (!doDataPartition(funcOp, *dataPartitonFactor))
      signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
