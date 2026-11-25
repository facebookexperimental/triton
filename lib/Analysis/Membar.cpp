#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <limits>
#include <optional>

#include <algorithm>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

/// Given a value produced by memdesc_index, possibly wrapped in transparent
/// memdesc views, narrow the parent buffer's interval to the sub-range actually
/// accessed. memdesc_index selects a contiguous slice along the leading
/// dimension, so a compile-time constant index identifies an exact byte range.
/// Nested memdesc_index operations are verifier-invalid.
static Interval<size_t> narrowIntervalForSubview(Value value,
                                                 Interval<size_t> interval) {
  triton::gpu::MemDescIndexOp indexOp;
  while (Operation *defOp = value.getDefiningOp()) {
    if ((indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(defOp)))
      break;
    if (!defOp->hasTrait<OpTrait::MemDescViewTrait>())
      return interval;
    value = defOp->getOperand(0);
  }
  if (!indexOp)
    return interval;

  APInt indexVal;
  if (!matchPattern(indexOp.getIndex(), m_ConstantInt(&indexVal)))
    return interval;

  auto parentType = cast<triton::gpu::MemDescType>(indexOp.getSrc().getType());
  int64_t dim0 = parentType.getShape()[0];
  size_t totalSize = interval.end() - interval.start();
  // Padded stages need the validated stride and extent from getStageGeometry;
  // equal fractions of the parent interval do not describe their byte ranges.
  if (dim0 <= 0 || indexVal.isNegative() || indexVal.getZExtValue() >= dim0 ||
      triton::gpu::isPaddedEncoding(parentType.getEncoding()) ||
      totalSize % dim0 != 0)
    return interval;

  size_t stride = totalSize / dim0;
  size_t newStart = interval.start() + indexVal.getSExtValue() * stride;
  return Interval<size_t>(newStart, newStart + stride);
}

static bool isConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) && constant == expected;
}

// Track both a residue relative to the induction value and signed i32 bounds.
// Congruence alone is insufficient: signed remainder may be negative, and an
// overflowing addition need not preserve a residue for a non-power-of-two ring.
struct StageExpression {
  unsigned offset;
  int64_t min;
  int64_t max;
};

static unsigned stageResidue(int64_t value, unsigned numStages) {
  int64_t residue = value % numStages;
  return residue < 0 ? residue + numStages : residue;
}

static std::optional<StageExpression>
getStageExpression(Value value, Value induction, unsigned numStages,
                   StageExpression inductionRange, unsigned depth = 0) {
  if (depth > 8 || value.getType() != induction.getType())
    return std::nullopt;
  if (value == induction)
    return inductionRange;
  auto recurse = [&](Value operand) {
    return getStageExpression(operand, induction, numStages, inductionRange,
                              depth + 1);
  };
  if (auto rem = value.getDefiningOp<arith::RemSIOp>()) {
    if (!isConstantInt(rem.getRhs(), numStages))
      return std::nullopt;
    auto source = recurse(rem.getLhs());
    if (source && source->min >= 0)
      return StageExpression{source->offset, 0, numStages - 1};
    return std::nullopt;
  }
  if (auto bitAnd = value.getDefiningOp<arith::AndIOp>()) {
    if (!llvm::isPowerOf2_32(numStages))
      return std::nullopt;
    Value source;
    if (isConstantInt(bitAnd.getRhs(), numStages - 1))
      source = bitAnd.getLhs();
    else if (isConstantInt(bitAnd.getLhs(), numStages - 1))
      source = bitAnd.getRhs();
    if (source)
      if (auto expression = recurse(source))
        return StageExpression{expression->offset, 0, numStages - 1};
    return std::nullopt;
  }

  Value source;
  APInt constant;
  int64_t delta = 0;
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (matchPattern(add.getRhs(), m_ConstantInt(&constant)))
      source = add.getLhs();
    else if (matchPattern(add.getLhs(), m_ConstantInt(&constant)))
      source = add.getRhs();
    if (source)
      delta = constant.getSExtValue();
  } else if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    if (matchPattern(sub.getRhs(), m_ConstantInt(&constant))) {
      source = sub.getLhs();
      delta = -constant.getSExtValue();
    }
  }
  if (source) {
    auto expression = recurse(source);
    if (!expression ||
        expression->min + delta < std::numeric_limits<int32_t>::min() ||
        expression->max + delta > std::numeric_limits<int32_t>::max())
      return std::nullopt;
    return StageExpression{
        stageResidue(static_cast<int64_t>(expression->offset) + delta,
                     numStages),
        expression->min + delta, expression->max + delta};
  }

  // For two stages, 1-x and x XOR 1 are also a unit residue offset. This
  // equivalence does not hold for arbitrary larger stage counts.
  if (numStages != 2)
    return std::nullopt;
  Value complemented;
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    if (isConstantInt(sub.getLhs(), 1))
      complemented = sub.getRhs();
  } else if (auto bitXor = value.getDefiningOp<arith::XOrIOp>()) {
    if (isConstantInt(bitXor.getLhs(), 1))
      complemented = bitXor.getRhs();
    else if (isConstantInt(bitXor.getRhs(), 1))
      complemented = bitXor.getLhs();
  }
  if (complemented)
    if (auto expression = recurse(complemented))
      if (expression->min >= 0 && expression->max <= 1)
        return StageExpression{expression->offset ^ 1, 0, 1};
  return std::nullopt;
}

// Only discoverStageLoops supplies an induction value. Recover its proven
// initial and exclusive upper bounds without assuming a particular residue.
static std::optional<unsigned> getStageOffset(Value value, Value induction,
                                              unsigned numStages) {
  auto argument = dyn_cast_or_null<BlockArgument>(induction);
  if (!argument)
    return std::nullopt;
  Block *header = argument.getOwner();
  auto branch = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  auto compare = branch ? branch.getCondition().getDefiningOp<arith::CmpIOp>()
                        : arith::CmpIOp();
  if (!compare || compare.getLhs() != induction ||
      compare.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;
  std::optional<int64_t> initial;
  for (Block *predecessor : header->getPredecessors()) {
    auto entry = dyn_cast<cf::BranchOp>(predecessor->getTerminator());
    APInt constant;
    if (entry && matchPattern(entry.getDestOperands()[argument.getArgNumber()],
                              m_ConstantInt(&constant)))
      initial = constant.getSExtValue();
  }
  int64_t max = std::numeric_limits<int32_t>::max() - 1;
  APInt upper;
  if (matchPattern(compare.getRhs(), m_ConstantInt(&upper)))
    max = upper.getSExtValue() - 1;
  if (!initial || *initial < 0 || *initial > max)
    return std::nullopt;
  auto expression =
      getStageExpression(value, induction, numStages, {0, *initial, max});
  if (!expression || expression->min < 0 || expression->max >= numStages)
    return std::nullopt;
  return expression->offset;
}

// Limit symbolic slices to a direct, stable allocation dominating the header.
// Descriptor block arguments, selects and reinterpretations are intentionally
// not traced to an underlying allocation.
static bool isStableStageParent(Value parent, Value induction) {
  if (!parent || !induction)
    return false;
  auto argument = dyn_cast<BlockArgument>(induction);
  auto alloc = parent.getDefiningOp<triton::gpu::LocalAllocOp>();
  if (!argument || !alloc)
    return false;
  Block *header = argument.getOwner();
  if (alloc->getBlock() == header ||
      alloc->getBlock()->getParent() != header->getParent())
    return false;
  DominanceInfo dominance(header->getParentOp());
  return dominance.dominates(alloc.getOperation(), &header->front());
}

struct StageGeometry {
  Value parent;
  Value index;
  size_t stride;
  size_t extent;
  unsigned numStages;
};

static std::optional<StageGeometry>
getStageGeometry(Value value, Interval<size_t> interval,
                 Allocation::BufferId bufferId, Allocation *allocation) {
  if (!allocation)
    return std::nullopt;
  triton::gpu::MemDescIndexOp index;
  while (Operation *def = value.getDefiningOp()) {
    if ((index = dyn_cast<triton::gpu::MemDescIndexOp>(def)))
      break;
    // These views stay within the indexed stage. Do not extend this to every
    // MemDescViewTrait: reinterpretation can change the physical stage bounds.
    if (!isa<triton::gpu::MemDescSubsliceOp, triton::gpu::MemDescTransOp>(def))
      return std::nullopt;
    value = def->getOperand(0);
  }
  if (!index)
    return std::nullopt;
  auto alloc = index.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
  if (!alloc || !alloc.isSharedMemoryAlloc())
    return std::nullopt;
  auto parentType = alloc.getType();
  auto stageType = index.getType();
  auto encoding = parentType.getEncoding();
  auto layout = dyn_cast<triton::gpu::LayoutEncodingTrait>(encoding);
  if (!layout || isa<triton::gpu::PartitionedSharedEncodingAttr>(encoding) ||
      parentType.getRank() != stageType.getRank() + 1 ||
      layout.getRank() != stageType.getRank() ||
      parentType.getShape().front() <= 0 ||
      parentType.getShape() != parentType.getAllocShape() ||
      stageType.getShape() != stageType.getAllocShape() ||
      parentType.getShape().drop_front() != stageType.getShape() ||
      encoding != stageType.getEncoding())
    return std::nullopt;

  auto ids = allocation->getBufferIds(alloc.getResult());
  if (ids.size() != 1 || ids.front() != bufferId ||
      allocation->getAllocatedInterval(bufferId) != interval)
    return std::nullopt;

  // Match Allocation's footprint and MemDescIndex lowering's stage stride.
  // Padding omits the gap after the last element, so dividing the footprint
  // by the stage count does not necessarily give the stage stride.
  unsigned bitWidth = getIntOrFloatOrPtrBitWidth(parentType.getElementType());
  if (bitWidth < 8 || bitWidth % 8)
    return std::nullopt;
  int64_t parentElems =
      triton::gpu::getAllocationElems(encoding, parentType.getAllocShape());
  int64_t stageElems =
      triton::gpu::getAllocationElems(encoding, stageType.getAllocShape());
  int64_t numStages = parentType.getShape().front();
  if (parentElems <= 0 || stageElems <= 0 ||
      parentElems > std::numeric_limits<int32_t>::max() ||
      parentElems % numStages || parentElems / numStages != stageElems)
    return std::nullopt;
  int64_t strideElems = stageElems;
  int64_t extentElems = stageElems;
  if (auto padded = triton::gpu::getPaddedEncoding(encoding)) {
    // Every stage must start at the same point in each padding period. Without
    // this alignment the first stage's stride cannot describe later stages.
    if (llvm::any_of(padded.getIntervals(), [&](unsigned interval) {
          return stageElems % interval != 0;
        }))
      return std::nullopt;
    parentElems = padded.getPaddedSize({parentElems});
    extentElems = padded.getPaddedSize({stageElems});
    // getPaddedSize(N+1)-1 includes the padding immediately before element N.
    strideElems = padded.getPaddedSize({stageElems + 1}) - 1;
  }
  // memdesc_index lowers the unpadded product and padded offset as i32. Bound
  // both, and check the last stage's extent without overflowing a product.
  if (extentElems <= 0 || strideElems < extentElems ||
      parentElems > std::numeric_limits<int32_t>::max() ||
      parentElems < extentElems || (parentElems - extentElems) % strideElems ||
      (parentElems - extentElems) / strideElems != numStages - 1)
    return std::nullopt;
  size_t bytesPerElement = bitWidth / 8;
  size_t parentBytes = interval.end() - interval.start();
  if (parentBytes % bytesPerElement ||
      parentBytes / bytesPerElement != static_cast<uint64_t>(parentElems))
    return std::nullopt;
  return StageGeometry{alloc.getResult(), index.getIndex(),
                       static_cast<size_t>(strideElems) * bytesPerElement,
                       static_cast<size_t>(extentElems) * bytesPerElement,
                       static_cast<unsigned>(numStages)};
}

AllocationSlice::AllocationSlice(Value value,
                                 Interval<size_t> allocationInterval,
                                 Allocation::BufferId bufferId,
                                 Allocation *allocation, Value stageBasis)
    : allocationInterval(narrowIntervalForSubview(value, allocationInterval)),
      bufferId(bufferId) {
  auto accessTy = cast<triton::gpu::MemDescType>(value.getType());
  this->accessTy = accessTy;

  // Get the memdesc_subslice information if present. If no subslice is
  // present the whole interval is accessed
  if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
    // The source supplies coordinates even if a preceding subslice has not
    // folded, or the descriptor is carried through control flow or a loop.
    subsliceOffsets = subslice.getOffsets();
    subsliceSource = subslice.getSrc();
  }

  auto geometry =
      getStageGeometry(value, allocationInterval, bufferId, allocation);
  if (!geometry)
    return;
  std::optional<unsigned> offset;
  APInt index;
  if (matchPattern(geometry->index, m_ConstantInt(&index)) &&
      !index.isNegative() && index.getZExtValue() < geometry->numStages) {
    offset = index.getZExtValue();
    size_t start = allocationInterval.start() + *offset * geometry->stride;
    this->allocationInterval = {start, start + geometry->extent};
  } else if (isStableStageParent(geometry->parent, stageBasis)) {
    offset = getStageOffset(geometry->index, stageBasis, geometry->numStages);
    if (offset)
      stage.basis = stageBasis;
  }
  if (!offset)
    return;
  stage.parent = geometry->parent;
  stage.parentInterval = allocationInterval;
  stage.stride = geometry->stride;
  stage.numStages = geometry->numStages;
  stage.offset = *offset;
}

AllocationSlice AllocationSlice::enterStageLoop(Value induction,
                                                unsigned initialValue) const {
  auto result = forgetStageLoop();
  if (!result.stage.parent ||
      !isStableStageParent(result.stage.parent, induction))
    return result;
  result.stage.basis = induction;
  result.stage.offset =
      stageResidue(static_cast<int64_t>(result.stage.offset) - initialValue,
                   result.stage.numStages);
  // The index expression describes the original SSA value, not the lifted
  // induction-relative stage that subsequent loop iterations will access.
  result.bufferIndexExpr = nullptr;
  // A lifted entry access now denotes any physical stage. Keeping its old
  // constant interval would incorrectly prove later-iteration accesses
  // disjoint.
  result.allocationInterval = result.stage.parentInterval;
  return result;
}

AllocationSlice AllocationSlice::advanceStageLoop(Value induction) const {
  if (!stage.basis)
    return *this;
  if (stage.basis != induction)
    return forgetStageLoop();
  auto result = *this;
  // old iv = new iv - 1 for the recognized +1 latch. Normalize so the set of
  // symbolic states is finite, independent of the loop's trip count.
  result.stage.offset = stage.offset ? stage.offset - 1 : stage.numStages - 1;
  result.bufferIndexExpr = nullptr;
  return result;
}

AllocationSlice AllocationSlice::forgetStageLoop() const {
  auto result = *this;
  if (stage.basis) {
    result.allocationInterval = stage.parentInterval;
    result.stage = {};
    result.bufferIndexExpr = nullptr;
  }
  return result;
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  if (stage.basis && stage.basis == other.stage.basis &&
      stage.parent == other.stage.parent &&
      stage.parentInterval == other.stage.parentInterval &&
      stage.stride == other.stage.stride && bufferId == other.bufferId &&
      stage.numStages == other.stage.numStages &&
      stage.offset != other.stage.offset)
    return false;

  // Physical footprints include every possible origin across loop iterations,
  // retaining CTA and memory-space identity.
  if (!triton::mayOverlap(physicalFootprint, other.physicalFootprint))
    return false;

  // Compare indices of the same source descriptor to prove disjoint slots,
  // including arguments without allocator IDs.
  if (bufferId == other.bufferId &&
      areBufferIndicesProvablyDifferent(*this, other))
    return false;

  // Compare logical offsets only for subslices of the same source descriptor.
  if (!subsliceSource || subsliceSource != other.subsliceSource)
    return true;

  auto shapeA = accessTy.getShape();
  auto shapeB = other.accessTy.getShape();
  // Chek if all subslice region dimensions have some intersection
  // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
  // If any dimension doesn't intersect, we are looking at disjoint subslices
  for (size_t i = 0; i < subsliceOffsets.size(); ++i) {
    int64_t startA = subsliceOffsets[i];
    int64_t endA = startA + shapeA[i];
    int64_t startB = other.subsliceOffsets[i];
    int64_t endB = startB + shapeB[i];

    // Is A completely before B? Is B completely before A? If so, disjoint
    if (endA <= startB || endB <= startA)
      return false;
  }

  // All dimensions of subslices have some intersection
  return true;
}

void AllocationSlice::print(raw_ostream &os) const {
  os << "interval=[" << allocationInterval.start() << ","
     << allocationInterval.end() << ")";

  if (bufferId != Allocation::InvalidBufferId)
    os << " buffer=" << bufferId;
  if (argumentIndex)
    os << " argument=" << *argumentIndex;

  os << " offsets=[";
  if (!subsliceOffsets.empty()) {
    llvm::interleaveComma(subsliceOffsets, os);
  } else {
    os << "unknown";
  }
  os << "]";

  os << " shape=";
  if (accessTy) {
    llvm::interleave(accessTy.getShape(), os, "x");
    os << " layout=" << accessTy.getEncoding();
  } else {
    os << "? layout=unknown";
  }
}

static bool isFullCTABarrier(Operation *op) {
  if (isa<gpu::BarrierOp>(op))
    return true;
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op))
    return barrier.hasLocal();
  return false;
}

void MembarStageAnalysis::discoverStageLoops(FunctionOpInterface function) {
  stageLoops.clear();
  DominanceInfo dominance(function.getOperation());
  for (Block &header : function.getBlocks()) {
    auto branch = dyn_cast<cf::CondBranchOp>(header.getTerminator());
    if (!branch || !llvm::hasSingleElement(header.without_terminator()))
      continue;
    auto compare = branch.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!compare || compare->getBlock() != &header ||
        compare.getPredicate() != arith::CmpIPredicate::slt)
      continue;
    auto induction = dyn_cast<BlockArgument>(compare.getLhs());
    if (!induction || induction.getOwner() != &header ||
        !induction.getType().isInteger(32))
      continue;

    Block *body = branch.getTrueDest();
    if (body == &header || branch.getFalseDest() == &header ||
        branch.getFalseDest() == body || !body->getArguments().empty() ||
        body->getSinglePredecessor() != &header)
      continue;
    auto latch = dyn_cast<cf::BranchOp>(body->getTerminator());
    if (!latch || latch.getDest() != &header)
      continue;
    auto predecessors = llvm::to_vector(header.getPredecessors());
    if (predecessors.size() != 2)
      continue;
    Block *entry = predecessors[0] == body ? predecessors[1] : predecessors[0];
    if (entry == body || branch.getFalseDest() == entry)
      continue;
    auto entryBranch = dyn_cast<cf::BranchOp>(entry->getTerminator());
    if (!entryBranch || entryBranch.getDest() != &header)
      continue;

    // A loop-invariant signed upper bound and +1 step ensure that each
    // executing induction value is nonnegative and the increment cannot
    // overflow: iv < ub <= INT_MAX implies iv <= INT_MAX-1.
    Value upperBound = compare.getRhs();
    if (upperBound.getParentBlock() == &header ||
        upperBound.getParentBlock() == body ||
        !dominance.dominates(upperBound, &header.front()))
      continue;
    unsigned argIndex = induction.getArgNumber();
    APInt initial;
    if (!matchPattern(entryBranch.getDestOperands()[argIndex],
                      m_ConstantInt(&initial)) ||
        initial.isNegative())
      continue;
    auto increment =
        latch.getDestOperands()[argIndex].getDefiningOp<arith::AddIOp>();
    if (!increment || increment->getBlock() != body ||
        !((increment.getLhs() == induction &&
           isConstantInt(increment.getRhs(), 1)) ||
          (increment.getRhs() == induction &&
           isConstantInt(increment.getLhs(), 1))))
      continue;

    // An unconditional full CTA barrier bounds inter-wave skew to one
    // iteration. Arrival, partition, atomic and scheduling barriers do not
    // establish this proof. Discover before inserting any new barriers.
    bool hasBodyBarrier = false;
    bool unsupported = false;
    for (Operation &op : body->without_terminator()) {
      if (op.getNumRegions() || isa<CallOpInterface>(&op)) {
        unsupported = true;
        break;
      }
      hasBodyBarrier |= isFullCTABarrier(&op);
    }
    if (unsupported || !hasBodyBarrier)
      continue;

    // Only accesses after this entry barrier can be lifted into the first
    // iteration's coordinates. Nested control or calls after it are outside
    // this bounded proof.
    bool hasEntryBarrier = false;
    for (Operation *op = entryBranch->getPrevNode(); op;
         op = op->getPrevNode()) {
      if (isFullCTABarrier(op)) {
        hasEntryBarrier = true;
        break;
      }
      if (op->getNumRegions() || isa<CallOpInterface>(op))
        break;
    }
    if (!hasEntryBarrier)
      continue;

    stageLoops.push_back({entry, &header, body, induction,
                          static_cast<unsigned>(initial.getZExtValue())});
  }
}

Value MembarStageAnalysis::getStageBasis(Operation *operation) const {
  for (const StageLoop &loop : stageLoops)
    if (operation->getBlock() == loop.body)
      return loop.induction;
  return {};
}

BlockInfo MembarStageAnalysis::transferStageEdge(const BlockInfo &info,
                                                 Block *from, Block *to) const {
  for (const StageLoop &loop : stageLoops) {
    if (from == loop.entry && to == loop.header)
      return info.mapSlices([&](const AllocationSlice &slice) {
        return slice.enterStageLoop(loop.induction, loop.initialValue);
      });
    if (from == loop.body && to == loop.header)
      return info.mapSlices([&](const AllocationSlice &slice) {
        return slice.advanceStageLoop(loop.induction);
      });
    if (from == loop.header && to == loop.body)
      return info;
  }
  // No symbolic coordinates escape the recognized edges. In particular the
  // exit can contain any physical stage, including on a zero-trip path.
  return info.mapSlices(
      [](const AllocationSlice &slice) { return slice.forgetStageLoop(); });
}

BlockInfo MembarOrFenceAnalysis::transferEdge(const BlockInfo &info,
                                              Block *from, Block *to) const {
  return transferStageEdge(info, from, to);
}

MembarInfo MembarAnalysis::transferEdge(const MembarInfo &info, Block *from,
                                        Block *to) const {
  MembarInfo result = info;
  // Entry-prefix effects are historical; only pending effects move with the
  // current iteration's coordinates.
  result.pending = transferStageEdge(info.pending, from, to);
  return result;
}

void MembarOrFenceAnalysis::run(FunctionOpInterface function,
                                FuncMapT &funcMap) {
  // Discover the qualifying barriers before the analysis inserts new ones.
  discoverStageLoops(function);
  triton::PostOrderFunctionAnalysis<BlockInfo>::run(function, funcMap);
}

void MembarAnalysis::run(FunctionOpInterface function, FuncMapT &funcMap) {
  scratchSyncOps.clear();
  // Discover the qualifying barriers before the analysis inserts new ones.
  discoverStageLoops(function);
  triton::PostOrderFunctionAnalysis<MembarInfo>::run(function, funcMap);
  if (!scratchSync)
    return;

  // A later fixed-point visit can discover a CTA hazard that supersedes a
  // scratch-only WAR. Keep emission out of the short-circuiting hazard query
  // and out of the fixed-point traversal itself.
  OpBuilder builder(function.getContext());
  function.walk([&](Operation *op) {
    if (!scratchSyncOps.contains(op))
      return;
    builder.setInsertionPoint(op);
    scratchSync.emitBefore(op, builder);
  });
}

void BlockInfo::invalidateIterationInfo() {
  transformSlices([](AllocationSlice slice) {
    slice.invalidateIterationInfo();
    return SmallVector<AllocationSlice>{std::move(slice)};
  });
}

AllocationSlice AllocationSlice::translateToCallsite(
    CallOpInterface call, FunctionOpInterface callee,
    triton::BufferRegionAnalysis &regions) const {
  assert(!argumentIndex && "argument effects must be bound to call operands");
  AllocationSlice shifted = *this;
  // Unknown accesses span the whole frame; shifting their endpoint overflows.
  if (allocationInterval != Interval<size_t>{}) {
    size_t offset = regions.getCallOffset(call);
    shifted.allocationInterval = Interval<size_t>(
        allocationInterval.start() + offset, allocationInterval.end() + offset);
  }
  shifted.physicalFootprint =
      regions.translateToCallsite(physicalFootprint, call, callee);
  shifted.bufferId = Allocation::InvalidBufferId;
  shifted.stage = {};
  shifted.invalidateIterationInfo();
  return shifted;
}

namespace {

enum class ThreadSyncKind {
  Ordinary,
  Publication,
  // Each warp publishes its completed effects without a CTA rendezvous.
  WarpPublication,
  // Each thread publishes its own effects. The barrier completes only after
  // every thread has arrived.
  ThreadPublication,
  // The operation does not observe another thread's memory.
  Independent,
  // Completions only establish completion or acquire state; they do
  // not access payload or consume another completion's unpublished effects.
  Completion,
  CompletionNeedsSync,
  // These completions do not publish writes to global memory.
  SharedCompletionNeedsSync,
  // Finish waits before a notification can let a peer advance their phase.
  MBarrierWait,
};

struct ThreadSyncInfo {
  // Rendezvous requirements, not barriers provided by the operation itself.
  ThreadSyncKind kind = ThreadSyncKind::Ordinary;
  ThreadIssuer::Kind issuer = ThreadIssuer::Unknown;

  bool requiresBefore() const { return kind == ThreadSyncKind::Publication; }

  bool requiresWarpBefore() const {
    return kind == ThreadSyncKind::WarpPublication;
  }

  bool requiresAfter() const {
    return kind == ThreadSyncKind::CompletionNeedsSync ||
           kind == ThreadSyncKind::SharedCompletionNeedsSync;
  }

  bool hasThreadDemand() const {
    return kind == ThreadSyncKind::Ordinary ||
           kind == ThreadSyncKind::Publication;
  }
};

ThreadSyncInfo getThreadSyncInfo(Operation *op) {
  // Prints go to the host and an assert only aborts. They often run under a
  // thread-zero branch, where a barrier would deadlock.
  if (isa<triton::PrintOp, triton::AssertOp>(op))
    return {ThreadSyncKind::Independent};
  // Acquires and proxy fences do not consume payload. Wait deps only keep
  // allocations live; each thread fences its own completed accesses.
  if (isa<ttng::WaitBarrierOp>(op))
    return {ThreadSyncKind::MBarrierWait};
  if (isa<ttng::FenceAsyncSharedOp>(op))
    return {ThreadSyncKind::Completion};
  if (isa<triton::gpu::AsyncWaitOp>(op))
    return {ThreadSyncKind::SharedCompletionNeedsSync};
  if (auto wait = dyn_cast<ttng::TMAStoreWaitOp>(op);
      wait && wait.getReadOnly())
    return {ThreadSyncKind::SharedCompletionNeedsSync};
  if (op->hasTrait<OpTrait::MemWaitOpTrait>() ||
      isa<ttng::TensormapFenceproxyAcquireOp>(op))
    return {ThreadSyncKind::CompletionNeedsSync};
  if (auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    // WGMMA waits only synchronize the issuing warp group.
    return {!wait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4
                ? ThreadSyncKind::CompletionNeedsSync
                : ThreadSyncKind::Completion};
  }
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(op)) {
    // Init and inval use the same full-mask election, or thread zero before
    // SM90.
    return {ThreadSyncKind::Ordinary, ThreadIssuer::Warp0Leader};
  }
  // A per-thread arrival converted to per-warp arrival keeps the warp-scope
  // rendezvous of the per-warp form.
  if (auto barrier = dyn_cast<triton::gpu::MBarrierOpInterface>(op);
      barrier && barrier.isPerWarp())
    return {ThreadSyncKind::WarpPublication};
  if (ttng::arrivesFromEveryThread(op))
    return {ThreadSyncKind::ThreadPublication};
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op)) {
    auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
    auto fromCTA = arrive ? arrive.getFromCTA()
                          : cast<ttng::BarrierExpectOp>(op).getFromCTA();
    // Nonidentity routing may use several lanes to signal the peer CTAs, and
    // a predicated per-thread arrival comes from the threads it selects.
    bool fromThread0 = !fromCTA && !(arrive && arrive.getPerThread());
    return {ThreadSyncKind::Publication,
            fromThread0 ? ThreadIssuer::Thread0 : ThreadIssuer::Unknown};
  }
  // MMAs and commits are issued by the same full-mask election in warp zero,
  // so a commit tracks every earlier MMA of its region.
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    return {mma.getCompletionBarriers().empty() ? ThreadSyncKind::Ordinary
                                                : ThreadSyncKind::Publication,
            ThreadIssuer::Warp0Leader};
  if (isa<ttng::TCGen5CommitOp>(op))
    return {ThreadSyncKind::Publication, ThreadIssuer::Warp0Leader};
  return {};
}

// Whether pending thread effects must rendezvous before `demands`.
bool requiresThreadSyncFor(const BlockInfo::ThreadSyncState &sync,
                           const BlockInfo::ThreadDemandState &demands,
                           bool hasSharedAccesses) {
  if (!demands.hasDemand)
    return false;
  // Waits, fences and arrivals only have to finish before another thread's
  // publication, or before a notification can let a peer advance a waited
  // phase. Private counter updates are already ordered by the slice hazards.
  if (demands.mayIssueAhead && !sync.syncIssuers.empty() &&
      ((!demands.publicationIssuers.empty() &&
        !sync.syncIssuers.hasSameKnownIssuer(demands.publicationIssuers)) ||
       (sync.peerWaitNeedsSync && demands.mayNotifyPeer)))
    return true;
  if (sync.effectIssuers.empty())
    return false;
  return sync.completion == CompletionSync::All ||
         (sync.completion == CompletionSync::Shared &&
          (hasSharedAccesses || demands.hasDemandBeyondGlobalReads)) ||
         (!demands.publicationIssuers.empty() &&
          !sync.effectIssuers.hasSameKnownIssuer(demands.publicationIssuers));
}

// A barrier initialization or invalidation is pending.
bool hasPendingBarrierLifetime(const BlockInfo &pending) {
  return llvm::any_of(pending.syncWriteSlices, [](const auto &entry) {
    return llvm::any_of(entry.second, [](Operation *op) {
      return isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(op);
    });
  });
}

// Whether the slice hazards can miss a pending barrier initialization or
// invalidation. An MMA with completion barriers is modeled as completing every
// pending effect, and a per-thread arrival records no slices.
bool missesBarrierLifetime(Operation *op) {
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    return !mma.getCompletionBarriers().empty();
  return ttng::arrivesFromEveryThread(op);
}

// Effects that only synchronize. Their mbarrier updates and proxy ordering
// carry no payload for a later publication to make visible.
bool carriesNoPayload(Operation *op) {
  return isa<ttng::WaitBarrierOp, ttng::FenceAsyncSharedOp,
             ttng::ArriveBarrierOp, ttng::BarrierExpectOp,
             ttng::TCGen5CommitOp>(op);
}

// Whether a store writes each address from a single thread. The lowering
// stores the copies that one thread holds once, but every thread that holds a
// copy of an element writes it.
bool storesEachAddressFromOneThread(triton::gpu::LocalStoreOp store) {
  auto srcTy = cast<RankedTensorType>(store.getSrc().getType());
  if (!isa_and_nonnull<triton::gpu::DistributedEncodingTrait>(
          srcTy.getEncoding()))
    return false;
  MLIRContext *ctx = store.getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  triton::LinearLayout layout =
      triton::gpu::toLinearLayout(srcTy).removeZeroBasesAlongDim(kReg);
  return layout
      .sublayout({kReg, kLane, kWarp}, llvm::to_vector(layout.getOutDimNames()))
      .isInjective();
}

// Whether `before` and `after` are two executions of one register-to-shared
// store into a view that is fixed for the whole kernel or warp-specialized
// region. When the stored layout gives each element to one thread, every
// execution writes each address from that same thread, which orders its own
// writes by program order, so the last write to an address always comes from
// the latest execution. A layout that replicates elements across lanes or
// warps does not qualify: each copy is written, and one thread's write from an
// earlier execution can land after another thread's write from a later one.
bool isRepeatedLocalStore(Operation *before, Operation *after,
                          bool beforeIsRead, bool afterIsRead) {
  if (before != after || beforeIsRead || afterIsRead)
    return false;
  auto store = dyn_cast<triton::gpu::LocalStoreOp>(before);
  if (!store || !storesEachAddressFromOneThread(store))
    return false;
  // The view must be computed once, before any control flow of a kernel or
  // warp-specialized region, so every execution writes the same addresses.
  Operation *def = store.getDst().getDefiningOp();
  Region *region = before->getParentRegion();
  if (!def || def->getParentRegion() != region ||
      def->getBlock() != &region->front())
    return false;
  Operation *parent = region->getParentOp();
  if (auto func = dyn_cast<FunctionOpInterface>(parent))
    return triton::isKernel(func);
  return isa<triton::gpu::WarpSpecializeOp,
             triton::gpu::WarpSpecializePartitionsOp>(parent);
}

bool hasIssuedEffects(const BlockInfo::ThreadSyncState &sync) {
  return !sync.effectIssuers.empty() || !sync.syncIssuers.empty();
}

// Every observer stays in this region and CTA. Follow views and selects;
// calls, captures and raw-address escapes reject the proof.
bool hasRegionLocalUses(Value root) {
  if (ttng::hasCrossCTASharedAccess(root.getDefiningOp()).value_or(false))
    return false;
  SmallVector<Value> worklist{root};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->hasTrait<OpTrait::MemDescViewTrait>() ||
          isa<arith::SelectOp>(user)) {
        llvm::append_range(worklist, user->getResults());
        continue;
      }
      if (user->getParentRegion() != root.getParentRegion())
        return false;
      // TMEM loads and stores only access the issuing CTA's tensor memory.
      if (isa<triton::gpu::LocalDeallocOp, ttng::TMEMLoadOp, ttng::TMEMStoreOp>(
              user))
        continue;
      // Wait dependencies only keep allocations live.
      if (auto wait = dyn_cast<ttng::WaitBarrierOp>(user);
          wait && wait.getAlloc() != value)
        continue;
      auto barrier = dyn_cast<triton::gpu::MBarrierOpInterface>(user);
      if (barrier && llvm::is_contained(barrier.getBarriers(), value)) {
        if (ttng::hasCrossCTAMBarrierUse(barrier) ||
            ttng::hasCGABroadcast(
                cast<triton::gpu::MemDescType>(value.getType())))
          return false;
        if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(user))
          continue;
        for (const auto &access : triton::getMemoryAccesses(user))
          if (access.value == value &&
              access.sharedKind != triton::gpu::SharedKind::Barrier)
            return false;
        continue;
      }
      auto crossCTA = ttng::hasCrossCTASharedAccess(user);
      if (!crossCTA || *crossCTA)
        return false;
    }
  }
  return true;
}

bool hasOnlyGlobalReads(Operation *op) {
  auto effects = getEffectsRecursively(op);
  return effects && !effects->empty() &&
         llvm::all_of(*effects, [](const auto &effect) {
           return isa<MemoryEffects::Read>(effect.getEffect()) &&
                  isa<triton::GlobalMemory>(effect.getResource());
         });
}

ThreadIssuer getThreadIssuer(Operation *op) {
  auto kind = getThreadSyncInfo(op).issuer;
  return kind == ThreadIssuer::Unknown
             ? ThreadIssuer{}
             : ThreadIssuer{kind, op->getParentRegion()};
}

bool haveSameThreadSyncIssuer(Operation *lhs, Operation *rhs) {
  auto issuer = getThreadIssuer(lhs);
  return issuer.kind != ThreadIssuer::Unknown && issuer == getThreadIssuer(rhs);
}

} // namespace

bool DivergentControlFlow::isLoopBranch(Operation *branch) {
  // A branch inside a loop that can leave it decides whether to repeat the
  // loop. A branch whose paths all stay in the loop only skips part of an
  // iteration, even when one of them goes straight back to the header:
  // canonicalization threads the empty merge block of an if at the end of a
  // loop body into the header.
  Block *block = branch->getBlock();
  Region *region = block->getParent();
  if (region->hasOneBlock())
    return false;
  std::unique_ptr<CFGLoopInfo> &regionLoops = loops[region];
  if (!regionLoops)
    regionLoops = std::make_unique<CFGLoopInfo>(dominance.getDomTree(region));
  CFGLoop *loop = regionLoops->getLoopFor(block);
  return loop && llvm::any_of(branch->getSuccessors(), [&](Block *successor) {
           return !loop->contains(successor);
         });
}

ArrayRef<Block *> DivergentControlFlow::getSkippableBlocks(Operation *branch) {
  auto it = skippableBlocks.find(branch);
  if (it != skippableBlocks.end())
    return it->second;
  // The paths meet again at the first block that post-dominates the branch.
  Block *block = branch->getBlock();
  SmallVector<Block *> worklist(branch->getSuccessors());
  DenseSet<Block *> seen{block};
  SmallVector<Block *> skippable;
  while (!worklist.empty()) {
    Block *next = worklist.pop_back_val();
    if (!seen.insert(next).second || postDominance.postDominates(next, block))
      continue;
    skippable.push_back(next);
    llvm::append_range(worklist, next->getSuccessors());
  }
  return skippableBlocks[branch] = std::move(skippable);
}

void DivergentControlFlow::findDivergentValues() {
  DenseSet<Value> divergent;
  SmallVector<Value> worklist;
  auto add = [&](Value value) {
    if (divergent.insert(value).second)
      worklist.push_back(value);
  };
  auto addArguments = [&](Block *block) {
    for (BlockArgument argument : block->getArguments())
      add(argument);
  };
  // A device function can be called with values that differ between threads.
  auto funcOp = cast<FunctionOpInterface>(function);
  if (!triton::isKernel(funcOp) && !funcOp.getFunctionBody().empty())
    addArguments(&funcOp.getFunctionBody().front());
  // A warp vote or ballot is the same in every lane of a warp, but can differ
  // between the warps even when its operands do not.
  function->walk([&](Operation *op) {
    if (isa<gpu::ThreadIdOp, gpu::LaneIdOp, triton::gpu::WarpIdOp,
            triton::gpu::WarpVoteOp, ttng::VoteBallotSyncOp,
            triton::ElementwiseInlineAsmOp, CallOpInterface>(op))
      for (Value result : op->getResults())
        add(result);
  });
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      for (Value result : user->getResults())
        add(result);
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        bool forwarded = false;
        for (auto [index, successor] : llvm::enumerate(user->getSuccessors())) {
          SuccessorOperands operands = branch.getSuccessorOperands(index);
          for (BlockArgument argument : successor->getArguments()) {
            if (operands.getOperandIndex(argument.getArgNumber()) ==
                use.getOperandNumber()) {
              add(argument);
              forwarded = true;
            }
          }
        }
        if (forwarded)
          continue;
        // A divergent condition decides which values reach the blocks where
        // the paths meet again.
        if (isLoopBranch(user)) {
          for (Block *successor : user->getSuccessors())
            addArguments(successor);
          continue;
        }
        ArrayRef<Block *> skippable = getSkippableBlocks(user);
        DenseSet<Block *> inside(skippable.begin(), skippable.end());
        inside.insert(user->getBlock());
        for (Block *block : inside)
          for (Block *successor : block->getSuccessors())
            if (!inside.contains(successor))
              addArguments(successor);
        continue;
      }
      // Each partition receives a capture as the matching region argument.
      if (isa<triton::gpu::WarpSpecializePartitionsOp>(user)) {
        for (Region &region : user->getRegions())
          if (!region.empty())
            add(region.getArgument(use.getOperandNumber()));
        continue;
      }
      // Structured control flow forwards values to its regions and results.
      Operation *parent = isa<RegionBranchTerminatorOpInterface>(user)
                              ? user->getParentOp()
                              : user;
      if (isa<RegionBranchOpInterface>(parent)) {
        for (Value result : parent->getResults())
          add(result);
        for (Region &region : parent->getRegions())
          if (!region.empty())
            addArguments(&region.front());
      }
    }
  }
  divergentValues = std::move(divergent);
}

bool DivergentControlFlow::mayDiverge(Value value) {
  if (!divergentValues)
    findDivergentValues();
  return divergentValues->contains(value);
}

bool DivergentControlFlow::isDivergentBranch(Operation *op) {
  Value condition;
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    condition = ifOp.getCondition();
  else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op))
    condition = switchOp.getArg();
  else if (auto branch = dyn_cast<cf::CondBranchOp>(op))
    condition = branch.getCondition();
  else if (auto switchOp = dyn_cast<cf::SwitchOp>(op))
    condition = switchOp.getFlag();
  else
    return false;
  return mayDiverge(condition) &&
         (isa<scf::IfOp, scf::IndexSwitchOp>(op) || !isLoopBranch(op));
}

void DivergentControlFlow::walkSkippable(
    Operation *op, function_ref<void(Operation *)> visit) {
  if (isa<scf::IfOp, scf::IndexSwitchOp>(op)) {
    for (Region &region : op->getRegions())
      region.walk(visit);
    return;
  }
  // The visitor may query other branches, which can grow the cache.
  SmallVector<Block *> blocks(getSkippableBlocks(op));
  for (Block *block : blocks)
    for (Operation &nested : *block)
      nested.walk(visit);
}

bool DivergentControlFlow::isUnderDivergentBranch(Operation *op) {
  if (!divergentBlocks) {
    DenseSet<Block *> blocks;
    function->walk([&](Operation *branch) {
      if (isa<cf::CondBranchOp, cf::SwitchOp>(branch) &&
          isDivergentBranch(branch))
        for (Block *block : getSkippableBlocks(branch))
          blocks.insert(block);
    });
    divergentBlocks = std::move(blocks);
  }
  // Stop at the function body or at a warp-specialized region, which every
  // thread of its warps enters.
  for (Operation *parent = op->getParentOp(); parent;
       op = parent, parent = op->getParentOp()) {
    if (divergentBlocks->contains(op->getBlock()) ||
        (isa<scf::IfOp, scf::IndexSwitchOp>(parent) &&
         isDivergentBranch(parent)))
      return true;
    if (isa<FunctionOpInterface, triton::gpu::WarpSpecializeOp,
            triton::gpu::WarpSpecializePartitionsOp>(parent))
      return false;
  }
  return false;
}

bool IssuerSummary::hasSameKnownIssuer(const IssuerSummary &other) const {
  return issuer && issuer->kind != ThreadIssuer::Unknown && *this == other;
}

void IssuerSummary::join(const IssuerSummary &other) {
  if (other.empty())
    return;
  if (empty()) {
    *this = other;
  } else if (*this != other) {
    issuer = ThreadIssuer{};
  }
}

void BlockInfo::joinThreadState(const BlockInfo &other) {
  const auto &sync = other.threadSync;
  threadSync.effectIssuers.join(sync.effectIssuers);
  threadSync.completion = std::max(threadSync.completion, sync.completion);
  threadSync.syncIssuers.join(sync.syncIssuers);
  threadSync.peerWaitNeedsSync |= sync.peerWaitNeedsSync;
  const auto &demands = other.threadDemands;
  threadDemands.hasDemand |= demands.hasDemand;
  threadDemands.hasDemandBeyondGlobalReads |=
      demands.hasDemandBeyondGlobalReads;
  threadDemands.publicationIssuers.join(demands.publicationIssuers);
  threadDemands.mayNotifyPeer |= demands.mayNotifyPeer;
  threadDemands.mayIssueAhead |= demands.mayIssueAhead;
}

bool MembarAnalysis::isRegionLocal(Value value) {
  auto *footprint = regions.getFootprint(value);
  return footprint &&
         llvm::all_of(footprint->regionInfo.views, [&](auto &view) {
           Value root = view.allocation->getResult(0);
           auto [it, inserted] =
               regionLocalAllocations.try_emplace(root, false);
           if (inserted)
             it->second = hasRegionLocalUses(root);
           return it->second;
         });
}

bool MembarAnalysis::mayNotifyPeer(Operation *op) {
  // Nested operations are checked separately; region scratch is private.
  if (isa<RegionBranchOpInterface, triton::AtomicLoadOp, triton::AtomicPollOp>(
          op))
    return false;
  // A weak shared or tensor memory store cannot release a peer. The peer
  // observes it only after a barrier, which is checked as a notification
  // itself.
  if (isa<triton::gpu::LocalStoreOp, triton::gpu::LocalAllocOp,
          ttng::TMEMStoreOp, ttng::TMEMAllocOp>(op))
    return false;
  auto effects = getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [&](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect()) &&
           (!effect.getValue() || !isRegionLocal(effect.getValue()));
  });
}

bool MembarAnalysis::requiresThreadSync(const BlockInfo &pending,
                                        const BlockInfo &effects) {
  // The maps also include implicit scratch and translated callee accesses.
  bool hasSharedAccesses =
      !effects.syncReadSlices.empty() || !effects.syncWriteSlices.empty();
  return requiresThreadSyncFor(pending.threadSync, effects.threadDemands,
                               hasSharedAccesses);
}

Operation *MembarAnalysis::syncIfNeeded(Operation *op, const BlockInfo &effects,
                                        MembarInfo *membarInfo,
                                        OpBuilder *builder, bool cluster) {
  if (!builder)
    return nullptr;
  auto &pending = membarInfo->pending;
  auto canSkip = [&](Operation *before, Operation *after, bool beforeIsRead,
                     bool afterIsRead, Allocation *allocation,
                     const AllocationSlice &beforeSlice,
                     const AllocationSlice &afterSlice) {
    // Effects issued by the same thread are ordered with respect to each other.
    // A fixed thread-zero issuer and an elected warp-zero leader may differ.
    return (hasIssuedEffects(pending.threadSync) &&
            hasIssuedEffects(effects.threadSync) &&
            haveSameThreadSyncIssuer(before, after)) ||
           isRepeatedLocalStore(before, after, beforeIsRead, afterIsRead) ||
           (filter && filter(before, after, beforeIsRead, afterIsRead,
                             allocation, beforeSlice, afterSlice));
  };
  // Every thread must observe a barrier initialization or invalidation before
  // it next uses the barrier, even when the slice hazards cannot see that use.
  bool barrierLifetimeHazard = !cluster && missesBarrierLifetime(op) &&
                               hasPendingBarrierLifetime(pending);
  if (!barrierLifetimeHazard && !requiresThreadSync(pending, effects) &&
      !pending.isIntersected(effects, canSkip, &allocation, sliceFilter))
    return nullptr;
  // The barrier clears incoming state. The operation's own effects still
  // follow it, including a scratch write that conflicts with a pending read.
  builder->setInsertionPoint(op);
  Operation *barrier = insertBarrier(op, builder, cluster);
  membarInfo->sync();
  return barrier;
}

Operation *MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                         bool cluster) {
  OpBuilder::InsertionGuard g(*builder);
  if (cluster)
    return ttng::ClusterBarrierOp::create(*builder, op->getLoc());
  return triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                        triton::gpu::AddrSpace::Local);
}

static Allocation::BufferId getScratchBufferId(Operation *op,
                                               Allocation *allocation) {
  // A call's allocation belongs to the callee and is translated separately.
  if (isa<CallOpInterface>(op))
    return Allocation::InvalidBufferId;
  return allocation->getBufferId(op);
}

static bool scratchBufferUsesWarpSync(Operation *op) {
  if (isa<ttng::TensormapCreateOp>(op))
    return true;
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;

  auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(cvt.getType());
  auto kWarp = StringAttr::get(op->getContext(), "warp");
  return mlir::isCvtDimSync(srcTy, dstTy, kWarp);
}

triton::BarrierStages getLocalBarrierStages(Operation *op,
                                            Allocation *allocation) {
  triton::BarrierStages stages;
  // The local-memory mask guarantees ordering of local memory accesses.
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op)) {
    stages.beforeMemoryEffects = stages.afterMemoryEffects = barrier.hasLocal();
    return stages;
  }
  // Explicit barriers have no accesses between their leading and trailing
  // stages. A relaxed cluster rendezvous does not order memory accesses.
  if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op)) {
    stages.beforeMemoryEffects = stages.afterMemoryEffects =
        !barrier.getRelaxed();
    return stages;
  }
  // Pure layout conversions and reductions can still use shared scratch and
  // internal barriers even without descriptor memory effects.
  auto scratchBufferId = getScratchBufferId(op, allocation);
  bool hasScratchBarrier = scratchBufferId != Allocation::InvalidBufferId &&
                           !scratchBufferUsesWarpSync(op);

  // Atomic polls always end in a rendezvous. With scratch, the rendezvous is
  // between the scratch write and read; otherwise it follows all effects.
  if (isa<triton::AtomicPollOp>(op)) {
    stages.betweenMemoryEffects = hasScratchBarrier;
    stages.afterMemoryEffects = !hasScratchBarrier;
    return stages;
  }

  if (auto atomic = dyn_cast<triton::AtomicOpInterface>(op)) {
    // Atomic result broadcast uses a scratch write, rendezvous, and read for
    // every memory semantic, including relaxed.
    return triton::getAtomicBarrierStages(atomic.getMemSemantic(),
                                          hasScratchBarrier);
  }

  // Scratch-backed operations contain a rendezvous between their scratch
  // write and read phases. Warp-specialization boundaries synchronize before
  // the operation.
  stages.betweenMemoryEffects = hasScratchBarrier;
  // Fork retains GPU-dialect barriers and cluster waits as
  // self-synchronizing; arrive/expect/commit follow upstream scheduling.
  stages.beforeMemoryEffects =
      isa<gpu::BarrierOp, ttng::ClusterWaitOp, triton::gpu::WarpYieldOp,
          triton::gpu::WarpReturnOp>(op);

  // The first launch rendezvous publishes captures. The second finishes their
  // reads before any partition starts its body and reuses capture storage.
  if (isa<triton::gpu::WarpSpecializeOp>(op)) {
    stages.beforeMemoryEffects = !hasScratchBarrier;
    stages.afterMemoryEffects = true;
  }
  // MMA completion barriers are committed after the MMA's memory effects.
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    stages.afterMemoryEffects = !mma.getCompletionBarriers().empty();
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op))
    stages.beforeMemoryEffects =
        !wgWait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4;
  return stages;
}

void MembarAnalysis::updateSuccessor(Operation *terminator, Block *successor,
                                     MembarInfo *membarInfo) {
  if (bufferIndexAnalysis.isBackedgeSuccessor(terminator, successor)) {
    membarInfo->pending.invalidateIterationInfo();
    membarInfo->entryBlockInfo.invalidateIterationInfo();
  }
}

void MembarAnalysis::updateExitState(MembarInfo *membarInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // identities and loop coordinates are not meaningful there.
  auto forgetStageLoop = [](const AllocationSlice &slice) {
    return slice.forgetStageLoop();
  };
  membarInfo->pending = membarInfo->pending.mapSlices(forgetStageLoop);
  membarInfo->entryBlockInfo =
      membarInfo->entryBlockInfo.mapSlices(forgetStageLoop);
  membarInfo->pending.invalidateIterationInfo();
  membarInfo->entryBlockInfo.invalidateIterationInfo();
}

triton::BarrierStages MembarAnalysis::getBarrierStages(Operation *op) {
  return getLocalBarrierStages(op, &allocation);
}

bool MembarAnalysis::hasThreadEffects(Operation *op) {
  auto effects = getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const auto &effect) {
    return !isa<triton::gpu::SharedMemory>(effect.getResource()) ||
           !isa<MemoryEffects::Allocate, MemoryEffects::Free>(
               effect.getEffect());
  });
}

BlockInfo MembarAnalysis::getThreadEffects(Operation *op) {
  BlockInfo effects;
  auto sync = getThreadSyncInfo(op);
  if (carriesNoPayload(op)) {
    effects.threadSync.syncIssuers = IssuerSummary{getThreadIssuer(op)};
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op))
      effects.threadSync.peerWaitNeedsSync = !isRegionLocal(wait.getAlloc());
    return effects;
  }
  effects.threadSync.effectIssuers = IssuerSummary{getThreadIssuer(op)};
  if (sync.kind == ThreadSyncKind::CompletionNeedsSync)
    effects.threadSync.completion = CompletionSync::All;
  else if (sync.kind == ThreadSyncKind::SharedCompletionNeedsSync)
    effects.threadSync.completion = CompletionSync::Shared;
  return effects;
}

void MembarAnalysis::addThreadDemand(BlockInfo &effects, Operation *op) {
  effects.threadDemands.hasDemand = true;
  effects.threadDemands.mayNotifyPeer = mayNotifyPeer(op);
  effects.threadDemands.mayIssueAhead = !waitsForWholeRegion(op);
  effects.threadDemands.hasDemandBeyondGlobalReads = !hasOnlyGlobalReads(op);
  if (getThreadSyncInfo(op).requiresBefore())
    effects.threadDemands.publicationIssuers =
        IssuerSummary{getThreadIssuer(op)};
}

// The operation cannot take effect before every thread of its region reaches
// it, so the threads have finished their earlier waits, fences and arrivals.
// Unlike a barrier, this does not order memory.
bool MembarAnalysis::waitsForWholeRegion(Operation *op) {
  // A named barrier completes only after every thread it counts arrives. That
  // includes the whole region only if every thread of the region executes it.
  if (isa<ttng::NamedBarrierArriveOp, ttng::NamedBarrierWaitOp>(op))
    return ttng::countsEveryThread(op) &&
           !divergence.isUnderDivergentBranch(op);
  // The lanes of a single warp leave a wait loop together because its branch
  // is uniform, and an elected lane issues these only after every lane has
  // executed the earlier fences and arrivals.
  return isa<ttng::MMAv5OpInterface, ttng::TCGen5CommitOp,
             ttng::AsyncTMACopyGlobalToLocalOp,
             ttng::AsyncTMACopyLocalToGlobalOp>(op) &&
         triton::gpu::maybeLookupNumWarps(op) == 1;
}

BlockInfo MembarAnalysis::getOperationEffects(Operation *op) {
  auto sync = getThreadSyncInfo(op);
  // Region control flow is visited by the dataflow engine. Its children
  // contribute their own effects; implicit scratch belongs to this op. So does
  // a predicated region when the analysis walks into it; otherwise the region
  // is opaque and carries its body's effects itself.
  bool controlFlow =
      isa<BranchOpInterface, RegionBranchOpInterface,
          RegionBranchTerminatorOpInterface>(op) ||
      (isa<triton::gpu::WarpPredicateOp>(op) && analyzesPredicatedRegions());
  bool isReturn = op->hasTrait<OpTrait::ReturnLike>();
  // Cluster barriers contribute synchronization stages, not payload effects.
  // Scheduling-only fences (e.g. AMD rocdl.sched.barrier) carry no memory
  // semantics; pending completions flow through to the real barrier.
  bool hasEffects =
      sync.requiresBefore() || sync.requiresAfter() ||
      getScratchBufferId(op, &allocation) != Allocation::InvalidBufferId ||
      (!controlFlow && !isReturn && !isa<ttng::ClusterBarrierOp>(op) &&
       !isa<triton::gpu::SchedulingBarrierOpInterface>(op) &&
       hasThreadEffects(op));

  BlockInfo effects;
  if (hasEffects) {
    effects = getThreadEffects(op);
    if (sync.hasThreadDemand())
      addThreadDemand(effects, op);
  }
  if (isReturn && isa<FunctionOpInterface>(op->getParentOp()))
    addThreadDemand(effects, op);
  return effects;
}

bool MembarAnalysis::requiresSyncBeforeBranch(Operation *branch,
                                              const BlockInfo &pending) {
  bool pendingLifetime = hasPendingBarrierLifetime(pending);
  if ((!hasIssuedEffects(pending.threadSync) && !pendingLifetime) ||
      !divergence.isDivergentBranch(branch))
    return false;
  auto it = branchDemands.find(branch);
  if (it == branchDemands.end()) {
    SmallVector<GuardedDemand> demands;
    divergence.walkSkippable(branch, [&](Operation *op) {
      // Barriers that this analysis inserts must not change the summary.
      if (isa<triton::gpu::BarrierOp, ttng::ClusterBarrierOp>(op))
        return;
      GuardedDemand demand{getOperationEffects(op).threadDemands,
                           missesBarrierLifetime(op)};
      if (demand.threadDemands.hasDemand || demand.missesBarrierLifetime)
        demands.push_back(demand);
    });
    it = branchDemands.try_emplace(branch, std::move(demands)).first;
  }
  // Slice hazards of the guarded operations are not checked here.
  return llvm::any_of(it->second, [&](const GuardedDemand &demand) {
    return (pendingLifetime && demand.missesBarrierLifetime) ||
           requiresThreadSyncFor(pending.threadSync, demand.threadDemands,
                                 /*hasSharedAccesses=*/false);
  });
}

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op);
      barrier && barrier.isWarp()) {
    membarInfo->syncWarps();
    return;
  }
  // Replace any provisional decision whenever this operation is revisited.
  scratchSyncOps.erase(op);
  if (isa<CallOpInterface>(op)) {
    auto forgetStageLoop = [](const AllocationSlice &slice) {
      return slice.forgetStageLoop();
    };
    membarInfo->pending = membarInfo->pending.mapSlices(forgetStageLoop);
    membarInfo->entryBlockInfo =
        membarInfo->entryBlockInfo.mapSlices(forgetStageLoop);
  }
  // A divergent branch can skip what it guards in only some threads of the
  // region, and a barrier those threads skip deadlocks. Synchronize before the
  // branch if a guarded operation needs pending thread effects finished.
  if (builder && requiresSyncBeforeBranch(op, membarInfo->pending)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }
  BlockInfo effects = getOperationEffects(op);

  // Finish completions that block every memory demand once before a loop.
  // Shared-copy completions can still overlap independent global reads.
  if (builder && bufferIndexAnalysis.entersLoop(op) &&
      membarInfo->pending.threadSync.completion == CompletionSync::All) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }
  updateMemoryEffects(op, membarInfo, funcMap, builder, /*cluster=*/false,
                      std::move(effects));
}

SmallVector<AllocationSlice> MembarAnalysis::getAllocationSlices(
    Value value, Value stageBasis,
    std::optional<triton::gpu::SharedKind> sharedKind) {
  auto function = cast<FunctionOpInterface>(allocation.getOperation());
  // Device-function views use their own frame until imported at a call;
  // foreign or mixed frames, including returned descriptors, stay unknown.
  auto *footprint = regions.getFootprint(value, function);
  SmallVector<AllocationSlice> slices;
  auto addSlice = [&](Interval<size_t> interval,
                      Allocation::BufferId bufferId) {
    auto slice =
        AllocationSlice(value, interval, bufferId, &allocation, stageBasis);
    // Attach unconditionally as upstream does: the disjointness proof checks
    // same base, indexed source and modulus, so argument-rooted indices are
    // sound here too.
    slice.bufferIndexExpr =
        bufferIndexAnalysis.makeSlice(value, interval, bufferId)
            .bufferIndexExpr;
    slice.physicalFootprint = footprint;
    slice.sharedKind = sharedKind;
    slices.push_back(std::move(slice));
  };
  Allocation::BufferIdSetT bufferIds;
  if (accessMode == AccessMode::AllocatorAliasesOnly) {
    // Allocation identity survives unknown geometry. Argument effects
    // have no local IDs; retain them until a caller binds their allocation.
    bufferIds = allocation.getAllBufferIdsWithAliases(value);
    for (unsigned argument : allocation.getAliasedArgumentIndices(value)) {
      AllocationSlice slice(Interval<size_t>{});
      slice.argumentIndex = argument;
      slice.sharedKind = sharedKind;
      slices.push_back(std::move(slice));
    }
  } else if (footprint) {
    // Use every lattice origin: function-local alias analysis can miss roots
    // returned through calls or joined with descriptor arguments.
    for (const auto &view : footprint->regionInfo.views) {
      auto ids = view.allocation
                     ? allocation.getBufferIds(view.allocation->getResult(0))
                     : SmallVector<Allocation::BufferId>{};
      if (ids.empty()) {
        bufferIds.clear();
        break;
      }
      bufferIds.insert(ids.begin(), ids.end());
    }
  }
  // Unknown allocations use an unbounded interval. View disjointness and
  // RAW/WAR/WAW checks still determine whether a barrier is needed.
  if (accessMode == AccessMode::AllSharedAccesses && bufferIds.empty())
    addSlice(Interval<size_t>{}, Allocation::InvalidBufferId);
  for (auto bufferId : bufferIds)
    addSlice(allocation.getAllocatedInterval(bufferId), bufferId);
  return slices;
}

void MembarAnalysis::updateMemoryEffects(Operation *op, MembarInfo *membarInfo,
                                         FuncMapT *funcMap, OpBuilder *builder,
                                         bool cluster, BlockInfo curBlockInfo) {
  auto barrierStages = getBarrierStages(op);
  if (barrierStages.beforeMemoryEffects) {
    // Model a leading barrier before handling the operation's effects.
    membarInfo->sync();
  }

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    auto summary = getCallSummary(
        call, *funcMap, [&](MembarInfo &summary, FunctionOpInterface callee) {
          summary.transformSlices([&](const AllocationSlice &slice) {
            if (!slice.argumentIndex)
              return SmallVector<AllocationSlice>{
                  slice.translateToCallsite(call, callee, regions)};
            Value actual = call.getArgOperands()[*slice.argumentIndex];
            auto slices = getAllocationSlices(actual, {}, slice.sharedKind);
            // A callee can reinterpret the argument's view. Retain the caller's
            // allocation IDs, but cover each whole allocation without shifting.
            for (AllocationSlice &bound : slices) {
              bound.physicalFootprint = nullptr;
              bound.invalidateIterationInfo();
            }
            return slices;
          });
        });
    assert(summary && "expected callee summary");
    syncIfNeeded(op, summary->entryBlockInfo, membarInfo, builder, cluster);
    membarInfo->applyCallSummary(*summary);
    return;
  }

  // Intra-function dependencies
  // Explicit buffer
  // Per-thread mbarrier arrival has release semantics; thread order and the
  // wait already synchronize its SMEM ops. Cluster barriers still need effects.
  bool isPerThreadArrive = !cluster && ttng::arrivesFromEveryThread(op);
  if (!isPerThreadArrive) {
    for (const auto &access : triton::getMemoryAccesses(op)) {
      Value value = access.value;
      auto memory = cast<triton::gpu::MemDescType>(value.getType());
      if (!isa<triton::gpu::SharedMemorySpaceAttr>(memory.getMemorySpace()))
        continue;
      // Shared effects cover only descriptor elements, including gather/scatter
      // and async copies. Footprints retain padding, partitions and CTA
      // identity.
      for (const AllocationSlice &slice :
           getAllocationSlices(value, getStageBasis(op), access.sharedKind)) {
        if (access.isWrite)
          curBlockInfo.syncWriteSlices[slice].insert(op);
        if (access.isRead)
          curBlockInfo.syncReadSlices[slice].insert(op);
      }
    }
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  auto scratchBufferId = getScratchBufferId(op, &allocation);
  std::optional<AllocationSlice> scratchSlice;
  if (scratchBufferId != Allocation::InvalidBufferId) {
    bool hasExplicitSharedDeps = !curBlockInfo.syncReadSlices.empty() ||
                                 !curBlockInfo.syncWriteSlices.empty();
    if (hasExplicitSharedDeps &&
        !isa<triton::gpu::LocalAtomicScatterRMWOp>(op)) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = allocation.getAllocatedInterval(scratchBufferId);
    scratchSlice.emplace(interval);
    scratchSlice->physicalFootprint = regions.getScratchFootprint(op);
    curBlockInfo.syncWriteSlices[*scratchSlice].insert(op);
    bool needsScratchSync = false;
    auto scratchFilter = [&](Operation *lhs, Operation *rhs, bool lhsIsRead,
                             bool rhsIsRead, Allocation *allocation,
                             const AllocationSlice &lhsSlice,
                             const AllocationSlice &rhsSlice) {
      if (filter && filter(lhs, rhs, lhsIsRead, rhsIsRead, allocation, lhsSlice,
                           rhsSlice))
        return true;
      if (!scratchSync || !scratchSync.canReuse(lhs, rhs, allocation))
        return false;
      needsScratchSync |= lhsIsRead && !rhsIsRead;
      return true;
    };
    // A pending thread completion still needs its CTA rendezvous even when
    // scratch ownership can be reused; scratch sync releases only per-warp
    // partitions, never workgroup thread state.
    auto insertCTABarrier =
        requiresThreadSync(membarInfo->pending, curBlockInfo) ||
        membarInfo->pending.isIntersected(curBlockInfo, scratchFilter,
                                          &allocation);
    // Like syncIfNeeded, an analysis-only update (no builder) inserts nothing.
    if (insertCTABarrier && builder) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder, cluster);
      membarInfo->sync();
    } else if (!insertCTABarrier && needsScratchSync) {
      // Preserve all pending effects: this synchronization releases only the
      // proven per-warp scratch ownership, not arbitrary workgroup accesses.
      scratchSyncOps.insert(op);
    }
  }

  // Scratch effects already went through the scratch-reuse decision above;
  // running the plain barrier check would force a CTA barrier anyway.
  if (!scratchSlice)
    syncIfNeeded(op, curBlockInfo, membarInfo, builder, cluster);

  if (builder && !cluster && getThreadSyncInfo(op).requiresWarpBefore() &&
      !membarInfo->warpsSynced) {
    builder->setInsertionPoint(op);
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local,
                                   triton::gpu::BarrierScope::Warp);
    membarInfo->syncWarps();
  }

  if (scratchSlice) {
    if (barrierStages.betweenMemoryEffects) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      bool hasThreadEffects = !curBlockInfo.threadSync.effectIssuers.empty();
      membarInfo->addBlockInfo(curBlockInfo);
      membarInfo->sync();
      curBlockInfo.sync();
      // Final scratch reads still need to finish before a later publication.
      if (hasThreadEffects)
        curBlockInfo = getThreadEffects(op);
    }
    curBlockInfo.syncReadSlices[*scratchSlice].insert(op);
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  membarInfo->addBlockInfo(curBlockInfo);

  if (barrierStages.afterMemoryEffects) {
    // Model a trailing barrier after handling the operation's effects.
    membarInfo->sync();
  }
}

void ModuleMembarAnalysis::run() { run<MembarAnalysis>(); }

} // namespace mlir
