#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <deque>
#include <limits>
#include <optional>

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
  if (dim0 <= 0 || totalSize % dim0 != 0)
    return interval;

  size_t stride = totalSize / dim0;
  size_t newStart = interval.start() + indexVal.getSExtValue() * stride;
  return Interval<size_t>(newStart, newStart + stride);
}

static bool isConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) && constant == expected;
}

// Recognize only parity of the induction value supplied by discoverStageLoops.
// In particular, a scalar stage index unrelated to that value is not a proof
// that different waves access the same stage.
static std::optional<unsigned> getStageParity(Value value, Value induction,
                                              unsigned depth = 0) {
  if (!induction || depth > 3)
    return std::nullopt;
  if (auto rem = value.getDefiningOp<arith::RemSIOp>()) {
    if (rem.getLhs() == induction && isConstantInt(rem.getRhs(), 2))
      return 0;
  }
  if (auto bitAnd = value.getDefiningOp<arith::AndIOp>()) {
    if ((bitAnd.getLhs() == induction && isConstantInt(bitAnd.getRhs(), 1)) ||
        (bitAnd.getRhs() == induction && isConstantInt(bitAnd.getLhs(), 1)))
      return 0;
  }
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
    if (auto parity = getStageParity(complemented, induction, depth + 1))
      return *parity ^ 1;
  return std::nullopt;
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
    if (!isa<triton::gpu::MemDescSubsliceOp,
             triton::gpu::MemDescTransOp>(def))
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
  if (!layout ||
      isa<triton::gpu::PartitionedSharedEncodingAttr>(encoding) ||
      parentType.getRank() != stageType.getRank() + 1 ||
      layout.getRank() != stageType.getRank() ||
      parentType.getShape().front() != 2 ||
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
  // Padding omits the gap after the last element, so interval.size()/2 is not
  // necessarily the byte offset of stage 1.
  unsigned bitWidth = getIntOrFloatOrPtrBitWidth(parentType.getElementType());
  if (bitWidth < 8 || bitWidth % 8)
    return std::nullopt;
  int64_t parentElems = triton::gpu::getAllocationElems(
      encoding, parentType.getAllocShape());
  int64_t stageElems = triton::gpu::getAllocationElems(
      encoding, stageType.getAllocShape());
  if (parentElems <= 0 || stageElems <= 0 ||
      stageElems >= std::numeric_limits<int32_t>::max() ||
      parentElems % 2 || parentElems / 2 != stageElems)
    return std::nullopt;
  int64_t strideElems = stageElems;
  int64_t extentElems = stageElems;
  if (auto padded = triton::gpu::getPaddedEncoding(encoding)) {
    parentElems = padded.getPaddedSize({parentElems});
    extentElems = padded.getPaddedSize({stageElems});
    // getPaddedSize(N+1)-1 includes the padding immediately before element N.
    strideElems = padded.getPaddedSize({stageElems + 1}) - 1;
  }
  if (extentElems <= 0 || strideElems < extentElems ||
      parentElems <= strideElems || parentElems - strideElems != extentElems)
    return std::nullopt;
  size_t bytesPerElement = bitWidth / 8;
  size_t parentBytes = interval.end() - interval.start();
  if (parentBytes % bytesPerElement ||
      parentBytes / bytesPerElement != static_cast<uint64_t>(parentElems))
    return std::nullopt;
  return StageGeometry{alloc.getResult(), index.getIndex(),
                       static_cast<size_t>(strideElems) * bytesPerElement};
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
    // We know there aren't subslices before the one because of subslice::fold
    // Still need to check this for where a fold isn't possible (control flow)
    // and when a subslice is carried in a loop
    if (accessTy.getAllocShape() == subslice.getSrc().getType().getShape()) {
      subsliceOffsets = SmallVector<int64_t>(subslice.getOffsets());
    }
  }

  auto geometry =
      getStageGeometry(value, allocationInterval, bufferId, allocation);
  if (!geometry)
    return;
  std::optional<unsigned> parity;
  if (isConstantInt(geometry->index, 0))
    parity = 0;
  else if (isConstantInt(geometry->index, 1))
    parity = 1;
  else if (isStableStageParent(geometry->parent, stageBasis)) {
    parity = getStageParity(geometry->index, stageBasis);
    if (parity)
      stage.basis = stageBasis;
  }
  if (!parity)
    return;
  stage.parent = geometry->parent;
  stage.parentInterval = allocationInterval;
  stage.stride = geometry->stride;
  stage.parity = *parity;
}

AllocationSlice AllocationSlice::enterStageLoop(Value induction,
                                                 unsigned initialParity) const {
  auto result = forgetStageLoop();
  if (!result.stage.parent ||
      !isStableStageParent(result.stage.parent, induction))
    return result;
  result.stage.basis = induction;
  result.stage.parity ^= initialParity;
  // A lifted entry access now denotes either physical stage. Keeping its old
  // constant interval would incorrectly prove later-iteration accesses disjoint.
  result.allocationInterval = result.stage.parentInterval;
  return result;
}

AllocationSlice AllocationSlice::advanceStageLoop(Value induction) const {
  if (!stage.basis)
    return *this;
  if (stage.basis != induction)
    return forgetStageLoop();
  auto result = *this;
  // old parity(iv) = new parity(iv) XOR 1 for the recognized +1 latch.
  result.stage.parity ^= 1;
  return result;
}

AllocationSlice AllocationSlice::forgetStageLoop() const {
  auto result = *this;
  if (stage.basis) {
    result.allocationInterval = stage.parentInterval;
    result.stage = {};
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
      stage.parity != other.stage.parity)
    return false;

  // If access types are unknown, assume intersection
  if (!accessTy || !other.accessTy)
    return true;

  // If offsets are unknown, conservatively assume overlap
  if (subsliceOffsets.empty() || other.subsliceOffsets.empty())
    return true;

  // If layouts differ, we assume intersection as we currently only work on
  // logical elements
  if (accessTy.getEncoding() != other.accessTy.getEncoding())
    return true;

  auto shapeA = SmallVector<int64_t>(accessTy.getShape());
  auto shapeB = SmallVector<int64_t>(other.accessTy.getShape());
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

void MembarOrFenceAnalysis::discoverStageLoops(FunctionOpInterface function) {
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

    stageLoops.push_back(
        {entry, &header, body, induction, static_cast<unsigned>(initial[0])});
  }
}

Value MembarOrFenceAnalysis::getStageBasis(Operation *operation) const {
  for (const StageLoop &loop : stageLoops)
    if (operation->getBlock() == loop.body)
      return loop.induction;
  return {};
}

BlockInfo MembarOrFenceAnalysis::transferStageLoops(const BlockInfo &info,
                                                    Block *from,
                                                    Block *to) const {
  for (const StageLoop &loop : stageLoops) {
    if (from == loop.entry && to == loop.header)
      return info.mapSlices([&](const AllocationSlice &slice) {
        return slice.enterStageLoop(loop.induction, loop.initialParity);
      });
    if (from == loop.body && to == loop.header)
      return info.mapSlices([&](const AllocationSlice &slice) {
        return slice.advanceStageLoop(loop.induction);
      });
    if (from == loop.header && to == loop.body)
      return info;
  }
  // No symbolic coordinates escape the recognized edges. In particular the
  // exit can contain either physical stage, including on a zero-trip path.
  return info.mapSlices(
      [](const AllocationSlice &slice) { return slice.forgetStageLoop(); });
}

void MembarOrFenceAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarOrFenceAnalysis::resolve(FunctionOpInterface funcOp,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  discoverStageLoops(funcOp);
  // Initialize the blockList. Operations are organized into "virtual blocks",
  // which represent segments of straight-line code analyzed by each iteration
  // of the dataflow analysis. Virtual blocks abstract over both control flow
  // represented by basic blocks and block successors (i.e. `BranchOpInterface`)
  // and control flow represented by regions (i.e. `RegionBranchOpInterface`).
  //
  // A virtual block consists of a parent block and a starting iterator, where
  // the virtual block starts on the operation *after* the starting iterator. A
  // null iterator is used to represent the beginning of the block. The virtual
  // block ends at any region branch operation or the basic block terminator.
  // Thus, basic blocks are broken up into multiple virtual blocks at each
  // region operation.
  //
  // Entry virtual blocks are represented by a null iterator. Populate the
  // blockList with the entry virtual blocks in the function. Then, each
  // iteration scans until a terminator or region branch operation is found.
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;
  // Start the analysis from the entry block of the function.
  blockList.emplace_back(&funcOp.getBlocks().front(), Block::iterator());

  // A fixed point algorithm
  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        block.second.isValid() ? std::next(block.second) : block.first->begin();
    for (Operation &op : llvm::make_range(startIt, block.first->end())) {
      // Update inputBlockInfo based on the current operation. Note that we do
      // this before we process terminators and branch-like ops, because some of
      // them (e.g. WarpSpecializePartitionsOp) may have synchronizing effects.
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block. The block transfer function is not monotonic,
    // so overwrite the output state entirely.
    outputBlockInfoMap[block] = inputBlockInfo;
    // Update the successors
    for (VirtualBlock successor : successors) {
      auto transferred = transferStageLoops(outputBlockInfoMap[block],
                                           block.first, successor.first);
      inputBlockInfoMap[successor].join(transferred);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](triton::ReturnOp returnOp) {
    // A basic block can be broken into several virtual blocks. Find all virtual
    // blocks that belong to the basic block containing the return.
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[block, blockInfo] : outputBlockInfoMap) {
      if (block.first == returnOp->getBlock())
        virtualBlocks.emplace_back(block, blockInfo);
    }
    // The return is a terminator, so the virtual block that contains this
    // return starts after all other ones. Find it by comparing the start
    // iterators of the virtual blocks.
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      assert(lhs.first.first == rhs.first.first);
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });

    funcBlockInfo.join(maxIt->second);
  });
}

void MembarOrFenceAnalysis::visitTerminator(
    Operation *op, SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // The successors of an operation with regions can be queried via an
    // interface. The operation branches to the entry blocks of its region
    // successors. It can also branch to after itself.
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isOperation()) {
        successors.emplace_back(br->getBlock(), br->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
  // reason. Check that the parent is actually a `RegionBranchOpInterface`.
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    // Check the successors of a region branch terminator. It can branch to
    // another region of its parent operation or to after the parent op.
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isOperation()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // Otherwise, it could be a return op
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
}

bool containsLocalBarrier(Operation *op) {
  if (isa<triton::AtomicPollOp>(op))
    return true;
  if (auto atomic = dyn_cast<triton::AtomicOpInterface>(op))
    return atomic.getMemSemantic() != triton::MemSemantic::RELAXED;
  if (isa<gpu::BarrierOp>(op))
    return true;
  if (isa<triton::nvidia_gpu::ClusterBarrierOp>(op))
    return true;
  if (isa<triton::nvidia_gpu::ClusterWaitOp>(op))
    return true;
  if (auto arrive = dyn_cast<triton::nvidia_gpu::ArriveBarrierOp>(op))
    return !arrive.getPerThread();
  if (isa<triton::gpu::WarpSpecializePartitionsOp>(op))
    return true;
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op))
    return barrier.hasLocal();
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op))
    return !wgWait.getWarpGroupLocal() && triton::gpu::lookupNumWarps(op) > 4;
  return false;
}

struct LocalBarrierStages {
  // Stages are independent: for example, a release atomic with scratch has
  // both a leading ordering barrier and a scratch rendezvous.
  bool beforeMemoryEffects = false;
  bool afterMemoryEffects = false;
  bool betweenMemoryEffects = false;
};

static Allocation::BufferId getScratchBufferId(Operation *op,
                                               Allocation *allocation) {
  // A call's allocation belongs to the callee and is translated separately.
  if (isa<triton::CallOp>(op))
    return Allocation::InvalidBufferId;
  return allocation->getBufferId(op);
}

static bool scratchBufferUsesWarpSync(Operation *op) {
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;

  auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(cvt.getType());
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  auto kWarp = StringAttr::get(op->getContext(), "warp");
  return mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
}

static LocalBarrierStages getLocalBarrierStages(Operation *op,
                                                Allocation *allocation) {
  LocalBarrierStages stages;
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
    auto sem = atomic.getMemSemantic();
    stages.beforeMemoryEffects = sem == triton::MemSemantic::RELEASE ||
                                 sem == triton::MemSemantic::ACQUIRE_RELEASE;
    stages.afterMemoryEffects =
        !hasScratchBarrier && (sem == triton::MemSemantic::ACQUIRE ||
                               sem == triton::MemSemantic::ACQUIRE_RELEASE);
    // Scalar-result broadcast uses a scratch write, rendezvous, and read for
    // every memory semantic, including relaxed.
    stages.betweenMemoryEffects = hasScratchBarrier;
    return stages;
  }

  // Scratch-backed operations contain a rendezvous between their scratch
  // write and read phases. Other barrier-like operations behave as a barrier
  // immediately before the operation.
  stages.betweenMemoryEffects = hasScratchBarrier;
  stages.beforeMemoryEffects = containsLocalBarrier(op);
  return stages;
}

// Returns true if the same block has a later wait or local barrier before any
// memory effect or nested control flow. Scheduling-only fences carrying
// ttg::SchedulingBarrierOpInterface (e.g. the AMD rocdl.sched.barrier that
// brackets the real ttg.barrier a tlx.workgroup_barrier interposes) have no
// cross-wave memory semantics and are skipped: treating one as a stopping point
// would make Membar insert a redundant barrier right after the async wait --
// doubling the workgroup barrier and stalling a hand-written ping-pong
// schedule.
static bool hasSyncPointBeforeMemoryEffect(Operation *op,
                                           Allocation *allocation) {
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (isa<triton::gpu::SchedulingBarrierOpInterface>(next))
      continue;
    if (isa<ttng::BarrierExpectOp>(next))
      return true;

    auto stages = getLocalBarrierStages(next, allocation);
    if (stages.beforeMemoryEffects ||
        next->hasTrait<mlir::OpTrait::MemWaitOpTrait>())
      return true;

    // A contained barrier follows the operation's incoming shared-memory
    // effects, so it cannot protect those effects from the preceding wait.
    if (stages.betweenMemoryEffects)
      return false;

    // Barriers classified as "after" have no shared-memory effects before
    // them. Currently these are non-scratch atomics and polls.
    if (stages.afterMemoryEffects)
      return true;

    if (isa<RegionBranchOpInterface>(next) || !isMemoryEffectFree(next))
      return false;
  }
  return false;
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<CallOpInterface>(op))
    *blockInfo = blockInfo->mapSlices(
        [](const AllocationSlice &slice) { return slice.forgetStageLoop(); });
  auto arrive = dyn_cast<triton::nvidia_gpu::ArriveBarrierOp>(op);

  // A later CTA-wide synchronization can also synchronize this wait, provided
  // no memory is accessed before reaching it.
  if (auto wgWait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
    if (!wgWait.getWarpGroupLocal() &&
        triton::gpu::lookupNumWarps(wgWait) > 4 &&
        hasSyncPointBeforeMemoryEffect(wgWait, allocation)) {
      wgWait->setAttr("warpGroupLocal", builder->getUnitAttr());
    }
  }

  auto barrierStages = getLocalBarrierStages(op, allocation);
  if (barrierStages.beforeMemoryEffects) {
    // Model a leading local barrier before handling the operation's effects.
    blockInfo->sync();
  }

  // If the current op is an (async) memory wait and there is no later sync
  // point before memory is accessed, insert a barrier op and sync. This avoids
  // redundant barriers by deferring the barrier to the later sync point.
  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
      !hasSyncPointBeforeMemoryEffect(op, allocation)) {
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = getScratchBufferId(op, allocation);
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable())) {
      auto calleeBlockInfo = funcBlockInfoMap->lookup(callee);
      auto callBufferId = allocation->getBufferId(op);
      size_t callOffset = 0;
      if (callBufferId != Allocation::InvalidBufferId)
        callOffset = allocation->getAllocatedInterval(callBufferId).start();
      curBlockInfo = translateBlockInfoToCallsite(calleeBlockInfo, callOffset);
    }
  } else {
    // Intra-function dependencies
    //
    // For perThread ArriveBarrierOp, skip all SMEM hazard tracking.
    // mbarrier.arrive has release semantics and mbarrier.wait has acquire
    // semantics, so no CTA-wide bar.sync is needed before a perThread arrive.
    // Each thread's program order guarantees its own SMEM ops are visible
    // before its arrive, and the mbarrier accumulates all arrivals before
    // releasing the waiter.
    bool isPerThreadArrive = arrive && arrive.getPerThread();

    if (!isPerThreadArrive) {
      if (auto memoryEffectOpInterface =
              dyn_cast<MemoryEffectOpInterface>(op)) {
        // Explicit buffer
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
            effectInstances;
        memoryEffectOpInterface.getEffects(effectInstances);
        for (auto effectInstance : effectInstances) {
          if (auto value = effectInstance.getValue()) {
            for (auto bufferId :
                 allocation->getAllBufferIdsWithAliases(value)) {
              if (bufferId != Allocation::InvalidBufferId) {
                auto interval = allocation->getAllocatedInterval(bufferId);
                auto slice = AllocationSlice(value, interval, bufferId,
                                             allocation, getStageBasis(op));

                if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                  curBlockInfo.syncWriteSlices[slice].insert(op);
                else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                  curBlockInfo.syncReadSlices[slice].insert(op);
              }
            }
          }
        }
      }
    }
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    bool hasExplicitSharedDeps = !curBlockInfo.syncReadSlices.empty() ||
                                 !curBlockInfo.syncWriteSlices.empty();
    if (hasExplicitSharedDeps &&
        !isa<triton::gpu::LocalAtomicScatterRMWOp>(op)) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncWriteSlices[scratchSlice].insert(op);
    auto insertCTABarrier =
        blockInfo->isIntersected(curBlockInfo, filter, allocation);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    if (insertCTABarrier)
      blockInfo->sync();

    if (barrierStages.betweenMemoryEffects) {
      // The internal barrier synchronizes all incoming effects. Do not carry
      // them past the operation; only effects after the barrier are outgoing.
      blockInfo->join(curBlockInfo);
      blockInfo->sync();
      curBlockInfo.sync();
    }
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter, allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);

  if (barrierStages.afterMemoryEffects) {
    // Model a trailing local barrier after handling the operation's effects.
    blockInfo->sync();
  }
}
} // namespace mlir
