#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <limits>

namespace mlir {

#define GEN_PASS_DEF_NVGPULOWERCLUSTERREDUCETOTTGIR
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

using ClusterReduceOp = ttng::ClusterReduceOp;

SmallVector<Value> cloneCombine(OpBuilder &builder, ClusterReduceOp op,
                                ValueRange lhs, ValueRange rhs) {
  Block &combine = op.getCombineOp().front();
  unsigned arity = op.getPartials().size();
  IRMapping mapping;
  for (unsigned i = 0; i < arity; ++i) {
    mapping.map(combine.getArgument(i), lhs[i]);
    mapping.map(combine.getArgument(arity + i), rhs[i]);
  }
  for (Operation &nested : combine.without_terminator())
    builder.clone(nested, mapping);
  return llvm::map_to_vector(
      combine.getTerminator()->getOperands(),
      [&](Value result) { return mapping.lookupOrDefault(result); });
}

SmallVector<Value> combineValues(OpBuilder &builder, ClusterReduceOp op,
                                 ValueRange lhs, ValueRange rhs) {
  if (!isa<RankedTensorType>(lhs.front().getType()))
    return cloneCombine(builder, op, lhs, rhs);

  assert(lhs.size() == 1 && rhs.size() == 1 &&
         "tensor cluster reductions have one component");
  Block &combine = op.getCombineOp().front();
  Operation *scalarCombiner =
      combine.getTerminator()->getOperand(0).getDefiningOp();
  assert(scalarCombiner && scalarCombiner->getNumOperands() == 2 &&
         scalarCombiner->getNumResults() == 1 &&
         "tensor cluster reductions require a binary scalar combiner");

  OperationState state(op.getLoc(), scalarCombiner->getName());
  state.addOperands({lhs.front(), rhs.front()});
  state.addTypes(lhs.front().getType());
  state.addAttributes(scalarCombiner->getAttrs());
  return {builder.create(state)->getResult(0)};
}

SmallVector<Value> reduceTuple(OpBuilder &builder, ClusterReduceOp op,
                               ValueRange tensors) {
  auto reduce = triton::ReduceOp::create(builder, op.getLoc(), tensors,
                                         /*axis=*/0,
                                         /*reduction_ordering=*/nullptr);
  unsigned arity = tensors.size();
  SmallVector<Type> argTypes(2 * arity, builder.getF32Type());
  SmallVector<Location> argLocs(2 * arity, op.getLoc());
  Block *body =
      builder.createBlock(&reduce.getCombineOp(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);
  auto combined =
      cloneCombine(builder, op, body->getArguments().take_front(arity),
                   body->getArguments().take_back(arity));
  triton::ReduceReturnOp::create(builder, op.getLoc(), combined);
  builder.setInsertionPointAfter(reduce);
  SmallVector<Value> results;
  for (Value result : reduce.getResults())
    results.push_back(result);
  return results;
}

LogicalResult lowerCollective(ClusterReduceOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  int64_t groupSize = op.getGroupSize();
  int64_t arity = op.getPartials().size();
  Type componentType = op.getPartials().front().getType();
  auto tensorType = dyn_cast<RankedTensorType>(componentType);
  int64_t elements = tensorType ? tensorType.getNumElements() : 1;
  int64_t expectedBytes = sizeof(float) * elements * arity * (groupSize - 1);
  if (expectedBytes > std::numeric_limits<int32_t>::max())
    return op.emitError("cluster-reduce transaction byte count exceeds i32");

  SmallVector<int64_t> slotShape = tensorType
                                       ? llvm::to_vector(tensorType.getShape())
                                       : SmallVector<int64_t>{1};
  unsigned slotRank = slotShape.size();
  SmallVector<unsigned> slotOrder;
  for (unsigned dim = slotRank; dim > 0; --dim)
    slotOrder.push_back(dim - 1);
  SmallVector<unsigned> slotOnes(slotRank, 1);
  auto slotCTA = ttg::CGAEncodingAttr::fromSplitParams(context, slotOnes,
                                                       slotOnes, slotOrder);
  auto slotEncoding = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1,
                                                           slotOrder, slotCTA);

  SmallVector<int64_t> bufferShape{groupSize};
  llvm::append_range(bufferShape, slotShape);
  unsigned bufferRank = bufferShape.size();
  SmallVector<unsigned> bufferOrder;
  for (unsigned dim = bufferRank; dim > 0; --dim)
    bufferOrder.push_back(dim - 1);
  SmallVector<unsigned> bufferOnes(bufferRank, 1);
  auto bufferCTA = ttg::CGAEncodingAttr::fromSplitParams(
      context, bufferOnes, bufferOnes, bufferOrder);
  auto bufferEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, bufferOrder, bufferCTA);
  auto smemSpace = ttg::SharedMemorySpaceAttr::get(context);
  auto barrierCTA =
      ttg::CGAEncodingAttr::fromSplitParams(context, {1}, {1}, {0});
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTA);
  RankedTensorType exchangeType;
  if (tensorType) {
    exchangeType = tensorType;
  } else {
    int numWarps = ttg::lookupNumWarps(op);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
    auto exchangeEncoding = ttg::BlockedEncodingAttr::get(
        context, {1}, {static_cast<unsigned>(threadsPerWarp)},
        {static_cast<unsigned>(numWarps)}, {0}, slotCTA);
    exchangeType =
        RankedTensorType::get({1}, builder.getF32Type(), exchangeEncoding);
  }
  auto bufferType = ttg::MemDescType::get(bufferShape, builder.getF32Type(),
                                          bufferEncoding, smemSpace, true);
  auto slotType = ttg::MemDescType::get(slotShape, builder.getF32Type(),
                                        slotEncoding, smemSpace, true);

  Value rank =
      triton::nvgpu::ClusterCTAIdOp::create(builder, loc, builder.getI32Type());
  SmallVector<Value> partials, buffers, localSlots;
  for (Value partial : op.getPartials()) {
    partials.push_back(tensorType ? partial
                                  : triton::SplatOp::create(
                                        builder, loc, exchangeType, partial));
    Value buffer = ttg::LocalAllocOp::create(builder, loc, bufferType, Value());
    buffers.push_back(buffer);
    localSlots.push_back(
        ttg::MemDescIndexOp::create(builder, loc, slotType, buffer, rank));
  }

  auto barrierType = ttg::MemDescType::get({1}, builder.getI64Type(),
                                           barrierEncoding, smemSpace, true);
  Value barrier = ttg::LocalAllocOp::create(builder, loc, barrierType, Value());
  ttng::InitBarrierOp::create(builder, loc, barrier, 1);
  Value pred = arith::ConstantIntOp::create(builder, loc, 1, 1);
  ttng::BarrierExpectOp::create(builder, loc, barrier, expectedBytes, pred);
  ttng::ClusterArriveOp::create(builder, loc, false);
  ttng::ClusterWaitOp::create(builder, loc);

  for (auto [partial, slot] : llvm::zip(partials, localSlots))
    ttg::LocalStoreOp::create(builder, loc, partial, slot);
  for (int peer = 0; peer < groupSize; ++peer) {
    Value peerRank = arith::ConstantIntOp::create(builder, loc, peer, 32);
    Value isRemote = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, rank, peerRank);
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isRemote,
                                  /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    for (auto [partial, slot] : llvm::zip(partials, localSlots))
      ttg::AsyncRemoteShmemStoreOp::create(builder, loc, partial, slot,
                                           peerRank, barrier);
    builder.setInsertionPointAfter(ifOp);
  }

  Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
  ttng::WaitBarrierOp::create(builder, loc, barrier, phase, pred);
  auto loadRank = [&](int peer) {
    Value peerRank = arith::ConstantIntOp::create(builder, loc, peer, 32);
    SmallVector<Value> loaded;
    for (Value buffer : buffers) {
      Value slot =
          ttg::MemDescIndexOp::create(builder, loc, slotType, buffer, peerRank);
      loaded.push_back(
          ttg::LocalLoadOp::create(builder, loc, exchangeType, slot));
    }
    return tensorType ? loaded : reduceTuple(builder, op, loaded);
  };

  SmallVector<Value> results = loadRank(0);
  for (int peer = 1; peer < groupSize; ++peer)
    results = combineValues(builder, op, results, loadRank(peer));
  ttg::BarrierOp::create(builder, loc, ttg::AddrSpace::Local);
  op.replaceAllUsesWith(results);
  op.erase();
  return success();
}

class NVGPULowerClusterReduceToTTGIRPass
    : public impl::NVGPULowerClusterReduceToTTGIRBase<
          NVGPULowerClusterReduceToTTGIRPass> {
public:
  void runOnOperation() override {
    SmallVector<ClusterReduceOp> collectives;
    getOperation().walk([&](ClusterReduceOp op) { collectives.push_back(op); });
    // Validate the complete set before mutating IR so a malformed later
    // collective cannot leave the module partially lowered.
    for (ClusterReduceOp op : collectives)
      if (failed(verify(op.getOperation())))
        return signalPassFailure();
    for (ClusterReduceOp op : collectives)
      if (failed(lowerCollective(op)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir
