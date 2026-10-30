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
  int64_t expectedBytes = sizeof(float) * arity * (groupSize - 1);
  if (expectedBytes > std::numeric_limits<int32_t>::max())
    return op.emitError("cluster-reduce transaction byte count exceeds i32");

  auto cta1d = ttg::CGAEncodingAttr::fromSplitParams(context, {1}, {1}, {0});
  auto smem1d =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, cta1d);
  auto cta2d =
      ttg::CGAEncodingAttr::fromSplitParams(context, {1, 1}, {1, 1}, {1, 0});
  auto smem2d =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {1, 0}, cta2d);
  auto smemSpace = ttg::SharedMemorySpaceAttr::get(context);
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  auto exchangeEncoding = ttg::BlockedEncodingAttr::get(
      context, {1}, {static_cast<unsigned>(threadsPerWarp)},
      {static_cast<unsigned>(numWarps)}, {0}, cta1d);
  auto exchangeType =
      RankedTensorType::get({1}, builder.getF32Type(), exchangeEncoding);
  auto bufferType = ttg::MemDescType::get({groupSize, 1}, builder.getF32Type(),
                                          smem2d, smemSpace, true);
  auto slotType =
      ttg::MemDescType::get({1}, builder.getF32Type(), smem1d, smemSpace, true);

  Value rank =
      triton::nvgpu::ClusterCTAIdOp::create(builder, loc, builder.getI32Type());
  SmallVector<Value> partials, buffers, localSlots;
  for (Value partial : op.getPartials()) {
    partials.push_back(
        triton::SplatOp::create(builder, loc, exchangeType, partial));
    Value buffer = ttg::LocalAllocOp::create(builder, loc, bufferType, Value());
    buffers.push_back(buffer);
    localSlots.push_back(
        ttg::MemDescIndexOp::create(builder, loc, slotType, buffer, rank));
  }

  auto barrierType =
      ttg::MemDescType::get({1}, builder.getI64Type(), smem1d, smemSpace, true);
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
    return reduceTuple(builder, op, loaded);
  };

  SmallVector<Value> results = loadRank(0);
  for (int peer = 1; peer < groupSize; ++peer)
    results = cloneCombine(builder, op, results, loadRank(peer));
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
