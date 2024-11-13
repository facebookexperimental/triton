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
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <unordered_set>

#define DEBUG_TYPE "triton-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

static int getWarpGroupId(Operation *op) {
  if (!op->hasAttr("async_task_id"))
    return -1;
  auto producerTask = op->getAttrOfType<DenseIntElementsAttr>("async_task_id");
  if (producerTask.getValues<int>().size() > 1)
    return -1;
  return producerTask.getValues<int>()[0];
}

static bool isHeavyComputation(Operation *op) {
  // include exp2, mulf, addf 1D. Somehow we don't go through reduction
  // when checking dependencies
  if (!isa<arith::MulFOp>(op) && !isa<math::Exp2Op>(op) &&
      !isa<arith::AddFOp>(op) && !isa<mlir::triton::ReduceOp>(op))
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!tensorTy)
    return false;
  if (tensorTy.getRank() < 1)
    return false;
  return true;
}

static Value createGetAsyncTaskId(OpBuilder &builder, Operation *op) {
  // builder.create<ttng::GetAsyncTaskIdOp>(loc);
  auto loc = op->getLoc();
  Value _4 = builder.create<arith::ConstantIntOp>(loc, 4, 32);
  Value warpId = builder.create<ttng::GetCanonicalWarpIdOp>(loc);
  Value asyncTaskId = builder.create<arith::DivUIOp>(loc, warpId, _4);
  return asyncTaskId;
}

#define GEN_PASS_DEF_TRITONGPUPINGPONGSYNC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUPingPongSyncPass
    : public impl::TritonGPUPingPongSyncBase<TritonGPUPingPongSyncPass> {
public:
  using impl::TritonGPUPingPongSyncBase<
      TritonGPUPingPongSyncPass>::TritonGPUPingPongSyncBase;

  enum class ResourceType {
    Gemm,
    OtherComp,
  };

  void getNestedFor(scf::IfOp ifOp, SmallVector<scf::ForOp> &loops) {
    ifOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  }
  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numConsumerGroups != 2)
      return;

    // Insert sync points in ForOp for consumer warp groups.
    // Enable this pass when number of consumer warp groups > 1 and the consumer
    // warp groups run the same code.
    SmallVector<scf::ForOp> loops;
    // Identify ForOps for consumer warp groups.
    getOperation()->walk([&](scf::IfOp ifOp) {
      int wgId = getWarpGroupId(ifOp);
      if (wgId == 1 || wgId == 2) {
        // get ForOp nested under.
        getNestedFor(ifOp, loops);
        if (mlir::triton::tools::getBoolEnv("DATA_PARTITION_FA")) {
          // Goes through ConvertLayout. ExperimentalDescriptorStore will need
          // scratch smem.
          ifOp->walk([&](Operation *op) {
            if (llvm::isa<triton::gpu::ConvertLayoutOp>(op) ||
                llvm::isa<tt::ExperimentalDescriptorStoreOp>(op)) {
              op->setAttr(
                  "allocation.copy",
                  IntegerAttr::get(IntegerType::get(op->getContext(), 32), 1));
            }
          });
        }
      }
    });

    if (loops.size() != 1)
      return;

    // Step 2: handles data partitioning along program_id.
    // Go through all ops in the function, replace program_id
    // if (warp_group_id == 1) 2*tl.program_id(0)
    // if (warp_group_id == 2) 2*tl.program_id(0)+1
    // i.e num_consumer_wgs * tl.program_id(0) + (warp_group_id - 1)
    // for consumer wgs only.
    // Done replacing program_id. Start adding barriers for pingpong.
    if (!mlir::triton::tools::getBoolEnv("ENABLE_PINGPONG"))
      return;

    // For warp group 1:
    //   At beginning of tensor core ops: bar.sync 9, 256
    //   At end of tensor core ops: bar.arrive 10, 256
    // For warp group 2:
    // Under ifOp for consumer, prior to the forOp, add an ifOp with
    // bar.arrive 9, 256
    //   At beginning of tensor core ops: bar.sync 10, 256
    //   At end of tensor core ops: bar.arrive 9, 256
    // Identify beginning of tensor core ops.
    Operation *startOfGemm = nullptr;
    Operation *endOfGemm = nullptr;
    auto forOp = loops[0];
    OpBuilder builder(forOp);
    for (auto &op : forOp.getBody()->without_terminator()) {
      // Find start of tensor core ops first. Then find end of tensor core ops
      // (i.e start of cuda core ops).
      if (startOfGemm && endOfGemm)
        break;
      bool isCudaCore = isHeavyComputation(&op);
      if (op.hasTrait<OpTrait::DotLike>() && !isCudaCore &&
          startOfGemm == nullptr) {
        startOfGemm = &op;
        continue;
      }
      if (!op.hasTrait<OpTrait::DotLike>() && isCudaCore && startOfGemm) {
        endOfGemm = &op;
        break;
      }
    }
    if (!startOfGemm || !endOfGemm)
      return;

    LLVM_DEBUG({
      LDBG("found start of tensor core ops");
      startOfGemm->dump();
    });
    LLVM_DEBUG({
      LDBG("found end of tensor core ops");
      endOfGemm->dump();
    });

    // Add ifOp right before forOp
    builder.setInsertionPoint(forOp);
    auto forLoc = forOp->getLoc();
    Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);
    Value c_9 = builder.create<arith::ConstantIntOp>(forLoc, 9, 32);
    Value c_2 = builder.create<arith::ConstantIntOp>(forLoc, 2, 32);
    Value curWarpGroupId = createGetAsyncTaskId(builder, forOp);
    auto pred = builder.create<arith::CmpIOp>(forLoc, arith::CmpIPredicate::eq,
                                              curWarpGroupId, c_2);
    auto ifOp = builder.create<scf::IfOp>(forLoc, pred, /*else=*/false);
    builder.setInsertionPoint(ifOp.thenYield());
    builder.create<ttng::NamedBarrierArriveOp>(forLoc, c_9, numThreads);

    builder.setInsertionPoint(startOfGemm);
    auto loc = startOfGemm->getLoc();

    Value c_8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
    Value syncBarrier = builder.create<arith::AddIOp>(loc, c_8, curWarpGroupId);
    // 8 + warpGroupId
    builder.create<ttng::NamedBarrierWaitOp>(loc, syncBarrier, numThreads);

    builder.setInsertionPoint(endOfGemm);
    auto loc2 = endOfGemm->getLoc();
    Value c_11 = builder.create<arith::ConstantIntOp>(loc2, 11, 32);
    Value arriveBarrier =
        builder.create<arith::SubIOp>(loc2, c_11, curWarpGroupId);
    // 11 - warpGroupId
    builder.create<ttng::NamedBarrierArriveOp>(loc2, arriveBarrier, numThreads);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
