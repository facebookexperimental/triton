#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <set>

#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define DEBUG_TYPE "tritongpu-warp-spec-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static Value createThreadIdOp(OpBuilder &builder, Location loc) {
  Value threadId = builder.create<::mlir::gpu::ThreadIdOp>(
      loc, builder.getIndexType(), ::mlir::gpu::Dimension::x);
  auto cast = builder.create<UnrealizedConversionCastOp>(
      loc, TypeRange{builder.getIntegerType(32)}, ValueRange{threadId});
  return cast.getResult(0);
}

// Lower to use GetCanonicalWarpIdOp.
// In Hopper, each task is a warpgroup consisting of 4 warps.
static const int WARPS_PER_TASK = 4;
static const int THREADS_PER_TASK = 128;
void lowerGetAsyncTaskIdOp(Operation *parentOp, int numConsumerGroups) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::GetAsyncTaskIdOp op) {
    auto loc = op.getLoc();
    OpBuilder builder(op);
    Value _4 = builder.create<arith::ConstantIntOp>(loc, WARPS_PER_TASK, 32);
    Value warpId = builder.create<ttng::GetCanonicalWarpIdOp>(loc);
    Value asyncTaskId = builder.create<arith::DivUIOp>(loc, warpId, _4);
    op.getResult().replaceAllUsesWith(asyncTaskId);

    LLVM_DEBUG({
      LDBG("erasing GetAsyncTask");
      op->dump();
    });
    eraseOps.insert(op);
  });
  for (Operation *op : eraseOps)
    op->erase();
}

Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool emptyBarrier) {
  auto loc = op->getLoc();
  assert(isa<ttng::ProducerAcquireOp>(op) || isa<ttng::ConsumerWaitOp>(op));
  Value curPhase;
  if (auto acq = dyn_cast<ttng::ProducerAcquireOp>(op))
    curPhase = acq.getPhase();
  else if (auto wait = dyn_cast<ttng::ConsumerWaitOp>(op))
    curPhase = wait.getPhase();
  if (emptyBarrier) {
    // curPhase = curPhase xor True for emptyBarrier.
    Value _1_1b = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    curPhase = builder.create<mlir::arith::XOrIOp>(loc, curPhase, _1_1b);
  }
  LLVM_DEBUG(curPhase.dump());
  return curPhase;
}

void processProducerAcquireOp(OpBuilder &builder, ttng::ProducerAcquireOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, true);
  auto i32Ty = builder.getIntegerType(32);
  phase = builder.create<arith::ExtUIOp>(loc, i32Ty, phase);
  auto waitOp = builder.create<ttng::WaitBarrierOp>(loc, bufferEmpty, phase);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processProducerCommitOp(OpBuilder &builder, ttng::ProducerCommitOp op,
                             Value bufferFull, ttng::TokenLoadType loadType) {
  auto loc = op.getLoc();
  int txCnt = 0;
  ttng::MBarrierArriveOp arriveOp;

  if (loadType == ttng::TokenLoadType::TMALoadOp) {
    // Only thread 0 arrives for TMA load.
    Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value threadId = createThreadIdOp(builder, loc);
    Value pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               threadId, _0);
    arriveOp = builder.create<ttng::MBarrierArriveOp>(
        loc, bufferFull, pred, /*remoteCTAId*/ nullptr, /*trackAsyncOp*/ false,
        txCnt);
  } else {
    // Each thread arrives.
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    arriveOp = builder.create<ttng::MBarrierArriveOp>(
        loc, bufferFull, pred, /*remoteCTAId*/ nullptr, /*trackAsyncOp*/ true,
        txCnt);
  }

  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

void processConsumerWaitOp(OpBuilder &builder, ttng::ConsumerWaitOp op,
                           Value bufferFull) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, false);
  auto i32Ty = builder.getIntegerType(32);
  phase = builder.create<arith::ExtUIOp>(loc, i32Ty, phase);
  auto waitOp = builder.create<ttng::WaitBarrierOp>(loc, bufferFull, phase);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processConsumerReleaseOp(OpBuilder &builder, ttng::ConsumerReleaseOp op,
                              Value bufferEmpty, int numCTAs) {
  auto loc = op.getLoc();
  auto arriveOp = builder.create<ttng::MBarrierArriveOp>(
      loc, bufferEmpty, nullptr, nullptr, false, 0);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

void lowerTokenOperations(Operation *parentOp, int numCTAs,
                          int numConsumerGroups) {
  SmallVector<Operation *> deprecatedOps;
  SmallVector<Operation *> deprecatedTokenOps;
  DenseSet<Operation *> warpSpecOps;
  DenseMap<Operation *, Value> tokenToFull;
  DenseMap<Operation *, Value> tokenToEmpty;
  parentOp->walk([&](ttng::CreateTokenOp createTokenOp) {
    ttng::TokenLoadType loadType = createTokenOp.getLoadType();
    MLIRContext *context = createTokenOp.getContext();
    OpBuilder builder(createTokenOp);
    Location loc = createTokenOp.getLoc();

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto barrierCTALayout =
        ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, {0}, barrierCTALayout);
    Type barrierMemDescType =
        ttg::MemDescType::get({createTokenOp.getNum()}, builder.getI64Type(),
                              barrierEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    Type singleBarrierMemDescType =
        ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    // These are created prior to warp_specialize.
    Value bufferFullArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    Value bufferEmptyArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    tokenToFull[createTokenOp.getOperation()] = bufferFullArray;
    tokenToEmpty[createTokenOp.getOperation()] = bufferEmptyArray;

    for (unsigned i = 0; i < createTokenOp.getNum(); i++) {
      Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
      Value barrierFullView = builder.create<ttg::MemDescSubviewOp>(
          loc, singleBarrierMemDescType, bufferFullArray, idx);
      unsigned bufferFullCount =
          loadType == ttng::TokenLoadType::TMALoadOp ? 1 : THREADS_PER_TASK;
      builder.create<ttng::InitBarrierOp>(loc, barrierFullView,
                                          bufferFullCount);

      Value barrierEmptyView = builder.create<ttg::MemDescSubviewOp>(
          loc, singleBarrierMemDescType, bufferEmptyArray, idx);
      builder.create<ttng::InitBarrierOp>(loc, barrierEmptyView,
                                          THREADS_PER_TASK);
    }

    assert(numCTAs == 1 && "remote CTA is not supported yet");
    builder.create<mlir::gpu::BarrierOp>(loc);

    // Helper function for extracting one index from bufferFullArray.
    auto extractBufferFull = [&](Location loc, Value idx) -> Value {
      return builder.create<ttg::MemDescSubviewOp>(
          loc, singleBarrierMemDescType, bufferFullArray, idx);
    };

    // Helper function for extracting one index from bufferEmptyArray.
    auto extractBufferEmpty = [&](Location loc, Value idx) -> Value {
      return builder.create<ttg::MemDescSubviewOp>(
          loc, singleBarrierMemDescType, bufferEmptyArray, idx);
    };
    auto handleOneUser = [&](Operation *user) -> bool {
      // Here builder is at the user, make sure usage of values outside of
      // warp_specialize is via capture if user is in a partition region.
      // We need bufferFullArray and bufferEmptyArray.
      if (auto op = dyn_cast<ttng::ProducerAcquireOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        auto pOp = user->getParentOp();
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        processProducerAcquireOp(builder, op, bufferEmpty);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttng::ProducerCommitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        processProducerCommitOp(builder, op, bufferFull, loadType);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttng::ConsumerWaitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        processConsumerWaitOp(builder, op, bufferFull);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttng::ConsumerReleaseOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        processConsumerReleaseOp(builder, op, bufferEmpty, numCTAs);
        deprecatedOps.push_back(user);
        return true;
      }
      return false;
    };

    // Process token users: ProducerAcquireOp, ProducerCommitOp, ConsumerWaitOp,
    // and ConsumerReleaseOp.
    for (OpOperand &use : createTokenOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      bool handled = handleOneUser(user);
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned opndNum = use.getOperandNumber();
        // Handle the regions. Trace uses of the argument corresponding to the
        // captured value.
        for (Region *region : wsOp.getPartitionRegions()) {
          LDBG("-- region " << region->getNumArguments());
          auto tArg = region->getArgument(opndNum);
          for (Operation *tUser : tArg.getUsers()) {
            builder.setInsertionPoint(tUser);
            // Use of TokenOp via capture of warp_specialize.
            handleOneUser(tUser);
          }
        }
        warpSpecOps.insert(user);
      } else if (!handled) {
        llvm_unreachable("Unexpected user of token");
      }
    }

    deprecatedTokenOps.push_back(createTokenOp);
  });
  for (auto op : deprecatedOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    op->erase();
  }
  unsigned tokenRemoval = 0;
  // Map from tokenOp to bufferFullArray, bufferEmptyArray.
  // If a tokenOp is used by warp_specialize, remove it and add
  // buffer[Full|Empty]Array.

  for (auto op : deprecatedTokenOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    ++tokenRemoval;
    if (auto tokenOp = dyn_cast<ttng::CreateTokenOp>(op)) {
      // Check to see if it is used by warpSpec. If yes, eraseOperand and
      // eraseArgument.
      for (OpOperand &use : tokenOp->getUses()) {
        Operation *user = use.getOwner();
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
          unsigned opndNum = use.getOperandNumber();
          LDBG("wsOp user numOperands: " << wsOp->getNumOperands() << " idx "
                                         << opndNum);

          LLVM_DEBUG({
            LDBG("prior to erasing " << tokenRemoval);
            parentOp->dump();
          });
          wsOp->eraseOperand(opndNum);
          Value empty = tokenToEmpty[op];
          Value full = tokenToFull[op];
          wsOp->insertOperands(wsOp.getNumOperands(), full);
          wsOp->insertOperands(wsOp.getNumOperands(), empty);
          // Handle the regions.
          for (Region *region : wsOp.getPartitionRegions()) {
            LDBG("-- region " << region->getNumArguments());
            auto tArg = region->getArgument(opndNum);
            for (Operation *tUser : tArg.getUsers()) {
              LLVM_DEBUG({
                LDBG("user for arg");
                tUser->dump();
              });
            }
            region->eraseArgument(opndNum);
            BlockArgument arg =
                region->addArgument(full.getType(), full.getLoc());
            replaceAllUsesInRegionWith(full, arg, *region);
            BlockArgument arg2 =
                region->addArgument(empty.getType(), empty.getLoc());
            replaceAllUsesInRegionWith(empty, arg2, *region);
          }
        }
      }
    }
    op->erase();
  }

  assert(numCTAs == 1 && "remote CTA is not supported yet");
  LLVM_DEBUG({
    LDBG("after lowering");
    parentOp->dump();
  });
}

#define GEN_PASS_DEF_TRITONGPUWSLOWERING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// This pass lowers WS-specific operations.
class TritonGPUWSLowering
    : public impl::TritonGPUWSLoweringBase<TritonGPUWSLowering> {
public:
  using impl::TritonGPUWSLoweringBase<
      TritonGPUWSLowering>::TritonGPUWSLoweringBase;

  void runOnOperation() override {
    // Disable WarpSpec if numConsumerGroups is zero.
    if (numConsumerGroups == 0)
      return;
    ModuleOp mod = getOperation();
    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);

    lowerGetAsyncTaskIdOp(mod, numConsumerGroups);
    lowerTokenOperations(mod, numCTAs, numConsumerGroups);

    // We assume number of warps per warp group is 4.
    // With Warp Spec, the effective warps per CTA is
    // number of warp groups * 4, but within each warp group, layout will use
    // num_warps of 4, since tensors are not distributed between the groups.
    //
    // Loads usually happen in one producer warp groups. num_warps of 4 makes
    // sense because only the 4 warps from the producer warp group are
    // participating in the load.
    //
    // But at some point (at least when we launch the kernel!) we really do need
    // to know that the CTA has 8 or 12 warps in it. Attribute
    // "num-warp-groups-per-cta" can be used to calculate the total number of
    // warps.
    auto builder = OpBuilder::atBlockBegin(mod.getBody());
    int numWarps = triton::gpu::lookupNumWarps(mod);
    mod->setAttr("ttg.total-num-warps",
                 builder.getI32IntegerAttr(numWarps * (1 + numConsumerGroups)));
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
