#include "CodePartitionUtility.h"
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
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/JSON.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace nvgpu = ::mlir::triton::nvgpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-lower-mem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Can a TMA copy land directly in a buffer with this encoding? Mirrors
// verifyTMAEncoding() in TritonNvidiaGPU/IR/Ops.cpp: the destination of an
// async_tma_copy_global_to_local must be NVMMA shared, untransposed, and (once
// the descriptor carries an encoding) agree with it. A consumer-owned buffer
// can legitimately fail this -- a block-scale load is consumed through a
// #ttg.shared_linear alloc that a memdesc_reshape/trans chain feeds to
// tmem_copy -- in which case the load needs its own NVMMA landing buffer.
static bool isValidTMADestEncoding(Attribute bufferEnc, Attribute descEnc) {
  auto nvmma = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(bufferEnc);
  if (!nvmma || nvmma.getTransposed())
    return false;
  // No descriptor encoding yet: only the NVMMA requirement is checked.
  if (!descEnc)
    return true;
  auto descNvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(descEnc);
  // Encodings may differ in rank for rank-reducing loads, so compare fields.
  return descNvmma && descNvmma.getTransposed() == nvmma.getTransposed() &&
         descNvmma.getSwizzlingByteWidth() == nvmma.getSwizzlingByteWidth() &&
         descNvmma.getElementBitWidth() == nvmma.getElementBitWidth() &&
         descNvmma.getFp4Padded() == nvmma.getFp4Padded();
}

static bool isScheduleRematerialization(Operation *op) {
  return isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp>(op);
}

static void eraseDeadScheduleRematerializations(Value value) {
  SmallVector<Operation *> worklist(value.getUsers());
  SmallVector<Operation *> candidates;
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isScheduleRematerialization(op))
      continue;
    if (!visited.insert(op).second)
      continue;
    candidates.push_back(op);
    for (Value result : op->getResults())
      llvm::append_range(worklist, result.getUsers());
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *&op : llvm::reverse(candidates)) {
      if (op && op->use_empty()) {
        op->erase();
        op = nullptr;
        changed = true;
      }
    }
  }
}

LogicalResult doConvertDescriptorLoadsToNVWS(triton::FuncOp funcOp) {
  // Schedule rematerialization can leave dead per-task clones attached to a
  // descriptor load. Remove only dead layout-rematerialization chains before
  // deriving the replacement local_load's consumer task IDs; whole-function
  // DCE would also erase intentionally dead consumers in test/debug IR.
  SmallVector<tt::DescriptorLoadOp> loads;
  funcOp.walk([&](tt::DescriptorLoadOp load) { loads.push_back(load); });

  for (tt::DescriptorLoadOp load : loads) {
    eraseDeadScheduleRematerializations(load.getResult());
    if (load.getResult().use_empty()) {
      load.erase();
      continue;
    }

    auto tensorType = dyn_cast<RankedTensorType>(load.getType());
    if (!tensorType)
      return load.emitError("expected a ranked tensor descriptor load result");
    Attribute descEnc =
        cast<tt::TensorDescType>(load.getDesc().getType()).getSharedLayout();

    ttg::LocalStoreOp soleStore;
    ttg::LocalAllocOp soleAlloc;
    if (load->hasOneUse())
      soleStore = dyn_cast<ttg::LocalStoreOp>(*load->getUsers().begin());
    if (load->hasOneUse())
      soleAlloc = dyn_cast<ttg::LocalAllocOp>(*load->getUsers().begin());

    // Only reuse the consumer's buffer when the TMA copy can actually target
    // it; otherwise fall through to the register-consumed path below, which
    // allocates an NVMMA buffer and local_loads out of it (what the upstream
    // tt.descriptor_load lowering does for every load).
    if (soleStore && !isValidTMADestEncoding(
                         soleStore.getDst().getType().getEncoding(), descEnc))
      soleStore = nullptr;
    if (soleAlloc &&
        !isValidTMADestEncoding(soleAlloc.getType().getEncoding(), descEnc))
      soleAlloc = nullptr;

    OpBuilderWithAsyncTaskIds builder(load);
    builder.setInsertionPoint(load);
    Value buffer;
    if (soleStore) {
      buffer = soleStore.getDst();
    } else if (soleAlloc) {
      auto oldType = cast<ttg::MemDescType>(soleAlloc.getType());
      auto bufferType = ttg::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), /*mutableMemory=*/true);
      auto newAlloc = builder.createWithAsyncTaskIds<ttg::LocalAllocOp>(
          soleAlloc.getLoc(), bufferType);
      newAlloc->setAttrs(soleAlloc->getAttrs());
      triton::replaceUsesAndPropagateType(builder, soleAlloc,
                                          newAlloc.getResult());
      buffer = newAlloc.getResult();
      builder.setInsertionPointAfter(newAlloc);
    } else {
      auto encoding =
          ttng::getEncodingFromDescriptor(load, tensorType, load.getDesc());
      auto memorySpace = ttg::SharedMemorySpaceAttr::get(load.getContext());
      auto bufferType = ttg::MemDescType::get(
          tensorType.getShape(), tensorType.getElementType(), encoding,
          memorySpace, /*mutableMemory=*/true);
      buffer = builder
                   .createWithAsyncTaskIds<ttg::LocalAllocOp>(load.getLoc(),
                                                              bufferType)
                   .getResult();
    }

    if (Operation *bufferDef = buffer.getDefiningOp();
        bufferDef && bufferDef->getBlock() == load->getBlock() &&
        load->isBeforeInBlock(bufferDef))
      bufferDef->moveBefore(load);

    int64_t txCount =
        ttng::getDescriptorLoadBytes(cast<ttg::MemDescType>(buffer.getType()));
    auto nvwsLoad = builder.createWithAsyncTaskIds<ttnvws::DescriptorLoadOp>(
        load.getLoc(), load.getDesc(), load.getIndices(), txCount, buffer,
        load.getCache(), load.getEvict());
    nvwsLoad->setAttrs(load->getAttrs());

    if (soleStore) {
      soleStore.erase();
    } else if (soleAlloc) {
      soleAlloc.erase();
    } else {
      // Register-consumed load (e.g. the m/Di softmax metadata). Tag the
      // local_load with the *consumer* partitions and their loop schedule --
      // not the producer's -- so the nvws.descriptor_load's SMEM buffer serves
      // directly as the cross-partition channel (one buffer). Inheriting the
      // producer task id instead materializes the value in the producer
      // partition and forces doBufferAllocation to create a *second*
      // cross-partition channel buffer, doubling the SMEM footprint that the
      // original tt.descriptor_load path kept unified (SMEM/TMEM overflow in FA
      // backward).
      builder.setAsyncTaskIdsFromValueUsers(load.getResult());
      builder.setLoopScheduleInfoFromOp(*load->getUsers().begin());
      auto localLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
          load.getLoc(), load.getType(), buffer);
      load.replaceAllUsesWith(localLoad.getResult());
    }
    load.erase();
  }

  bool hasUnconvertedLoad = false;
  funcOp.walk([&](tt::DescriptorLoadOp load) {
    load.emitError("descriptor load was not converted for AutoWS");
    hasUnconvertedLoad = true;
  });
  return failure(hasUnconvertedLoad);
}

Value createBufferView(OpBuilderWithAsyncTaskIds &builder, Value alloc,
                       Value idx) {
  assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
         "Expected MemDescType");
  auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape;
  assert(allocDescType.getShape().size() > 1 &&
         "Expected multi-dimensional memdesc (e.g., Nx...) for subview");
  shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
               allocDescType.getShape().end());
  auto viewDescType = triton::gpu::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory());
  return triton::gpu::MemDescIndexOp::create(builder, alloc.getLoc(),
                                             viewDescType, alloc, idx);
}

namespace {

Value getTMALoadBufferForStage(OpBuilderWithAsyncTaskIds &builder, Value buffer,
                               Value bufferIdx) {
  auto currentView = buffer.getDefiningOp<ttg::MemDescIndexOp>();
  if (!currentView)
    return buffer;
  return createBufferView(builder, currentView.getSrc(), bufferIdx);
}

} // namespace

static Value mapBarrierTo2CTALeader(OpBuilderWithAsyncTaskIds &builder,
                                    Location loc, Value barrier) {
  MLIRContext *ctx = builder.getContext();
  Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
      loc, builder.getI32Type());
  Value negTwo =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, -2, 32);
  Value leaderRank =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, ctaId, negTwo);
  auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      barrierTy.getShape(), barrierTy.getElementType(), barrierTy.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierTy.getMutableMemory(), barrierTy.getAllocShape());
  return builder.createWithAsyncTaskIds<ttng::MapToRemoteBufferOp>(
      loc, remoteTy, barrier, leaderRank);
}

static Value mapBarrierTo2CTAPeer(OpBuilderWithAsyncTaskIds &builder,
                                  Location loc, Value barrier, Value ctaId) {
  MLIRContext *ctx = builder.getContext();
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 32);
  Value peerRank =
      builder.createWithAsyncTaskIds<arith::XOrIOp>(loc, ctaId, one);
  auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      barrierTy.getShape(), barrierTy.getElementType(), barrierTy.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierTy.getMutableMemory(), barrierTy.getAllocShape());
  return builder.createWithAsyncTaskIds<ttng::MapToRemoteBufferOp>(
      loc, remoteTy, barrier, peerRank);
}

static bool useDirectTwoCTAWait(Operation *anchor) {
  auto funcOp = anchor->getParentOfType<tt::FuncOp>();
  bool enabled = false;
  funcOp.walk([&](Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>("tt.autows");
    if (!attr)
      return;
    auto parsed = llvm::json::parse(attr.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return;
    }
    auto *object = parsed->getAsObject();
    if (object)
      enabled |= object->getBoolean("two_cta_tma_direct_wait").value_or(false);
  });
  return enabled;
}

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<ttnvws::DescriptorLoadOp> &tmaLoads,
                            Value barrierAlloc, Value bufferIdx,
                            Value bufferIdxExtract, Value phase,
                            Operation *headProducer, Operation *headConsumer,
                            Operation *headConsumerSameLevel,
                            ArrayRef<int> additionalConsumerTaskIds,
                            DictionaryAttr consumerWaitConstraints) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  // A cooperative two-CTA load moves the pair's bytes through the leader's
  // barrier, so the leader must expect twice the per-CTA transaction count.
  // The group must be homogeneous: mixing protocols would give the fused
  // barrier inconsistent completion routing and expected-byte semantics.
  int64_t sizeInBytes = 0;
  bool twoCTA = tmaLoads.front()->hasAttr(ttng::AttrTwoCTALoadName);
  if (!llvm::all_of(tmaLoads, [twoCTA](ttnvws::DescriptorLoadOp load) {
        return load->hasAttr(ttng::AttrTwoCTALoadName) == twoCTA;
      })) {
    tmaLoads.front().emitError(
        "TMA barrier fusion cannot mix cooperative and per-CTA loads");
    return nullptr;
  }
  for (auto tmaLoad : tmaLoads) {
    bool cooperative = tmaLoad->hasAttr(ttng::AttrTwoCTALoadName);
    sizeInBytes += tmaLoad.getTxCount() * (cooperative ? 2 : 1);
  }

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  builder.setLoopScheduleInfoFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  Value tmaBarrier =
      twoCTA ? mapBarrierTo2CTALeader(builder, loc, prodBarrier) : prodBarrier;
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  // Only the pair leader publishes the expected byte count; the follower's
  // completion is relayed to the leader's barrier.
  Value expectPred = pred;
  if (twoCTA) {
    Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
        loc, builder.getI32Type());
    Value two =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 2, 32);
    Value rankInPair =
        builder.createWithAsyncTaskIds<arith::RemUIOp>(loc, ctaId, two);
    Value zero =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
    expectPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankInPair, zero);
  }
  builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, expectPred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto tmaLoad : tmaLoads) {
    builder.setInsertionPoint(tmaLoad);
    builder.setAsyncTaskIdsFromOp(tmaLoad);
    builder.setLoopScheduleInfoFromOp(tmaLoad);
    Value pipelineBuffer =
        getTMALoadBufferForStage(builder, tmaLoad.getResult(), bufferIdx);
    auto copyOp =
        builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
            tmaLoad.getLoc(), tmaLoad.getDesc(), tmaLoad.getIndices(),
            tmaBarrier, pipelineBuffer, pred);
    if (tmaLoad->hasAttr(ttng::AttrTwoCTALoadName))
      copyOp.setTwoCta(true);
    copy = copyOp;
  }

  // Create a wait_barrier before the first consumer.
  // For data-partitioned channels, shared ops (consBarrier, phase, pred)
  // need ALL consumer task IDs so they survive specializeRegion.
  builder.setInsertionPoint(headConsumerSameLevel);
  SmallVector<int> consumerTaskIds;
  for (int id : getAsyncTaskIds(headConsumer))
    consumerTaskIds.push_back(id);
  for (int id : additionalConsumerTaskIds)
    consumerTaskIds.push_back(id);
  builder.setAsynTaskIdsFromArray(consumerTaskIds);
  builder.setLoopScheduleInfoFromOp(headConsumerSameLevel);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
      loc, builder.getI32Type(), phase);
  Value waitPred =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  bool directTwoCTAWait = twoCTA && useDirectTwoCTAWait(headConsumer);
  Value followerWaitPred;
  Value peerBarrier;
  if (twoCTA) {
    Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
        loc, builder.getI32Type());
    Value two =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 2, 32);
    Value rankInPair =
        builder.createWithAsyncTaskIds<arith::RemUIOp>(loc, ctaId, two);
    Value zero =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
    waitPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankInPair, zero);
    if (!directTwoCTAWait) {
      followerWaitPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, rankInPair, zero);
      peerBarrier = mapBarrierTo2CTAPeer(builder, loc, consBarrier, ctaId);
    }
  }

  // Create one WaitBarrierOp per consumer task ID.
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase, waitPred, /*deps=*/ValueRange{},
      consumerWaitConstraints);
  if (twoCTA && !directTwoCTAWait) {
    // The hardware CTA-group TMA transaction completes the leader's local
    // mbarrier. Relay that completion to the follower's corresponding local
    // barrier, then let the follower wait on it. A cluster barrier is not
    // valid here: only this warp-specialized consumer partition executes the
    // handoff, while cluster barriers require participation from every warp.
    builder.createWithAsyncTaskIds<ttng::FenceAsyncSharedOp>(
        loc, /*bCluster=*/false);
    builder.createWithAsyncTaskIds<ttng::ArriveBarrierOp>(
        loc, peerBarrier, /*count=*/1, waitPred);
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, followerWaitPred, /*deps=*/ValueRange{},
        consumerWaitConstraints);
  }
  for (int extraTaskId : additionalConsumerTaskIds) {
    builder.setAsynTaskIdsFromArray({extraTaskId});
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, waitPred,
        /*deps=*/ValueRange{}, consumerWaitConstraints);
    if (twoCTA && !directTwoCTAWait) {
      builder.createWithAsyncTaskIds<ttng::FenceAsyncSharedOp>(
          loc, /*bCluster=*/false);
      builder.createWithAsyncTaskIds<ttng::ArriveBarrierOp>(
          loc, peerBarrier, /*count=*/1, waitPred);
      builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
          loc, consBarrier, phase, followerWaitPred, /*deps=*/ValueRange{},
          consumerWaitConstraints);
    }
  }

  for (auto tmaLoad : tmaLoads)
    tmaLoad.erase();
  builder.clearLoopScheduleInfo();
  return copy;
}

} // namespace mlir
