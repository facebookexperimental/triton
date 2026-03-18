#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-multi-cta-reduction"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_NVGPUMULTICTAREDUCTION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

static int getNumClusterCTAs(ModuleOp moduleOp) {
  int dimX = 1, dimY = 1, dimZ = 1;
  if (auto attr = moduleOp->getAttrOfType<IntegerAttr>("ttg.cluster-dim-x"))
    dimX = attr.getInt();
  if (auto attr = moduleOp->getAttrOfType<IntegerAttr>("ttg.cluster-dim-y"))
    dimY = attr.getInt();
  if (auto attr = moduleOp->getAttrOfType<IntegerAttr>("ttg.cluster-dim-z"))
    dimZ = attr.getInt();
  return dimX * dimY * dimZ;
}

static SmallVector<triton::ReduceOp> findReduceConsumers(scf::ForOp forOp) {
  SmallVector<triton::ReduceOp> reduces;
  for (auto result : forOp.getResults()) {
    for (auto *user : result.getUsers()) {
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(user))
        reduces.push_back(reduceOp);
    }
  }
  return reduces;
}

/// Transform a multi-CTA annotated loop: partition iterations across CTAs and
/// generate cross-CTA DSM exchange for any downstream tt.reduce consumers.
static LogicalResult transformMultiCTALoop(scf::ForOp forOp,
                                           int numClusterCTAs) {
  if (numClusterCTAs <= 1) {
    forOp->removeAttr("tt.multi_cta");
    return success();
  }

  auto *context = forOp->getContext();
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  auto i32Ty = builder.getI32Type();

  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  auto ivType = lb.getType();

  // Step 1: Get CTA rank within the cluster.
  Value ctaRankI32 = builder.create<triton::nvgpu::ClusterCTAIdOp>(loc, i32Ty);
  Value numCTAsI32 = builder.create<arith::ConstantIntOp>(
      loc, static_cast<int64_t>(numClusterCTAs), /*width=*/32);

  // Cast to the loop IV type if needed.
  Value ctaRank = (ivType == i32Ty)
                      ? ctaRankI32
                      : static_cast<Value>(builder.create<arith::IndexCastOp>(
                            loc, ivType, ctaRankI32));
  Value numCTAs = (ivType == i32Ty)
                      ? numCTAsI32
                      : static_cast<Value>(builder.create<arith::IndexCastOp>(
                            loc, ivType, numCTAsI32));

  // Step 2: Partition loop range across CTAs.
  Value range = builder.create<arith::SubIOp>(loc, ub, lb);

  // Verify divisibility: floor division drops remainder iterations.
  APInt rangeConst;
  if (matchPattern(range, m_ConstantInt(&rangeConst)) &&
      rangeConst.getZExtValue() % numClusterCTAs != 0) {
    return forOp.emitError("multi-CTA loop range (")
           << rangeConst.getZExtValue()
           << ") is not evenly divisible by numClusterCTAs (" << numClusterCTAs
           << "); remainder iterations would be silently dropped";
  }

  Value chunkSize = builder.create<arith::DivUIOp>(loc, range, numCTAs);
  Value offset = builder.create<arith::MulIOp>(loc, ctaRank, chunkSize);
  Value myLB = builder.create<arith::AddIOp>(loc, lb, offset);
  Value myUB = builder.create<arith::AddIOp>(loc, myLB, chunkSize);

  forOp.setLowerBound(myLB);
  forOp.setUpperBound(myUB);
  forOp->removeAttr("tt.multi_cta");

  // Step 3: For each tt.reduce consumer, generate cross-CTA DSM exchange
  //         of the SCALAR result (4 bytes, like TLX) instead of the full
  //         accumulator tensor (4096 bytes which exceeds DSM limits).
  namespace ttg = triton::gpu;
  namespace ttng = triton::nvidia_gpu;

  auto smemSpace = ttg::SharedMemorySpaceAttr::get(context);

  auto reduces = findReduceConsumers(forOp);
  for (auto reduceOp : reduces) {
    // The reduce produces scalar f32. We exchange it as a small tensor.
    builder.setInsertionPointAfter(reduceOp);
    Value partialResult = reduceOp->getResult(0);
    Type scalarType = partialResult.getType();

    unsigned elemBytes = scalarType.getIntOrFloatBitWidth() / 8;
    int expectedBytes = elemBytes * (numClusterCTAs - 1);

    // Get the reduce's input encoding to derive warp count.
    auto origSrcTy =
        cast<RankedTensorType>(reduceOp.getOperands()[0].getType());
    auto origEnc = origSrcTy.getEncoding();
    auto origBlockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(origEnc);
    if (!origBlockedEnc) {
      return reduceOp.emitError(
                 "multi-CTA reduction requires BlockedEncodingAttr on reduce "
                 "input, got ")
             << origEnc;
    }

    // Create a 1D CTA layout with no cluster splitting.
    auto ctaLayout1d = ttg::CTALayoutAttr::get(
        context, /*CTAsPerCGA=*/{1}, /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto smemEncoding1d = ttg::SwizzledSharedEncodingAttr::get(
        context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
        /*order=*/{0}, ctaLayout1d);

    // Create a 1-element blocked encoding with sizePerThread=[1].
    // CRITICAL: Using the original encoding's sizePerThread (e.g., [4]) would
    // cause getTotalElemsPerThread to return 4, making reduceWithinThreads
    // accumulate 4 copies of the scalar instead of 1.
    unsigned numWarps = 1;
    for (auto w : origBlockedEnc.getWarpsPerCTA())
      numWarps *= w;
    auto exchange1dEnc = ttg::BlockedEncodingAttr::get(
        context, /*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
        /*warpsPerCTA=*/{numWarps}, /*order=*/{0}, ctaLayout1d);
    auto tensor1dTy = RankedTensorType::get({1}, scalarType, exchange1dEnc);

    // a) Allocate DSM buffer: [numCTAs x 1] rank-2 in shared memory.
    auto ctaLayout2d = ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1, 1},
                                               /*CTASplitNum=*/{1, 1},
                                               /*CTAOrder=*/{1, 0});
    auto smemEncoding2d = ttg::SwizzledSharedEncodingAttr::get(
        context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
        /*order=*/{1, 0}, ctaLayout2d);
    auto bufType = ttg::MemDescType::get({numClusterCTAs, 1}, scalarType,
                                         smemEncoding2d, smemSpace, true);
    Value dsmBuf = builder.create<ttg::LocalAllocOp>(loc, bufType, Value());

    // b) Allocate barrier.
    auto barBufType = ttg::MemDescType::get({1}, builder.getI64Type(),
                                            smemEncoding1d, smemSpace, true);
    Value barrier = builder.create<ttg::LocalAllocOp>(loc, barBufType, Value());
    // init_barrier count = 1: only BarrierExpectOp counts as an arrival.
    // The st.async.mbarrier::complete_tx::bytes ops deliver bytes but do NOT
    // count as arrivals. Using numClusterCTAs-1 here causes deadlock for >2
    // CTAs.
    builder.create<ttng::InitBarrierOp>(loc, barrier, 1);

    // c) Wrap scalar in tensor<1xf32> using the exchange encoding.
    Value partialTensor =
        builder.create<triton::SplatOp>(loc, tensor1dTy, partialResult);

    // d) Get my slot in dsmBuf: memdesc<1xf32> (rank-1).
    auto dsmSlotType =
        ttg::MemDescType::get({1}, scalarType, smemEncoding1d, smemSpace, true);
    Value mySlot = builder.create<ttg::MemDescIndexOp>(loc, dsmSlotType, dsmBuf,
                                                       ctaRankI32);

    // Match TLX ordering exactly:
    //   barrier_expect -> cluster_arrive/wait -> local_store -> async_remote ->
    //   wait_barrier
    Value predTrue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    builder.create<ttng::BarrierExpectOp>(loc, barrier, expectedBytes,
                                          predTrue);
    builder.create<ttng::ClusterArriveOp>(loc, false);
    builder.create<ttng::ClusterWaitOp>(loc);

    // e) Store my partial to my slot AFTER cluster sync (matching TLX).
    builder.create<ttg::LocalStoreOp>(loc, partialTensor, mySlot);

    // f) Send partial to other CTAs (skip self).
    for (int i = 0; i < numClusterCTAs; ++i) {
      Value iVal = builder.create<arith::ConstantIntOp>(loc, i, 32);
      Value isNotMe = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, ctaRankI32, iVal);
      auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, isNotMe,
                                            /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      builder.create<ttg::AsyncRemoteShmemStoreOp>(loc, partialTensor, mySlot,
                                                   iVal, barrier);
      builder.setInsertionPointAfter(ifOp);
    }

    // g) Wait for all remote stores.
    Value phaseZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    builder.create<ttng::WaitBarrierOp>(loc, barrier, phaseZero, predTrue);

    // h) Accumulate: load each slot, add with arith.addf on tensor<1xf32>.
    if (!isa<FloatType>(scalarType)) {
      return reduceOp.emitError("multi-CTA cross-CTA accumulation only "
                                "supports floating-point "
                                "reductions, got ")
             << scalarType;
    }
    Value firstSlotIdx = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value firstSlot = builder.create<ttg::MemDescIndexOp>(loc, dsmSlotType,
                                                          dsmBuf, firstSlotIdx);
    Value combined =
        builder.create<ttg::LocalLoadOp>(loc, tensor1dTy, firstSlot);
    for (int i = 1; i < numClusterCTAs; ++i) {
      Value iVal = builder.create<arith::ConstantIntOp>(loc, i, 32);
      Value slot =
          builder.create<ttg::MemDescIndexOp>(loc, dsmSlotType, dsmBuf, iVal);
      Value loaded = builder.create<ttg::LocalLoadOp>(loc, tensor1dTy, slot);
      combined = builder.create<arith::AddFOp>(loc, combined, loaded);
    }

    // i) Extract scalar from tensor<1xf32> via tt.reduce(axis=0).
    auto finalReduce =
        builder.create<triton::ReduceOp>(loc, SmallVector<Value>{combined}, 0);
    IRMapping finalMapping;
    reduceOp.getCombineOp().cloneInto(&finalReduce.getCombineOp(),
                                      finalMapping);
    Value finalResult = finalReduce->getResult(0);

    // j) Replace uses of the original reduce result with the final result.
    //    Replace ALL uses EXCEPT: the reduceOp itself and ops in our DSM chain
    //    (which are between reduceOp and finalResult in the block).
    SmallVector<OpOperand *> usesToReplace;
    for (auto &use : partialResult.getUses()) {
      Operation *user = use.getOwner();
      if (user == reduceOp.getOperation())
        continue;
      // Skip users in different blocks (isBeforeInBlock requires same block).
      if (user->getBlock() != reduceOp->getBlock())
        continue;
      // Skip ops in our DSM chain: they are AFTER reduceOp but BEFORE or AT
      // finalReduce. Everything AFTER finalReduce should be replaced.
      if (reduceOp->isBeforeInBlock(user) &&
          !finalReduce->isBeforeInBlock(user))
        continue;
      usesToReplace.push_back(&use);
    }
    for (auto *use : usesToReplace) {
      use->set(finalResult);
    }
  }

  LDBG("Transformed multi-CTA loop at " << loc << " with " << numClusterCTAs
                                        << " CTAs, " << reduces.size()
                                        << " reduces");
  return success();
}

} // namespace

class NVGPUMultiCTAReductionPass
    : public impl::NVGPUMultiCTAReductionBase<NVGPUMultiCTAReductionPass> {
public:
  using impl::NVGPUMultiCTAReductionBase<
      NVGPUMultiCTAReductionPass>::NVGPUMultiCTAReductionBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    int numCTAs = getNumClusterCTAs(moduleOp);

    moduleOp->walk([&](triton::FuncOp funcOp) {
      SmallVector<scf::ForOp> multiCTALoops;
      funcOp->walk([&](scf::ForOp forOp) {
        if (forOp->hasAttr("tt.multi_cta"))
          multiCTALoops.push_back(forOp);
      });

      for (auto forOp : multiCTALoops) {
        if (failed(transformMultiCTALoop(forOp, numCTAs))) {
          forOp.emitError("failed to transform multi-CTA loop");
          return signalPassFailure();
        }
      }
    });
  }
};

} // namespace mlir
