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

/// Check that the loop body only accumulates via addition.
/// For each iter_arg, the corresponding yield operand must be defined by
/// arith::AddFOp or arith::AddIOp with one operand being the iter_arg itself.
/// This ensures the loop is a pure additive accumulation that can be safely
/// partitioned across CTAs (each CTA computes a partial sum).
static LogicalResult verifyAdditiveAccumulation(scf::ForOp forOp) {
  auto yieldedValues = forOp.getYieldedValues();
  auto iterArgs = forOp.getRegionIterArgs();

  for (unsigned i = 0; i < yieldedValues.size(); ++i) {
    Value yielded = yieldedValues[i];
    Value iterArg = iterArgs[i];

    Operation *defOp = yielded.getDefiningOp();
    if (!defOp)
      return forOp.emitError(
          "multi-CTA loop: yield operand is not defined by an operation; "
          "only additive accumulation (acc += ...) is supported");

    if (!isa<arith::AddFOp, arith::AddIOp>(defOp))
      return forOp.emitError("multi-CTA loop: yield operand is defined by ")
             << defOp->getName()
             << ", but only arith.addf/arith.addi accumulation is supported; "
                "the loop body must be a pure additive accumulation (acc += "
                "...)";

    bool usesIterArg = llvm::any_of(defOp->getOperands(),
                                    [&](Value v) { return v == iterArg; });
    if (!usesIterArg)
      return forOp.emitError(
          "multi-CTA loop: the add operation in the loop body does not use "
          "the loop's iter_arg as an operand; only patterns like "
          "acc = acc + x are supported");
  }
  return success();
}

/// Check that a triton::ReduceOp's combine region is a pure addition.
/// The combine region must contain exactly one arith.addf or arith.addi
/// (plus block args and yield), and no other arithmetic operations.
static LogicalResult verifyReduceCombinerIsAdd(triton::ReduceOp reduceOp) {
  Region &combineRegion = reduceOp.getCombineOp();
  bool hasAdd = false;
  bool hasNonAddArith = false;

  combineRegion.walk([&](Operation *op) {
    if (isa<arith::AddFOp, arith::AddIOp>(op)) {
      hasAdd = true;
    } else if (isa<arith::MaxNumFOp, arith::MaxSIOp, arith::MaxUIOp,
                   arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp,
                   arith::MaximumFOp, arith::MinimumFOp, arith::MulFOp,
                   arith::MulIOp, arith::XOrIOp, arith::AndIOp, arith::OrIOp>(
                   op)) {
      hasNonAddArith = true;
    }
  });

  if (!hasAdd)
    return reduceOp.emitError(
        "multi-CTA reduction requires an additive reduce combiner "
        "(arith.addf or arith.addi), but none was found in the combine "
        "region; only sum reductions can be distributed across CTAs");
  if (hasNonAddArith)
    return reduceOp.emitError(
        "multi-CTA reduction requires a pure additive reduce combiner, "
        "but found non-additive arithmetic in the combine region; "
        "only sum reductions (not max, min, mul, etc.) are supported");

  return success();
}

/// Transform a multi-CTA annotated loop: partition iterations across CTAs and
/// generate cross-CTA DSM exchange for any downstream tt.reduce consumers.
static LogicalResult transformMultiCTALoop(scf::ForOp forOp,
                                           int numClusterCTAs) {
  if (numClusterCTAs <= 1) {
    forOp->removeAttr("tt.multi_cta");
    return success();
  }

  // Validate that this loop is a pure additive accumulation and that
  // downstream reduces use an add combiner. This ensures correctness:
  // partitioning a non-additive loop (e.g., max, mul) across CTAs and
  // combining partial results with addition would produce wrong results.
  if (failed(verifyAdditiveAccumulation(forOp)))
    return failure();

  auto reduces = findReduceConsumers(forOp);
  for (auto reduceOp : reduces) {
    if (failed(verifyReduceCombinerIsAdd(reduceOp)))
      return failure();
  }

  auto *context = forOp->getContext();
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  auto i32Ty = builder.getI32Type();

  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  auto ivType = lb.getType();

  // Step 1: Get CTA rank within the cluster.
  Value ctaRankI32 = triton::nvgpu::ClusterCTAIdOp::create(builder, loc, i32Ty);
  Value numCTAsI32 = arith::ConstantIntOp::create(
      builder, loc, static_cast<int64_t>(numClusterCTAs), /*width=*/32);

  // Cast to the loop IV type if needed.
  Value ctaRank = (ivType == i32Ty)
                      ? ctaRankI32
                      : static_cast<Value>(arith::IndexCastOp::create(
                            builder, loc, ivType, ctaRankI32));
  Value numCTAs = (ivType == i32Ty)
                      ? numCTAsI32
                      : static_cast<Value>(arith::IndexCastOp::create(
                            builder, loc, ivType, numCTAsI32));

  // Step 2: Partition loop range across CTAs.
  Value range = arith::SubIOp::create(builder, loc, ub, lb);

  // Verify divisibility: floor division drops remainder iterations.
  APInt rangeConst;
  if (matchPattern(range, m_ConstantInt(&rangeConst)) &&
      rangeConst.getZExtValue() % numClusterCTAs != 0) {
    return forOp.emitError("multi-CTA loop range (")
           << rangeConst.getZExtValue()
           << ") is not evenly divisible by numClusterCTAs (" << numClusterCTAs
           << "); remainder iterations would be silently dropped";
  }

  Value chunkSize = arith::DivUIOp::create(builder, loc, range, numCTAs);
  Value offset = arith::MulIOp::create(builder, loc, ctaRank, chunkSize);
  Value myLB = arith::AddIOp::create(builder, loc, lb, offset);
  Value myUB = arith::AddIOp::create(builder, loc, myLB, chunkSize);

  forOp.setLowerBound(myLB);
  forOp.setUpperBound(myUB);
  forOp->removeAttr("tt.multi_cta");

  // Step 3: For each tt.reduce consumer, generate cross-CTA DSM exchange.
  //         The reduce may produce either a scalar (1D accumulator reduced to
  //         axis=0) or a tensor (2D accumulator reduced along one axis, e.g.,
  //         tensor<BLOCK_SIZE_M x f32>). We exchange resultSize * elemBytes
  //         per CTA via DSM, matching the TLX pattern for multi-row blocks.
  namespace ttg = triton::gpu;
  namespace ttng = triton::nvidia_gpu;

  auto smemSpace = ttg::SharedMemorySpaceAttr::get(context);

  for (auto reduceOp : reduces) {
    builder.setInsertionPointAfter(reduceOp);
    Value partialResult = reduceOp->getResult(0);
    Type resultType = partialResult.getType();

    // Detect scalar vs tensor result.
    Type elemType;
    int64_t resultSize;
    bool isScalar = !isa<RankedTensorType>(resultType);
    if (isScalar) {
      elemType = resultType;
      resultSize = 1;
    } else {
      auto tensorTy = cast<RankedTensorType>(resultType);
      elemType = tensorTy.getElementType();
      resultSize = tensorTy.getNumElements();
    }

    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    int expectedBytes = elemBytes * resultSize * (numClusterCTAs - 1);

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
    auto ctaLayout1d = ttg::CTAEncodingAttr::fromSplitParams(
        context, /*CTAsPerCGA=*/{1}, /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto smemEncoding1d = ttg::SwizzledSharedEncodingAttr::get(
        context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
        /*order=*/{0}, ctaLayout1d);

    // Create exchange encoding with sizePerThread=[1].
    // CRITICAL: Using the original encoding's sizePerThread (e.g., [4]) would
    // cause getTotalElemsPerThread to return 4, making reduceWithinThreads
    // accumulate 4 copies of the scalar instead of 1.
    unsigned numWarps = 1;
    for (auto w : origBlockedEnc.getWarpsPerCTA())
      numWarps *= w;
    auto exchange1dEnc = ttg::BlockedEncodingAttr::get(
        context, /*sizePerThread=*/{1}, /*threadsPerWarp=*/{32},
        /*warpsPerCTA=*/{numWarps}, /*order=*/{0}, ctaLayout1d);
    auto exchangeTy =
        RankedTensorType::get({resultSize}, elemType, exchange1dEnc);

    // a) Allocate DSM buffer: [numCTAs x resultSize] rank-2 in shared memory.
    auto ctaLayout2d =
        ttg::CTAEncodingAttr::fromSplitParams(context, /*CTAsPerCGA=*/{1, 1},
                                              /*CTASplitNum=*/{1, 1},
                                              /*CTAOrder=*/{1, 0});
    auto smemEncoding2d = ttg::SwizzledSharedEncodingAttr::get(
        context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
        /*order=*/{1, 0}, ctaLayout2d);
    auto bufType = ttg::MemDescType::get({numClusterCTAs, resultSize}, elemType,
                                         smemEncoding2d, smemSpace, true);
    Value dsmBuf = ttg::LocalAllocOp::create(builder, loc, bufType, Value());

    // b) Allocate barrier.
    auto barBufType = ttg::MemDescType::get({1}, builder.getI64Type(),
                                            smemEncoding1d, smemSpace, true);
    Value barrier =
        ttg::LocalAllocOp::create(builder, loc, barBufType, Value());
    // init_barrier count = 1: only BarrierExpectOp counts as an arrival.
    // The st.async.mbarrier::complete_tx::bytes ops deliver bytes but do NOT
    // count as arrivals. Using numClusterCTAs-1 here causes deadlock for >2
    // CTAs.
    ttng::InitBarrierOp::create(builder, loc, barrier, 1);

    // c) Wrap/convert the partial result into the exchange tensor type.
    Value partialTensor;
    if (isScalar) {
      partialTensor =
          triton::SplatOp::create(builder, loc, exchangeTy, partialResult);
    } else {
      partialTensor =
          ttg::ConvertLayoutOp::create(builder, loc, exchangeTy, partialResult);
    }

    // d) Get my slot in dsmBuf: memdesc<resultSize x elemType> (rank-1).
    auto dsmSlotType = ttg::MemDescType::get({resultSize}, elemType,
                                             smemEncoding1d, smemSpace, true);
    Value mySlot = ttg::MemDescIndexOp::create(builder, loc, dsmSlotType,
                                               dsmBuf, ctaRankI32);

    // Match TLX ordering exactly:
    //   barrier_expect -> cluster_arrive/wait -> local_store -> async_remote ->
    //   wait_barrier
    Value predTrue = arith::ConstantIntOp::create(builder, loc, 1, 1);
    ttng::BarrierExpectOp::create(builder, loc, barrier, expectedBytes,
                                  predTrue);
    ttng::ClusterArriveOp::create(builder, loc, false);
    ttng::ClusterWaitOp::create(builder, loc);

    // e) Store my partial to my slot AFTER cluster sync (matching TLX).
    ttg::LocalStoreOp::create(builder, loc, partialTensor, mySlot);

    // f) Send partial to other CTAs (skip self).
    for (int i = 0; i < numClusterCTAs; ++i) {
      Value iVal = arith::ConstantIntOp::create(builder, loc, i, 32);
      Value isNotMe = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne, ctaRankI32, iVal);
      auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isNotMe,
                                    /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      ttg::AsyncRemoteShmemStoreOp::create(builder, loc, partialTensor, mySlot,
                                           iVal, barrier);
      builder.setInsertionPointAfter(ifOp);
    }

    // g) Wait for all remote stores.
    Value phaseZero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    ttng::WaitBarrierOp::create(builder, loc, barrier, phaseZero, predTrue);

    // h) Accumulate: load each slot, add with arith.addf.
    if (!isa<FloatType>(elemType)) {
      return reduceOp.emitError("multi-CTA cross-CTA accumulation only "
                                "supports floating-point "
                                "reductions, got ")
             << elemType;
    }
    Value firstSlotIdx = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value firstSlot = ttg::MemDescIndexOp::create(builder, loc, dsmSlotType,
                                                  dsmBuf, firstSlotIdx);
    Value combined =
        ttg::LocalLoadOp::create(builder, loc, exchangeTy, firstSlot);
    for (int i = 1; i < numClusterCTAs; ++i) {
      Value iVal = arith::ConstantIntOp::create(builder, loc, i, 32);
      Value slot =
          ttg::MemDescIndexOp::create(builder, loc, dsmSlotType, dsmBuf, iVal);
      Value loaded = ttg::LocalLoadOp::create(builder, loc, exchangeTy, slot);
      combined = arith::AddFOp::create(builder, loc, combined, loaded);
    }

    // i) Extract the final result from the accumulated exchange tensor.
    Value finalResult;
    if (isScalar) {
      // Scalar case: extract from tensor<1xelemType> via tt.reduce(axis=0).
      auto finalReduce = triton::ReduceOp::create(
          builder, loc, SmallVector<Value>{combined}, 0);
      IRMapping finalMapping;
      reduceOp.getCombineOp().cloneInto(&finalReduce.getCombineOp(),
                                        finalMapping);
      finalResult = finalReduce->getResult(0);
    } else {
      // Tensor case: convert back from exchange encoding to original encoding.
      finalResult =
          ttg::ConvertLayoutOp::create(builder, loc, resultType, combined);
    }

    // j) Replace uses of the original reduce result with the final result.
    //    Replace ALL uses EXCEPT: the reduceOp itself and ops in our DSM chain
    //    (which are between reduceOp and finalResult in the block).
    Operation *finalOp = finalResult.getDefiningOp();
    SmallVector<OpOperand *> usesToReplace;
    for (auto &use : partialResult.getUses()) {
      Operation *user = use.getOwner();
      if (user == reduceOp.getOperation())
        continue;
      // Skip users in different blocks (isBeforeInBlock requires same block).
      if (user->getBlock() != reduceOp->getBlock())
        continue;
      // Skip ops in our DSM chain: they are AFTER reduceOp but BEFORE or AT
      // finalOp. Everything AFTER finalOp should be replaced.
      if (reduceOp->isBeforeInBlock(user) && !finalOp->isBeforeInBlock(user))
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
