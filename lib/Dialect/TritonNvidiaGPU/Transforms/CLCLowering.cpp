// CLC (Cluster Launch Control) tile-scheduler lowering.
//
// Implements Stages 1, 2 and 4 of the design in
// docs/design/triton-clc-tile-scheduler.md:
//
//   Stage 1 (clc-split):        ttng.clc_advance -> ttng.clc_try_cancel_async
//                               (-> !ttg.async.token) + ttng.clc_read
//   Stage 2 (clc-hoist):        promote ttng.clc_try_cancel_async to the top of
//                               its block so the CLC latency overlaps compute
//   Stage 4 (clc-materialize):  token form -> response buffer + completion
//                               mbarrier + low-level try_cancel/expect (issue)
//                               and wait/load/decode (read), with a
//                               loop-carried phase. Explicit clusters also get
//                               an empty barrier that guards response reuse.
//
// Stage 3 (AutoWS handling) is intentionally NOT here; it lands in a follow-up
// branch and slots between Stage 2 and Stage 4. The CUDA-style ctas_per_cga
// path is supported because it keeps num-ctas == 1 and assigns a distinct
// program id to each CTA. Triton's num-ctas > 1 path remains unsupported.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCLCSPLITPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUCLCHOISTPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUCLCMATERIALIZEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace ttg = triton::gpu;

namespace {

// A CLC response is 16 bytes; the low-level ops require a rank-1 memdesc of
// exactly 2 x i64 (see verifyCLCResultMemdesc).
constexpr int kClcResponseBytes = 16;

int getExplicitClusterSize(ModuleOp mod) {
  if (ttg::TritonGPUDialect::getNumCTAs(mod) != 1)
    return 1;
  auto dims = ttg::TritonGPUDialect::getClusterDims(mod);
  return dims[0] * dims[1] * dims[2];
}

Value createClcResponseAlloc(OpBuilder &b, Location loc) {
  MLIRContext *ctx = b.getContext();
  auto cga = ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/1);
  SmallVector<unsigned> order{0};
  Attribute enc = ttg::SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cga);
  auto memSpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto ty = ttg::MemDescType::get({2}, b.getI64Type(), enc, memSpace,
                                  /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(b, loc, ty);
}

//===--------------------------------------------------------------------===//
// Stage 1 - split clc_advance into the token form.
//===--------------------------------------------------------------------===//
struct CLCSplitPass
    : public impl::TritonNvidiaGPUCLCSplitPassBase<CLCSplitPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto tokTy = ttg::AsyncTokenType::get(ctx);
    SmallVector<Type> readTys{
        IntegerType::get(ctx, 1), IntegerType::get(ctx, 32),
        IntegerType::get(ctx, 32), IntegerType::get(ctx, 32)};
    getOperation().walk([&](CLCAdvanceOp adv) {
      OpBuilder b(adv);
      auto issue = CLCTryCancelAsyncOp::create(b, adv.getLoc(), tokTy);
      auto read = CLCReadOp::create(b, adv.getLoc(), readTys,
                                    ValueRange{issue.getToken()});
      adv->replaceAllUsesWith(read->getResults());
      adv->erase();
    });
  }
};

//===--------------------------------------------------------------------===//
// Stage 2 - hoist clc_try_cancel_async to overlap the CLC latency.
//
// clc_try_cancel_async has no operands, so within its block it can move to the
// front freely; that places the issue above the tile's compute (loop-body top)
// so the request resolves while the tile is computed. The move is restricted to
// the persistent-loop body (the scf.while after-region), the only block whose
// front is a safe issue point -- an arbitrary block's front may hold required
// setup. TODO: cross-basic-block hoisting / unification (e.g. clc_advance in
// both arms of an if/else) is a follow-up; see the design doc.
//===--------------------------------------------------------------------===//
struct CLCHoistPass
    : public impl::TritonNvidiaGPUCLCHoistPassBase<CLCHoistPass> {
  void runOnOperation() override {
    getOperation().walk([&](CLCTryCancelAsyncOp issue) {
      Block *blk = issue->getBlock();
      auto whileOp = dyn_cast<scf::WhileOp>(blk->getParentOp());
      if (!whileOp || blk != &whileOp.getAfter().front())
        return;
      if (&blk->front() != issue.getOperation())
        issue->moveBefore(&blk->front());
    });
  }
};

//===--------------------------------------------------------------------===//
// Stage 4 - materialize the token form into mbarrier ops.
//===--------------------------------------------------------------------===//
struct CLCMaterializePass
    : public impl::TritonNvidiaGPUCLCMaterializePassBase<CLCMaterializePass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Triton's num-ctas path gives one program id to the whole cluster. CLC
    // currently supports either a single CTA or the CUDA-style ctas_per_cga
    // path, where num-ctas stays 1 and ttg.cluster-dim-* describes the physical
    // cluster whose CTAs each have their own program id.
    if (ttg::TritonGPUDialect::getNumCTAs(mod) != 1) {
      bool hasCLC = false;
      mod.walk([&](Operation *op) {
        if (isa<CLCTryCancelAsyncOp, CLCReadOp, CLCAdvanceOp>(op))
          hasCLC = true;
      });
      if (hasCLC) {
        mod.emitError("CLC tile scheduler does not support Triton's num-ctas "
                      "cluster semantics; use num-ctas == 1, optionally with "
                      "explicit ctas_per_cga cluster dimensions");
        return signalPassFailure();
      }
      return;
    }

    SmallVector<CLCReadOp> reads;
    mod.walk([&](CLCReadOp read) { reads.push_back(read); });
    for (CLCReadOp read : reads)
      if (failed(materialize(read)))
        return signalPassFailure();
  }

  LogicalResult materialize(CLCReadOp read) {
    auto issue = read.getToken().getDefiningOp<CLCTryCancelAsyncOp>();
    if (!issue)
      return read.emitError("clc_read token is not from clc_try_cancel_async");

    auto whileOp = read->getParentOfType<scf::WhileOp>();
    if (!whileOp || issue->getParentOfType<scf::WhileOp>() != whileOp)
      return read.emitError("CLC fetch must live in a single persistent "
                            "scf.while (Stage 3 / cross-region CLC not yet "
                            "supported in this branch)");

    Location loc = read.getLoc();
    OpBuilder b(whileOp);
    Type i32 = b.getI32Type();
    int clusterSize = getExplicitClusterSize(read->getParentOfType<ModuleOp>());

    // Loop-carried phase, initialized to 0.
    Value zero =
        arith::ConstantIntOp::create(b, loc, /*value=*/0, /*width=*/32);
    scf::WhileOp newLoop =
        replaceWhileOpWithNewSignature(b, whileOp, {zero}, {i32});
    whileOp->erase();

    // Response buffer + completion mbarrier, allocated before the loop. The
    // barrier ops take a single-buffer view of the (1-deep) barrier allocation.
    // Explicit clusters also use a second barrier to rendezvous before reuse.
    b.setInsertionPoint(newLoop);
    Value resp = createClcResponseAlloc(b, loc);
    Value barAlloc = createBarrierAlloc(newLoop, /*numBarriers=*/1);
    Value bar = createSingleBufferView(b, barAlloc, 0);
    Value emptyBar;
    if (clusterSize > 1) {
      Value emptyBarAlloc =
          createBarrierAlloc(newLoop, /*numBarriers=*/1, clusterSize);
      emptyBar = createSingleBufferView(b, emptyBarAlloc, 0);
    }

    // Forward the phase from the before-region into the after-region and read
    // it back as the wait phase.
    Value phaseBefore = newLoop.getBeforeArguments().back();
    Value phaseAfter = newLoop.getAfterArguments().back();
    auto condOp =
        cast<scf::ConditionOp>(newLoop.getBefore().front().getTerminator());
    condOp.getArgsMutable().append(phaseBefore);

    // Materialize the issue: barrier_expect + clc_try_cancel.
    OpBuilder ib(issue);
    if (emptyBar) {
      // The first wait passes immediately because an initialized mbarrier has
      // phase 0. Later waits block until every CTA has consumed the previous
      // multicast response and arrived on the lead CTA's empty barrier.
      Value rank =
          mlir::triton::nvgpu::ClusterCTAIdOp::create(ib, issue.getLoc(), i32);
      Value zero = arith::ConstantIntOp::create(ib, issue.getLoc(), 0, 32);
      Value isLeader = arith::CmpIOp::create(
          ib, issue.getLoc(), arith::CmpIPredicate::eq, rank, zero);
      Value one = arith::ConstantIntOp::create(ib, issue.getLoc(), 1, 32);
      Value emptyPhase =
          arith::XOrIOp::create(ib, issue.getLoc(), phaseAfter, one);
      WaitBarrierOp::create(ib, issue.getLoc(), emptyBar, emptyPhase, isLeader);
    }
    Value pred = arith::ConstantIntOp::create(ib, issue.getLoc(), /*value=*/1,
                                              /*width=*/1);
    BarrierExpectOp::create(ib, issue.getLoc(), bar, kClcResponseBytes, pred);
    CLCTryCancelOp::create(ib, issue.getLoc(), resp, bar);

    // Materialize the read: wait_barrier + clc_load_result + decode.
    OpBuilder rb(read);
    WaitBarrierOp::create(rb, loc, bar, phaseAfter);
    Value clcResult = CLCLoadResultOp::create(rb, loc, resp);
    Value isValid = CLCIsCanceledOp::create(rb, loc, clcResult);
    Value x = CLCGetProgramIdOp::create(rb, loc, clcResult, 0);
    Value y = CLCGetProgramIdOp::create(rb, loc, clcResult, 1);
    Value z = CLCGetProgramIdOp::create(rb, loc, clcResult, 2);
    if (emptyBar) {
      // Remote mbarrier waits are illegal, so all CTAs arrive on rank zero's
      // empty barrier and only rank zero waits on its local view above. Skip
      // the final arrival when cancellation failed because the loop exits and
      // the response storage will not be reused.
      // TODO: Represent this response-buffer release with an explicit op that
      // links resp and emptyBar, then let ProxyFenceInsertion place the fence.
      FenceAsyncSharedOp::create(rb, loc, /*bCluster=*/false);
      Value zero = arith::ConstantIntOp::create(rb, loc, 0, 32);
      auto emptyBarTy = cast<ttg::MemDescType>(emptyBar.getType());
      auto remoteBarTy = ttg::MemDescType::get(
          emptyBarTy.getShape(), emptyBarTy.getElementType(),
          emptyBarTy.getEncoding(),
          SharedClusterMemorySpaceAttr::get(rb.getContext()),
          emptyBarTy.getMutableMemory(), emptyBarTy.getAllocShape());
      Value remoteEmptyBar =
          MapToRemoteBufferOp::create(rb, loc, remoteBarTy, emptyBar, zero);
      ArriveBarrierOp::create(rb, loc, remoteEmptyBar, /*count=*/1, isValid);
    }
    read->replaceAllUsesWith(ValueRange{isValid, x, y, z});
    read->erase();
    issue->erase();

    // Toggle the phase once per iteration in the loop yield.
    auto yieldOp =
        cast<scf::YieldOp>(newLoop.getAfter().front().getTerminator());
    OpBuilder yb(yieldOp);
    Value one = arith::ConstantIntOp::create(yb, yieldOp.getLoc(), /*value=*/1,
                                             /*width=*/32);
    Value toggled =
        arith::XOrIOp::create(yb, yieldOp.getLoc(), phaseAfter, one);
    yieldOp.getResultsMutable().append(toggled);

    return success();
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
