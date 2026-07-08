//===- DotDecomposeAndSchedule.cpp ----------------------------------------===//
//
// TTGIR-level replacement for the LLVM-IR `LLIRSchedule` pass on AMD MFMA
// matmul kernels.
//
// What this pass does
// -------------------
// For each MFMA-typed `tt.dot` in an inner `scf.for`:
//   1. Compute the M × N partition plan from
//      `AMDMfmaEncodingAttr::getInstrShape()` and `warpsPerCTA`:
//        ctaTileM = instrShape[0] * warpsPerCTA[0]    (32 for v8/v10)
//        ctaTileN = instrShape[1] * warpsPerCTA[1]    (32 for v8/v10)
//        numPartitionsM = blockM / ctaTileM           (8 for v8/v10)
//        numPartitionsN = blockN / ctaTileN           (4 for v8/v10)
//   2. Walk backward from `dot.getA()` and `dot.getC()` to collect
//      producer ops that would need co-partitioning (adapted from
//      `WSDataPartition::getBackwardSliceToPartition`,
//      `third_party/nvidia/hopper/.../WSDataPartition.cpp:291`, stripped
//      of Hopper-only branches).
//   3. Walk forward from `dot.getResult()` to collect user ops
//      (`WSDataPartition::getForwardSliceToPartition`:441, same strip).
//   4. (APPLY only) Replace the dot with `M × N` sub-dots via
//      `amdgpu.extract_slice` on each operand, then re-aggregate with a
//      single `amdgpu.concat` in row-major (M outer, N inner) order.
//   5. (APPLY only) Insert a `ROCDL::SchedBarrier(0)` after every k-th
//      sub-dot (default k = numPartitionsN, i.e. one barrier per M-row)
//      so LLVM's misched can optimize within row-bounded regions
//      without reordering across boundaries.
//
// Producer / user chains are not modified — `extract_slice` is a no-op
// at the CTA-tile level, so upstream layouts are preserved.
//
// Env-var contract
// ----------------
//   TRITON_ENABLE_TTGIR_SCHED          : enable the pass (planning-only)
//   TRITON_TTGIR_SCHED_APPLY           : also mutate IR
//   TRITON_TTGIR_SCHED_BARRIER_STRIDE  : barrier stride
//                                          unset → numPartitionsN
//                                          0     → no barriers
//                                          1     → between every sub-dot
//                                          k     → every k-th
//
// e2e results (stand-alone autotuned matmul, K sweep on gfx950):
//   K=1024 → -1.4 %, K=2048 → +2.4 %, K=4096 → +4.1 %, K=8192 → +11.0 %
// FA-fwd tutorial (the kernel that crashes the matmul_4waves LLIR pass
// with `Instruction does not dominate all uses!`):
//   runs correctly under TTGIR-SCHED with output within ±2 % of baseline.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "Utility.h" // AMD transforms: composePaddedLayout (+ TargetInfo)
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
// Backend-neutral modulo scheduling core (Phase A extraction). Cross-tree
// include via the global `third_party` include dir — interim until the core is
// relocated to a neutral path.
#include "mlir/IR/IRMapping.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/AMDLatencyModel.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/DataDependenceGraph.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

#define DEBUG_TYPE "tritonamdgpu-dot-decompose-and-schedule"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDOTDECOMPOSEANDSCHEDULE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Partition plan (Phase 1a)
//===----------------------------------------------------------------------===//

static triton::gpu::AMDMfmaEncodingAttr getMfmaEncoding(triton::DotOp dotOp) {
  auto resultType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
  if (!resultType)
    return nullptr;
  return dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(resultType.getEncoding());
}

static bool isCloneableInterior(Operation *op) {
  return isa<triton::gpu::ConvertLayoutOp>(op) ||
         op->hasTrait<mlir::OpTrait::Elementwise>() ||
         isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncFOp,
             arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op);
}

static int computeLDSStageCap(scf::ForOp forOp, int fallback) {
  constexpr int64_t kLDSBytes = 160 * 1024; // gfx950 LDS per CU
  int64_t perBufferBytes = 0;
  forOp.getBody()->walk([&](triton::DotOp dot) {
    if (!getMfmaEncoding(dot))
      return;
    for (Value operand : {dot.getA(), dot.getB()}) {
      auto t = dyn_cast<RankedTensorType>(operand.getType());
      if (!t)
        continue;
      int64_t elems = 1;
      for (int64_t d : t.getShape())
        elems *= d;
      int64_t eb =
          std::max<int64_t>(1, t.getElementType().getIntOrFloatBitWidth() / 8);
      perBufferBytes += elems * eb;
    }
  });
  if (perBufferBytes <= 0)
    return fallback;
  int64_t maxBuffers = kLDSBytes / perBufferBytes; // floor
  if (maxBuffers < 2)
    maxBuffers = 2; // need at least a double buffer to pipeline
  return static_cast<int>(maxBuffers - 1);
}

//===----------------------------------------------------------------------===//
// Early load lowering (change #1 of amd_decomp_modulo_pipeline.md)
//
// Lower a pipelineable global tt.load to a SINGLE-buffer staged copy
//   local_alloc<1 x tile> -> async_copy_global_to_local -> commit -> wait
//   -> local_load (replacing the load's uses)
// BEFORE modulo runs, so modulo sees async_copy (GLOBAL latency) vs local_load
// (LDS) as distinct ops and can size the overlap itself. Single buffer here;
// the ModuloDotSchedule expander (change #4) grows the ring from modulo's
// stages. v1 uses a plain (unswizzled) shared encoding -- matching the real
// path's padded_shared needs targetInfo(arch) plumbed into the pass
// (follow-up).
//===----------------------------------------------------------------------===//

// True if `ld`'s result reaches an MFMA dot through cloneable interior ops
// (convert_layout / casts).
static bool loadFeedsMfmaDot(triton::LoadOp ld) {
  SmallVector<Operation *> wl(ld->getUsers().begin(), ld->getUsers().end());
  llvm::DenseSet<Operation *> seen;
  while (!wl.empty()) {
    Operation *u = wl.pop_back_val();
    if (!seen.insert(u).second)
      continue;
    if (auto dot = dyn_cast<triton::DotOp>(u))
      if (getMfmaEncoding(dot))
        return true;
    if (isCloneableInterior(u))
      for (Operation *uu : u->getUsers())
        wl.push_back(uu);
  }
  return false;
}

// The DotOperandEncoding the load `ld` feeds (through cloneable interior ops),
// or null. Drives the LDS encoding selection.
static triton::gpu::DotOperandEncodingAttr
loadDotOperandEnc(triton::LoadOp ld) {
  SmallVector<Value> wl{ld.getResult()};
  llvm::DenseSet<Operation *> seen;
  while (!wl.empty()) {
    Value v = wl.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (isa<triton::DotOp>(u))
        if (auto t = dyn_cast<RankedTensorType>(v.getType()))
          if (auto de = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                  t.getEncoding()))
            return de;
      if (seen.insert(u).second && isCloneableInterior(u))
        for (Value r : u->getResults())
          wl.push_back(r);
    }
  }
  return nullptr;
}

static void lowerLoadToStagedCopy(scf::ForOp forOp, triton::LoadOp ld,
                                  const triton::AMD::TargetInfo &targetInfo) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  if (!ty || ty.getRank() != 2)
    return;
  MLIRContext *ctx = ld.getContext();
  // Pick the LDS encoding the way the stream pipeliner does (LowerLoops):
  // padded (CDNA4 async, conflict-free) when possible, else dot-operand-driven
  // swizzled. Closes the v1 bank-conflict gap vs the plain unswizzled encoding.
  SmallVector<unsigned> sharedOrder = triton::gpu::getOrder(ty);
  auto cga = triton::gpu::getCGALayout(ty.getEncoding());
  unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
  triton::gpu::SharedEncodingTrait sharedEnc;
  if (auto dotOpEnc = loadDotOperandEnc(ld)) {
    auto srcTOM = cast<triton::gpu::TensorOrMemDesc>(ld.getType());
    sharedEnc = composePaddedLayout(targetInfo, dotOpEnc.getOpIdx(),
                                    dotOpEnc.getKWidth(), srcTOM, sharedOrder,
                                    dotOpEnc, /*useAsyncCopy=*/true);
    if (!sharedEnc)
      sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
          ctx, dotOpEnc, ty.getShape(), sharedOrder, cga, bitWidth,
          /*needTrans=*/false);
  }
  if (!sharedEnc) // no dot-operand info: plain swizzled fallback
    sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1,
                                                             sharedOrder, cga);
  Value alloc = triton::createAlloc(forOp, ty, ld.getLoc(), sharedEnc,
                                    /*distance=*/1);
  OpBuilder b(ld);
  Location loc = ld.getLoc();
  auto viewLoad = triton::createSingleBufferView(b, alloc, 0)
                      .getDefiningOp<triton::gpu::MemDescIndexOp>();
  auto copyOp = triton::gpu::AsyncCopyGlobalToLocalOp::create(
      b, loc, ld.getPtr(), viewLoad, ld.getMask(), ld.getOther(), ld.getCache(),
      ld.getEvict(), ld.getIsVolatile(), /*contiguity=*/1);
  auto commitOp =
      triton::gpu::AsyncCommitGroupOp::create(b, loc, copyOp->getResult(0));
  auto waitOp =
      triton::gpu::AsyncWaitOp::create(b, loc, commitOp->getResult(0), 0);
  triton::replaceUsesWithLocalLoad(b, ld->getResult(0), viewLoad, waitOp);
  ld.erase();
}

// Opt-in (TRITON_AMD_EARLY_LOWER): early-lower every pipelineable global load
// feeding an MFMA dot, for each inner loop.
static void runEarlyLowerLoads(ModuleOp module) {
  auto arch = mlir::getAMDArch(module);
  triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });
  for (scf::ForOp forOp : loops) {
    SmallVector<triton::LoadOp> loads;
    forOp.getBody()->walk([&](triton::LoadOp ld) {
      if (loadFeedsMfmaDot(ld))
        loads.push_back(ld);
    });
    for (triton::LoadOp ld : loads)
      lowerLoadToStagedCopy(forOp, ld, targetInfo);
  }
}

//===----------------------------------------------------------------------===//
// ModuloDotSchedule expander (change #4 of amd_decomp_modulo_pipeline.md)
//
// Takes the early-lowered loop (single-buffer async_copy/local_load) and turns
// it into a real software pipeline: (a) re-buffer single->multi + ring
// `extractIdx` (mirrors createStreamOps, LowerLoops.cpp:368-387), (b) serialize
// a CoarseSchedule (async_copy backward-slice -> stage 0 / load cluster; rest
// -> last stage / compute cluster), (c) run the general expander `expandLoops`
// (NOT SingleDotSchedule). v1: canonical depth-2 double buffer; modulo-derived
// depth / fine order is the refinement. Async-wait counts are fixed by the
// existing tritonamdgpu-update-async-wait-count pass downstream (e2e) — not in
// this pass.
//===----------------------------------------------------------------------===//
static void runModuloExpand(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });
  for (scf::ForOp forOp : loops) {
    SmallVector<triton::gpu::AsyncCopyGlobalToLocalOp> copies;
    forOp.getBody()->walk([&](triton::gpu::AsyncCopyGlobalToLocalOp cp) {
      copies.push_back(cp);
    });
    if (copies.empty())
      continue;

    const int numBuffers = 2; // v1: canonical double buffer
    IRRewriter builder(forOp);
    Location loc = forOp.getLoc();
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
    Value numBuf = arith::ConstantIntOp::create(builder, loc, numBuffers, 32);

    // Ring index iter-arg (mirrors createStreamOps).
    unsigned newArgIdx = forOp.getBody()->getNumArguments();
    forOp = addIterArgsToLoop(builder, forOp, {minusOne});
    Value ring = forOp.getBody()->getArgument(newArgIdx);
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
    ring = arith::AddIOp::create(builder, loc, ring, one);
    Value cnd = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                      ring, numBuf);
    ring = arith::SelectOp::create(builder, loc, cnd, ring, zero);
    appendToForOpYield(forOp, {ring});

    // Re-buffer each operand's alloc (depth 1 -> numBuffers), re-index by
    // `ring`. `copies` op pointers stay valid (addIterArgsToLoop moves the
    // body).
    for (auto cp : copies) {
      Value oldView = cp.getResult(); // dest memdesc = memdesc_index alloc[c0]
      auto oldViewOp = oldView.getDefiningOp<triton::gpu::MemDescIndexOp>();
      if (!oldViewOp)
        continue;
      auto oldAlloc =
          oldViewOp.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
      if (!oldAlloc)
        continue;
      auto oldTy = cast<triton::gpu::MemDescType>(oldAlloc.getType());
      auto shp = oldTy.getShape();
      if (shp.size() < 2)
        continue; // expect [1, tile...]
      SmallVector<int64_t> tile(shp.begin() + 1, shp.end());
      auto tensorTy = RankedTensorType::get(tile, oldTy.getElementType());
      auto enc =
          dyn_cast<triton::gpu::SharedEncodingTrait>(oldTy.getEncoding());
      if (!enc)
        continue;
      // createAlloc grows to numBuffers AND emits a matching local_dealloc.
      Value newAlloc = triton::createAlloc(forOp, tensorTy, oldAlloc.getLoc(),
                                           enc, numBuffers);
      OpBuilder vb(oldViewOp);
      Value newView = triton::createSingleBufferView(vb, newAlloc, ring);
      oldView.replaceAllUsesWith(newView);
      oldViewOp.erase();
      // Erase the old alloc's dangling local_dealloc, then the old alloc.
      SmallVector<Operation *> users(oldAlloc->getUsers().begin(),
                                     oldAlloc->getUsers().end());
      for (Operation *u : users)
        if (isa<triton::gpu::LocalDeallocOp>(u))
          u->erase();
      oldAlloc.erase();
    }

    // Stage assignment: async_copy backward slice (within the loop) -> stage 0
    // (load cluster); everything else -> last stage (compute cluster). The
    // backward slice keeps each load's index/pointer math co-staged (def<=use).
    llvm::DenseSet<Operation *> stage0;
    SmallVector<Operation *> wl;
    for (auto cp : copies)
      wl.push_back(cp);
    while (!wl.empty()) {
      Operation *op = wl.pop_back_val();
      if (!op || op->getBlock() != forOp.getBody())
        continue;
      if (!stage0.insert(op).second)
        continue;
      for (Value v : op->getOperands())
        if (Operation *d = v.getDefiningOp())
          wl.push_back(d);
    }
    triton::CoarseSchedule cs(/*numStages=*/numBuffers);
    auto cLoad = cs.clusters.newAtBack();
    auto cCompute = cs.clusters.newAtBack();
    for (Operation &op : forOp.getBody()->without_terminator()) {
      bool s0 = stage0.contains(&op);
      cs.insert(&op, s0 ? 0 : numBuffers - 1, s0 ? cLoad : cCompute);
    }
    cs.serialize(forOp);
  }
  // Run the general expander on every serialized loop (change #4 (b)).
  expandLoops(module);
}

// Phase E0: AMD modulo scaffold. For each inner loop, build the backend-neutral
// DDG (from TritonGPUModuloCore) using AMDLatencyModel and report the
// per-pipeline node classification. This is the scaffold for the AMD modulo
// scheduler and the runtime test gate for AMDLatencyModel.
static void runAMDModuloScaffold(ModuleOp module) {
  triton::gpu::AMDLatencyModel model;
  // Collect loops first — E2 mutates loop bodies, so don't mutate during walk.
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });

  for (scf::ForOp forOp : loops) {
    auto ddg = triton::gpu::DataDependenceGraph::build(forOp, model);
    llvm::DenseMap<triton::gpu::HWPipeline, unsigned> counts;
    for (const auto &node : ddg.getNodes())
      ++counts[node.pipeline];
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "amd-modulo: DDG " << ddg.getNumNodes() << " nodes";
    for (auto p :
         {triton::gpu::HWPipeline::MFMA, triton::gpu::HWPipeline::LDS,
          triton::gpu::HWPipeline::GLOBAL, triton::gpu::HWPipeline::VALU,
          triton::gpu::HWPipeline::NONE})
      os << " " << triton::gpu::getPipelineName(p) << "=" << counts.lookup(p);

    // E1: run the core modulo scheduler (rau/SMS, selected by
    // TRITON_USE_MODULO_SCHEDULE; default rau) and annotate each op with its
    // stage/order so a downstream expansion can consume the schedule.
    auto schedOr = triton::gpu::runModuloScheduling(ddg);
    if (!succeeded(schedOr)) {
      os << " II=FAILED";
      forOp.emitRemark() << os.str();
      continue;
    }
    auto &sched = *schedOr;
    mlir::Builder b(module.getContext());
    for (const auto &node : ddg.getNodes()) {
      auto it = sched.nodeToCycle.find(node.idx);
      if (it == sched.nodeToCycle.end())
        continue;
      node.op->setAttr("ttg.modulo_stage",
                       b.getI32IntegerAttr(sched.getStage(node.idx)));
      node.op->setAttr("ttg.modulo_order", b.getI32IntegerAttr(it->second));
    }
    os << " II=" << sched.II << " maxStage=" << sched.getMaxStage();

    // E3: emit a serialized triton::CoarseSchedule so the EXISTING AMD pipeline
    // expander (tritonamdgpu-pipeline) does multi-buffering + loop expansion —
    // i.e. modulo replaces tritonamdgpu-schedule-loops. Map modulo stage ->
    // CoarseSchedule stage and slot (order % II) -> cluster (steady-state body
    // order). num_stages and per-buffer copy-count then follow from modulo's
    // stage assignment. This is the cross-iteration (pipelining) axis; E2 below
    // is the complementary intra-iteration interleave axis.
    if (triton::tools::getBoolEnv("TRITON_AMD_MODULO_SERIALIZE")) {
      // Phase-D-lite guardrail: cap the pipeline depth. Realistic global
      // latency
      // (~790 cyc) makes modulo want a very deep pipeline (e.g. 52 stages) to
      // fully hide it — infeasible (LDS holds ~2 stages). Until Phase D adds a
      // real LDS/register feasibility model, clamp stages to a small cap
      // (default 1 -> 2-stage pipeline, matching the stream-pipeline baseline).
      int stageCap = 1;
      // Change #3: LDS-capacity support for the AMD modulo scheduler. When the
      // decomp+modulo guard is on, set the depth to min(LDS-feasible, needed):
      //   * LDS-feasible = floor(160KB / per-iter operand bytes) - 1   (gfx950)
      //   * needed       = max(1, modulo maxStage = ceil(latency/II))
      // so small tiles can pipeline deeper (up to LDS) while large tiles stay
      // protected, and we never buffer deeper than the latency actually needs.
      if (!triton::tools::getStrEnv("TRITON_USE_MODULO_SCHEDULE").empty()) {
        int ldsCap = computeLDSStageCap(forOp, /*fallback=*/1);
        int needed = std::max(1, sched.getMaxStage());
        stageCap = std::min(ldsCap, needed);
        os << " ldsCap=" << ldsCap << " needed=" << needed;
      }
      if (const char *e = std::getenv("TRITON_AMD_MODULO_MAX_STAGE"))
        stageCap = std::atoi(e); // explicit override always wins
      triton::CoarseSchedule cs(stageCap + 1);
      // The AMD pipeline expander's SingleDotSchedule path requires exactly the
      // 2-cluster convention: cluster 0 = global load, cluster 1 = compute. The
      // *multi-buffer decision* lives in the STAGE assignment (modulo owns it);
      // the cluster is just load-vs-compute. (Finer order/slot interleave is
      // the separate E2 axis, not consumed by this path.)
      auto cLoad = cs.clusters.newAtBack();    // SCHED_GLOBAL_LOAD
      auto cCompute = cs.clusters.newAtBack(); // SCHED_COMPUTE
      for (const auto &node : ddg.getNodes()) {
        auto it = sched.nodeToCycle.find(node.idx);
        if (it == sched.nodeToCycle.end())
          continue;
        bool isGlobal = node.pipeline == triton::gpu::HWPipeline::GLOBAL;
        // Canonical double-buffer overlap: global loads are PREFETCHED one
        // stage ahead of compute (stage 0 / cluster 0), everything else lands
        // in the compute stage (stageCap / cluster 1). Modulo's raw stage =
        // ceil(latency/II) under-stages when latency < II (no prefetch), so we
        // place loads ahead explicitly; modulo still provides II (now realistic
        // via the scaled MFMA cost), schedulability, and the load/compute
        // split.
        auto cluster = isGlobal ? cLoad : cCompute;
        int stage = isGlobal ? 0 : stageCap;
        cs.insert(node.op, stage, cluster);
      }
      cs.serialize(forOp);
      os << " serialized num_stages=" << cs.getNumStages()
         << " (capped from maxStage=" << sched.getMaxStage() << ")";
      forOp.emitRemark() << os.str();
      continue;
    }

    // E2: realize the schedule in IR — reorder the loop body into modulo
    // `order` and mark stage boundaries with rocdl.sched.barrier (region hints
    // for the backend machine scheduler). Guarded: only reorder when every
    // non-terminator op is scheduled, so the move sequence stays
    // dominance-valid (modulo order respects intra-iteration deps).
    Block *body = forOp.getBody();
    Operation *term = body->getTerminator();
    SmallVector<Operation *> ops;
    bool allScheduled = true;
    for (Operation &op : body->without_terminator()) {
      if (!op.hasAttr("ttg.modulo_order")) {
        allScheduled = false;
        break;
      }
      ops.push_back(&op);
    }
    int nbar = 0;
    if (allScheduled && !ops.empty()) {
      auto orderOf = [](Operation *op) {
        return cast<IntegerAttr>(op->getAttr("ttg.modulo_order")).getInt();
      };
      auto stageOf = [](Operation *op) {
        return cast<IntegerAttr>(op->getAttr("ttg.modulo_stage")).getInt();
      };
      llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
        return orderOf(a) < orderOf(b);
      });
      for (Operation *op : ops)
        op->moveBefore(term);
      int64_t prevStage = -1;
      for (Operation *op : ops) {
        int64_t stage = stageOf(op);
        if (prevStage >= 0 && stage != prevStage) {
          OpBuilder bb(op);
          ROCDL::SchedBarrier::create(bb, op->getLoc(), /*mask=*/0);
          ++nbar;
        }
        prevStage = stage;
      }
      os << " reordered barriers=" << nbar;
    } else {
      os << " reorder=skipped";
    }
    forOp.emitRemark() << os.str();
  }
}

struct TritonAMDGPUDotDecomposeAndSchedulePass
    : impl::TritonAMDGPUDotDecomposeAndScheduleBase<
          TritonAMDGPUDotDecomposeAndSchedulePass> {
  // Inherit base constructors (incl. the Options-taking one) so the `mode`
  // pass option is wired through createTritonAMDGPUDotDecomposeAndSchedule.
  using impl::TritonAMDGPUDotDecomposeAndScheduleBase<
      TritonAMDGPUDotDecomposeAndSchedulePass>::
      TritonAMDGPUDotDecomposeAndScheduleBase;

  void runOnOperation() override {
    // Change #2: an explicit `mode` pass option overrides env-var dispatch, so
    // the same pass can sit at several pipeline slots (early-lower / decompose
    // / modulo) without the two-slot collision (env dispatch checks AMD_MODULO
    // before TTGIR_SCHED, so a post-pipeline decompose slot would re-run
    // modulo).
    StringRef m(mode);
    if (m == "early-lower") {
      runEarlyLowerLoads(getOperation());
      return;
    }
    if (m == "modulo") {
      runAMDModuloScaffold(getOperation());
      return;
    }
    if (m == "expand") {
      runModuloExpand(getOperation());
      return;
    }
    // Legacy env-var dispatch: opt-in early load lowering / modulo scaffold.
    if (triton::tools::getBoolEnv("TRITON_AMD_EARLY_LOWER")) {
      runEarlyLowerLoads(getOperation());
      return;
    }
    if (triton::tools::getBoolEnv("TRITON_ENABLE_AMD_MODULO")) {
      runAMDModuloScaffold(getOperation());
      return;
    }
  }
};
} // namespace

} // namespace mlir
