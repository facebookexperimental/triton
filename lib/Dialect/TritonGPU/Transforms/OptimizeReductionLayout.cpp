//===- OptimizeReductionLayout.cpp ----------------------------------------===//
//
// M2 (bitequiv): pick a cheaper operand layout for `inner_tree` reductions.
//
// An `inner_tree` reduction is LAYOUT-INVARIANT by construction: the lowering
// (ReduceOpToLLVM) emits a balanced within-thread tree plus a count-up
// warp-shuffle xor sequence that fixes the FP association order independent of
// where the elements live. So for a reduce carrying `reduction_ordering =
// "inner_tree"` we may freely change the operand's data layout WITHOUT changing
// the result bits -- the whole valid-layout space is a free performance knob.
// (This is exactly the freedom `OptimizeThreadLocality` exploits for *unordered*
// reduces, which it must skip for ordered ones -- see its `hasDefinedOrdering`
// bail. This pass fills that gap for the ordered case.)
//
// STRATEGY. The cost of a reduction is dominated by the cross-warp shared-memory
// stage, which runs iff `warpsPerCTA[axis] != 1` (two barriers + smem round trip
// + a second butterfly). We build the "ideal" reduce layout: reduce the axis
// WITHIN a warp -- put lanes on the axis (cheap shuffles) and, crucially, no
// warps on the axis (warpsPerCTA[axis] = 1) -- and spread the warps over the
// kept dimensions. The remaining axis extent is carried in registers (a free
// within-thread fold). Then a cost model decides whether the win (deleting the
// cross-warp stage) beats the cost of the `convert_layout` we must insert on the
// operand; if not, we leave the reduce alone.
//
// The reduction op, axis, extent and ordering are NEVER changed, so the numeric
// result is bit-identical (verified by the bitequiv eval hook: correctness stage
// must report 0 bit-changed). Scope: within-CTA blocked-layout reductions. The
// pass runs LAST in make_ttgir so its layout is not reverted by a later pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEREDUCTIONLAYOUT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// floor(log2(v)) for v >= 1.
static unsigned ilog2(int64_t v) {
  unsigned b = 0;
  while ((int64_t(1) << (b + 1)) <= v)
    ++b;
  return b;
}

// Build the "ideal" reduce layout by distributing hardware bits: lanes onto the
// axis first (reduce within a warp), then remaining lanes to kept dims; warps
// onto kept dims (axis gets none if possible); the leftover axis extent carried
// in registers. Powers of two throughout, so the products stay exactly warpSize
// and numWarps. Returns {} if it cannot improve on `cur`.
static BlockedEncodingAttr makeIdeal(triton::ReduceOp red, BlockedEncodingAttr cur,
                                     int axis, ArrayRef<int64_t> shape,
                                     int numWarps, int warpSize) {
  unsigned rank = shape.size();
  SmallVector<unsigned> laneB(rank, 0), warpB(rank, 0), sizeB(rank, 0);
  SmallVector<unsigned> O(cur.getOrder());

  // 1. Lanes: axis first, then kept dims in memory order.
  int laneBudget = ilog2(warpSize);
  unsigned onAxis = std::min<unsigned>(laneBudget, ilog2(shape[axis]));
  laneB[axis] = onAxis;
  laneBudget -= onAxis;
  for (unsigned k : O) {
    if (laneBudget == 0)
      break;
    if (int(k) == axis)
      continue;
    unsigned take = std::min<unsigned>(laneBudget, ilog2(shape[k]));
    laneB[k] += take;
    laneBudget -= take;
  }
  if (laneBudget > 0) // shape too small to hold a full warp: pile leftover on axis
    laneB[axis] += laneBudget;

  // 2. Warps: kept dims only (keep the axis warp-synchronous), by remaining room.
  int warpBudget = ilog2(numWarps);
  for (unsigned k : O) {
    if (warpBudget == 0)
      break;
    if (int(k) == axis)
      continue;
    unsigned room = ilog2(shape[k]) > laneB[k] ? ilog2(shape[k]) - laneB[k] : 0;
    unsigned take = std::min<unsigned>(warpBudget, room);
    warpB[k] += take;
    warpBudget -= take;
  }
  if (warpBudget > 0) // kept dims cannot absorb all warps: fall back to axis (partial)
    warpB[axis] += warpBudget;

  // 3. Registers: carry the rest of the axis extent as a contiguous fold.
  unsigned axisCovered = laneB[axis] + warpB[axis];
  if (ilog2(shape[axis]) > axisCovered)
    sizeB[axis] = ilog2(shape[axis]) - axisCovered;

  SmallVector<unsigned> T(rank), W(rank), S(rank);
  for (unsigned d = 0; d < rank; ++d) {
    T[d] = 1u << laneB[d];
    W[d] = 1u << warpB[d];
    S[d] = 1u << sizeB[d];
  }
  auto cand = BlockedEncodingAttr::get(red.getContext(), S, T, W, O, cur.getCGALayout());
  if (cand == cur)
    return {};
  return cand;
}

// Rewrite `red` to consume its operands in layout `cand`. We cannot reuse
// convertDistributedOpEncoding: it applies the operand encoding to the RESULT
// types too, but a reduce result is a reduced-rank SliceEncoding. So: convert
// each operand blocked->cand, clone the reduce (preserving its combine region),
// retype the clone's results to slice-of-cand, then convert results back.
static void rewriteOperandLayout(triton::ReduceOp red, BlockedEncodingAttr cand,
                                 int axis) {
  OpBuilder b(red);
  Location loc = red.getLoc();
  Operation *redOp = red.getOperation(); // ReduceOp is variadic: index results via Operation*

  SmallVector<Value> converted;
  for (Value operand : red.getOperands()) {
    auto ty = cast<RankedTensorType>(operand.getType());
    auto newTy = RankedTensorType::get(ty.getShape(), ty.getElementType(), cand);
    converted.push_back(ConvertLayoutOp::create(b, loc, newTy, operand));
  }

  Operation *clone = b.clone(*redOp);
  for (unsigned i = 0; i < converted.size(); ++i)
    clone->setOperand(i, converted[i]);
  auto sliceEnc = SliceEncodingAttr::get(red.getContext(), axis, cand);
  for (unsigned i = 0; i < clone->getNumResults(); ++i) {
    auto rt = cast<RankedTensorType>(redOp->getResult(i).getType());
    clone->getResult(i).setType(
        RankedTensorType::get(rt.getShape(), rt.getElementType(), sliceEnc));
  }

  b.setInsertionPointAfter(clone);
  for (unsigned i = 0; i < redOp->getNumResults(); ++i) {
    Value back = ConvertLayoutOp::create(b, loc, redOp->getResult(i).getType(),
                                         clone->getResult(i));
    redOp->getResult(i).replaceAllUsesWith(back);
  }
  redOp->erase();
}

struct OptimizeReductionLayoutPass
    : public impl::TritonGPUOptimizeReductionLayoutBase<
          OptimizeReductionLayoutPass> {
  using Base = impl::TritonGPUOptimizeReductionLayoutBase<
      OptimizeReductionLayoutPass>;
  using Base::Base;
  void runOnOperation() override {
    // `strategy`, `minUnderparallel`, `maxElemsPerThread` are pass Options (see
    // Passes.td). The in-pipeline call sets them from `knobs.nvidia` (compiler.py);
    // LIT tests pass them as pass args. `strategy == "off"` -> no-op (bit-safe).
    if (strategy == "off")
      return;

    ModuleOp mod = getOperation();
    int warpSize = TritonGPUDialect::getThreadsPerWarp(mod);

    SmallVector<triton::ReduceOp> targets;
    mod.walk([&](triton::ReduceOp red) {
      auto ord = red.getReductionOrderingAttr();
      if (ord && ord.getValue() == "inner_tree")
        targets.push_back(red);
    });

    for (triton::ReduceOp red : targets) {
      auto srcTy = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!srcTy)
        continue;
      auto blocked = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
      if (!blocked)
        continue;
      // Bit-safety: a mul-fed reduce (dot product, operand = arith.mulf) can
      // FMA-contract the mul with the combine add under enable_fp_fusion. That
      // contraction is layout-sensitive and NOT covered by inner_tree (which only
      // fixes the add tree), so relayouting would change the bits. Skip it.
      // Checking operand[0] is sufficient: FMA contraction only arises for a
      // single-operand sum(a*b); variadic reduces (e.g. welford) use non-add
      // combines and are not inner_tree add-reduces, so no other operand can
      // introduce the mul+add contraction this guard protects against.
      if (auto *def = red.getOperands()[0].getDefiningOp())
        if (isa<arith::MulFOp>(def))
          continue;
      int axis = red.getAxis();
      ReduceOpHelper helper(red);
      if (!helper.isReduceWithinCTA()) // needs CTASplitNum[axis] == 1
        continue;
      if (helper.isWarpSynchronous()) // already no cross-warp stage
        continue;

      int numWarps = lookupNumWarps(red);

      // Register-pressure guard: the reduce-friendly layout carries the axis slice
      // per thread, so if the tile has more elements per thread than the register
      // file holds (~256 for f32), the relayout spills and loses. Skip those (they
      // are the low-warp-count / very-large-tile corner). Env-tunable.
      int64_t tileElems = 1;
      for (int64_t d : srcTy.getShape())
        tileElems *= d;
      int64_t elemsPerThread = tileElems / (int64_t(warpSize) * numWarps);
      if (elemsPerThread > maxElemsPerThread)
        continue;

      BlockedEncodingAttr cand =
          makeIdeal(red, blocked, axis, srcTy.getShape(), numWarps, warpSize);
      if (!cand)
        continue;

      // Cost guard: the win is converting cross-warp / register reduction work into
      // cheap intra-warp shuffles, which only pays for the convert_layout when the
      // axis is BOTH gaining lanes AND currently badly under-parallelized.
      unsigned curIntra = blocked.getThreadsPerWarp()[axis];
      unsigned newIntra = cand.getThreadsPerWarp()[axis];
      int64_t axisExtent = srcTy.getShape()[axis];
      if (newIntra <= curIntra)
        continue; // candidate does not add intra-warp parallelism to the axis
      if (axisExtent < int64_t(curIntra) * minUnderparallel)
        continue; // axis already well spread over lanes -> convert not worth it

      rewriteOperandLayout(red, cand, axis);
    }
  }
};

} // namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
