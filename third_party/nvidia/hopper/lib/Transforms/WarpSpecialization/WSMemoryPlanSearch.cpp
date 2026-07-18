// Beam-search driver for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §3.3, Step 6).
//
// Places buffers one at a time in the OrderingPolicy's sequence. At each level
// a partial plan branches into: (a) join buffer `b` into each existing block
// the Packer deems legal, and (b) open a new block for `b`. Infeasible branches
// are dropped; the surviving partials are ranked and truncated to the beam
// width W.
//
// The copy dimension is solved in closed form per grouping (CopySolver), not
// branched: it is used both to rank partials (a good grouping frees budget for
// more copies -> higher score) and to finalize the leaf plans. Because all
// partials at a given level have placed the SAME prefix of buffers, ranking by
// score alone is apples-to-apples — no optimistic remainder term is needed.

#include "WSMemoryPlanSearch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "nvgpu-ws-memory-planner"

namespace mlir {
namespace wsplan {

namespace {

/// Apply the CopySolver to `plan` (a grouping) and return its scored copy. Used
/// both for beam ranking and leaf finalization so the two stay consistent.
Plan scoreWithCopies(const BufferModel &model, const Packer &packer,
                     const Budget &budget, const CostModel &cost,
                     const CopySolver &copies, Plan plan) {
  CopyMap cm = copies.solve(model, packer, plan, budget, cost);
  for (Block &blk : plan.blocks) {
    auto it = cm.find(blk.id);
    if (it != cm.end())
      blk.copies = it->second;
  }
  plan.score = cost.score(plan);
  return plan;
}

/// Return `plan` with `b` appended to block index `blockIdx`.
Plan withJoin(Plan plan, const Packer &packer, BufferId b, unsigned blockIdx) {
  Block &blk = plan.blocks[blockIdx];
  blk.members.push_back(b);
  blk.placement[b] = packer.place(blk, b);
  plan.blockOf[b] = blockIdx;
  return plan;
}

/// Return `plan` with a fresh single-member block for `b`.
Plan withNewBlock(Plan plan, const Packer &packer, BufferId b) {
  Block blk;
  blk.id = static_cast<BlockId>(plan.blocks.size());
  blk.members.push_back(b);
  blk.copies = 1;
  unsigned idx = plan.blocks.size();
  plan.blocks.push_back(std::move(blk));
  plan.blocks[idx].placement[b] = packer.place(plan.blocks[idx], b);
  plan.blockOf[b] = idx;
  return plan;
}

} // namespace

TopKPlans beamSearch(const BufferModel &model, const OrderingPolicy &ordering,
                     const Packer &packer, const CostModel &cost,
                     const CopySolver &copies, const Budget &budget, unsigned W,
                     unsigned K) {
  TopKPlans out;
  if (model.buffers().empty() || W == 0 || K == 0)
    return out;

  SmallVector<BufferId> seq = ordering.order(model);

  SmallVector<Plan> beam;
  beam.emplace_back(); // empty partial

  for (BufferId b : seq) {
    SmallVector<Plan> next;
    for (const Plan &p : beam) {
      // (a) join an existing legal block
      for (unsigned gi = 0; gi < p.blocks.size(); ++gi) {
        if (!packer.legalJoin(p, b, p.blocks[gi].id))
          continue;
        Plan cand = withJoin(p, packer, b, gi);
        if (packer.feasible(cand, budget))
          next.push_back(std::move(cand));
      }
      // (b) open a new block
      Plan fresh = withNewBlock(p, packer, b);
      if (packer.feasible(fresh, budget))
        next.push_back(std::move(fresh));
    }

    if (next.empty()) {
      // No legal+feasible placement for `b` from any partial. This should not
      // happen (opening a new single-buffered block is always available unless
      // even one copy overflows the budget); surface it rather than silently
      // returning a truncated result.
      LLVM_DEBUG(llvm::dbgs()
                 << "[ws-plan] no feasible placement for buffer " << b << "\n");
      return out;
    }

    // Rank by best achievable score for the grouping so far, then keep top-W.
    SmallVector<std::pair<double, unsigned>> ranked;
    ranked.reserve(next.size());
    for (unsigned i = 0; i < next.size(); ++i) {
      Plan scored =
          scoreWithCopies(model, packer, budget, cost, copies, next[i]);
      ranked.push_back({scored.score, i});
    }
    llvm::stable_sort(ranked, [](const std::pair<double, unsigned> &x,
                                 const std::pair<double, unsigned> &y) {
      return x.first > y.first; // best-first
    });

    SmallVector<Plan> pruned;
    unsigned keep = std::min<unsigned>(W, ranked.size());
    for (unsigned i = 0; i < keep; ++i)
      pruned.push_back(std::move(next[ranked[i].second]));
    if (ranked.size() > keep)
      LLVM_DEBUG(llvm::dbgs()
                 << "[ws-plan] beam truncated " << ranked.size() << " -> "
                 << keep << " partials at buffer " << b << "\n");
    beam = std::move(pruned);
  }

  // Finalize: solve copies + score every beam leaf, then take the top K.
  SmallVector<Plan> leaves;
  leaves.reserve(beam.size());
  for (Plan &p : beam)
    leaves.push_back(scoreWithCopies(model, packer, budget, cost, copies, p));
  llvm::stable_sort(
      leaves, [](const Plan &x, const Plan &y) { return x.score > y.score; });

  unsigned n = std::min<unsigned>(K, leaves.size());
  for (unsigned i = 0; i < n; ++i)
    out.push_back(std::move(leaves[i]));
  return out;
}

} // namespace wsplan
} // namespace mlir
