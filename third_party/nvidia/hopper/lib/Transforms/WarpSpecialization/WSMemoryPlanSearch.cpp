// Beam-search driver for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §3.3, Step 6).
//
// Places buffers one at a time in the OrderingPolicy's sequence. At each level
// a partial plan branches into: (a) join buffer `b` into each existing block
// the Packer deems legal, and (b) open a new block for `b`. Infeasible branches
// are dropped; the surviving partials are ranked and truncated to the beam
// width W.
//
// Partial groupings use CopySolver's legacy greedy result for ranking (a good
// grouping frees budget for more copies -> higher score). Complete leaves
// branch over the solver's bounded copy-count frontier before global top-K
// selection. Because all partials at a given level have placed the SAME prefix
// of buffers, ranking by score alone is apples-to-apples — no optimistic
// remainder term is needed.

#include "WSMemoryPlanSearch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <optional>

#define DEBUG_TYPE "nvgpu-ws-memory-planner"

namespace mlir {
namespace wsplan {

namespace {

/// Apply one copy assignment to `plan`, validate it, and score it.
std::optional<Plan> applyCopiesAndScore(const BufferModel &model,
                                        const Packer &packer,
                                        const Budget &budget,
                                        const CostModel &cost,
                                        const CopySafetyValidator &validator,
                                        Plan plan, const CopyMap &cm) {
  for (Block &blk : plan.blocks) {
    auto it = cm.find(blk.id);
    if (it != cm.end())
      blk.copies = it->second;
  }
  SmallVector<CopySafetyFailure> failures;
  if (!validator.validate(model, plan, &failures)) {
    LLVM_DEBUG(for (const CopySafetyFailure &failure : failures) {
      llvm::dbgs() << "[ws-plan] reject unsafe block " << failure.block
                   << " buffer " << failure.buffer << ": proposed "
                   << failure.proposedCopies << ", required "
                   << failure.requiredCopies
                   << " (missing release -> overwrite ordering)\n";
    });
    return std::nullopt;
  }
  if (!packer.feasible(plan, budget)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[ws-plan] reject copy-solved plan over budget\n");
    return std::nullopt;
  }
  plan.score = cost.score(plan);
  return plan;
}

/// Rank a partial grouping with the solver's first (legacy greedy) result.
std::optional<Plan>
scoreWithGreedyCopies(const BufferModel &model, const Packer &packer,
                      const Budget &budget, const CostModel &cost,
                      const CopySolver &copies,
                      const CopySafetyValidator &validator, Plan plan) {
  CopyMap cm = copies.solve(model, packer, plan, budget, cost);
  return applyCopiesAndScore(model, packer, budget, cost, validator,
                             std::move(plan), cm);
}

static bool samePlan(const Plan &a, const Plan &b) {
  if (a.blocks.size() != b.blocks.size())
    return false;
  for (unsigned i = 0; i < a.blocks.size(); ++i) {
    const Block &aBlock = a.blocks[i];
    const Block &bBlock = b.blocks[i];
    if (aBlock.copies != bBlock.copies ||
        aBlock.countsTowardBudget != bBlock.countsTowardBudget ||
        aBlock.members != bBlock.members)
      return false;
    for (BufferId member : aBlock.members) {
      auto aPlacement = aBlock.placement.find(member);
      auto bPlacement = bBlock.placement.find(member);
      bool aMissing = aPlacement == aBlock.placement.end();
      bool bMissing = bPlacement == bBlock.placement.end();
      if (aMissing != bMissing)
        return false;
      if (aMissing)
        continue;
      if (aPlacement->second.rowOffset != bPlacement->second.rowOffset ||
          aPlacement->second.colOffset != bPlacement->second.colOffset)
        return false;
    }
  }
  return true;
}

static void appendUnique(SmallVectorImpl<Plan> &plans, Plan plan) {
  if (llvm::none_of(plans,
                    [&](const Plan &other) { return samePlan(plan, other); }))
    plans.push_back(std::move(plan));
}

/// Prefer a conservative grouping that introduces the fewest reuse edges.
/// Among equally conservative plans, keep buffers that are adjacent in the
/// model's deterministic order together, preferring reuse among the earliest
/// such buffers. This preserves a structurally distinct, low-aliasing
/// candidate without using latency as an admission rule or naming any
/// particular operand.
static bool isMoreConservative(const Plan &a, const Plan &b) {
  if (a.blocks.size() != b.blocks.size())
    return a.blocks.size() > b.blocks.size();

  auto fragmentation = [](const Plan &plan) {
    uint64_t holes = 0;
    for (const Block &block : plan.blocks) {
      if (block.members.empty())
        continue;
      auto [minIt, maxIt] =
          std::minmax_element(block.members.begin(), block.members.end());
      holes +=
          static_cast<uint64_t>(*maxIt - *minIt + 1) - block.members.size();
    }
    return holes;
  };
  uint64_t aHoles = fragmentation(a), bHoles = fragmentation(b);
  if (aHoles != bHoles)
    return aHoles < bHoles;

  auto orderedGroupSizes = [](const Plan &plan) {
    SmallVector<std::pair<BufferId, unsigned>> groups;
    for (const Block &block : plan.blocks) {
      if (block.members.empty())
        continue;
      groups.push_back(
          {*std::min_element(block.members.begin(), block.members.end()),
           static_cast<unsigned>(block.members.size())});
    }
    llvm::sort(groups);
    SmallVector<unsigned> sizes;
    for (auto [first, size] : groups)
      sizes.push_back(size);
    return sizes;
  };
  SmallVector<unsigned> aSizes = orderedGroupSizes(a);
  SmallVector<unsigned> bSizes = orderedGroupSizes(b);
  return std::lexicographical_compare(bSizes.begin(), bSizes.end(),
                                      aSizes.begin(), aSizes.end());
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
                     const CopySolver &copies,
                     const CopySafetyValidator &validator, const Budget &budget,
                     unsigned W, unsigned K) {
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
      auto scored = scoreWithGreedyCopies(model, packer, budget, cost, copies,
                                          validator, next[i]);
      if (scored)
        ranked.push_back({scored->score, i});
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

  // Finalize: branch over bounded copy assignments for every grouping leaf,
  // validate + score each concrete plan, deduplicate, then take global top-K.
  SmallVector<Plan> leaves;
  leaves.reserve(beam.size() * K);
  for (Plan &p : beam) {
    CopyMaps copyMaps = copies.enumerate(model, packer, p, budget, cost, K);
    for (const CopyMap &cm : copyMaps) {
      if (auto scored = applyCopiesAndScore(model, packer, budget, cost,
                                            validator, p, cm))
        appendUnique(leaves, std::move(*scored));
    }
  }
  llvm::stable_sort(
      leaves, [](const Plan &x, const Plan &y) { return x.score > y.score; });

  // Keep the existing best-scored plan at rank zero. When grouping choices
  // exist, reserve rank one for the least aggressive feasible reuse topology;
  // fill the remainder by score. This prevents a small top-K from containing
  // only maximally packed plans with different reuse edges.
  appendUnique(out, leaves.front());
  bool hasGroupingDiversity = llvm::any_of(leaves, [&](const Plan &plan) {
    return plan.blocks.size() != leaves.front().blocks.size();
  });
  if (K > 1 && hasGroupingDiversity) {
    auto conservative = std::max_element(
        leaves.begin(), leaves.end(),
        [](const Plan &a, const Plan &b) { return isMoreConservative(b, a); });
    appendUnique(out, *conservative);
  }
  for (const Plan &plan : leaves) {
    if (out.size() == K)
      break;
    appendUnique(out, plan);
  }
  return out;
}

} // namespace wsplan
} // namespace mlir
