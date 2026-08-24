// Copy allocation for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §2.3, Step 4).
//
// Given a fixed grouping (Plan.blocks), decide each block's multi-buffer depth
// (`copies`). Two parts:
//
//   1. Correctness floors, set first and never reverted for budget:
//        copies >= max(cross-stage stageSpan of members)          (deadlock)
//        copies >= sum(entries of members)                        (slot
//        collision)
//   2. Discretionary copies added greedily by benefit density
//        (score delta / footprint delta) until no block has positive benefit
//        or none fits the budget. Because per-copy latency benefit is concave,
//        this greedy is the exact optimum for the separable knapsack.
//
// The CostModel is the single source of the objective — marginal benefit is a
// score delta — so this solver needs no knowledge of the latency formula. The
// Packer supplies pool-specific footprint and feasibility.

#include "WSMemoryPlanSearch.h"

#include <algorithm>
#include <limits>

namespace mlir {
namespace wsplan {

namespace {

/// Collapse a Footprint to a single scalar for density comparison. SMEM
/// footprints carry only `bytes`; TMEM only `rows`/`cols`; the unused fields
/// are zero, so the sum picks out the relevant pool's magnitude.
static double footprintScalar(const Footprint &f) {
  return static_cast<double>(f.bytes) +
         static_cast<double>(f.rows) * static_cast<double>(f.cols);
}

/// Correctness copy floor for a block (docs §2.2).
///
/// Combines two independent floors:
///   - cross-stage (`stageSpan`): a member consumed at N distinct `loop.stage`
///     values needs N copies so two stages never alias one slot.
///   - slot-collision (`entries`, summed over members): members fused into one
///     block share a single physical allocation and are addressed by a *static*
///     per-member slot index, so the block needs one slot per member entry.
///
/// The slot term is an assignment floor, not a liveness result: two members
/// that happen to be serialized still get distinct slots, because the index is
/// fixed at lowering time rather than derived from a concurrency analysis.
/// Consequently a two-member fused block floors at 2 even when one slot would
/// be safe under a drain-between-members schedule. `entries()` is currently 1
/// per member (see the data-partition TODO in the BufferModel builders), so
/// today the slot floor equals the member count.
static unsigned copyFloor(const BufferModel &model, const Block &blk) {
  unsigned crossStage = 1;
  unsigned slot = 0;
  for (BufferId b : blk.members) {
    crossStage = std::max(crossStage, model.stageSpan(b));
    slot += model.entries(b);
  }
  return std::max({crossStage, slot, 1u});
}

class GreedyCopySolver : public CopySolver {
public:
  CopyMap solve(const BufferModel &model, const Packer &packer,
                const Plan &grouping, const Budget &budget,
                const CostModel &cost) const override {
    // Work on a local copy so we can probe score/footprint at tentative depths.
    Plan work = grouping;

    // Part 1: correctness floors (exempt from budget).
    for (Block &blk : work.blocks)
      blk.copies = copyFloor(model, blk);

    // Part 2: greedy discretionary increase by benefit density.
    while (true) {
      double baseScore = cost.score(work);
      int bestIdx = -1;
      double bestDensity = -std::numeric_limits<double>::infinity();

      for (unsigned i = 0; i < work.blocks.size(); ++i) {
        Block &blk = work.blocks[i];
        double footBefore = footprintScalar(packer.footprint(blk));

        blk.copies += 1; // tentative
        bool feas = packer.feasible(work, budget);
        double dScore = cost.score(work) - baseScore;
        double footAfter = footprintScalar(packer.footprint(blk));
        blk.copies -= 1; // revert

        if (!feas || dScore <= 0.0)
          continue; // over budget, or benefit already saturated

        double dFoot = footAfter - footBefore;
        double density = dFoot > 0.0
                             ? dScore / dFoot
                             : dScore * std::numeric_limits<double>::max();
        if (density > bestDensity) {
          bestDensity = density;
          bestIdx = static_cast<int>(i);
        }
      }

      if (bestIdx < 0)
        break; // nothing beneficial fits
      work.blocks[bestIdx].copies += 1;
    }

    CopyMap result;
    for (const Block &blk : work.blocks)
      result[blk.id] = blk.copies;
    return result;
  }
};

class StaticCopySafetyValidator : public CopySafetyValidator {
public:
  bool validate(const BufferModel &model, const Plan &plan,
                SmallVectorImpl<CopySafetyFailure> *failures) const override {
    bool safe = true;
    for (const Block &blk : plan.blocks) {
      unsigned memberEntries = 0;
      for (BufferId b : blk.members)
        memberEntries += model.entries(b);
      for (BufferId b : blk.members) {
        unsigned required = std::max({1u, memberEntries, model.stageSpan(b)});
        if (blk.copies >= required)
          continue;
        safe = false;
        if (failures)
          failures->push_back({blk.id, b, blk.copies, required});
      }
    }
    return safe;
  }
};

} // namespace

std::unique_ptr<CopySolver> createGreedyCopySolver() {
  return std::make_unique<GreedyCopySolver>();
}

std::unique_ptr<CopySafetyValidator> createStaticCopySafetyValidator() {
  return std::make_unique<StaticCopySafetyValidator>();
}

} // namespace wsplan
} // namespace mlir
