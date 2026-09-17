// Copy allocation for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §2.3, Step 4).
//
// Given a fixed grouping (Plan.blocks), enumerate bounded multi-buffer depths.
// The first result is the existing greedy allocation, preserving TOPK=1.
// Further results explore the structural space from the correctness-floor
// vector in increasing edit distance:
//
//   1. Correctness floors, set first and never reverted for budget:
//        copies >= max(cross-stage stageSpan of members)          (deadlock)
//        copies >= sum(entries of members)                        (slot
//        collision)
//   2. Discretionary copies added greedily by benefit density for result zero
//        (score delta / footprint delta) until no block has positive benefit
//        or none fits the budget. Because per-copy latency benefit is concave,
//        this greedy is the exact optimum for the separable knapsack.
//   3. Alternative feasible vectors up to each block's representable maximum,
//        independent of the latency model, for runtime selection.
//
// The CostModel is the single source of the objective — marginal benefit is a
// score delta — so this solver needs no knowledge of the latency formula. The
// Packer supplies pool-specific footprint and feasibility.

#include "WSMemoryPlanSearch.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <vector>

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

/// Maximum representable discretionary depth for a block. All members share
/// one physical ring, so the narrowest member capability is the block cap.
/// Correctness wins over configuration: a floor above the cap remains legal
/// here and will be diagnosed by the ordinary pool-budget backstop if needed.
static unsigned copyCeiling(const BufferModel &model, const Block &blk) {
  unsigned ceiling = std::numeric_limits<unsigned>::max();
  for (BufferId b : blk.members)
    ceiling = std::min(ceiling, model.maxCopies(b));
  if (ceiling == std::numeric_limits<unsigned>::max())
    ceiling = 1;
  return std::max(copyFloor(model, blk), ceiling);
}

static CopyMap makeCopyMap(const Plan &grouping, ArrayRef<unsigned> depths) {
  CopyMap result;
  for (unsigned i = 0; i < grouping.blocks.size(); ++i)
    result[grouping.blocks[i].id] = depths[i];
  return result;
}

static Plan withCopies(Plan plan, ArrayRef<unsigned> depths) {
  for (unsigned i = 0; i < plan.blocks.size(); ++i)
    plan.blocks[i].copies = depths[i];
  return plan;
}

static bool sameCopyMap(const Plan &grouping, const CopyMap &a,
                        const CopyMap &b) {
  for (const Block &blk : grouping.blocks) {
    if (a.lookup(blk.id) != b.lookup(blk.id))
      return false;
  }
  return true;
}

static void appendUnique(CopyMaps &maps, const Plan &grouping, CopyMap map) {
  if (llvm::none_of(maps, [&](const CopyMap &other) {
        return sameCopyMap(grouping, map, other);
      }))
    maps.push_back(std::move(map));
}

class GreedyCopySolver : public CopySolver {
public:
  CopyMaps enumerate(const BufferModel &model, const Packer &packer,
                     const Plan &grouping, const Budget &budget,
                     const CostModel &cost, unsigned limit) const override {
    CopyMaps results;
    if (limit == 0)
      return results;

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
        if (blk.copies >= copyCeiling(model, blk))
          continue;
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

    SmallVector<unsigned> greedyDepths;
    greedyDepths.reserve(work.blocks.size());
    for (const Block &blk : work.blocks)
      greedyDepths.push_back(blk.copies);
    appendUnique(results, grouping, makeCopyMap(grouping, greedyDepths));
    if (results.size() == limit)
      return results;

    // Enumerate alternatives from the correctness-floor vector. FIFO breadth
    // first gives the deterministic policy documented for the search:
    // baseline, every one-block +1 neighbor, then combinations. This order is
    // structural; latency only ranks the resulting complete plans.
    struct State {
      std::vector<unsigned> depths;
    };
    std::queue<State> frontier;
    std::set<std::vector<unsigned>> seen;
    std::vector<unsigned> floors;
    SmallVector<unsigned> ceilings;
    floors.reserve(grouping.blocks.size());
    ceilings.reserve(grouping.blocks.size());
    for (const Block &blk : grouping.blocks) {
      floors.push_back(copyFloor(model, blk));
      ceilings.push_back(copyCeiling(model, blk));
    }
    frontier.push({floors});
    seen.insert(floors);

    while (!frontier.empty() && results.size() < limit) {
      State state = std::move(frontier.front());
      frontier.pop();

      Plan candidate = withCopies(grouping, state.depths);
      bool feasible = packer.feasible(candidate, budget);
      if (feasible)
        appendUnique(results, grouping, makeCopyMap(grouping, state.depths));

      // Feasibility is monotone in copy depth, so an over-budget state cannot
      // have a feasible descendant.
      if (!feasible)
        continue;
      for (unsigned i = 0; i < state.depths.size(); ++i) {
        if (state.depths[i] >= ceilings[i])
          continue;
        State next = state;
        ++next.depths[i];
        if (seen.insert(next.depths).second)
          frontier.push(std::move(next));
      }
    }
    return results;
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
