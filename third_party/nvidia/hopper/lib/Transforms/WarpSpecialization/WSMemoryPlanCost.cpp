// Latency-hiding cost model for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §4, Step 5).
//
// The same function orders candidates and (via CopySolver) drives copy
// allocation:
//
//   Score(P) = Σ_block Σ_member freq·min(copies·II, L) − λ·occupancy_penalty
//
// `score` is the exact value of a *complete* plan (copies finalized). `bound`
// is an admissible optimistic upper bound used by branch-and-bound: it assumes
// every buffer (placed or not) can hide its full producer latency and drops the
// (non-negative) penalty, so it never underestimates any completion's score.

#include "WSMemoryPlanSearch.h"

#include <algorithm>

namespace mlir {
namespace wsplan {

namespace {

/// Latency hidden by giving buffer `b` `copies` slots: freq · min(copies·II,
/// L). Concave/diminishing in `copies` — the property that makes the
/// CopySolver's greedy-by-benefit-density exact (docs §2.3).
static double hiddenLatency(const BufferModel &m, BufferId b, unsigned copies,
                            double II) {
  double reach = static_cast<double>(copies) * II;
  return m.freq(b) * std::min(reach, m.latency(b));
}

class LatencyCostModel : public CostModel {
public:
  LatencyCostModel(const BufferModel &model, double II, double lambda)
      : model(model), II(II), lambda(lambda) {}

  double score(const Plan &plan) const override {
    double benefit = 0.0;
    for (const Block &blk : plan.blocks)
      for (BufferId b : blk.members)
        benefit += hiddenLatency(model, b, blk.copies, II);

    if (lambda == 0.0)
      return benefit;

    // Placeholder occupancy proxy: total resource footprint (a block occupies
    // max(member size) · copies). Only active when λ>0, which is a deferred
    // open item (docs §8); refined alongside a real occupancy model.
    double penalty = 0.0;
    for (const Block &blk : plan.blocks) {
      double blockUnits = 0.0;
      for (BufferId b : blk.members) {
        Footprint f = model.size(b);
        double units =
            static_cast<double>(f.bytes) +
            static_cast<double>(f.rows) * static_cast<double>(f.cols);
        blockUnits = std::max(blockUnits, units);
      }
      penalty += blockUnits * static_cast<double>(blk.copies);
    }
    return benefit - lambda * penalty;
  }

  double bound(const PartialPlan &partial,
               ArrayRef<BufferId> remaining) const override {
    // Optimistic: assume full latency hidden for every buffer (min(·,L) ≤ L)
    // and drop the penalty (≥ 0). This is an upper bound on any completion.
    double b = 0.0;
    for (const Block &blk : partial.blocks)
      for (BufferId id : blk.members)
        b += model.freq(id) * model.latency(id);
    for (BufferId id : remaining)
      b += model.freq(id) * model.latency(id);
    return b;
  }

private:
  const BufferModel &model;
  double II;
  double lambda;
};

} // namespace

std::unique_ptr<CostModel> createLatencyCostModel(const BufferModel &model,
                                                  double II, double lambda) {
  return std::make_unique<LatencyCostModel>(model, II, lambda);
}

} // namespace wsplan
} // namespace mlir
