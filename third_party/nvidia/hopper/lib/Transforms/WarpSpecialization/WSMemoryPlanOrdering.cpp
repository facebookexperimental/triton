// Ordering policies for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §3.2, Step 2).
//
// The ordering decides the sequence in which buffers are offered to the beam
// search. Per the design invariant (docs §3.1) it must NOT affect whether any
// plan is legal/feasible — only which partials the beam retains and how hard
// the bound prunes. Legality lives entirely in the Packer/BufferModel
// predicates, so a new ordering can be added here without touching any other
// module.

#include "WSMemoryPlanSearch.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace wsplan {

namespace {

/// Sort buffers by liveness start, with a deterministic BufferId tie-break so
/// the search is reproducible. Refactor target for the existing program-order
/// sorts (WSMemoryPlanner.cpp: sortChannelsByProgramOrder /
/// getWSBufferUsageOrder).
class LivenessStartOrder : public OrderingPolicy {
public:
  SmallVector<BufferId> order(const BufferModel &model) const override {
    ArrayRef<BufferId> src = model.buffers();
    SmallVector<BufferId> ids(src.begin(), src.end());
    llvm::stable_sort(ids, [&](BufferId a, BufferId b) {
      size_t sa = model.liveness(a).start();
      size_t sb = model.liveness(b).start();
      if (sa != sb)
        return sa < sb;
      return a < b; // deterministic tie-break
    });
    return ids;
  }
};

/// Dependency topological order (docs §8 / Step 10). Places a buffer's
/// producers before the buffer, so reuse candidates are offered only once their
/// live-range predecessors are settled — pruning harder when reuse is
/// dependency-dominated (TMEM). Deterministic Kahn's algorithm with a
/// (liveness-start, BufferId) tie-break; a defensive fallback breaks any cycle.
///
/// Adding this touches no other module — the proof that the docs §3.1 ordering
/// invariant held.
class TopologicalOrder : public OrderingPolicy {
public:
  SmallVector<BufferId> order(const BufferModel &model) const override {
    ArrayRef<BufferId> ids = model.buffers();

    // Edge b -> a when a depends on b; indeg[a] counts a's dependencies.
    DenseMap<BufferId, unsigned> indeg;
    for (BufferId a : ids)
      indeg[a] = 0;
    for (BufferId a : ids)
      for (BufferId b : ids)
        if (a != b && model.dependsOn(a, b))
          indeg[a] += 1;

    auto earlier = [&](BufferId x, BufferId y) {
      size_t sx = model.liveness(x).start();
      size_t sy = model.liveness(y).start();
      if (sx != sy)
        return sx < sy;
      return x < y; // deterministic tie-break
    };

    DenseSet<BufferId> done;
    SmallVector<BufferId> result;
    result.reserve(ids.size());

    while (result.size() < ids.size()) {
      // Pick the earliest ready (indeg 0) node; if none (cycle), fall back to
      // the earliest remaining node regardless of indegree.
      BufferId best = 0;
      bool found = false;
      for (BufferId x : ids) {
        if (done.count(x) || indeg[x] != 0)
          continue;
        if (!found || earlier(x, best)) {
          best = x;
          found = true;
        }
      }
      if (!found) {
        for (BufferId x : ids) {
          if (done.count(x))
            continue;
          if (!found || earlier(x, best)) {
            best = x;
            found = true;
          }
        }
      }

      done.insert(best);
      result.push_back(best);
      for (BufferId a : ids)
        if (!done.count(a) && a != best && indeg[a] > 0 &&
            model.dependsOn(a, best))
          indeg[a] -= 1;
    }
    return result;
  }
};

} // namespace

std::unique_ptr<OrderingPolicy> createOrderingPolicy(StringRef name) {
  if (name.empty() || name == "liveness")
    return std::make_unique<LivenessStartOrder>();
  if (name == "topo")
    return std::make_unique<TopologicalOrder>();
  // Unknown names return nullptr so the caller can diagnose a bad
  // --mem-plan-order value.
  return nullptr;
}

} // namespace wsplan
} // namespace mlir
