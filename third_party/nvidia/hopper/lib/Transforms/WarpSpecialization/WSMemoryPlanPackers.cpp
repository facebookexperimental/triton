// Packers for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §5.3/§5.4, Steps 3 & 7).
//
// A Packer defines pool-specific block semantics: whether a buffer may join a
// block (reuse legality), whether a partial plan fits the pool budget, a
// block's footprint, and (TMEM) placement. Legality is expressed as explicit
// predicates over the partial plan, never inferred from ordering (docs §3.1).
//
// SmemPacker (Step 3): circular multi-buffer reuse groups. TmemPacker (Step 7):
// row-owners with time-multiplexed (liveness-disjoint) column reuse, copies
// pinned to 1.

#include "WSMemoryPlanSearch.h"

#include <algorithm>

namespace mlir {
namespace wsplan {

namespace {

/// Locate a block by id within a plan (ids are assigned densely at creation,
/// but search by id so we never depend on vector position).
static const Block *findBlock(const Plan &plan, BlockId id) {
  for (const Block &blk : plan.blocks)
    if (blk.id == id)
      return &blk;
  return nullptr;
}

/// SMEM: a block is a circular multi-buffer reuse group. Physical size is the
/// largest member times the shared copy count; members must share an encoding.
class SmemPacker : public Packer {
public:
  explicit SmemPacker(const BufferModel &model) : model(model) {}

  bool legalJoin(const PartialPlan &, BufferId, BlockId) const override {
    // No SMEM circular reuse grouping. Matching encoding + basic block is NOT
    // sufficient for safe circular reuse: two concurrently-live buffers (e.g.
    // an addmm/GEMM A and B operand) satisfy it yet must not share one circular
    // buffer — doing so corrupts data (deterministic wrong results on addmm,
    // races on GEMM). The upstream heuristic defaults `smem-circular-reuse` OFF
    // for the same reason. The search therefore only multi-buffers individual
    // buffers (own block, own copy count), which is always safe. Safe reuse
    // grouping (the true rotating-entry condition, not just encoding + scope)
    // is future work; see docs §8.
    return false;
  }

  Footprint footprint(const Block &blk) const override {
    uint64_t maxBytes = 0;
    for (BufferId m : blk.members)
      maxBytes = std::max(maxBytes, model.size(m).bytes);
    Footprint f;
    f.bytes = maxBytes * static_cast<uint64_t>(blk.copies);
    return f;
  }

  bool feasible(const PartialPlan &plan, const Budget &budget) const override {
    uint64_t total = 0;
    for (const Block &blk : plan.blocks)
      total += footprint(blk).bytes;
    return total <= budget.smemBytes;
  }

  Placement place(const Block &, BufferId) const override {
    return Placement{}; // SMEM blocks are sized by max member; no offsets.
  }

private:
  const BufferModel &model;
};

/// TMEM: a block is a row-owner; members time-multiplex the same rows x cols
/// space (all at column offset 0). Reuse is legal only when a candidate shares
/// the members' encoding, has disjoint liveness, AND is bidirectionally
/// data-dependent with them — the automatic-reuse predicate the existing placer
/// uses (hasPotentialReuse score 1). The data dependency is load-bearing:
/// liveness-disjoint-but-independent buffers can be concurrent across warp
/// partitions despite disjoint op-id intervals. Copies stay 1 (caller pins).
class TmemPacker : public Packer {
public:
  explicit TmemPacker(const BufferModel &model) : model(model) {}

  bool legalJoin(const PartialPlan &plan, BufferId b,
                 BlockId blockId) const override {
    const Block *blk = findBlock(plan, blockId);
    if (!blk || blk->members.empty())
      return false;
    EncodingKey key = model.encoding(b);
    Interval<size_t> live = model.liveness(b);
    for (BufferId m : blk->members) {
      if (model.encoding(m) != key)
        return false;
      // Match the proven automatic-reuse predicate (hasPotentialReuse score 1):
      // disjoint liveness AND a bidirectional data dependency. The dependency
      // is essential — two liveness-disjoint but *independent* buffers can live
      // in different warp-specialized partitions and be concurrent at run time
      // despite disjoint op-id intervals, so sharing columns would
      // race/deadlock.
      if (live.intersects(model.liveness(m)))
        return false;
      if (!model.dependsOn(b, m) && !model.dependsOn(m, b))
        return false;
    }
    return true;
  }

  Footprint footprint(const Block &blk) const override {
    Footprint f;
    for (BufferId m : blk.members) {
      Footprint mf = model.size(m);
      f.rows = std::max(f.rows, mf.rows);
      f.cols = std::max(f.cols, mf.cols);
    }
    return f;
  }

  bool feasible(const PartialPlan &plan, const Budget &budget) const override {
    unsigned totalRows = 0;
    for (const Block &blk : plan.blocks) {
      Footprint f = footprint(blk);
      if (f.cols > budget.tmemCols)
        return false;
      totalRows += f.rows;
    }
    return totalRows <= budget.tmemRows;
  }

  Placement place(const Block &, BufferId) const override {
    return Placement{}; // time-multiplexed: all members share offset 0.
  }

private:
  const BufferModel &model;
};

} // namespace

std::unique_ptr<Packer> createSmemPacker(const BufferModel &model) {
  return std::make_unique<SmemPacker>(model);
}

std::unique_ptr<Packer> createTmemPacker(const BufferModel &model) {
  return std::make_unique<TmemPacker>(model);
}

} // namespace wsplan
} // namespace mlir
