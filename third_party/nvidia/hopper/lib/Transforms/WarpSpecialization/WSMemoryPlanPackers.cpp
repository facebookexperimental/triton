// Packers for the memory plan-space search
// (docs/MemoryPlannerSearchSpace.plan.md §5.3/§5.4, Steps 3 & 7).
//
// A Packer defines pool-specific block semantics: whether a buffer may join a
// block (reuse legality), whether a partial plan fits the pool budget, a
// block's footprint, and (TMEM) placement. Legality is expressed as explicit
// predicates over the partial plan, never inferred from ordering (docs §3.1).
//
// This file currently holds the SMEM packer (Step 3). The TMEM packer (Step 7)
// wraps the existing backtracking placer and lands here next.

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

  bool legalJoin(const PartialPlan &plan, BufferId b,
                 BlockId blockId) const override {
    const Block *blk = findBlock(plan, blockId);
    if (!blk || blk->members.empty())
      return false;
    // Circular reuse requires (a) a common encoding and (b) all endpoints in
    // one basic block, captured by a shared reuseScope (verifyReuseGroup1).
    // Producer/ consumer separation within the block is handled by the circular
    // slot indexing, so no liveness check is needed (docs §5.3).
    // TODO(neutral-reuse): relax the encoding gate under
    // TRITON_WS_NEUTRAL_REUSE.
    EncodingKey key = model.encoding(b);
    unsigned scope = model.reuseScope(b);
    for (BufferId m : blk->members)
      if (model.encoding(m) != key || model.reuseScope(m) != scope)
        return false;
    return true;
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

} // namespace

std::unique_ptr<Packer> createSmemPacker(const BufferModel &model) {
  return std::make_unique<SmemPacker>(model);
}

} // namespace wsplan
} // namespace mlir
