#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WSMEMORYPLANSEARCH_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WSMEMORYPLANSEARCH_H_

// Plan-space search for the AutoWS memory planner.
//
// This header declares the modular seams described in
// docs/MemoryPlannerSearchSpace.plan.md. The search treats the planner's
// *output* (an allocation plan: grouping + copies + placement) as the search
// variable, enumerates the legal + feasible plans via a beam search, scores
// them with a latency-aware cost model, and keeps the top-K.
//
// Design invariant (docs §3.1): legality and feasibility are checked as
// explicit predicates against the partial plan, never inferred from a buffer's
// position in the ordering. This is what lets `OrderingPolicy` be swapped
// (liveness-start -> topological) without touching any other module.

#include "triton/Analysis/Allocation.h" // mlir::Interval

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace wsplan {

//===----------------------------------------------------------------------===//
// Value types
//===----------------------------------------------------------------------===//

/// Index into `BufferModel`'s buffer list. Decouples the search from the
/// underlying `Operation *` so the model can be built from IR or synthesized
/// in unit tests.
using BufferId = unsigned;

/// Index of a physical block within a `Plan`.
using BlockId = unsigned;

/// What role a buffer plays. Drives cost-model weighting and TMEM rotation
/// legality (accumulator vs operand — see docs §5.4 / §8).
enum class BufferKind : uint8_t {
  Other = 0,
  TMALoad,     // TMA / descriptor load (high producer latency)
  Operand,     // MMA operand staged in SMEM/TMEM
  Accumulator, // MMA operand-D (loop-carried acc token)
  Staging,     // TMA store/reduce staging buffer
};

/// Resource footprint of a block. SMEM uses `bytes`; TMEM uses `rows`/`cols`.
/// Each `Packer` interprets only the fields relevant to its pool.
struct Footprint {
  uint64_t bytes = 0; // SMEM
  unsigned rows = 0;  // TMEM
  unsigned cols = 0;  // TMEM
};

/// Physical pool limits. SMEM: `smemBytes`. TMEM: `tmemRows` x `tmemCols`
/// (512 rows on Blackwell, minus reserved scale columns).
struct Budget {
  uint64_t smemBytes = 0;
  unsigned tmemRows = 0;
  unsigned tmemCols = 0;
};

/// Reuse-compatibility key: two buffers may share a block only if their keys
/// compare equal (unless neutral-reuse relaxes it). Mirrors
/// `areReuseEncodingsCompatible` (WSMemoryPlanner.cpp:1702).
struct EncodingKey {
  Type elementType;
  Attribute encoding;

  bool operator==(const EncodingKey &o) const {
    return elementType == o.elementType && encoding == o.encoding;
  }
  bool operator!=(const EncodingKey &o) const { return !(*this == o); }
};

/// Placement of a buffer within its block. SMEM buffers use the default
/// (0, 0) — SMEM blocks are sized by `max(size) * copies`. TMEM buffers carry
/// a column offset within the owner's row space (`buffer.offset`).
struct Placement {
  unsigned rowOffset = 0;
  unsigned colOffset = 0;
};

/// A physical block: a reuse group sharing one `buffer.id`. `copies` is the
/// multi-buffer depth (shared by all members). `placement` holds per-member
/// offsets (TMEM only).
struct Block {
  BlockId id = 0;
  SmallVector<BufferId, 4> members;
  unsigned copies = 1;
  DenseMap<BufferId, Placement> placement;
};

/// A (possibly partial) allocation plan: the set of blocks plus a buffer->block
/// index. A plan is *complete* once every buffer in the model is assigned.
struct Plan {
  SmallVector<Block> blocks;
  DenseMap<BufferId, unsigned> blockOf; // buffer -> index into `blocks`
  double score = 0.0;                   // set by CostModel on complete plans

  bool assigns(BufferId b) const { return blockOf.count(b); }
};

/// A partial plan is structurally identical to a complete one; the engine
/// tracks completeness by how many buffers have been placed.
using PartialPlan = Plan;

/// Copy-count decision per block, produced by `CopySolver`.
using CopyMap = DenseMap<BlockId, unsigned>;

/// The K best complete plans, best-first.
using TopKPlans = SmallVector<Plan>;

//===----------------------------------------------------------------------===//
// Module interfaces (docs §3.2)
//===----------------------------------------------------------------------===//

/// Normalized per-buffer facts, built once from IR + schedule annotations
/// (or synthesized in tests). No policy logic lives here.
class BufferModel {
public:
  virtual ~BufferModel() = default;

  virtual ArrayRef<BufferId> buffers() const = 0;

  virtual Footprint size(BufferId) const = 0;
  virtual Interval<size_t> liveness(BufferId) const = 0;
  virtual unsigned stageSpan(BufferId) const = 0; // cross-stage floor
  virtual unsigned entries(BufferId) const = 0;   // slot-collision floor
  virtual EncodingKey encoding(BufferId) const = 0;
  virtual BufferKind kind(BufferId) const = 0;

  // Reuse-scope id: two buffers may share a multi-buffered reuse group only if
  // their scopes match. Captures the "all producers/consumers in one basic
  // block" invariant that circular accumCnt staggering requires
  // (verifyReuseGroup1); buffers whose endpoints span blocks get a unique scope
  // so they never group. See docs §5.3.
  virtual unsigned reuseScope(BufferId) const = 0;

  // producer L_b — issue-to-result latency the builder gets on demand from
  // ttg::NVLatencyModel (docs §4 / Step 0 revised); no scheduler annotation.
  virtual double latency(BufferId) const = 0;
  virtual double freq(BufferId) const = 0; // trip count

  /// True if `a` (transitively) depends on `b` via def-use. Used by
  /// `TopologicalOrder` and by TMEM reuse legality.
  virtual bool dependsOn(BufferId a, BufferId b) const = 0;
};

/// Produces the sequence in which buffers are placed. THE swap seam: changing
/// this must not change whether any plan is admitted (docs §3.1).
class OrderingPolicy {
public:
  virtual ~OrderingPolicy() = default;
  virtual SmallVector<BufferId> order(const BufferModel &) const = 0;
};

/// Factory for the `--mem-plan-order` knob. Known names:
///   "liveness" (default) — sort by liveness start; deterministic tie-break.
///   "topo"               — dependency topological order (docs §8 / Step 10).
/// Returns nullptr for an unknown name.
std::unique_ptr<OrderingPolicy> createOrderingPolicy(StringRef name);

/// Pool-specific block semantics: whether a buffer may join a block, whether a
/// partial plan fits the budget, a block's footprint, and (TMEM) placement.
class Packer {
public:
  virtual ~Packer() = default;

  virtual bool legalJoin(const PartialPlan &, BufferId, BlockId) const = 0;
  virtual bool feasible(const PartialPlan &, const Budget &) const = 0;
  virtual Footprint footprint(const Block &) const = 0;
  virtual Placement place(const Block &, BufferId) const = 0;
};

/// Scores complete plans and bounds partial ones. `bound` must be optimistic
/// (an upper bound on any completion's score) for branch-and-bound to be exact;
/// under beam it is used only as the ranking key.
class CostModel {
public:
  virtual ~CostModel() = default;
  virtual double score(const Plan &) const = 0;
  virtual double bound(const PartialPlan &,
                       ArrayRef<BufferId> remaining) const = 0;
};

/// Given a fixed grouping, allocates copies per block under the budget. The
/// latency benefit of the c-th copy is concave, so greedy-by-benefit-density is
/// exact (docs §2.3). Correctness floors (cross-stage, slot-collision) are hard
/// constraints seeded from the `BufferModel` and are set even if they exceed
/// the budget (the HW limit is the backstop — docs §2.2). Needs the `Packer`
/// for pool-specific footprint/feasibility, and the `CostModel` as the single
/// source of the objective (marginal benefit = score delta).
class CopySolver {
public:
  virtual ~CopySolver() = default;
  virtual CopyMap solve(const BufferModel &, const Packer &,
                        const Plan &grouping, const Budget &,
                        const CostModel &) const = 0;
};

/// Concave-knapsack copy allocator (docs §2.3, Step 4).
std::unique_ptr<CopySolver> createGreedyCopySolver();

/// SMEM packer (docs §5.3, Step 3): each block is sized `max(member bytes) *
/// copies`. NOTE: `legalJoin` currently returns false unconditionally -- the
/// search does NOT circular-reuse-group SMEM buffers. Matching encoding + basic
/// block is not sufficient for safe circular reuse (two concurrently-live
/// operands satisfy it yet must not share one circular buffer), so every SMEM
/// buffer gets its own block/copy count, which is always safe. Real reuse
/// grouping (the rotating-entry condition, not just encoding + scope) is future
/// work -- see docs §8 and the SmemPacker::legalJoin comment. Holds a reference
/// to `model`, which must outlive the returned object.
std::unique_ptr<Packer> createSmemPacker(const BufferModel &model);

/// TMEM packer (docs §5.4, Step 7): blocks are row-owners whose members
/// time-multiplex the same rows x cols space. A join is legal when encodings
/// match, the candidate's liveness is disjoint from every member's, AND the
/// candidate is bidirectionally data-dependent with them
/// (BufferModel::dependsOn either way) -- mirroring hasPotentialReuse's proven
/// predicate. The dependency is load-bearing: two liveness-disjoint but
/// *independent* buffers can live in different warp partitions and be
/// concurrent at run time despite disjoint op-id intervals, so encoding +
/// liveness alone is unsafe. Footprint is `max(member rows) x max(member
/// cols)`; feasibility is total rows <= 512 and per-owner cols <= 512. Copies
/// stay 1 (non-accumulator TMEM multi-copy is not yet legal — the caller pins
/// them). Holds a reference to `model`, which must outlive it.
std::unique_ptr<Packer> createTmemPacker(const BufferModel &model);

/// Latency-hiding cost model (docs §4). `II` is the loop initiation interval
/// (`tt.modulo_ii`); `lambda` weights the occupancy penalty (default 0 = pure
/// latency hiding, docs §8 open item 4). Holds a reference to `model`, which
/// must outlive the returned object.
std::unique_ptr<CostModel> createLatencyCostModel(const BufferModel &model,
                                                  double II,
                                                  double lambda = 0.0);

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Beam search over grouping+placement, with the copy dimension solved in
/// closed form per grouping (docs §3.3). Returns the K best complete plans.
///   W = beam width (partials retained per level)
///   K = number of top plans to return
TopKPlans beamSearch(const BufferModel &model, const OrderingPolicy &ordering,
                     const Packer &packer, const CostModel &cost,
                     const CopySolver &copies, const Budget &budget, unsigned W,
                     unsigned K);

} // namespace wsplan
} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WSMEMORYPLANSEARCH_H_
