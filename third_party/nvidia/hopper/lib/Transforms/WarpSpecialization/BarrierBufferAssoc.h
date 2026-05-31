#ifndef NV_DIALECT_HOPPER_TRANSFORMS_BARRIERBUFFERASSOC_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_BARRIERBUFFERASSOC_H_

// Milestone 1 of the Barrier Analysis & Verification subsystem
// (see barrier_analysis.md).
//
// Associates every WSBarrier endpoint with the set of physical buffer(s)
// (buffer.id, optional buffer.offset) it guards. The association is computed
// from the autoWS `Channel` structures (where the guarded buffer is still
// known), stamped as discardable dotted-key attributes -- mirroring the
// `buffer.id` / `buffer.offset` mechanism of the memory planner -- and
// forwarded through token lowering so it survives to the verifier stage.
//
// This module is analysis-only; it never mutates control flow or removes,
// merges, or moves barriers.

#include "CodePartitionUtility.h" // Channel, ReuseConfig (namespace mlir)
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

// Discardable attribute keys stamped on barrier endpoints / token ops. Dotted
// keys mirror the memory planner's `buffer.id` / `buffer.offset` convention so
// there is zero op-schema (TableGen) churn.
constexpr llvm::StringLiteral kBarrierGuardsAttr = "barrier.guards";
constexpr llvm::StringLiteral kBarrierOffsetsAttr = "barrier.offsets";

// A physical buffer (or TMEM column sub-range) guarded by a WSBarrier endpoint.
// `offset` is the `buffer.offset` column offset for TMEM sub-range reuse; it is
// 0 for whole-buffer guards.
struct GuardedBuffer {
  int id = -1;
  int offset = 0;

  bool operator==(const GuardedBuffer &o) const {
    return id == o.id && offset == o.offset;
  }
  bool operator<(const GuardedBuffer &o) const {
    return id != o.id ? id < o.id : offset < o.offset;
  }
};

// barrier endpoint op -> the set of physical buffers it guards.
using BarrierBufferMap =
    llvm::DenseMap<Operation *, llvm::SmallVector<GuardedBuffer, 2>>;

// autoWS frontend: compute the guarded buffers for `channel`, unioning the
// whole reuse group when `channel` participates in one. Pure derivation from
// `Channel::getAllocOp()` + the `buffer.id` / `buffer.offset` attributes that
// the memory planner already set.
llvm::SmallVector<GuardedBuffer, 2> computeChannelGuards(Channel *channel,
                                                         ReuseConfig *config);

// Writer: stamp `barrier.guards` / `barrier.offsets` onto `op`. No-op when
// `guards` is empty. `op` is typically a token op (at emission) or a lowered
// ttng.wait/arrive_barrier (after forwarding through lowering).
void stampBarrierGuards(Operation *op, llvm::ArrayRef<GuardedBuffer> guards);

// Reader: read the stamped guards off `op` (mirrors `getBufferId`). Returns an
// empty vector when `op` carries no guards.
llvm::SmallVector<GuardedBuffer, 2> getBarrierGuards(Operation *op);

// Forward the guard attributes from `from` to `to`, mirroring the way
// `getConstraintsAttr()` is forwarded in WSLowerToken.cpp. No-op when `from`
// carries no guards.
void forwardBarrierGuards(Operation *from, Operation *to);

// True if `op` is a WSBarrier endpoint -- an NVWS token endpoint
// (producer_acquire / producer_commit / consumer_wait / consumer_release) or a
// lowered HW barrier (ttng.wait_barrier / ttng.arrive_barrier).
bool isBarrierEndpoint(Operation *op);

// Build the barrier->buffer map for `funcOp` by scanning every WSBarrier
// endpoint and reading its stamped guards. Endpoints with no guards are
// omitted from the map (use dumpAndVerifyBarrierBufferMap to flag them).
BarrierBufferMap buildBarrierBufferMap(triton::FuncOp funcOp);

// Debug dumper + self-consistency check. Prints `endpoint -> guarded buffers`
// for every WSBarrier endpoint and returns the number of *orphan* endpoints
// (WS barriers that resolve to zero buffers). Only WS barriers are counted as
// orphans: ttng barriers without WSBarrier constraints (e.g. TMA-store
// barriers) are ignored.
unsigned dumpAndVerifyBarrierBufferMap(triton::FuncOp funcOp,
                                       const BarrierBufferMap &map,
                                       llvm::raw_ostream &os);

// Gated debug entry point. When the TRITON_DUMP_BARRIER_GUARDS environment
// variable is set, build and print the barrier->buffer map for `funcOp` (and
// its orphan count) to llvm::errs(). No-op otherwise. Intended to be invoked
// from the pipeline right after token lowering.
void dumpBarrierGuardsIfEnabled(triton::FuncOp funcOp);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_BARRIERBUFFERASSOC_H_
