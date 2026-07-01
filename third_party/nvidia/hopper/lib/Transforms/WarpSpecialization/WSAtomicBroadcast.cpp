//===- WSAtomicBroadcast.cpp - Cross-partition run-once atomic support ----===//
//
// Support for a side-effecting, loop-carried `tt.atomic_rmw` that appears in a
// dynamic (work-stealing) persistent kernel's `scf.while` loop, e.g.
//
//     tile_id = start_pid
//     while tile_id < num_tiles:
//         ... compute tile ...
//         tile_id = tl.atomic_add(tile_counter, 1)   // claim next tile
//
// Under AutoWS the persistent `scf.while` is cloned once per partition. Cloning
// is correct for the *pure* static update (`tile_id += NUM_SMS`) but wrong for
// the *side-effecting, non-deterministic* atomic: task-id propagation assigns
// the atomic (whose result is the loop-carried value used by every partition)
// to *all* partitions, so each warp group bumps the counter independently, the
// partitions diverge onto different tiles, and their producer/consumer barriers
// never match -> runtime deadlock.
//
// This pass runs immediately after `doTaskIdPropagate`. For each
// `tt.atomic_rmw` it classifies the op into one of three cases (see the design
// doc `docs/CrossPartitionAtomicSupport.md`):
//
//   1. Single-partition atomic  -> pass through unchanged (identity).
//   2. All-partition, scalar, loop-carried atomic (the dynamic tile counter)
//      -> transform: execute the atomic once in the producer/load partition and
//      broadcast the scalar result to every partition through an SMEM slot
//      guarded by a full/empty mbarrier handshake, so all partitions' while
//      conditions agree on the same tile id.
//   3. Anything else -> gracefully reject warp specialization (leave the kernel
//      unspecialized-but-compilable, never crash).
//
//===----------------------------------------------------------------------===//

#include "CodePartitionUtility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-atomic-broadcast"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

enum class AtomicWSCase {
  PassThrough, // case 1: mapped to a single partition
  Transform,   // case 2: all-partition, scalar, loop-carried counter
  Reject       // case 3: unsupported shape -> bail out of WS
};

// The full set of partitions the kernel is being specialized into. Derived from
// the union of task ids on the enclosing persistent loop.
static SmallVector<AsyncTaskId> getAllPartitions(scf::WhileOp whileOp) {
  return getAsyncTaskIds(whileOp.getOperation());
}

// Return the enclosing persistent `scf.while` whose after-region terminator
// (`scf.yield`) directly forwards `result` as a loop-carried value, or nullptr.
// This is what makes the atomic result "loop-carried": its value drives the
// next iteration's `scf.condition` in every partition.
static scf::WhileOp getLoopCarryingWhile(Value result) {
  for (OpOperand &use : result.getUses()) {
    auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
    if (!yield)
      continue;
    if (auto whileOp = dyn_cast<scf::WhileOp>(yield->getParentOp()))
      return whileOp;
  }
  return nullptr;
}

// A scalar atomic has neither a tensor result nor a tensor (scatter) address.
static bool isScalarAtomic(tt::AtomicRMWOp atomicOp) {
  if (isa<RankedTensorType>(atomicOp.getResult().getType()))
    return false;
  if (isa<RankedTensorType>(atomicOp.getPtr().getType()))
    return false;
  return true;
}

static AtomicWSCase classifyAtomic(tt::AtomicRMWOp atomicOp,
                                   scf::WhileOp &carryingWhileOut) {
  SmallVector<AsyncTaskId> taskIds = getAsyncTaskIds(atomicOp.getOperation());

  // Case 1: mapped to exactly one partition -> no cross-partition concern.
  if (taskIds.size() <= 1)
    return AtomicWSCase::PassThrough;

  // Case 2 requires a scalar atomic whose result is the loop-carried value of
  // an enclosing persistent `scf.while`, replicated to *every* partition.
  if (!isScalarAtomic(atomicOp)) {
    LDBG("reject: non-scalar / scatter atomic replicated across partitions");
    return AtomicWSCase::Reject;
  }
  scf::WhileOp whileOp = getLoopCarryingWhile(atomicOp.getResult());
  if (!whileOp) {
    LDBG("reject: replicated scalar atomic is not while-loop-carried");
    return AtomicWSCase::Reject;
  }
  SmallVector<AsyncTaskId> allParts = getAllPartitions(whileOp);
  if (taskIds.size() != allParts.size()) {
    LDBG("reject: atomic replicated to a strict subset of partitions");
    return AtomicWSCase::Reject;
  }
  carryingWhileOut = whileOp;
  return AtomicWSCase::Transform;
}

// Owner of the run-once atomic = the producer / TMA-load partition already
// assigned by PartitionSchedulingMeta. We locate it by the partition that owns
// the loop's TMA loads. Falls back to the smallest partition id.
static AsyncTaskId getOwnerPartition(scf::WhileOp whileOp,
                                     ArrayRef<AsyncTaskId> allParts) {
  AsyncTaskId owner = allParts.front();
  bool found = false;
  whileOp.getAfterBody()->walk([&](Operation *op) {
    if (found)
      return;
    if (isa<tt::DescriptorLoadOp, ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      auto ids = getAsyncTaskIds(op);
      if (ids.size() == 1) {
        owner = ids[0];
        found = true;
      }
    }
  });
  return owner;
}

// Transform a case-2 atomic (see file header). `depth` is the tile-prefetch
// depth; v1 only supports depth == 1.
//
// Rather than hand-synthesizing barriers, we install only the *data path*: run
// the atomic once in the owner partition, splat its scalar result into a 1-CTA
// SMEM slot, and read it back (+ tt.unsplat) in every partition, rewiring the
// while's loop-carried tile id to the broadcast value. The producer→consumer
// synchronization is left to `doCodePartitionPost`, which already turns a
// cross-partition `local_store`→`local_load` into (N) SMEM channels with the
// correct full/empty mbarriers and phase — i.e. the broadcast is expressed in
// terms of an existing, tested AutoWS channel rather than a bespoke handshake.
static LogicalResult transformAtomic(triton::FuncOp funcOp,
                                     tt::AtomicRMWOp atomicOp,
                                     scf::WhileOp whileOp, int depth) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<AsyncTaskId> allParts = getAllPartitions(whileOp);
  AsyncTaskId owner = getOwnerPartition(whileOp, allParts);
  Location loc = atomicOp.getLoc();

  // The atomic result must be the loop-carried yield operand we rewrite.
  Value atomicRes = atomicOp.getResult();
  auto yieldOp = whileOp.getYieldOp();
  int yieldIdx = -1;
  for (auto [i, v] : llvm::enumerate(yieldOp.getOperands()))
    if (v == atomicRes)
      yieldIdx = i;
  if (yieldIdx < 0)
    return failure();

  // Function-scope SMEM slot holding the claimed scalar (becomes a WS capture).
  // The per-slot shape is a single element; the channel machinery prepends the
  // buffer count (numBuffers) when it multi-buffers this channel. `depth` will
  // drive that buffer count once the channel is registered as multi-buffered.
  (void)depth;
  OpBuilder fb(funcOp);
  fb.setInsertionPointToStart(&funcOp.getBody().front());
  Attribute smem = ttg::SharedMemorySpaceAttr::get(ctx);
  auto numCTAs = ttg::lookupNumCTAs(funcOp);
  auto cga = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
  auto slotEnc = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cga);
  auto elemTy = atomicRes.getType();
  auto slotTy = ttg::MemDescType::get({1}, elemTy, slotEnc, smem,
                                      /*mutableMemory=*/true);
  Value slot = ttg::LocalAllocOp::create(
      fb, NameLoc::get(StringAttr::get(ctx, "atomic_bcast_slot"), loc), slotTy,
      Value());

  // Single-element tensor for the scalar round-trip: splat -> local_store ->
  // local_load -> tt.unsplat. The default blocked layout mirrors the
  // scalar-load pipelining pattern (LowerLoops.cpp). Single-element tensors are
  // excluded from partition register-budget classification
  // (OptimizePartitionWarps), so this does not reclassify an otherwise
  // non-tensor partition.
  int numWarps = ttg::lookupNumWarps(funcOp);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      funcOp->getParentOfType<ModuleOp>());
  auto bcastLayout = ttg::getDefaultBlockedEncoding(ctx, {1}, numWarps,
                                                    threadsPerWarp, numCTAs);
  auto bcastTy = RankedTensorType::get({1}, elemTy, bcastLayout);

  OpBuilderWithAsyncTaskIds b(ctx);

  // Owner: run the atomic once, then publish its result into the slot.
  setAsyncTaskIds(atomicOp, {owner});
  b.setAsynTaskIdsFromArray({owner});
  b.setInsertionPointAfter(atomicOp);
  Value splat = b.createWithAsyncTaskIds<tt::SplatOp>(loc, bcastTy, atomicRes)
                    .getResult();
  b.createWithAsyncTaskIds<ttg::LocalStoreOp>(loc, splat, slot);

  // Every partition: read the broadcast value back and unsplat to a scalar.
  b.setAsynTaskIdsFromArray(allParts);
  Value loaded =
      b.createWithAsyncTaskIds<ttg::LocalLoadOp>(loc, bcastTy, slot, Value())
          .getResult();
  Value bcastScalar =
      b.createWithAsyncTaskIds<tt::UnsplatOp>(loc, elemTy, loaded).getResult();

  // Rewire the loop-carried tile id to the broadcast value; the atomic now
  // feeds only the slot store, so it is cloned into the owner partition alone.
  yieldOp->setOperand(yieldIdx, bcastScalar);
  return success();
}

} // namespace

// Entry point: classify and (where applicable) transform every `tt.atomic_rmw`.
// Returns failure() when an atomic forces a graceful warp-specialization
// reject. The caller is responsible for stripping WS metadata (via
// removeWarpSpecializeAttr, which clears both the partition ids and the
// async_task_ids) so the kernel is left unspecialized-but-compilable; this
// keeps a single source of truth for the reject teardown shared with the other
// AutoWS bail-outs.
LogicalResult doAtomicBroadcast(triton::FuncOp funcOp, int tilePrefetchDepth) {
  SmallVector<tt::AtomicRMWOp> atomics;
  funcOp.walk([&](tt::AtomicRMWOp op) { atomics.push_back(op); });

  for (auto atomicOp : atomics) {
    scf::WhileOp whileOp;
    AtomicWSCase kind = classifyAtomic(atomicOp, whileOp);
    switch (kind) {
    case AtomicWSCase::PassThrough:
      continue;
    case AtomicWSCase::Reject:
      LDBG("Warp specialization does not support this atomic shape. Skipping.");
      return failure();
    case AtomicWSCase::Transform:
      if (failed(
              transformAtomic(funcOp, atomicOp, whileOp, tilePrefetchDepth))) {
        LDBG("atomic broadcast transform failed; skipping WS.");
        return failure();
      }
      continue;
    }
  }
  return success();
}

} // namespace mlir
