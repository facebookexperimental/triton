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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// depth: the number of buffer slots the broadcast channel is multi-buffered
// with. depth == 1 is the single-stage broadcast; depth > 1 lets the owner
// claim and publish up to `depth` tile ids ahead of the slower consumer
// partitions, so the persistent loop does not serialize on the tile-id handoff.
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
    if (v == atomicRes) {
      yieldIdx = i;
      break;
    }
  if (yieldIdx < 0)
    return failure();

  // Function-scope SMEM slot holding the claimed scalar (becomes a WS capture).
  // The per-slot shape is a single element; the channel machinery prepends the
  // buffer count (numBuffers) when it multi-buffers this channel. For depth > 1
  // we stamp the requested count on the alloc (kAtomicBroadcastCopiesAttrName)
  // so the memory planner pins this otherwise non-innermost broadcast channel
  // to `depth` buffers; the accumCnt already threaded through the persistent
  // scf.while then rotates the slot/phase across iterations.
  OpBuilder fb(funcOp);
  fb.setInsertionPointToStart(&funcOp.getBody().front());
  Attribute smem = ttg::SharedMemorySpaceAttr::get(ctx);
  auto numCTAs = ttg::lookupNumCTAs(funcOp);
  auto cga = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
  auto slotEnc = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cga);
  auto elemTy = atomicRes.getType();
  auto slotTy = ttg::MemDescType::get({1}, elemTy, slotEnc, smem,
                                      /*mutableMemory=*/true);
  auto slotOp = ttg::LocalAllocOp::create(
      fb, NameLoc::get(StringAttr::get(ctx, "atomic_bcast_slot"), loc), slotTy,
      Value());
  if (depth > 1)
    slotOp->setAttr(kAtomicBroadcastCopiesAttrName,
                    fb.getI32IntegerAttr(depth));
  Value slot = slotOp;

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

//===--------------------------------------------------------------------===//
// CLC tile-scheduler fetch broadcast (Stage 3 of the CLC lowering).
//
// A dynamic-persistent kernel driven by `tl.clc_tile_scheduler` fetches its
// next tile with the token pair `ttng.clc_try_cancel_async` (->
// !ttg.async.token) + `ttng.clc_read` (token -> {isValid, x, y, z}). Those
// decoded results are the loop-carried values of the persistent `scf.while` and
// are used by every partition (loop condition + tile compute), so task-id
// propagation replicates the read to *all* partitions -- which would make each
// warp group run its own try_cancel, claim a different pending cluster, and
// diverge -> deadlock. This is the exact analogue of the atomic tile-counter
// handled above, so we handle it the same way: run the fetch once in the owner
// (producer) partition and broadcast the decoded results to every partition
// through SMEM slots. The owner-local CLC completion barrier is materialized
// later by the `clc-materialize` pass (which runs after AutoWS).
//===--------------------------------------------------------------------===//

enum class CLCWSCase {
  PassThrough, // mapped to a single partition
  Transform,   // all-partition, loop-carried fetch
  Reject       // unsupported -> bail out of WS
};

static CLCWSCase classifyCLC(ttng::CLCReadOp readOp, scf::WhileOp &whileOut) {
  SmallVector<AsyncTaskId> taskIds = getAsyncTaskIds(readOp.getOperation());
  if (taskIds.size() <= 1)
    return CLCWSCase::PassThrough;

  auto whileOp = readOp->getParentOfType<scf::WhileOp>();
  if (!whileOp) {
    LDBG("reject: clc_read replicated across partitions but not in a while");
    return CLCWSCase::Reject;
  }
  if (!readOp.getToken().getDefiningOp<ttng::CLCTryCancelAsyncOp>()) {
    LDBG("reject: clc_read token is not from clc_try_cancel_async");
    return CLCWSCase::Reject;
  }
  SmallVector<AsyncTaskId> allParts = getAllPartitions(whileOp);
  if (taskIds.size() != allParts.size()) {
    LDBG("reject: clc_read replicated to a strict subset of partitions");
    return CLCWSCase::Reject;
  }
  whileOut = whileOp;
  return CLCWSCase::Transform;
}

// Broadcast a scalar produced in `owner` to every partition through a
// function-scope SMEM slot (splat -> local_store in owner; local_load ->
// unsplat in all partitions). Mirrors the atomic case; the store/load edge
// becomes SMEM channels in doCodePartitionPost. `b`'s insertion point must be
// after the scalar's producer.
static Value broadcastScalarThroughSmem(triton::FuncOp funcOp,
                                        OpBuilderWithAsyncTaskIds &b,
                                        AsyncTaskId owner,
                                        ArrayRef<AsyncTaskId> allParts,
                                        Value scalar, Location loc) {
  MLIRContext *ctx = funcOp.getContext();
  Type origTy = scalar.getType();

  // Sub-32-bit integers (e.g. the `is_valid` i1) don't have a well-formed SMEM
  // tensor layout for the round-trip; widen to i32 for transport and narrow
  // back on the far side.
  Type transportTy = origTy;
  bool widen = false;
  if (auto intTy = dyn_cast<IntegerType>(origTy);
      intTy && intTy.getWidth() < 32) {
    transportTy = b.getI32Type();
    widen = true;
  }

  OpBuilder fb(funcOp);
  fb.setInsertionPointToStart(&funcOp.getBody().front());
  Attribute smem = ttg::SharedMemorySpaceAttr::get(ctx);
  auto numCTAs = ttg::lookupNumCTAs(funcOp);
  auto cga = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
  auto slotEnc = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cga);
  auto slotTy = ttg::MemDescType::get({1}, transportTy, slotEnc, smem,
                                      /*mutableMemory=*/true);
  Value slot = ttg::LocalAllocOp::create(
      fb, NameLoc::get(StringAttr::get(ctx, "clc_bcast_slot"), loc), slotTy,
      Value());

  int numWarps = ttg::lookupNumWarps(funcOp);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      funcOp->getParentOfType<ModuleOp>());
  auto bcastLayout = ttg::getDefaultBlockedEncoding(ctx, {1}, numWarps,
                                                    threadsPerWarp, numCTAs);
  auto bcastTy = RankedTensorType::get({1}, transportTy, bcastLayout);

  b.setAsynTaskIdsFromArray({owner});
  Value toStore = scalar;
  if (widen)
    toStore = b.createWithAsyncTaskIds<arith::ExtUIOp>(loc, transportTy, scalar)
                  .getResult();
  Value splat =
      b.createWithAsyncTaskIds<tt::SplatOp>(loc, bcastTy, toStore).getResult();
  b.createWithAsyncTaskIds<ttg::LocalStoreOp>(loc, splat, slot);

  b.setAsynTaskIdsFromArray(allParts);
  Value loaded =
      b.createWithAsyncTaskIds<ttg::LocalLoadOp>(loc, bcastTy, slot, Value())
          .getResult();
  Value unsplat =
      b.createWithAsyncTaskIds<tt::UnsplatOp>(loc, transportTy, loaded)
          .getResult();
  if (widen)
    unsplat = b.createWithAsyncTaskIds<arith::TruncIOp>(loc, origTy, unsplat)
                  .getResult();
  return unsplat;
}

// Transform: run the CLC fetch once in the owner partition and broadcast each
// loop-carried decoded result to every partition.
static LogicalResult transformCLC(triton::FuncOp funcOp, ttng::CLCReadOp readOp,
                                  scf::WhileOp whileOp) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<AsyncTaskId> allParts = getAllPartitions(whileOp);
  AsyncTaskId owner = getOwnerPartition(whileOp, allParts);
  Location loc = readOp.getLoc();

  auto issue = readOp.getToken().getDefiningOp<ttng::CLCTryCancelAsyncOp>();
  if (!issue)
    return failure();

  auto yieldOp = whileOp.getYieldOp();

  // Run the fetch (issue + read) in the owner partition alone.
  setAsyncTaskIds(issue, {owner});
  setAsyncTaskIds(readOp, {owner});

  OpBuilderWithAsyncTaskIds b(ctx);
  b.setInsertionPointAfter(readOp);

  // Broadcast each loop-carried result; the read's results only feed the yield.
  for (Value res : readOp.getResults()) {
    int yieldIdx = -1;
    for (auto [i, v] : llvm::enumerate(yieldOp.getOperands()))
      if (v == res) {
        yieldIdx = i;
        break;
      }
    if (yieldIdx < 0)
      continue; // not loop-carried (e.g. an unused/DCE'd coordinate)
    Value bcast =
        broadcastScalarThroughSmem(funcOp, b, owner, allParts, res, loc);
    yieldOp->setOperand(yieldIdx, bcast);
  }
  return success();
}

} // namespace

// Entry point: handle every cross-partition, run-once, loop-carried "claim the
// next tile" producer in a dynamic-persistent kernel. Two kinds share the exact
// same run-once + SMEM-broadcast idea and are handled together here:
//   * the `tt.atomic_rmw` global tile counter, and
//   * the CLC tile-scheduler fetch (`ttng.clc_read` fed by
//     `ttng.clc_try_cancel_async`).
// Each producer is classified into pass-through (single partition), transform
// (run once in the owner/producer partition + broadcast the loop-carried
// result(s) to all partitions), or reject. Returns failure() to force a
// graceful warp-specialization reject; the caller strips WS metadata via
// removeWarpSpecializeAttr (which clears both partition ids and
// async_task_ids), leaving the kernel unspecialized-but-compilable — one source
// of truth for the reject teardown shared with the other AutoWS bail-outs.
LogicalResult doDynamicTileBroadcast(triton::FuncOp funcOp,
                                     int tilePrefetchDepth) {
  // Atomic global tile counter.
  SmallVector<tt::AtomicRMWOp> atomics;
  funcOp.walk([&](tt::AtomicRMWOp op) { atomics.push_back(op); });
  for (auto atomicOp : atomics) {
    scf::WhileOp whileOp;
    switch (classifyAtomic(atomicOp, whileOp)) {
    case AtomicWSCase::PassThrough:
      break;
    case AtomicWSCase::Reject:
      LDBG("Warp specialization does not support this atomic shape. Skipping.");
      return failure();
    case AtomicWSCase::Transform:
      if (failed(
              transformAtomic(funcOp, atomicOp, whileOp, tilePrefetchDepth))) {
        LDBG("atomic broadcast transform failed; skipping WS.");
        return failure();
      }
      break;
    }
  }

  // CLC tile-scheduler fetch.
  SmallVector<ttng::CLCReadOp> reads;
  funcOp.walk([&](ttng::CLCReadOp op) { reads.push_back(op); });
  for (auto readOp : reads) {
    scf::WhileOp whileOp;
    switch (classifyCLC(readOp, whileOp)) {
    case CLCWSCase::PassThrough:
      break;
    case CLCWSCase::Reject:
      LDBG("Warp specialization does not support this CLC fetch. Skipping.");
      return failure();
    case CLCWSCase::Transform:
      if (failed(transformCLC(funcOp, readOp, whileOp))) {
        LDBG("CLC broadcast transform failed; skipping WS.");
        return failure();
      }
      break;
    }
  }
  return success();
}

} // namespace mlir
