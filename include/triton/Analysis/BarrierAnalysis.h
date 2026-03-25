#ifndef TRITON_ANALYSIS_BARRIER_ANALYSIS_H
#define TRITON_ANALYSIS_BARRIER_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

class FuncOp;

// ===----------------------------------------------------------------------===
// Barrier objects
// ===----------------------------------------------------------------------===

/// The semantic role of a barrier usage in the producer-consumer protocol.
///
/// AutoWS token lowering converts each `nvws.create_token` into a *full* and
/// an *empty* mbarrier array.  The four token ops map to barrier operations
/// on these two arrays:
///
///   ProducerAcquire  – wait   on the *empty* barrier (buffer is free to fill)
///   ProducerCommit   – arrive on the *full*  barrier (data is ready)
///   ConsumerWait     – wait   on the *full*  barrier (data is available)
///   ConsumerRelease  – arrive on the *empty* barrier (buffer is consumed)
enum class BarrierObjectRole {
  ProducerAcquire,
  ProducerCommit,
  ConsumerWait,
  ConsumerRelease,
};

/// A single use-site of a barrier (one wait, arrive, expect, or commit op).
struct BarrierObjectUsage {
  BarrierObjectRole role;
  int taskId;    // warp-specialized partition id (-1 if unknown)
  Operation *op; // the MLIR op that uses the barrier
};

/// One barrier allocation – a `ttg.local_alloc` producing an i64 memdesc
/// (mbarrier array in shared memory) or a hardware named barrier.
struct BarrierObject {
  unsigned id = 0;
  enum Kind { MBarrier, Named };
  Kind kind = MBarrier;

  Operation *allocOp = nullptr; // the alloc op that creates this barrier
  std::string name;             // human-readable name from NameLoc
  int64_t offset = -1;          // byte offset in SMEM (-1 = unallocated)

  SmallVector<BarrierObjectUsage> usages; // all use-sites
};

/// A full/empty barrier pair that together synchronize one producer-consumer
/// data transfer.  Originates from a single `nvws.create_token` which lowers
/// to two `ttg.local_alloc` ops (one full, one empty).
struct BarrierPair {
  unsigned id = 0;
  BarrierObject *fullBarrier = nullptr;  // ProducerCommit / ConsumerWait
  BarrierObject *emptyBarrier = nullptr; // ProducerAcquire / ConsumerRelease

  SmallVector<struct MemoryObject *> memoryObjects; // data buffers guarded by this pair
};

// ===----------------------------------------------------------------------===
// Memory objects
// ===----------------------------------------------------------------------===

/// Whether a partition writes to (Producer) or reads from (Consumer) a buffer.
enum class MemoryObjectRole {
  Producer,
  Consumer,
};

/// A single use-site of a memory object by a warp-specialized partition.
struct MemoryObjectUsage {
  MemoryObjectRole role;
  int taskId;    // warp-specialized partition id (-1 if unknown)
  Operation *op; // the MLIR op that accesses the buffer
};

/// A shared-memory or tensor-memory buffer that carries data between producer
/// and consumer partitions, guarded by a BarrierPair.
struct MemoryObject {
  unsigned id = 0;
  enum MemoryKind { SMEM, TMEM };
  MemoryKind memoryKind = SMEM;

  Operation *allocOp = nullptr; // ttg.local_alloc (SMEM) or ttng.tmem_alloc (TMEM)
  std::string name;             // human-readable name from NameLoc
  int64_t offset = -1;          // byte offset from allocation.offset attribute

  SmallVector<MemoryObjectUsage> usages; // all use-sites

  BarrierPair *barrierPair = nullptr; // the barrier pair guarding this buffer
};

// ===----------------------------------------------------------------------===
// Barrier Graph
// ===----------------------------------------------------------------------===

/// Complete barrier-and-memory graph for one `tt.func`.
///
/// Owns all BarrierObject, BarrierPair, and MemoryObject instances via
/// unique_ptr.  Raw pointers in BarrierPair::memoryObjects,
/// MemoryObject::barrierPair, etc. are non-owning back-references valid
/// for the lifetime of this struct.
struct BarrierGraph {
  SmallVector<std::unique_ptr<BarrierObject>> barrierObjects;
  SmallVector<std::unique_ptr<BarrierPair>> barrierPairs;
  SmallVector<std::unique_ptr<MemoryObject>> memoryObjects;
};

// ===----------------------------------------------------------------------===
// Entry points
// ===----------------------------------------------------------------------===

/// Build the barrier/memory graph for a single function.
BarrierGraph buildBarrierAnalysis(FuncOp funcOp);

/// Dump the graph as human-readable tables to `os`.
void dumpBarrierTable(const BarrierGraph &result, llvm::raw_ostream &os);

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BARRIER_ANALYSIS_H
