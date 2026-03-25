#include "triton/Analysis/BarrierAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "barrier-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

// ===----------------------------------------------------------------------===
// Helpers
// ===----------------------------------------------------------------------===

// Matches getOutermostNameFromLoc in WarpSpecialization/Utility.h.
// TODO: Deduplicate once that utility moves to a public header.
static std::string getNameFromLoc(Location loc) {
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc))
    return getNameFromLoc(callSiteLoc.getCallee());
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return "";
}

/// Get the SSA name of a value (e.g., "%0", "%v_31").
static std::string getSSAName(Value v) {
  std::string name;
  llvm::raw_string_ostream os(name);
  v.printAsOperand(os, OpPrintingFlags().assumeVerified());
  return name;
}

/// Get the name for an alloc op: prefer NameLoc, fall back to SSA name.
static std::string getAllocName(Operation *op) {
  std::string name = getNameFromLoc(op->getLoc());
  if (!name.empty())
    return name;
  if (op->getNumResults() > 0)
    return getSSAName(op->getResult(0));
  return "<anon>";
}

/// Check if a value is used by ttng.init_barrier (indicating it's a barrier).
static bool isUsedByInitBarrier(Value v) {
  for (Operation *user : v.getUsers()) {
    // init_barrier operates on memdesc_index results, so check both the
    // value directly and its memdesc_index results.
    if (isa<ttng::InitBarrierOp>(user))
      return true;
    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
      for (Operation *indexUser : indexOp->getResult(0).getUsers()) {
        if (isa<ttng::InitBarrierOp>(indexUser))
          return true;
      }
    }
  }
  return false;
}

/// Check if a local_alloc produces a data buffer (non-barrier smem alloc).
static bool isDataBufferAlloc(Operation *op) {
  auto allocOp = dyn_cast<ttg::LocalAllocOp>(op);
  if (!allocOp)
    return false;
  auto mdType = dyn_cast<ttg::MemDescType>(allocOp.getType());
  if (!mdType)
    return false;
  return !mdType.getElementType().isInteger(64);
}

/// Check if a tmem_alloc produces a tensor memory buffer.
static bool isTmemAlloc(Operation *op) {
  return isa<ttng::TMEMAllocOp>(op);
}

/// Get the allocation.offset attribute from an op, or -1 if not set.
static int64_t getAllocOffset(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("allocation.offset"))
    return attr.getInt();
  return -1;
}

// Matches getAsyncTaskIds in WarpSpecialization/Utility.h but returns
// just the first id (or -1).  Also checks the kPartitionAttrName fallback.
// TODO: Deduplicate once getAsyncTaskIds moves to a public header.
static int getTaskId(Operation *op) {
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    if (!attr.empty())
      return attr[0];
  }
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kPartitionAttrName)) {
    if (!attr.empty())
      return attr[0];
  }
  return -1;
}

/// Get the index from a memdesc_index op, or -1 if not a constant.
static int64_t getMemDescIndex(ttg::MemDescIndexOp indexOp) {
  Value idxVal = indexOp.getIndex();
  if (auto cst = idxVal.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = idxVal.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return -1;
}

// ===----------------------------------------------------------------------===
// Step 1: Classify warp_specialize arguments into barriers and memory
// ===----------------------------------------------------------------------===
//
// Traverse all arguments of ttg.warp_specialize.  For each argument that is
// a ttg.local_alloc, check if it feeds ttng.init_barrier — those are barrier
// allocs.  Everything else (non-i64 local_alloc, tmem_alloc) is a memory
// alloc.  Build the initial BarrierObject and MemoryObject lists.

struct WarpSpecInfo {
  ttg::WarpSpecializeOp wsOp;
  // Map from warp_specialize operand index → BarrierObject* (barrier args)
  DenseMap<unsigned, BarrierObject *> argIdxToBarrier;
  // Map from warp_specialize operand index → MemoryObject* (memory args)
  DenseMap<unsigned, MemoryObject *> argIdxToMemory;
};

static void classifyWarpSpecArgs(ttg::WarpSpecializeOp wsOp,
                                 BarrierGraph &result,
                                 WarpSpecInfo &info) {
  info.wsOp = wsOp;
  unsigned nextBarrierId = result.barrierObjects.size();
  unsigned nextMemoryId = result.memoryObjects.size();

  auto captures = wsOp.getExplicitCaptures();
  for (unsigned i = 0; i < captures.size(); ++i) {
    Value capture = captures[i];
    Operation *defOp = capture.getDefiningOp();
    if (!defOp)
      continue;

    // Check if this is a barrier alloc: local_alloc with i64 element type
    // whose value is used by init_barrier.
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(defOp)) {
      auto mdType = dyn_cast<ttg::MemDescType>(allocOp.getType());
      if (!mdType)
        continue;

      if (mdType.getElementType().isInteger(64) &&
          isUsedByInitBarrier(capture)) {
        // This is a barrier alloc.
        auto obj = std::make_unique<BarrierObject>();
        obj->id = nextBarrierId++;
        obj->kind = BarrierObject::MBarrier;
        obj->allocOp = defOp;
        obj->name = getAllocName(defOp);
        obj->offset = getAllocOffset(defOp);

        LDBG("Step1: barrier arg[" << i << "] name=" << obj->name
                                    << " offset=" << obj->offset);

        info.argIdxToBarrier[i] = obj.get();
        result.barrierObjects.push_back(std::move(obj));
      } else if (!mdType.getElementType().isInteger(64)) {
        // This is a data buffer in SMEM.
        auto obj = std::make_unique<MemoryObject>();
        obj->id = nextMemoryId++;
        obj->memoryKind = MemoryObject::SMEM;
        obj->allocOp = defOp;
        obj->name = getAllocName(defOp);
        obj->offset = getAllocOffset(defOp);

        LDBG("Step1: memory arg[" << i << "] name=" << obj->name
                                   << " kind=SMEM offset=" << obj->offset);

        info.argIdxToMemory[i] = obj.get();
        result.memoryObjects.push_back(std::move(obj));
      }
    } else if (isTmemAlloc(defOp)) {
      // This is a data buffer in TMEM.
      auto obj = std::make_unique<MemoryObject>();
      obj->id = nextMemoryId++;
      obj->memoryKind = MemoryObject::TMEM;
      obj->allocOp = defOp;
      obj->name = getAllocName(defOp);
      obj->offset = getAllocOffset(defOp);

      LDBG("Step1: memory arg[" << i << "] name=" << obj->name
                                 << " kind=TMEM offset=" << obj->offset);

      info.argIdxToMemory[i] = obj.get();
      result.memoryObjects.push_back(std::move(obj));
    }
  }
}

// ===----------------------------------------------------------------------===
// Step 2: Traverse partitions to collect usages
// ===----------------------------------------------------------------------===
//
// For each partition in ttg.warp_specialize, the block arguments map 1:1 to
// the warp_specialize's explicit captures.  Walk each partition looking for
// ttg.memdesc_index ops that index into our collected barrier/memory args.
//
// For barriers: record BarrierObjectUsage with tentative roles —
//   wait → ProducerAcquire (tentative), arrive → ProducerCommit (tentative).
//   Roles will be refined when we build BarrierPairs later.
//
// For memory: record MemoryObjectUsage —
//   writes (local_store, tc_gen5_mma, tmem_store, etc.) → Producer,
//   reads (local_load, tmem_load, etc.) → Consumer.

/// Check if an operation writes to a memdesc (producer of data).
static bool isMemoryWrite(Operation *op) {
  return isa<ttg::LocalStoreOp>(op) ||
         isa<ttng::TCGen5MMAOp>(op) ||
         isa<ttng::TMEMStoreOp>(op) ||
         isa<ttng::TCGen5CommitOp>(op) ||
         isa<ttng::BarrierExpectOp>(op);
}

/// Check if an operation reads from a memdesc (consumer of data).
static bool isMemoryRead(Operation *op) {
  return isa<ttg::LocalLoadOp>(op) ||
         isa<ttng::TMEMLoadOp>(op);
}

/// Determine the partition id for a region.  The default region is partition 0,
/// and the i-th partition region is partition (i+1).
static int getPartitionId(ttg::WarpSpecializeOp wsOp, Region *region) {
  if (region == &wsOp.getDefaultRegion())
    return 0;
  auto partitions = wsOp.getPartitionRegions();
  for (auto [idx, partRegion] : llvm::enumerate(partitions)) {
    if (&*partRegion == region)
      return idx + 1;
  }
  return -1;
}

static void collectUsagesInRegion(Region &region, int partitionId,
                                  const DenseMap<unsigned, BarrierObject *> &argIdxToBarrier,
                                  const DenseMap<unsigned, MemoryObject *> &argIdxToMemory,
                                  bool isDefaultRegion,
                                  ValueRange wsCaptures) {
  // Build a map from block argument (or captured value in default region)
  // to the warp_specialize operand index.
  DenseMap<Value, unsigned> valueToArgIdx;

  if (isDefaultRegion) {
    // Default region implicitly captures — the values used inside are the
    // same SSA values as the warp_specialize operands.
    for (unsigned i = 0; i < wsCaptures.size(); ++i)
      valueToArgIdx[wsCaptures[i]] = i;
  } else {
    // Partition regions have block arguments mapping 1:1 to captures.
    Block &block = region.front();
    for (unsigned i = 0; i < block.getNumArguments() && i < wsCaptures.size();
         ++i)
      valueToArgIdx[block.getArgument(i)] = i;
  }

  // Walk all memdesc_index ops in this region.
  region.walk([&](ttg::MemDescIndexOp indexOp) {
    Value src = indexOp.getSrc();
    auto it = valueToArgIdx.find(src);
    if (it == valueToArgIdx.end())
      return; // Not indexing into a warp_specialize argument.

    unsigned argIdx = it->second;
    int64_t slotIndex = getMemDescIndex(indexOp);

    // Check if this indexes a barrier arg.
    auto barIt = argIdxToBarrier.find(argIdx);
    if (barIt != argIdxToBarrier.end()) {
      BarrierObject *barrier = barIt->second;
      // Determine role from downstream usage of the indexed barrier.
      for (Operation *user : indexOp->getResult(0).getUsers()) {
        BarrierObjectRole role;
        if (isa<ttng::WaitBarrierOp>(user)) {
          // Tentative: will be refined to ProducerAcquire or ConsumerWait
          // when we build BarrierPairs.
          role = BarrierObjectRole::ProducerAcquire;
        } else if (isa<ttng::ArriveBarrierOp>(user)) {
          // Tentative: will be refined to ProducerCommit or ConsumerRelease.
          role = BarrierObjectRole::ProducerCommit;
        } else if (isa<ttng::BarrierExpectOp>(user) ||
                   isa<ttng::TCGen5CommitOp>(user)) {
          role = BarrierObjectRole::ProducerCommit;
        } else if (isa<ttng::InitBarrierOp>(user)) {
          continue; // Skip init ops.
        } else {
          continue; // Unknown barrier user.
        }
        barrier->usages.push_back({role, partitionId, indexOp});

        LDBG("Step2: barrier usage name=" << barrier->name
              << " partition=" << partitionId
              << " role=" << (role == BarrierObjectRole::ProducerAcquire
                                  ? "wait" : "arrive")
              << " slot=" << slotIndex);
      }
      return;
    }

    // Check if this indexes a memory arg.
    auto memIt = argIdxToMemory.find(argIdx);
    if (memIt != argIdxToMemory.end()) {
      MemoryObject *memObj = memIt->second;
      // Determine role from downstream usage.
      for (Operation *user : indexOp->getResult(0).getUsers()) {
        MemoryObjectRole role;
        if (isMemoryWrite(user)) {
          role = MemoryObjectRole::Producer;
        } else if (isMemoryRead(user)) {
          role = MemoryObjectRole::Consumer;
        } else {
          continue;
        }
        memObj->usages.push_back({role, partitionId, indexOp});

        LDBG("Step2: memory usage name=" << memObj->name
              << " partition=" << partitionId
              << " role=" << (role == MemoryObjectRole::Producer
                                  ? "producer" : "consumer")
              << " slot=" << slotIndex);
      }

      // Also record the index as the offset if not yet set.
      if (memObj->offset == -1 && slotIndex >= 0)
        memObj->offset = slotIndex;
    }
  });
}

static void collectUsages(ttg::WarpSpecializeOp wsOp,
                           const WarpSpecInfo &info) {
  auto captures = wsOp.getExplicitCaptures();

  // Default region (partition 0) — uses implicit captures.
  collectUsagesInRegion(wsOp.getDefaultRegion(), /*partitionId=*/0,
                        info.argIdxToBarrier, info.argIdxToMemory,
                        /*isDefaultRegion=*/true, captures);

  // Partition regions (partition 1, 2, ...).
  for (auto [idx, partRegion] : llvm::enumerate(wsOp.getPartitionRegions())) {
    collectUsagesInRegion(*partRegion, /*partitionId=*/idx + 1,
                          info.argIdxToBarrier, info.argIdxToMemory,
                          /*isDefaultRegion=*/false, captures);
  }
}

// ===----------------------------------------------------------------------===
// Entry point
// ===----------------------------------------------------------------------===

BarrierGraph buildBarrierAnalysis(FuncOp funcOp) {
  BarrierGraph result;

  // Find all warp_specialize ops in the function.
  funcOp->walk([&](ttg::WarpSpecializeOp wsOp) {
    // Step 1: Classify warp_specialize arguments.
    WarpSpecInfo info;
    classifyWarpSpecArgs(wsOp, result, info);

    // Step 2: Traverse partitions to collect barrier and memory usages.
    collectUsages(wsOp, info);
  });

  return result;
}

// ===----------------------------------------------------------------------===
// Table dump
// ===----------------------------------------------------------------------===

static const char *roleToStr(BarrierObjectRole role) {
  switch (role) {
  case BarrierObjectRole::ProducerAcquire: return "ProdAcq";
  case BarrierObjectRole::ProducerCommit:  return "ProdCmt";
  case BarrierObjectRole::ConsumerWait:    return "ConsWait";
  case BarrierObjectRole::ConsumerRelease: return "ConsRel";
  }
  return "?";
}

static const char *memRoleToStr(MemoryObjectRole role) {
  switch (role) {
  case MemoryObjectRole::Producer: return "Producer";
  case MemoryObjectRole::Consumer: return "Consumer";
  }
  return "?";
}

void dumpBarrierTable(const BarrierGraph &result, llvm::raw_ostream &os) {
  os << "\n";
  os << "=== Barrier Analysis ===\n\n";

  // Table 1: Barrier Objects
  os << "--- Barrier Objects ---\n";
  os << "ID   Name                 Kind   Offset\n";
  os << "-------------------------------------------\n";
  for (auto &alloc : result.barrierObjects) {
    const char *name = alloc->name.empty() ? "<anon>" : alloc->name.c_str();
    const char *kind =
        alloc->kind == BarrierObject::MBarrier ? "mbar" : "named";
    os << llvm::format("%-4u %-20s %-6s %-8ld\n", alloc->id, name, kind,
                       alloc->offset);
  }

  // Table 2: Barrier Pairs (placeholder — will be populated in later steps)
  os << "\n--- Barrier Pairs ---\n";
  os << "ID   Full(ID)   Empty(ID)  #Mem\n";
  os << "-------------------------------------\n";
  for (auto &pair : result.barrierPairs) {
    unsigned fullId = pair->fullBarrier ? pair->fullBarrier->id : 0;
    unsigned emptyId = pair->emptyBarrier ? pair->emptyBarrier->id : 0;
    os << llvm::format("%-4u %-10u %-10u %-8lu\n", pair->id, fullId, emptyId,
                       (unsigned long)pair->memoryObjects.size());
  }

  // Table 3: Memory Objects
  os << "\n--- Memory Objects ---\n";
  os << "ID   Name             Kind   Offset   BarPair\n";
  os << "------------------------------------------------\n";
  for (auto &mem : result.memoryObjects) {
    const char *name = mem->name.empty() ? "<anon>" : mem->name.c_str();
    const char *kind =
        mem->memoryKind == MemoryObject::SMEM ? "SMEM" : "TMEM";
    unsigned bpId = mem->barrierPair ? mem->barrierPair->id : 0;
    os << llvm::format("%-4u %-16s %-6s %-8ld %-8u\n", mem->id, name, kind,
                       mem->offset, bpId);
  }

  // Table 4: Barrier Usages
  os << "\n--- Barrier Usages ---\n";
  os << "Barrier              ID   Role       Task\n";
  os << "--------------------------------------------\n";
  for (auto &alloc : result.barrierObjects) {
    const char *name = alloc->name.empty() ? "<anon>" : alloc->name.c_str();
    for (auto &u : alloc->usages) {
      os << llvm::format("%-20s %-4u %-10s %-6d\n", name, alloc->id,
                         roleToStr(u.role), u.taskId);
    }
  }

  // Table 5: Memory Usages
  os << "\n--- Memory Usages ---\n";
  os << "Memory               ID   Role       Task\n";
  os << "--------------------------------------------\n";
  for (auto &mem : result.memoryObjects) {
    const char *name = mem->name.empty() ? "<anon>" : mem->name.c_str();
    for (auto &u : mem->usages) {
      os << llvm::format("%-20s %-4u %-10s %-6d\n", name, mem->id,
                         memRoleToStr(u.role), u.taskId);
    }
  }

  os << "\n=== End Barrier Analysis ===\n";
}

} // namespace mlir::triton
