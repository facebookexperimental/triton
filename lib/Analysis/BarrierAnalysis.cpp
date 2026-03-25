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
static bool isTmemAlloc(Operation *op) { return isa<ttng::TMEMAllocOp>(op); }

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
  // Set of warp_specialize operand indices that are barrier allocs.
  DenseSet<unsigned> barrierArgIndices;
  // Map from (argIdx, slotIndex) → BarrierObject* for per-slot barriers.
  DenseMap<std::pair<unsigned, int64_t>, BarrierObject *> slotToBarrier;
  // Map from warp_specialize operand index → all per-slot BarrierObjects.
  DenseMap<unsigned, SmallVector<BarrierObject *>> argIdxToBarrierSlots;
  // Map from named barrier index → BarrierObject* for named barriers.
  DenseMap<int64_t, BarrierObject *> namedBarriers;
  // Map from warp_specialize operand index → MemoryObject* (memory args)
  DenseMap<unsigned, MemoryObject *> argIdxToMemory;
};

static void classifyWarpSpecArgs(ttg::WarpSpecializeOp wsOp,
                                 BarrierGraph &result, WarpSpecInfo &info) {
  info.wsOp = wsOp;
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
        // Scan memdesc_index → init_barrier chains to find per-slot indices.
        info.barrierArgIndices.insert(i);
        std::string allocName = getAllocName(defOp);

        for (Operation *user : capture.getUsers()) {
          auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user);
          if (!indexOp)
            continue;
          // Check if this memdesc_index feeds an init_barrier.
          bool feedsInit = false;
          for (Operation *indexUser : indexOp->getResult(0).getUsers()) {
            if (isa<ttng::InitBarrierOp>(indexUser)) {
              feedsInit = true;
              break;
            }
          }
          if (!feedsInit)
            continue;

          int64_t slot = getMemDescIndex(indexOp);
          auto key = std::make_pair(i, slot);
          if (info.slotToBarrier.count(key))
            continue; // Already created for this slot.

          auto obj = std::make_unique<BarrierObject>();
          obj->id = result.barrierObjects.size();
          obj->kind = BarrierObject::MBarrier;
          obj->allocOp = defOp;
          obj->name = allocName;
          obj->offset = slot;

          LDBG("Step1: barrier arg[" << i << "] name=" << allocName
                                     << " slot=" << slot);

          info.slotToBarrier[key] = obj.get();
          info.argIdxToBarrierSlots[i].push_back(obj.get());
          result.barrierObjects.push_back(std::move(obj));
        }
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
  return isa<ttg::LocalStoreOp>(op) || isa<ttng::TCGen5MMAOp>(op) ||
         isa<ttng::TMEMStoreOp>(op) || isa<ttng::TCGen5CommitOp>(op) ||
         isa<ttng::BarrierExpectOp>(op);
}

/// Check if an operation reads from a memdesc (consumer of data).
static bool isMemoryRead(Operation *op) {
  return isa<ttg::LocalLoadOp>(op) || isa<ttng::TMEMLoadOp>(op);
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
                                  WarpSpecInfo &info, BarrierGraph &result,
                                  bool isDefaultRegion, ValueRange wsCaptures) {
  // Build a map from block argument (or captured value in default region)
  // to the warp_specialize operand index.
  DenseMap<Value, unsigned> valueToArgIdx;

  // Map from pre-indexed barrier memdesc_index results to (argIdx, slotIndex).
  // Only needed for the default region where pre-computed indices are used
  // directly by barrier ops (e.g., TLX TTGIR).
  DenseMap<Value, std::pair<unsigned, int64_t>> preIndexedBarriers;

  if (isDefaultRegion) {
    // Default region implicitly captures — the values used inside are the
    // same SSA values as the warp_specialize operands.
    for (unsigned i = 0; i < wsCaptures.size(); ++i) {
      valueToArgIdx[wsCaptures[i]] = i;

      // For barrier allocs, also map their pre-indexed memdesc_index results
      // (defined before warp_specialize) so we can detect direct uses.
      if (info.barrierArgIndices.contains(i)) {
        for (Operation *user : wsCaptures[i].getUsers()) {
          if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
            // Only map indices defined outside the warp_specialize regions.
            if (!region.isAncestor(indexOp->getParentRegion())) {
              int64_t slot = getMemDescIndex(indexOp);
              preIndexedBarriers[indexOp->getResult(0)] = {i, slot};
            }
          }
        }
      }
    }
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
    if (info.barrierArgIndices.contains(argIdx)) {
      // Collect the roles from downstream usage of the indexed barrier.
      SmallVector<std::pair<BarrierObjectRole, Operation *>> roles;
      for (Operation *user : indexOp->getResult(0).getUsers()) {
        if (isa<ttng::WaitBarrierOp>(user)) {
          roles.push_back({BarrierObjectRole::ProducerAcquire, user});
        } else if (isa<ttng::ArriveBarrierOp>(user)) {
          roles.push_back({BarrierObjectRole::ProducerCommit, user});
        } else if (isa<ttng::BarrierExpectOp>(user) ||
                   isa<ttng::TCGen5CommitOp>(user) ||
                   isa<ttng::TCGen5MMAOp>(user) ||
                   isa<ttng::TCGen5MMAScaledOp>(user) ||
                   isa<ttng::AsyncTMACopyGlobalToLocalOp>(user)) {
          // tc_gen5_mma and async_tma_copy implicitly arrive on their barrier
          // operands.  barrier_expect and tc_gen5_commit are explicit arrivals.
          roles.push_back({BarrierObjectRole::ProducerCommit, user});
        }
        // Skip init_barrier and unknown ops.
      }
      if (roles.empty())
        return;

      // Determine which BarrierObjects to attach usages to.
      SmallVector<BarrierObject *> targets;
      if (slotIndex >= 0) {
        // Constant index → link to the specific per-slot BarrierObject.
        auto key = std::make_pair(argIdx, slotIndex);
        auto slotIt = info.slotToBarrier.find(key);
        if (slotIt != info.slotToBarrier.end())
          targets.push_back(slotIt->second);
      } else {
        // Dynamic index → link to all per-slot BarrierObjects for this alloc.
        auto slotsIt = info.argIdxToBarrierSlots.find(argIdx);
        if (slotsIt != info.argIdxToBarrierSlots.end())
          targets = slotsIt->second;
      }

      for (BarrierObject *barrier : targets) {
        for (auto &[role, user] : roles) {
          barrier->usages.push_back({role, partitionId, indexOp});
          LDBG("Step2: barrier usage name="
               << barrier->name << " partition=" << partitionId << " role="
               << (role == BarrierObjectRole::ProducerAcquire ? "wait"
                                                              : "arrive")
               << " slot=" << barrier->offset);
        }
      }
      return;
    }

    // Check if this indexes a memory arg.
    auto memIt = info.argIdxToMemory.find(argIdx);
    if (memIt != info.argIdxToMemory.end()) {
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

        LDBG("Step2: memory usage name="
             << memObj->name << " partition=" << partitionId << " role="
             << (role == MemoryObjectRole::Producer ? "producer" : "consumer")
             << " slot=" << slotIndex);
      }

      // Also record the index as the offset if not yet set.
      if (memObj->offset == -1 && slotIndex >= 0)
        memObj->offset = slotIndex;
    }
  });

  // Walk barrier ops that use pre-indexed values (default region only, e.g.,
  // TLX TTGIR where memdesc_index results from before warp_specialize are
  // used directly by wait_barrier/arrive_barrier in the default region).
  if (!preIndexedBarriers.empty()) {
    auto recordPreIndexedUsage = [&](Operation *op, Value barrierVal,
                                     BarrierObjectRole role) {
      auto it = preIndexedBarriers.find(barrierVal);
      if (it == preIndexedBarriers.end())
        return;
      auto [argIdx, slot] = it->second;
      // Use the memdesc_index op (which has a result with an SSA name)
      // instead of the barrier consumer op (which may have no results).
      Operation *nameOp = barrierVal.getDefiningOp();
      if (!nameOp)
        nameOp = op;
      SmallVector<BarrierObject *> targets;
      if (slot >= 0) {
        auto key = std::make_pair(argIdx, slot);
        auto slotIt = info.slotToBarrier.find(key);
        if (slotIt != info.slotToBarrier.end())
          targets.push_back(slotIt->second);
      } else {
        auto slotsIt = info.argIdxToBarrierSlots.find(argIdx);
        if (slotsIt != info.argIdxToBarrierSlots.end())
          targets = slotsIt->second;
      }
      for (BarrierObject *barrier : targets) {
        barrier->usages.push_back({role, partitionId, nameOp});
        LDBG("Step2: pre-indexed barrier usage name="
             << barrier->name << " partition=" << partitionId << " slot="
             << barrier->offset);
      }
    };

    region.walk([&](ttng::WaitBarrierOp op) {
      recordPreIndexedUsage(op, op.getAlloc(),
                            BarrierObjectRole::ProducerAcquire);
    });
    region.walk([&](ttng::ArriveBarrierOp op) {
      recordPreIndexedUsage(op, op.getAlloc(),
                            BarrierObjectRole::ProducerCommit);
    });
    region.walk([&](ttng::BarrierExpectOp op) {
      recordPreIndexedUsage(op, op.getAlloc(),
                            BarrierObjectRole::ProducerCommit);
    });
    region.walk([&](ttng::TCGen5CommitOp op) {
      recordPreIndexedUsage(op, op.getBarrier(),
                            BarrierObjectRole::ProducerCommit);
    });
  }

  // Walk named barrier ops (arrive_barrier_named / wait_barrier_named).
  auto getOrCreateNamedBarrier = [&](int64_t barIdx) -> BarrierObject * {
    auto it = info.namedBarriers.find(barIdx);
    if (it != info.namedBarriers.end())
      return it->second;
    auto obj = std::make_unique<BarrierObject>();
    obj->id = result.barrierObjects.size();
    obj->kind = BarrierObject::Named;
    obj->allocOp = nullptr;
    obj->name = "named_bar_" + std::to_string(barIdx);
    obj->offset = barIdx;
    auto *ptr = obj.get();
    info.namedBarriers[barIdx] = ptr;
    result.barrierObjects.push_back(std::move(obj));
    return ptr;
  };

  region.walk([&](ttng::NamedBarrierArriveOp arriveOp) {
    auto cst = arriveOp.getBar().getDefiningOp<arith::ConstantIntOp>();
    if (!cst)
      return;
    BarrierObject *barrier = getOrCreateNamedBarrier(cst.value());
    barrier->usages.push_back(
        {BarrierObjectRole::ProducerCommit, partitionId, arriveOp});
    LDBG("Step2: named barrier arrive idx=" << cst.value()
                                            << " partition=" << partitionId);
  });

  region.walk([&](ttng::NamedBarrierWaitOp waitOp) {
    auto cst = waitOp.getBar().getDefiningOp<arith::ConstantIntOp>();
    if (!cst)
      return;
    BarrierObject *barrier = getOrCreateNamedBarrier(cst.value());
    barrier->usages.push_back(
        {BarrierObjectRole::ProducerAcquire, partitionId, waitOp});
    LDBG("Step2: named barrier wait idx=" << cst.value()
                                          << " partition=" << partitionId);
  });
}

static void collectUsages(ttg::WarpSpecializeOp wsOp, WarpSpecInfo &info,
                          BarrierGraph &result) {
  auto captures = wsOp.getExplicitCaptures();

  // Default region (partition 0) — uses implicit captures.
  collectUsagesInRegion(wsOp.getDefaultRegion(), /*partitionId=*/0, info,
                        result,
                        /*isDefaultRegion=*/true, captures);

  // Partition regions (partition 1, 2, ...).
  for (auto [idx, partRegion] : llvm::enumerate(wsOp.getPartitionRegions())) {
    collectUsagesInRegion(*partRegion, /*partitionId=*/idx + 1, info, result,
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
    // Barrier objects are created per-slot here (not in Step 1).
    collectUsages(wsOp, info, result);
  });

  return result;
}

// ===----------------------------------------------------------------------===
// Table dump
// ===----------------------------------------------------------------------===

static const char *roleToStr(BarrierObjectRole role) {
  switch (role) {
  case BarrierObjectRole::ProducerAcquire:
    return "ProdAcq";
  case BarrierObjectRole::ProducerCommit:
    return "ProdCmt";
  case BarrierObjectRole::ConsumerWait:
    return "ConsWait";
  case BarrierObjectRole::ConsumerRelease:
    return "ConsRel";
  }
  return "?";
}

static const char *memRoleToStr(MemoryObjectRole role) {
  switch (role) {
  case MemoryObjectRole::Producer:
    return "Producer";
  case MemoryObjectRole::Consumer:
    return "Consumer";
  }
  return "?";
}

/// Get a human-readable name for an op: prefer NameLoc, fall back to SSA name.
static std::string getOpName(Operation *op) {
  std::string name = getNameFromLoc(op->getLoc());
  if (!name.empty())
    return name;
  if (op->getNumResults() > 0)
    return getSSAName(op->getResult(0));
  return "<anon>";
}

/// Format a usage list as an inline array: [{name, role, task}, ...].
static void formatUsages(llvm::raw_ostream &os,
                         ArrayRef<BarrierObjectUsage> usages) {
  os << "[";
  for (unsigned i = 0; i < usages.size(); ++i) {
    if (i > 0)
      os << ", ";
    auto &u = usages[i];
    std::string name = u.op ? getOpName(u.op) : "?";
    os << "{" << name << ", " << roleToStr(u.role) << ", " << u.taskId << "}";
  }
  os << "]";
}

static void formatUsages(llvm::raw_ostream &os,
                         ArrayRef<MemoryObjectUsage> usages) {
  os << "[";
  for (unsigned i = 0; i < usages.size(); ++i) {
    if (i > 0)
      os << ", ";
    auto &u = usages[i];
    std::string name = u.op ? getOpName(u.op) : "?";
    os << "{" << name << ", " << memRoleToStr(u.role) << ", " << u.taskId
       << "}";
  }
  os << "]";
}

void dumpBarrierTable(const BarrierGraph &result, llvm::raw_ostream &os) {
  os << "\n=== Barrier Analysis ===\n\n";

  // Table 1: Barrier Objects
  os << "--- Barrier Objects ---\n";
  os << "ID   Name                 Offset   Kind   Usages\n";
  os << "---------------------------------------------------------------\n";
  for (auto &obj : result.barrierObjects) {
    const char *kindStr =
        obj->kind == BarrierObject::MBarrier ? "mbar" : "named";
    os << llvm::format("%-4u %-20s %-8ld %-6s ", obj->id, obj->name.c_str(),
                       obj->offset, kindStr);
    formatUsages(os, obj->usages);
    os << "\n";
  }

  // Table 2: Memory Objects
  os << "\n--- Memory Objects ---\n";
  os << "ID   Name                 Offset   Kind   Usages\n";
  os << "---------------------------------------------------------------\n";
  for (auto &obj : result.memoryObjects) {
    const char *kindStr =
        obj->memoryKind == MemoryObject::SMEM ? "SMEM" : "TMEM";
    os << llvm::format("%-4u %-20s %-8ld %-6s ", obj->id, obj->name.c_str(),
                       obj->offset, kindStr);
    formatUsages(os, obj->usages);
    os << "\n";
  }

  os << "\n=== End Barrier Analysis ===\n";
}

} // namespace mlir::triton
