#include "mlir/IR/BuiltinOps.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUALLOCATEWARPGROUPS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

constexpr int32_t kMinRegistersForAssertOrPrint = 32;

static bool regionUsesAssertOrPrint(Region &region) {
  return region
      .walk([](Operation *op) {
        return isa<AssertOp, PrintOp>(op) ? WalkResult::interrupt()
                                          : WalkResult::advance();
      })
      .wasInterrupted();
}

static int32_t minRegistersForRegion(Region &region, bool instrumented,
                                     int32_t minRegisters) {
  // Work around an NVIDIA driver bug: regions that may call device system
  // functions such as assert or print need at least 32 registers.
  return instrumented || regionUsesAssertOrPrint(region)
             ? std::max(minRegisters, kMinRegistersForAssertOrPrint)
             : minRegisters;
}

// A module with exactly one AutoWS-generated WarpSpecializeOp can use direct
// warp-ID dispatch. Require the AutoWS tag so manual TLX is not opted in
// implicitly; TLX requests this lowering via async_tasks(exclusive=True).
static void enableSingleWarpSpecializeForAutoWS(ModuleOp mod) {
  if (mod->hasAttr(AttrSingleWarpSpecializeName))
    return;

  SmallVector<WarpSpecializeOp> wsOps;
  mod.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  if (wsOps.size() != 1)
    return;

  bool hasAutoWSTag = false;
  wsOps.front()->walk([&](Operation *op) {
    hasAutoWSTag |= op->hasAttr(kWarpSpecializeTagAttrName);
  });
  if (hasAutoWSTag)
    setHasSingleWarpSpecialize(mod, true);
}

// Given a `ttg.warp_specialize` with a certain number of existing warps, pad it
// with extra warps until it has the same number of full warp groups as the
// largest partitioning. This ensures that all threads can be present to
// surrender registers.
static void padToMaxWarpGroups(WarpSpecializeOp op, int numExtraWarpGroups) {
  int numExtraWarps = op.getTotalPartitionWarps();
  int warpsToAdd = numExtraWarpGroups * 4 - numExtraWarps;
  assert(warpsToAdd >= 0);

  // Fill it with powers of 2.
  SmallVector<int> paddingPartitionSizes;
  while (warpsToAdd > 0) {
    int paddingSize = llvm::NextPowerOf2(warpsToAdd) / 2;
    paddingPartitionSizes.push_back(paddingSize);
    warpsToAdd -= paddingSize;
  }

  auto partitions = cast<WarpSpecializePartitionsOp>(
      op.getPartitionOpHolder().front().front());
  OperationState state(partitions.getLoc(), partitions.getOperationName(),
                       partitions.getOperands(), /*types=*/{});
  for (Region *region : partitions.getRegions())
    state.addRegion()->takeBody(*region);

  SmallVector<int32_t> partitionNumWarps(op.getPartitionNumWarps());
  for (int paddingSize : paddingPartitionSizes) {
    partitionNumWarps.push_back(paddingSize);

    Block &body = state.addRegion()->emplaceBlock();
    for (Value capture : op.getPartitionOp().getExplicitCaptures())
      body.addArgument(capture.getType(), capture.getLoc());
    OpBuilder b(op.getContext());
    b.setInsertionPointToStart(&body);
    WarpReturnOp::create(b, op.getLoc());
  }
  op.setPartitionNumWarps(partitionNumWarps);

  // Set the requested registers as low as legally possible for the padded
  // partitions that do nothing. `nvvm.setmaxregister` requires the per-thread
  // register count to be in [24, 256] (a multiple of 8) on all sm_90+ targets
  // (Hopper and Blackwell), so 24 is the floor. Using a below-minimum value
  // (e.g. 16) is only masked when a padding partition shares a warp group with
  // a real partition requesting >= 24 (the warp group's request is the max over
  // its members); when it does not, the illegal value reaches `setmaxregister`
  // and lowering fails the verifier.
  constexpr int32_t kMinLegalRegisters = 24;
  if (auto reqRegs = op.getRequestedRegisters()) {
    SmallVector<int32_t> newReqRegs(*reqRegs);
    newReqRegs.append(paddingPartitionSizes.size(), kMinLegalRegisters);
    op.setRequestedRegisters(newReqRegs);
  }

  OpBuilder b(partitions);
  b.create(state);
  partitions.erase();
}

namespace {
struct AllocateWarpGroups
    : public mlir::triton::gpu::impl::TritonGPUAllocateWarpGroupsBase<
          AllocateWarpGroups> {
  using TritonGPUAllocateWarpGroupsBase::TritonGPUAllocateWarpGroupsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool laterInstrumentation = instrumented.getValue();
    enableSingleWarpSpecializeForAutoWS(mod);

    // First determine the maximum number of extra warps.
    int maxExtraWarps = 0;
    mod.walk([&](WarpSpecializeOp op) {
      maxExtraWarps = std::max<int>(maxExtraWarps, op.getTotalPartitionWarps());
    });

    // Round this up to the nearest warpgroup (multiple of 4) and then pad each
    // `ttg.warp_specialize` to the nearest warpgroup.
    int numExtraWarpGroups = llvm::divideCeil(maxExtraWarps, 4);
    mod.walk([&](WarpSpecializeOp op) {
      padToMaxWarpGroups(op, numExtraWarpGroups);
    });

    int baseNumWarps = lookupNumWarps(mod);

    // Compute the total number of warps required at any given time.
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();

      // Allocate the start IDs such that the largest warpgroups have lower
      // starting warp IDs.
      // FIXME: Handle aligning warp group IDs to 4 for TMEM.
      SmallVector<std::pair<unsigned, int32_t>> idxAndSize;
      for (auto [i, size] : llvm::enumerate(arr))
        idxAndSize.emplace_back(i, size);

      // If user-provided warpGroupStartIds exist, they cover only the
      // original (non-padding) partitions. Respect the user-provided IDs
      // for those partitions and assign IDs to padding partitions after.
      SmallVector<int32_t> startIds(arr.size());
      if (auto existingIds = op.getWarpGroupStartIds();
          existingIds && existingIds->size() < arr.size()) {
        // User provided IDs for the first N partitions. Compute the max
        // warp used by those, then assign padding partitions after.
        int maxWarp = 0;
        for (auto [id, numWarps] :
             llvm::zip(*existingIds, arr.take_front(existingIds->size())))
          maxWarp = std::max(maxWarp, (int)(id + numWarps));

        // Copy user-provided IDs.
        for (unsigned i = 0; i < existingIds->size(); ++i)
          startIds[i] = (*existingIds)[i];

        // Assign padding partitions sequentially after the real ones.
        int startId = maxWarp;
        for (unsigned i = existingIds->size(); i < arr.size(); ++i) {
          startIds[i] = startId;
          startId += arr[i];
        }
      } else {
        // No user-provided IDs (or they cover all partitions already).
        // Sort by size descending (stable to preserve order for equal sizes).
        llvm::stable_sort(idxAndSize, [&](auto lhs, auto rhs) {
          return lhs.second > rhs.second;
        });

        int startId = baseNumWarps;
        for (auto [i, size] : idxAndSize) {
          startIds[i] = startId;
          startId += size;
        }
      }
      op.setWarpGroupStartIds(startIds);
    });

    Builder b(&getContext());
    mod->setAttr("ttg.total-num-warps",
                 b.getI32IntegerAttr(baseNumWarps + numExtraWarpGroups * 4));

    bool needsRegisterOptimization = false;
    mod.walk([&](WarpSpecializeOp op) {
      if (op.getRequestedRegisters() || op.getDefaultRequestedRegisters())
        needsRegisterOptimization = true;
    });

    if (!needsRegisterOptimization)
      return;

    // Determine the maximum number of registers per thread. This may have
    // been set by the user.
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int maxnreg;
    if (auto maxnregAttr =
            mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName)) {
      maxnreg = maxnregAttr.getInt();
    } else {
      // Assume the user wants to use all 64K registers.
      maxnreg = (64 * 1024) / (baseNumWarps + numExtraWarpGroups * 4) /
                threadsPerWarp;
      maxnreg = maxnreg / 8 * 8;
    }

    struct WarpGroupInfo {
      SmallVector<Region *> partitions;
      int maxRequestedRegs = -1;
      unsigned numWarps = 0;
    };
    struct WarpGroupPartition {
      int startId;
      Region *partition;
      int32_t estRegs;
      int numWarps;
    };

    // Compute register allocation for each warp specialize op.
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();
      auto startIds = *op.getWarpGroupStartIds();

      // Require that an estimate has been set and that we have even warpgroups.
      auto regsAttr = op.getRequestedRegisters();
      auto defaultRegsAttr = op.getDefaultRequestedRegisters();
      if ((!regsAttr && !defaultRegsAttr) ||
          op.getTotalPartitionWarps() % 4 != 0)
        return;

      SmallVector<int32_t> requestedRegisters(arr.size(), -1);
      if (regsAttr)
        llvm::copy(*regsAttr, requestedRegisters.begin());

      // Group the partitions into warpgroups.
      SmallVector<WarpGroupPartition> orderedPartitions;
      for (auto [startId, partition, estRegs, numWarps] : llvm::zip(
               startIds, op.getPartitionRegions(), requestedRegisters, arr)) {
        int minRegs =
            minRegistersForRegion(*partition, laterInstrumentation, estRegs);
        orderedPartitions.push_back({startId, partition, minRegs, numWarps});
      }
      llvm::sort(orderedPartitions,
                 [&](auto lhs, auto rhs) { return lhs.startId < rhs.startId; });

      // Iterate over the partitions and assign them to warp groups. Determine
      // the maximum number of requested registers per warp group.
      // A requested value of -1 is a sentinel meaning "give me an even share
      // of leftover registers." When computing the per-warp-group max, -1
      // values are skipped; a warp group is sentinel only if all its
      // partitions are -1 (maxRequestedRegs stays at the initial -1).
      SmallVector<WarpGroupInfo> warpGroups;
      for (auto [startId, partition, estRegs, numWarps] : orderedPartitions) {
        if (startId % 4 == 0) {
          warpGroups.push_back(WarpGroupInfo{});
        }
        warpGroups.back().partitions.push_back(partition);
        if (estRegs > 0) {
          int estRegsCeil8 = llvm::divideCeil(estRegs, 8) * 8;
          warpGroups.back().maxRequestedRegs =
              std::max<int>(warpGroups.back().maxRequestedRegs, estRegsCeil8);
        }
        warpGroups.back().numWarps += numWarps;
      }

      SmallVector<int32_t> maxnregsPerPartition(1 + arr.size());

      int totalWarps = baseNumWarps;
      for (const WarpGroupInfo &wg : warpGroups)
        totalWarps += wg.numWarps;
      int remainingRegs = maxnreg * totalWarps * threadsPerWarp;
      int sharingThreads = 0;

      int defaultRegs = -1;
      if (defaultRegsAttr) {
        defaultRegs = minRegistersForRegion(
            op.getDefaultRegion(), laterInstrumentation, *defaultRegsAttr);
        remainingRegs -= defaultRegs * baseNumWarps * threadsPerWarp;
      } else {
        sharingThreads += baseNumWarps * threadsPerWarp;
      }

      for (const WarpGroupInfo &wg : warpGroups) {
        assert(wg.numWarps % 4 == 0);
        if (wg.maxRequestedRegs > 0)
          remainingRegs -= wg.maxRequestedRegs * wg.numWarps * threadsPerWarp;
        else
          sharingThreads += wg.numWarps * threadsPerWarp;
      }
      if (remainingRegs < 0)
        return;

      int sharedRegs = -1;
      if (sharingThreads > 0) {
        sharedRegs = remainingRegs / sharingThreads;
        sharedRegs = sharedRegs / 8 * 8;
        int minSharedRegs = 24;
        if (defaultRegs < 0)
          minSharedRegs = minRegistersForRegion(
              op.getDefaultRegion(), laterInstrumentation, minSharedRegs);
        if (sharedRegs < minSharedRegs)
          return;
      }

      maxnregsPerPartition.front() = defaultRegs > 0 ? defaultRegs : sharedRegs;
      for (const WarpGroupInfo &wg : warpGroups) {
        int regs = wg.maxRequestedRegs > 0 ? wg.maxRequestedRegs : sharedRegs;
        for (Region *region : wg.partitions)
          maxnregsPerPartition[1 + region->getRegionNumber()] = regs;
      }

      op.setActualRegisters(maxnregsPerPartition);

      // Set the initial max number of registers. This is needed for PTXAS to
      // cooperate.
      mod->setAttr(AttrMaxRegistersName,
                   Builder(op.getContext()).getI32IntegerAttr(maxnreg));
    });
  }
};
} // namespace
