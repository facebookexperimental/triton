#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

// clang-format off
// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
// C = CTAs, B = buffers, K = mbarriers, T = logical ConSan thread bit slots,
// P = max number of warp-specialize partitions tracked by ConSan (16 for now).
//
// Name                   | Storage | Rank/Type          | Description
// -----------------------|---------|--------------------|------------
// buffers                | tensor  | <C x B x i64>      | Base pointers of all (sub)buffers
// barriers               | tensor  | <C x K x i64>      | Pointers to all individual mbarriers
// barrierStates          | scratch | <C x K x i64>      | Packed barrier phase (bit 0), arrival counts (bits[1..20] init, [21..40] current), and signed tx-count (bits[41..61]); zero means invalid/uninitialized
// barrierWriteRecipients | scratch | <C x K x i32>      | CTA bitsets of EffectWrites rows published by each barrier
// waiting                | scratch | <C x K x i32>      | Two bits per thread: waiting flag bit (LSB), stored phase bit (bit 1)
// writeVisibility        | scratch | <C x B x i64>      | Per-buffer thread-visibility bitmask (bit i => thread i visible)
// readVisibility         | scratch | <C x B x T x i64>  | Per-buffer, per-thread visibility lanes (row-updated; values are bitmasks)
// writeTracking          | scratch | <C x B x K x i8>   | Map buffers -> barriers that track writes
// readTracking           | scratch | <C x B x K x i64>  | Map buffers -> barriers that track reads
// outstandingCommits
//   (async/wgmma)        | scratch | <C x B x P x i8>   | Number of outstanding commits per buffer/base partition-thread (2D replaces prior 1D)
// clang-format on

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

static llvm::StringMap<ConSanHooksFactory> &getHooksRegistry() {
  static llvm::StringMap<ConSanHooksFactory> registry;
  return registry;
}

void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory) {
  getHooksRegistry()[key] = std::move(factory);
}

std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key) {
  auto it = getHooksRegistry().find(key);
  if (it != getHooksRegistry().end())
    return it->second();
  return nullptr;
}

namespace {

// OpBuilder listener tracking operations added to the builder to be wrapped
// with a lock acquire/release pair.
class CriticalSectionListener : public ImplicitLocOpBuilder::Listener {
public:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint /*previous*/) override {
    if (firstOp == nullptr) {
      firstOp = op;
    }
    lastOp = op;
  }
  void maybeWrapWithCriticalSection(ImplicitLocOpBuilder &b,
                                    AuxDataMap &auxData, Value pred) {
    Operation *_firstOp = firstOp;
    Operation *_lastOp = lastOp;
    if (firstOp != nullptr && lastOp != nullptr) {
      assert(firstOp->getParentRegion() == lastOp->getParentRegion());
      b.setInsertionPoint(_firstOp);
      tti::ExperimentalLockAcquireOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
      b.setInsertionPointAfter(_lastOp);
      tti::ExperimentalLockReleaseOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
    }
  }

private:
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::MMAv5OpInterface, ttng::TCGen5CommitOp, ttng::TMEMCopyOp>(
      op);
}

std::optional<int> maybeGetPartitionIdx(Operation *op) {
  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
    return op->getParentRegion()->getRegionNumber();
  }
  if (Operation *parent = op->getParentOp()) {
    return maybeGetPartitionIdx(parent);
  }
  return std::nullopt;
}

int getCurrentThread(Operation *op, const ConSanTargetHooks *hooks) {
  // Default partition is 0, other partitions are idx + 1
  int thread = maybeGetPartitionIdx(op).value_or(-1) + 1;
  if (hooks->isTMAOp(op)) {
    thread += TMA_THREAD_OFFSET;
    return thread;
  }
  if (isTensorCoreOp(op)) {
    thread += TC_THREAD_OFFSET;
    return thread;
  }
  if (hooks->isCLCOp(op)) {
    thread += CLC_THREAD_OFFSET;
    return thread;
  }
  return thread;
}

int getBaseThread(int thread) { return thread % NUM_THREADS; }

// Peer threads are the equivalent threads in the TMA, TC, CLC and normal
// thread classes.
// If a thread is a base thread, return the mask with the peers, otherwise
// return the mask with the thread itself.
uint64_t getThreadPeersMask(int thread) {
  uint64_t mask = 1ULL << thread;
  if (thread < NUM_THREADS) {
    mask |= 1ULL << (thread + TMA_THREAD_OFFSET);
    mask |= 1ULL << (thread + TC_THREAD_OFFSET);
    mask |= 1ULL << (thread + CLC_THREAD_OFFSET);
  }
  return mask;
}

int getActiveMask(Operation *op) {
  int numParts = 1;

  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializeOp>()) {
    numParts = wsOp.getPartitionRegions().size() + 1;
  }
  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
    numParts = wsOp.getPartitionRegions().size() + 1;
  }
  int activeMask = 0;
  for (int i = 0; i < numParts; ++i)
    activeMask |= (1 << i);
  return activeMask;
}

bool shouldInitializeAllocations() {
  std::string envValue = tt::tools::getStrEnv("TRITON_CONSAN_INIT_ALLOCATIONS");
  if (envValue.empty())
    return true;
  if (auto enabled = tt::tools::isEnvValueBool(envValue))
    return *enabled;
  llvm::report_fatal_error("TRITON_CONSAN_INIT_ALLOCATIONS must be a boolean");
}

llvm::APInt getIntegerNaNPattern(unsigned bitWidth) {
  switch (bitWidth) {
  case 16:
    // 0x7FC0 is a NaN in both bfloat16 and float16 interpretations.
    return llvm::APInt(16, 0x7FC0);
  case 32:
    return llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()).bitcastToAPInt();
  case 64:
    return llvm::APFloat::getNaN(llvm::APFloat::IEEEdouble()).bitcastToAPInt();
  default:
    return llvm::APInt::getAllOnes(bitWidth);
  }
}

Value createPoisonTensor(ImplicitLocOpBuilder &b,
                         ttg::MemDescType memDescType) {
  auto region = b.getInsertionBlock()->getParent();
  Type elementType = memDescType.getElementType();
  RankedTensorType poisonType;
  if (isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
    auto encoding = ttng::getDefaultLayoutForTmemLdSt(
        memDescType, ttg::lookupNumWarps(region));
    poisonType =
        RankedTensorType::get(memDescType.getShape(), elementType, encoding);
  } else {
    auto encoding = ttg::getDefaultBlockedEncoding(
        b.getContext(), memDescType.getShape(), ttg::lookupNumWarps(region),
        ttg::lookupThreadsPerWarp(b), ttg::lookupNumCTAs(b));
    encoding = ttg::BlockedEncodingAttr::get(
        b.getContext(), encoding.getSizePerThread(),
        encoding.getThreadsPerWarp(), encoding.getWarpsPerCTA(),
        encoding.getOrder(), ttg::getCGALayout(memDescType.getEncoding()));
    poisonType =
        RankedTensorType::get(memDescType.getShape(), elementType, encoding);
  }

  DenseElementsAttr poison;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    poison = DenseElementsAttr::get(
        poisonType, llvm::APFloat::getNaN(floatType.getFloatSemantics()));
  } else if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    poison = DenseElementsAttr::get(
        poisonType, getIntegerNaNPattern(integerType.getWidth()));
  } else {
    llvm::report_fatal_error(
        "ConSan allocation initialization expects integer or float elements");
  }
  return arith::ConstantOp::create(b, b.getLoc(), poisonType, poison);
}

Value createSingleBufferView(ImplicitLocOpBuilder &b, Value alloc,
                             int64_t buffer) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocType.getShape().begin() + 1,
                             allocType.getShape().end());
  auto viewType = ttg::MemDescType::get(
      shape, allocType.getElementType(), allocType.getEncoding(),
      allocType.getMemorySpace(), allocType.getMutableMemory());
  Value index = arith::ConstantIntOp::create(b, buffer, 32);
  return ttg::MemDescIndexOp::create(b, b.getLoc(), viewType, alloc, index);
}

void initializeAllocation(ImplicitLocOpBuilder &b, Value alloc) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> leaves;
  unsigned storeRank = allocType.getRank();
  if (isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace())) {
    storeRank = 2;
  } else {
    auto encoding = dyn_cast<ttg::LayoutEncodingTrait>(allocType.getEncoding());
    assert(encoding && "shared allocation must have a layout encoding");
    storeRank = encoding.getRank();
  }

  if (allocType.getRank() == storeRank) {
    leaves.push_back(alloc);
  } else {
    assert(allocType.getRank() == storeRank + 1 &&
           "only single-dimension multibuffer allocations are supported");
    for (int64_t buffer = 0; buffer < allocType.getDimSize(0); ++buffer)
      leaves.push_back(createSingleBufferView(b, alloc, buffer));
  }

  bool isTensorMemory =
      isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace());
  ttg::AddrSpace barrierSpace =
      isTensorMemory
          ? (ttg::AddrSpace::TensorRead | ttg::AddrSpace::TensorWrite)
          : ttg::AddrSpace::Local;
  // Synchronize warps, so in case of re-used memory we won't start poisoning
  // memory that is still being used, and finish poisoning before the kernel's
  // first real use of the allocation.
  ttg::BarrierOp::create(b, b.getLoc(), barrierSpace);
  for (Value leaf : leaves) {
    auto leafType = cast<ttg::MemDescType>(leaf.getType());
    Value poison = createPoisonTensor(b, leafType);
    if (isTensorMemory) {
      Value pred = arith::ConstantIntOp::create(b, 1, 1);
      ttng::TMEMStoreOp::create(b, leaf, poison, pred);
    } else {
      ttg::LocalStoreOp::create(b, poison, leaf);
    }
  }
  ttg::BarrierOp::create(b, b.getLoc(), barrierSpace);
}

bool canInitializeAllocation(Value alloc) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  if (!isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace()))
    return true;
  unsigned numWarps = ttg::lookupNumWarps(alloc.getDefiningOp());
  return numWarps % 4 == 0;
}

Value currentCTAMask(ImplicitLocOpBuilder &b) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, 1, 32),
                               ctaId);
}

Value allCTAsMask(ImplicitLocOpBuilder &b) {
  int numCTAs = ttg::lookupNumCTAs(b);
  assert(numCTAs <= 16 && "ConSan CTA bitsets assume at most 16 CTAs");
  return arith::ConstantIntOp::create(b, (1u << numCTAs) - 1, 32);
}

uint16_t getBlockBroadcastMask(Value alloc) {
  auto allocTy = cast<ttg::MemDescType>(alloc.getType());
  auto kBlock = StringAttr::get(alloc.getContext(), "block");
  return toLinearLayout(allocTy).getFreeVariableMasks().lookup(kBlock);
}

Value createCTABitset(ImplicitLocOpBuilder &b, uint32_t pattern,
                      uint32_t baseMask) {
  // Create a CTA bitset by shifting `pattern` by the non-broadcast CTA bits of
  // the current CTA.
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value base = arith::AndIOp::create(
      b, ctaId, arith::ConstantIntOp::create(b, baseMask, 32));
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, pattern, 32),
                               base);
}

Value getMulticastRecipientCTAs(ImplicitLocOpBuilder &b, Value alloc) {
  // Return the CTA rows touched by an alloc: current CTA for
  // non-broadcast allocs, or all CTAs in the current multicast group.
  uint16_t broadcastMask = getBlockBroadcastMask(alloc);
  if (!broadcastMask)
    return currentCTAMask(b);
  int numCTAs = ttg::lookupNumCTAs(b);
  auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastMask);
  return createCTABitset(b, encoding.pattern, encoding.fixedBits);
}

Value getLeaderCTA(ImplicitLocOpBuilder &b, Value barrier) {
  uint16_t broadcastMask = getBlockBroadcastMask(barrier);
  if (!broadcastMask)
    return currentCTAMask(b);
  int numCTAs = ttg::lookupNumCTAs(b);
  auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastMask);
  return createCTABitset(b, /*pattern=*/1, encoding.fixedBits);
}

Value getMulticastBarrierRecipientCTAs(ImplicitLocOpBuilder &b, Value result,
                                       Value barrier) {
  uint32_t resultBroadcastMask = getBlockBroadcastMask(result);
  uint32_t barrierBroadcastMask = getBlockBroadcastMask(barrier);
  int numCTAs = ttg::lookupNumCTAs(b);
  uint32_t recipientBroadcastMask =
      resultBroadcastMask & ~barrierBroadcastMask & (numCTAs - 1);
  auto encoding =
      ttng::getTMAMulticastMaskEncoding(numCTAs, recipientBroadcastMask);
  uint32_t baseMask =
      ~(resultBroadcastMask | barrierBroadcastMask) & (numCTAs - 1);
  return createCTABitset(b, encoding.pattern, baseMask);
}

Value getRecipientCTAsForBroadcastMasks(ImplicitLocOpBuilder &b,
                                        ArrayRef<uint16_t> broadcastMasks) {
  if (broadcastMasks.empty())
    return currentCTAMask(b);

  int numCTAs = ttg::lookupNumCTAs(b);
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value recipientCTAs = arith::ConstantIntOp::create(b, 0, 32);
  // Match eager tcgen05_commit lowering in
  // DotOpToLLVM/MMAv5.cpp:createMMACommit: build one concrete recipient bitset
  // per descriptor, then OR those bitsets.
  for (uint16_t broadcastBits : broadcastMasks) {
    // Compute the map that goes from cta_id to lead_cta_id (fixedBits)
    // and the pattern that goes from cta_0 to its multicast group (pattern).
    auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
    Value fixedBitsVal =
        arith::ConstantIntOp::create(b, encoding.fixedBits, 32);
    Value base = arith::AndIOp::create(b, ctaId, fixedBitsVal);
    Value patternVal = arith::ConstantIntOp::create(b, encoding.pattern, 32);
    Value descRecipientCTAs = arith::ShLIOp::create(b, patternVal, base);
    recipientCTAs = arith::OrIOp::create(b, recipientCTAs, descRecipientCTAs);
  }
  return recipientCTAs;
}

SmallVector<uint16_t> getTensorCoreBarrierBroadcastMasks(Operation *op) {
  assert(isTensorCoreOp(op) && "expected a tensor-core op");
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  SmallVector<Value> commitDescs;
  if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    llvm::append_range(commitDescs, commitOp.getDescs());
  } else if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    if (mmaOp.getMulticast()) {
      if (isa<ttg::SharedEncodingTrait>(mmaOp.getA().getType().getEncoding()))
        commitDescs.push_back(mmaOp.getA());
      commitDescs.push_back(mmaOp.getB());
    }
  } else if (isa<ttng::TMEMCopyOp, ttng::TCGen5MMAScaledOp>(op)) {
    // TODO: we should support descs for tc_gen5_mma_scaled.
  } else {
    llvm_unreachable("unknown tensor-core op");
  }
  return ttng::getCTABroadcastMasks(twoCTAs, commitDescs);
}

Value getBarrierRecipientCTAs(ImplicitLocOpBuilder &b, Operation *op);

Value getMemEffectRecipientCTAs(ImplicitLocOpBuilder &b, Operation *op) {
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    if (copyOp.getMulticast())
      return getMulticastRecipientCTAs(b, copyOp.getResult());
    return currentCTAMask(b);
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op);
      gatherOp && gatherOp.getMulticast())
    return getMulticastRecipientCTAs(b, gatherOp.getResult());
  if (isa<ttng::CLCTryCancelOp>(op))
    return allCTAsMask(b);
  if (isTensorCoreOp(op))
    return getRecipientCTAsForBroadcastMasks(
        b, ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), {}));
  return currentCTAMask(b);
}

Value getBarrierRecipientCTAs(ImplicitLocOpBuilder &b, Operation *op) {
  if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op))
    return getLeaderCTA(b, expectOp.getAlloc());
  if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op))
    return getLeaderCTA(b, arriveOp.getAlloc());
  if (auto arriveOp = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(op))
    return getLeaderCTA(b, arriveOp.getBarrier());
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    if (copyOp.getMulticast())
      return getMulticastBarrierRecipientCTAs(b, copyOp.getResult(),
                                              copyOp.getBarrier());
    return getLeaderCTA(b, copyOp.getBarrier());
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    if (gatherOp.getMulticast())
      return getMulticastBarrierRecipientCTAs(b, gatherOp.getResult(),
                                              gatherOp.getBarrier());
    return getLeaderCTA(b, gatherOp.getBarrier());
  }
  if (isa<ttng::CLCTryCancelOp>(op))
    return allCTAsMask(b);

  if (isTensorCoreOp(op))
    return getRecipientCTAsForBroadcastMasks(
        b, getTensorCoreBarrierBroadcastMasks(op));
  return currentCTAMask(b);
}

class ConcurrencySanitizerImpl {
public:
  ConcurrencySanitizerImpl(ModuleOp module, const ConSanTargetHooks *hooks)
      : module(module), hooks(hooks) {}

  void run() {
    tti::FunctionBuilder funcBuilder(module, auxData);
    auxData.populateAndPassToWarpSpecialize(module, funcBuilder, hooks);

    tt::FuncOp entryPoint = tti::getEntryPoint(module);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    instrumentMemoryOperations(b, funcBuilder);
    initializeAllocations();
  }

private:
  void initializeAllocations() {
    if (!shouldInitializeAllocations())
      return;

    SmallVector<Operation *> allocationsToInitialize;
    module.walk([&](Operation *op) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (!alloc.getSrc())
          allocationsToInitialize.push_back(op);
      }
      if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
        if (!alloc.getSrc())
          allocationsToInitialize.push_back(op);
      }
    });

    for (Operation *op : allocationsToInitialize) {
      ImplicitLocOpBuilder b(op->getLoc(), op);
      b.setInsertionPointAfter(op);
      Value alloc = op->getResult(0);
      if (canInitializeAllocation(alloc))
        initializeAllocation(b, alloc);
    }
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b,
                                  tti::FunctionBuilder &funcBuilder) {
    module.walk([&](Operation *op) {
      CriticalSectionListener listener;
      b.setListener(&listener);

      int thread = getCurrentThread(op, hooks);
      int baseThread = getBaseThread(thread);
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
        // Place insert point after specific ops:
        // allocs - we want to
        //   check if it is not overwriting any earlier allocation, but the
        //   memref value can be referenced only after it is created.
        b.setInsertionPointAfter(op);
      }

      if (auto info = hooks->getBarrierWaitInfo(op)) {
        // For waits we want to instrument it before and after, so we do it
        // manually inside instrumentBarrierWait (disable the critical section
        // listener and return early)
        b.setListener(nullptr);
        instrumentBarrierWait(op, info->alloc, info->phase, info->pred, thread,
                              baseThread, funcBuilder);
        return;
      }

      instrumentMemEffects(b, op, thread, funcBuilder);
      b.setLoc(op->getLoc());
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        auto partitionRegions = wsOp.getPartitionRegions();
        if (!partitionRegions.empty()) {
          uint64_t destMask = 0;
          for (size_t idx = 0, e = partitionRegions.size(); idx < e; ++idx)
            destMask |= getThreadPeersMask(idx + 1);
          if (destMask) {
            for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
              funcBuilder.createCopyWriteVisibilityCall(b, thread, destMask,
                                                        nullptr, memType, op);
              funcBuilder.createCopyReadVisibilityCall(b, thread, destMask,
                                                       nullptr, memType, op);
            }
          }
        }
      }
      if (auto info = hooks->getBarrierInitInfo(op)) {
        Value pred = hooks->getIssuerCTAPred(b, op);
        funcBuilder.createVerifyBarrierCanInitCall(b, info->alloc, pred, op,
                                                   currentCTAMask(b));
        funcBuilder.createInitBarrierStateCall(b, info->alloc, info->count,
                                               pred, op);
      }
      if (auto info = hooks->getBarrierInvalidateInfo(op)) {
        Value barrier = info->alloc;
        Value pred = hooks->getIssuerCTAPred(b, op);
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, pred, op,
                                                       currentCTAMask(b));
        funcBuilder.createInvalidateBarrierStateCall(b, barrier, pred, op);
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createClearBarrierWriteTrackingCall(b, barrier, pred,
                                                          memType, op);
          funcBuilder.createClearBarrierReadTrackingCall(b, barrier, pred,
                                                         memType, op);
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        if (!auxData.commits[CommitKind::AsyncCp].empty())
          funcBuilder.createCommitAccessesCall(b, thread, nullptr,
                                               CommitKind::AsyncCp, op);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferWritesCall(
            b, baseThread, getThreadPeersMask(thread), asyncWaitOp.getNum(),
            nullptr, CommitKind::AsyncCp, MemType::SHARED_MEM, op);
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferReadsCall(
            b, baseThread, getThreadPeersMask(thread),
            wgmmaWaitOp.getPendings(), nullptr, CommitKind::Wgmma,
            MemType::SHARED_MEM, op);
      }
      if (auto info = hooks->getWaitOpInfo(op)) {
        if (info->transferWrites && info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferBothCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        } else if (info->transferWrites) {
          funcBuilder.createClearOutstandingCommitsTransferWritesCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        } else if (info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferReadsCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        }
      }

      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  void instrumentBarrierWait(Operation *op, Value alloc, Value phase,
                             Value pred, int thread, int baseThread,
                             tti::FunctionBuilder &funcBuilder) {
    ImplicitLocOpBuilder wb(op->getLoc(), op);
    pred = tti::maybeAnd(wb, pred, hooks->getIssuerCTAPred(wb, op));
    Value lock = auxData.lock.at(op).value;
    // Pre-wait: mark waiting threads and check for deadlock.
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    funcBuilder.createVerifyBarrierInitializedCall(wb, alloc, pred, op,
                                                   currentCTAMask(wb));
    funcBuilder.createSetWaitingCall(wb, alloc, baseThread, phase, pred, op);
    funcBuilder.createCheckAllActiveWaitingCall(wb, getActiveMask(op), pred,
                                                op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
    // Post-wait: transfer visible writes and reads to all peer threads,
    // and clear waiting for this barrier.
    assert(!auxData.barriers.empty() &&
           "barrier descriptors must exist when instrumenting wait");
    wb.setInsertionPointAfter(op);
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      funcBuilder.createTransferVisibleWritesCall(
          wb, alloc, getThreadPeersMask(thread), pred, memType, op);
      funcBuilder.createTransferVisibleReadsCall(
          wb, alloc, getThreadPeersMask(thread), pred, memType, op);
    }
    funcBuilder.createClearWaitingCall(wb, alloc, baseThread, pred, op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
  }

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op, int thread,
                            tti::FunctionBuilder &funcBuilder) {
    int baseThread = getBaseThread(thread);
    std::optional<MemEffectsOpInfo> opInfo = hooks->getMemEffectsOpInfo(op);
    if (!opInfo) {
      return;
    }
    Value pred = opInfo->pred;
    Value issuerCTAPred = hooks->getIssuerCTAPred(b, op);
    pred = tti::maybeAnd(b, pred, issuerCTAPred);
    Value effectRecipientCTAs = getMemEffectRecipientCTAs(b, op);
    for (auto effect : opInfo->operandEffects) {
      Value buf = effect.buf;
      auto bufType = cast<ttg::MemDescType>(buf.getType());
      MemType memType = MemType::TENSOR_MEM;
      if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
        memType = MemType::SHARED_MEM;
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, effectRecipientCTAs,
                       opInfo->commitKind);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetReadVisibilityCall(
              b, buf, effect.length, getThreadPeersMask(thread), pred, memType,
              op, effectRecipientCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(b, buf, effect.length,
                                                     baseThread, pred, memType,
                                                     opInfo->commitKind, op);
        }
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, effectRecipientCTAs,
                       opInfo->commitKind);
        addReadChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                      thread, effect.operandName, effectRecipientCTAs,
                      opInfo->commitKind);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetWriteVisibilityCall(
              b, buf, effect.length, getThreadPeersMask(thread), pred, memType,
              op, effectRecipientCTAs);
          funcBuilder.createClearWriteTrackingCall(
              b, buf, effect.length, pred, memType, op, effectRecipientCTAs);
          funcBuilder.createClearReadVisibilityCall(
              b, buf, effect.length, pred, memType, op, effectRecipientCTAs);
          funcBuilder.createClearReadTrackingCall(
              b, buf, effect.length, pred, memType, op, effectRecipientCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(b, buf, effect.length,
                                                     baseThread, pred, memType,
                                                     opInfo->commitKind, op);
        }
      }
    }
    for (const auto &barrierInfo : opInfo->barriers) {
      Value barrier = barrierInfo.barrier;
      Value combinedPred = tti::maybeAnd(b, barrierInfo.pred, pred);
      Value recipientCTAs = getBarrierRecipientCTAs(b, op);
      funcBuilder.createVerifyBarrierInitializedCall(b, barrier, combinedPred,
                                                     op, recipientCTAs);
      if (barrierInfo.trackingMode ==
          MemEffectsOpInfo::BarrierTrackingMode::Frontier) {
        // If the op has barriers, we treat it as a commit emitted for each
        // barrier.
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createTrackVisibleWritesCall(
              b, barrier, thread, combinedPred, memType, op, recipientCTAs);
          funcBuilder.createTrackVisibleReadsCall(
              b, barrier, thread, combinedPred, memType, op, recipientCTAs);
        }
      } else if (barrierInfo.trackingMode ==
                 MemEffectsOpInfo::BarrierTrackingMode::EffectWrites) {
        for (const auto &effect : opInfo->operandEffects) {
          if (effect.rw != MemEffectsOpInfo::Effects::Write)
            continue;
          auto bufType = cast<ttg::MemDescType>(effect.buf.getType());
          MemType memType = MemType::TENSOR_MEM;
          if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding()))
            memType = MemType::SHARED_MEM;
          funcBuilder.createTrackBarrierWriteForBufferCall(
              b, barrier, effect.buf, effect.length, combinedPred, memType, op,
              recipientCTAs, effectRecipientCTAs,
              barrierInfo.diagonalEffectRecipientCTAs);
        }
      }
      if (barrierInfo.count > 0 || barrierInfo.txCount != 0) {
        funcBuilder.createVerifyBarrierArriveCall(
            b, barrier, barrierInfo.count, combinedPred, op, recipientCTAs,
            barrierInfo.txCount);
        funcBuilder.createUpdateBarrierStateCall(
            b, barrier, barrierInfo.count, combinedPred, op, recipientCTAs,
            barrierInfo.txCount);
      }
    }
    if (opInfo->implicitCommit) {
      assert(opInfo->trackingKind ==
             MemEffectsOpInfo::TrackingKind::CommitCount);
      funcBuilder.createCommitAccessesCall(b, baseThread, pred,
                                           opInfo->commitKind, op);
    }
  }

  void addWriteChecks(ImplicitLocOpBuilder &b,
                      tti::FunctionBuilder &funcBuilder, Operation *op,
                      Value buf, uint32_t length, Value pred, MemType memType,
                      int thread, const std::string &operandName,
                      Value recipientCTAs,
                      CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyWriteVisibilityCall(
        b, buf, length, thread, operandName, pred, memType, op, recipientCTAs);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks->getOutstandingWriteCommitKinds()) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks->isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, buf, length, getBaseThread(thread), commitKindDesc.operationDesc,
            pred, memType, commitKindDesc.kind, op, recipientCTAs, excludeSelf);
      }
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value buf, uint32_t length, Value pred,
                     MemType memType, int thread,
                     const std::string &operandName, Value recipientCTAs,
                     CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyReadVisibilityCall(
        b, buf, length, thread, operandName, pred, memType, op, recipientCTAs);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks->getOutstandingReadCommitKinds()) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks->isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, buf, length, getBaseThread(thread), commitKindDesc.operationDesc,
            pred, memType, commitKindDesc.kind, op, recipientCTAs, excludeSelf);
      }
    }
  }

  ModuleOp module;
  AuxDataMap auxData;
  const ConSanTargetHooks *hooks;
};

} // namespace

void runConcurrencySanitizer(ModuleOp module, const ConSanTargetHooks *hooks) {
  assert(hooks && "hooks must not be null");
  ConcurrencySanitizerImpl impl(module, hooks);
  impl.run();
}

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto targetAttr = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    assert(targetAttr && "module missing ttg.target attribute");
    StringRef target = targetAttr.strref();
    StringRef key = target.starts_with("cuda:")  ? "nvidia"
                    : target.starts_with("hip:") ? "amd"
                                                 : "";
    auto hooks = createConSanHooks(key);
    assert(hooks && "no ConSan hooks registered for target");
    runConcurrencySanitizer(module, hooks.get());
  }
};

} // namespace instrument
} // namespace triton
} // namespace mlir
