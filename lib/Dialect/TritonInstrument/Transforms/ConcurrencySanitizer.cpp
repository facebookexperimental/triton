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

// clang-format off
// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
//
// Name              | Storage | Rank/Type       | Description
// ------------------|---------|-----------------|------------
// buffers           | tensor  | <B x i64>       | Base pointers of all (sub)buffers
// barriers          | tensor  | <K x i64>       | Pointers to all individual mbarriers
// barrierStates     | scratch | <K x i32>       | Packed barrier phase (bit 0) and arrival counts (bits[1..8] init, [9..16] current); zero means invalid/uninitialized
// waiting           | scratch | <K x i32>       | Two bits per thread: waiting flag bit (LSB), stored phase bit (bit 1)
// writeVisibility   | scratch | <B x i64>       | Per-buffer thread-visibility bitmask (bit i => thread i visible)
// readVisibility    | scratch | <B x T x i64>   | Per-buffer, per-thread visibility lanes (row-updated; values are bitmasks)
// writeTracking     | scratch | <B x K x i8>    | Map buffers -> barriers that track writes
// readTracking      | scratch | <B x K x i64>   | Map buffers -> barriers that track reads
// outstandingCommits
//   (async/wgmma)   | scratch | <B x T x i8>    | Number of outstanding commits per buffer/thread (2D replaces prior 1D)
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
  return isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp, ttng::TCGen5CommitOp>(
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
  return thread;
}

int getBaseThread(int thread) { return thread % NUM_THREADS; }

// Peer threads are the equivalent threads in the TMA, TC and normal
// thread classes.
// If a thread is a base thread, return the mask with the peers, otherwise
// return the mask with the thread itself.
uint64_t getThreadPeersMask(int thread) {
  uint64_t mask = 1ULL << thread;
  if (thread < NUM_THREADS) {
    mask |= 1ULL << (thread + TMA_THREAD_OFFSET);
    mask |= 1ULL << (thread + TC_THREAD_OFFSET);
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
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op) ||
          hooks->isPostInstrumentedOp(op)) {
        // Place insert point after specific ops:
        // allocs - we want to
        //   check if it is not overwriting any earlier allocation, but the
        //   memref value can be referenced only after it is created.
        // wait barriers - we can update aux data only after the wait is
        //   completed
        b.setInsertionPointAfter(op);
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
        funcBuilder.createVerifyBarrierCanInitCall(b, info->alloc, op);
        funcBuilder.createInitBarrierStateCall(b, info->alloc, info->count, op);
      }
      if (auto invalOp = dyn_cast<ttng::InvalBarrierOp>(op)) {
        Value barrier = invalOp.getAlloc();
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, nullptr,
                                                       invalOp);
        funcBuilder.createInvalidateBarrierStateCall(b, barrier, invalOp);
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createClearBarrierWriteTrackingCall(b, barrier, nullptr,
                                                          memType, invalOp);
          funcBuilder.createClearBarrierReadTrackingCall(b, barrier, nullptr,
                                                         memType, invalOp);
        }
      }
      if (auto info = hooks->getBarrierWaitInfo(op)) {
        instrumentBarrierWait(b, listener, op, info->alloc, info->phase,
                              info->pred, thread, baseThread, funcBuilder);
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
        if (info->transferWrites) {
          funcBuilder.createClearOutstandingCommitsTransferWritesCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        } else {
          funcBuilder.createClearOutstandingCommitsTransferReadsCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        }
      }

      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  void instrumentBarrierWait(ImplicitLocOpBuilder &b,
                             CriticalSectionListener &listener, Operation *op,
                             Value alloc, Value phase, Value pred, int thread,
                             int baseThread,
                             tti::FunctionBuilder &funcBuilder) {
    // Pre-wait: mark waiting threads and check for deadlock.
    {
      CriticalSectionListener preListener;
      b.setListener(&preListener);
      b.setInsertionPoint(op);
      funcBuilder.createVerifyBarrierInitializedCall(b, alloc, pred, op);
      funcBuilder.createSetWaitingCall(b, alloc, baseThread, phase, pred, op);
      funcBuilder.createCheckAllActiveWaitingCall(b, getActiveMask(op), pred,
                                                  op);
      preListener.maybeWrapWithCriticalSection(b, auxData, pred);
      b.setListener(&listener);
      b.setInsertionPointAfter(op);
    }
    // Post-wait: transfer visible writes and reads to all peer threads,
    // and clear waiting for this barrier
    assert(!auxData.barriers.empty() &&
           "barrier descriptors must exist when instrumenting wait");
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      funcBuilder.createTransferVisibleWritesCall(
          b, alloc, getThreadPeersMask(thread), pred, memType, op);
      funcBuilder.createTransferVisibleReadsCall(
          b, alloc, getThreadPeersMask(thread), pred, memType, op);
    }
    funcBuilder.createClearWaitingCall(b, alloc, baseThread, pred, op);
  }

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op, int thread,
                            tti::FunctionBuilder &funcBuilder) {
    int baseThread = getBaseThread(thread);
    std::optional<MemEffectsOpInfo> opInfo = hooks->getMemEffectsOpInfo(op);
    if (!opInfo) {
      return;
    }
    Value pred = opInfo->pred;
    auto combinePredicates = [&](Value barrierPred) -> Value {
      if (barrierPred && pred) {
        return arith::AndIOp::create(b, b.getLoc(), barrierPred, pred);
      }
      return barrierPred ? barrierPred : pred;
    };
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
                       thread, effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetReadVisibilityCall(b, buf, effect.length,
                                                  getThreadPeersMask(thread),
                                                  pred, memType, op);
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
                       thread, effect.operandName);
        addReadChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                      thread, effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetWriteVisibilityCall(b, buf, effect.length,
                                                   getThreadPeersMask(thread),
                                                   pred, memType, op);
          funcBuilder.createClearWriteTrackingCall(b, buf, effect.length, pred,
                                                   memType, op);
          funcBuilder.createClearReadVisibilityCall(b, buf, effect.length, pred,
                                                    memType, op);
          funcBuilder.createClearReadTrackingCall(b, buf, effect.length, pred,
                                                  memType, op);
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
      Value combinedPred = combinePredicates(barrierInfo.pred);
      funcBuilder.createVerifyBarrierInitializedCall(b, barrier, combinedPred,
                                                     op);
      if (barrierInfo.trackingMode ==
          MemEffectsOpInfo::BarrierTrackingMode::Frontier) {
        // If the op has barriers, we treat it as a commit emitted for each
        // barrier.
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createTrackVisibleWritesCall(b, barrier, thread,
                                                   combinedPred, memType, op);
          funcBuilder.createTrackVisibleReadsCall(b, barrier, thread,
                                                  combinedPred, memType, op);
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
              b, barrier, effect.buf, effect.length, combinedPred, memType, op);
        }
      }
      if (barrierInfo.count > 0) {
        funcBuilder.createVerifyBarrierArriveCall(b, barrier, barrierInfo.count,
                                                  combinedPred, op);
        funcBuilder.createUpdateBarrierStateCall(b, barrier, barrierInfo.count,
                                                 combinedPred, op);
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
                      int thread, const std::string &operandName) {
    funcBuilder.createVerifyWriteVisibilityCall(b, buf, length, thread,
                                                operandName, pred, memType, op);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "async_copy_global_to_shared",
          pred, memType, CommitKind::AsyncCp, op);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value buf, uint32_t length, Value pred,
                     MemType memType, int thread,
                     const std::string &operandName) {
    funcBuilder.createVerifyReadVisibilityCall(b, buf, length, thread,
                                               operandName, pred, memType, op);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "warpgroup_mma operand read",
          pred, memType, CommitKind::Wgmma, op);
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "async_copy_shared_to_global",
          pred, memType, CommitKind::TmaStore, op);
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
