#ifndef NVIDIA_NVWS_TRANSFORMS_MEMORY_PLANNER_NVWS_ADAPTER_H_
#define NVIDIA_NVWS_TRANSFORMS_MEMORY_PLANNER_NVWS_ADAPTER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir::triton::nvws::planner {

enum class DataChannelKind { SMEMPost, TMEMPost };

// This interface intentionally mirrors Meta-AWS ChannelPost. MemoryPlanner.cpp
// can therefore retain Meta's algorithms while channel discovery remains an
// NVWS representation adapter over ttg.partition annotations.
struct Channel {
  Channel(int producer, ArrayRef<int> consumers, Operation *allocOp,
          unsigned uniqID, DataChannelKind channelKind)
      : producer(producer), consumers(consumers.begin(), consumers.end()),
        allocOp(allocOp), uniqID(uniqID), channelKind(channelKind) {}
  virtual ~Channel() = default;

  virtual Operation *getAllocOp() const { return allocOp; }
  virtual Operation *getSrcOp() const = 0;
  virtual Operation *getDstOp() const = 0;
  virtual void getDstOps(SmallVectorImpl<Operation *> &dsts) const = 0;

  int producer;
  SmallVector<int> consumers;
  Operation *allocOp;
  unsigned uniqID;
  DataChannelKind channelKind;
};

struct ChannelPost final : Channel {
  ChannelPost(int producer, ArrayRef<int> consumers, Operation *allocOp,
              Operation *srcOp, ArrayRef<Operation *> dstOps, unsigned uniqID)
      : Channel(producer, consumers, allocOp, uniqID,
                DataChannelKind::SMEMPost),
        explicitSrcOp(srcOp), explicitDstOps(dstOps.begin(), dstOps.end()) {}

  Operation *getSrcOp() const override { return explicitSrcOp; }
  Operation *getDstOp() const override {
    return explicitDstOps.empty() ? nullptr : explicitDstOps.back();
  }
  void getDstOps(SmallVectorImpl<Operation *> &dsts) const override {
    dsts.append(explicitDstOps.begin(), explicitDstOps.end());
  }

  Operation *explicitSrcOp;
  SmallVector<Operation *> explicitDstOps;
};

using LocalDataChannelPost = ChannelPost;

struct TmemDataChannelPost final : Channel {
  TmemDataChannelPost(int producer, ArrayRef<int> consumers,
                      Operation *allocOp, bool isOperandD,
                      bool isOperandDNoAcc, unsigned uniqID)
      : Channel(producer, consumers, allocOp, uniqID,
                DataChannelKind::TMEMPost),
        isOperandD(isOperandD), isOperandDNoAcc(isOperandDNoAcc) {}

  TmemDataChannelPost(int producer, ArrayRef<int> consumers,
                      Operation *allocOp, Operation *srcOp,
                      ArrayRef<Operation *> dstOps, bool isOperandD,
                      bool isOperandDNoAcc, bool isPlannerOnly,
                      unsigned uniqID)
      : Channel(producer, consumers, allocOp, uniqID,
                DataChannelKind::TMEMPost),
        explicitSrcOp(srcOp), explicitDstOps(dstOps.begin(), dstOps.end()),
        isOperandD(isOperandD), isOperandDNoAcc(isOperandDNoAcc),
        isPlannerOnly(isPlannerOnly) {}

  Operation *getSrcOp() const override;
  Operation *getDstOp() const override;
  void getDstOps(SmallVectorImpl<Operation *> &dsts) const override;

  Operation *explicitSrcOp = nullptr;
  SmallVector<Operation *> explicitDstOps;
  bool isOperandD;
  bool isOperandDNoAcc;
  bool isSameIterGuard = false;
  bool isPlannerOnly = false;
};

LogicalResult collectPostChannels(
    SmallVectorImpl<std::unique_ptr<Channel>> &channels, FuncOp funcOp);

// Translate Meta planner results into the additional attributes consumed by
// NVWS InsertSemas. These adapters may not change buffer.id or buffer.copy.
LogicalResult emitSmemPlanAnnotations(
    FuncOp funcOp, ArrayRef<Channel *> channels, int smemAllocAlgo,
    bool smemCircularReuse, const DenseSet<Operation *> &eligibleAllocs);
void emitTmemOwnerOffsets(FuncOp funcOp);

Operation *getSameLevelOp(Operation *producer, Operation *consumer);
SmallVector<Operation *> getActualConsumers(Operation *consumer);
bool hasLoopCarriedAccToken(Operation *tmemAlloc, scf::ForOp forOp);

void dumpCombinedGraph(SmallVector<std::unique_ptr<Channel>> &channels,
                       FuncOp funcOp, llvm::raw_ostream &os);
void dumpTmemBufferLiveness(
    SmallVector<nvidia_gpu::TMEMAllocOp> &allocs,
    DenseMap<Operation *, Interval<size_t>> &allocToIntervals,
    DenseMap<Operation *, nvidia_gpu::TMemAllocation> &allocToSize,
    DenseMap<Operation *, TmemDataChannelPost *> &allocToChannel,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os);
void dumpSmemBufferLiveness(
    llvm::MapVector<Allocation::BufferId, std::pair<Interval<size_t>, size_t>>
        &bufferInfo,
    DenseMap<Allocation::BufferId, Operation *> &bufferOwners,
    SmallVector<Channel *> &channels, llvm::raw_ostream &os);

} // namespace mlir::triton::nvws::planner

#endif // NVIDIA_NVWS_TRANSFORMS_MEMORY_PLANNER_NVWS_ADAPTER_H_
