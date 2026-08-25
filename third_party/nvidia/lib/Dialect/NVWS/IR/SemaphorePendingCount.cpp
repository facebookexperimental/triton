/* Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nvidia/include/Dialect/NVWS/IR/SemaphorePendingCount.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

namespace mlir::triton::nvws {
namespace {

std::optional<int>
getReleaseAsyncContribution(ArrayAttr asyncOps,
                            std::optional<AsyncOp> &unsupportedAsyncOp) {
  int contribution = 0;
  for (Attribute asyncOp : asyncOps) {
    auto kind = cast<AsyncOpAttr>(asyncOp).getValue();
    switch (kind) {
    case AsyncOp::TC5MMA:
    case AsyncOp::TMALoad:
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
    case AsyncOp::TMEMCopy:
      ++contribution;
      break;
    case AsyncOp::CpAsync:
      unsupportedAsyncOp = kind;
      return std::nullopt;
    default:
      llvm_unreachable("unknown async op");
    }
  }

  return contribution;
}

bool directlyUsesSemaphore(Operation *operation, Value semaphore) {
  for (Value operand : operation->getOperands())
    if (operand == semaphore)
      return true;
  return false;
}

SmallVector<Operation *> getSemaphoreUsersInProgramOrder(SemaphoreCreateOp op) {
  SmallVector<Operation *> orderedUsers;
  Operation *scope = op->getParentOp();
  if (!scope) {
    for (Operation *user : op->getUsers())
      orderedUsers.push_back(user);
    return orderedUsers;
  }

  Value semaphore = op.getResult();
  scope->walk<WalkOrder::PreOrder>([&](Operation *candidate) {
    if (candidate == op.getOperation())
      return;
    if (directlyUsesSemaphore(candidate, semaphore))
      orderedUsers.push_back(candidate);
  });
  return orderedUsers;
}

} // namespace

SemaphorePendingCountAnalysis
analyzeSemaphorePendingCount(SemaphoreCreateOp op) {
  SemaphorePendingCountAnalysis analysis;
  // Compute pending count from release waves.  A wave is the set of releases on
  // this semaphore since the previous acquire on the same semaphore.  Multiple
  // partitions in one wave are true fan-in; releases separated by acquires are
  // sequential reuse of the same semaphore and must not be unioned together.
  llvm::DenseMap<int, int> globalPartitionContrib;
  llvm::DenseMap<int, int> currentWaveContrib;
  int currentWavePendingCount = 0;
  int pendingCount = 0;

  auto recordCurrentWave = [&]() {
    if (currentWavePendingCount == 0)
      return;
    pendingCount = std::max(pendingCount, currentWavePendingCount);
    currentWaveContrib.clear();
    currentWavePendingCount = 0;
  };

  for (Operation *user : getSemaphoreUsersInProgramOrder(op)) {
    if (isa<SemaphoreAcquireOp>(user)) {
      recordCurrentWave();
      continue;
    }

    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp || !gpu::hasPartition(user))
      continue;

    auto partitionIds = gpu::getPartitionIds(user);
    if (partitionIds.size() != 1) {
      // Pending-count analysis is per releasing partition. If the same
      // semaphore must be released from multiple partitions, model that with
      // multiple semaphore.release ops, each carrying one partition id, with
      // associated acquire token.
      analysis.invalidPartitionArity = partitionIds.size();
      return analysis;
    }

    std::optional<AsyncOp> unsupportedAsyncOp;
    auto contribution = getReleaseAsyncContribution(releaseOp.getAsyncOps(),
                                                    unsupportedAsyncOp);
    if (!contribution) {
      analysis.unsupportedAsyncOp = unsupportedAsyncOp;
      return analysis;
    }
    // First-class arrive multiplicity (absent = 1 for legacy producers):
    // with this folded in, the analysis sees exactly what the lowering
    // emits and pending-count verification is exact equality.
    if (auto countAttr = releaseOp.getArriveCountAttr())
      *contribution *= countAttr.getInt();

    int partitionId = partitionIds.front();
    auto [globalIt, globalInserted] =
        globalPartitionContrib.try_emplace(partitionId, contribution.value());
    // Repeated releases from the same partition are allowed only if they model
    // the same logical participant and therefore imply the same number of
    // arrivals.
    if (!globalInserted && globalIt->second != contribution.value()) {
      analysis.inconsistentPartitionId = partitionId;
      analysis.expectedContribution = globalIt->second;
      analysis.actualContribution = contribution.value();
      return analysis;
    }

    auto [waveIt, waveInserted] =
        currentWaveContrib.try_emplace(partitionId, contribution.value());
    (void)waveIt;
    if (waveInserted) {
      currentWavePendingCount += contribution.value();
      continue;
    }
  }
  recordCurrentWave();

  // Outside partitioned warp-specialized code there may be no partitioned
  // releases at all. Keep the historical fallback of one expected arrival.
  analysis.pendingCount = pendingCount == 0 ? 1 : pendingCount;
  return analysis;
}

} // namespace mlir::triton::nvws
