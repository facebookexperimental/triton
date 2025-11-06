#include "ReorderLoads.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

DenseMap<ttng::MMAv5OpInterface, int> determineMMAPriority(tt::FuncOp &funcOp) {
  DenseMap<ttng::MMAv5OpInterface, int> mmaPriority;
  // Collect all the MMA ops. Right now we require every MMA op be annotated
  // with LoopSchedule information.
  SmallVector<ttng::MMAv5OpInterface> allMMAs;
  bool isSupported = true;
  funcOp.walk([&](ttng::MMAv5OpInterface mmaOp) {
    if (!mmaOp->hasAttr(tt::kLoopStageAttrName) ||
        !mmaOp->hasAttr(tt::kLoopClusterAttrName)) {
      isSupported = false;
      return;
    }
    allMMAs.push_back(mmaOp);
  });
  llvm::sort(allMMAs, [&](ttng::MMAv5OpInterface a, ttng::MMAv5OpInterface b) {
    if (a->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName).getInt() <
        b->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName).getInt()) {
      return true;
    } else if (a->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName)
                   .getInt() <
               b->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName).getInt()) {
      return true;
    } else {
      if (a->getBlock() != b->getBlock()) {
        isSupported = false;
        // Doesn't matter. Will abort.
        return true;
      } else {
        return a->isBeforeInBlock(b);
      }
    }
  });
  if (isSupported) {
    for (int i = 0; i < allMMAs.size(); ++i) {
      mmaPriority[allMMAs[i]] = i;
    }
  }
  return mmaPriority;
}

// Collect all MMA users of a given OP.
// When you have OP -> MMA -> MMA only the first MMA is included.
void collectOpInitialMMAUsers(Operation *op,
                              SmallVector<ttng::MMAv5OpInterface> &users) {
  for (auto user : op->getUsers()) {
    if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      users.push_back(mma);
    } else {
      collectOpInitialMMAUsers(user, users);
    }
  }
}

// Collect all the MMA users derived from the tt::DescriptorLoadOp.
// When you have load -> MMA -> MMA only the first MMA is included.
SmallVector<ttng::MMAv5OpInterface>
collectInitialMMAUsers(tt::DescriptorLoadOp &loadOp) {
  SmallVector<ttng::MMAv5OpInterface> users;
  collectOpInitialMMAUsers(loadOp, users);
  return users;
}

void annotateLoads(tt::FuncOp &funcOp) {
  // Determine the relative order of each MMA op using the
  // pipelining information + program order to break ties.
  auto mmaPriority = determineMMAPriority(funcOp);
  funcOp.walk([&](tt::DescriptorLoadOp loadOp) {
    if (loadOp->hasAttr(kMMAPriorityAttr)) {
      return;
    }
    // Assign a load to its earliest user.
    auto users = collectInitialMMAUsers(loadOp);
    int priority = mmaPriority.size();
    for (auto &user : users) {
      if (!mmaPriority.count(user)) {
        // If any MMA is unknown we disable reordering
        // those sections.
        priority = -1;
      } else {
        priority = std::min(priority, mmaPriority[user]);
      }
    }
    if (priority == mmaPriority.size()) {
      priority = -1;
    }
    // Give each load a priority.
    loadOp->setAttr(
        kMMAPriorityAttr,
        IntegerAttr::get(IntegerType::get(loadOp->getContext(), 32), priority));
  });
}

} // namespace mlir
