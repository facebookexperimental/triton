#include "ReorderLoads.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

DenseMap<ttng::MMAv5OpInterface, int> determineMMAPriority(tt::FuncOp &funcOp) {
  DenseMap<ttng::MMAv5OpInterface, int> mmaPriority;
  return mmaPriority;
}

SmallVector<ttng::MMAv5OpInterface>
collectMMAUsers(tt::DescriptorLoadOp &loadOp) {
  SmallVector<ttng::MMAv5OpInterface> users;
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
    auto users = collectMMAUsers(loadOp);
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
