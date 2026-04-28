#ifndef NVIDIA_NVWS_TRANSFORMS_WS_UTILITY_H_
#define NVIDIA_NVWS_TRANSFORMS_WS_UTILITY_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::nvws {

using AsyncTaskId = int;

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);
void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks);

struct LoopScheduleInfo {
  IntegerAttr stage;
  IntegerAttr cluster;
};

class OpBuilderWithAsyncTaskIds : public OpBuilder {
public:
  OpBuilderWithAsyncTaskIds(MLIRContext *context) : OpBuilder(context) {}

  explicit OpBuilderWithAsyncTaskIds(Operation *op) : OpBuilder(op) {
    setAsyncTaskIdsFromOp(op);
    setLoopScheduleInfoFromOp(op);
  }

  void setAsynTaskIdsFromArray(ArrayRef<AsyncTaskId> newAsyncTaskIds) {
    asyncTaskIds =
        SmallVector<AsyncTaskId>(newAsyncTaskIds.begin(), newAsyncTaskIds.end());
  }

  void setAsyncTaskIdsFromOp(Operation *op) {
    setAsynTaskIdsFromArray(nvws::getAsyncTaskIds(op));
  }

  SmallVector<AsyncTaskId> getAsyncTaskIds() { return asyncTaskIds; }

  template <typename OpTy, typename... Args>
  OpTy createWithAsyncTaskIds(Args &&...args) {
    OpTy op = OpTy::create(*this, std::forward<Args>(args)...);
    if (!asyncTaskIds.empty())
      nvws::setAsyncTaskIds(op, asyncTaskIds);
    setOpLoopScheduleInfo(op);
    return op;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = createWithAsyncTaskIds<OpTy>(std::forward<Args>(args)...);
    setOpLoopScheduleInfo(op);
    return op;
  }

  void setLoopScheduleInfoFromInfo(LoopScheduleInfo newLoopScheduleInfo) {
    loopScheduleInfo = newLoopScheduleInfo;
  }

  void setLoopScheduleInfoFromOp(Operation *op);
  void clearLoopScheduleInfo() { loopScheduleInfo = {nullptr, nullptr}; }
  LoopScheduleInfo getLoopScheduleInfo() { return loopScheduleInfo; }

private:
  void setOpLoopScheduleInfo(Operation *op);

  SmallVector<AsyncTaskId> asyncTaskIds;
  LoopScheduleInfo loopScheduleInfo = {nullptr, nullptr};
};

Location appendToNameLoc(Location loc, StringRef suffix, MLIRContext *ctx);

} // namespace mlir::triton::nvws

#endif // NVIDIA_NVWS_TRANSFORMS_WS_UTILITY_H_
