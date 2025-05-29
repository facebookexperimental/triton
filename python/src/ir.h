#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace ir {

namespace py = pybind11;
using namespace mlir;
using namespace triton;

// A custom op builder that keeps track of the last location
class TritonOpBuilder {
public:
  TritonOpBuilder(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }
  MLIRContext *getContext() { return builder->getContext(); }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  void setLastLoc(Location loc) {
    if (lineInfoEnabled)
      lastLoc = std::make_unique<Location>(loc);
  }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(FileLineColLoc::get(context, fileName, line, column));
  }

  Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
  bool lineInfoEnabled = !triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO");
};
} // namespace ir
