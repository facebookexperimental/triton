//===- CudaWarningsPass.cpp - CUDA target-specific warnings pass ---------===//
//
// Emits warnings for performance-impacting patterns on specific CUDA GPUs.
//
// Currently warns on FP64 math operations for GB300 (SM103), which has 1/28th
// the FP64 throughput of GB200.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CUDAWARNINGS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

/// Check if a type is or contains f64.
static bool containsF64(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.isF64();
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsF64(shapedType.getElementType());
  return false;
}

/// Check if an operation has any f64 operands or results.
static bool hasF64OperandOrResult(Operation *op) {
  for (Type type : op->getResultTypes())
    if (containsF64(type))
      return true;
  for (Value operand : op->getOperands())
    if (containsF64(operand.getType()))
      return true;
  return false;
}

/// Check if an operation is an FP64 math operation.
static bool isFP64MathOp(Operation *op) {
  if (!hasF64OperandOrResult(op))
    return false;

  // Arith dialect floating-point operations that implement
  // ArithFastMathInterface are FP math ops, but we exclude casts (ExtFOp,
  // TruncFOp, etc.) which implement the interface for fastmath propagation but
  // aren't compute ops.
  if (isa<arith::ArithFastMathInterface>(op) && !isa<CastOpInterface>(op))
    return true;

  // Math dialect operations (exp, sin, cos, sqrt, fma, etc.)
  if (op->getDialect() && op->getDialect()->getNamespace() == "math")
    return true;

  // Triton compute operations
  if (isa<triton::DotOp, triton::ReduceOp, triton::ScanOp>(op))
    return true;

  return false;
}

/// Check if a function name is a Triton builtin/internal function.
static bool isBuiltinFunction(llvm::StringRef funcName) {
  return funcName.starts_with("triton.");
}

/// Get the parent function of an operation by recursively walking up parents.
static std::string getParentFunctionName(Operation *op) {
  triton::FuncOp func = op->getParentOfType<triton::FuncOp>();
  if (func) {
    return func.getName().str();
  } else {
    return "";
  }
}

/// Format function names from a set into a comma-separated string.
static std::string formatFunctionNames(const llvm::StringSet<> &funcNames) {
  if (funcNames.empty())
    return "Kernel";

  llvm::SmallVector<llvm::StringRef, 4> names;
  for (const auto &entry : funcNames)
    names.push_back(entry.getKey());

  // Sort for deterministic output
  llvm::sort(names);

  if (names.size() == 1)
    return names[0].str();

  // Multiple kernels - join with commas
  std::string result;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      result += ", ";
    result += names[i].str();
  }
  return result;
}

/// Collect FP64 performance warnings for a module.
/// Returns a vector of warning messages (empty if no warnings).
static std::vector<std::string>
collectFloat64PerformanceWarnings(ModuleOp module) {
  std::vector<std::string> warnings;
  llvm::StringSet<> functionsWithFP64Math;

  module.walk([&](Operation *op) {
    if (isFP64MathOp(op)) {
      std::string funcName = getParentFunctionName(op);
      if (!funcName.empty() && !isBuiltinFunction(funcName)) {
        functionsWithFP64Math.insert(funcName);
      }
    }
    return WalkResult::advance();
  });

  if (!functionsWithFP64Math.empty()) {
    std::string kernelNames = formatFunctionNames(functionsWithFP64Math);
    std::string message =
        "PERFORMANCE WARNING: " + kernelNames +
        " contains FP64 (double-precision) math operations on a "
        "GB300 GPU. FP64 math throughput was reduced significantly on "
        "GB300 (1/28th GB200 throughput). Consider using tl.float32 for "
        "compute-intensive operations, possibly with algorithmic changes "
        "for numeric stability.";
    warnings.push_back(message);
  }

  return warnings;
}

struct CudaWarningsPass
    : public mlir::triton::impl::CudaWarningsBase<CudaWarningsPass> {
  using CudaWarningsBase::CudaWarningsBase;

  // Pass is defined solely for lit test integration. Use
  // collectCudaWarnings directly from Python in the compiler.

  void runOnOperation() override {
    if (computeCapability == 103) {
      auto warnings = collectFloat64PerformanceWarnings(getOperation());
      for (const auto &message : warnings) {
        getOperation().emitWarning() << message;
      }
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createCudaWarningsPass(int32_t computeCapability) {
  return std::make_unique<CudaWarningsPass>(
      CudaWarningsOptions{computeCapability});
}

std::vector<std::string> collectCudaWarnings(ModuleOp module,
                                             int32_t computeCapability) {
  std::vector<std::string> warnings;
  if (computeCapability == 103) {
    warnings = collectFloat64PerformanceWarnings(module);
  }
  return warnings;
}

} // namespace mlir::triton
