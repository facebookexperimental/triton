//===- TMAVerificationPass.cpp - TMA Regression Detection ------*- C++ -*-===//
//
// Triton-Lint - Static analysis tool for Triton kernels
//
//===----------------------------------------------------------------------===//
//
// This pass verifies that TMA (Tensor Memory Accelerator) operations were
// properly inserted by Triton's optimization passes. It detects regressions
// where TMA-eligible patterns are present but TMA operations are missing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "triton-lint/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

// Include Triton dialect headers
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

/**
*   **`[TRITON_LINT_JSON_INFO]`** - Informational events:

    *   `{"event": "started", "compute_capability": 90}`
    *   `{"event": "skipped", "reason": "Target is not Hopper+", ...}`
    *   `{"event": "no_cc_attr", "message": "...", "default_cc": 90}`
*   **`[TRITON_LINT_JSON_TMA_OP]`** - Each TMA operation found:

    *   `{"operation": "ttng.async_tma_copy_global_to_local"}`
*   **`[TRITON_LINT_JSON_SUMMARY]`** - Overall verification summary:

    *   `{"pass": "triton-lint-verify-tma", "status": "passed", "tma_ops_found":
9,
...}`
*   **`[TRITON_LINT_JSON_REGRESSION]`** - Regression detection:

    *   `{"type": "regression_detected", "message": "...", "tma_ops_found": 0,
...}`
*   **`[TRITON_LINT_JSON_DETAIL]`** - Details for each missed opportunity:

    *   `{"index": 0, "operation": "tt.load", "critical": true, "reason":
"..."}`
**/

#define DEBUG_TYPE "triton-lint-tma-verification"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton_lint {

namespace {

/// Information about a missed TMA opportunity
struct TMAOpportunity {
  Operation *op;
  std::string reason;
  bool isCritical; // Block pointer patterns that must use TMA
  size_t transferSize;
};

/// Pass to verify TMA usage
class TMAVerificationPass
    : public PassWrapper<TMAVerificationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TMAVerificationPass)

  StringRef getArgument() const override { return "triton-lint-verify-tma"; }

  StringRef getDescription() const override {
    return "Verify TMA operations were inserted (regression detection)";
  }

  /// Emit JSON summary remark
  void emitJSONSummary(ModuleOp module, int tmaOps, int regularOps,
                       const SmallVector<TMAOpportunity> &opportunities) const {
    int criticalMisses = 0;
    for (const auto &opp : opportunities) {
      if (opp.isCritical)
        criticalMisses++;
    }

    auto diag = module.emitRemark();
    diag << "[TRITON_LINT_JSON_SUMMARY] {";
    diag << "\"pass\": \"triton-lint-verify-tma\", ";
    diag << "\"status\": \"" << (opportunities.empty() ? "passed" : "failed")
         << "\", ";
    diag << "\"tma_ops_found\": " << tmaOps << ", ";
    diag << "\"regular_mem_ops\": " << regularOps << ", ";
    diag << "\"tma_eligible_ops\": " << opportunities.size() << ", ";
    diag << "\"critical_misses\": " << criticalMisses;
    diag << "}";
  }

  /// Emit JSON details for each missed opportunity
  void emitJSONDetails(const SmallVector<TMAOpportunity> &opportunities) const {
    for (size_t i = 0; i < opportunities.size(); ++i) {
      const auto &opp = opportunities[i];
      auto diag = opp.op->emitRemark();

      diag << "[TRITON_LINT_JSON_DETAIL] {";
      diag << "\"index\": " << i << ", ";
      diag << "\"operation\": \"" << opp.op->getName().getStringRef() << "\", ";
      diag << "\"critical\": " << (opp.isCritical ? "true" : "false") << ", ";
      diag << "\"transfer_size_bytes\": " << opp.transferSize << ", ";
      diag << "\"reason\": \"";
      // Escape quotes in reason
      for (char c : opp.reason) {
        if (c == '"')
          diag << "\\\"";
        else if (c == '\\')
          diag << "\\\\";
        else if (c == '\n')
          diag << "\\n";
        else
          diag << c;
      }
      diag << "\"";
      diag << "}";
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Get target compute capability
    int computeCapability = getComputeCapability(module);

    // TMA is only for Hopper+ (sm_90)
    if (computeCapability < 90) {
      auto diag = module.emitRemark();
      diag << "[TRITON_LINT_JSON_INFO] {";
      diag << "\"event\": \"skipped\", ";
      diag << "\"reason\": \"Target is not Hopper+\", ";
      diag << "\"compute_capability\": " << computeCapability << ", ";
      diag << "\"required_cc\": 90";
      diag << "}";
      return;
    }

    auto infoDiag = module.emitRemark();
    infoDiag << "[TRITON_LINT_JSON_INFO] {";
    infoDiag << "\"event\": \"started\", ";
    infoDiag << "\"compute_capability\": " << computeCapability;
    infoDiag << "}";

    // Collect statistics
    int tmaOps = 0;
    int regularMemOps = 0;
    SmallVector<TMAOpportunity> missedOpportunities;

    // Walk all operations
    module.walk([&](Operation *op) {
      // Count TMA operations
      if (isTMAOperation(op)) {
        tmaOps++;

        // Get function name if available
        std::string funcName = "unknown";
        if (auto funcOp = op->getParentOfType<triton::FuncOp>()) {
          funcName = funcOp.getName().str();
        }

        auto diag = op->emitRemark();
        diag << "[TRITON_LINT_JSON_TMA_OP] {";
        diag << "\"operation\": \"" << op->getName().getStringRef() << "\", ";
        diag << "\"function\": \"" << funcName << "\"";
        diag << "}";
        return;
      }

      // Check loads
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        regularMemOps++;

        if (shouldUseTMA(loadOp)) {
          TMAOpportunity opp;
          opp.op = op;
          opp.isCritical = isBlockPointerLoad(loadOp);
          opp.transferSize = getTransferSize(loadOp);
          opp.reason = buildLoadDiagnostic(opp.isCritical, opp.transferSize);
          missedOpportunities.push_back(opp);
        }
      }

      // Check stores
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        regularMemOps++;

        if (shouldUseTMA(storeOp)) {
          TMAOpportunity opp;
          opp.op = op;
          opp.isCritical = isBlockPointerStore(storeOp);
          opp.transferSize = getTransferSize(storeOp);
          opp.reason = buildStoreDiagnostic(opp.isCritical, opp.transferSize);
          missedOpportunities.push_back(opp);
        }
      }
    });

    // Report results - JSON summary is emitted below

    // Emit JSON summary remark
    emitJSONSummary(module, tmaOps, regularMemOps, missedOpportunities);

    if (!missedOpportunities.empty()) {
      // Emit detailed JSON for each missed opportunity
      emitJSONDetails(missedOpportunities);
      signalPassFailure();
    }
  }

private:
  /// Check if operation is a TMA operation
  bool isTMAOperation(Operation *op) const {
    StringRef opName = op->getName().getStringRef();
    return opName.contains("async_tma_copy") ||
           opName.contains("async_tma_reduce") ||
           opName.contains("async_tma_gather") ||
           opName.contains("async_tma_scatter") ||
           opName.contains("make_tma_desc");
  }

  /// Check if load should use TMA
  bool shouldUseTMA(triton::LoadOp loadOp) const {
    if (isBlockPointerLoad(loadOp)) {
      return true;
    }

    size_t transferSize = getTransferSize(loadOp);
    if (transferSize >= 65536) { // 64KB threshold
      return true;
    }

    return false;
  }

  /// Check if store should use TMA
  bool shouldUseTMA(triton::StoreOp storeOp) const {
    if (isBlockPointerStore(storeOp)) {
      return true;
    }

    size_t transferSize = getTransferSize(storeOp);
    if (transferSize >= 65536) {
      return true;
    }

    return false;
  }

  /// Check if load uses a block pointer
  bool isBlockPointerLoad(triton::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
    return isBlockPointer(ptr);
  }

  /// Check if store uses a block pointer
  bool isBlockPointerStore(triton::StoreOp storeOp) const {
    Value ptr = storeOp.getPtr();
    return isBlockPointer(ptr);
  }

  /// Check if value comes from make_block_ptr
  bool isBlockPointer(Value ptr) const {
    if (!ptr)
      return false;

    Operation *defOp = ptr.getDefiningOp();
    if (!defOp)
      return false;

    StringRef opName = defOp->getName().getStringRef();
    if (opName.contains("make_block_ptr"))
      return true;

    // Check through expand/broadcast operations
    if (opName.contains("expand") || opName.contains("broadcast")) {
      if (defOp->getNumOperands() > 0) {
        return isBlockPointer(defOp->getOperand(0));
      }
    }

    return false;
  }

  /// Get transfer size for load
  size_t getTransferSize(triton::LoadOp loadOp) const {
    Type resultType = loadOp.getResult().getType();
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resultType)) {
      return calculateTransferSize(tensorType);
    }
    return 0;
  }

  /// Get transfer size for store
  size_t getTransferSize(triton::StoreOp storeOp) const {
    Type valueType = storeOp.getValue().getType();
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(valueType)) {
      return calculateTransferSize(tensorType);
    }
    return 0;
  }

  /// Calculate transfer size in bytes
  size_t calculateTransferSize(RankedTensorType type) const {
    Type elementType = type.getElementType();
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    size_t elementSize = (bitWidth + 7) / 8;

    size_t numElements = 1;
    for (auto dim : type.getShape()) {
      if (dim > 0)
        numElements *= dim;
    }

    return elementSize * numElements;
  }

  /// Build diagnostic for load
  std::string buildLoadDiagnostic(bool isCritical, size_t transferSize) const {
    std::string msg;

    if (isCritical) {
      msg = "CRITICAL: Block pointer load should use TMA but found tt.load. ";
      msg +=
          "This is a primary TMA target pattern that must use TMA on Hopper.";
    } else {
      msg = "Large transfer (";
      msg += std::to_string(transferSize / 1024);
      msg += " KB) eligible for TMA but using regular load.";
    }

    return msg;
  }

  /// Build diagnostic for store
  std::string buildStoreDiagnostic(bool isCritical, size_t transferSize) const {
    std::string msg;

    if (isCritical) {
      msg = "CRITICAL: Block pointer store should use TMA but found tt.store.";
    } else {
      msg = "Large transfer (";
      msg += std::to_string(transferSize / 1024);
      msg += " KB) eligible for TMA but using regular store.";
    }

    return msg;
  }

  /// Get compute capability from module
  int getComputeCapability(ModuleOp module) const {
    if (auto ccAttr =
            module->getAttrOfType<IntegerAttr>("nvidia.compute_capability")) {
      return ccAttr.getInt();
    }

    if (auto ccAttr = module->getAttrOfType<IntegerAttr>(
            "triton_gpu.compute_capability")) {
      return ccAttr.getInt();
    }

    // No compute capability found - emit info and default to 90
    auto diag = module.emitRemark();
    diag << "[TRITON_LINT_JSON_INFO] {";
    diag << "\"event\": \"no_cc_attr\", ";
    diag << "\"message\": \"No compute_capability attribute found\", ";
    diag << "\"default_cc\": 90";
    diag << "}";
    return 90;
  }

  /// Emit regression report as JSON remarks
  void emitRegressionReport(ModuleOp module,
                            const SmallVector<TMAOpportunity> &opportunities,
                            int tmaOps, int regularOps) const {
    int criticalMisses = 0;
    for (const auto &opp : opportunities) {
      if (opp.isCritical)
        criticalMisses++;
    }

    // Emit module-level regression summary as remark
    auto diag = module.emitRemark();
    diag << "[TRITON_LINT_JSON_REGRESSION] {";
    diag << "\"type\": \"regression_detected\", ";
    diag << "\"message\": \"TMA-eligible operations using regular memory ops "
            "instead of TMA\", ";
    diag << "\"tma_ops_found\": " << tmaOps << ", ";
    diag << "\"regular_mem_ops\": " << regularOps << ", ";
    diag << "\"tma_eligible_ops\": " << opportunities.size() << ", ";
    diag << "\"critical_misses\": " << criticalMisses;
    diag << "}";
  }
};

} // namespace

/// Create the TMA verification pass
std::unique_ptr<Pass> createTMAVerificationPass() {
  return std::make_unique<TMAVerificationPass>();
}

/// Register all Triton-Lint passes
void registerTritonLintPasses() { PassRegistration<TMAVerificationPass>(); }

} // namespace triton_lint
} // namespace mlir

// External registration function for dynamic loading
extern "C" {

LLVM_ATTRIBUTE_WEAK void registerPasses() {
  mlir::triton_lint::registerTritonLintPasses();
}

// MLIR plugin entry point
LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Triton-Lint", "v0.1",
          []() { mlir::triton_lint::registerTritonLintPasses(); }};
}

} // extern "C"
