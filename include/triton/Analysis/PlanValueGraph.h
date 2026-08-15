#ifndef TRITON_ANALYSIS_PLANVALUEGRAPH_H
#define TRITON_ANALYSIS_PLANVALUEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton {
class FuncOp;

namespace plan {

struct PlanUse {
  std::string operationId;
  unsigned operandNumber = 0;
};

struct PlanOperationRecord {
  std::string id;
  std::string kind;
  std::string locator;
  std::string identityQuality;
  unsigned ordinal = 0;
  unsigned resultCount = 0;
};

struct PlanValueRecord {
  std::string id;
  std::string locator;
  std::string identityQuality;
  std::string category;
  std::string type;
  std::string originKind;
  std::string definingOperationId;
  int64_t resultNumber = -1;
  int64_t argumentNumber = -1;
  std::string argumentRole;
  std::vector<int64_t> logicalShape;
  std::string elementType;
  std::string encoding;
  std::optional<int64_t> logicalBytes;
  std::vector<PlanUse> uses;
};

struct PlanLineageEdge {
  std::string source;
  std::string destination;
  std::string kind;
  int64_t iterationDistance = 0;
  std::map<std::string, std::string> stringParameters;
  std::map<std::string, int64_t> integerParameters;
};

struct PlanDiagnostic {
  std::string severity;
  std::string code;
  std::string message;
  std::string operationId;
  std::string valueId;
};

/// Analysis-only description of operation/value identity and value lineage for
/// one final structured TTGIR function. It deliberately does not model memory
/// hazards, liveness, or physical register allocation.
class PlanValueGraph {
public:
  static FailureOr<PlanValueGraph> build(FuncOp function);

  LogicalResult verify(bool strict = false) const;
  llvm::json::Object toJSON() const;

  StringRef getFunctionName() const { return functionName; }
  StringRef getSemanticFingerprint() const { return semanticFingerprint; }
  ArrayRef<PlanOperationRecord> getOperations() const { return operations; }
  ArrayRef<PlanValueRecord> getValues() const { return values; }
  ArrayRef<PlanLineageEdge> getLineageEdges() const { return lineageEdges; }
  ArrayRef<PlanDiagnostic> getDiagnostics() const { return diagnostics; }

private:
  std::string functionName;
  std::string semanticFingerprint;
  std::vector<PlanOperationRecord> operations;
  std::vector<PlanValueRecord> values;
  std::vector<PlanLineageEdge> lineageEdges;
  std::vector<PlanDiagnostic> diagnostics;
};

/// Serialize a module-level sidecar. The graphs are sorted by function name;
/// each graph is already deterministically sorted by stable ID.
std::string serializePlanValueGraphs(ArrayRef<PlanValueGraph> graphs,
                                     ModuleOp module);

} // namespace plan
} // namespace mlir::triton

#endif // TRITON_ANALYSIS_PLANVALUEGRAPH_H
