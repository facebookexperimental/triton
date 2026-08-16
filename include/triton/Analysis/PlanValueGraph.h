#ifndef TRITON_ANALYSIS_PLANVALUEGRAPH_H
#define TRITON_ANALYSIS_PLANVALUEGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
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

/// A stable structured block and its direct operation order. Operation
/// positions are local to the block; they are not machine cycles.
struct PlanBlockRecord {
  std::string id;
  std::string parentOperationId;
  int64_t regionNumber = -1;
  int64_t blockNumber = -1;
  std::vector<std::string> operations;
};

/// Half-open static TTGIR program-order interval within one block.
struct PlanLiveSegment {
  std::string valueId;
  std::string blockId;
  std::string startOperationId;
  std::string endOperationId;
  int64_t startPosition = 0;
  int64_t endPosition = 0;
  bool liveIn = false;
  bool liveOut = false;
  bool crossesBackedge = false;
  int64_t iterationDistance = 0;
};

/// Normalized index selecting one logical slot of an LDS allocation.
struct PlanSlotExpression {
  std::string kind;
  std::string text;
  std::string baseValueId;
  int64_t coefficient = 0;
  int64_t offset = 0;
  int64_t modulus = 0;
  std::vector<int64_t> possibleSlots;
};

struct PlanSlotPath {
  std::string rootValueId;
  std::vector<PlanSlotExpression> indices;
};

/// Logical relationship between a memdesc value and local_alloc roots.
struct PlanAliasRecord {
  std::string valueId;
  std::string viewKind;
  std::string sourceValueId;
  std::vector<std::string> rootValueIds;
  std::vector<int64_t> staticOffsets;
  std::vector<int64_t> order;
  std::vector<PlanSlotPath> slotPaths;
};

struct PlanMemoryAccess {
  std::string operationId;
  std::string valueId;
  std::string effect;
  bool pendingAsync = false;
  std::vector<std::string> rootValueIds;
  std::vector<PlanSlotPath> slotPaths;
};

struct PlanLdsAllocationRecord {
  std::string rootValueId;
  std::string allocationOperationId;
  std::optional<int64_t> logicalBytes;
  std::optional<int64_t> alignment;
  std::vector<std::string> aliases;
  std::vector<PlanLiveSegment> liveSegments;
};

struct PlanLivenessResult {
  std::vector<PlanBlockRecord> blocks;
  std::vector<PlanLiveSegment> liveSegments;
  std::vector<PlanAliasRecord> aliases;
  std::vector<PlanMemoryAccess> memoryAccesses;
  std::vector<PlanLdsAllocationRecord> ldsAllocations;
  std::vector<PlanDiagnostic> diagnostics;
};

/// A static TTGIR program point that participates in an asynchronous LDS
/// lifetime. iterationDistance is a dynamic structured-loop distance, not a
/// cycle count.
struct PlanAsyncFrontier {
  std::string operationId;
  std::string blockId;
  std::string kind;
  std::string precision;
  int64_t position = -1;
  int64_t iterationDistance = 0;
};

struct PlanAsyncTransaction {
  std::string id;
  std::string producerOperationId;
  std::string destinationValueId;
  std::string direction;
  std::string commitGroupId;
  std::string precision;
  std::vector<std::string> rootValueIds;
  std::vector<PlanSlotPath> slotPaths;
  std::vector<PlanAsyncFrontier> completionFrontiers;
  std::vector<PlanAsyncFrontier> visibilityFrontiers;
  std::vector<PlanAsyncFrontier> consumerFrontiers;
  std::vector<PlanAsyncFrontier> releaseFrontiers;
  std::vector<PlanAsyncFrontier> overwriteFrontiers;
};

struct PlanAsyncGroup {
  std::string id;
  std::string commitOperationId;
  std::vector<std::string> transactionIds;
};

struct PlanAsyncWaitCompletion {
  std::string groupId;
  int64_t iterationDistance = 0;
  std::string precision;
};

struct PlanAsyncWaitRecord {
  std::string operationId;
  int64_t retainedGroupCount = 0;
  std::vector<PlanAsyncWaitCompletion> completedGroups;
  std::vector<std::string> possiblyOutstandingGroups;
  std::string precision;
};

struct PlanLdsReuseHazard {
  std::string severity;
  std::string code;
  std::string message;
  std::string transactionId;
  std::string operationId;
  std::string rootValueId;
};

struct PlanAsyncLifetimeResult {
  std::vector<PlanAsyncTransaction> transactions;
  std::vector<PlanAsyncGroup> groups;
  std::vector<PlanAsyncWaitRecord> waits;
  std::vector<PlanLdsReuseHazard> hazards;
  std::vector<PlanDiagnostic> diagnostics;
};

/// A semantic dependency between stable TTGIR operations. The distance is a
/// structured-loop iteration distance, not a latency or cycle count.
struct PlanDependencyEdge {
  std::string id;
  std::string kind;
  std::string sourceOperationId;
  std::string destinationOperationId;
  std::string sourceValueId;
  std::string destinationValueId;
  std::string rootValueId;
  std::string precision;
  std::string reason;
  int64_t iterationDistance = 0;
};

/// Maximum block-local overlap at one static TTGIR program point. Logical
/// tensor bytes are whole distributed tensor bytes, not per-wave VGPR bytes.
struct PlanPeakLiveSet {
  std::string blockId;
  std::string operationId;
  int64_t position = 0;
  int64_t logicalTensorBytes = 0;
  int64_t unknownTensorValueCount = 0;
  int64_t logicalLdsBytes = 0;
  std::vector<std::string> tensorValueIds;
  std::vector<std::string> ldsRootValueIds;
};

struct PlanResourceSummary {
  int64_t logicalLdsAllocationBytes = 0;
  int64_t unknownLdsAllocations = 0;
  int64_t peakLogicalTensorBytes = 0;
  int64_t peakLogicalTensorCount = 0;
  int64_t peakUnknownTensorValueCount = 0;
  int64_t peakLogicalLdsBytes = 0;
  int64_t maxAsyncIterationDistance = 0;
  int64_t maxPossiblyOutstandingGroups = 0;
  int64_t maxLogicalSlotDepth = 1;
  std::map<std::string, int64_t> pipelineClassCounts;
};

struct PlanUnresolvedFact {
  std::string severity;
  std::string code;
  std::string message;
  std::string operationId;
  std::string valueId;
  std::string importance;
  std::string status;
};

struct PlanAuditAnalysisResult {
  std::vector<PlanDependencyEdge> dependencies;
  std::vector<PlanPeakLiveSet> peakLiveSets;
  PlanResourceSummary resources;
  std::vector<PlanUnresolvedFact> unresolvedFacts;
  std::vector<PlanDiagnostic> diagnostics;
};

FailureOr<PlanLivenessResult> analyzePlanLiveness(
    FuncOp function,
    const llvm::DenseMap<Operation *, std::string> &operationIds,
    const llvm::DenseMap<Value, std::string> &valueIds,
    ArrayRef<PlanLineageEdge> lineageEdges);

FailureOr<PlanAsyncLifetimeResult> analyzePlanAsyncLifetimes(
    FuncOp function,
    const llvm::DenseMap<Operation *, std::string> &operationIds,
    const llvm::DenseMap<Value, std::string> &valueIds,
    ArrayRef<PlanBlockRecord> blocks, ArrayRef<PlanAliasRecord> aliases,
    ArrayRef<PlanMemoryAccess> memoryAccesses);

FailureOr<PlanAuditAnalysisResult> analyzePlanAudit(
    ArrayRef<PlanOperationRecord> operations, ArrayRef<PlanValueRecord> values,
    ArrayRef<PlanLineageEdge> lineageEdges, ArrayRef<PlanBlockRecord> blocks,
    ArrayRef<PlanLiveSegment> liveSegments,
    ArrayRef<PlanMemoryAccess> memoryAccesses,
    ArrayRef<PlanLdsAllocationRecord> ldsAllocations,
    ArrayRef<PlanAsyncTransaction> asyncTransactions,
    ArrayRef<PlanAsyncGroup> asyncGroups,
    ArrayRef<PlanAsyncWaitRecord> asyncWaits,
    ArrayRef<PlanLdsReuseHazard> ldsReuseHazards,
    ArrayRef<PlanDiagnostic> diagnostics);

/// Analysis-only description of operation/value identity, value lineage,
/// static program-order liveness, logical LDS aliases, asynchronous lifetime,
/// semantic dependencies, and logical overlap for one final structured TTGIR
/// function. It deliberately does not model physical register allocation,
/// physical LDS offsets, or hardware-cycle lifetime.
class PlanValueGraph {
public:
  /// Build the stable graph. When requested, retain the transient mapping from
  /// live IR operations to their stable IDs for an immediately following
  /// analysis or mutation pass.
  static FailureOr<PlanValueGraph>
  build(FuncOp function,
        llvm::DenseMap<Operation *, std::string> *operationBindings = nullptr);

  LogicalResult verify(bool strict = false) const;
  llvm::json::Object toJSON() const;

  StringRef getFunctionName() const { return functionName; }
  StringRef getSemanticFingerprint() const { return semanticFingerprint; }
  ArrayRef<PlanOperationRecord> getOperations() const { return operations; }
  ArrayRef<PlanValueRecord> getValues() const { return values; }
  ArrayRef<PlanLineageEdge> getLineageEdges() const { return lineageEdges; }
  ArrayRef<PlanDiagnostic> getDiagnostics() const { return diagnostics; }
  ArrayRef<PlanBlockRecord> getBlocks() const { return blocks; }
  ArrayRef<PlanLiveSegment> getLiveSegments() const { return liveSegments; }
  ArrayRef<PlanAliasRecord> getAliases() const { return aliases; }
  ArrayRef<PlanMemoryAccess> getMemoryAccesses() const {
    return memoryAccesses;
  }
  ArrayRef<PlanLdsAllocationRecord> getLdsAllocations() const {
    return ldsAllocations;
  }
  ArrayRef<PlanAsyncTransaction> getAsyncTransactions() const {
    return asyncTransactions;
  }
  ArrayRef<PlanAsyncGroup> getAsyncGroups() const { return asyncGroups; }
  ArrayRef<PlanAsyncWaitRecord> getAsyncWaits() const { return asyncWaits; }
  ArrayRef<PlanLdsReuseHazard> getLdsReuseHazards() const {
    return ldsReuseHazards;
  }
  ArrayRef<PlanDependencyEdge> getDependencyEdges() const {
    return dependencyEdges;
  }
  ArrayRef<PlanPeakLiveSet> getPeakLiveSets() const { return peakLiveSets; }
  const PlanResourceSummary &getResourceSummary() const {
    return resourceSummary;
  }
  ArrayRef<PlanUnresolvedFact> getUnresolvedFacts() const {
    return unresolvedFacts;
  }

private:
  std::string functionName;
  std::string semanticFingerprint;
  std::vector<PlanOperationRecord> operations;
  std::vector<PlanValueRecord> values;
  std::vector<PlanLineageEdge> lineageEdges;
  std::vector<PlanDiagnostic> diagnostics;
  std::vector<PlanBlockRecord> blocks;
  std::vector<PlanLiveSegment> liveSegments;
  std::vector<PlanAliasRecord> aliases;
  std::vector<PlanMemoryAccess> memoryAccesses;
  std::vector<PlanLdsAllocationRecord> ldsAllocations;
  std::vector<PlanAsyncTransaction> asyncTransactions;
  std::vector<PlanAsyncGroup> asyncGroups;
  std::vector<PlanAsyncWaitRecord> asyncWaits;
  std::vector<PlanLdsReuseHazard> ldsReuseHazards;
  std::vector<PlanDependencyEdge> dependencyEdges;
  std::vector<PlanPeakLiveSet> peakLiveSets;
  PlanResourceSummary resourceSummary;
  std::vector<PlanUnresolvedFact> unresolvedFacts;
};

/// Serialize a module-level sidecar. The graphs are sorted by function name;
/// each graph is already deterministically sorted by stable ID.
std::string serializePlanValueGraphs(
    ArrayRef<PlanValueGraph> graphs, ModuleOp module,
    StringRef passPosition = "after_warp_pipeline_conversion_before_scf_to_cf");

} // namespace plan
} // namespace mlir::triton

#endif // TRITON_ANALYSIS_PLANVALUEGRAPH_H
