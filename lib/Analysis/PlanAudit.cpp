#include "triton/Analysis/PlanValueGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

namespace mlir::triton::plan {
namespace {

struct OperationPoint {
  std::string blockId;
  int64_t position = -1;
  bool inLoop = false;
};

enum class AliasRelation { Disjoint, Same, MayAlias };

static AliasRelation compareExpression(const PlanSlotExpression &lhs,
                                       const PlanSlotExpression &rhs) {
  if (lhs.text == rhs.text)
    return AliasRelation::Same;
  if (!lhs.possibleSlots.empty() && !rhs.possibleSlots.empty()) {
    bool intersects = llvm::any_of(lhs.possibleSlots, [&](int64_t slot) {
      return llvm::is_contained(rhs.possibleSlots, slot);
    });
    if (!intersects)
      return AliasRelation::Disjoint;
  }
  if (lhs.kind == "constant" && rhs.kind == "constant")
    return AliasRelation::Disjoint;
  return AliasRelation::MayAlias;
}

static AliasRelation comparePath(const PlanSlotPath &lhs,
                                 const PlanSlotPath &rhs) {
  if (lhs.rootValueId != rhs.rootValueId)
    return AliasRelation::Disjoint;
  AliasRelation result = AliasRelation::Same;
  size_t common = std::min(lhs.indices.size(), rhs.indices.size());
  for (size_t index = 0; index < common; ++index) {
    AliasRelation relation =
        compareExpression(lhs.indices[index], rhs.indices[index]);
    if (relation == AliasRelation::Disjoint)
      return relation;
    if (relation == AliasRelation::MayAlias)
      result = relation;
  }
  if (lhs.indices.size() != rhs.indices.size())
    result = AliasRelation::MayAlias;
  return result;
}

static AliasRelation comparePaths(ArrayRef<PlanSlotPath> lhs,
                                  ArrayRef<PlanSlotPath> rhs) {
  if (lhs.empty() || rhs.empty())
    return AliasRelation::MayAlias;
  AliasRelation result = AliasRelation::Disjoint;
  for (const PlanSlotPath &left : lhs)
    for (const PlanSlotPath &right : rhs) {
      AliasRelation relation = comparePath(left, right);
      if (relation == AliasRelation::Same)
        return relation;
      if (relation == AliasRelation::MayAlias)
        result = relation;
    }
  return result;
}

static std::string commonRoot(const PlanMemoryAccess &lhs,
                              const PlanMemoryAccess &rhs) {
  for (StringRef root : lhs.rootValueIds)
    if (llvm::is_contained(rhs.rootValueIds, root))
      return root.str();
  return "";
}

static std::string memoryKind(StringRef source, StringRef destination) {
  if (source == "write" && destination == "read")
    return "memory_raw";
  if (source == "read" && destination == "write")
    return "memory_war";
  if (source == "write" && destination == "write")
    return "memory_waw";
  return "";
}

static std::string pipelineClass(StringRef kind) {
  if (kind.contains("scheduled_mfma") || kind == "tt.dot" ||
      kind.contains("wmma"))
    return "mfma";
  if (kind.contains("barrier") || kind.contains("async_wait") ||
      kind.contains("async_commit"))
    return "synchronization";
  if (kind.contains("local_") || kind.contains("memdesc") ||
      kind.contains("lds"))
    return "lds";
  if ((kind.contains("load") || kind.contains("store")) &&
      !kind.contains("local"))
    return "global";
  if (kind.starts_with("arith.") || kind.starts_with("math.") ||
      kind.starts_with("tt.reduce") || kind.starts_with("tt.scan"))
    return "valu";
  return "other";
}

static std::optional<int64_t> slotReusePeriod(const PlanMemoryAccess &access) {
  if (access.slotPaths.empty())
    return std::nullopt;
  int64_t period = 1;
  for (const PlanSlotPath &path : access.slotPaths)
    for (const PlanSlotExpression &index : path.indices) {
      if (index.kind == "unknown" ||
          (index.kind == "induction" && index.modulus == 0))
        return std::nullopt;
      if (index.modulus > 0) {
        int64_t coefficient = std::abs(index.coefficient);
        period = std::lcm(period,
                          index.modulus / std::gcd(coefficient, index.modulus));
      }
    }
  return period;
}

class AuditAnalysis {
public:
  AuditAnalysis(ArrayRef<PlanOperationRecord> operations,
                ArrayRef<PlanValueRecord> values,
                ArrayRef<PlanLineageEdge> lineageEdges,
                ArrayRef<PlanBlockRecord> blocks,
                ArrayRef<PlanLiveSegment> liveSegments,
                ArrayRef<PlanMemoryAccess> memoryAccesses,
                ArrayRef<PlanLdsAllocationRecord> ldsAllocations,
                ArrayRef<PlanAsyncTransaction> asyncTransactions,
                ArrayRef<PlanAsyncGroup> asyncGroups,
                ArrayRef<PlanAsyncWaitRecord> asyncWaits,
                ArrayRef<PlanLdsReuseHazard> ldsReuseHazards,
                ArrayRef<PlanDiagnostic> diagnostics)
      : operations(operations), values(values), lineageEdges(lineageEdges),
        blocks(blocks), liveSegments(liveSegments),
        memoryAccesses(memoryAccesses), ldsAllocations(ldsAllocations),
        asyncTransactions(asyncTransactions), asyncGroups(asyncGroups),
        asyncWaits(asyncWaits), ldsReuseHazards(ldsReuseHazards),
        diagnostics(diagnostics) {}

  PlanAuditAnalysisResult run() {
    indexRecords();
    addSsaDependencies();
    addMemoryDependencies();
    addAsyncDependencies();
    computePeaksAndResources();
    classifyUnresolvedFacts();
    sortResult();
    return std::move(result);
  }

private:
  void indexRecords() {
    for (const PlanOperationRecord &operation : operations) {
      operationById[operation.id] = &operation;
      ++result.resources.pipelineClassCounts[pipelineClass(operation.kind)];
    }
    for (const PlanValueRecord &value : values)
      valueById[value.id] = &value;
    for (const PlanBlockRecord &block : blocks) {
      const PlanOperationRecord *parent =
          operationById[block.parentOperationId];
      bool inLoop =
          parent && (parent->kind == "scf.for" || parent->kind == "scf.while");
      for (auto [position, operation] : llvm::enumerate(block.operations))
        points[operation] = {block.id, static_cast<int64_t>(position), inLoop};
    }
    for (const PlanAsyncGroup &group : asyncGroups)
      groupById[group.id] = &group;
  }

  void addDependency(PlanDependencyEdge edge) {
    std::string key = edge.kind + "|" + edge.sourceOperationId + "|" +
                      edge.destinationOperationId + "|" + edge.sourceValueId +
                      "|" + edge.destinationValueId + "|" + edge.rootValueId +
                      "|" + std::to_string(edge.iterationDistance) + "|" +
                      edge.precision;
    if (!dependencyKeys.insert(key).second)
      return;
    edge.id = "dependency:" + key;
    result.dependencies.push_back(std::move(edge));
  }

  void addSsaDependencies() {
    for (const PlanValueRecord &value : values) {
      if (value.definingOperationId.empty())
        continue;
      for (const PlanUse &use : value.uses)
        addDependency({"", "ssa", value.definingOperationId, use.operationId,
                       value.id, value.id, "", "exact", "direct SSA def-use",
                       0});
    }

    for (const PlanLineageEdge &lineage : lineageEdges) {
      if (lineage.iterationDistance <= 0)
        continue;
      const PlanValueRecord *source = valueById[lineage.source];
      const PlanValueRecord *destination = valueById[lineage.destination];
      if (!source || !destination || source->definingOperationId.empty())
        continue;
      for (const PlanUse &use : destination->uses)
        addDependency({"", "loop_carried_ssa", source->definingOperationId,
                       use.operationId, source->id, destination->id, "",
                       "structured_exact", "scf iter_arg/yield backedge",
                       lineage.iterationDistance});
    }
  }

  void addMemoryDependencies() {
    for (size_t leftIndex = 0; leftIndex < memoryAccesses.size(); ++leftIndex) {
      const PlanMemoryAccess &left = memoryAccesses[leftIndex];
      if (left.effect != "read" && left.effect != "write")
        continue;
      auto leftPoint = points.find(left.operationId);
      if (leftPoint == points.end())
        continue;
      for (size_t rightIndex = 0; rightIndex < memoryAccesses.size();
           ++rightIndex) {
        if (leftIndex == rightIndex)
          continue;
        const PlanMemoryAccess &right = memoryAccesses[rightIndex];
        std::string kind = memoryKind(left.effect, right.effect);
        if (kind.empty() || left.operationId == right.operationId)
          continue;
        auto rightPoint = points.find(right.operationId);
        if (rightPoint == points.end() ||
            leftPoint->second.blockId != rightPoint->second.blockId)
          continue;
        std::string root = commonRoot(left, right);
        if (root.empty())
          continue;
        AliasRelation relation = comparePaths(left.slotPaths, right.slotPaths);
        if (relation == AliasRelation::Disjoint)
          continue;

        int64_t distance = 0;
        if (leftPoint->second.position >= rightPoint->second.position) {
          if (!leftPoint->second.inLoop)
            continue;
          distance = 1;
        }
        std::string precision =
            distance > 0
                ? "conservative_cross_iteration_alias"
                : (relation == AliasRelation::Same ? "exact"
                                                   : "conservative_alias");
        std::string reason = relation == AliasRelation::Same
                                 ? "same normalized LDS slot"
                                 : "LDS slot paths may alias";
        addDependency({"", kind, left.operationId, right.operationId,
                       left.valueId, right.valueId, root, precision, reason,
                       distance});
      }
      if (left.effect == "write" && leftPoint->second.inLoop) {
        if (std::optional<int64_t> period = slotReusePeriod(left)) {
          std::string root =
              left.rootValueIds.empty() ? "" : left.rootValueIds.front();
          addDependency({"", "memory_waw", left.operationId, left.operationId,
                         left.valueId, left.valueId, root, "structured_exact",
                         "same static loop write reuses its logical slot",
                         *period});
        }
      }
    }
  }

  void addAsyncDependencies() {
    for (const PlanAsyncTransaction &transaction : asyncTransactions) {
      const PlanAsyncGroup *group = groupById[transaction.commitGroupId];
      std::string root = transaction.rootValueIds.empty()
                             ? ""
                             : transaction.rootValueIds.front();
      std::string commit = group ? group->commitOperationId : "";
      if (!commit.empty())
        addDependency({"", "async_commit", transaction.producerOperationId,
                       commit, transaction.destinationValueId,
                       transaction.destinationValueId, root,
                       transaction.precision, "async write enters commit group",
                       0});

      for (const PlanAsyncFrontier &completion :
           transaction.completionFrontiers) {
        if (!commit.empty())
          addDependency({"", "async_completion", commit, completion.operationId,
                         transaction.destinationValueId,
                         transaction.destinationValueId, root,
                         completion.precision, "commit group completed by wait",
                         completion.iterationDistance});
      }
      for (const PlanAsyncFrontier &visibility :
           transaction.visibilityFrontiers) {
        const PlanAsyncFrontier *completion = nearestFrontier(
            transaction.completionFrontiers, visibility.iterationDistance);
        addDependency({"", "barrier_visibility",
                       completion ? completion->operationId : commit,
                       visibility.operationId, transaction.destinationValueId,
                       transaction.destinationValueId, root,
                       visibility.precision,
                       "CTA barrier makes completed LDS write visible",
                       visibility.iterationDistance});
      }
      for (const PlanAsyncFrontier &consumer : transaction.consumerFrontiers) {
        const PlanAsyncFrontier *visibility = nearestFrontier(
            transaction.visibilityFrontiers, consumer.iterationDistance);
        if (visibility)
          addDependency({"", "async_consumer", visibility->operationId,
                         consumer.operationId, transaction.destinationValueId,
                         transaction.destinationValueId, root,
                         consumer.precision, "visible LDS value is consumed",
                         consumer.iterationDistance});
      }
      for (const PlanAsyncFrontier &release : transaction.releaseFrontiers) {
        const PlanAsyncFrontier *consumer = nearestFrontier(
            transaction.consumerFrontiers, release.iterationDistance);
        if (consumer)
          addDependency({"", "consumer_release", consumer->operationId,
                         release.operationId, transaction.destinationValueId,
                         transaction.destinationValueId, root,
                         release.precision,
                         "CTA barrier releases consumed LDS slot",
                         release.iterationDistance});
      }
      for (const PlanAsyncFrontier &overwrite :
           transaction.overwriteFrontiers) {
        const PlanAsyncFrontier *release = nearestFrontier(
            transaction.releaseFrontiers, overwrite.iterationDistance);
        addDependency(
            {"", "slot_reuse",
             release ? release->operationId : transaction.producerOperationId,
             overwrite.operationId, transaction.destinationValueId,
             transaction.destinationValueId, root,
             release ? overwrite.precision : "conservative_missing_release",
             release ? "released LDS slot may be overwritten"
                     : "overwrite has no proven release frontier",
             overwrite.iterationDistance});
      }
    }
  }

  const PlanAsyncFrontier *
  nearestFrontier(ArrayRef<PlanAsyncFrontier> frontiers,
                  int64_t iterationDistance) const {
    const PlanAsyncFrontier *best = nullptr;
    for (const PlanAsyncFrontier &frontier : frontiers) {
      if (frontier.iterationDistance > iterationDistance)
        continue;
      if (!best || frontier.iterationDistance > best->iterationDistance)
        best = &frontier;
    }
    return best;
  }

  void computePeaksAndResources() {
    for (const PlanLdsAllocationRecord &allocation : ldsAllocations) {
      if (allocation.logicalBytes)
        result.resources.logicalLdsAllocationBytes += *allocation.logicalBytes;
      else
        ++result.resources.unknownLdsAllocations;
    }

    for (const PlanMemoryAccess &access : memoryAccesses)
      for (const PlanSlotPath &path : access.slotPaths)
        for (const PlanSlotExpression &index : path.indices) {
          int64_t depth = index.modulus;
          if (index.kind == "constant" && index.offset >= 0)
            depth = std::max<int64_t>(depth, index.offset + 1);
          result.resources.maxLogicalSlotDepth =
              std::max(result.resources.maxLogicalSlotDepth, depth);
        }

    for (const PlanAsyncTransaction &transaction : asyncTransactions)
      for (const auto *frontiers :
           {&transaction.completionFrontiers, &transaction.visibilityFrontiers,
            &transaction.consumerFrontiers, &transaction.releaseFrontiers,
            &transaction.overwriteFrontiers})
        for (const PlanAsyncFrontier &frontier : *frontiers)
          result.resources.maxAsyncIterationDistance =
              std::max(result.resources.maxAsyncIterationDistance,
                       frontier.iterationDistance);
    for (const PlanAsyncWaitRecord &wait : asyncWaits)
      result.resources.maxPossiblyOutstandingGroups =
          std::max<int64_t>(result.resources.maxPossiblyOutstandingGroups,
                            wait.possiblyOutstandingGroups.size());

    for (const PlanBlockRecord &block : blocks) {
      PlanPeakLiveSet best;
      best.blockId = block.id;
      for (int64_t position = 0;
           position <= static_cast<int64_t>(block.operations.size());
           ++position) {
        PlanPeakLiveSet candidate;
        candidate.blockId = block.id;
        candidate.position = position;
        if (position < static_cast<int64_t>(block.operations.size()))
          candidate.operationId = block.operations[position];

        std::set<std::string> tensorValues;
        for (const PlanLiveSegment &segment : liveSegments) {
          if (segment.blockId != block.id ||
              !(segment.startPosition <= position &&
                position < segment.endPosition))
            continue;
          const PlanValueRecord *value = valueById[segment.valueId];
          if (!value || value->category != "tensor_register_logical" ||
              !tensorValues.insert(value->id).second)
            continue;
          candidate.tensorValueIds.push_back(value->id);
          if (value->logicalBytes)
            candidate.logicalTensorBytes += *value->logicalBytes;
          else
            ++candidate.unknownTensorValueCount;
        }

        std::set<std::string> ldsRoots;
        for (const PlanLdsAllocationRecord &allocation : ldsAllocations) {
          bool alive = llvm::any_of(
              allocation.liveSegments, [&](const PlanLiveSegment &segment) {
                return segment.blockId == block.id &&
                       segment.startPosition <= position &&
                       position < segment.endPosition;
              });
          if (!alive || !ldsRoots.insert(allocation.rootValueId).second)
            continue;
          candidate.ldsRootValueIds.push_back(allocation.rootValueId);
          if (allocation.logicalBytes)
            candidate.logicalLdsBytes += *allocation.logicalBytes;
        }
        llvm::sort(candidate.tensorValueIds);
        llvm::sort(candidate.ldsRootValueIds);
        if (peakKey(candidate) > peakKey(best))
          best = std::move(candidate);
      }
      result.peakLiveSets.push_back(std::move(best));
    }

    for (const PlanPeakLiveSet &peak : result.peakLiveSets) {
      result.resources.peakLogicalTensorBytes = std::max(
          result.resources.peakLogicalTensorBytes, peak.logicalTensorBytes);
      result.resources.peakLogicalTensorCount = std::max<int64_t>(
          result.resources.peakLogicalTensorCount, peak.tensorValueIds.size());
      result.resources.peakUnknownTensorValueCount =
          std::max(result.resources.peakUnknownTensorValueCount,
                   peak.unknownTensorValueCount);
      result.resources.peakLogicalLdsBytes =
          std::max(result.resources.peakLogicalLdsBytes, peak.logicalLdsBytes);
    }
  }

  static std::tuple<int64_t, size_t, int64_t, int64_t, int64_t>
  peakKey(const PlanPeakLiveSet &peak) {
    return {peak.logicalTensorBytes, peak.tensorValueIds.size(),
            peak.unknownTensorValueCount, peak.logicalLdsBytes, -peak.position};
  }

  void classifyUnresolvedFacts() {
    for (const PlanDiagnostic &diagnostic : diagnostics) {
      const PlanValueRecord *value = valueById[diagnostic.valueId];
      bool importantValue =
          value && (value->category == "tensor_register_logical" ||
                    value->category == "memdesc");
      bool acceptedFallback = diagnostic.code == "identity_collision" &&
                              diagnostic.severity != "error";
      result.unresolvedFacts.push_back(
          {diagnostic.severity, diagnostic.code, diagnostic.message,
           diagnostic.operationId, diagnostic.valueId,
           acceptedFallback ? "advisory"
                            : (importantValue ? "important" : "advisory"),
           acceptedFallback ? "accepted_deterministic_fallback" : "open"});
    }
    for (const PlanLdsReuseHazard &hazard : ldsReuseHazards)
      result.unresolvedFacts.push_back(
          {hazard.severity, hazard.code, hazard.message, hazard.operationId,
           hazard.rootValueId, "important", "open"});
  }

  void sortResult() {
    llvm::sort(result.dependencies, [](const auto &lhs, const auto &rhs) {
      return lhs.id < rhs.id;
    });
    llvm::sort(result.peakLiveSets, [](const auto &lhs, const auto &rhs) {
      return lhs.blockId < rhs.blockId;
    });
    llvm::sort(result.unresolvedFacts, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.importance, lhs.status, lhs.severity, lhs.code,
                      lhs.operationId, lhs.valueId) <
             std::tie(rhs.importance, rhs.status, rhs.severity, rhs.code,
                      rhs.operationId, rhs.valueId);
    });
  }

  ArrayRef<PlanOperationRecord> operations;
  ArrayRef<PlanValueRecord> values;
  ArrayRef<PlanLineageEdge> lineageEdges;
  ArrayRef<PlanBlockRecord> blocks;
  ArrayRef<PlanLiveSegment> liveSegments;
  ArrayRef<PlanMemoryAccess> memoryAccesses;
  ArrayRef<PlanLdsAllocationRecord> ldsAllocations;
  ArrayRef<PlanAsyncTransaction> asyncTransactions;
  ArrayRef<PlanAsyncGroup> asyncGroups;
  ArrayRef<PlanAsyncWaitRecord> asyncWaits;
  ArrayRef<PlanLdsReuseHazard> ldsReuseHazards;
  ArrayRef<PlanDiagnostic> diagnostics;
  PlanAuditAnalysisResult result;
  std::map<std::string, const PlanOperationRecord *> operationById;
  std::map<std::string, const PlanValueRecord *> valueById;
  std::map<std::string, const PlanAsyncGroup *> groupById;
  std::map<std::string, OperationPoint> points;
  std::set<std::string> dependencyKeys;
};

} // namespace

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
    ArrayRef<PlanDiagnostic> diagnostics) {
  return AuditAnalysis(operations, values, lineageEdges, blocks, liveSegments,
                       memoryAccesses, ldsAllocations, asyncTransactions,
                       asyncGroups, asyncWaits, ldsReuseHazards, diagnostics)
      .run();
}

} // namespace mlir::triton::plan
