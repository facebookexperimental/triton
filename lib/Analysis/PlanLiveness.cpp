#include "triton/Analysis/PlanValueGraph.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace mlir::triton::plan {
namespace {

using RootSet = llvm::SmallSetVector<Value, 4>;

static bool isMemDesc(Value value) {
  return value && isa<gpu::MemDescType>(value.getType());
}

static std::optional<int64_t> getConstant(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

static std::optional<int64_t> getLogicalBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  int64_t bits = elementType.getIntOrFloatBitWidth();
  int64_t elements = shaped.getNumElements();
  if (bits <= 0 || elements < 0 ||
      elements > (std::numeric_limits<int64_t>::max() - 7) / bits)
    return std::nullopt;
  return (elements * bits + 7) / 8;
}

static int64_t positiveModulo(int64_t value, int64_t modulus) {
  if (modulus <= 0)
    return value;
  int64_t result = value % modulus;
  return result < 0 ? result + modulus : result;
}

struct IndexExpr {
  std::string kind = "unknown";
  std::string baseValueId;
  int64_t coefficient = 0;
  int64_t offset = 0;
  int64_t modulus = 0;
  std::vector<int64_t> possibleSlots;
};

static std::vector<int64_t> makeRange(int64_t begin, int64_t end,
                                      int64_t step) {
  std::vector<int64_t> values;
  if (step <= 0 || end < begin || end - begin > 1024)
    return values;
  for (int64_t value = begin; value < end && values.size() < 64; value += step)
    values.push_back(value);
  return values;
}

static IndexExpr
normalizeIndex(Value value, int64_t depth,
               const llvm::DenseMap<Value, std::string> &valueIds,
               unsigned recursion = 0) {
  IndexExpr result;
  result.baseValueId = valueIds.lookup(value);
  if (recursion > 16)
    return result;

  if (std::optional<int64_t> constant = getConstant(value)) {
    result.kind = "constant";
    result.offset = *constant;
    result.possibleSlots = {*constant};
    return result;
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto loop =
            dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        loop && argument == loop.getInductionVar()) {
      result.kind = "induction";
      result.coefficient = 1;
      if (auto lower = getConstant(loop.getLowerBound()))
        if (auto upper = getConstant(loop.getUpperBound()))
          if (auto step = getConstant(loop.getStep()))
            result.possibleSlots = makeRange(*lower, *upper, *step);
      return result;
    }
  }

  Operation *op = value.getDefiningOp();
  if (!op)
    return result;

  if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(
          op))
    return normalizeIndex(op->getOperand(0), depth, valueIds, recursion + 1);

  auto normalizeBinaryConstant = [&](Value variable, int64_t constant,
                                     int64_t variableSign,
                                     int64_t constantSign) {
    IndexExpr nested = normalizeIndex(variable, depth, valueIds, recursion + 1);
    if (nested.kind == "unknown")
      return nested;
    nested.coefficient *= variableSign;
    nested.offset = nested.offset * variableSign + constant * constantSign;
    if (nested.modulus > 0) {
      nested.coefficient = positiveModulo(nested.coefficient, nested.modulus);
      nested.offset = positiveModulo(nested.offset, nested.modulus);
      nested.possibleSlots.clear();
      for (int64_t slot = 0; slot < nested.modulus && slot < 64; ++slot)
        nested.possibleSlots.push_back(slot);
    } else if (!nested.possibleSlots.empty()) {
      for (int64_t &slot : nested.possibleSlots)
        slot = slot * variableSign + constant * constantSign;
      llvm::sort(nested.possibleSlots);
      nested.possibleSlots.erase(
          std::unique(nested.possibleSlots.begin(), nested.possibleSlots.end()),
          nested.possibleSlots.end());
    }
    return nested;
  };

  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    std::optional<int64_t> lhs = getConstant(op->getOperand(0));
    std::optional<int64_t> rhs = getConstant(op->getOperand(1));
    bool isSub = isa<arith::SubIOp>(op);
    if (!lhs && rhs)
      return normalizeBinaryConstant(op->getOperand(0), *rhs, 1,
                                     isSub ? -1 : 1);
    if (lhs && !rhs)
      return normalizeBinaryConstant(op->getOperand(1), *lhs, isSub ? -1 : 1,
                                     1);
  }

  if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
    std::optional<int64_t> modulus = getConstant(op->getOperand(1));
    if (modulus && *modulus > 0) {
      result =
          normalizeIndex(op->getOperand(0), depth, valueIds, recursion + 1);
      if (result.kind == "unknown") {
        result.baseValueId = valueIds.lookup(op->getOperand(0));
        result.coefficient = 1;
      }
      result.kind = "modulo";
      result.modulus = *modulus;
      result.coefficient = positiveModulo(result.coefficient, *modulus);
      result.offset = positiveModulo(result.offset, *modulus);
      result.possibleSlots.clear();
      for (int64_t slot = 0; slot < *modulus && slot < 64; ++slot)
        result.possibleSlots.push_back(slot);
      return result;
    }
  }

  if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
    std::optional<int64_t> lhs = getConstant(andOp.getLhs());
    std::optional<int64_t> rhs = getConstant(andOp.getRhs());
    Value variable = lhs ? andOp.getRhs() : andOp.getLhs();
    std::optional<int64_t> mask = lhs ? lhs : rhs;
    if (mask && *mask >= 0 && llvm::isPowerOf2_64(*mask + 1)) {
      result = normalizeIndex(variable, depth, valueIds, recursion + 1);
      if (result.kind == "unknown") {
        result.baseValueId = valueIds.lookup(variable);
        result.coefficient = 1;
      }
      result.kind = "modulo";
      result.modulus = *mask + 1;
      result.coefficient = positiveModulo(result.coefficient, result.modulus);
      result.offset = positiveModulo(result.offset, result.modulus);
      result.possibleSlots.clear();
      for (int64_t slot = 0; slot < result.modulus && slot < 64; ++slot)
        result.possibleSlots.push_back(slot);
      return result;
    }
  }

  return result;
}

static std::string formatIndex(IndexExpr expr) {
  if (expr.kind == "constant")
    return "constant(" + std::to_string(expr.offset) + ")";
  if (expr.kind == "unknown")
    return "unknown(" + expr.baseValueId + ")";
  std::string affine;
  if (expr.coefficient == 1)
    affine = expr.baseValueId;
  else
    affine = std::to_string(expr.coefficient) + "*" + expr.baseValueId;
  if (expr.offset > 0)
    affine += "+" + std::to_string(expr.offset);
  else if (expr.offset < 0)
    affine += std::to_string(expr.offset);
  if (expr.modulus > 0)
    return "(" + affine + ") mod " + std::to_string(expr.modulus);
  return affine;
}

static PlanSlotExpression
makeSlotExpression(Value index, int64_t depth,
                   const llvm::DenseMap<Value, std::string> &valueIds) {
  IndexExpr normalized = normalizeIndex(index, depth, valueIds);
  PlanSlotExpression result;
  result.kind = normalized.kind;
  result.baseValueId = normalized.baseValueId;
  result.coefficient = normalized.coefficient;
  result.offset = normalized.offset;
  result.modulus = normalized.modulus;
  result.possibleSlots = std::move(normalized.possibleSlots);
  result.text = formatIndex(normalized);
  return result;
}

static std::string
blockId(Block *block, int64_t blockNumber,
        const llvm::DenseMap<Operation *, std::string> &ids) {
  Region *region = block->getParent();
  Operation *parent = region ? region->getParentOp() : nullptr;
  std::string parentId = ids.lookup(parent);
  if (parentId.empty())
    parentId = "function";
  return "block:" + parentId +
         ":region:" + std::to_string(region ? region->getRegionNumber() : -1) +
         ":index:" + std::to_string(blockNumber);
}

static void
collectBlocks(Region &region,
              const llvm::DenseMap<Operation *, std::string> &operationIds,
              std::vector<PlanBlockRecord> &records,
              llvm::DenseMap<Block *, std::string> &blockIds,
              llvm::DenseMap<Operation *, int64_t> &positions) {
  Operation *parent = region.getParentOp();
  std::string parentId = operationIds.lookup(parent);
  for (auto [blockNumber, block] : llvm::enumerate(region.getBlocks())) {
    PlanBlockRecord record;
    record.id = blockId(&block, blockNumber, operationIds);
    record.parentOperationId = parentId;
    record.regionNumber = region.getRegionNumber();
    record.blockNumber = blockNumber;
    blockIds[&block] = record.id;
    for (auto [position, operation] : llvm::enumerate(block.getOperations())) {
      positions[&operation] = position;
      record.operations.push_back(operationIds.lookup(&operation));
      for (Region &nested : operation.getRegions())
        collectBlocks(nested, operationIds, records, blockIds, positions);
    }
    records.push_back(std::move(record));
  }
}

static SmallVector<Value> getAliasSources(Value value) {
  SmallVector<Value> sources;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    Operation *parent = block->getParentOp();
    if (auto loop = dyn_cast_or_null<scf::ForOp>(parent)) {
      if (argument == loop.getInductionVar())
        return sources;
      unsigned index = argument.getArgNumber() - 1;
      if (index < loop.getInitArgs().size()) {
        sources.push_back(loop.getInitArgs()[index]);
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        sources.push_back(yield.getResults()[index]);
      }
    } else if (auto loop = dyn_cast_or_null<scf::WhileOp>(parent)) {
      unsigned index = argument.getArgNumber();
      if (block == loop.getBeforeBody()) {
        if (index < loop.getInits().size())
          sources.push_back(loop.getInits()[index]);
        if (index < loop.getYieldOp().getResults().size())
          sources.push_back(loop.getYieldOp().getResults()[index]);
      } else if (block == loop.getAfterBody() &&
                 index < loop.getConditionOp().getArgs().size()) {
        sources.push_back(loop.getConditionOp().getArgs()[index]);
      }
    }
    return sources;
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return sources;
  Operation *op = result.getOwner();
  unsigned index = result.getResultNumber();
  if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
    if (op->getNumOperands())
      sources.push_back(op->getOperand(0));
  } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
    sources.push_back(select.getTrueValue());
    sources.push_back(select.getFalseValue());
  } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (index < yield.getResults().size())
      sources.push_back(yield.getResults()[index]);
  } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
    if (index < loop.getConditionOp().getArgs().size())
      sources.push_back(loop.getConditionOp().getArgs()[index]);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto appendYield = [&](Region &region) {
      if (region.empty())
        return;
      if (auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator()))
        if (index < yield.getResults().size())
          sources.push_back(yield.getResults()[index]);
    };
    appendYield(ifOp.getThenRegion());
    appendYield(ifOp.getElseRegion());
  }
  llvm::erase_if(sources, [](Value source) { return !isMemDesc(source); });
  return sources;
}

static std::vector<int64_t> getI32Array(Operation *op, StringRef name) {
  std::vector<int64_t> values;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(name))
    for (int32_t value : attr.asArrayRef())
      values.push_back(value);
  else if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name))
    values.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
  return values;
}

static std::string slotPathKey(const PlanSlotPath &path) {
  std::string key = path.rootValueId;
  for (const PlanSlotExpression &index : path.indices)
    key += "|" + index.text;
  return key;
}

static void mergePaths(std::vector<PlanSlotPath> &destination,
                       ArrayRef<PlanSlotPath> source) {
  std::set<std::string> keys;
  for (const PlanSlotPath &path : destination)
    keys.insert(slotPathKey(path));
  for (const PlanSlotPath &path : source)
    if (keys.insert(slotPathKey(path)).second)
      destination.push_back(path);
  llvm::sort(destination, [](const auto &lhs, const auto &rhs) {
    return slotPathKey(lhs) < slotPathKey(rhs);
  });
}

static bool isPendingAsync(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.contains("async") || name == "amdg.buffer_load_to_local";
}

static std::string effectName(MemoryEffects::Effect *effect) {
  if (isa<MemoryEffects::Read>(effect))
    return "read";
  if (isa<MemoryEffects::Write>(effect))
    return "write";
  if (isa<MemoryEffects::Allocate>(effect))
    return "allocate";
  if (isa<MemoryEffects::Free>(effect))
    return "free";
  return "unknown";
}

static void mergeSegment(PlanLiveSegment &destination,
                         const PlanLiveSegment &source) {
  if (source.startPosition < destination.startPosition) {
    destination.startPosition = source.startPosition;
    destination.startOperationId = source.startOperationId;
  }
  if (source.endPosition > destination.endPosition) {
    destination.endPosition = source.endPosition;
    destination.endOperationId = source.endOperationId;
  }
  destination.liveIn |= source.liveIn;
  destination.liveOut |= source.liveOut;
  destination.crossesBackedge |= source.crossesBackedge;
  destination.iterationDistance =
      std::max(destination.iterationDistance, source.iterationDistance);
}

} // namespace

FailureOr<PlanLivenessResult> analyzePlanLiveness(
    FuncOp function,
    const llvm::DenseMap<Operation *, std::string> &operationIds,
    const llvm::DenseMap<Value, std::string> &valueIds,
    ArrayRef<PlanLineageEdge> lineageEdges) {
  PlanLivenessResult result;
  llvm::DenseMap<Block *, std::string> blockIds;
  llvm::DenseMap<Operation *, int64_t> positions;
  collectBlocks(function.getBody(), operationIds, result.blocks, blockIds,
                positions);
  llvm::sort(result.blocks,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });

  std::map<std::string, int64_t> backedgeDistances;
  for (const PlanLineageEdge &edge : lineageEdges)
    if (edge.iterationDistance > 0) {
      backedgeDistances[edge.source] =
          std::max(backedgeDistances[edge.source], edge.iterationDistance);
      backedgeDistances[edge.destination] =
          std::max(backedgeDistances[edge.destination], edge.iterationDistance);
    }

  Liveness liveness(function);
  for (const auto &[value, valueId] : valueIds) {
    llvm::DenseMap<Block *, SmallVector<Operation *>> liveByBlock;
    for (Operation *operation : liveness.resolveLiveness(value))
      if (operation && operation->getBlock())
        liveByBlock[operation->getBlock()].push_back(operation);

    for (const PlanBlockRecord &blockRecord : result.blocks) {
      Block *block = nullptr;
      for (const auto &[candidate, id] : blockIds)
        if (id == blockRecord.id) {
          block = candidate;
          break;
        }
      if (!block)
        continue;
      const LivenessBlockInfo *info = liveness.getLiveness(block);
      if (!info)
        continue;
      bool liveIn = info->isLiveIn(value);
      bool liveOut = info->isLiveOut(value);
      bool definedHere = (value.getDefiningOp() &&
                          value.getDefiningOp()->getBlock() == block) ||
                         (isa<BlockArgument>(value) &&
                          cast<BlockArgument>(value).getOwner() == block);
      SmallVector<Operation *> liveOperations = liveByBlock.lookup(block);
      if (!liveIn && !liveOut && !definedHere && liveOperations.empty())
        continue;

      int64_t operationCount = blockRecord.operations.size();
      int64_t start = liveIn ? 0 : operationCount;
      int64_t end = liveOut ? operationCount : 0;
      std::string startOperation;
      std::string endOperation;
      if (Operation *definition = value.getDefiningOp();
          definition && definition->getBlock() == block) {
        start = std::min(start, positions.lookup(definition));
        startOperation = operationIds.lookup(definition);
      } else if (isa<BlockArgument>(value) &&
                 cast<BlockArgument>(value).getOwner() == block) {
        start = 0;
      }
      for (Operation *operation : liveOperations) {
        int64_t position = positions.lookup(operation);
        if (!liveIn && position <= start) {
          start = position;
          startOperation = operationIds.lookup(operation);
        }
        if (!liveOut && position + 1 >= end) {
          end = position + 1;
          endOperation = operationIds.lookup(operation);
        }
      }
      if (end < start)
        end = start;
      if (value.getDefiningOp() && end == start)
        end = std::min(operationCount, start + 1);

      int64_t distance = backedgeDistances[valueId];
      if (!distance && liveIn && liveOut) {
        Operation *parent = block->getParentOp();
        if (isa_and_nonnull<scf::ForOp, scf::WhileOp>(parent))
          distance = 1;
      }
      result.liveSegments.push_back({valueId, blockRecord.id, startOperation,
                                     endOperation, start, end, liveIn, liveOut,
                                     distance > 0, distance});
    }
  }
  llvm::sort(result.liveSegments, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.valueId, lhs.blockId, lhs.startPosition,
                    lhs.endPosition) < std::tie(rhs.valueId, rhs.blockId,
                                                rhs.startPosition,
                                                rhs.endPosition);
  });

  llvm::DenseMap<Value, RootSet> roots;
  for (const auto &[value, id] : valueIds)
    if (isMemDesc(value) &&
        isa_and_nonnull<gpu::LocalAllocOp>(value.getDefiningOp()))
      roots[value].insert(value);

  for (unsigned round = 0; round < valueIds.size(); ++round) {
    bool changed = false;
    for (const auto &[value, id] : valueIds) {
      if (!isMemDesc(value) ||
          isa_and_nonnull<gpu::LocalAllocOp>(value.getDefiningOp()))
        continue;
      for (Value source : getAliasSources(value))
        for (Value root : roots.lookup(source))
          changed |= roots[value].insert(root);
    }
    if (!changed)
      break;
  }

  llvm::DenseMap<Value, std::vector<PlanSlotPath>> paths;
  for (const auto &[value, id] : valueIds)
    if (isMemDesc(value) &&
        isa_and_nonnull<gpu::LocalAllocOp>(value.getDefiningOp()))
      paths[value] = {{id, {}}};

  for (unsigned round = 0; round < valueIds.size(); ++round) {
    bool changed = false;
    for (const auto &[value, id] : valueIds) {
      if (!isMemDesc(value) ||
          isa_and_nonnull<gpu::LocalAllocOp>(value.getDefiningOp()))
        continue;
      std::vector<PlanSlotPath> candidates;
      for (Value source : getAliasSources(value))
        mergePaths(candidates, paths.lookup(source));
      Operation *op = value.getDefiningOp();
      if (op && op->getName().getStringRef() == "ttg.memdesc_index" &&
          op->getNumOperands() >= 2) {
        int64_t depth = 0;
        if (auto sourceType =
                dyn_cast<gpu::MemDescType>(op->getOperand(0).getType());
            sourceType && !sourceType.getShape().empty())
          depth = sourceType.getShape().front();
        PlanSlotExpression expression =
            makeSlotExpression(op->getOperand(1), depth, valueIds);
        for (PlanSlotPath &path : candidates)
          if (path.indices.size() < 8)
            path.indices.push_back(expression);
        if (expression.kind == "unknown")
          result.diagnostics.push_back(
              {"warning", "unresolved_lds_slot",
               "memdesc_index could not be normalized to a named slot",
               operationIds.lookup(op), id});
        for (int64_t possible : expression.possibleSlots)
          if (depth > 0 && (possible < 0 || possible >= depth))
            result.diagnostics.push_back(
                {"error", "lds_slot_out_of_bounds",
                 "normalized memdesc_index may exceed the leading dimension",
                 operationIds.lookup(op), id});
      }
      size_t oldSize = paths[value].size();
      mergePaths(paths[value], candidates);
      changed |= paths[value].size() != oldSize;
    }
    if (!changed)
      break;
  }

  for (const auto &[value, id] : valueIds) {
    if (!isMemDesc(value))
      continue;
    PlanAliasRecord alias;
    alias.valueId = id;
    Operation *op = value.getDefiningOp();
    alias.viewKind =
        op ? op->getName().getStringRef().str() : "external_memdesc";
    SmallVector<Value> sources = getAliasSources(value);
    if (sources.size() == 1)
      alias.sourceValueId = valueIds.lookup(sources.front());
    for (Value root : roots.lookup(value))
      alias.rootValueIds.push_back(valueIds.lookup(root));
    llvm::sort(alias.rootValueIds);
    alias.rootValueIds.erase(
        std::unique(alias.rootValueIds.begin(), alias.rootValueIds.end()),
        alias.rootValueIds.end());
    if (op) {
      alias.staticOffsets = getI32Array(op, "offsets");
      alias.order = getI32Array(op, "order");
    }
    alias.slotPaths = paths.lookup(value);
    result.aliases.push_back(std::move(alias));
  }
  llvm::sort(result.aliases, [](const auto &lhs, const auto &rhs) {
    return lhs.valueId < rhs.valueId;
  });

  for (const auto &[operation, operationId] : operationIds) {
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value value = effect.getValue();
      if (!isMemDesc(value) || roots.lookup(value).empty())
        continue;
      PlanMemoryAccess access;
      access.operationId = operationId;
      access.valueId = valueIds.lookup(value);
      access.effect = effectName(effect.getEffect());
      access.pendingAsync = isPendingAsync(operation);
      for (Value root : roots.lookup(value))
        access.rootValueIds.push_back(valueIds.lookup(root));
      llvm::sort(access.rootValueIds);
      access.slotPaths = paths.lookup(value);
      result.memoryAccesses.push_back(std::move(access));
    }
  }
  llvm::sort(result.memoryAccesses, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.operationId, lhs.valueId, lhs.effect) <
           std::tie(rhs.operationId, rhs.valueId, rhs.effect);
  });
  result.memoryAccesses.erase(
      std::unique(result.memoryAccesses.begin(), result.memoryAccesses.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return std::tie(lhs.operationId, lhs.valueId, lhs.effect) ==
                           std::tie(rhs.operationId, rhs.valueId, rhs.effect);
                  }),
      result.memoryAccesses.end());

  for (const auto &[value, valueId] : valueIds) {
    auto alloc = value.getDefiningOp<gpu::LocalAllocOp>();
    if (!alloc)
      continue;
    PlanLdsAllocationRecord allocation;
    allocation.rootValueId = valueId;
    allocation.allocationOperationId = operationIds.lookup(alloc);
    allocation.logicalBytes = getLogicalBytes(value.getType());
    if (IntegerAttr alignment = alloc->getAttrOfType<IntegerAttr>("alignment"))
      allocation.alignment = alignment.getInt();

    std::map<std::string, PlanLiveSegment> unionByBlock;
    for (const PlanAliasRecord &alias : result.aliases) {
      if (!llvm::is_contained(alias.rootValueIds, valueId))
        continue;
      allocation.aliases.push_back(alias.valueId);
      for (const PlanLiveSegment &segment : result.liveSegments) {
        if (segment.valueId != alias.valueId)
          continue;
        auto [it, inserted] = unionByBlock.insert({segment.blockId, segment});
        if (!inserted)
          mergeSegment(it->second, segment);
      }
    }
    llvm::sort(allocation.aliases);
    for (auto &[block, segment] : unionByBlock) {
      segment.valueId = valueId;
      allocation.liveSegments.push_back(std::move(segment));
    }
    result.ldsAllocations.push_back(std::move(allocation));
  }
  llvm::sort(result.ldsAllocations, [](const auto &lhs, const auto &rhs) {
    return lhs.rootValueId < rhs.rootValueId;
  });
  llvm::sort(result.diagnostics, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.severity, lhs.code, lhs.operationId, lhs.valueId,
                    lhs.message) < std::tie(rhs.severity, rhs.code,
                                            rhs.operationId, rhs.valueId,
                                            rhs.message);
  });
  return result;
}

} // namespace mlir::triton::plan
