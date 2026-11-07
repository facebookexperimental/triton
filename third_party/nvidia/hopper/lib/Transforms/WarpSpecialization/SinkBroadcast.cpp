#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

#define GEN_PASS_DEF_NVGPUSINKBROADCAST
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

bool hasOnlyReadEffects(Operation *op) {
  if (isMemoryEffectFree(op)) {
    return true;
  }

  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface) {
    return false;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  if (effects.empty()) {
    return true;
  }

  return llvm::all_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect());
  });
}

bool mayWriteOrHaveUnknownEffects(Operation *op) {
  if (isMemoryEffectFree(op)) {
    return false;
  }

  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface) {
    return true;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return !isa<MemoryEffects::Read>(effect.getEffect());
  });
}

bool hasInterveningWriteOrUnknownEffects(Operation *producer, Operation *user) {
  for (Operation *it = producer->getNextNode(); it && it != user;
       it = it->getNextNode()) {
    if (mayWriteOrHaveUnknownEffects(it)) {
      return true;
    }
  }
  return false;
}

bool isSinkableProducer(Operation *producer, Operation *user) {
  if (!producer || producer->getBlock() != user->getBlock() ||
      !producer->isBeforeInBlock(user) || producer->getNumRegions() != 0 ||
      producer->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  if (!hasOnlyReadEffects(producer)) {
    return false;
  }

  if (!isMemoryEffectFree(producer) &&
      hasInterveningWriteOrUnknownEffects(producer, user)) {
    return false;
  }

  return true;
}

bool collectSinkableChain(Operation *producer, Operation *user,
                          llvm::SetVector<Operation *> &chain) {
  if (!isSinkableProducer(producer, user)) {
    return false;
  }

  if (chain.contains(producer)) {
    return true;
  }

  // Include read-only loads but do not pull their address arithmetic along.
  // The goal is to keep the value materialization close to the broadcast use
  // without unnecessarily perturbing shared descriptor/index setup.
  if (isMemoryEffectFree(producer)) {
    for (Value operand : producer->getOperands()) {
      Operation *operandProducer = operand.getDefiningOp();
      if (!operandProducer) {
        continue;
      }
      if (!collectSinkableChain(operandProducer, user, chain)) {
        return false;
      }
    }
  }

  chain.insert(producer);
  return true;
}

bool allUsesAreInChainOrByUser(Operation *op,
                               const llvm::DenseSet<Operation *> &chain,
                               Operation *user) {
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      if (owner != user && !chain.contains(owner)) {
        return false;
      }
    }
  }
  return true;
}

struct BroadcastChain {
  triton::BroadcastOp broadcast;
  SmallVector<Operation *> ops;
};

SmallVector<BroadcastChain> getSinkableBroadcastChains(Operation *op) {
  SmallVector<BroadcastChain> chains;
  llvm::SmallDenseSet<Operation *, 4> seenBroadcasts;

  if (op->getNumOperands() < 2 || op->getNumRegions() != 0 ||
      !op->hasTrait<OpTrait::Elementwise>() || !isMemoryEffectFree(op)) {
    return chains;
  }

  for (Value operand : op->getOperands()) {
    auto broadcast = operand.getDefiningOp<triton::BroadcastOp>();
    if (!broadcast || !seenBroadcasts.insert(broadcast).second) {
      continue;
    }

    llvm::SetVector<Operation *> chain;
    if (!collectSinkableChain(broadcast, op, chain)) {
      continue;
    }

    chains.push_back(
        {broadcast, SmallVector<Operation *>(chain.begin(), chain.end())});
  }

  return chains;
}

bool sinkBroadcastOperands(Operation *op) {
  SmallVector<BroadcastChain> chains = getSinkableBroadcastChains(op);
  if (chains.empty()) {
    return false;
  }

  bool changed = false;
  OpBuilder builder(op);
  for (const BroadcastChain &chain : chains) {
    triton::BroadcastOp broadcast = chain.broadcast;
    Value oldResult = broadcast.getResult();

    llvm::DenseSet<Operation *> chainSet;
    for (Operation *chainOp : chain.ops) {
      chainSet.insert(chainOp);
    }

    bool canMoveOriginal = llvm::all_of(chain.ops, [&](Operation *chainOp) {
      return allUsesAreInChainOrByUser(chainOp, chainSet, op);
    });

    if (canMoveOriginal) {
      for (Operation *chainOp : chain.ops) {
        if (chainOp->getNextNode() != op) {
          chainOp->moveBefore(op);
        }
        changed = true;
      }
      continue;
    }

    IRMapping mapping;
    for (Operation *chainOp : chain.ops) {
      builder.clone(*chainOp, mapping);
    }
    Value newResult = mapping.lookup(oldResult);
    for (OpOperand &operand : op->getOpOperands()) {
      if (operand.get() == oldResult) {
        operand.set(newResult);
      }
    }
    if (oldResult.use_empty()) {
      broadcast.erase();
    }
    changed = true;
  }

  return changed;
}

class NVGPUSinkBroadcastPass
    : public impl::NVGPUSinkBroadcastBase<NVGPUSinkBroadcastPass> {
public:
  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation()->walk([&](Operation *op) {
      if (!getSinkableBroadcastChains(op).empty()) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      if (!op->getBlock()) {
        continue;
      }
      sinkBroadcastOperands(op);
    }
  }
};

} // namespace

} // namespace mlir
