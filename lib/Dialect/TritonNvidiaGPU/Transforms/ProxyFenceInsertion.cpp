#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Function.h"
#include "triton/Analysis/MemoryFrontier.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <cstdint>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// Track generic-proxy frontiers with exact physical regions. Async-proxy
// accesses are retained only while they can still conflict with generic
// accesses preceding the current function; this lets callers place a fence
// before a call without inventing incoming effects in the callee.
namespace {

enum : uint8_t { kCTAScope = 1, kClusterScope = 2 };

uint8_t getProxyFenceScope(Operation *op) {
  auto load = dyn_cast<TMALoadLikeOpInterface>(op);
  auto mma = dyn_cast<MMAv5OpInterface>(op);
  bool cluster = (load && load.getMulticast()) || (mma && mma.getTwoCtas()) ||
                 (isa<TMEMCopyOp>(op) && getModuleTwoCTAs(op)) ||
                 (isa<CLCTryCancelOp>(op) && gpu::lookupNumCTAs(op) > 1);
  return cluster ? kClusterScope : kCTAScope;
}

// Whether a TMA load may write another CTA's shared memory: it multicasts, or
// its destination layout spans CTAs (the lowering may then target a peer
// through mapa). Otherwise it writes only the issuing CTA's shared memory.
bool mayWritePeerSmem(TMALoadLikeOpInterface load) {
  auto copy = dyn_cast<AsyncTMACopyGlobalToLocalOp>(load.getOperation());
  return load.getMulticast() || (copy && copy.getMulticastTargets()) ||
         gpu::lookupNumCTAs(load) > 1;
}

// Whether a release reaches a barrier outside the arriving CTA.
bool isRemoteArrive(Operation *op) {
  auto arrive = dyn_cast<ArriveBarrierOp>(op);
  if (!arrive)
    return false;
  return isa<SharedClusterMemorySpaceAttr>(
             arrive.getAlloc().getType().getMemorySpace()) ||
         arrive.getFromCTA() || arrive.getMulticastCTA() != 0 ||
         gpu::lookupNumCTAs(op) > 1;
}

using BufferAccess = const BufferRegionFootprint *;

struct ProxyBlockInfo {
  using Frontier = ScopedMemoryFrontier<uint8_t>;

  // Generic accesses since the last proxy fence and async accesses before the
  // first proxy fence reachable from function entry.
  Frontier generic;
  Frontier async;

  // Scopes fenced on every path from the function entry.
  uint8_t entryGenericFenced = 0;

  ProxyBlockInfo &join(const ProxyBlockInfo &other) {
    generic.join(other.generic);
    async.join(other.async);
    entryGenericFenced &= other.entryGenericFenced;
    return *this;
  }

  bool operator==(const ProxyBlockInfo &other) const {
    return std::tie(generic, async, entryGenericFenced) ==
           std::tie(other.generic, other.async, other.entryGenericFenced);
  }

  void fenceGeneric(uint8_t scopes) {
    if (scopes & kClusterScope)
      scopes |= kCTAScope;
    generic.eraseScopes(scopes);
    entryGenericFenced |= scopes;
  }
};

// Async-proxy shared-memory writes of a function, recorded at cluster scope
// so that a CTA-scope fence (which leaves the cluster bit of a generic access
// in place) does not hide the reads it fenced from the matching below.
struct AsyncWrites {
  // Every async-proxy shared-memory write.
  ProxyBlockInfo::Frontier all;
  // The writes that may come from, or land in, another CTA: multicast or
  // cross-CTA TMA loads and every non-TMA-load async writer (e.g. a remote
  // shared-memory copy).
  ProxyBlockInfo::Frontier peer;
};

struct ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using FuncMapT = PostOrderFunctionAnalysis<ProxyBlockInfo>::FuncMapT;

  ProxyFenceFunctionAnalysis(BufferRegionAnalysis &regions, uint8_t scopes,
                             const AsyncWrites &refills)
      : regions(regions), scopes(scopes), refills(refills) {}

  // Scope of the fence before a marked release. The refill that follows the
  // release is issued by the producer after it observes the arrive, so the
  // fence must order the reads fenced here before that TMA write. When the
  // arrive stays in this CTA and every async write that may touch the buffers
  // accessed since the last cluster-scope fence is a TMA load into this CTA's
  // own shared memory, the reads, the release, the acquire and the write all
  // stay inside one CTA and shared::cta suffices. If those buffers cannot be
  // matched to a refill, keep the module scope.
  uint8_t getReleaseFenceScope(Operation *op,
                               const ProxyBlockInfo &state) const {
    if (!(scopes & kClusterScope))
      return kCTAScope;
    if (isRemoteArrive(op) ||
        state.generic.hasHazard(refills.peer, kClusterScope) ||
        !state.generic.hasHazard(refills.all, kClusterScope))
      return kClusterScope;
    return kCTAScope;
  }

  void applyEffects(Operation *op, const ProxyBlockInfo &effects,
                    ProxyBlockInfo &state, OpBuilder &builder) {
    for (uint8_t scope : {kClusterScope, kCTAScope}) {
      if ((scopes & scope) && state.generic.hasHazard(effects.async, scope)) {
        builder.setInsertionPoint(op);
        FenceAsyncSharedOp::create(builder, op->getLoc(),
                                   scope == kClusterScope);
        state.fenceGeneric(scope);
        break;
      }
    }

    state.async.join(effects.async, scopes & ~state.entryGenericFenced);
    if (isa<CallOpInterface>(op) && effects.entryGenericFenced)
      state.fenceGeneric(effects.entryGenericFenced);
    state.generic.join(effects.generic);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      state->fenceGeneric(fence.getBCluster() ? kClusterScope : kCTAScope);
      return;
    }

    // An autoWS release whose buffer another partition refills with TMA. That
    // async write is in a different region, so the frontiers here never see
    // the hazard; fence the generic reads before the release unconditionally.
    // The release may have been promoted to a named barrier by now.
    if (isa<ArriveBarrierOp, NamedBarrierArriveOp>(op) &&
        op->hasAttr(kFenceGenericReadsAttrName)) {
      uint8_t scope = getReleaseFenceScope(op, *state);
      // Reuse a fence already in place (this pass revisits loop bodies),
      // widening it when cluster scope is needed.
      auto prevFence = dyn_cast_or_null<FenceAsyncSharedOp>(op->getPrevNode());
      if (!prevFence) {
        builder->setInsertionPoint(op);
        FenceAsyncSharedOp::create(*builder, op->getLoc(),
                                   scope == kClusterScope);
      } else if (scope == kClusterScope && !prevFence.getBCluster()) {
        prevFence.setBCluster(true);
      }
      state->fenceGeneric(scope);
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto effects = getCallSummary(
          call, *funcMap,
          [&](ProxyBlockInfo &effects, FunctionOpInterface callee) {
            for (auto *frontier : {&effects.generic, &effects.async})
              frontier->transformAccesses([&](BufferAccess access) {
                return regions.translateToCallsite(access, call, callee);
              });
          });
      if (effects)
        applyEffects(op, *effects, *state, *builder);
      return;
    }

    ProxyBlockInfo effects;
    // Beta ordering: barrier invalidation writes its backing shared memory.
    // If a later async op reuses those bytes as payload, the invalidation
    // needs the same proxy ordering as any other generic write. Other barrier
    // effects stay excluded: they synchronize the barrier, not payload bytes.
    bool isInvalBarrier = isa<InvalBarrierOp>(op);
    // Beta safety net: treat TMA-load shared reads as async even if the
    // effect description omits the async-proxy kind.
    bool isTmaLoad = isa<AsyncTMACopyGlobalToLocalOp>(op);
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      if (!access.isShared() ||
          (access.isShared(gpu::SharedKind::Barrier) && !isInvalBarrier))
        continue;

      bool async = access.isShared(gpu::SharedKind::Async) ||
                   (isTmaLoad && access.isRead);
      uint8_t accessScopes = async ? getProxyFenceScope(op) : scopes;
      ProxyBlockInfo::Frontier &frontier =
          async ? effects.async : effects.generic;
      BufferAccess footprint = regions.getFootprint(access.value);
      if (access.isRead)
        frontier.addRead(footprint, accessScopes);
      if (access.isWrite)
        frontier.addWrite(footprint, accessScopes);
    }

    if (op->hasAttr("allocation.size")) {
      BufferAccess scratch = regions.getScratchFootprint(op);
      effects.generic.addRead(scratch, scopes);
      effects.generic.addWrite(scratch, scopes);
    }
    applyEffects(op, effects, *state, *builder);
  }

  BufferRegionAnalysis &regions;
  uint8_t scopes;
  const AsyncWrites &refills;
};

} // namespace

struct ProxyFenceInsertionPass
    : public impl::TritonGPUProxyFenceInsertionBase<ProxyFenceInsertionPass> {
  using impl::TritonGPUProxyFenceInsertionBase<
      ProxyFenceInsertionPass>::TritonGPUProxyFenceInsertionBase;

  void runOnOperation() override {
    if (computeCapability < 90)
      return;

    ModuleOp module = getOperation();
    uint8_t scopes = 0;
    module.walk([&](Operation *op) {
      // Include TMA loads even if their effect description omits the
      // async-proxy kind (beta safety net).
      if (hasSharedAccess(op, gpu::SharedKind::Async) ||
          isa<AsyncTMACopyGlobalToLocalOp>(op))
        scopes |= getProxyFenceScope(op) | kCTAScope;
      return scopes & kClusterScope ? WalkResult::interrupt()
                                    : WalkResult::advance();
    });
    if (!scopes)
      return;

    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return signalPassFailure();

    // Footprints are per allocation frame, so collect async writes per
    // function. Every function gets an entry up front: the analyses keep
    // references.
    DenseMap<Operation *, AsyncWrites> refills;
    module.walk([&](FunctionOpInterface func) {
      AsyncWrites &writes = refills[func];
      func.walk([&](Operation *op) {
        auto record = [&](Value value, bool peer) {
          BufferAccess footprint = regions->getFootprint(value);
          writes.all.addWrite(footprint, kClusterScope);
          if (peer)
            writes.peer.addWrite(footprint, kClusterScope);
        };
        if (auto load = dyn_cast<TMALoadLikeOpInterface>(op)) {
          record(load.getResult(), mayWritePeerSmem(load));
          return;
        }
        for (const MemoryAccess &access : getMemoryAccesses(op))
          if (access.isWrite && access.isShared(gpu::SharedKind::Async))
            record(access.value, /*peer=*/true);
      });
    });

    ProxyFenceFunctionAnalysis::runModule(
        module, [&](FunctionOpInterface func) {
          return ProxyFenceFunctionAnalysis(*regions, scopes,
                                            refills.find(func)->second);
        });
  }
};

} // namespace mlir::triton::nvidia_gpu
