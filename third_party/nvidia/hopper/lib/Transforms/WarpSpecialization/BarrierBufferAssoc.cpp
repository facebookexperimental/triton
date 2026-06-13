#include "BarrierBufferAssoc.h"

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdlib>

namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;

namespace mlir {

// Nested WSBarrier key inside the `constraints` DictionaryAttr. Kept as a local
// literal (rather than including WSBarrierAnalysis.h) to keep this module's
// dependencies minimal; must stay in sync with WSBarrierAttr::kKey.
static constexpr llvm::StringLiteral kWSBarrierKey = "WSBarrier";
static constexpr llvm::StringLiteral kConstraintsAttr = "constraints";

// Append the (buffer.id, buffer.offset) of `alloc` to `out`, deduplicated.
static void appendGuard(llvm::SmallVectorImpl<GuardedBuffer> &out,
                        Operation *alloc) {
  if (!alloc)
    return;
  auto idAttr = alloc->getAttrOfType<IntegerAttr>("buffer.id");
  if (!idAttr)
    return;
  GuardedBuffer gb;
  gb.id = static_cast<int>(idAttr.getInt());
  if (auto offAttr = alloc->getAttrOfType<IntegerAttr>("buffer.offset"))
    gb.offset = static_cast<int>(offAttr.getInt());
  if (!llvm::is_contained(out, gb))
    out.push_back(gb);
}

llvm::SmallVector<GuardedBuffer, 2> computeChannelGuards(Channel *channel,
                                                         ReuseConfig *config) {
  llvm::SmallVector<GuardedBuffer, 2> guards;
  if (!channel)
    return guards;
  appendGuard(guards, channel->getAllocOp());

  // Union the whole reuse group: a single WSBarrier guards every physical
  // buffer that shares the reused memory.
  if (config) {
    for (unsigned g = 0, e = config->getGroupSize(); g < e; ++g) {
      ReuseGroup *grp = config->getGroup(g);
      bool member = llvm::any_of(
          grp->channels, [&](Channel *c) { return c == channel; });
      if (!member)
        continue;
      for (Channel *c : grp->channels)
        appendGuard(guards, c->getAllocOp());
    }
  }
  llvm::sort(guards);
  return guards;
}

void stampBarrierGuards(Operation *op, llvm::ArrayRef<GuardedBuffer> guards) {
  if (!op || guards.empty())
    return;
  llvm::SmallVector<int32_t, 2> ids, offsets;
  ids.reserve(guards.size());
  offsets.reserve(guards.size());
  for (const GuardedBuffer &g : guards) {
    ids.push_back(g.id);
    offsets.push_back(g.offset);
  }
  MLIRContext *ctx = op->getContext();
  op->setAttr(kBarrierGuardsAttr, DenseI32ArrayAttr::get(ctx, ids));
  op->setAttr(kBarrierOffsetsAttr, DenseI32ArrayAttr::get(ctx, offsets));
}

llvm::SmallVector<GuardedBuffer, 2> getBarrierGuards(Operation *op) {
  llvm::SmallVector<GuardedBuffer, 2> guards;
  if (!op)
    return guards;
  auto ids = op->getAttrOfType<DenseI32ArrayAttr>(kBarrierGuardsAttr);
  if (!ids)
    return guards;
  auto offs = op->getAttrOfType<DenseI32ArrayAttr>(kBarrierOffsetsAttr);
  for (int i = 0, e = ids.size(); i < e; ++i) {
    GuardedBuffer gb;
    gb.id = ids[i];
    gb.offset = (offs && i < offs.size()) ? offs[i] : 0;
    guards.push_back(gb);
  }
  return guards;
}

void forwardBarrierGuards(Operation *from, Operation *to) {
  if (!from || !to)
    return;
  if (Attribute ids = from->getAttr(kBarrierGuardsAttr))
    to->setAttr(kBarrierGuardsAttr, ids);
  if (Attribute offs = from->getAttr(kBarrierOffsetsAttr))
    to->setAttr(kBarrierOffsetsAttr, offs);
}

bool isBarrierEndpoint(Operation *op) {
  return isa<ttnvws::ProducerAcquireOp, ttnvws::ProducerCommitOp,
             ttnvws::ConsumerWaitOp, ttnvws::ConsumerReleaseOp,
             ttng::WaitBarrierOp, ttng::ArriveBarrierOp>(op);
}

// Whether `op` carries the WSBarrier marker in its `constraints` dict. NVWS
// token endpoints are always WS barriers; HW barriers are WS barriers only when
// lowered from a WS token (which forwards the constraints).
static bool isWSBarrier(Operation *op) {
  if (isa<ttnvws::ProducerAcquireOp, ttnvws::ProducerCommitOp,
          ttnvws::ConsumerWaitOp, ttnvws::ConsumerReleaseOp>(op))
    return true;
  if (auto dict = op->getAttrOfType<DictionaryAttr>(kConstraintsAttr))
    return dict.get(kWSBarrierKey) != nullptr;
  return false;
}

BarrierBufferMap buildBarrierBufferMap(triton::FuncOp funcOp) {
  BarrierBufferMap map;
  funcOp.walk([&](Operation *op) {
    if (!isBarrierEndpoint(op))
      return;
    auto guards = getBarrierGuards(op);
    if (!guards.empty())
      map[op] = std::move(guards);
  });
  return map;
}

unsigned dumpAndVerifyBarrierBufferMap(triton::FuncOp funcOp,
                                       const BarrierBufferMap &map,
                                       llvm::raw_ostream &os) {
  unsigned orphans = 0;
  os << "==== barrier->buffer map for @" << funcOp.getName() << " ====\n";
  funcOp.walk([&](Operation *op) {
    if (!isBarrierEndpoint(op))
      return;
    auto it = map.find(op);
    if (it == map.end() || it->second.empty()) {
      // Only WS barriers are expected to resolve to a buffer; ignore
      // non-WS HW barriers (e.g. TMA-store barriers).
      if (isWSBarrier(op)) {
        ++orphans;
        os << "  [ORPHAN] " << op->getName() << " : <no guarded buffer>\n";
      }
      return;
    }
    os << "  " << op->getName() << " : {";
    llvm::interleaveComma(it->second, os, [&](const GuardedBuffer &g) {
      os << "id=" << g.id;
      if (g.offset)
        os << ",off=" << g.offset;
    });
    os << "}\n";
  });
  os << "==== orphans: " << orphans << " ====\n";
  return orphans;
}

void dumpBarrierGuardsIfEnabled(triton::FuncOp funcOp) {
  // Use std::getenv directly (mirroring TRITON_DUMP_WS_GRAPHS in
  // WSMemoryPlanner) rather than getBoolEnv, which asserts the variable name
  // is in Triton's recognized env-var registry.
  if (!std::getenv("TRITON_DUMP_BARRIER_GUARDS"))
    return;
  BarrierBufferMap map = buildBarrierBufferMap(funcOp);
  dumpAndVerifyBarrierBufferMap(funcOp, map, llvm::errs());
}

} // namespace mlir
