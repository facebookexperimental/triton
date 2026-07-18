#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUINSERTTMEMALIASWARBARRIERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

struct TMemBase {
  int64_t row = 0;
  int64_t col = 0;
};

struct TMemView {
  Value root;
  std::optional<TMemBase> base;
};

// Resolve a TMEM view to its root and physical base address. TMEM pointers use
// the low 16 bits for a 32-bit column offset and the high 16 bits for a row
// offset. Reinterpret/transpose/reshape views do not move the base pointer.
static TMemView resolveTMemView(Value v) {
  TMemBase base;
  bool knownBase = true;
  llvm::DenseSet<Value> visited;
  while (v && visited.insert(v).second) {
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Operation *parent = ba.getOwner()->getParentOp();
      if (auto part = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
        auto captures = part.getExplicitCaptures();
        if (ba.getArgNumber() >= captures.size())
          break;
        v = captures[ba.getArgNumber()];
        continue;
      }
      break;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      break;
    if (auto index = dyn_cast<ttg::MemDescIndexOp>(def)) {
      APInt indexValue;
      if (matchPattern(index.getIndex(), m_ConstantInt(&indexValue))) {
        auto size = ttng::getTmemAllocSizes(index.getType());
        base.col += indexValue.getSExtValue() * size.numCols;
      } else {
        knownBase = false;
      }
      v = index.getSrc();
      continue;
    }
    if (auto subslice = dyn_cast<ttng::TMEMSubSliceOp>(def)) {
      uint32_t offset = ttng::getTMemSubSliceOffset(subslice.getSrc().getType(),
                                                    subslice.getN());
      base.col += offset & 0xffff;
      base.row += offset >> 16;
      v = subslice.getSrc();
      continue;
    }
    if (isa<ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
            ttg::MemDescReshapeOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    if (def->hasTrait<OpTrait::MemDescViewTrait>()) {
      // Preserve the root for conservative aliasing, but do not pretend an
      // unsupported view has a known physical base.
      knownBase = false;
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return {v, knownBase ? std::optional<TMemBase>(base) : std::nullopt};
}

// Return true for a synchronization point that covers every warp in the
// current task. Producer/consumer mbarriers and partial named barriers are not
// sufficient: they do not rendezvous all warps that can share a TMEM lane.
static bool isFullTaskBarrier(Operation *op) {
  if (isa<ttg::BarrierOp>(op))
    return true;
  auto named = dyn_cast<ttng::NamedBarrierWaitOp>(op);
  if (!named)
    return false;

  APInt count;
  if (!matchPattern(named.getNumThreads(), m_ConstantInt(&count)))
    return false;
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  int taskThreads =
      ttg::lookupNumWarps(op) * ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  return count.getSExtValue() == taskThreads;
}

using WarpFootprint = SmallVector<llvm::DenseSet<uint64_t>>;

// Compute the 32-bit TMEM cells touched by each warp. The physical layout is
// shared with tcgen05 lowering, including its implicit warp-to-row mapping.
static LogicalResult computeWarpFootprint(RankedTensorType regTy,
                                          ttg::MemDescType tmemTy,
                                          TMemBase base,
                                          WarpFootprint &warpFp) {
  MLIRContext *ctx = regTy.getContext();
  auto physical = ttng::computeTMemLdStPhysicalLayout(regTy, tmemTy);
  if (failed(physical))
    return failure();
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kRow = StringAttr::get(ctx, "row");
  StringAttr kCol = StringAttr::get(ctx, "col");
  if (!llvm::is_contained(physical->getInDimNames(), kWarp) ||
      !llvm::is_contained(physical->getInDimNames(), kReg) ||
      !llvm::is_contained(physical->getInDimNames(), kLane) ||
      !llvm::is_contained(physical->getOutDimNames(), kRow) ||
      !llvm::is_contained(physical->getOutDimNames(), kCol))
    return failure();
  int nWarp = physical->getInDimSize(kWarp);
  int nReg = physical->getInDimSize(kReg);
  int nLane = physical->getInDimSize(kLane);
  int bitWidth = tmemTy.getElementTypeBitWidth();
  warpFp.assign(nWarp, {});
  SmallVector<std::pair<StringAttr, int32_t>> pt;
  for (StringAttr d : physical->getInDimNames())
    pt.push_back({d, 0});
  for (int w = 0; w < nWarp; ++w)
    for (int r = 0; r < nReg; ++r)
      for (int l = 0; l < nLane; ++l) {
        for (auto &p : pt)
          p.second = (p.first == kReg)    ? r
                     : (p.first == kLane) ? l
                     : (p.first == kWarp) ? w
                                          : 0;
        int64_t row = 0, col = 0;
        for (auto &o : physical->apply(pt)) {
          if (o.first == kRow)
            row = o.second;
          else if (o.first == kCol)
            col = o.second;
        }
        uint64_t physicalRow = base.row + row;
        uint64_t physicalCol = base.col + (col * bitWidth) / 32;
        warpFp[w].insert((physicalRow << 32) | physicalCol);
      }
  return success();
}

static bool hasCrossWarpOverlap(ttng::TMEMLoadOp rd, ttng::TMEMStoreOp st) {
  TMemView readView = resolveTMemView(rd.getSrc());
  TMemView writeView = resolveTMemView(st.getDst());
  if (readView.root != writeView.root)
    return false;
  if (!readView.base || !writeView.base)
    return true;

  WarpFootprint readFp, writeFp;
  if (failed(computeWarpFootprint(rd.getResult().getType(),
                                  rd.getSrc().getType(), *readView.base,
                                  readFp)) ||
      failed(computeWarpFootprint(st.getSrc().getType(), st.getDst().getType(),
                                  *writeView.base, writeFp)))
    return true;
  for (size_t a = 0; a < writeFp.size(); ++a)
    for (size_t b = 0; b < readFp.size(); ++b) {
      if (a == b)
        continue;
      for (uint64_t cell : writeFp[a])
        if (readFp[b].contains(cell))
          return true;
    }
  return false;
}

// Scan a block in program order. Multiple reads of one allocation remain live
// until a full-task barrier; a non-overlapping store does not prove that an
// earlier read has completed in another warp.
static unsigned processBlock(Block &block) {
  unsigned inserted = 0;
  DenseMap<Value, SmallVector<ttng::TMEMLoadOp>> liveReads;
  for (Operation &opRef : llvm::make_early_inc_range(block)) {
    Operation *op = &opRef;
    if (auto ld = dyn_cast<ttng::TMEMLoadOp>(op)) {
      Value root = resolveTMemView(ld.getSrc()).root;
      if (root)
        liveReads[root].push_back(ld);
    } else if (isFullTaskBarrier(op)) {
      liveReads.clear();
    } else if (auto st = dyn_cast<ttng::TMEMStoreOp>(op)) {
      Value root = resolveTMemView(st.getDst()).root;
      auto it = liveReads.find(root);
      if (it != liveReads.end()) {
        bool hazard = llvm::any_of(it->second, [&](ttng::TMEMLoadOp ld) {
          return hasCrossWarpOverlap(ld, st);
        });
        if (hazard) {
          OpBuilder b(op);
          ttg::BarrierOp::create(b, op->getLoc(), ttg::AddrSpace::All);
          ++inserted;
          liveReads.clear();
        }
      }
    }
    for (Region &region : op->getRegions())
      for (Block &nested : region)
        inserted += processBlock(nested);
  }
  return inserted;
}

} // namespace

class TritonNvidiaGPUInsertTMemAliasWARBarrierPass
    : public impl::TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase<
          TritonNvidiaGPUInsertTMemAliasWARBarrierPass> {
public:
  using TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase::
      TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (Region &region : mod->getRegions())
      for (Block &block : region)
        processBlock(block);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
