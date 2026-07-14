#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUWARNTMEMALIASWARPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Trace a TMEM memdesc value back to its root allocation, following
// metadata-only view ops (memdesc_reinterpret / _index / _subslice / _trans)
// and warp_specialize partition captures. Two accesses whose roots are the
// same SSA value refer to the same physical TMEM (e.g. aliased qk/P or dp/dsT
// declared via storage_alias and lowered to memdesc_reinterpret of one alloc).
static Value getTmemRoot(Value v) {
  while (v) {
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Operation *parent = ba.getOwner()->getParentOp();
      if (auto part = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
        v = part->getOperand(ba.getArgNumber());
        continue;
      }
      break;
    }
    Operation *def = v.getDefiningOp();
    if (def && def->hasTrait<OpTrait::MemDescViewTrait>()) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

// A true intra-CTA execution barrier (bar.sync) that orders the partition's
// warps against each other. mbarrier arrive/wait (ttng.*_barrier) are async
// producer/consumer signals and do NOT order the warps within a task.
static bool isOrderingBarrier(Operation *op) {
  return isa<ttg::BarrierOp, ttng::NamedBarrierWaitOp>(op);
}

// Per-warp footprint of a tcgen05 access in physical TMEM.
// Phi = regLayout.invertAndCompose(tmemLayout) maps (register, lane, warp[,
// block]) -> (row, col). When `cellMode` is true the footprint is the set of
// (row, col) cells (encoded row*nCol+col); otherwise it is the set of rows
// only. Rows are physical TMEM lanes (0-127) and share a common frame across
// element types (dtype only repacks columns), so row sets are comparable even
// when the two accesses have different dtypes; columns are only comparable when
// the two accesses share the same memdesc frame (same type / no relative
// offset), which is what `cellMode` gates on.
static LogicalResult
computeWarpFootprint(RankedTensorType regTy, ttg::MemDescType tmemTy,
                     bool cellMode,
                     SmallVectorImpl<llvm::DenseSet<int64_t>> &warpFp) {
  MLIRContext *ctx = regTy.getContext();
  LinearLayout reg = ttg::toLinearLayout(regTy);
  LinearLayout tmem = ttg::toLinearLayout(tmemTy);
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kRow = StringAttr::get(ctx, "row");
  StringAttr kCol = StringAttr::get(ctx, "col");
  if (!llvm::is_contained(reg.getInDimNames(), kWarp) ||
      !llvm::is_contained(reg.getInDimNames(), kReg) ||
      !llvm::is_contained(reg.getInDimNames(), kLane) ||
      !llvm::is_contained(tmem.getInDimNames(), kRow))
    return failure();
  LinearLayout phi = reg.invertAndCompose(tmem);
  int nWarp = phi.getInDimSize(kWarp);
  int nReg = phi.getInDimSize(kReg);
  int nLane = phi.getInDimSize(kLane);
  int64_t nCol = phi.getOutDimSize(kCol);
  warpFp.assign(nWarp, {});
  SmallVector<std::pair<StringAttr, int32_t>> pt;
  for (StringAttr d : phi.getInDimNames())
    pt.push_back({d, 0});
  for (int w = 0; w < nWarp; ++w)
    for (int r = 0; r < nReg; ++r)
      for (int l = 0; l < nLane; ++l) {
        for (auto &p : pt)
          p.second = (p.first == kReg) ? r
                     : (p.first == kLane) ? l
                     : (p.first == kWarp) ? w
                                          : 0;
        int64_t row = 0, col = 0;
        for (auto &o : phi.apply(pt)) {
          if (o.first == kRow)
            row = o.second;
          else if (o.first == kCol)
            col = o.second;
        }
        warpFp[w].insert(cellMode ? row * nCol + col : row);
      }
  return success();
}

// The commenter's exact condition: a cross-warp write-after-read exists iff some
// warp's store footprint overlaps a *different* warp's read footprint. If every
// warp only overwrites what it itself read, the reuse is warp-local and safe.
// Same-frame accesses (identical memdesc type, e.g. an accumulator
// read-modify-write) are compared at (row, col) cell granularity; different-
// frame aliases (e.g. an f32 qk region reused as an f16 P region, whose column
// frames differ by an unmodeled offset) fall back to a conservative row-level
// comparison.
static bool hasCrossWarpOverlap(ttng::TMEMLoadOp rd, ttng::TMEMStoreOp st) {
  bool sameFrame = rd.getSrc().getType() == st.getDst().getType();
  SmallVector<llvm::DenseSet<int64_t>> readFp, writeFp;
  if (failed(computeWarpFootprint(rd.getResult().getType(), rd.getSrc().getType(),
                                  sameFrame, readFp)) ||
      failed(computeWarpFootprint(st.getSrc().getType(), st.getDst().getType(),
                                  sameFrame, writeFp)))
    return false; // can't prove a hazard -> stay silent
  int n = std::min(readFp.size(), writeFp.size());
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      if (a == b)
        continue;
      for (int64_t cell : writeFp[a])
        if (readFp[b].contains(cell))
          return true;
    }
  return false;
}

static void findHazards(Block &block,
                        function_ref<void(Operation *, Operation *)> report) {
  DenseMap<Value, Operation *> liveReadOp; // TMEM root -> last unbarriered read
  for (Operation &opRef : block) {
    Operation *op = &opRef;
    if (auto ld = dyn_cast<ttng::TMEMLoadOp>(op)) {
      liveReadOp[getTmemRoot(ld.getSrc())] = op;
    } else if (isOrderingBarrier(op)) {
      liveReadOp.clear();
    } else if (auto st = dyn_cast<ttng::TMEMStoreOp>(op)) {
      Value root = getTmemRoot(st.getDst());
      auto it = liveReadOp.find(root);
      if (it != liveReadOp.end()) {
        auto rd = cast<ttng::TMEMLoadOp>(it->second);
        if (hasCrossWarpOverlap(rd, st))
          report(op, it->second);
        liveReadOp.erase(it); // region overwritten; drop the pending read
      }
    }
    for (Region &region : op->getRegions())
      for (Block &nested : region)
        findHazards(nested, report);
  }
}

static std::string hazardMessage(Operation *store, Operation *read) {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "aliased-TMEM write-after-read hazard: this tcgen05 store overwrites "
        "a TMEM region that another warp reads earlier in the same task "
        "(cross-warp footprint overlap) with no intervening barrier; under warp "
        "skew a fast warp's store can clobber a chunk a slow warp is still "
        "reading. Insert a task-scoped barrier between the read and the store. "
        "[store: "
     << store->getLoc() << "; read: " << read->getLoc() << "]";
  return msg;
}

} // namespace

// Public: collect warning strings (CudaWarnings-style; surfaced from Python via
// warnings.warn). Does not modify the IR.
std::vector<std::string> collectTMemAliasWARWarnings(ModuleOp module) {
  std::vector<std::string> warnings;
  for (Region &region : module->getRegions())
    for (Block &block : region)
      findHazards(block, [&](Operation *st, Operation *rd) {
        warnings.push_back(hazardMessage(st, rd));
      });
  return warnings;
}

class TritonNvidiaGPUWarnTMemAliasWARPass
    : public impl::TritonNvidiaGPUWarnTMemAliasWARPassBase<
          TritonNvidiaGPUWarnTMemAliasWARPass> {
public:
  using TritonNvidiaGPUWarnTMemAliasWARPassBase::
      TritonNvidiaGPUWarnTMemAliasWARPassBase;

  // In-IR diagnostics for lit testing (-verify-diagnostics).
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (Region &region : mod->getRegions())
      for (Block &block : region)
        findHazards(block, [](Operation *st, Operation *rd) {
          InFlightDiagnostic diag = st->emitWarning() << hazardMessage(st, rd);
          diag.attachNote(rd->getLoc()) << "the aliased TMEM read is here";
        });
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
