#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTELHSTOTMEMPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

enum class OpndAMemType { Unspecified, SMem, TMem };

static unsigned getUnsignedEnv(StringRef name, unsigned fallback) {
  std::string value = triton::tools::getStrEnv(name.str());
  unsigned parsed;
  if (value.empty() || StringRef(value).getAsInteger(10, parsed))
    return fallback;
  return parsed;
}

static SmallVector<uint64_t> enumerateMemorySpaceMasks(unsigned candidateCount,
                                                       uint64_t limit) {
  SmallVector<uint64_t> masks{0};
  for (unsigned i = 0; i < candidateCount && masks.size() < limit; ++i)
    masks.push_back(uint64_t{1} << i);
  const uint64_t total = uint64_t{1} << candidateCount;
  for (uint64_t mask = 1; mask < total && masks.size() < limit; ++mask) {
    if ((mask & (mask - 1)) != 0)
      masks.push_back(mask);
  }
  return masks;
}

static void dumpMemorySpaceCandidates(ArrayRef<uint64_t> masks,
                                      unsigned selected,
                                      unsigned candidateCount) {
  auto path = triton::tools::getStrEnv("TRITON_WS_SEARCH_MANIFEST");
  if (path.empty())
    return;
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Append);
  if (ec)
    return;
  for (unsigned rank = 0; rank < masks.size(); ++rank) {
    os << "{\"kind\": \"memory-space\", \"rank\": " << rank
       << ", \"selected\": " << (rank == selected ? "true" : "false")
       << ", \"candidate_count\": " << candidateCount << ", \"lhs_tmem\": [";
    bool first = true;
    for (unsigned i = 0; i < candidateCount; ++i) {
      if (!(masks[rank] & (uint64_t{1} << i)))
        continue;
      os << (first ? "" : ", ") << i;
      first = false;
    }
    os << "]}\n";
  }
}

/// Extract the memory type for opndA from a tt.autows annotation.
static OpndAMemType getOpndAMemType(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("tt.autows");
  if (!attr)
    return OpndAMemType::Unspecified;
  auto parsed = llvm::json::parse(attr.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return OpndAMemType::Unspecified;
  }
  auto *obj = parsed->getAsObject();
  if (!obj)
    return OpndAMemType::Unspecified;
  auto *channelsArr = obj->getArray("channels");
  if (!channelsArr)
    return OpndAMemType::Unspecified;
  for (auto &elem : *channelsArr) {
    auto str = elem.getAsString();
    if (!str)
      continue;
    StringRef channel = *str;
    if (!channel.consume_front("opndA,"))
      continue;
    StringRef memType = channel.take_front(channel.find(','));
    if (memType == "smem")
      return OpndAMemType::SMem;
    if (memType == "tmem")
      return OpndAMemType::TMem;
  }
  return OpndAMemType::Unspecified;
}

template <class MMAOpTy>
Attribute getLHSTMemLayout(MMAOpTy tcGen5MMAOp, gpu::MemDescType lhsTMEMType) {
  int numWarps = ttg::lookupNumWarps(tcGen5MMAOp);
  return nvidia_gpu::getDefaultLayoutForTmemLdSt(lhsTMEMType, numWarps);
}

template <class MMAOpTy> static bool hasTransposedLHSSibling(MMAOpTy mmaOp) {
  auto localAllocOp = mmaOp.getA().template getDefiningOp<ttg::LocalAllocOp>();
  if (!localAllocOp || !localAllocOp.getSrc())
    return false;
  Value src = localAllocOp.getSrc();
  for (Operation *srcUser : src.getUsers()) {
    auto otherAlloc = dyn_cast<ttg::LocalAllocOp>(srcUser);
    if (!otherAlloc)
      continue;
    for (Operation *allocUser : otherAlloc->getResult(0).getUsers()) {
      auto transOp = dyn_cast<ttg::MemDescTransOp>(allocUser);
      if (!transOp)
        continue;
      for (Operation *transUser : transOp->getResult(0).getUsers()) {
        if (auto otherMma = dyn_cast<TCGen5MMAOp>(transUser)) {
          if (otherMma.getA() == transOp->getResult(0))
            return true;
        } else if (auto otherScaled = dyn_cast<TCGen5MMAScaledOp>(transUser)) {
          if (otherScaled.getA() == transOp->getResult(0))
            return true;
        }
      }
    }
  }
  return false;
}

template <class MMAOpTy> static bool isLegalLHSConversion(MMAOpTy mmaOp) {
  auto lhs = mmaOp.getA();
  auto localAllocOp = lhs.template getDefiningOp<ttg::LocalAllocOp>();
  if (!localAllocOp || !localAllocOp.getSrc() ||
      localAllocOp->getParentRegion() != mmaOp->getParentRegion())
    return false;
  Value src = localAllocOp.getSrc();
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto accTMemEncoding =
      dyn_cast<TensorMemoryEncodingAttr>(mmaOp.getD().getType().getEncoding());
  if (!srcType || !accTMemEncoding)
    return false;
  unsigned elemBitWidth =
      lhs.getType().getElementType().getIntOrFloatBitWidth();
  if (!llvm::is_contained({8, 16, 32}, elemBitWidth) ||
      isFp4Padded(lhs.getType().getEncoding()))
    return false;
  auto cgaLayout = triton::gpu::getCGALayout(srcType.getEncoding());
  auto aTMemEncoding = TensorMemoryEncodingAttr::get(
      mmaOp->getContext(), accTMemEncoding.getBlockM(),
      lhs.getType().getShape()[1], /*colStride=*/1, cgaLayout,
      accTMemEncoding.getTwoCTAs(),
      accTMemEncoding.getCtaMode() == TensorMemoryCTAMode::TwoCTA_RHS
          ? TensorMemoryCTAMode::TwoCTA_LHS
          : accTMemEncoding.getCtaMode());
  auto lhsTMemType = ttg::MemDescType::get(
      lhs.getType().getShape(), lhs.getType().getElementType(), aTMemEncoding,
      TensorMemorySpaceAttr::get(mmaOp->getContext()),
      /*mutableMemory=*/false);
  return isDistributedLayoutTMemCompatible(mmaOp, srcType, lhsTMemType) ||
         !comesFromLoadOrBlockArg(src) ||
         triton::tools::getBoolEnv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION");
}

template <class MMAOpTy> class LHSToTMem : public OpRewritePattern<MMAOpTy> {
public:
  LHSToTMem(MLIRContext *context, const DenseSet<Operation *> *searchPromotions)
      : OpRewritePattern<MMAOpTy>(context), searchPromotions(searchPromotions) {
  }

  LogicalResult matchAndRewrite(MMAOpTy tcGen5MMAOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = tcGen5MMAOp->getContext();
    Location loc = tcGen5MMAOp.getLoc();
    auto lhs = tcGen5MMAOp.getA();
    auto localAllocOp = lhs.template getDefiningOp<ttg::LocalAllocOp>();
    if (!localAllocOp)
      return failure();
    // Limit the liverange of the TMem allocations to single block.
    if (localAllocOp->getParentRegion() != tcGen5MMAOp->getParentRegion())
      return failure();
    Value src = localAllocOp.getSrc();
    // Check tt.autows annotation for explicit opndA memory type.
    // If annotated as "smem", skip promotion. If "tmem", promote directly
    // (skip the transposed-shared-source heuristic). If no annotation,
    // fall through to the heuristic.
    const OpndAMemType opndAMem = getOpndAMemType(tcGen5MMAOp);
    if (opndAMem == OpndAMemType::SMem)
      return failure();
    const bool annotatedTmem = opndAMem == OpndAMemType::TMem;
    const bool searchPromoted =
        searchPromotions && searchPromotions->contains(tcGen5MMAOp);

    // If the same source value is also allocated and transposed for use as
    // operand A of another gen5 MMA, skip promotion. The transposed path
    // cannot be promoted to tmem, so keeping both in smem avoids a redundant
    // tmem allocation and copy for the same data. This covers both:
    //   1. Same local_alloc used directly + through memdesc_trans
    //   2. Separate local_allocs from the same src, one transposed
    if (!annotatedTmem && !searchPromoted &&
        hasTransposedLHSSibling(tcGen5MMAOp))
      return failure();
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcLayout = srcType.getEncoding();
    auto accTMemEncoding = dyn_cast<TensorMemoryEncodingAttr>(
        tcGen5MMAOp.getD().getType().getEncoding());
    auto cgaLayout = triton::gpu::getCGALayout(srcLayout);
    // TMem encoding for A operand is the same as for D (Acc), with colStride 1,
    // i.e. densely packed in TMEM.
    unsigned elemBitWidth =
        lhs.getType().getElementType().getIntOrFloatBitWidth();
    if (!llvm::is_contained({8, 16, 32}, elemBitWidth)) {
      return failure();
    }
    // Padded fp4 operand cannot be trivially promoted to TMEM.
    if (isFp4Padded(lhs.getType().getEncoding())) {
      return failure();
    }
    const unsigned colStride = 1;
    auto aTMemEncoding = TensorMemoryEncodingAttr::get(
        context, accTMemEncoding.getBlockM(), lhs.getType().getShape()[1],
        colStride, cgaLayout, accTMemEncoding.getTwoCTAs(),
        accTMemEncoding.getCtaMode() == TensorMemoryCTAMode::TwoCTA_RHS
            ? TensorMemoryCTAMode::TwoCTA_LHS
            : accTMemEncoding.getCtaMode());
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    ttg::MemDescType lhsMemDescType = ttg::MemDescType::get(
        lhs.getType().getShape(), lhs.getType().getElementType(), aTMemEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    bool layoutTmemCompatible =
        isDistributedLayoutTMemCompatible(tcGen5MMAOp, srcType, lhsMemDescType);
    Attribute newLayout = srcLayout;
    if (!layoutTmemCompatible) {
      if (!comesFromLoadOrBlockArg(src) ||
          triton::tools::getBoolEnv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION")) {
        newLayout = getLHSTMemLayout(tcGen5MMAOp, lhsMemDescType);
      } else {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(localAllocOp);
    if (newLayout != srcLayout) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy = ty.cloneWithEncoding(newLayout);
      src = ttg::ConvertLayoutOp::create(rewriter, loc, newTy, src);
    }
    Value tMemAlloc = TMEMAllocOp::create(rewriter, loc, lhsMemDescType, src);
    if (searchPromoted)
      tMemAlloc.getDefiningOp()->setAttr("allocation.memorySpaceSearch",
                                         UnitAttr::get(context));
    tcGen5MMAOp.getAMutable().assign(tMemAlloc);
    return success();
  }

private:
  const DenseSet<Operation *> *searchPromotions;
};
} // namespace

class TritonNvidiaGPUPromoteLHSToTMemPass
    : public impl::TritonNvidiaGPUPromoteLHSToTMemPassBase<
          TritonNvidiaGPUPromoteLHSToTMemPass> {
public:
  using TritonNvidiaGPUPromoteLHSToTMemPassBase<
      TritonNvidiaGPUPromoteLHSToTMemPass>::
      TritonNvidiaGPUPromoteLHSToTMemPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Rank zero preserves the existing heuristic. Higher ranks enumerate
    // subsets of LHS operands rejected solely because a transposed sibling
    // consumes the same source. These are legal-but-resource-sensitive choices
    // such as FA backward's direct dsT path; runtime search decides whether the
    // extra TMEM copy is worthwhile.
    SmallVector<Operation *> candidates;
    auto collectCandidate = [&](auto mmaOp) {
      if (getOpndAMemType(mmaOp) != OpndAMemType::Unspecified)
        return;
      if (!isLegalLHSConversion(mmaOp))
        return;
      if (hasTransposedLHSSibling(mmaOp))
        candidates.push_back(mmaOp);
    };
    m.walk([&](Operation *op) {
      if (auto mma = dyn_cast<TCGen5MMAOp>(op))
        collectCandidate(mma);
      else if (auto scaled = dyn_cast<TCGen5MMAScaledOp>(op))
        collectCandidate(scaled);
    });

    unsigned topK =
        std::max(1u, getUnsignedEnv("TRITON_WS_MEMORY_SPACE_TOPK", 1));
    unsigned pick = getUnsignedEnv("TRITON_WS_MEMORY_SPACE_PICK", 0);
    unsigned searchable = std::min<unsigned>(candidates.size(), 20);
    uint64_t total = uint64_t{1} << searchable;
    uint64_t retained =
        std::min<uint64_t>(total, std::max<uint64_t>(topK, uint64_t{pick} + 1));
    SmallVector<uint64_t> masks =
        enumerateMemorySpaceMasks(searchable, retained);
    unsigned selected = std::min<unsigned>(pick, masks.size() - 1);
    dumpMemorySpaceCandidates(masks, selected, searchable);
    DenseSet<Operation *> searchPromotions;
    for (unsigned i = 0; i < searchable; ++i)
      if (masks[selected] & (uint64_t{1} << i))
        searchPromotions.insert(candidates[i]);

    RewritePatternSet patterns(context);
    patterns.add<LHSToTMem<TCGen5MMAOp>>(context, &searchPromotions);
    patterns.add<LHSToTMem<TCGen5MMAScaledOp>>(context, &searchPromotions);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
