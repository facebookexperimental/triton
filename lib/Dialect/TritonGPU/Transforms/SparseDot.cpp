#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::SharedEncodingTrait;

// The functions below are defined in AccelerateMatmul.cpp.
namespace mlir::triton::gpu {
SmallVector<unsigned, 3>
getWarpsPerTile(Operation *dotOp, ArrayRef<int64_t> shape, int version,
                int numWarps, const SmallVector<unsigned, 3> &instrShape);
int computeOrigBitWidth(Value x);
Value getSharedMemMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                             int opIdx, bool allowTranspose);
} // namespace mlir::triton::gpu

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_SPARSEBLOCKEDTOMMAPASS
#define GEN_PASS_DEF_SPARSEREMOVELAYOUTCONVERSIONPASS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

constexpr int kThreadsPerWarp = 32;
// Each 16x16 original sparse matrix tile requires 16 metadata values of
// 16-bit size, where the first thread (T0) in each 4-thread group holds two
// such values in a register (32-bit).
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-matrix-storage
constexpr int kTileSize = 16;
constexpr int kMetaElementsBitSize = 2;
// Metadata elements are packed into 16-bits values.
constexpr int kMetaElementsPerPackedValue = 16 / kMetaElementsBitSize;
constexpr int kColumnsPerCtaTile = kTileSize / kMetaElementsPerPackedValue;

// TODO(sparsity) upstream this
namespace {
std::optional<int> maybeLookupAttr(Operation *op, const char *AttrName) {
  if (isa<ModuleOp, FuncOp>(op)) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(AttrName))
      return attr.getInt();
  } else if (auto partitions =
                 dyn_cast<WarpSpecializePartitionsOp>(op->getParentOp())) {
    unsigned idx = op->getParentRegion()->getRegionNumber();
    return partitions.getParentOp().getPartitionNumWarps()[idx];
  }
  if (Operation *parent = op->getParentOp())
    return maybeLookupAttr(parent, AttrName);
  return {};
}

int lookupNumThreadsPerWarp(Operation *op) {
  std::optional<int> threadsPerWarp =
      maybeLookupAttr(op, AttrNumThreadsPerWarp);
  if (!threadsPerWarp) {
    op->emitOpError("is not contained within a context that specifies the "
                    "number of threads per warp");
    llvm::report_fatal_error(
        "failed to lookup the number of threads per warp, the "
        "surrounding module should contain a " +
        Twine(AttrNumWarpsName) + " attribute");
  }
  return *threadsPerWarp;
}

int lookupNumCTAs(Operation *op) {
  std::optional<int> numCTAs = maybeLookupAttr(op, AttrNumCTAsName);
  if (!numCTAs) {
    op->emitOpError(
        "is not contained within a context that specifies the number of ctas");
    llvm::report_fatal_error("failed to lookup the number of ctas, the "
                             "surrounding module should contain a " +
                             Twine(AttrNumWarpsName) + " attribute");
  }
  return *numCTAs;
}
} // namespace

class SparseBlockedToMMA : public RewritePattern {
  using ConvertLayoutOp = triton::gpu::ConvertLayoutOp;
  using NvidiaMmaEncodingAttr = triton::gpu::NvidiaMmaEncodingAttr;

public:
  SparseBlockedToMMA(MLIRContext *context, int compute_capability)
      : RewritePattern(SparseDotOp::getOperationName(), 2, context),
        compute_capability_(compute_capability) {}

  // TODO(sparsity): this comes from AccelerateMatmul.cpp.
  //                 Maybe combine it.
  //                 Otherwise switch everything to camelCase like upstream.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto dotOp = cast<SparseDotOp>(op);
    auto context = op->getContext();
    llvm::outs().flush();
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    // Check data-types and SM compatibility
    RankedTensorType ret_type = dotOp.getType();
    if (!ret_type.getEncoding() ||
        isa<NvidiaMmaEncodingAttr>(ret_type.getEncoding()))
      return failure();

    assert(compute_capability_ >= 80 &&
           "SparseDot is only supported on Ampere or higher");
    bool allow_v3 = !triton::tools::getBoolEnv("DISABLE_MMA_V3");
    // Sparse dot is supported for MMA v2 and v3 only, and sm100/sm120 should
    // use MMA v2 (v3 is Hopper-only).
    int triton_mma_version = compute_capability_ == 90 && allow_v3 ? 3 : 2;

    // get MMA encoding and new return type given the number of warps
    auto ret_shape_per_cta = triton::gpu::getShapePerCTA(ret_type);
    auto mod = op->getParentOfType<ModuleOp>();
    int num_warps = lookupNumWarps(mod);
    auto cta_layout = triton::gpu::getCTALayout(ret_type.getEncoding());

    auto instr_shape =
        mmaVersionToInstrShape(triton_mma_version, ret_shape_per_cta,
                               getElementTypeOrSelf(a.getType()), num_warps);
    auto warps_per_tile = mlir::triton::gpu::getWarpsPerTile(
        dotOp, ret_shape_per_cta, triton_mma_version, num_warps, instr_shape);
    NvidiaMmaEncodingAttr mma_enc = NvidiaMmaEncodingAttr::get(
        context, triton_mma_version, /*versionMinor=*/0, warps_per_tile,
        cta_layout, instr_shape);
    auto newRetType = RankedTensorType::get(ret_type.getShape(),
                                            ret_type.getElementType(), mma_enc);

    // convert accumulator
    auto acc = dotOp.getOperand(2);
    auto new_acc =
        rewriter.create<ConvertLayoutOp>(acc.getLoc(), newRetType, acc);

    // convert metadata
    Value meta = dotOp.getAMeta();
    auto meta_type = cast<RankedTensorType>(meta.getType());
    // TODO(sparsity) look more closely at CTA Layout. It should match A's CTA
    // Layout?
    meta_type =
        RankedTensorType::get(meta_type.getShape(), meta_type.getElementType(),
                              SparseDotMetaEncodingAttr::get(context, mma_enc));
    meta = rewriter.create<ConvertLayoutOp>(meta.getLoc(), meta_type, meta);

    Operation *newDot = nullptr;

    if (triton_mma_version == 2) { // MMAV2
      int min_bit_width = std::min(triton::gpu::computeOrigBitWidth(a),
                                   triton::gpu::computeOrigBitWidth(b));
      int k_width = 32 / min_bit_width;

      // convert A operand
      auto new_a_encoding =
          DotOperandEncodingAttr::get(context, 0, mma_enc, k_width);
      auto a_type = cast<RankedTensorType>(a.getType());
      a_type = RankedTensorType::get(a_type.getShape(), a_type.getElementType(),
                                     new_a_encoding);
      a = rewriter.create<ConvertLayoutOp>(a.getLoc(), a_type, a);

      // convert B operand
      auto new_b_encoding =
          DotOperandEncodingAttr::get(context, 1, mma_enc, k_width);
      auto b_type = cast<RankedTensorType>(b.getType());
      b_type = RankedTensorType::get(b_type.getShape(), b_type.getElementType(),
                                     new_b_encoding);
      b = rewriter.create<ConvertLayoutOp>(b.getLoc(), b_type, b);

      newDot = rewriter.create<SparseDotOp>(dotOp.getLoc(), newRetType, a, b,
                                            new_acc, meta);
    } else { // MMAV3
      assert(triton_mma_version == 3 &&
             "Sparsity is only supported with MMAV2 or higher");
      auto elt_type = dotOp.getA().getType().getElementType();
      // In MMAV3 transpose is only supported for f16 and bf16.
      bool allow_transpose = elt_type.isF16() || elt_type.isBF16();
      // Shared memory allocations that will be used by the dot op.
      a = triton::gpu::getSharedMemMMAOperand(a, rewriter, 0, allow_transpose);
      b = triton::gpu::getSharedMemMMAOperand(b, rewriter, 1, allow_transpose);
      newDot = rewriter.create<triton::nvidia_gpu::SparseWarpGroupDotOp>(
          dotOp.getLoc(), newRetType, a, b, new_acc, meta);
    }

    // convert back to return type
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, ret_type,
                                                 newDot->getResult(0));
    return success();
  }

private:
  int compute_capability_;
};

struct SparseBlockedToMMAPass
    : public impl::SparseBlockedToMMAPassBase<SparseBlockedToMMAPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    auto compute_capability = getNVIDIAComputeCapability(module);
    auto pattern =
        std::make_unique<SparseBlockedToMMA>(context, compute_capability);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

struct SparseRemoveLayoutConversionPass
    : public impl::SparseRemoveLayoutConversionPassBase<
          SparseRemoveLayoutConversionPass> {
  void runOnOperation() override {
    getOperation().walk([&](triton::gpu::ConvertLayoutOp op) {
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      // Skip if the source is already in shared memory.
      auto src_encoding =
          cast<RankedTensorType>(op.getSrc().getType()).getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(src_encoding)) {
        return;
      }
      auto dst_type = cast<RankedTensorType>(op.getType());
      // Skip if the destination is not a sparse dot meta.
      if (!isa<SparseDotMetaEncodingAttr>(dst_type.getEncoding())) {
        return;
      }

      // TODO(sparsity): is src_encoding always DistributedEncodingTrait?
      auto order =
          cast<DistributedEncodingTrait>(src_encoding).getDefaultOrder();

      auto shared_layout =
          builder.getAttr<triton::gpu::SwizzledSharedEncodingAttr>(
              // Packing metadata elements together. No swizzling.
              /*vec=*/kMetaElementsPerPackedValue, /*perPhase=*/1,
              /*maxPhase=*/1,
              /*order=*/order, triton::gpu::getCTALayout(src_encoding));
      auto mem_type = triton::gpu::MemDescType::get(
          dst_type.getShape(), dst_type.getElementType(), shared_layout,
          builder.getAttr<triton::gpu::SharedMemorySpaceAttr>());
      Value alloc =
          builder.create<triton::gpu::LocalAllocOp>(mem_type, op.getSrc());
      Value convert = builder.create<triton::gpu::LocalLoadOp>(dst_type, alloc);
      op.replaceAllUsesWith(convert);
      op.erase();
    });
  }
};

// TODO(sparsity): do we need SparseLocalLoadToLLVM (from upstream patches)?

} // namespace gpu
} // namespace triton
} // namespace mlir
