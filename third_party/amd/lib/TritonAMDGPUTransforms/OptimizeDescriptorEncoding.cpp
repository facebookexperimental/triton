#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

#define DEBUG_TYPE "tritonamdgpu-optimize-descriptor-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::PaddedSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEncPadded(Value loadedValue,
                                      const TargetFeatures &targetFeatures) {
  ttg::PaddedSharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::PaddedSharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::PaddedSharedEncodingAttr>(memDesc.getEncoding());
      auto newAttr = getSharedEncIfAllUsersAreDotEncPadded(user->getResult(0),
                                                           targetFeatures);

      if (!newAttr.has_value())
        return std::nullopt;

      auto interval = tempAttr.getIntervals()[0];
      auto padding = tempAttr.getPaddings()[0];
      // Update interval and padding if we find a compatible shared encoding
      // down the chain
      if (newAttr.has_value() && *newAttr) {
        interval = newAttr->getIntervals()[0];
        padding = newAttr->getPaddings()[0];
      }

      tempAttr = ttg::PaddedSharedEncodingAttr::get(
          tempAttr.getContext(), interval, padding,
          tempAttr.getLinearComponent());
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto order = getOrderForMemory(srcTy);
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOperand conversions support
      // arbitrary shared memory ordering
      if (rank == 3) {
        // Move the batch dimension (dim #0) to be the last so that it will be
        // the slowest varying dimension.
        for (unsigned i = 0; i < rank; ++i)
          if (order[i] != 0)
            sharedOrder.emplace_back(order[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = order;
      }

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        // For async descriptor loads, enable padding.
        tempAttr =
            composePaddedLayout(targetFeatures, dotOpEnc.getOpIdx(),
                                dotOpEnc.getKWidth(), srcTy, sharedOrder);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto dotEnc = getDotEncoding<ttg::AMDWmmaEncodingAttr>(
                userResult, &opIdx, &vecSize)) {
          tempAttr =
              composePaddedLayout(targetFeatures, opIdx, vecSize, srcTy, order);
        }
      }
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}
} // anonymous namespace

namespace mlir {

// Walk the uses of descriptor loads and find a favorable encoding to use.
// Attach the desired encoding as a discardable attribute to descriptor loads.
// assignMemoryLayouts will propagate this attribute to rest of the descriptors
static void computeDesiredEncodingAttr(mlir::ModuleOp &m) {
  auto targetFeatures = TargetFeatures::fromModuleOp(m);
  for (auto f : m.getOps<tt::FuncOp>()) {
    f.walk([&](tt::DescriptorLoadOp load) {
      auto paddedEncoding =
          getSharedEncIfAllUsersAreDotEncPadded(load, targetFeatures);
      if (paddedEncoding) {
        load->setDiscardableAttr("tt.desired_encoding", *paddedEncoding);
        LDBG("Desired encoding: " << *paddedEncoding);
      }
    });
  }
}

// TLX kernels emit `amdgpu.async_tdm_*` directly, bypassing
// `tt.descriptor_load` / `tt.descriptor_store`. The memdesc operand
// (destination for loads, source for stores) carries the encoding chosen by
// TLX (e.g. WMMA-tuned `composePaddedLayout` when feeding `tt.dot`). Without
// any propagation, the descriptor's `TensorDescType` keeps the fallback
// encoding from `AssignDescriptorMemoryLayouts`, while the alloc gets the
// TLX-picked encoding. The TDM hardware lowering in `LoadStoreOpToLLVM` reads
// stride from the descriptor type but reads/writes the alloc — a stride
// mismatch causes out-of-bounds LDS access.
//
// This pass copies the memdesc encoding back to the descriptor type so the
// two sides agree by construction. If multiple TDM ops share a descriptor
// with conflicting memdesc encodings, we error out (no good way to pick one
// over the other; TLX kernels currently never hit this).
static LogicalResult alignTDMDescriptorEncodings(mlir::ModuleOp &m) {
  llvm::DenseMap<Value, Attribute> descToEncoding;

  auto record = [&](Operation *op, Value desc,
                    Attribute encoding) -> WalkResult {
    auto [it, inserted] = descToEncoding.try_emplace(desc, encoding);
    if (!inserted && it->second != encoding) {
      op->emitError() << "TDM ops using the same descriptor require "
                         "conflicting memdesc layouts";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  WalkResult result = m.walk([&](Operation *op) {
    if (auto load = dyn_cast<tt::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
      auto memDescTy = cast<ttg::MemDescType>(load.getResult().getType());
      return record(load, load.getDesc(), memDescTy.getEncoding());
    }
    if (auto store = dyn_cast<tt::amdgpu::AsyncTDMCopyLocalToGlobalOp>(op)) {
      auto memDescTy = cast<ttg::MemDescType>(store.getSrc().getType());
      return record(store, store.getDesc(), memDescTy.getEncoding());
    }
    // tdm_prefetch carries no memdesc and so cannot anchor an encoding;
    // it piggy-backs on whatever the corresponding load/store decided.
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  for (auto [desc, encoding] : descToEncoding) {
    auto descTy = cast<tt::TensorDescType>(desc.getType());
    auto blockTy = descTy.getBlockType();
    // Adjust order/CGA fields of paddedEncoding/swizzled/nvmma to the
    // descriptor's block shape so a future rank-reducing TDM doesn't desync.
    auto sharedEnc = cast<ttg::SharedEncodingTrait>(encoding);
    Attribute fittedEnc =
        ttg::updateEncodingForShape(desc.getDefiningOp(), sharedEnc, blockTy);
    desc.setType(tt::TensorDescType::get(blockTy.getShape(),
                                         blockTy.getElementType(), fittedEnc));
  }

  auto ctx = m.getContext();
  for (auto func : m.getOps<tt::FuncOp>()) {
    SmallVector<Type> argTypes(func.getBlocks().front().getArgumentTypes());
    SmallVector<Type> resultTypes(func.getResultTypes());
    func.setFunctionType(FunctionType::get(ctx, argTypes, resultTypes));
  }
  return success();
}

class AMDGPUAssignDescriptorMemoryLayouts
    : public ttg::AssignDescriptorMemoryLayouts {
public:
  AMDGPUAssignDescriptorMemoryLayouts() = default;
  ~AMDGPUAssignDescriptorMemoryLayouts() override = default;

private:
  Attribute buildFallbackSharedEncoding(mlir::MLIRContext *ctx,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<unsigned> order,
                                        ttg::CGAEncodingAttr cgaLayout,
                                        Type elementType) override;
  bool isCompatibleSharedEncoding(Attribute enc) override;
};

Attribute AMDGPUAssignDescriptorMemoryLayouts::buildFallbackSharedEncoding(
    mlir::MLIRContext *ctx, ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
    ttg::CGAEncodingAttr cgaLayout, Type elementType) {
  return buildDefaultTDMDescriptorEncoding(ctx, shape, order, cgaLayout,
                                           elementType);
}

bool AMDGPUAssignDescriptorMemoryLayouts::isCompatibleSharedEncoding(
    Attribute enc) {
  return isa<ttg::PaddedSharedEncodingAttr, ttg::SwizzledSharedEncodingAttr>(
      enc);
}

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDESCRIPTORENCODING
#include "TritonAMDGPUTransforms/Passes.h.inc"

// This pass assigns encoding to each descriptor in the function. Descriptors
// are created using `tl.make_tensor_descriptor` or passed in as arguments to
// the kernel. They are used by TDM load/store/gather/scatter. We assign
// shared memory encoding (e.g., padded) to the descriptors and use it for
// deriving encodings on descriptor ops including load/store/gather/scatter.
// The pass works in two phases: First, we derive a favorable encoding for
// each descriptor based on its uses (e.g., load -> tt.dot) and store it as
// EncodingInfo for each descriptor. The EncodingInfo is propagated to other
// desc descriptors through fixed point iteration. Finally, the computed
// EncodingInfo is fully materialized and assigned to the descriptors.
// Example:
//   %d = tt.make_tensor_descriptor ...   ; no encoding yet
//   %r = scf.for ... iter_args(%di = %d) -> ... {
//     %x = tt.descriptor_load %di ... ; use gives desired encoding for %di
//     %y = tt.dot %x, %b, %c           ; encoding from dot
//     scf.yield %di                    ; same encoding is propagated
//   }
//
class TritonAMDGPUOptimizeDescriptorEncodingPass
    : public impl::TritonAMDGPUOptimizeDescriptorEncodingBase<
          TritonAMDGPUOptimizeDescriptorEncodingPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    computeDesiredEncodingAttr(m);

    AMDGPUAssignDescriptorMemoryLayouts assignMemoryLayouts;
    assignMemoryLayouts.assignMemoryLayouts(m);

    if (failed(alignTDMDescriptorEncodings(m))) {
      signalPassFailure();
      return;
    }

    // Remove temporary discardable attributes used during encoding assignment
    for (auto f : m.getOps<tt::FuncOp>()) {
      f.walk([](tt::DescriptorLoadOp load) {
        load->removeDiscardableAttr("tt.desired_encoding");
      });
    }
  }
};

} // namespace mlir
