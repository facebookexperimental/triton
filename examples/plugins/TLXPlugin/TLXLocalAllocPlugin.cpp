/// TLX memory ops ported to Triton extension plugin custom ops.
///
/// This plugin implements the core local memory operations:
///   - tlx_local_alloc:            ttg::LocalAllocOp (SMEM) with default layout
///   - tlx_local_alloc_tmem:       ttng::TMEMAllocOp (TMEM / Blackwell)
///   - tlx_local_view:             ttg::MemDescIndexOp (subview into buffer)
///   - tlx_local_store:            ttg::LocalStoreOp (register -> SMEM)
///   - tlx_local_load:             ttg::LocalLoadOp (SMEM -> register)
///
/// The composite custom op `tlx_local_alloc` constructs the shared memory
/// encoding attribute (SwizzledSharedEncoding for 1D, NVMMASharedEncoding
/// for 2D+) and the MemDescType internally, so the Python DSL doesn't need
/// `make_swizzled_shared_encoding_attr` or `make_nv_mma_shared_encoding_attr`
/// on TritonOpBuilder.
///
/// The alias/storageAlias paths for local_alloc require TLX dialect ops
/// (LocalAliasOp, StorageAliasLocalAllocOp) and need a full Dialect Plugin.
///
/// TMEM paths for local_load (TMEMLoadOp) and local_store (TMEMStoreOp)
/// also require TLX dialect ops (RequireLayoutOp, ReleaseLayoutOp) and
/// need a full Dialect Plugin.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"
#include <unordered_map>

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// ---------------------------------------------------------------------------
// Helper: extract a constant integer from an arith.constant Value
// ---------------------------------------------------------------------------

static std::optional<int64_t> extractConstantInt(mlir::Value v) {
  if (auto constIntOp =
          mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(
              v.getDefiningOp()))
    return constIntOp.value();
  if (auto constOp =
          mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(v.getDefiningOp())) {
    if (auto intAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

// ===========================================================================
// tlx_local_alloc — SMEM allocation with automatic layout encoding
//
// This composite custom op creates a LocalAllocOp with the appropriate shared
// memory layout encoding constructed internally. The Python DSL passes the
// element type via a type-carrier scalar value and the full shape (including
// the num-buffers dimension) as i32 constants.
//
// Layout selection:
//   - Per-buffer rank == 1 (full shape has 2 dims): SwizzledSharedEncoding
//     with default parameters (vec=1, perPhase=1, maxPhase=1)
//   - Per-buffer rank >= 2 (full shape has 3+ dims): NVMMASharedEncoding
//     using the shape-based builder that computes swizzling automatically
//
// Operand convention:
//   operands[0] = result slot (overwritten with LocalAllocOp result)
//   operands[1] = type carrier — a scalar constant whose Type determines
//                 the element type (e.g., arith.constant 0.0 : f16)
//   operands[2..N] = full shape dimensions as i32 constants
//                    (first dim is num-buffers, rest is per-buffer shape)
// ===========================================================================

static TritonPluginResult
createLocalAllocSmem(const char *handle, TritonOpBuilder &self,
                     std::vector<mlir::Value> &operands) {
  if (operands.size() < 4) // result + type_carrier + at least 2 shape dims
    return TP_GENERIC_FAILURE;

  // Extract element type from the type-carrier value
  mlir::Type elemType = operands[1].getType();

  // Extract full shape from remaining operands
  llvm::SmallVector<int64_t> fullShape;
  for (unsigned i = 2; i < operands.size(); ++i) {
    auto dimVal = extractConstantInt(operands[i]);
    if (!dimVal)
      return TP_GENERIC_FAILURE;
    fullShape.push_back(*dimVal);
  }

  // Per-buffer rank (full shape minus the leading num-buffers dimension)
  unsigned perBufferRank = fullShape.size() - 1;
  auto *context = self.getBuilder().getContext();

  // Create default CGA layout (single CTA)
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, perBufferRank);

  // Construct the shared memory encoding based on per-buffer rank
  mlir::Attribute encoding;
  if (perBufferRank == 1) {
    // 1D: SwizzledSharedEncoding with default params
    llvm::SmallVector<unsigned> order;
    for (int i = perBufferRank - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    encoding = ttg::SwizzledSharedEncodingAttr::get(
        context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  } else {
    // 2D+: NVMMASharedEncoding using shape-based builder
    // Extract per-buffer shape (skip the leading num dimension)
    llvm::SmallVector<int64_t> perBufferShape(fullShape.begin() + 1,
                                              fullShape.end());
    llvm::SmallVector<unsigned> order;
    for (int i = perBufferRank - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    encoding = ttg::NVMMASharedEncodingAttr::get(
        context, perBufferShape, order, cgaLayout, elemType,
        /*fp4Padded=*/false);
  }

  // Create the MemDescType
  auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto memDescType = ttg::MemDescType::get(fullShape, elemType, encoding,
                                           memorySpace,
                                           /*mutableMemory=*/true);

  operands[0] = self.create<ttg::LocalAllocOp>(memDescType);
  return TP_SUCCESS;
}

// ===========================================================================
// tlx_local_alloc_tmem — TMEM allocation (Blackwell)
//
// Operand convention:
//   operands[0] = type-carrier / result slot
//                 Input type must be MemDescType with TensorMemorySpaceAttr.
//                 Overwritten with the TMEMAllocOp result.
// ===========================================================================

static TritonPluginResult
createLocalAllocTmem(const char *handle, TritonOpBuilder &self,
                     std::vector<mlir::Value> &operands) {
  if (operands.empty())
    return TP_GENERIC_FAILURE;

  auto memDescType =
      mlir::dyn_cast<ttg::MemDescType>(operands[0].getType());
  if (!memDescType)
    return TP_GENERIC_FAILURE;

  if (!mlir::isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace()))
    return TP_GENERIC_FAILURE;

  operands[0] = self.create<ttng::TMEMAllocOp>(memDescType, /*src=*/nullptr);
  return TP_SUCCESS;
}

// ===========================================================================
// tlx_local_view — Index into a multi-buffer allocation
//
// Maps to ttg::MemDescIndexOp. Given a multi-buffer MemDesc (e.g.,
// <2x64x64xf16, ...>) and an index, returns a subview with the leading
// dimension dropped (e.g., <64x64xf16, ...>). For 1D inputs the subview
// shape is <1>.
//
// Operand convention:
//   operands[0] = result slot (overwritten with MemDescIndexOp result)
//   operands[1] = source MemDesc (multi-buffer allocation)
//   operands[2] = buffer index (i32 scalar)
// ===========================================================================

static TritonPluginResult
createLocalView(const char *handle, TritonOpBuilder &self,
                std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return TP_GENERIC_FAILURE;

  mlir::Value localAlloc = operands[1];
  mlir::Value bufferIdx = operands[2];

  auto localAllocType =
      mlir::dyn_cast<ttg::MemDescType>(localAlloc.getType());
  if (!localAllocType)
    return TP_GENERIC_FAILURE;

  auto localAllocShape = localAllocType.getShape();
  mlir::Type memDescType;
  if (localAllocShape.size() == 1) {
    // 1D: subview is shape [1]
    memDescType = ttg::MemDescType::get(
        {1}, localAllocType.getElementType(), localAllocType.getEncoding(),
        localAllocType.getMemorySpace(),
        /*mutableMemory=*/localAllocType.getMutableMemory());
  } else {
    // N-D: drop the leading dimension
    memDescType = ttg::MemDescType::get(
        localAllocShape.drop_front(), localAllocType.getElementType(),
        localAllocType.getEncoding(), localAllocType.getMemorySpace(),
        /*mutableMemory=*/localAllocType.getMutableMemory());
  }

  operands[0] =
      self.create<ttg::MemDescIndexOp>(memDescType, localAlloc, bufferIdx);
  return TP_SUCCESS;
}

// ===========================================================================
// tlx_local_store — Store register tensor into SMEM buffer
//
// Maps to ttg::LocalStoreOp.
//
// Operand convention (after pybind prepends result slot at [0]):
//   operands[0] = result slot (unused for stores)
//   operands[1] = destination MemDesc (SMEM subview)
//   operands[2] = source register tensor (RankedTensorType)
//
// No result is produced (store is a side-effecting op).
// ===========================================================================

static TritonPluginResult
createLocalStore(const char *handle, TritonOpBuilder &self,
                 std::vector<mlir::Value> &operands) {
  if (operands.size() < 3)
    return TP_GENERIC_FAILURE;

  mlir::Value dst = operands[1];
  mlir::Value src = operands[2];

  if (!mlir::isa<ttg::MemDescType>(dst.getType()))
    return TP_GENERIC_FAILURE;

  self.create<ttg::LocalStoreOp>(src, dst);
  return TP_SUCCESS;
}

// ===========================================================================
// tlx_local_load — Load from SMEM buffer into register tensor
//
// Maps to ttg::LocalLoadOp. The result type is a RankedTensorType derived
// from the source MemDesc's shape and element type (no encoding, since the
// register layout is determined later by the compiler).
//
// Operand convention:
//   operands[0] = result slot (overwritten with LocalLoadOp result)
//   operands[1] = source MemDesc (SMEM subview)
//   operands[2] = (optional) async token
// ===========================================================================

static TritonPluginResult
createLocalLoad(const char *handle, TritonOpBuilder &self,
                std::vector<mlir::Value> &operands) {
  if (operands.size() < 2)
    return TP_GENERIC_FAILURE;

  mlir::Value subView = operands[1];

  auto subViewType = mlir::dyn_cast<ttg::MemDescType>(subView.getType());
  if (!subViewType)
    return TP_GENERIC_FAILURE;

  auto newType = mlir::RankedTensorType::get(subViewType.getShape(),
                                             subViewType.getElementType());

  // Optional async token in operands[2]
  mlir::Value asyncToken;
  if (operands.size() > 2)
    asyncToken = operands[2];

  operands[0] = self.create<ttg::LocalLoadOp>(newType, subView, asyncToken);
  return TP_SUCCESS;
}

// ===========================================================================
// Plugin registration
// ===========================================================================

using CustomOpFn = TritonPluginResult (*)(const char *, TritonOpBuilder &,
                                          std::vector<mlir::Value> &);

static const char *LOCAL_ALLOC_SMEM = "tlx_local_alloc";
static const char *LOCAL_ALLOC_TMEM = "tlx_local_alloc_tmem";
static const char *LOCAL_VIEW = "tlx_local_view";
static const char *LOCAL_STORE = "tlx_local_store";
static const char *LOCAL_LOAD = "tlx_local_load";

static std::unordered_map<std::string, CustomOpFn> customOpMap = {
    {LOCAL_ALLOC_SMEM, createLocalAllocSmem},
    {LOCAL_ALLOC_TMEM, createLocalAllocTmem},
    {LOCAL_VIEW, createLocalView},
    {LOCAL_STORE, createLocalStore},
    {LOCAL_LOAD, createLocalLoad},
};

static std::vector<const char *> customOpNames = {
    LOCAL_ALLOC_SMEM, LOCAL_ALLOC_TMEM, LOCAL_VIEW, LOCAL_STORE, LOCAL_LOAD,
};

TRITON_PLUGIN_API
tritonEnumeratePluginCustomOps(uint32_t *count, const char **handles) {
  if (!count)
    return TP_GENERIC_FAILURE;
  *count = static_cast<uint32_t>(customOpNames.size());
  if (!handles)
    return TP_SUCCESS;
  for (unsigned i = 0; i < customOpNames.size(); ++i)
    handles[i] = customOpNames[i];
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonAddPluginCustomOp(const char *handle, TritonOpBuilder &self,
                        std::vector<mlir::Value> &operands) {
  std::string key(handle);
  auto it = customOpMap.find(key);
  if (it == customOpMap.end())
    return TP_GENERIC_FAILURE;
  return it->second(handle, self, operands);
}
