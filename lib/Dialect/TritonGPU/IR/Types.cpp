#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::gpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"

static constexpr llvm::StringRef kMutableMemory = "mutable";

Type MemDescType::parse(AsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> dimensions; // required
  if (failed(parser.parseDimensionList(dimensions, /*allowDynamic=*/false)))
    return Type();

  Type elementType; // required
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute encoding; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(encoding)))
    return Type();

  Attribute memorySpace; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(memorySpace)))
    return Type();

  bool mutableMemory = false;      // optional
  SmallVector<int64_t> allocShape; // optional
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword(kMutableMemory))) {
      mutableMemory = true;
      if (succeeded(parser.parseOptionalComma())) {
        if (failed(parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                             /*withTrailingX=*/false))) {
          return Type();
        }
      }
    } else if (failed(parser.parseDimensionList(allocShape,
                                                /*allowDynamic=*/false,
                                                /*withTrailingX=*/false))) {
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();

  if (!allocShape.empty())
    return MemDescType::getChecked(loc, parser.getContext(), dimensions,
                                   elementType, encoding, memorySpace,
                                   mutableMemory, allocShape);

  return MemDescType::getChecked(loc, parser.getContext(), dimensions,
                                 elementType, encoding, memorySpace,
                                 mutableMemory, dimensions);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  auto shape = getShape();
  for (auto dim : shape)
    printer << dim << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  if (getMemorySpace())
    printer << ", " << getMemorySpace();
  if (getMutableMemory())
    printer << ", " << kMutableMemory;
  auto allocShape = getAllocShape();
  if (allocShape != shape) {
    printer << ", " << allocShape[0];
    for (auto dim : allocShape.drop_front(1)) {
      printer << "x" << dim;
    }
  }
  printer << ">";
}

LogicalResult MemDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  Attribute encoding, Attribute memorySpace,
                                  bool mutableMemory,
                                  ArrayRef<int64_t> allocShape) {
  if (shape.empty()) {
    return emitError() << "rank 0 memdesc is not allowed";
  }
  auto mma = dyn_cast_if_present<NVMMASharedEncodingAttr>(encoding);
  if (mma && mma.isPaddedFp6() && !elementType.isInteger(8))
    return emitError() << "FP6 shared tiles require i8 storage";
  unsigned bitwidth = getIntOrFloatOrPtrBitWidth(elementType);
  if (bitwidth != 1 && bitwidth < 8)
    return emitError() << "element type bit width must be 1 or at least 8; got "
                       << bitwidth;
  if (llvm::is_contained(shape, 0))
    return emitError() << "shape has 0 dimension";
  if (llvm::is_contained(allocShape, 0))
    return emitError() << "alloc shape has 0 dimension";
  if (allocShape.size() < shape.size())
    return emitError()
           << "alloc shape must have at least as many dimensions as shape";
  bool isDummyTMEM = isa<triton::tlx::DummyTMEMLayoutAttr>(encoding);
  auto layoutEncoding = dyn_cast_if_present<LayoutEncodingTrait>(encoding);
  if (!layoutEncoding && !isDummyTMEM)
    return emitError() << encoding << " is not a valid encoding";
  // A pow2 alloc view's allocation is verified once the checks below accept
  // it; the TMEM checks drop its pipelining dimension.
  std::optional<ArrayRef<int64_t>> npotPow2AllocShape;
  if (!isDummyTMEM) {
    auto rank = layoutEncoding.getRank();
    if (isa<nvidia_gpu::TensorMemoryEncodingAttr>(encoding)) {
      if (shape.size() != 2 && shape.size() != 3)
        return emitError() << "rank must be 2 or 3";
    } else if (isa<SharedEncodingTrait>(encoding)) {
      if (!(rank == shape.size() || rank == shape.size() - 1))
        return emitError() << "rank must be equal to or one less than "
                           << "the shape size. Got " << rank << " and "
                           << shape.size();
    } else if (isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(encoding)) {
      if (shape.size() != 2 && shape.size() != 3)
        return emitError() << "tensor-memory scale descriptors must have rank "
                              "2 or 3; got "
                           << shape.size();
    }
    // Every layout dimension must be a power of 2; only a leading pipeline
    // dimension and a shared buffer's first dimension may have another
    // positive size.
    if (isNpotPow2AllocView(encoding, shape))
      npotPow2AllocShape = allocShape;
    else if (!hasPow2MemDescLayoutShape(encoding, shape))
      return emitError()
             << "shape must have power-of-2 and non-zero dimensions; got "
             << shape;
    // A shared allocation with an NPOT tile is laid out on its pow2 alloc
    // shape, whose footprint verifyNpotPow2AllocView checks it holds.
    if (!(npotPow2AllocShape && isa<SharedEncodingTrait>(encoding)) &&
        !hasPow2MemDescLayoutShape(encoding, allocShape))
      return emitError()
             << "alloc shape must have power-of-2 and non-zero dimensions; got "
             << allocShape;
  }
  if (llvm::any_of(
          llvm::zip(shape, allocShape.take_back(shape.size())),
          [](auto pair) { return std::get<0>(pair) > std::get<1>(pair); }))
    return emitError() << "shape must be less than or equal to allocShape. "
                       << "shape = " << shape
                       << ", allocShape = " << allocShape;
  auto ctx = encoding.getContext();
  if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(encoding)) {
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    unsigned bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth < 8)
      return emitError()
             << "tensor-memory element type bit width must be at least 8; got "
             << bitwidth;
    if (bitwidth * enc.getColStride() > 32) {
      return emitError()
             << "bitwidth * colStride must be less than or equal to 32. Got "
             << bitwidth << " and " << enc.getColStride();
    }
    auto ctaSplit = enc.getCGALayout().getCTASplitNum();
    auto blockN = std::min<int32_t>(enc.getBlockN(), shape.back());
    // A row slice (tmem_subslice with dim = 0) may hold fewer rows than a
    // block; its allocation may not.
    if (allocShape[allocShape.size() - 2] < enc.getBlockM() * ctaSplit[0] ||
        shape[shape.size() - 1] < blockN * ctaSplit[1]) {
      return emitError() << "the tensor shape must be at least "
                         << enc.getBlockM() * ctaSplit[0] << "x"
                         << blockN * ctaSplit[1] << ". Got " << shape;
    }
  } else if (auto enc = dyn_cast<SharedEncodingTrait>(encoding)) {
    if (memorySpace != SharedMemorySpaceAttr::get(ctx) &&
        memorySpace != nvidia_gpu::SharedClusterMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be SharedMemorySpace or "
                            "SharedClusterMemorySpace for shared encoding. "
                         << "Got " << memorySpace;
    }
    if (bitwidth % 8 != 0)
      return emitError()
             << "shared-memory element type bit width must be a multiple of 8; "
                "got "
             << bitwidth;
    // Its layout is cut to the cover shape, which keeps whole digits only at
    // power-of-two extents.
    if (isa<SharedLinearEncodingAttr>(encoding) &&
        !llvm::all_of(getNpotPow2AllocShape(dropPipeliningDim(allocShape, enc)),
                      llvm::isPowerOf2_64))
      return emitError() << "shared linear encoding needs power-of-two layout "
                            "dimensions; got "
                         << allocShape;
  } else if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                 encoding)) {
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    if (allocShape.size() != 2 && allocShape.size() != 3) {
      return emitError() << "Scales must be 2D or 3D (with multibuffering)";
    }
    auto bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth != 8) {
      return emitError() << "bitwidth must be 8";
    }
    // TMEM stores require 16 independently addressable rows per CTA. A 3D
    // allocation is multibuffered, so its leading dimension counts buffers.
    auto layoutAllocShape = dropPipeliningDim(allocShape, enc);
    auto shapePerCTA = getShapePerCTA(enc, layoutAllocShape);
    if (shapePerCTA[0] < 16)
      return emitError() << "tensor-memory scale allocations require at least "
                            "16 rows per CTA; got "
                         << shapePerCTA[0];
    if (enc.getCtaMode() == nvidia_gpu::TensorMemoryCTAMode::TwoCTA_RHS) {
      auto rhsShapePerCTA = getShapePerCTA(enc, layoutAllocShape);
      if (rhsShapePerCTA != ArrayRef<int64_t>({128, 4})) {
        return emitError()
               << "twocta_rhs tensor-memory scales require a per-CTA "
                  "allocation shape of 128x4; got "
               << rhsShapePerCTA;
      }
    }
  } else if (isa<triton::tlx::DummyTMEMLayoutAttr>(encoding)) {
    // Dummy TMEM layout for deferred resolution - allow any shape for TMEM
    // The layout will be resolved to a concrete encoding during layout
    // propagation (e.g., TensorMemoryScalesEncodingAttr for scales)
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
  }

  // These encodings are also SharedEncodingTraits but have additional rules.
  if (auto enc = dyn_cast<PartitionedSharedEncodingAttr>(encoding)) {
    auto inner = enc.getPartitionLayout();
    if (isa<PartitionedSharedEncodingAttr>(inner)) {
      return emitError()
             << "nested PartitionedSharedEncodingAttr is not supported";
    }
    if (isa<NVMMASharedEncodingAttr>(inner)) {
      return emitError() << "NVMMASharedEncodingAttr is not supported as a "
                            "partitionLayout";
    }

    auto blockShape =
        getShapePerCTA(enc, dropPipeliningDim(allocShape, encoding));
    unsigned partitionDim = enc.getPartitionDim();
    unsigned numLogicalPieces = enc.getNumLogicalPieces();
    if (blockShape[partitionDim] % numLogicalPieces != 0) {
      return emitError()
             << "per-CTA allocation extent along partitionDim must be "
                "divisible by numPartitions * numGroups; got "
             << blockShape[partitionDim] << " and " << numLogicalPieces;
    }

    SmallVector<int64_t> pieceShape(blockShape);
    pieceShape[partitionDim] /= numLogicalPieces;
    auto verifyFixedShape = [&](const LinearLayout &layout) -> LogicalResult {
      auto shape = llvm::to_vector_of<int64_t>(layout.getOutDimSizes());
      auto actual = getShapePerCTA(enc, shape);
      if (actual != pieceShape) {
        return emitError() << "partitionLayout does not match the per-CTA "
                              "logical piece shape; expected "
                           << pieceShape << ", got " << actual;
      }
      return success();
    };

    if (auto padded = dyn_cast<PaddedSharedEncodingAttr>(inner)) {
      if (failed(verifyFixedShape(padded.getLinearComponent())))
        return failure();
    } else if (auto linear = dyn_cast<SharedLinearEncodingAttr>(inner)) {
      if (failed(verifyFixedShape(linear.getLinearLayout())))
        return failure();
    }
  } else if (auto enc = dyn_cast<PaddedSharedEncodingAttr>(encoding)) {
    auto rank = enc.getRank();
    // Ensure linear component's outDims match the alloc size ignoring
    // pipelining dimension
    auto outDims = standardOutDimNames(ctx, rank);
    const auto &ll = enc.getLinearComponent();
    auto expectedShape = dropPipeliningDim(allocShape, enc);

    for (auto d = 0; d < rank; d++) {
      if (ll.getOutDimSize(outDims[d]) != expectedShape[d]) {
        return emitError() << "Mismatch in expected shape for dimension " << d
                           << ". Expected: " << expectedShape[d]
                           << ", got: " << ll.getOutDimSize(outDims[d]);
      }
    }
  } else if (auto enc = dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    SmallVector<int64_t> shapePerCTA(getShapePerCTA(enc, allocShape));
    SmallVector<int64_t> blockShape(
        dropPipeliningDim(ArrayRef(shapePerCTA), enc));
    // Under TRITON_ALLOW_NPOT an NPOT tile is laid out and sized as its pow2
    // alloc shape, so the swizzle must fit the pow2 alloc shape.
    if (triton::tools::isNpotAllowed())
      blockShape = getNpotPow2AllocShape(blockShape);
    if (failed(getTMABlockShape(blockShape, enc.getElementBitWidth(),
                                enc.getSwizzlingByteWidth(), enc.isPaddedFp4(),
                                enc.getTransposed(), /*packedSize=*/false,
                                emitError, TMAMode::Tiled)))
      return failure();
  } else if (auto enc = dyn_cast<SharedLinearEncodingAttr>(encoding)) {
    auto blockShape = dropPipeliningDim(allocShape, enc);
    const LinearLayout &ll = enc.getLinearLayout();
    for (auto [dim, size, llSize] :
         llvm::enumerate(blockShape, ll.getOutDimSizes())) {
      if (size == llSize)
        continue;
      return emitError() << "Mismatch in expected shape for dimension " << dim
                         << ". Expected: " << size << ", got: " << llSize;
    }
  }

  if (npotPow2AllocShape)
    return verifyNpotPow2AllocView(emitError, encoding, shape,
                                   *npotPow2AllocShape);
  return success();
}

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::gpu::TritonGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"
      >();
}
