#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir {
namespace triton {
namespace NVIDIA {

// The descriptor format is described in the spec:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Unnamed fields are not used.
// Bit 52 is the leading-dimension stride mode: 0 = byte-offset relative
// (default), 1 = byte-address absolute (sm_103a only, used by K=96 mxf4nvf4
// where the packed K row is 48 bytes and would overflow the 128B boundary).
// PTX ISA 9.2 Table 42 (§9.7.16.4.1); absolute mode introduced for sm_103a
// per §9.7.16.4.1.1.
union SMEMDescriptor {
  uint64_t descriptor;
  struct {
    uint64_t baseAddress : 14;
    uint64_t : 2;
    uint64_t leadDimensionBaseOffset : 14;
    uint64_t : 2;
    uint64_t strideDimensionBaseOffset : 14;
    uint64_t : 3;
    uint64_t matrixBaseOffset : 3;
    uint64_t lboAbsoluteMode : 1; // bit 52
    uint64_t : 9;                 // bits 53-61 (fixed 0 per Table 42)
    uint64_t swizzlingMode : 2;
  };
};
static_assert(sizeof(SMEMDescriptor) == 8,
              "SMEMDescriptor must be exactly 64 bits wide");

struct MMASMEMDescriptor {
  SMEMDescriptor descriptor;
  int32_t swizzlingByteWidth;
  int32_t bitwidth;
  bool transposed;
  bool fp4Padded;
  // True when the descriptor uses absolute-address leading-dimension mode
  // (bit 52 = 1), which is required for K=96 mxf4nvf4 on sm_103a. In this
  // mode the LBO field encodes the absolute SMEM address of the second
  // chunk; smemLoad has to OR in the runtime base address contribution.
  bool lboAbsoluteMode = false;
};

struct MemDescOperand {
  Value base;
  std::optional<int> offset;
};

// Abstract class to calculate the address of a shared or tensor memory slice.
class DotOpMmaMemLoader {
public:
  virtual ~DotOpMmaMemLoader() = default;
  // Given the starting coordinates of the logical tensor (i.e. reps *
  // ctaTileSize), return the associated memory descriptor for SMEM / TMEM.
  virtual MemDescOperand memLoad(int a, int b,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc) const = 0;
};

class DotOpMmaSmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaSmemLoader() = default;

  DotOpMmaSmemLoader(MMASMEMDescriptor desc, Value baseSrcb128,
                     LinearLayout llInv)
      : desc(desc), baseSrcb128(baseSrcb128), ll(std::move(llInv)) {}

  static FailureOr<DotOpMmaSmemLoader>
  build(Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
        Value smemBase, ArrayRef<unsigned> instrShape, unsigned MNdim,
        int mmaVersion, bool isFp4 = false,
        std::optional<RankedTensorType> mmaTy = std::nullopt) {
    auto ctx = rewriter.getContext();
    auto kOffset = str_attr("offset");
    auto bitwidth = memTy.getElementType().getIntOrFloatBitWidth();

    // Unified path for NVMMASharedEncoding: use computeSMEMDescriptor (the
    // ISA-formula helper) for both pow2 and NPOT shapes. This replaces the
    // brute-force getDescriptor search and the cross-validation assert that
    // checked they agreed.
    if (auto encoding = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
            memTy.getEncoding())) {
      auto allocShape = memTy.getAllocShape().take_back(memTy.getRank());
      bool hasNpotDim = llvm::any_of(allocShape, [](int64_t s) {
        return s > 0 && !llvm::isPowerOf2_64(s);
      });

      if (isFp4 && encoding.getFp4Padded() && hasNpotDim) {
        // fp4Padded requires 128B swizzle and has a special padded packing
        // format. NPOT dims are unlikely with fp4Padded (which requires
        // contigBytes divisible by 128), but bail out if it happens.
        return failure();
      }

      // Pseudoinverse for smemLoad coordinate routing: NPOT contiguous dim uses
      // the 3-dim split layout (single algebra per dim); else the 2-dim layout.

      LinearLayout llInv;
      if (hasNpotDim) {
        auto splitOpt =
            gpu::nvmmaSharedToSplitLinearLayout(allocShape, encoding);
        if (splitOpt) {
          // NPOT contiguous dim: use 3-dim split pseudoinverse.
          llInv = splitOpt->pseudoinvert();
        } else {
          // NPOT non-contiguous dim (e.g., NPOT M with pow2 K):
          // standard 2-dim pseudoinverse is correct because the NPOT dim
          // is uncoupled from swizzle.
          auto fwdFull = toLinearLayout(memTy);
          auto fwdWithBlock = gpu::getLayoutWithinBlock(fwdFull);
          auto outDimNames = llvm::to_vector(fwdWithBlock.getOutDimNames());
          auto fwdLocal =
              fwdWithBlock.sublayout({str_attr("offset")}, outDimNames);
          llInv = fwdLocal.pseudoinvert();
        }
      } else {
        // Pow2: standard pseudoinverse.
        llInv = toLinearLayout(memTy).pseudoinvert();
      }

      if (isFp4) {
        // fp4 (E2M1): prepend identity1D(2) for packed-byte offset; halve
        // bitwidth.
        auto dims = to_vector(llInv.getInDimNames());
        auto trans = llInv.getBasis(dims[0], 0, kOffset) == 1;
        llInv =
            LinearLayout::identity1D(2, dims[trans ? 0 : 1], kOffset) * llInv;
        bitwidth /= 2;
      }

      return buildFromLL(loc, rewriter, llInv, bitwidth, smemBase, instrShape,
                         MNdim, mmaVersion, memTy, mmaTy);
    }

    // Non-NVMMAShared: fall back to the brute-force getDescriptor build().
    auto llInv = toLinearLayout(memTy).pseudoinvert();
    if (isFp4) {
      auto dims = to_vector(llInv.getInDimNames());
      auto trans = llInv.getBasis(dims[0], 0, kOffset) == 1;
      llInv = LinearLayout::identity1D(2, dims[trans ? 0 : 1], kOffset) * llInv;
      bitwidth /= 2;
    }

    return build(loc, rewriter, llInv, bitwidth, smemBase, instrShape, MNdim,
                 mmaVersion, mmaTy);
  }

  static FailureOr<DotOpMmaSmemLoader>
  build(Location loc, RewriterBase &rewriter, const LinearLayout &ll,
        int bitwidth, Value smemBase, ArrayRef<unsigned> instrShape,
        unsigned MNdim, int mmaVersion,
        std::optional<RankedTensorType> mmaTy = std::nullopt) {
    // ll is a map from two dimensions (dim0, dim1) or (row, col) into offsets
    // and blocks
    auto ctx = rewriter.getContext();
    auto kOffset = str_attr("offset");
    auto kBlock = str_attr("block");
    assert(ll.getNumOutDims() == 2);
    assert(ll.hasOutDim(kOffset) && ll.hasOutDim(kBlock));

    assert(mmaVersion == 3 || mmaVersion == 5);
    // Just needed for MMAv3
    assert(mmaTy.has_value() == (mmaVersion == 3));
    assert(MNdim < 2);
    assert(instrShape.size() == 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Due to having a 16B alignment, we can compute the offsets in 128b
    // elements
    // TODO We should assert in the verifier that the alignment is at least 16B
    smemBase = b.ptrtoint(i32_ty, smemBase);
    Value baseSrcb128 = b.lshr(smemBase, b.i32_val(4));

    if (mmaVersion == 3) {
      auto mmaLl = gpu::toLinearLayout(mmaTy.value());
      auto outDims = to_vector(mmaLl.getOutDimNames());
      auto kWarp = str_attr("warp");
      // Map from warps into the MN dimension
      auto mmaWarps = mmaLl.sublayout({kWarp}, {outDims[MNdim]}) *
                      LinearLayout::identity1D(1, kWarp, outDims[1 - MNdim]);
      // Map from warps to offsets in bitwidth elements
      auto warpToOffset = mmaWarps.compose(ll);
      // Map from warps to offsets in 128b elements
      auto maybeWarpToOffsetb128 =
          divideLeft(warpToOffset,
                     LinearLayout::zeros1D(1, kWarp, kOffset, 128 / bitwidth));
      assert(maybeWarpToOffsetb128.has_value());
      // zero out the first two warp bases to have a warpgroup to offset map
      auto bases = maybeWarpToOffsetb128->getBases();
      assert(maybeWarpToOffsetb128->getNumOutDims() == 2);
      bases[kWarp][0] = {0, 0};
      bases[kWarp][1] = {0, 0};
      auto warpGroupToOffsetb128 =
          LinearLayout(std::move(bases), warpToOffset.getOutDims(),
                       /*requireSurjective=*/false);
      Value warpId = mlir::triton::gpu::WarpIdOp::create(rewriter, loc);
      Value warpStrideb128 =
          applyLinearLayout(loc, rewriter, warpGroupToOffsetb128,
                            {{kWarp, warpId}})[0]
              .second;
      baseSrcb128 = b.add(baseSrcb128, warpStrideb128);
    }

    for (auto [dim, instrSize] : llvm::zip(ll.getInDimNames(), instrShape)) {
      if (instrSize <= ll.getInDimSize(dim))
        continue;
      auto inDims = ll.getInDims();
      return mlir::emitError(loc)
             << "instruction shape [" << instrShape[0] << ", " << instrShape[1]
             << "] is too large for the layout with block size ["
             << inDims[0].second << ", " << inDims[1].second << "]";
    }

    auto desc = getDescriptor(loc, ll, instrShape, bitwidth, MNdim, mmaVersion);
    if (failed(desc))
      return failure();

    return DotOpMmaSmemLoader{*desc, baseSrcb128, ll};
  }

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc) const {
    auto *ctx = loc.getContext();
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto dims = to_vector(ll.getInDimNames());
    SmallVector<std::pair<StringAttr, int32_t>> applyArgs;
    if (dims.size() == 3) {
      // Split-dim pseudoinverse: dims = {dim0, contig_intra, contig_phase}
      // The split layout is in core-tile coordinates (not tensor coords):
      //   dim0 = rows (non-contiguous direction)
      //   contig_intra + contig_phase = cols (contiguous direction, NPOT)
      //
      // The caller passes (a, b) in tensor coordinates:
      //   For non-transposed: a = row (non-contig), b = col (contig)
      //   For transposed: a = first tensor dim (contig), b = second
      //   (non-contig)
      //
      // For non-transposed: a -> dim0, b -> contig_intra/contig_phase (correct)
      // For transposed: a -> contig_intra/contig_phase, b -> dim0 (swap needed)
      int32_t rowCoord, colCoord;
      if (desc.transposed) {
        rowCoord = b; // b = non-contiguous dim -> dim0 (rows)
        colCoord = a; // a = contiguous dim -> dim1 (cols, split)
      } else {
        rowCoord = a; // a = non-contiguous dim -> dim0 (rows)
        colCoord = b; // b = contiguous dim -> dim1 (cols, split)
      }
      int32_t phaseSize = ll.getInDimSize(dims[1]); // contig_intra is pow2
      int32_t colIntra = colCoord % phaseSize;
      int32_t colPhase = colCoord / phaseSize;
      applyArgs = {
          {dims[0], rowCoord}, {dims[1], colIntra}, {dims[2], colPhase}};
    } else {
      applyArgs = {{dims[0], a}, {dims[1], b}};
    }
    auto offsetBlock = ll.apply(applyArgs);
    int32_t offsetElems = offsetBlock[0].second;
    // NPOT (buildFromLL) pseudoinverse may lack the "block" out dim; the
    // standard path has both "offset" and "block".
    if (ll.getNumOutDims() == 2) {
      int32_t block = offsetBlock[1].second;
      assert(block == 0);
    }
    // For sub-byte types (e.g., fp4 with bitwidth=4), odd offsetElems would
    // silently truncate (4/8 = 0). The fp4 path in build() prepends
    // identity1D(2) to guarantee even offsets; assert that invariant here.
    assert(desc.bitwidth >= 8 || offsetElems % 2 == 0);
    int32_t smemByteOffsetb8 = offsetElems * desc.bitwidth / 8;
    auto currDesc = desc.descriptor;
    if (desc.swizzlingByteWidth > 0) {
      uint32_t mask = (desc.swizzlingByteWidth >> 4) - 1;
      currDesc.matrixBaseOffset = (smemByteOffsetb8 / 128) & mask;
    } else {
      // swizzle=0: the offset is carried entirely by the base address, so the
      // matrixBaseOffset phase field is 0.
      currDesc.matrixBaseOffset = 0;
    }
    // PTX ISA 9.2 §9.7.16.3.1.3: K=96 absolute-LBO mode requires matrix base
    // offset == 0 (restriction 3). SMEM allocation must place the K=96 region
    // on the 128B swizzle pattern boundary; if not, the descriptor would be
    // silently wrong on B300 silicon.
    // Hard error (not assert): compiled out under NDEBUG (mode/opt), so a
    // misaligned K=96 SMEM region would otherwise silently miscompute on B300.
    if (desc.lboAbsoluteMode && currDesc.matrixBaseOffset != 0)
      llvm::report_fatal_error(
          "K=96 fp4 SMEM region must start on the 128B swizzle pattern "
          "boundary "
          "(matrix base offset must be 0; PTX ISA 9.2 §9.7.16.3.1.3 "
          "restriction 3)");
    currDesc.baseAddress = 0;
    int32_t smemByteOffsetb128 = smemByteOffsetb8 >> 4;
    // Compute the base address at runtime to prevent LLVM from folding the
    // per-tile offset into a unique 64-bit constant. This produces a short
    // dependency chain (add→and→zext→add) that helps hide WGMMA latency.
    Value fullAddrb128 = tb.add(baseSrcb128, tb.i32_val(smemByteOffsetb128));
    Value addrMasked = tb.and_(fullAddrb128, tb.i32_val(0x3FFF));
    Value addr64 = tb.zext(i64_ty, addrMasked);
    Value descVal = tb.add(tb.int_val(64, currDesc.descriptor), addr64);
    if (desc.lboAbsoluteMode) {
      // K=96 mxf4nvf4: the LBO field (bits 16-29) holds the absolute SMEM
      // address of the "second chunk" of the 48-byte packed K=96 run, which the
      // hardware reads as two chunks to stay within the aligned 128B boundary
      // (PTX ISA 9.2 §9.7.16.3.1.2). The chunk1 address is NOT a fixed
      // base + 64B: per CUTLASS 4.4.2
      // (cute/atom/mma_traits_sm100.hpp:4609-4621, `desc_a.leading_byte_offset_
      // = desc_next_a.start_address_`) the absolute LBO is the *layout-derived*
      // start address of the K-block run where the operand's K data continues,
      // computed by the SAME smem layout that placed chunk0 -- i.e. the
      // start_address_ of the "next" K-block descriptor, in 16B units, exactly
      // like chunk0's own start_address_ (Table 42 / CUTLASS SmemDescriptor:
      // start_address_ = smem_addr >> 4).
      //
      // Triton emits a SINGLE MMA per K=96 tile (mmaSizeK=96, numRepK=1) over a
      // freshly padded 128B SW128 atom that holds K=96 (= 3 contiguous 16B
      // blocks, bytes [0,48)) in the front of the atom. That run starts on the
      // 128B pattern boundary (matrixBaseOffset==0, enforced above) and ends at
      // byte 48, so it does NOT straddle the 128B boundary. This is exactly
      // CUTLASS's non-straddling case (mainloop MMA0:
      // sm103_blockscaled_mma_warpspecialized.hpp:1155-1156), where the "next"
      // K-block descriptor is block 0 of the SAME buffer -> its start_address_
      // equals chunk0's own start address. Hence chunk1's absolute address is
      // chunk0's base address (`addrMasked`), NOT `addrMasked + 64B`.
      //
      // The previous code synthesized `addrMasked + 4` (b128 units = +64
      // bytes), which modeled a straddling second 64B half that does not exist
      // for the packed 48B run: with only 48 real bytes, base+64 reads entirely
      // into the 128B atom's padding, producing byte-identical garbage on GB300
      // (T264996227). Deriving the LBO from the layout (chunk0 base) instead of
      // a constant offset is what every working NVMMAShared descriptor path
      // does (the start/LBO addresses are one function of the encoding + smem
      // base).
      Value secondChunkAddrb128 = addrMasked;
      Value lboField = tb.and_(secondChunkAddrb128, tb.i32_val(0x3FFF));
      Value lboShifted = tb.shl(tb.zext(i64_ty, lboField), tb.int_val(64, 16));
      descVal = tb.or_(descVal, lboShifted);
    }
    return descVal;
  }
  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override {
    return {smemLoad(a, b, rewriter, loc), std::nullopt};
  }

  MMASMEMDescriptor &getDescriptor() { return desc; }

private:
  MMASMEMDescriptor desc;
  Value baseSrcb128;
  LinearLayout ll;

  // Construct the descriptor and loader from encoding parameters and a
  // pseudoinverse layout. Uses computeSMEMDescriptor (ISA-formula) for
  // the descriptor and handles warp stride computation for MMAv3.
  //
  // Handles all layout shapes:
  //   - 2-dim pseudoinverse (pow2, or NPOT in non-contiguous dim only)
  //   - 3-dim pseudoinverse (NPOT in contiguous dim, split into
  //     contig_intra + contig_phase)
  static FailureOr<DotOpMmaSmemLoader>
  buildFromLL(Location loc, RewriterBase &rewriter, const LinearLayout &llInv,
              int bitwidth, Value smemBase, ArrayRef<unsigned> instrShapeArray,
              unsigned MNdim, int mmaVersion, gpu::MemDescType memTy,
              std::optional<RankedTensorType> mmaTy) {
    auto ctx = rewriter.getContext();
    auto kOffset = str_attr("offset");
    auto instrShape = to_vector(instrShapeArray);
    assert(instrShape.size() == 2);
    assert(MNdim < 2);

    // Extract encoding parameters directly.
    auto encoding =
        cast<triton::gpu::NVMMASharedEncodingAttr>(memTy.getEncoding());

    // Compute LBO and SBO using ISA canonical formulas (PTX ISA 9.2,
    // section 9.7.16.3) directly from encoding parameters. This avoids
    // building core tile LLs, pseudoinverting them, and reading basis
    // values -- which is fragile for NPOT dimensions.
    auto descResult = computeSMEMDescriptor(encoding, instrShape, bitwidth,
                                            MNdim, mmaVersion, memTy,
                                            /*isFp4=*/bitwidth == 4);
    if (failed(descResult)) {
      return failure();
    }
    auto mmaSMEMDesc = *descResult;

    // Base address computation (same as the standard build path).
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    smemBase = b.ptrtoint(i32_ty, smemBase);
    Value baseSrcb128 = b.lshr(smemBase, b.i32_val(4));

    if (mmaVersion == 3 && mmaTy.has_value()) {
      // For MMAv3, compute warp stride arithmetically.
      auto mmaLl = gpu::toLinearLayout(mmaTy.value());
      auto outDims = to_vector(mmaLl.getOutDimNames());
      auto kWarp = str_attr("warp");

      auto mmaWarps = mmaLl.sublayout({kWarp}, {outDims[MNdim]}) *
                      LinearLayout::identity1D(1, kWarp, outDims[1 - MNdim]);

      // For 3-dim split pseudoinverse (dim0, contig_intra, contig_phase),
      // reshape mmaWarps output to match llInv input dims.  Warps only
      // advance in the MN dim, not K, so the split dims (which decompose
      // the contiguous K dim) get zero contribution.
      auto llInvInDims = to_vector(llInv.getInDimNames());
      if (llInvInDims.size() == 3) {
        SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
        for (auto dim : llInvInDims) {
          newOutDims.push_back({dim, llInv.getInDimSize(dim)});
        }
        auto newProduct =
            std::accumulate(newOutDims.begin(), newOutDims.end(), int64_t{1},
                            [](int64_t a, auto &p) { return a * p.second; });
        if (mmaWarps.getTotalOutDimSizeProduct() == newProduct) {
          // Products match (typical SM100 path): reshape directly.
          mmaWarps = mmaWarps.reshapeOuts(newOutDims);
        } else {
          // NPOT-SM90: mmaWarps has a pow2 K-dim but llInv's K is split (NPOT).
          // Warps don't advance in K, so rebuild mmaWarps at llInv's dim sizes,
          // zero-padding each basis for the K-split dims.
          LinearLayout::BasesT adjustedBases;
          for (auto &[inDim, dimBases] : mmaWarps.getBases()) {
            std::vector<std::vector<int32_t>> paddedBases;
            for (auto &basis : dimBases) {
              std::vector<int32_t> padded(basis.begin(), basis.end());
              while (padded.size() < newOutDims.size()) {
                padded.push_back(0);
              }
              paddedBases.push_back(std::move(padded));
            }
            adjustedBases[inDim] = std::move(paddedBases);
          }
          mmaWarps = LinearLayout(std::move(adjustedBases), newOutDims,
                                  /*requireSurjective=*/false);
        }
      }

      auto warpToOffset = mmaWarps.compose(llInv);
      auto bases = warpToOffset.getBases();
      int numOutDims = warpToOffset.getNumOutDims();
      // Zero out warp group bases (first 2 warp bases = intra-warp-group).
      SmallVector<int32_t> zeroBasis(numOutDims, 0);
      bases[kWarp][0] =
          std::vector<int32_t>(zeroBasis.begin(), zeroBasis.end());
      bases[kWarp][1] =
          std::vector<int32_t>(zeroBasis.begin(), zeroBasis.end());
      int elemsPer128b = 128 / bitwidth;
      int offsetIdx = 0;
      for (auto [dimName, dimSize] : warpToOffset.getOutDims()) {
        if (dimName == kOffset) {
          break;
        }
        offsetIdx++;
      }
      for (auto &[dimName, dimBases] : bases) {
        for (auto &basisVec : dimBases) {
          assert(basisVec[offsetIdx] % elemsPer128b == 0 &&
                 "warp offset must be 128b-aligned");
          basisVec[offsetIdx] /= elemsPer128b;
        }
      }
      auto outDimsWarp = warpToOffset.getOutDims();
      for (auto &[dimName, dimSize] : outDimsWarp) {
        if (dimName == kOffset) {
          dimSize = std::max(int32_t{1}, dimSize / elemsPer128b);
        }
      }
      auto warpGroupToOffsetb128 = LinearLayout(std::move(bases), outDimsWarp,
                                                /*requireSurjective=*/false);
      Value warpId = mlir::triton::gpu::WarpIdOp::create(rewriter, loc);
      Value warpStrideb128 =
          applyLinearLayout(loc, rewriter, warpGroupToOffsetb128,
                            {{kWarp, warpId}})[0]
              .second;
      baseSrcb128 = b.add(baseSrcb128, warpStrideb128);
    }

    // baseSrcb128 must remain i32: smemLoad() adds an i32 per-tile offset to
    // it (line ~341) and only zexts the masked result to i64. Returning an i64
    // here triggers "'llvm.add' op requires the same type for all operands
    // and results" in ConvertTritonGPUToLLVM. The standard build() path stores
    // the i32 value directly; mirror that here.
    return DotOpMmaSmemLoader{mmaSMEMDesc, baseSrcb128, llInv};
  }

  static FailureOr<MMASMEMDescriptor>
  getDescriptor(Location loc, const LinearLayout &ll,
                ArrayRef<unsigned> instrShape, int bitwidth, unsigned MNdim,
                int mmaVersion) {
    // ll is a map from allocShape into offsets and blocks
    auto dims = to_vector(ll.getInDimNames());
    auto ctx = dims[0].getContext();
    auto kOffset = str_attr("offset");

    // Any CGALayout, it's not really used within getCoreMatrixLinearLayout
    auto CGALayout = triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, 2);

    for (bool fp4Padded : (bitwidth == 4 ? SmallVector<bool>({false, true})
                                         : SmallVector<bool>({false}))) {
      for (auto transposed : {false, true}) {
        for (int swizzling : {0, 32, 64, 128}) {
          // FIXME: getCoreMatrixLinearLayout does not accept bitwidth < 8
          auto shmemEnc = triton::gpu::NVMMASharedEncodingAttr::get(
              ctx, swizzling, transposed, std::max(8, bitwidth), fp4Padded,
              CGALayout);
          auto shmemTile =
              getCoreMatrixLinearLayout(shmemEnc, /*disableSwizzle=*/false);
          // Rename out dims to match the original layout (in case the dims were
          // (row, col))
          auto outDims = to_vector(shmemTile.getOutDims());
          outDims[0].first = dims[0];
          outDims[1].first = dims[1];
          shmemTile = LinearLayout(shmemTile.getBases(), outDims,
                                   /*requireSurjective=*/false);
          // unpack the fp4 layout
          if (bitwidth == 4) {
            shmemTile =
                LinearLayout::identity1D(2, kOffset, dims[1]) * shmemTile;
          }

          // getCoreMatrixLinearLayout gives the k-contiguous tile
          // shmemTile is a layout onto a matrix with shape
          // If swizzling != 0: 8 x (8 * swizzling / bitwidth)
          // If swizzling == 0: 8 x (8 * 16 / bitwidth)
          assert(shmemTile.getOutDimSize(dims[0]) == 8);
          // Multiply by 2 if fp4Padded as the matrix has half the core
          // matrix has half the number of elements
          assert(shmemTile.getOutDimSize(dims[1]) * (fp4Padded ? 2 : 1) ==
                 8 * std::max(16, swizzling) / bitwidth);

          if (transposed) {
            shmemTile = transposeLinearLayout(shmemTile, {1, 0});
          }
          // Pseudoinvert as fp4 may have padding
          auto shmemTileInv = shmemTile.pseudoinvert();

          // The PTX docs are wrong in subtle ways:
          // 1) LBO can be specified for kContig && swizzled != 0
          //    PTX says it's assumed to be 1, but  we can in fact use it
          // 2) The Cute layouts for kContig && swizzled != 0 are wrong
          int lbo = 0, sbo = 0;
          int leadingDim = transposed ? 0 : 1;
          int stridedDim = transposed ? 1 : 0;
          // The lbo / sbo is swapped for swizzling == 0 and MNContig lol
          bool MNContig = (MNdim == 0) == transposed;
          if (swizzling == 0 && MNContig) {
            std::swap(leadingDim, stridedDim);
          }
          auto log2RowsTile = shmemTileInv.getInDimSizeLog2(dims[leadingDim]);
          if (llvm::Log2_32(instrShape[leadingDim]) > log2RowsTile) {
            lbo = ll.getBasis(dims[leadingDim], log2RowsTile, kOffset);
          }

          auto log2ColsTile = shmemTileInv.getInDimSizeLog2(dims[stridedDim]);
          if (llvm::Log2_32(instrShape[stridedDim]) > log2ColsTile) {
            sbo = ll.getBasis(dims[stridedDim], log2ColsTile, kOffset);
          }

          // Pad the tile up to the full instruction shape with the relevant
          // stride if the instruction shape is larger than the tile
          auto bases = shmemTileInv.getBases();
          for (int d : {0, 1}) {
            // 'tile' with the atom tile according to the lbo/sbo rules
            for (int i = 1;
                 i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]);
                 i *= 2) {
              auto stride = ll.getBasis(
                  dims[d], shmemTileInv.getInDimSizeLog2(dims[d]), kOffset);
              bases[dims[d]].push_back({stride * i});
            }
          }
          auto maxBasis = 0;
          for (auto dimBases : llvm::make_second_range(bases)) {
            for (auto basis : dimBases) {
              maxBasis = std::max(maxBasis, basis[0]);
            }
          }
          // Multiply by 2 or round up to the next power of 2
          shmemTileInv = LinearLayout(std::move(bases),
                                      {{kOffset, llvm::NextPowerOf2(maxBasis)}},
                                      /*requireSurjective=*/false);
          // Add a trivial block dimension as getReps expects both layouts to
          // have the same outdims
          shmemTileInv *=
              LinearLayout::identity1D(1, dims[0], str_attr("block"));

          auto reps = getReps(ll, shmemTileInv);
          if (reps.has_value()) {
            SMEMDescriptor desc;
            desc.descriptor = mmaVersion == 5 ? 1ULL << 46 : 0ULL;
            // The lbo / sbo is defined wrt. the 128b elements
            desc.leadDimensionBaseOffset = (lbo * bitwidth / 8) >> 4;
            desc.strideDimensionBaseOffset = (sbo * bitwidth / 8) >> 4;
            switch (swizzling) {
            case 0:
              desc.swizzlingMode = 0;
              break;
            case 32:
              desc.swizzlingMode = 3;
              break;
            case 64:
              desc.swizzlingMode = 2;
              break;
            case 128:
              desc.swizzlingMode = 1;
              break;
            default:
              llvm_unreachable("Unsupported swizzling size.");
            }
            return MMASMEMDescriptor{/* .descriptor = */ desc,
                                     /* .swizzlingByteWidth = */ swizzling,
                                     /* .bitwidth = */ bitwidth,
                                     /* .transposed = */ transposed,
                                     /* .fp4Padded = */ fp4Padded,
                                     /* .lboAbsoluteMode = */ false};
          }
        }
      }
    }
    return failure();
  }

  // Compute SMEM descriptor LBO/SBO directly from encoding parameters using
  // ISA canonical formulas (PTX ISA 9.2, section 9.7.16.3).
  // This is the unified descriptor computation used by both pow2 and NPOT
  // paths. It replaces the brute-force getDescriptor() search and works
  // correctly for all dimension sizes.
  //
  // The core matrix tile is always 8 (rows) x tileCols (cols) before transpose.
  //   tileCols = 8 * max(16, swizzleBytes) / elemBitWidth
  //
  // Physical strides for one tile:
  //   Contiguous-axis (tileCols elements): tileCols * elemBytes bytes
  //   Non-contiguous-axis (8 rows): 8 * contigDimSize * elemBytes bytes
  //
  // LBO/SBO assignment:
  //   Normal: LBO = contiguous-axis stride, SBO = non-contiguous-axis stride
  //   MNContig + no-swizzle swap: LBO = non-contig stride, SBO = contig stride
  //
  // Strides derive independently from the encoding + allocShape.
  static FailureOr<MMASMEMDescriptor>
  computeSMEMDescriptor(triton::gpu::NVMMASharedEncodingAttr encoding,
                        ArrayRef<unsigned> instrShape, int bitwidth,
                        unsigned MNdim, int mmaVersion, gpu::MemDescType memTy,
                        bool isFp4) {
    int swizzling = encoding.getSwizzlingByteWidth();
    bool transposed = encoding.getTransposed();
    bool fp4Padded = encoding.getFp4Padded();

    // Determine the contiguous dimension size from the alloc shape.
    // transposed==false: last dim (dim1) is contiguous
    // transposed==true:  first dim (dim0) is contiguous
    auto allocShape = memTy.getAllocShape();
    assert(allocShape.size() >= 2 && "Expected at least 2D allocation shape");
    int contigDimIdx = transposed ? 0 : (allocShape.size() - 1);
    int64_t contigDimSize = allocShape[contigDimIdx];

    // Core matrix tile dimensions.
    // tileRows = 8 (always, for the non-contiguous/strided axis)
    // tileCols = number of elements along the contiguous axis per tile
    int tileRows = 8;
    int tileCols = 8 * std::max(16, swizzling) / bitwidth;

    // Physical stride between rows in the SMEM allocation.
    // For swizzle > 0, the core tile data is interleaved by the swizzle, so the
    // tile is a contiguous block of tileCols * tileRows elements. The stride
    // from one core tile to the next (in either direction) is based on the tile
    // size: SBO = coreTileBytes, LBO = numRowGroups * coreTileBytes.
    //
    // For swizzle == 0, no interleaving occurs: rows are stored with stride
    // = contigAllocStride (the pow2-rounded contiguous dim size, since
    // getAllocationShapePerCTA rounds NPOT dims to pow2 for swizzle=0). The
    // strides from one tile to the next are:
    //   - Along non-contiguous (rows): tileRows * contigAllocStride * elemBytes
    //   - Along contiguous (cols): tileCols * elemBytes
    int64_t contigAllocStride = contigDimSize;
    if (swizzling == 0 && !llvm::isPowerOf2_64(contigDimSize)) {
      contigAllocStride = llvm::NextPowerOf2(contigDimSize);
    }

    int nonContigDimIdx = transposed ? (int)(allocShape.size() - 1) : 0;
    int64_t nonContigDimSize = allocShape[nonContigDimIdx];
    if (!llvm::isPowerOf2_64(nonContigDimSize)) {
      nonContigDimSize = llvm::NextPowerOf2(nonContigDimSize);
    }
    int64_t numRowGroups = nonContigDimSize / tileRows;

    // Compute stride bytes. For sub-byte types (FP4: bitwidth=4), compute in
    // bits first to avoid integer truncation from bitwidth/8=0.
    int64_t sboStrideBytes, lboStrideBytes;
    if (swizzling > 0) {
      // Swizzled: tiles are contiguous blocks in memory.
      int64_t coreTileBytes = (int64_t)tileCols * tileRows * bitwidth / 8;
      sboStrideBytes = coreTileBytes;
      lboStrideBytes = numRowGroups * coreTileBytes;
    } else {
      // No swizzle: simple row-major layout with row stride =
      // contigAllocStride. sboStrideBytes = stride from one 8-row tile-group to
      // the next
      //                = tileRows * contigAllocStride * elemBytes
      // lboStrideBytes = stride from one tileCols-wide column-group to the next
      //                = tileCols * elemBytes
      sboStrideBytes = (int64_t)tileRows * contigAllocStride * bitwidth / 8;
      lboStrideBytes = (int64_t)tileCols * bitwidth / 8;
    }

    int leadingDim = transposed ? 0 : 1;
    int stridedDim = transposed ? 1 : 0;
    int tileExtentLeading = tileCols;
    int tileExtentStrided = tileRows;

    bool MNContig = (MNdim == 0) == transposed;
    bool swapLboSbo = (swizzling == 0 && MNContig);
    if (swapLboSbo) {
      std::swap(leadingDim, stridedDim);
      std::swap(tileExtentLeading, tileExtentStrided);
    }

    int lboDescriptor = 0;
    if ((int)instrShape[leadingDim] > tileExtentLeading) {
      lboDescriptor = (swapLboSbo ? sboStrideBytes : lboStrideBytes) >> 4;
    }

    int sboDescriptor = 0;
    if ((int)instrShape[stridedDim] > tileExtentStrided) {
      sboDescriptor = (swapLboSbo ? lboStrideBytes : sboStrideBytes) >> 4;
    }

    // K=96 mxf4nvf4 .block16 (sm_103a only): the packed K=96 row is 48 bytes
    // and would overflow the 128B SMEM boundary if packed contiguously, so the
    // descriptor must use the absolute-address LBO mode (bit 52 = 1). In that
    // mode the LBO field encodes the absolute SMEM address of the second 64B
    // chunk; the compile-time descriptor sets bit 52 and zeros the LBO field,
    // and smemLoad ORs in the runtime address (base + 64B). Per PTX ISA 9.2
    // §9.7.16.3.1.3, absolute mode requires 128B swizzle and K-major
    // (transpose = 0). instrShape leading dim is the contiguous (K) extent.
    bool isFp4K96 = isFp4 && (int)instrShape[transposed ? 0 : 1] == 96;
    bool useAbsoluteLbo = isFp4K96;
    if (useAbsoluteLbo) {
      // Hard error (not assert): the operand allocation must hand us a 128B
      // encoding for absolute-LBO mode (forced in AccelerateMatmul's
      // getSharedMemoryMMAOperand). The assert was compiled out under NDEBUG
      // (mode/opt), which previously let a non-128B-swizzle operand silently
      // emit a self-contradictory descriptor and miscompute on B300. Trap at
      // compile time instead. PTX ISA 9.2 §9.7.16.3.1.3 restriction 1.
      if (swizzling != 128)
        llvm::report_fatal_error(
            "K=96 mxf4nvf4 absolute-LBO mode requires 128B swizzle "
            "(PTX ISA 9.2 §9.7.16.3.1.3 restriction 1)");
      // K-major (ISA restriction 2) is already guaranteed here: useAbsoluteLbo
      // is set via isFp4K96, which tests the CONTIGUOUS dim
      // (instrShape[transposed ? 0 : 1] == 96), so K==96 is the contiguous axis
      // for both A (transposed = false, K at dim1) and B (transposed = true, K
      // at dim0). The encoding's `transposed` flag indicates which axis is
      // contiguous, NOT the MMA instruction's transpose bit (enforced at
      // instruction-descriptor emission), so it is expected to be true for the
      // B operand and must NOT be rejected. LBO field is filled at runtime (see
      // smemLoad), so zero it here so the compile-time bit pattern doesn't
      // collide with the runtime add.
      lboDescriptor = 0;
    }

    // Construct the descriptor
    SMEMDescriptor desc;
    desc.descriptor = mmaVersion == 5 ? 1ULL << 46 : 0ULL;
    desc.leadDimensionBaseOffset = lboDescriptor;
    desc.strideDimensionBaseOffset = sboDescriptor;
    if (useAbsoluteLbo) {
      desc.lboAbsoluteMode = 1;
    }
    switch (swizzling) {
    case 0:
      desc.swizzlingMode = 0;
      break;
    case 32:
      desc.swizzlingMode = 3;
      break;
    case 64:
      desc.swizzlingMode = 2;
      break;
    case 128:
      desc.swizzlingMode = 1;
      break;
    default:
      llvm_unreachable("Unsupported swizzling size.");
    }
    return MMASMEMDescriptor{/* .descriptor = */ desc,
                             /* .swizzlingByteWidth = */ swizzling,
                             /* .bitwidth = */ bitwidth,
                             /* .transposed = */ transposed,
                             /* .fp4Padded = */ fp4Padded,
                             /* .lboAbsoluteMode = */ useAbsoluteLbo};
  }
};

// Helper class to load tensor memory following MMAv5 layout.
class DotOpMmaV5TmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaV5TmemLoader() {}
  static DotOpMmaV5TmemLoader build(Location loc, RewriterBase &rewriter,
                                    gpu::MemDescType memTy, Value tmemBase);

  MemDescOperand tmemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                          Location loc) const;

  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override {
    return tmemLoad(a, b, rewriter, loc);
  }

private:
  DotOpMmaV5TmemLoader(LinearLayout ll, Value address, int bitwidth)
      : ll(std::move(ll)), address(address), bitwidth(bitwidth) {}

  LinearLayout ll;
  Value address;
  int bitwidth;
};

static Value getOffsetedBase(Value v, gpu::MemDescType memDescTy,
                             const TypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  TritonLLVMOpBuilder tb(loc, rewriter);
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, v, llvmElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto base = smemObj.getBase();
  return tb.gep(base.getType(), llvmElemTy, base, offset);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
