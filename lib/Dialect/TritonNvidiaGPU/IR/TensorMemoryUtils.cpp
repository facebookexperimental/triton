#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

namespace {

constexpr int maxRegisters = 256;
constexpr int largestTmemLoadStore = 128;

// Similar to largestVectorisation in TritonGPUToLLVM/Utility.cpp
std::optional<std::tuple<LinearLayout, ColumnAction, int>>
getVec(const LinearLayout &cvt, const LinearLayout &tile, int maxnreg) {
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kCol = StringAttr::get(ctx, "col");
  LinearLayout reps, vec;
  ColumnAction perm;
  // Heuristic:
  // Do not use more than half the registers as otherwise it's prone to spilling
  assert(maxnreg / 2 <= largestTmemLoadStore);
  auto maxReg = maxnreg / 2;
  // Heuristic:
  // If maxnreg is 256 and we need more than one message, we don't use max
  // vectorisation as ptxas' scheduler breaks...
  if (maxnreg == 256 && cvt.getInDimSize(kReg) > maxReg) {
    maxReg /= 2;
  }
  auto maxVec = maxReg / tile.getInDimSize(kReg);
  int i = 1;
  for (; i <= maxVec; i *= 2) {
    vec = LinearLayout::identity1D(i, kReg, kCol);
    auto vecTile = tile * vec;
    auto maybePerm = regPermForDivide(cvt, vecTile, /*left=*/true);
    if (!maybePerm) {
      break;
    }
    // nb. We could remove this part once we are confident the algo works
    perm = *maybePerm;
    auto newCvt = maybePerm->apply(cvt);
    auto maybeReps = getReps(newCvt, vecTile);
    if (!maybeReps.has_value()) {
      break;
    }
    reps = *maybeReps;
  }
  if (i == 1) {
    // Couldn't lower the tile
    return std::nullopt;
  }
  // i is the smallest power of 2 that *cannot* be used to lower the tile
  // so we return i / 2.
  assert(i > 1);
  return std::make_tuple(std::move(reps), std::move(perm),
                         (i / 2) * tile.getInDimSize(kReg));
}
} // namespace

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
int getContextualMaxNReg(Operation *op) {
  // Check the immediate parent op to see if it places a register constraint.
  auto getFromParent = [](Operation *op) -> std::optional<int> {
    Operation *parent = op->getParentOp();
    if (auto mod = dyn_cast<ModuleOp>(parent)) {
      if (auto attr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
        return attr.getInt();
      return {};
    }

    if (auto partitions = dyn_cast<WarpSpecializePartitionsOp>(parent)) {
      // Check if the partition has reduced registers.
      unsigned idx = op->getParentRegion()->getRegionNumber();
      if (auto actRegisters = partitions.getParentOp().getActualRegisters())
        return (*actRegisters)[1 + idx];
      if (auto requestedRegisters =
              partitions.getParentOp().getRequestedRegisters())
        return (*requestedRegisters)[idx];
      return {};
    }

    if (auto wsOp = dyn_cast<WarpSpecializeOp>(op->getParentOp())) {
      // Check the register usage of the default warpgroup.
      if (auto actRegisters = wsOp.getActualRegisters())
        return actRegisters->front();
      return {};
    }

    return {};
  };

  // PTXAS validates the register usage of `tcgen05.ld` and `tcgen05.st`
  // instructions based on the static number of registers set on the module, not
  // the dynamic allocation. This just means the register limit used for the
  // purpose of subtiling TMEM messages cannot be higher than the module's.
  auto mod = op->getParentOfType<ModuleOp>();
  int maxnreg = maxRegisters;

  for (; op != mod; op = op->getParentOp()) {
    if (std::optional<int> limit = getFromParent(op)) {
      maxnreg = std::min(maxnreg, *limit);
      break;
    }
  }

  if (auto maxnregAttr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
    maxnreg = std::min<int>(maxnreg, maxnregAttr.getInt());

  return maxnreg;
}

FailureOr<TMemLdStEncodingInfo>
lowerTMemLdSt(const LinearLayout &cvt, int maxnreg, int bitwidth, bool isScales,
              std::function<InFlightDiagnostic()> emitError,
              bool unpacked = false) {
  // We will fill in the returned value recursively (if it exists)

  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto info = lowerTMemLdSt(prmtCvt, maxnreg, bitwidth, isScales, emitError,
                              unpacked);
    if (failed(info))
      return failure();
    info->broadcast = std::move(removeBroadcastSrc);
    return info;
  }
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kRow = S("row");
  auto kCol = S("col");
  if (bitwidth < 32) {
    auto [bestContig, quot] =
        factorMaximalIdentityPrefix(cvt, kReg, kCol, 32 / bitwidth);
    bool padding = false;
    int newBitwidth = bitwidth;
    if (bestContig > 1) {
      // There are contiguous elements along kCol, so we can pack them into a
      // larger dtype
      unpacked = false;
      newBitwidth = bitwidth * bestContig;
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) *
                            LinearLayout::identity1D(2, kReg, kCol));
               bitwidth == 16 && maybeQuot) {
      // Unpacked just supported for bitwidth 16
      unpacked = true;
      quot = *maybeQuot;
      newBitwidth = 32;
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth))) {
      // We software-pad the elements when we either do not have enough elements
      // to fill a full 32b register, e.g., colN = 1 and colStride != 1 or when
      // bitwidth == 8 (this happens with scales with K=1).
      // These two cases are mostly supported for testing purposes.
      // A padded TMEM value occupies one physical column, so an unpacked
      // store would overwrite the adjacent column.
      unpacked = false;
      quot = *maybeQuot;
      padding = true;
      newBitwidth = 32;
    } else {
      if (emitError) {
        emitError() << "Failed to lower TMEM load/store: TMEM layout is not "
                       "packed or unpacked";
      }
      return failure();
    }
    // When unpacked each register moves 32/bitwidth (= 2) columns
    if (unpacked) {
      quot = LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) * quot;
    }
    auto info = lowerTMemLdSt(quot, maxnreg, newBitwidth, isScales, emitError,
                              unpacked);
    if (failed(info))
      return failure();
    if (bestContig > 1) {
      info->vec = bestContig;
    }
    if (unpacked) {
      info->unpacked = true;
    }
    if (padding) {
      info->padding = true;
    }
    return info;
  }

  assert(bitwidth == 32);

  // The algorithm goes as:
  // - Try to match the tile with one of the standard messages
  // - If it doesn't match, we use the 16x32bx2 message
  // Note that it can match one and only one of the layouts, even after register
  // reordering, as the layouts yield predetermined positions for the lanes
  // We store the instruction, the resulting reps layout, the permutation and
  // the number of registers per message
  std::optional<TMemLdStEncodingInfo> msgInfo;
  // A task of 1 or 2 warps covers only the lanes of its own warps.
  int numWarpBases = std::min(2, cvt.getInDimSizeLog2(S("warp")));
  for (auto atom : {TMemAccessAtom::I32x32b, TMemAccessAtom::I16x256b,
                    TMemAccessAtom::I16x64b, TMemAccessAtom::I16x128b}) {
    auto tile =
        getTileLayout(ctx, atom, unpacked, /*withWarp=*/true, numWarpBases);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      // Cannot match more than one
      msgInfo = {atom, std::get<0>(*maybeReps), std::get<1>(*maybeReps),
                 std::get<2>(*maybeReps)};
      break;
    }
  }
  std::optional<uint32_t> secondHalfOffset = std::nullopt;
  if (!msgInfo) {
    // Quotient by the smaller tile and then, if possible, we set the
    // secondHalfOffset to the last kLane basis
    auto tile = getTileLayout(ctx, TMemAccessAtom::I16x32bx2, unpacked,
                              /*withWarp=*/true, numWarpBases);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto [reps, perm, numRegsPerMessage] = std::move(*maybeReps);
      // Find the last kLane basis and use it as secondHalfOffset
      auto row = reps.getBasis(kLane, 4, kRow);
      auto col = reps.getBasis(kLane, 4, kCol);
      secondHalfOffset = (row << 16) | col;
      if (*secondHalfOffset == 0) {
        // The Buck toolchain still needs a nonzero second-half offset when
        // writing only 16 scale elements; the allocation is padded for this.
        if (!isScales) {
          if (emitError)
            emitError()
                << "Only supported for scales as we pad the allocation.";
          return failure();
        }
        secondHalfOffset = 1;
      }
      // We "quotient it out", meaning we remove the last basis from reps
      auto basis = reps.getBases();
      basis[kLane][4] = {0, 0};
      reps = LinearLayout(std::move(basis), reps.getOutDims(),
                          /*isSurjective=*/false);
      msgInfo = {TMemAccessAtom::I16x32bx2, reps, perm, numRegsPerMessage};
    }
  }

  if (!msgInfo) {
    if (emitError) {
      emitError()
          << "Failed to lower TMEM load/store: unsupported dst layout\n" +
                 cvt.toString();
    }
    return failure();
  }
  auto info = std::move(*msgInfo);
  info.secondHalfOffset = secondHalfOffset;
  return info;
}

FailureOr<MemDescType>
getNpotTMemChunkType(MemDescType memTy,
                     std::function<InFlightDiagnostic()> emitError) {
  ArrayRef<int64_t> shape = memTy.getShape().take_back(2);
  auto enc = dyn_cast<TensorMemoryEncodingAttr>(memTy.getEncoding());
  auto scales = dyn_cast<TensorMemoryScalesEncodingAttr>(memTy.getEncoding());
  // blockM=64 keeps NPOT N off: its unused TMEM columns per warp are unproven.
  if (!llvm::isPowerOf2_64(shape[0]) ||
      !(enc ? enc.getBlockM() == 128 &&
                  enc.getCtaMode() == TensorMemoryCTAMode::DEFAULT &&
                  enc.getCGALayout().getCTASplitNum()[1] == 1
            : scales && scales.getCtaMode() == TensorMemoryCTAMode::DEFAULT &&
                  getNumCTAs(scales) == 1)) {
    if (emitError)
      emitError() << "NPOT TMEM load/store is not yet supported for " << memTy;
    return failure();
  }
  int64_t c = LinearLayout::twoPartOf(shape[1]);
  return MemDescType::get({shape[0], c}, memTy.getElementType(),
                          memTy.getEncoding(), memTy.getMemorySpace(),
                          memTy.getMutableMemory(), memTy.getAllocShape());
}

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError) {
  // An NPOT N = c * q (q odd) takes the plan of its pow2 chunk of c columns,
  // and the outermost register digit, of radix q, steps reps across chunks.
  if (!llvm::all_of(regTy.getShape(), llvm::isPowerOf2_64)) {
    auto chunkTy = getNpotTMemChunkType(memTy, emitError);
    if (failed(chunkTy))
      return failure();
    auto *ctx = regTy.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    LinearLayout regLayout = toLinearLayout(regTy);
    // Four scale bytes share a 32-bit TMEM column, so a K chunk can't move
    // alone: the pow2 alloc shape moves instead, with its registers past K
    // padded or dropped.
    if (isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
      auto paddedTy = MemDescType::get(
          getNpotPow2AllocShape(memTy.getShape()), memTy.getElementType(),
          memTy.getEncoding(), memTy.getMemorySpace(), memTy.getMutableMemory(),
          memTy.getAllocShape());
      std::optional<LinearLayout> paddedLayout =
          getDistributedLayoutForTmemLdSt(
              paddedTy, TMemAccessAtom::I32x32b,
              regLayout.getInDimSize(StringAttr::get(ctx, "warp")));
      // At 8 warps the pow2 alloc shape puts its second warpgroup on a K column
      // the NPOT K lacks, so it takes regTy's warp basis in place of the
      // matching register.
      if (paddedLayout) {
        auto bases = paddedLayout->getBases();
        auto want = regLayout.getBases().lookup(StringAttr::get(ctx, "warp"));
        for (auto [w, basis] :
             llvm::zip(bases[StringAttr::get(ctx, "warp")], want))
          if (auto it = llvm::find(bases[kReg], basis); it != bases[kReg].end())
            std::swap(*it, w);
        paddedLayout =
            LinearLayout(std::move(bases), paddedLayout->getOutDims(),
                         paddedLayout->isSurjective());
      }
      SmallVector<int32_t> paddedRegs;
      // Padded register r and register paddedRegs[r] must name the same element
      // in every thread, so the lane, warp and block maps must agree.
      if (paddedLayout && regTy.getShape() == memTy.getShape() &&
          llvm::all_of(regLayout.getInDimNames(), [&](StringAttr dim) {
            return dim == kReg || regLayout.getBases().lookup(dim) ==
                                      paddedLayout->getBases().lookup(dim);
          })) {
        DenseMap<std::pair<int32_t, int32_t>, int32_t> regOf;
        auto elementOf = [&](const LinearLayout &ll, int32_t reg) {
          SmallVector<std::pair<StringAttr, int32_t>> ins;
          for (StringAttr dim : ll.getInDimNames())
            ins.push_back({dim, dim == kReg ? reg : 0});
          auto out = ll.apply(ins);
          return std::make_pair(out[0].second, out[1].second);
        };
        for (int32_t r = 0; r < regLayout.getInDimSize(kReg); ++r)
          regOf[elementOf(regLayout, r)] = r;
        for (int32_t r = 0; r < paddedLayout->getInDimSize(kReg); ++r)
          paddedRegs.push_back(
              regOf.lookup_or(elementOf(*paddedLayout, r), -1));
        if (llvm::count_if(paddedRegs, [](int32_t r) { return r >= 0; }) !=
            regLayout.getInDimSize(kReg))
          paddedRegs.clear();
      }
      if (paddedRegs.empty()) {
        if (emitError)
          emitError() << "NPOT TMEM scale load/store is not yet supported for "
                      << regTy;
        return failure();
      }
      auto paddedRegTy = RankedTensorType::get(
          paddedTy.getShape(), regTy.getElementType(),
          LinearEncodingAttr::get(ctx, std::move(*paddedLayout)));
      auto info = computeTMemLdStEncodingInfo(paddedRegTy, paddedTy, maxnreg,
                                              emitError);
      if (succeeded(info))
        info->npotPaddedRegs = std::move(paddedRegs);
      return info;
    }
    auto dims = to_vector(regLayout.getOutDimNames());
    int64_t m = regTy.getDimSize(0), c = chunkTy->getDimSize(1);
    int32_t q = regTy.getDimSize(1) / c;
    int32_t numBinary = regLayout.getNumLeadingRadix2Digits(kReg);
    auto bases = regLayout.getBases();
    bases[kReg].resize(numBinary);
    auto chunkLayout =
        LinearLayout::tryCreate(std::move(bases), {{dims[0], m}, {dims[1], c}},
                                /*requireSurjective=*/false);
    int32_t odd = 0;
    // A general offset table hides the odd digit's step from its basis.
    if (regLayout.hasOddPartForm() && chunkLayout &&
        regTy.getShape() == memTy.getShape().take_back(2) &&
        regLayout.getInDimSizeLog2(kReg) == numBinary + 1 &&
        regLayout.getInDimSize(kReg) == q << numBinary)
      odd = regLayout.getBasisContribution(kReg, numBinary, dims[1]).odd;
    if (odd == 0) {
      if (emitError)
        emitError() << "NPOT TMEM load/store is not yet supported for register "
                       "layout "
                    << regLayout.toString();
      return failure();
    }
    auto chunkRegTy = RankedTensorType::get(
        {m, c}, regTy.getElementType(),
        LinearEncodingAttr::get(ctx, std::move(*chunkLayout)));
    auto info =
        computeTMemLdStEncodingInfo(chunkRegTy, *chunkTy, maxnreg, emitError);
    if (failed(info))
      return failure();
    // Register block k is chunk k * odd mod q. Its memory column offset adds to
    // the chunk's own columns, which can pass q chunk widths (M blocks), so the
    // offsets go in a table, bounded by q chunk extents.
    int32_t numDigits = info->reps.getInDimSizeLog2(kReg);
    auto kCol = StringAttr::get(ctx, "col");
    auto repsBases = info->reps.getBases();
    auto radices = info->reps.getInputRadices();
    auto outDims = info->reps.getOutDims();
    repsBases[kReg].push_back({0, 0});
    radices[kReg].push_back(q);
    outDims[1].second *= q;
    llvm::MapVector<StringAttr, SmallVector<std::pair<StringAttr, int32_t>>>
        tableDigits;
    llvm::MapVector<StringAttr, std::vector<int32_t>> tableOffsets;
    tableDigits[kCol] = {{kReg, numDigits}};
    for (int32_t k = 0; k < q; ++k)
      tableOffsets[kCol].push_back(getTMemSubSliceOffset(
          memTy, int32_t(k * odd % q * c), memTy.getRank() - 1));
    std::string diagnostic;
    auto reps = LinearLayout::tryCreate(
        std::move(repsBases), std::move(radices), outDims,
        std::move(tableDigits), std::move(tableOffsets),
        /*requireSurjective=*/false, &diagnostic);
    if (!reps) {
      if (emitError)
        emitError() << "NPOT TMEM load/store is not yet supported: "
                    << diagnostic;
      return failure();
    }
    info->reps = std::move(*reps);
    SmallVector<size_t> action;
    for (const auto &basis :
         info->perm.apply(LinearLayout::identity1D(1 << numDigits, kReg, kReg))
             .getBases()
             .lookup(kReg))
      action.push_back(llvm::Log2_32(basis[0]));
    action.push_back(numDigits);
    SmallVector<int32_t> permRadices(numDigits, 2);
    permRadices.push_back(q);
    info->perm = ColumnAction(action, kReg, numDigits + 1, permRadices);
    return info;
  }
  auto memLayout = toLinearLayout(memTy);
  auto regLayout = toLinearLayout(regTy);
  auto *ctx = regTy.getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kBlock = S("block");
  auto kWarp = S("warp");
  auto kRow = S("row");
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    if (emitError) {
      emitError() << "The cga_layout of the register and memory layout must be "
                     "the same. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  cvt = maybeSublayout.value();
  // Warps 0-3 must map to row=32 and row=64 whether with broadcasting or not.
  // A task of 1 or 2 warps has only the first of these bases (it accesses a
  // row slice held in the lanes of its own warps).
  int numWarpBases = std::min(2, regLayout.getInDimSizeLog2(kWarp));
  bool warpsMatch = memLayout.getInDimSizeLog2(kRow) >= 5 + numWarpBases;
  for (int i = 0; warpsMatch && i < numWarpBases; ++i)
    warpsMatch =
        regLayout.getBasis(kWarp, i) == memLayout.getBasis(kRow, 5 + i);
  if (!warpsMatch) {
    if (emitError) {
      emitError() << "warps=1,2 must map to rows=32,64. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  // Such a task reaches only the lanes of its own warps, so the memory must
  // hold no row past them. A row slice keeps the row bits of its source with
  // the bases past the slice set to zero, so the bit count alone cannot tell
  // a slice from a whole buffer.
  bool rowsInWarps = true;
  for (int i = 5 + numWarpBases;
       numWarpBases < 2 && rowsInWarps && i < memLayout.getInDimSizeLog2(kRow);
       ++i)
    rowsInWarps = llvm::all_of(memLayout.getBasis(kRow, i),
                               [](int32_t b) { return b == 0; });
  if (!rowsInWarps) {
    if (emitError) {
      emitError() << "A task of " << (1 << numWarpBases)
                  << " warp(s) can only access the TMEM rows in the lanes of "
                     "its own warps. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  // Map warp bases to row=32 and row=64 in the cvt. This would be done
  // automatically in `invertAndCompose` if we had a different dimension name
  // for these rows. We can do this in the future if needed.
  auto bases = cvt.getBases();
  for (int i = 0; i < numWarpBases; ++i)
    bases[kWarp][i] = {32 << i, 0};
  cvt = LinearLayout(std::move(bases), cvt.getOutDims(),
                     /*isSurjective=*/cvt.isSurjective());

  bool isScales = isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding());
  int bitwidth = memTy.getElementTypeBitWidth();
  return lowerTMemLdSt(cvt, maxnreg, bitwidth, isScales, emitError);
}

FailureOr<LinearLayout> getTMemLdStMessageLayout(MLIRContext *ctx,
                                                 TMemAccessAtom atom,
                                                 bool unpacked,
                                                 int numRegsPerMessage) {
  auto kReg = StringAttr::get(ctx, "register");
  auto kCol = StringAttr::get(ctx, "col");
  LinearLayout atomLayout =
      getTileLayout(ctx, atom, unpacked, /*withWarp=*/false);
  int32_t atomRegs = atomLayout.getInDimSize(kReg);
  if (numRegsPerMessage % atomRegs != 0)
    return failure();
  return atomLayout *
         LinearLayout::identity1D(numRegsPerMessage / atomRegs, kReg, kCol);
}

FailureOr<SmallVector<DenseSet<uint32_t>>>
getTMemLdStWarpAddresses(RankedTensorType regTy, MemDescType memTy, int maxnreg,
                         uint32_t baseAddress) {
  auto info = computeTMemLdStEncodingInfo(regTy, memTy, maxnreg);
  if (failed(info))
    return failure();

  auto *ctx = regTy.getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  const LinearLayout &reps = info->reps;
  if (!reps.hasInDim(kReg) || !reps.hasInDim(kLane) || !reps.hasInDim(kWarp))
    return failure();

  auto getRowCol = [&](const auto &outputs) {
    std::optional<uint32_t> row;
    std::optional<uint32_t> col;
    for (auto [name, value] : outputs) {
      if (name == kRow)
        row = value;
      else if (name == kCol)
        col = value;
    }
    return std::make_pair(row, col);
  };
  auto encode = [](uint32_t row, uint32_t col) {
    assert(col < (1u << 16) && "TMEM column offset must fit in 16 bits");
    return (row << 16) | col;
  };

  auto messageLayout = getTMemLdStMessageLayout(ctx, info->atom, info->unpacked,
                                                info->numRegsPerMessage);
  if (failed(messageLayout))
    return failure();

  SmallVector<uint32_t> messageOffsets;
  for (int32_t messageReg = 0; messageReg < messageLayout->getInDimSize(kReg);
       ++messageReg) {
    for (int32_t lane = 0; lane < messageLayout->getInDimSize(kLane); ++lane) {
      auto [row, col] =
          getRowCol(messageLayout->apply({{kReg, messageReg}, {kLane, lane}}));
      if (!row || !col)
        return failure();
      messageOffsets.push_back(encode(*row, *col));
    }
  }

  SmallVector<DenseSet<uint32_t>> addressesByWarp;
  addressesByWarp.resize(reps.getInDimSize(kWarp));
  for (int32_t warp = 0; warp < reps.getInDimSize(kWarp); ++warp) {
    // Keep this warp-base calculation synchronized with lowerTMemLdSt.
    uint32_t warpOffset = (warp & 3) << (5 + 16);
    if (reps.getInDimSize(kWarp) > 4) {
      auto [row, col] =
          getRowCol(reps.apply({{kReg, 0}, {kLane, 0}, {kWarp, warp}}));
      if (!row || !col)
        return failure();
      warpOffset += encode(*row, *col);
    }

    DenseSet<uint32_t> addresses;
    for (int32_t reg = 0; reg < reps.getInDimSize(kReg);
         reg += info->numRegsPerMessage) {
      auto [messageRow, messageCol] =
          getRowCol(reps.apply({{kReg, reg}, {kLane, 0}, {kWarp, 0}}));
      if (!messageRow || !messageCol)
        return failure();
      uint32_t messageOffset = encode(*messageRow, *messageCol);

      for (uint32_t inMessageOffset : messageOffsets) {
        uint32_t address =
            baseAddress + warpOffset + messageOffset + inMessageOffset;
        addresses.insert(address);
        if (info->secondHalfOffset)
          addresses.insert(address + *info->secondHalfOffset);
      }
    }
    addressesByWarp[warp] = std::move(addresses);
  }
  return addressesByWarp;
}

bool supportsTMemLoadReduce(RankedTensorType regTy, MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError) {
  if (!llvm::all_of(regTy.getShape(), llvm::isPowerOf2_64)) {
    if (emitError)
      emitError() << "NPOT tmem_load reduction is not yet supported";
    return false;
  }
  auto encodingInfo = computeTMemLdStEncodingInfo(regTy, memTy, maxnreg);
  if (failed(encodingInfo)) {
    if (emitError)
      emitError() << "failed to compute TMEM encoding info";
    return false;
  }

  if (encodingInfo->unpacked) {
    if (emitError)
      emitError() << "tmem_load reduction requires packed format "
                     "(unpacked=false)";
    return false;
  }

  auto kReg = StringAttr::get(regTy.getContext(), "register");
  constexpr int dimM = 0, dimN = 1;
  auto regDims =
      toLinearEncoding(regTy).basesPerDim(kReg, /*skipBroadcast=*/true);

  // The fused ld.red instruction reduces the values that are already local to a
  // thread, so the N axis must live entirely in this thread's registers,
  // otherwise the N reduction would be partial and need cross-lane/warp/CTA
  // combining.
  if (regDims[dimN] != toLinearLayout(regTy).getOutDimSizes().begin()[dimN]) {
    if (emitError)
      emitError() << "tmem_load reduction with N dimension sharded across "
                     "threads is not supported.";
    return false;
  }

  // regDims[dimM] is the number of distinct M coordinates a single thread holds
  // across its registers. Require it to be 1, so each thread owns exactly one M
  // row.
  if (regDims[dimM] != 1) {
    if (emitError)
      emitError() << "tmem_load reduction with multiple M rows per thread is "
                     "not supported.";
    return false;
  }

  return true;
}

} // namespace mlir::triton::nvidia_gpu
