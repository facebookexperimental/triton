// PromoteLoadToTMA.cpp — Auto-promote eligible tt.load to a real TMA load.
//
// Detects masked, block-structured global loads --
//   tl.load(base + affine_block_offsets, mask = offs < shape)
// -- and rewrites them, at TTIR time, into the explicit-TMA representation the
// existing pipeline already lowers on Hopper (sm_90+):
//   %d = tt.make_tensor_descriptor(base, shape, strides) : !tt.tensordesc<...>
//   %v = tt.descriptor_load %d[tile_offsets] : ... -> tensor<BLOCK x dtype>
//
// The downstream nvidia pass `triton-nvidia-tma-lowering` turns descriptor_load
// into AsyncTMACopyGlobalToLocal (cp.async.bulk.tensor) and
// get_tensordesc_metadata records the host CUtensorMap params. Enabled
// per-kernel by the autotunable `auto_tma` compile option (a
// `triton.Config(..., auto_tma=True)` field) or, as a global default, the
// `TRITON_AUTO_TMA` env knob
// -- see make_ttir: `opt.auto_tma or knobs.nvidia.auto_tma`. Only invoked for
// sm_90+; non-eligible loads are left untouched.

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdlib>
#include <functional>
#include <string>

namespace tt = mlir::triton;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTELOADTOTMAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// CU_TENSOR_MAP_DATA_TYPE values (device-side; remapped to host in the
// launcher).
enum TMADataType {
  TMA_UINT8 = 0,
  TMA_UINT16 = 1,
  TMA_UINT32 = 2,
  TMA_UINT64 = 3,
  TMA_INT32 = 4,
  TMA_INT64 = 5,
  TMA_FLOAT16 = 6,
  TMA_FLOAT32 = 7,
  TMA_FLOAT64 = 8,
  TMA_BFLOAT16 = 9,
  TMA_FLOAT8_E4M3 = 13,
  TMA_FLOAT8_E5M2 = 14,
};

static std::optional<int> getTMADataType(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    switch (intTy.getWidth()) {
    case 8:
      return TMA_UINT8;
    case 16:
      return TMA_UINT16;
    case 32:
      return TMA_INT32;
    case 64:
      return TMA_INT64;
    default:
      return std::nullopt;
    }
  }
  if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
    if (floatTy.isF16())
      return TMA_FLOAT16;
    if (floatTy.isBF16())
      return TMA_BFLOAT16;
    if (floatTy.isF32())
      return TMA_FLOAT32;
    if (floatTy.isF64())
      return TMA_FLOAT64;
    if (llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType>(floatTy))
      return TMA_FLOAT8_E4M3;
    if (llvm::isa<Float8E5M2Type, Float8E5M2FNUZType>(floatTy))
      return TMA_FLOAT8_E5M2;
    return std::nullopt;
  }
  return std::nullopt;
}

// Per-dimension decomposition of a load's pointer expression.
struct DimInfo {
  int64_t blockSize;
  Value stride;     // outer-dim stride value (null if contiguous/innermost)
  Value offset;     // tile start offset value (program_id * BLOCK, in elements)
  int strideArgIdx; // func arg index for stride (-1 if contiguous)
  int shapeArgIdx = -1; // func arg index for global extent, recovered from a
                        // `offs % M` modulus (-1 if not modulo-bounded)
  // Loop-advanced dim (pointer-increment K loop): the tile offset is not a
  // fixed affine value but `loopIV * perIterElems`, reconstructed at rewrite
  // time.
  bool loopAdvanced = false;
  Value loopIV;             // scf.for induction variable
  int64_t perIterElems = 0; // per-iteration advance along this (contiguous) dim
};

struct DecomposedLoad {
  Value basePtr;
  int basePtrArgIndex;
  SmallVector<DimInfo> dims;
  // Uniform (per-program) scalar offset terms with no arange/expand_dims, e.g.
  // `off_z * stride_z` in a batched/FA kernel. These shift the whole tile, so
  // for a device descriptor they fold into the descriptor base pointer
  // (make_tensor_descriptor(addptr(base, Σ scalarBaseOffsets), ...)).
  SmallVector<Value> scalarBaseOffsets;
  bool valid;
};

static int getFuncArgIndex(Value v) {
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (isa<tt::FuncOp>(blockArg.getOwner()->getParentOp()))
      return blockArg.getArgNumber();
  }
  return -1;
}

static Value matchSplat(Value v) {
  if (auto splatOp = v.getDefiningOp<tt::SplatOp>())
    return splatOp.getSrc();
  return {};
}

static bool matchMakeRange(Value v, int64_t &size) {
  if (auto rangeOp = v.getDefiningOp<tt::MakeRangeOp>()) {
    if (rangeOp.getStart() == 0) {
      size = rangeOp.getEnd();
      return true;
    }
  }
  return false;
}

// True if v is a constant all-zeros value: tt.splat of a zero scalar, a dense
// splat zero (int or float) constant, or a scalar zero constant. Used for the
// select-clamp false value (below) and mirrors the OOB `other`-fill check in
// isTMAEligible -- both must be zero for the TMA zero-fill equivalence to hold.
static bool isConstantZero(Value v) {
  if (!v)
    return false;
  if (Value s = matchSplat(v))
    return matchPattern(s, m_Zero()) || matchPattern(s, m_AnyZeroFloat());
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue()))
      return d.isSplat() && d.getSplatValue<APInt>().isZero();
    if (auto d = dyn_cast<DenseFPElementsAttr>(c.getValue()))
      return d.isSplat() && d.getSplatValue<APFloat>().isZero();
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getValue().isZero();
  }
  return false;
}

// True if v is a bare tile-index expression -- make_range(0,B), or
// make_range + splat(tile_off) -- with NO stride multiply. The RemSI/Select
// OOB-idiom peels below record the modulus/clamp bound as this dim's global
// extent, which is a valid element-count extent only when the modulus/clamp
// wraps the RAW index. If it instead wraps a strided expression (e.g.
// (idx*stride) % M), the recorded arg is not an element extent and the
// descriptor's global_dim would be wrong -- so only peel a raw index.
static bool isRawIndexOffset(Value v) {
  int64_t b;
  if (matchMakeRange(v, b))
    return true;
  if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
    Value l = addOp.getLhs(), r = addOp.getRhs();
    for (int s = 0; s < 2; s++) {
      if (matchMakeRange(l, b) && matchSplat(r))
        return true;
      std::swap(l, r);
    }
  }
  return false;
}

// Match (make_range(0,BLOCK) + splat(tile_off)) * splat(stride), or the
// contiguous form make_range(0,BLOCK) + splat(tile_off).
static bool match1DOffset(Value offsetVal, DimInfo &info) {
  // Peel the no-mask OOB idiom `offs = (pid*BLOCK + arange) % M`: the modulus
  // is this dim's global extent. Record the modulus kernel arg as the shape
  // source and decompose the inner affine offset. (Upstream Triton matmuls wrap
  // operand offsets with `% M` / `% N` instead of masking M / N.) Promoting
  // such a load to TMA is correct: the descriptor uses global_dim = M with zero
  // OOB fill, and the kernel's epilogue store mask already discards the tail.
  if (auto remOp = offsetVal.getDefiningOp<arith::RemSIOp>()) {
    if (Value modSplat = matchSplat(remOp.getRhs())) {
      int modArg = getFuncArgIndex(modSplat);
      // Only peel when the modulus wraps the RAW index (not a
      // strided/byte-scaled expression); otherwise the modulus arg is not this
      // dim's element extent.
      if (modArg >= 0 && isRawIndexOffset(remOp.getLhs())) {
        info.shapeArgIdx = modArg;
        offsetVal = remOp.getLhs();
      }
    }
  }
  // Peel the other no-mask OOB idiom `offs = select(offs < M, offs, 0)`
  // (operand rows past M are clamped to 0). The select predicate's bound M is
  // this dim's global extent; decompose the true value (the clean affine
  // offset). Same correctness argument as the remsi case: TMA zero-fills OOB
  // and the epilogue store mask discards the tail.
  if (auto selOp = offsetVal.getDefiningOp<arith::SelectOp>()) {
    if (auto cmpOp = selOp.getCondition().getDefiningOp<arith::CmpIOp>()) {
      if (cmpOp.getPredicate() == arith::CmpIPredicate::slt &&
          selOp.getTrueValue() == cmpOp.getLhs()) {
        if (Value modSplat = matchSplat(cmpOp.getRhs())) {
          int modArg = getFuncArgIndex(modSplat);
          // Only a clamp to ZERO matches TMA's OOB zero-fill; a clamp to a
          // non-zero index would make the promoted load read different data
          // (zero) than the original (that index) for out-of-bounds rows.
          if (modArg >= 0 && isConstantZero(selOp.getFalseValue()) &&
              isRawIndexOffset(selOp.getTrueValue())) {
            info.shapeArgIdx = modArg;
            offsetVal = selOp.getTrueValue();
          }
        }
      }
    }
  }
  if (auto mulOp = offsetVal.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();
    for (int swap = 0; swap < 2; swap++) {
      auto addOp = lhs.getDefiningOp<arith::AddIOp>();
      Value strideSplat = matchSplat(rhs);
      if (addOp && strideSplat) {
        Value addLhs = addOp.getLhs();
        Value addRhs = addOp.getRhs();
        for (int swap2 = 0; swap2 < 2; swap2++) {
          int64_t blockSize;
          Value tileOff = matchSplat(addRhs);
          if (matchMakeRange(addLhs, blockSize) && tileOff) {
            info.blockSize = blockSize;
            info.stride = strideSplat;
            info.offset = tileOff;
            info.strideArgIdx = getFuncArgIndex(strideSplat);
            return true;
          }
          std::swap(addLhs, addRhs);
        }
      }
      std::swap(lhs, rhs);
    }
  }
  if (auto addOp = offsetVal.getDefiningOp<arith::AddIOp>()) {
    Value addLhs = addOp.getLhs();
    Value addRhs = addOp.getRhs();
    for (int swap = 0; swap < 2; swap++) {
      int64_t blockSize;
      Value tileOff = matchSplat(addRhs);
      if (matchMakeRange(addLhs, blockSize) && tileOff) {
        info.blockSize = blockSize;
        info.stride = {};
        info.offset = tileOff;
        info.strideArgIdx = -1;
        return true;
      }
      std::swap(addLhs, addRhs);
    }
  }
  // Bare make_range(0, BLOCK): a contiguous dim with no per-use tile offset.
  // Used by pointer-increment loops (`a_ptrs += BLOCK*stride`) where the tile
  // offset is carried in the advanced pointer, not recomputed. offset is left
  // null (== 0 at iteration 0); the loop-advance reconstruction fills the real
  // per-iteration offset (loopIV * perIterElems) at rewrite time.
  int64_t bsz;
  if (matchMakeRange(offsetVal, bsz)) {
    info.blockSize = bsz;
    info.stride = {};
    info.offset = {};
    info.strideArgIdx = -1;
    return true;
  }
  return false;
}

// Match a uniform integer constant tensor (tt.splat of an int constant, or a
// dense splat constant) — used for the per-iteration pointer increment.
static bool matchUniformIntConst(Value v, int64_t &out) {
  if (Value s = matchSplat(v)) {
    if (auto c = s.getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
        out = ia.getInt();
        return true;
      }
    return false;
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue()))
      if (d.isSplat()) {
        out = d.getSplatValue<APInt>().getSExtValue();
        return true;
      }
  return false;
}

struct LoopPtrInfo {
  bool isLoop = false;
  // Set when `ptr` IS an scf.for iter-arg pointer but its per-iteration advance
  // was NOT the recognized addptr(ptr, splat(C)). decomposePointer bails in
  // that case rather than decompose against the loop's INIT pointer (which
  // would fix the tile offset at iteration 0 -> wrong per-iteration index).
  bool unrecognized = false;
  Value iv;
  int64_t stepElems = 0;
};

// If `ptr` is an scf.for iter-arg pointer advanced each iteration by a uniform
// constant (`yield addptr(ptr, splat(C))`) — the classic pointer-increment K
// loop `a_ptrs += BLOCK_K*stride` — populate `info` (loop IV + per-iter element
// step) and return the loop's INIT pointer (the pointer at iteration 0) to be
// decomposed. Otherwise return `ptr` unchanged.
static Value resolveLoopCarriedPtr(Value ptr, LoopPtrInfo &info) {
  auto ba = dyn_cast<BlockArgument>(ptr);
  if (!ba)
    return ptr;
  auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp());
  if (!forOp)
    return ptr;
  unsigned argNo = ba.getArgNumber();
  if (argNo < 1) // 0 == induction variable
    return ptr;
  unsigned iterIdx = argNo - 1;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yielded = yieldOp.getOperand(iterIdx);
  auto adv = yielded.getDefiningOp<tt::AddPtrOp>();
  if (!adv || adv.getPtr() != ptr) {
    info.unrecognized = true; // loop-carried ptr, but advance not recognized
    return ptr;
  }
  int64_t stepC;
  if (!matchUniformIntConst(adv.getOffset(), stepC)) {
    info.unrecognized = true;
    return ptr;
  }
  info.isLoop = true;
  info.iv = forOp.getInductionVar();
  info.stepElems = stepC;
  return forOp.getInitArgs()[iterIdx];
}

static DecomposedLoad decomposePointer(Value ptr) {
  DecomposedLoad result;
  result.valid = false;
  auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrTy)
    return result;
  int rank = ptrTy.getRank();
  if (rank < 1 || rank > 5)
    return result;

  // Pointer-increment loops: if the load reads a loop-carried pointer advanced
  // by a uniform constant each iteration, decompose the loop's INIT pointer and
  // remember the loop IV + step to reconstruct the per-iteration tile offset.
  LoopPtrInfo loop;
  ptr = resolveLoopCarriedPtr(ptr, loop);
  // Loop-carried pointer with an unrecognized per-iteration advance: bail
  // rather than silently decompose against the INIT pointer (iteration-0 tile
  // offset).
  if (loop.unrecognized)
    return result;

  // Walk the (possibly multi-level) addptr/broadcast chain back to the base
  // pointer, collecting the offset operand at each level. Triton lowers
  // `base + d0_off + d1_off + ...` with mismatched per-dim broadcast shapes as
  // a CHAIN of addptr (with intervening broadcasts) rather than one summed
  // offset (e.g. addptr(broadcast(addptr(splat(base), m_off)), k_off)), so peel
  // each level instead of expecting a single addptr(splat(base), sum).
  SmallVector<Value> offsetVals;
  Value cur = ptr;
  for (int guard = 0; guard < 16; guard++) {
    if (auto bcast = cur.getDefiningOp<tt::BroadcastOp>()) {
      cur = bcast.getSrc();
      continue;
    }
    auto addPtrOp = cur.getDefiningOp<tt::AddPtrOp>();
    if (!addPtrOp)
      break;
    offsetVals.push_back(addPtrOp.getOffset());
    cur = addPtrOp.getPtr();
  }
  Value base = matchSplat(cur);
  if (!base)
    return result;
  // Peel scalar per-program addptr(s) folded into the base pointer before the
  // tile splat, e.g. `addptr(Qptr, off_z*stride_z)`. Collect their offsets as
  // per-program base shifts (folded into the descriptor base) and continue to
  // the underlying kernel-arg pointer.
  while (auto ap = base.getDefiningOp<tt::AddPtrOp>()) {
    result.scalarBaseOffsets.push_back(ap.getOffset());
    base = ap.getPtr();
  }
  result.basePtr = base;
  result.basePtrArgIndex = getFuncArgIndex(base);
  if (result.basePtrArgIndex < 0)
    return result;

  // Flatten any summed (addi) offsets across all chain levels into terms.
  SmallVector<Value> terms;
  std::function<void(Value)> collectTerms = [&](Value v) {
    if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
      collectTerms(addOp.getLhs());
      collectTerms(addOp.getRhs());
    } else {
      terms.push_back(v);
    }
  };
  for (Value ov : offsetVals)
    collectTerms(ov);

  if (rank == 1) {
    // Single-dim: match the whole per-level offset (make_range + splat), not
    // the addi-split pieces — match1DOffset expects the full 1d offset
    // expression. A rank-1 tile has exactly one affine offset level; if the
    // addptr chain contributed more than one level we can't be sure
    // match1DOffset captured them all (unlike the multi-dim branch below, which
    // flattens across levels), so bail rather than emit a partial tile offset.
    if (offsetVals.size() != 1)
      return result;
    Value inner = offsetVals[0];
    if (auto b = inner.getDefiningOp<tt::BroadcastOp>())
      inner = b.getSrc();
    DimInfo dim;
    if (match1DOffset(inner, dim)) {
      if (loop.isLoop && dim.strideArgIdx < 0) {
        // A pointer-increment loop's contiguous dim advances by
        // loopIV * perIterElems; the rewrite reconstructs the index from that
        // alone (init offset assumed 0, as in the canonical K-loop). If
        // match1DOffset also found a static tile offset we'd drop it, so bail.
        if (dim.offset)
          return result;
        dim.loopAdvanced = true;
        dim.loopIV = loop.iv;
        dim.perIterElems = loop.stepElems;
      }
      result.dims.push_back(dim);
      result.valid = true;
    }
    return result;
  }

  result.dims.resize(rank);
  SmallVector<bool> dimMatched(rank, false);
  for (Value term : terms) {
    Value inner = term;
    if (auto broadcastOp = inner.getDefiningOp<tt::BroadcastOp>())
      inner = broadcastOp.getSrc();
    // Uniform scalar offset (splat of a scalar, no arange/expand_dims): this is
    // a per-program base shift (e.g. `off_z * stride_z` in a batched/FA
    // kernel). Fold it into the descriptor base later instead of a per-dim
    // index.
    if (Value s = matchSplat(inner)) {
      if (s.getType().isIntOrIndex()) {
        result.scalarBaseOffsets.push_back(s);
        continue;
      }
    }
    // The canonical Triton idiom `offs[:, None] * stride` lowers to
    // muli(expand_dims(offs), splat(stride)) -- the multiply is applied AFTER
    // expand_dims, and no canonicalization hoists it through expand_dims. Peel
    // that form here and carry the stride into the DimInfo so multi-dim strided
    // loads (e.g. GEMM operands) decompose, not just the contiguous form.
    Value postExpandStride;
    if (auto mulOp = inner.getDefiningOp<arith::MulIOp>()) {
      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();
      for (int swap = 0; swap < 2; swap++) {
        if (lhs.getDefiningOp<tt::ExpandDimsOp>()) {
          if (Value strideSplat = matchSplat(rhs)) {
            inner = lhs;
            postExpandStride = strideSplat;
            break;
          }
        }
        std::swap(lhs, rhs);
      }
    }
    // NOTE: this peels a SINGLE expand_dims level, so it reliably handles rank
    // 1-2 tiles. A rank>=3 tile built with multiple expand_dims leaves src1d as
    // another ExpandDimsOp, match1DOffset fails, the dim stays unmatched, and
    // decomposePointer returns invalid (safe: load stays plain, never a wrong
    // descriptor). Broaden to a multi-level peel + targetDim mapping if rank>=3
    // auto-TMA is needed.
    if (auto expandOp = inner.getDefiningOp<tt::ExpandDimsOp>()) {
      int axis = expandOp.getAxis();
      Value src1d = expandOp.getSrc();
      int targetDim;
      if (rank == 2)
        targetDim = 1 - axis;
      else
        targetDim = (axis == 0) ? rank - 1 : axis - 1;
      if (targetDim >= 0 && targetDim < rank && !dimMatched[targetDim]) {
        DimInfo dim;
        if (match1DOffset(src1d, dim)) {
          // Stride applied post-expand (offs[:,None]*stride): the inner 1d is
          // the contiguous form, so overlay the peeled stride.
          if (postExpandStride && dim.strideArgIdx < 0 && !dim.stride) {
            dim.stride = postExpandStride;
            dim.strideArgIdx = getFuncArgIndex(postExpandStride);
          }
          result.dims[targetDim] = dim;
          dimMatched[targetDim] = true;
        }
      }
    }
  }
  for (int d = 0; d < rank; d++)
    if (!dimMatched[d])
      return result;
  // Pointer-increment loop: the contiguous dim is the one advanced each
  // iteration; its per-iteration tile offset is loopIV * stepElems. The rewrite
  // reconstructs the index from loopIV * perIterElems alone (init offset
  // assumed 0, as in the canonical K-loop). If that dim also carries a static
  // tile offset, promoting would silently drop it, so bail (leave valid =
  // false).
  if (loop.isLoop)
    for (int d = 0; d < rank; d++)
      if (result.dims[d].strideArgIdx < 0) {
        if (result.dims[d].offset)
          return result;
        result.dims[d].loopAdvanced = true;
        result.dims[d].loopIV = loop.iv;
        result.dims[d].perIterElems = loop.stepElems;
      }
  result.valid = true;
  return result;
}

// Reject only nonsensical (zero/negative) block dims. The 256-element box cap
// that cuTensorMapEncodeTiled requires is NOT enforced here anymore: the host
// recipe carries a getTMABlockShape-clamped box (computed in getAutoTmaRecipes,
// python/src/ir.cc -- the same clamp the device lowering uses), and the device
// AsyncTMACopyGlobalToLocal(/LocalToGlobal) lowering multi-copies that clamped
// box to fill the full-block SMEM tile. So an oversize tile (e.g. BLOCK=512) is
// promoted with a full-block SMEM result and a <=256 encoded box.
static bool blockDimsValid(const DecomposedLoad &decomp) {
  for (const DimInfo &dim : decomp.dims)
    if (dim.blockSize < 1)
      return false;
  return true;
}

// Shared TMA-eligibility core for a load/store tile (element type + decomposed
// pointer): TMA-encodable dtype, rank 1-5, every strided dim's stride is a
// kernel arg, exactly one contiguous dim, and the contiguous (inner) box is a
// positive multiple of 16 bytes. Block dims > 256 are allowed: the encoded box
// is getTMABlockShape-clamped and the device lowering multi-copies to fill the
// tile. Load-/store-specific checks live in the callers.
static bool isTMAEligibleCommon(Type elemTy, const DecomposedLoad &decomp) {
  if (!getTMADataType(elemTy))
    return false;
  int rank = decomp.dims.size();
  if (rank < 1 || rank > 5)
    return false;
  // A strided dim whose stride scalar is not a direct kernel argument (a
  // computed value or a loop-carried IV) has stride != null but strideArgIdx <
  // 0. The host recipe can only encode a stride via a kernel-arg index, and
  // several places use strideArgIdx < 0 as the "contiguous dim" predicate, so
  // reject.
  for (const DimInfo &dim : decomp.dims)
    if (dim.stride && dim.strideArgIdx < 0)
      return false;
  // Exactly one dim must be contiguous (stride == 1) -> TMA descriptor
  // innermost.
  int nContig = 0, contigDim = -1;
  for (int d = 0; d < rank; d++)
    if (decomp.dims[d].strideArgIdx < 0) {
      nContig++;
      contigDim = d;
    }
  if (nContig != 1)
    return false;
  // The contiguous (inner) box must be a positive multiple of 16 bytes: TMA
  // requires the inner-dimension byte size to be 16B-aligned.
  int elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  int64_t innerBytes = (int64_t)decomp.dims[contigDim].blockSize * elemBytes;
  if (innerBytes < 16 || innerBytes % 16 != 0)
    return false;
  // Reject only zero/negative block dims; the >256 box cap is handled at encode
  // time via a getTMABlockShape-clamped box + device multi-copy (see
  // blockDimsValid).
  if (!blockDimsValid(decomp))
    return false;
  return true;
}

static bool isTMAEligible(tt::LoadOp loadOp, const DecomposedLoad &decomp) {
  auto resultTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!resultTy)
    return false;
  if (!isTMAEligibleCommon(resultTy.getElementType(), decomp))
    return false;
  if (loadOp.getIsVolatile())
    return false;
  // 'other' (OOB fill) must be zero if present (TMA zero-fills).
  if (Value other = loadOp.getOther()) {
    bool isZero = false;
    if (auto splatOp = other.getDefiningOp<tt::SplatOp>()) {
      Value scalar = splatOp.getSrc();
      if (matchPattern(scalar, m_AnyZeroFloat()) ||
          matchPattern(scalar, m_Zero()))
        isZero = true;
    } else if (auto cstOp = other.getDefiningOp<arith::ConstantOp>()) {
      // A tensor `other=0` often lowers to a dense splat constant rather than
      // tt.splat(0) (e.g. multi-dim masked loads).
      if (auto dense = dyn_cast<DenseFPElementsAttr>(cstOp.getValue())) {
        if (dense.isSplat() && dense.getSplatValue<APFloat>().isZero())
          isZero = true;
      } else if (auto dense =
                     dyn_cast<DenseIntElementsAttr>(cstOp.getValue())) {
        if (dense.isSplat() && dense.getSplatValue<APInt>().isZero())
          isZero = true;
      }
    }
    if (!isZero)
      return false;
  }
  return true;
}

// Recursively check if root's defining-op chain transitively uses target.
// Memoized with a visited set: the SSA use-def graph is a DAG with heavy
// reconvergence (shared tile offsets), so without memoization a wide address
// expression could be re-walked exponentially. Visiting each node once bounds
// this to O(nodes).
static bool usesValueImpl(Value root, Value target,
                          llvm::SmallPtrSetImpl<Value> &visited, int depth) {
  if (depth > 12 || !root)
    return false;
  if (root == target)
    return true;
  if (!visited.insert(root).second)
    return false; // already explored this node
  auto *defOp = root.getDefiningOp();
  if (!defOp)
    return false;
  for (Value operand : defOp->getOperands())
    if (usesValueImpl(operand, target, visited, depth + 1))
      return true;
  return false;
}
static bool usesValue(Value root, Value target, int depth = 0) {
  llvm::SmallPtrSet<Value, 32> visited;
  return usesValueImpl(root, target, visited, depth);
}

// Find the func arg providing the global shape for a specific dimension.
// The mask is typically `and(offs_d0 < splat(M), offs_d1 < splat(N))`.
// We match each slt comparison's LHS back to the dim's tile_offset to
// associate the right bound with the right dimension.
// Find the kernel arg that supplies the global extent of `dim` -- the value the
// host CUtensorMap needs for global_dim[d]. We look for a comparison
// `offs_dim < splat(arg)` whose lhs is derived from this dim's tile offset
// (dim.offset, the pid*BLOCK tile start). A matching cmp's rhs splat arg is the
// global extent.
//
// We search the WHOLE kernel, not just this load's own mask: a typical GEMM
// masks only the K dim on its operand loads (M / N are assumed divisible, so
// the load itself carries no M / N bound), but the epilogue store DOES bound
// them (`offs_cm < M`, `offs_cn < N`). Because the tile start (pid*BLOCK) is
// one CSE'd SSA value shared by the operand-load offset and the store mask,
// usesValue ties such a store cmp to exactly this dim. If no comparison in the
// kernel bounds the dim, return -1 and the load is left unpromoted (we never
// fabricate an extent -- a wrong global_dim would mis-bound the TMA copy).
static bool cmpBoundsDim(arith::CmpIOp cmpOp, const DimInfo &dim, int &argIdx) {
  if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  if (!dim.offset || !usesValue(cmpOp.getLhs(), dim.offset))
    return false;
  Value bound = matchSplat(cmpOp.getRhs());
  if (!bound)
    return false;
  int idx = getFuncArgIndex(bound);
  if (idx < 0)
    return false;
  argIdx = idx;
  return true;
}

// A store mask may be reduced to the descriptor's globalDim bounds ONLY if it
// is exactly a rectangular per-dim boundary AND_d(offs_d < E_d). Any extra
// predicate (causal / data-dependent) would make the promoted TMA store write
// the whole in-bounds box, clobbering elements the original masked off -- a
// miscompile (unlike loads, where an extra in-bounds read is harmless). Return
// false if any AND-term of the mask is not a recognized per-dim boundary of
// some decomp dim.
static bool storeMaskIsRectangular(Value mask, const DecomposedLoad &decomp) {
  if (!mask)
    return true; // no mask -> nothing masked off beyond globalDim
  SmallVector<Value> terms;
  std::function<void(Value)> collect = [&](Value v) {
    if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
      collect(andOp.getLhs());
      collect(andOp.getRhs());
    } else {
      terms.push_back(v);
    }
  };
  collect(mask);
  for (Value t : terms) {
    Value c = t;
    if (auto bcast = c.getDefiningOp<tt::BroadcastOp>())
      c = bcast.getSrc();
    auto cmpOp = c.getDefiningOp<arith::CmpIOp>();
    int argIdx = -1;
    bool matched = false;
    for (const DimInfo &dim : decomp.dims)
      if (cmpBoundsDim(cmpOp, dim, argIdx)) {
        matched = true;
        break;
      }
    if (!matched)
      return false;
  }
  return true;
}

static int findShapeArgForDim(tt::FuncOp func, Value ownMask,
                              const DimInfo &dim) {
  if (!dim.offset)
    return -1;
  // 1) Prefer the op's own mask (most specific bound for the dim).
  if (Value mask = ownMask) {
    SmallVector<Value> cmpTerms;
    std::function<void(Value)> collectCmps = [&](Value v) {
      if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
        collectCmps(andOp.getLhs());
        collectCmps(andOp.getRhs());
      } else {
        cmpTerms.push_back(v);
      }
    };
    collectCmps(mask);
    for (Value cmpVal : cmpTerms) {
      // 2D/ND masks broadcast each per-dim comparison (offs[:,None] < M) up to
      // the load shape, so peel the broadcast to reach the cmpi.
      Value c = cmpVal;
      if (auto bcast = c.getDefiningOp<tt::BroadcastOp>())
        c = bcast.getSrc();
      int argIdx = -1;
      if (cmpBoundsDim(c.getDefiningOp<arith::CmpIOp>(), dim, argIdx))
        return argIdx;
    }
  }
  // 2) Fall back to any slt comparison in the kernel that bounds this dim
  //    (e.g. the epilogue store's mask supplies M / N for K-only-masked loads).
  int found = -1;
  func.walk([&](arith::CmpIOp cmpOp) {
    int argIdx = -1;
    if (cmpBoundsDim(cmpOp, dim, argIdx)) {
      found = argIdx;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Device mode hoists the make_tensor_descriptor above the load/store's
// enclosing scf.for so the lowered tensormap_create can be predicated. Any
// per-program scalar base offset folded into the descriptor base
// (scalarBaseOffsets) is an operand of that hoisted op, so it must be defined
// ABOVE the loop or the hoist would violate SSA dominance. Return false if any
// offset is defined inside the enclosing loop (then the caller skips
// promotion). No enclosing loop -> nothing to hoist past -> hoistable.
static bool scalarBaseOffsetsHoistable(Operation *op,
                                       const DecomposedLoad &decomp) {
  if (decomp.scalarBaseOffsets.empty())
    return true;
  auto forOp = op->getParentOfType<scf::ForOp>();
  if (!forOp)
    return true;
  Region &loopRegion = forOp.getRegion();
  for (Value off : decomp.scalarBaseOffsets)
    if (loopRegion.isAncestor(off.getParentRegion()))
      return false;
  return true;
}

// Whether `mask` carries an upper-bound comparison that is NOT attributable to
// any *tiled* dim (a dim with a tile offset). Such an "unattributed" bound may
// constrain an untiled/whole dim, whose true global extent could then exceed
// blockSize -- in which case the -2 constant-extent sentinel would encode too
// small an extent. Used to gate the whole-dim sentinel to genuinely unmasked
// dims. Conservative: an unrecognized comparison counts as unattributed, so the
// guard only ever declines to fabricate an extent (never emits a wrong one).
static bool maskHasUnattributedBound(Value mask, const DecomposedLoad &decomp) {
  if (!mask)
    return false;
  SmallVector<Value> cmpTerms;
  std::function<void(Value)> collectCmps = [&](Value v) {
    if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
      collectCmps(andOp.getLhs());
      collectCmps(andOp.getRhs());
    } else {
      cmpTerms.push_back(v);
    }
  };
  collectCmps(mask);
  for (Value cmpVal : cmpTerms) {
    Value c = cmpVal;
    if (auto bcast = c.getDefiningOp<tt::BroadcastOp>())
      c = bcast.getSrc();
    auto cmpOp = c.getDefiningOp<arith::CmpIOp>();
    if (!cmpOp)
      continue; // not a comparison we treat as a bound
    bool attributed = false;
    for (const DimInfo &dim : decomp.dims) {
      int argIdx = -1;
      if (dim.offset && cmpBoundsDim(cmpOp, dim, argIdx)) {
        attributed = true;
        break;
      }
    }
    if (!attributed)
      return true;
  }
  return false;
}

// Structural equality of scalar index expressions. The promote pass runs before
// CSE, so the tile offset used in a pointer (`muli(ki, BLOCK)`) and the same
// quantity recomputed in a mask bound (`K - muli(ki, BLOCK)`) are equal in
// value but distinct SSA. Compare by structure: same SSA leaf, equal constants,
// or matching add/mul (commutative) / sub (ordered) trees.
static bool sameScalarExpr(Value a, Value b, int depth = 0) {
  if (a == b)
    return true;
  // Depth cap keeps the commutative add/mul 4-way branching bounded
  // (tile-offset exprs are shallow, e.g. K - ki*BLOCK_K); a miss only costs a
  // promotion.
  if (depth > 6 || !a || !b)
    return false;
  Operation *oa = a.getDefiningOp();
  Operation *ob = b.getDefiningOp();
  if (!oa || !ob || oa->getName() != ob->getName())
    return false;
  if (auto ca = dyn_cast<arith::ConstantOp>(oa)) {
    auto cb = cast<arith::ConstantOp>(ob);
    return ca.getValue() == cb.getValue();
  }
  if (isa<arith::MulIOp, arith::AddIOp>(oa)) {
    Value a0 = oa->getOperand(0), a1 = oa->getOperand(1);
    Value b0 = ob->getOperand(0), b1 = ob->getOperand(1);
    return (sameScalarExpr(a0, b0, depth + 1) &&
            sameScalarExpr(a1, b1, depth + 1)) ||
           (sameScalarExpr(a0, b1, depth + 1) &&
            sameScalarExpr(a1, b0, depth + 1));
  }
  if (isa<arith::SubIOp>(oa))
    return sameScalarExpr(oa->getOperand(0), ob->getOperand(0), depth + 1) &&
           sameScalarExpr(oa->getOperand(1), ob->getOperand(1), depth + 1);
  return false;
}

// Recover a dim's global extent from a per-tile "remaining" bound of the form
//   localArange < E - tileOffset        (e.g. the reduction mask `arange < K -
//   ki*BLOCK_K`)
// Tiled loops mask the reduction dim against how much is LEFT in the current
// tile, so the bound is `subi(E, tileOffset)` rather than the direct extent E.
// We tie it to this dim by requiring the subtrahend to be exactly dim.offset
// (the tile start) and the lhs to be this dim's local make_range(0, blockSize).
// The minuend E is then the global-extent kernel arg.
static int findShapeArgFromTileBound(tt::FuncOp func, const DimInfo &dim) {
  if (!dim.offset)
    return -1;
  int found = -1;
  func.walk([&](arith::CmpIOp cmpOp) {
    if (found >= 0)
      return WalkResult::interrupt();
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
      return WalkResult::advance();
    // lhs: this dim's local range, peeled of expand_dims / broadcast.
    Value lhs = cmpOp.getLhs();
    if (auto b = lhs.getDefiningOp<tt::BroadcastOp>())
      lhs = b.getSrc();
    if (auto e = lhs.getDefiningOp<tt::ExpandDimsOp>())
      lhs = e.getSrc();
    int64_t bs = 0;
    if (!matchMakeRange(lhs, bs) || bs != dim.blockSize)
      return WalkResult::advance();
    // rhs: splat(subi(E, tileOffset)) with tileOffset structurally ==
    // dim.offset.
    Value rhs = matchSplat(cmpOp.getRhs());
    if (!rhs)
      return WalkResult::advance();
    auto subOp = rhs.getDefiningOp<arith::SubIOp>();
    if (!subOp || !sameScalarExpr(subOp.getRhs(), dim.offset))
      return WalkResult::advance();
    int argIdx = getFuncArgIndex(subOp.getLhs());
    if (argIdx < 0)
      return WalkResult::advance();
    found = argIdx;
    return WalkResult::interrupt();
  });
  return found;
}

static bool matchScalarIntConst(Value v, int64_t &out) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      out = ia.getInt();
      return true;
    }
  return false;
}

// Extent recovery for a loop-advanced (pointer-increment) dim. Its per-tile
// reduction bound is `localArange < E - (loopIV * perIterElems)`; match
// subi(E, muli(loopIV, perIterElems)) and return the extent arg E. The loop IV
// is matched by SSA identity (it is the actual induction var); the per-iter
// step by constant value.
static int findShapeArgForLoopDim(tt::FuncOp func, const DimInfo &dim) {
  if (!dim.loopAdvanced || !dim.loopIV)
    return -1;
  int found = -1;
  func.walk([&](arith::CmpIOp cmpOp) {
    if (found >= 0)
      return WalkResult::interrupt();
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
      return WalkResult::advance();
    // lhs: this dim's local range, peeled of expand_dims / broadcast. Required
    // (mirrors findShapeArgFromTileBound) so an unrelated slt whose RHS happens
    // to match subi(E, loopIV*step) can't select the wrong extent arg.
    Value lhs = cmpOp.getLhs();
    if (auto b = lhs.getDefiningOp<tt::BroadcastOp>())
      lhs = b.getSrc();
    if (auto e = lhs.getDefiningOp<tt::ExpandDimsOp>())
      lhs = e.getSrc();
    int64_t bs = 0;
    if (!matchMakeRange(lhs, bs) || bs != dim.blockSize)
      return WalkResult::advance();
    Value rhs = matchSplat(cmpOp.getRhs());
    if (!rhs)
      return WalkResult::advance();
    auto subOp = rhs.getDefiningOp<arith::SubIOp>();
    if (!subOp)
      return WalkResult::advance();
    auto mul = subOp.getRhs().getDefiningOp<arith::MulIOp>();
    if (!mul)
      return WalkResult::advance();
    auto isPerIter = [&](Value iv, Value c) {
      int64_t cv;
      return iv == dim.loopIV && matchScalarIntConst(c, cv) &&
             cv == dim.perIterElems;
    };
    if (!isPerIter(mul.getLhs(), mul.getRhs()) &&
        !isPerIter(mul.getRhs(), mul.getLhs()))
      return WalkResult::advance();
    int argIdx = getFuncArgIndex(subOp.getLhs());
    if (argIdx < 0)
      return WalkResult::advance();
    found = argIdx;
    return WalkResult::interrupt();
  });
  return found;
}

// A kernel arg is provably 16B-aligned if its ``tt.divisibility`` (in elements,
// from specialization) times the element size is a multiple of 16. Absent attr
// => alignment unknown => treated as not aligned (caller rejects).
static bool argIs16BAligned(tt::FuncOp func, int argIdx, int elemBytes) {
  auto attr = func.getArgAttrOfType<IntegerAttr>(argIdx, "tt.divisibility");
  if (!attr)
    return false;
  return (attr.getInt() * (int64_t)elemBytes) % 16 == 0;
}

// TMA requires the global (outer) strides to be 16-byte aligned. A strided
// dim's stride comes from a kernel arg whose element-divisibility is known from
// specialization (tt.divisibility); the byte stride is divisible by
// divisibility * elem_size. Contiguous innermost dims (strideArgIdx < 0) carry
// no outer stride and are always fine.
static bool dimStrideIs16BAligned(tt::FuncOp func, const DimInfo &dim,
                                  int elemBytes) {
  if (dim.strideArgIdx < 0)
    return true;
  return argIs16BAligned(func, dim.strideArgIdx, elemBytes);
}

// TMA-eligibility for a store's value tensor — mirrors isTMAEligible minus the
// load-only `other`/OOB-fill check (TMA store bounds via the descriptor's
// globalDim, so a masked store maps to a bounded TMA copy).
static bool isTMAEligibleStore(tt::StoreOp storeOp,
                               const DecomposedLoad &decomp) {
  auto valTy = dyn_cast<RankedTensorType>(storeOp.getValue().getType());
  if (!valTy)
    return false;
  // Shared eligibility core (dtype, rank, single contiguous dim, inner box a
  // 16B multiple, every box dim <= 256). The store drops only the load's OOB
  // `other`-fill check; the caller additionally requires the store mask to be a
  // pure rectangular boundary (storeMaskIsRectangular) before reducing it to
  // the descriptor's globalDim bounds, and verifies base-ptr / stride 16B
  // alignment.
  return isTMAEligibleCommon(valTy.getElementType(), decomp);
}

// Store promotion is WS-gated: the TMA store stages the value reg->smem then
// bulk-copies smem->global, which only pays off when overlapped with compute in
// a separate epilogue partition under warp specialization (standalone it just
// adds a round-trip).
//
// IR shape assumed: at the point this pass runs (TTIR, before WS lowering) a
// warp-specialized loop is an `scf.for` carrying the `tt.warp_specialize` unit
// attribute, set by `tl.range(warp_specialize=True)`. This is deliberately a
// narrow structural check -- if a future pipeline represents WS differently (a
// different op, or the attr on a non-ForOp construct) this returns false and
// store promotion is simply skipped (a missed optimization, never a
// miscompile). Broaden the walk here if/when such a representation is added.
static bool kernelIsWarpSpecialized(tt::FuncOp func) {
  bool ws = false;
  func.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("tt.warp_specialize")) {
      ws = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ws;
}

struct Promotion {
  tt::LoadOp loadOp;
  DecomposedLoad decomp;
  SmallVector<int> shapeArgIndices;
  // Logical-tile -> descriptor-memory dim order (the single contiguous dim is
  // moved last). Identity when the contiguous dim is already innermost.
  SmallVector<int> perm;
  // True when this load requires a device-built descriptor (in-kernel
  // tt.make_tensor_descriptor) rather than the host recipe path.
  bool useDeviceMode = false;
};

struct StorePromotion {
  tt::StoreOp storeOp;
  DecomposedLoad decomp;
  SmallVector<int> shapeArgIndices;
  bool useDeviceMode = false;
};

class TritonNvidiaGPUPromoteLoadToTMAPass
    : public impl::TritonNvidiaGPUPromoteLoadToTMAPassBase<
          TritonNvidiaGPUPromoteLoadToTMAPass> {
public:
  using impl::TritonNvidiaGPUPromoteLoadToTMAPassBase<
      TritonNvidiaGPUPromoteLoadToTMAPass>::
      TritonNvidiaGPUPromoteLoadToTMAPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    tt::FuncOp kernelFunc;
    int numKernels = 0;
    mod.walk([&](tt::FuncOp func) {
      if (tt::isKernel(func)) {
        ++numKernels;
        if (!kernelFunc)
          kernelFunc = func;
      }
      return WalkResult::skip();
    });
    if (!kernelFunc)
      return;
    // Recipes are recorded as a single module-wide ``ttg.auto_tma_recipes``
    // attribute whose ``desc_arg_index`` / ``*_arg_indices`` are relative to
    // one kernel's argument list, so this pass assumes exactly one kernel per
    // module (the invariant for the Triton JIT compile pipeline). Fail loudly
    // on a future multi-kernel module -- via a diagnostic that survives release
    // (NDEBUG) builds, unlike assert -- instead of silently promoting only the
    // first kernel and dropping the rest.
    if (numKernels != 1) {
      mod.emitError("PromoteLoadToTMA: expected exactly one kernel per module, "
                    "got ")
          << numKernels;
      return signalPassFailure();
    }

    // Device-descriptor mode: build tt.make_tensor_descriptor in-kernel instead
    // of a host recipe (handles per-program base + constant/whole-dim extents).
    // Controlled by the `device-mode` pass option
    // (knobs.nvidia.auto_tma_device); TRITON_AUTO_TMA_DEVICE stays as an env
    // fallback for ad-hoc runs.
    const char *devEnv = std::getenv("TRITON_AUTO_TMA_DEVICE");
    bool useDevice = deviceMode || (devEnv && std::string(devEnv) == "1");

    // Phase 1: collect eligible loads.
    SmallVector<Promotion, 0> promotions;
    kernelFunc.walk([&](tt::LoadOp loadOp) {
      DecomposedLoad decomp = decomposePointer(loadOp.getPtr());
      if (!decomp.valid || !isTMAEligible(loadOp, decomp))
        return;
      // Per-load host-vs-device decision: start with the global override, then
      // auto-escalate to device mode for loads that host recipe can't handle.
      bool needsDevice = useDevice;
      // scalarBaseOffsets (per-program base shifts like off_z*stride_z) require
      // device mode — the host recipe can only record base_ptr_arg_index.
      if (!decomp.scalarBaseOffsets.empty())
        needsDevice = true;
      // Device mode hoists the descriptor above enclosing loops; skip if a
      // scalar base offset is loop-defined (SSA dominance).
      if (needsDevice && !scalarBaseOffsetsHoistable(loadOp, decomp))
        return;
      int rank = decomp.dims.size();
      SmallVector<int> shapeArgs(rank, -1);
      for (int d = 0; d < rank; d++) {
        const DimInfo &dim = decomp.dims[d];
        // Extent sources, in priority order: (1) a modulus/select clamp on the
        // offset (dim.shapeArgIdx), (2) a direct bound `offs < E` anywhere in
        // the kernel, (3) a per-tile remaining bound `arange < E - tileOffset`.
        int s = dim.shapeArgIdx;
        if (s < 0)
          s = findShapeArgForDim(kernelFunc, loadOp.getMask(), dim);
        if (s < 0)
          s = findShapeArgFromTileBound(kernelFunc, dim);
        if (s < 0 && dim.loopAdvanced)
          s = findShapeArgForLoopDim(kernelFunc, dim);
        // A dim loaded WHOLE (no tile offset -> not tiled, e.g. the FA head dim
        // `offs_d = arange(0, HEAD_DIM)`) has global extent == blockSize, a
        // compile-time constant. Use the -2 sentinel so Phase 2 emits an
        // arith.constant extent. Host recipe can't encode a constant extent, so
        // this auto-escalates to device mode. Require the dim be genuinely
        // unmasked (no mask bound unattributable to a tiled dim): a
        // partially-masked untiled dim could have a true extent > blockSize, so
        // fabricating blockSize would under-size the descriptor.
        if (s < 0 && !dim.offset &&
            !maskHasUnattributedBound(loadOp.getMask(), decomp)) {
          s = -2;
          needsDevice = true;
        }
        shapeArgs[d] = s;
      }
      bool ok = true;
      for (int d = 0; d < rank; d++)
        if (shapeArgs[d] == -1) // -2 = constant extent (device mode), allowed
          ok = false;
      if (!ok)
        return;
      // TMA needs 16B-aligned outer strides; skip loads whose strides aren't
      // provably aligned (e.g. a runtime stride with no divisibility
      // guarantee).
      int elemBytes = cast<RankedTensorType>(loadOp.getResult().getType())
                          .getElementType()
                          .getIntOrFloatBitWidth() /
                      8;
      for (int d = 0; d < rank; d++)
        if (!dimStrideIs16BAligned(kernelFunc, decomp.dims[d], elemBytes))
          return;
      // TMA also requires the CUtensorMap global base address to be
      // 16B-aligned. The base pointer's byte alignment is divisibility *
      // elemBytes; reject if not provably 16B (mirrors the outer-stride check).
      // Absent attr => unknown
      // => reject, so a base marked divisible-by-1 (e.g. a sliced/offset
      // tensor) isn't promoted into an illegal descriptor.
      if (!argIs16BAligned(kernelFunc, decomp.basePtrArgIndex, elemBytes))
        return;
      Promotion p;
      p.loadOp = loadOp;
      p.decomp = decomp;
      p.shapeArgIndices = shapeArgs;
      p.useDeviceMode = needsDevice;
      // Descriptor memory order: keep dims in order but move the single
      // contiguous dim last (TMA descriptor innermost must be contiguous).
      int contigDim = -1;
      for (int d = 0; d < rank; d++)
        if (decomp.dims[d].strideArgIdx < 0)
          contigDim = d;
      for (int d = 0; d < rank; d++)
        if (d != contigDim)
          p.perm.push_back(d);
      p.perm.push_back(contigDim);
      promotions.push_back(std::move(p));
    });

    // Phase 1b: collect eligible stores, WS-gated (TMA store only pays off
    // under warp specialization). Only promote stores whose contiguous dim is
    // already innermost (the standard row-major C[M,N] output); a transposed
    // store would need the value transposed before descriptor_store -- skipped
    // for now.
    SmallVector<StorePromotion, 0> storePromotions;
    if (kernelIsWarpSpecialized(kernelFunc)) {
      kernelFunc.walk([&](tt::StoreOp storeOp) {
        DecomposedLoad decomp = decomposePointer(storeOp.getPtr());
        if (!decomp.valid || !isTMAEligibleStore(storeOp, decomp))
          return;
        // Per-store host-vs-device decision (same logic as loads).
        bool needsDevice = useDevice;
        if (!decomp.scalarBaseOffsets.empty())
          needsDevice = true;
        if (needsDevice && !scalarBaseOffsetsHoistable(storeOp, decomp))
          return;
        int rank = decomp.dims.size();
        // contiguous dim must already be innermost (identity perm).
        if (decomp.dims[rank - 1].strideArgIdx >= 0)
          return;
        // Reject any loop-advanced (pointer-increment) dim. Phase 2b builds the
        // store index directly from dim.offset and -- unlike the load rewrite
        // -- does NOT reconstruct loopIV * perIterElems, so a loop-advanced dim
        // would emit a wrong per-iteration index. This is also why the Phase 1b
        // extent search below omits the findShapeArgForLoopDim fallback that
        // Phase 1 uses: a loop-advanced dim never survives to need it. In
        // practice the output store is the row-major C tile, not a
        // pointer-increment operand, so this excludes nothing real.
        bool loopAdvanced = false;
        for (int d = 0; d < rank; d++)
          if (decomp.dims[d].loopAdvanced)
            loopAdvanced = true;
        if (loopAdvanced)
          return;
        // static_cast<unsigned>: dodge GCC 13 -Wstringop-overflow false
        // positive.
        SmallVector<int> shapeArgs(static_cast<unsigned>(rank), -1);
        for (int d = 0; d < rank; d++) {
          const DimInfo &dim = decomp.dims[d];
          int s = dim.shapeArgIdx;
          if (s < 0)
            s = findShapeArgForDim(kernelFunc, storeOp.getMask(), dim);
          if (s < 0)
            s = findShapeArgFromTileBound(kernelFunc, dim);
          if (s < 0 && !dim.offset &&
              !maskHasUnattributedBound(storeOp.getMask(), decomp)) {
            s = -2;
            needsDevice = true;
          }
          shapeArgs[d] = s;
        }
        bool ok = true;
        for (int d = 0; d < rank; d++)
          if (shapeArgs[d] == -1) // -2 = constant extent (device mode)
            ok = false;
        if (!ok)
          return;
        int elemBytes = cast<RankedTensorType>(storeOp.getValue().getType())
                            .getElementType()
                            .getIntOrFloatBitWidth() /
                        8;
        for (int d = 0; d < rank; d++)
          if (!dimStrideIs16BAligned(kernelFunc, decomp.dims[d], elemBytes))
            return;
        // TMA requires a 16B-aligned global base address (mirrors the load path
        // and the outer-stride check above).
        if (!argIs16BAligned(kernelFunc, decomp.basePtrArgIndex, elemBytes))
          return;
        // Only a provably rectangular store mask (AND_d(offs_d < E_d)) may be
        // reduced to the descriptor's globalDim bounds. An extra predicate
        // would make the TMA store write elements the original masked off ->
        // clobber.
        if (!storeMaskIsRectangular(storeOp.getMask(), decomp))
          return;
        StorePromotion p;
        p.storeOp = storeOp;
        p.decomp = decomp;
        p.shapeArgIndices = shapeArgs;
        p.useDeviceMode = needsDevice;
        storePromotions.push_back(std::move(p));
      });
    }

    if (promotions.empty() && storePromotions.empty())
      return;

    // Phase 2: rewrite each eligible load into a HOST-side TMA descriptor
    // passed as a compiler-synthesized kernel argument + descriptor_load, and
    // record a recipe so the launcher builds the CUtensorMap on the host (no
    // device global scratch). This routes auto-TMA through the shared launch.h
    // recipe core.
    //
    // The synthesized descriptor arg is tagged `tt.auto_tma` so the metadata
    // layer (get_tensordesc_metadata) skips it (it has no user-provided Python
    // arg); get_auto_tma_recipes() surfaces it to the launcher instead.
    OpBuilder builder(mod.getContext());
    MLIRContext *ctx = mod.getContext();
    Builder b(ctx);

    // Shared builder for the device (in-kernel) TMA descriptor, used by both
    // the load and store rewrites in device mode. Emits make_tensor_descriptor
    // in the given dim order (`perm`; identity for stores), hoisted via LICM
    // to the outermost scf.for where all descriptor operands are still
    // loop-invariant, so the lowered tensormap_create is rebuilt only when its
    // inputs actually change. Base ptr, shape, and stride args are kernel
    // function arguments (always invariant); only scalarBaseOffsets can be
    // loop-defined and thus limit the hoist distance. Phase 1
    // (scalarBaseOffsetsHoistable) has already ensured at least one level of
    // hoist is legal when an enclosing loop exists.
    auto buildDeviceDescriptor =
        [&](Operation *anchorOp, const DecomposedLoad &decomp,
            ArrayRef<int> shapeArgIndices, ArrayRef<int> perm, Location loc,
            Value &deviceZeroIdxOut) -> Value {
      OpBuilder::InsertionGuard guard(builder);
      // LICM: walk up nested scf::ForOps while all scalar base offsets are
      // defined outside the loop (base ptr and shape/stride args are kernel
      // function arguments, so they are invariant w.r.t. every loop).
      Operation *hoistPt = anchorOp;
      for (auto forOp = anchorOp->getParentOfType<scf::ForOp>(); forOp;
           forOp = forOp->getParentOfType<scf::ForOp>()) {
        Region &loopBody = forOp.getRegion();
        bool allInvariant = true;
        for (Value off : decomp.scalarBaseOffsets) {
          if (loopBody.isAncestor(off.getParentRegion())) {
            allInvariant = false;
            break;
          }
        }
        if (!allInvariant)
          break;
        hoistPt = forOp;
      }
      builder.setInsertionPoint(hoistPt);
      Type i32 = builder.getI32Type();
      deviceZeroIdxOut =
          arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
      SmallVector<Value> descShape, descStrides;
      SmallVector<int32_t> descBlock;
      for (int i = 0; i < (int)perm.size(); i++) {
        int d = perm[i];
        // Shape extent: kernel arg, or a compile-time constant (blockSize) for
        // a whole-dim (untiled) dim recovered as the -2 sentinel.
        Value shapeArg;
        if (shapeArgIndices[d] == -2) {
          shapeArg = arith::ConstantOp::create(
              builder, loc,
              builder.getI32IntegerAttr((int)decomp.dims[d].blockSize));
        } else {
          shapeArg = kernelFunc.getArgument(shapeArgIndices[d]);
          if (shapeArg.getType().isInteger(64))
            shapeArg = arith::TruncIOp::create(builder, loc, i32, shapeArg);
        }
        descShape.push_back(shapeArg);
        // Contiguous dim (strideArgIdx < 0) has unit stride;
        // make_tensor_descriptor needs an i64 stride operand per dim.
        int strideIdx = decomp.dims[d].strideArgIdx;
        Value strideV;
        if (strideIdx < 0) {
          strideV = arith::ConstantOp::create(builder, loc,
                                              builder.getI64IntegerAttr(1));
        } else {
          strideV = kernelFunc.getArgument(strideIdx);
          if (strideV.getType().isInteger(32))
            strideV = arith::ExtSIOp::create(builder, loc, builder.getI64Type(),
                                             strideV);
        }
        descStrides.push_back(strideV);
        descBlock.push_back((int32_t)decomp.dims[d].blockSize);
      }
      // Fold uniform per-program scalar offsets (off_z*stride_z, ...) into the
      // descriptor base -> a per-program (e.g. per-(batch,head)) view.
      Value descBase = decomp.basePtr;
      for (Value off : decomp.scalarBaseOffsets)
        descBase = tt::AddPtrOp::create(builder, loc, descBase.getType(),
                                        descBase, off);
      return tt::MakeTensorDescOp::create(
          builder, loc, descBase, descShape, descStrides, descBlock,
          /*isSignedInteger=*/false, tt::PaddingOption::PAD_ZERO);
    };
    auto i32Attr = [&](int v) { return b.getI32IntegerAttr(v); };
    SmallVector<Attribute> recipeAttrs;

    for (auto &promo : promotions) {
      tt::LoadOp loadOp = promo.loadOp;
      int rank = promo.decomp.dims.size();
      auto resultTy = cast<RankedTensorType>(loadOp.getResult().getType());
      Type elemTy = resultTy.getElementType();
      Location loc = loadOp.getLoc();

      // Build the descriptor in memory order (the single contiguous dim moved
      // last, per promo.perm). For an operand whose contiguous dim is already
      // innermost (e.g. A in C=A@B), perm is identity and this is the tile as
      // loaded. For a transposed operand (e.g. B[K,N]: K contiguous but N is
      // the tile innermost), the descriptor is built over the contiguous layout
      // and the loaded tile is transposed back -- mirroring explicit-TMA
      // `d.load().T`.
      bool isIdentity = true;
      for (int i = 0; i < rank; i++)
        if (promo.perm[i] != i)
          isIdentity = false;

      SmallVector<int64_t> permShape;
      for (int i = 0; i < rank; i++)
        permShape.push_back(resultTy.getShape()[promo.perm[i]]);
      // For the identity layout (contiguous dim already innermost) the load
      // result's encoding maps 1:1 onto the descriptor tile. For a permuted
      // (transposed-operand) layout the dims are reordered, so reusing the
      // load-result encoding -- whose per-dim order / sizePerThread follow the
      // LOGICAL shape -- would describe the permuted shape with the wrong dim
      // order. Leave the descriptor block encoding unset in that case and let
      // the downstream descriptor-encoding pass assign one matching the
      // memory-order tile.
      Attribute descEncoding =
          isIdentity ? resultTy.getEncoding() : Attribute();
      auto descTileTy = RankedTensorType::get(permShape, elemTy, descEncoding);

      builder.setInsertionPoint(loadOp);
      Type i32 = builder.getI32Type();

      // The descriptor fed to descriptor_load: device-built (deviceMode) or a
      // host-built descriptor passed as a synthesized kernel arg (recipe path).
      Value descVal;
      unsigned descArgIdx = 0;
      // Loop-invariant zero index for whole-dim (untiled) dims, created OUTSIDE
      // any loop so it needs no ttg.partition attr under warp specialization.
      Value deviceZeroIdx;
      if (promo.useDeviceMode) {
        descVal =
            buildDeviceDescriptor(loadOp, promo.decomp, promo.shapeArgIndices,
                                  promo.perm, loc, deviceZeroIdx);
      } else {
        // Synthesized host-side descriptor; block type is the memory-order
        // tile.
        auto descTy = tt::TensorDescType::get(descTileTy.getShape(),
                                              descTileTy.getElementType(),
                                              descTileTy.getEncoding());
        descArgIdx = kernelFunc.getNumArguments();
        auto argAttrs = b.getDictionaryAttr(
            {b.getNamedAttr("tt.auto_tma", b.getUnitAttr())});
        LogicalResult inserted =
            kernelFunc.insertArgument(descArgIdx, descTy, argAttrs, loc);
        assert(succeeded(inserted) &&
               "failed to insert TMA descriptor kernel argument");
        (void)inserted;
        descVal = kernelFunc.getArgument(descArgIdx);
      }
      Type i64 = builder.getI64Type();
      // Convert an integer/index value to a target integer width,
      // sign-extending or truncating as needed. descriptor_load indices must be
      // i32, but an offset may be i16 / i32 / i64 / index depending on how the
      // kernel computed it. Unexpected non-integer types are left unchanged and
      // get rejected by the op verifier rather than silently mis-typed.
      auto toIntWidth = [&](Value v, Type target) -> Value {
        Type ty = v.getType();
        if (ty == target)
          return v;
        if (ty.isIndex())
          return arith::IndexCastOp::create(builder, loc, target, v);
        auto srcInt = dyn_cast<IntegerType>(ty);
        auto dstInt = dyn_cast<IntegerType>(target);
        if (srcInt && dstInt) {
          if (srcInt.getWidth() > dstInt.getWidth())
            return arith::TruncIOp::create(builder, loc, target, v);
          return arith::ExtSIOp::create(builder, loc, target, v);
        }
        return v;
      };
      auto toI32 = [&](Value v) { return toIntWidth(v, i32); };

      SmallVector<Value> indices;
      for (int i = 0; i < rank; i++) {
        const DimInfo &dim = promo.decomp.dims[promo.perm[i]];
        if (dim.loopAdvanced) {
          // Reconstruct the per-iteration tile offset: loopIV * perIterElems.
          // Compute in i64 (perIterElems is i64, and loopIV * step can exceed
          // i32 for large trip counts / strides), narrowing to i32 only at the
          // descriptor_load index boundary.
          Value iv = toIntWidth(dim.loopIV, i64);
          Value step = arith::ConstantOp::create(
              builder, loc, builder.getI64IntegerAttr(dim.perIterElems));
          Value prod = arith::MulIOp::create(builder, loc, iv, step);
          indices.push_back(arith::TruncIOp::create(builder, loc, i32, prod));
        } else if (!dim.offset) {
          // Whole-dim (untiled, e.g. FA HEAD_DIM): index is 0. Use the hoisted
          // zero (deviceMode) so it isn't a constant inside the WS loop.
          indices.push_back(
              deviceZeroIdx ? deviceZeroIdx
                            : arith::ConstantOp::create(
                                  builder, loc, builder.getI32IntegerAttr(0)));
        } else {
          indices.push_back(toI32(dim.offset));
        }
      }

      auto newLoad = tt::DescriptorLoadOp::create(
          builder, loc, descTileTy, descVal, indices, tt::CacheModifier::NONE,
          tt::EvictionPolicy::NORMAL);

      Value loaded = newLoad.getResult();
      if (!isIdentity) {
        // Transpose the memory-order tile back to the kernel's logical tile.
        // order[i] = position of logical dim i within the memory-order tile.
        SmallVector<int32_t> order(rank);
        for (int i = 0; i < rank; i++)
          order[promo.perm[i]] = i;
        loaded = tt::TransOp::create(builder, loc, loaded, order);
      }

      loadOp.getResult().replaceAllUsesWith(loaded);
      loadOp.erase();

      // Device descriptors need no launcher recipe.
      if (promo.useDeviceMode)
        continue;

      // Record the recipe: how the launcher builds this CUtensorMap from the
      // existing scalar kernel args (base ptr / shape / stride) at launch time.
      // Emit per-dim fields in descriptor memory order (promo.perm) so they
      // match the descriptor arg's block type (contiguous dim last).
      SmallVector<Attribute> shapeIdx, strideIdx, blk;
      for (int i = 0; i < rank; i++) {
        int d = promo.perm[i];
        shapeIdx.push_back(i32Attr(promo.shapeArgIndices[d]));
        strideIdx.push_back(i32Attr(promo.decomp.dims[d].strideArgIdx));
        blk.push_back(i32Attr((int)promo.decomp.dims[d].blockSize));
      }
      int tmaDtype = getTMADataType(elemTy).value();
      int elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      SmallVector<NamedAttribute> fields = {
          b.getNamedAttr("desc_arg_index", i32Attr((int)descArgIdx)),
          b.getNamedAttr("base_ptr_arg_index",
                         i32Attr(promo.decomp.basePtrArgIndex)),
          b.getNamedAttr("shape_arg_indices", b.getArrayAttr(shapeIdx)),
          b.getNamedAttr("stride_arg_indices", b.getArrayAttr(strideIdx)),
          b.getNamedAttr("block_shape", b.getArrayAttr(blk)),
          b.getNamedAttr("elem_type", i32Attr(tmaDtype)),
          b.getNamedAttr("elem_size", i32Attr(elemBytes)),
          b.getNamedAttr("fp4_padded", i32Attr(0)),
          b.getNamedAttr("fill_mode", i32Attr(0)),
      };
      recipeAttrs.push_back(b.getDictionaryAttr(fields));
    }
    // Phase 2b: rewrite eligible stores into descriptor_store against a
    // synthesized host-side descriptor (symmetric to loads; the
    // descriptor_store TTGIR lowering creates the reg->smem staging +
    // async_tma_copy_local_to_global).
    for (auto &promo : storePromotions) {
      tt::StoreOp storeOp = promo.storeOp;
      int rank = promo.decomp.dims.size();
      Value value = storeOp.getValue();
      auto valTy = cast<RankedTensorType>(value.getType());
      Type elemTy = valTy.getElementType();
      Location loc = storeOp.getLoc();

      builder.setInsertionPoint(storeOp);
      Type i32 = builder.getI32Type();

      // Descriptor for the store: device-built (deviceMode) or host recipe arg.
      Value descVal;
      unsigned descArgIdx = 0;
      Value deviceZeroIdx;
      if (promo.useDeviceMode) {
        SmallVector<int> identity;
        for (int d = 0; d < rank; d++)
          identity.push_back(d);
        descVal =
            buildDeviceDescriptor(storeOp, promo.decomp, promo.shapeArgIndices,
                                  identity, loc, deviceZeroIdx);
      } else {
        auto descTy = tt::TensorDescType::get(
            valTy.getShape(), valTy.getElementType(), valTy.getEncoding());
        descArgIdx = kernelFunc.getNumArguments();
        auto argAttrs = b.getDictionaryAttr(
            {b.getNamedAttr("tt.auto_tma", b.getUnitAttr())});
        LogicalResult inserted =
            kernelFunc.insertArgument(descArgIdx, descTy, argAttrs, loc);
        assert(succeeded(inserted) &&
               "failed to insert TMA descriptor kernel argument");
        (void)inserted;
        descVal = kernelFunc.getArgument(descArgIdx);
      }
      // Same width handling as the load path's toIntWidth: descriptor_store
      // indices must be i32, but a tile offset may be index / i16 / i64 / etc.
      // depending on how the kernel computed it. Sign-extend or truncate as
      // needed; unexpected non-integer types fall through to the op verifier.
      auto toI32 = [&](Value v) -> Value {
        Type ty = v.getType();
        if (ty == i32)
          return v;
        if (ty.isIndex())
          return arith::IndexCastOp::create(builder, loc, i32, v);
        if (auto it = dyn_cast<IntegerType>(ty)) {
          if (it.getWidth() > 32)
            return arith::TruncIOp::create(builder, loc, i32, v);
          return arith::ExtSIOp::create(builder, loc, i32, v);
        }
        return v;
      };
      SmallVector<Value> indices;
      for (int d = 0; d < rank; d++) {
        Value off = promo.decomp.dims[d].offset;
        if (!off)
          indices.push_back(
              deviceZeroIdx ? deviceZeroIdx
                            : arith::ConstantOp::create(
                                  builder, loc, builder.getI32IntegerAttr(0)));
        else
          indices.push_back(toI32(off));
      }

      tt::DescriptorStoreOp::create(builder, loc, descVal, value, indices);
      storeOp.erase();

      // Device stores need no launcher recipe.
      if (promo.useDeviceMode)
        continue;

      SmallVector<Attribute> shapeIdx, strideIdx, blk;
      for (int d = 0; d < rank; d++) {
        shapeIdx.push_back(i32Attr(promo.shapeArgIndices[d]));
        strideIdx.push_back(i32Attr(promo.decomp.dims[d].strideArgIdx));
        blk.push_back(i32Attr((int)promo.decomp.dims[d].blockSize));
      }
      int tmaDtype = getTMADataType(elemTy).value();
      int elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      SmallVector<NamedAttribute> fields = {
          b.getNamedAttr("desc_arg_index", i32Attr((int)descArgIdx)),
          b.getNamedAttr("base_ptr_arg_index",
                         i32Attr(promo.decomp.basePtrArgIndex)),
          b.getNamedAttr("shape_arg_indices", b.getArrayAttr(shapeIdx)),
          b.getNamedAttr("stride_arg_indices", b.getArrayAttr(strideIdx)),
          b.getNamedAttr("block_shape", b.getArrayAttr(blk)),
          b.getNamedAttr("elem_type", i32Attr(tmaDtype)),
          b.getNamedAttr("elem_size", i32Attr(elemBytes)),
          b.getNamedAttr("fp4_padded", i32Attr(0)),
          b.getNamedAttr("fill_mode", i32Attr(0)),
      };
      recipeAttrs.push_back(b.getDictionaryAttr(fields));
    }
    mod->setAttr("ttg.auto_tma_recipes", b.getArrayAttr(recipeAttrs));
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
