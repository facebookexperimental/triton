#include "WarpUniformity.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::NVIDIA {
namespace detail {

//===----------------------------------------------------------------------===//
// Warp-uniformity analysis.
//===----------------------------------------------------------------------===//
//
// Per NVIDIA, PTX `bar.sync`/`bar.arrive` (the `.aligned` spellings of
// `barrier.sync`/`barrier.arrive`) only require warp-level alignment: all
// threads in a warp must execute the same barrier instruction. A condition
// may itself be data-dependent so long as it is warp-uniform (e.g. an
// iteration count). Divergence across warps -- e.g. different
// warp-specialize partitions, or branches on warp IDs -- is legal.
// Per-thread (intra-warp) divergence is not.
//
// The helpers below conservatively prove warp uniformity. Anything they cannot
// prove uniform is reported non-uniform, and the caller falls back to a form
// with no uniformity requirement. Straight-line code, warp-specialize
// partitions, and branches on constants, program IDs, or warp IDs keep the
// uniform form; branches on thread/lane IDs or on loaded data that can vary
// by thread within a warp -- and thread-varying operands -- do not.

int getUniformityWarpSize(Operation *op) {
  int warpSize = 32;
  if (auto mod = op->getParentOfType<ModuleOp>())
    warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  return warpSize > 0 ? warpSize : 32;
}

// Matches integer constants (arith or LLVM), looking through unrealized
// conversion casts left over from dialect conversion.
std::optional<APInt> matchIntConstant(Value v) {
  while (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      return std::nullopt;
    v = cast->getOperand(0);
  }
  Attribute attr;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    attr = cst.getValue();
  else if (auto cst = v.getDefiningOp<LLVM::ConstantOp>())
    attr = cst.getValue();
  else
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue();
  return std::nullopt;
}

// True if `cst` is a multiple of the warp size. Widens to 64 bits first: a
// narrow constant cannot hold the warp size (e.g. `APInt(5, 32)` truncates to
// 0), which would make the modulus a division by zero.
bool isWarpSizeMultiple(const APInt &cst, int warpSize) {
  unsigned width = std::max(cst.getBitWidth(), 64u);
  APInt wide = cst.zext(width);
  return wide.urem(APInt(width, static_cast<uint64_t>(warpSize))).isZero();
}

// Bit width of integer-like types, treating `index` as 64-bit. Returns 0 for
// non-integer types.
unsigned getIntBitWidth(Type t) {
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  if (isa<IndexType>(t))
    return 64;
  return 0;
}

// True for the thread ID and values derived from it without changing which
// warp each lane belongs to: casts, and addition/subtraction of multiples of
// the warp size (e.g. warp-group-relative thread IDs). Such values still form
// warp-aligned blocks, so dividing them by a multiple of the warp size yields
// a warp ID.
bool isThreadIdLike(Value v, int warpSize) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<::mlir::gpu::ThreadIdOp, NVVM::ThreadIdXOp, NVVM::ThreadIdYOp,
          NVVM::ThreadIdZOp>(def))
    return true;
  // Casts preserve warp-aligned blocks only above the lane-index bits: the
  // sign/truncation boundary must stay at or above the warp size's active bits
  // (6 for a warp of 32). Narrowing below that folds lanes from different
  // warps together, and sign-extending a narrow value remaps lanes within a
  // warp (e.g. lanes 16-31 of a 5-bit thread ID go negative while lanes 0-15
  // stay non-negative), so the quotient is no longer a warp ID. Zero extension
  // keeps values non-negative and is always safe.
  unsigned warpBits =
      APInt(64, static_cast<uint64_t>(warpSize)).getActiveBits();
  if (isa<arith::ExtUIOp, LLVM::ZExtOp>(def))
    return isThreadIdLike(def->getOperand(0), warpSize);
  if (isa<arith::ExtSIOp, LLVM::SExtOp>(def))
    return getIntBitWidth(def->getOperand(0).getType()) >= warpBits &&
           isThreadIdLike(def->getOperand(0), warpSize);
  if (isa<arith::TruncIOp, LLVM::TruncOp>(def))
    return getIntBitWidth(v.getType()) >= warpBits &&
           isThreadIdLike(def->getOperand(0), warpSize);
  if (isa<arith::IndexCastOp>(def) ||
      (isa<UnrealizedConversionCastOp>(def) && def->getNumOperands() == 1)) {
    // `index_cast` (and conversion casts) may widen or narrow, so both the
    // source and destination widths must preserve warp blocks.
    if (getIntBitWidth(def->getOperand(0).getType()) < warpBits ||
        getIntBitWidth(v.getType()) < warpBits)
      return false;
    return isThreadIdLike(def->getOperand(0), warpSize);
  }
  if (isa<arith::AddIOp, arith::SubIOp, LLVM::AddOp, LLVM::SubOp>(def)) {
    auto isMultiple = [&](Value w) {
      auto cst = matchIntConstant(w);
      return cst && isWarpSizeMultiple(*cst, warpSize);
    };
    // `tid - C` keeps lanes in the same warp-aligned block, but `C - tid`
    // reverses lane order within each warp (e.g. `(32 - tid) / 32` is 1 for
    // lane 0 and 0 for lanes 1-31), so the thread-ID-like value must be the
    // minuend. Addition commutes, so either operand may be the multiple.
    bool isSub = isa<arith::SubIOp, LLVM::SubOp>(def);
    if (isThreadIdLike(def->getOperand(0), warpSize) &&
        isMultiple(def->getOperand(1)))
      return true;
    return !isSub && isMultiple(def->getOperand(0)) &&
           isThreadIdLike(def->getOperand(1), warpSize);
  }
  return false;
}

// True for operations computing a warp ID from a thread ID: `tid / C` and
// `tid >> k` are constant within each warp when C (resp. 2^k) is a positive
// multiple of the warp size.
bool isWarpIdComputation(Operation *def, int warpSize) {
  if (isa<arith::DivUIOp, LLVM::UDivOp>(def)) {
    auto cst = matchIntConstant(def->getOperand(1));
    return cst && !cst->isZero() && isWarpSizeMultiple(*cst, warpSize) &&
           isThreadIdLike(def->getOperand(0), warpSize);
  }
  if (isa<arith::ShRUIOp, LLVM::LShrOp>(def)) {
    auto cst = matchIntConstant(def->getOperand(1));
    if (!cst || cst->getActiveBits() > 6)
      return false;
    APInt scale = APInt(64, 1).shl(static_cast<unsigned>(cst->getZExtValue()));
    return isWarpSizeMultiple(scale, warpSize) &&
           isThreadIdLike(def->getOperand(0), warpSize);
  }
  return false;
}

bool isWarpUniformValue(Value v, int warpSize, DenseSet<Value> &visiting,
                         DenseMap<Value, bool> &memo);

bool isWarpUniformBlockArg(BlockArgument arg, int warpSize,
                           DenseSet<Value> &visiting,
                           DenseMap<Value, bool> &memo) {
  Block *block = arg.getOwner();
  Operation *parent = block->getParentOp();
  // Kernel parameters are CTA-uniform. Callees are inlined before lowering,
  // so a function boundary here is the kernel entry.
  if (block->isEntryBlock() && isa<FunctionOpInterface>(parent))
    return true;
  if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
    unsigned idx = arg.getArgNumber();
    // Induction variable: uniform iff the loop bounds are uniform.
    if (idx == 0)
      return isWarpUniformValue(forOp.getLowerBound(), warpSize, visiting,
                                memo) &&
             isWarpUniformValue(forOp.getUpperBound(), warpSize, visiting,
                                memo) &&
             isWarpUniformValue(forOp.getStep(), warpSize, visiting, memo);
    auto yield = cast<scf::YieldOp>(block->getTerminator());
    return isWarpUniformValue(forOp.getInitArgs()[idx - 1], warpSize, visiting,
                              memo) &&
           isWarpUniformValue(yield.getOperands()[idx - 1], warpSize, visiting,
                              memo);
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
    // Before-region args are the op operands on the first iteration and the
    // after-region yields on later iterations; after-region args come from
    // the before-region `scf.condition` args.
    if (block == &whileOp.getBefore().front()) {
      unsigned idx = arg.getArgNumber();
      auto yield =
          cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
      return isWarpUniformValue(whileOp->getOperand(idx), warpSize, visiting,
                                memo) &&
             isWarpUniformValue(yield.getOperands()[idx], warpSize, visiting,
                                memo);
    }
    auto condOp =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    return isWarpUniformValue(condOp.getArgs()[arg.getArgNumber()], warpSize,
                              visiting, memo);
  }
  return false;
}

bool isWarpUniformValueImpl(Value v, int warpSize, DenseSet<Value> &visiting,
                            DenseMap<Value, bool> &memo) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return isWarpUniformBlockArg(arg, warpSize, visiting, memo);
  Operation *def = v.getDefiningOp();
  assert(def && "block arguments handled above");
  // Constants (of any type: the same value is visible to every thread).
  if (def->hasTrait<OpTrait::ConstantLike>())
    return true;
  // CTA-uniform sources: program IDs, block/grid sizes, warp IDs. These are
  // pure 0-operand ops that must not fall through to the generic rule, which
  // fails closed on unknown 0-operand ops (e.g. a new special-register read).
  if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp, ttg::WarpIdOp,
          ::mlir::gpu::BlockIdOp, ::mlir::gpu::GridDimOp,
          ::mlir::gpu::BlockDimOp, NVVM::BlockIdXOp, NVVM::BlockIdYOp,
          NVVM::BlockIdZOp, NVVM::GridDimXOp, NVVM::GridDimYOp,
          NVVM::GridDimZOp, NVVM::BlockDimXOp, NVVM::BlockDimYOp,
          NVVM::BlockDimZOp, NVVM::WarpSizeOp, NVVM::WarpIdOp,
          NVVM::ClusterIdXOp, NVVM::ClusterIdYOp, NVVM::ClusterIdZOp>(def))
    return true;
  // `elect.sync` returns true for exactly one thread: never warp-uniform,
  // regardless of its (uniform) member-mask operand.
  if (isa<NVVM::ElectSyncOp>(def))
    return false;
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def))
    return cast->getNumOperands() == 1 &&
           isWarpUniformValue(cast->getOperand(0), warpSize, visiting, memo);
  if (isWarpIdComputation(def, warpSize))
    return true;
  // Structured-control results: uniform iff the predicate and all incoming
  // values are uniform.
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (ifOp.getElseRegion().empty() ||
        !isWarpUniformValue(ifOp.getCondition(), warpSize, visiting, memo))
      return false;
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
    return isWarpUniformValue(thenYield.getOperands()[idx], warpSize, visiting,
                              memo) &&
           isWarpUniformValue(elseYield.getOperands()[idx], warpSize, visiting,
                              memo);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    // Even constant init/yield values can differ at the exit: a lane that
    // skips the loop returns the init, while another returns the yield.
    return isWarpUniformValue(forOp.getLowerBound(), warpSize, visiting,
                              memo) &&
           isWarpUniformValue(forOp.getUpperBound(), warpSize, visiting,
                              memo) &&
           isWarpUniformValue(forOp.getStep(), warpSize, visiting, memo) &&
           isWarpUniformValue(forOp.getInitArgs()[idx], warpSize, visiting,
                              memo) &&
           isWarpUniformValue(yield.getOperands()[idx], warpSize, visiting,
                              memo);
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    auto cond =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    // Results are the `scf.condition` args, not the after-region yields: a
    // uniform condition can still forward a lane-varying state value
    // (including on the zero-trip path). A lane-dependent condition can also
    // execute the loop a different number of times even if the forwarded
    // value is a constant.
    return isWarpUniformValue(cond.getCondition(), warpSize, visiting, memo) &&
           isWarpUniformValue(cond.getArgs()[idx], warpSize, visiting, memo);
  }
  if (auto execOp = dyn_cast<scf::ExecuteRegionOp>(def)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    // `scf.execute_region` can host multi-block CFG with the yield in a later
    // block; only a single block ending in `scf.yield` is analyzable, anything
    // else fails closed.
    if (!execOp.getRegion().hasOneBlock())
      return false;
    auto yield =
        dyn_cast<scf::YieldOp>(execOp.getRegion().front().getTerminator());
    return yield && idx < yield.getNumOperands() &&
           isWarpUniformValue(yield.getOperands()[idx], warpSize, visiting,
                              memo);
  }
  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (!isWarpUniformValue(switchOp.getArg(), warpSize, visiting, memo))
      return false;
    return llvm::all_of(switchOp->getRegions(), [&](Region &region) {
      auto yield = cast<scf::YieldOp>(region.front().getTerminator());
      return isWarpUniformValue(yield.getOperands()[idx], warpSize, visiting,
                                memo);
    });
  }
  // Any other region-holding op (reductions, warp-specialize partitions,
  // ...) may produce thread-varying results.
  if (def->getNumRegions() > 0)
    return false;
  if (isa<CallOpInterface>(def))
    return false;
  // A pure function of warp-uniform operands is warp-uniform. Require at
  // least one operand so unknown 0-operand ops fail closed (see above).
  if (def->getNumOperands() == 0 || !isMemoryEffectFree(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return isWarpUniformValue(operand, warpSize, visiting, memo);
  });
}

bool isWarpUniformValue(Value v, int warpSize, DenseSet<Value> &visiting,
                        DenseMap<Value, bool> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  // Backedges (loop-carried values) fail closed.
  if (!visiting.insert(v).second)
    return false;
  bool uniform = isWarpUniformValueImpl(v, warpSize, visiting, memo);
  visiting.erase(v);
  memo[v] = uniform;
  return uniform;
}

// True if every conditional branch that can reach `block` (including a
// conditional terminator in `block` itself) has a warp-uniform condition.
// Uniform conditions imply all threads in a warp follow the same path to
// `block`, hence execute it together.
bool reachingBranchesUniform(Block *block, int warpSize,
                             DenseSet<Value> &visiting,
                             DenseMap<Value, bool> &memo) {
  DenseSet<Block *> reaching;
  SmallVector<Block *> worklist{block};
  reaching.insert(block);
  while (!worklist.empty()) {
    for (Block *pred : worklist.pop_back_val()->getPredecessors())
      if (reaching.insert(pred).second)
        worklist.push_back(pred);
  }
  for (Block *cur : reaching) {
    if (cur->empty())
      continue;
    Operation *term = cur->getTerminator();
    if (!term)
      continue;
    Value cond;
    if (auto br = dyn_cast<cf::CondBranchOp>(term))
      cond = br.getCondition();
    else if (auto sw = dyn_cast<cf::SwitchOp>(term))
      cond = sw.getFlag();
    else if (auto br = dyn_cast<LLVM::CondBrOp>(term))
      cond = br.getCondition();
    else if (auto sw = dyn_cast<LLVM::SwitchOp>(term))
      cond = sw.getValue();
    else
      continue;
    if (!isWarpUniformValue(cond, warpSize, visiting, memo))
      return false;
  }
  return true;
}

// True if `op` executes warp-uniformly: whenever one thread in a warp
// executes it, all threads in the warp execute it together.
bool hasUniformExecution(Operation *op, int warpSize,
                         DenseSet<Value> &visiting,
                         DenseMap<Value, bool> &memo) {
  if (Block *block = op->getBlock();
      block && !reachingBranchesUniform(block, warpSize, visiting, memo))
    return false;
  Operation *parent = op->getParentOp();
  if (!parent)
    return false;
  if (isa<FunctionOpInterface>(parent))
    return true;
  if (auto ifOp = dyn_cast<scf::IfOp>(parent))
    return isWarpUniformValue(ifOp.getCondition(), warpSize, visiting, memo) &&
           hasUniformExecution(ifOp, warpSize, visiting, memo);
  if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
    if (!isWarpUniformValue(forOp.getLowerBound(), warpSize, visiting, memo) ||
        !isWarpUniformValue(forOp.getUpperBound(), warpSize, visiting, memo) ||
        !isWarpUniformValue(forOp.getStep(), warpSize, visiting, memo))
      return false;
    return hasUniformExecution(forOp, warpSize, visiting, memo);
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
    auto condOp =
        dyn_cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    return condOp &&
           isWarpUniformValue(condOp.getCondition(), warpSize, visiting,
                              memo) &&
           hasUniformExecution(whileOp, warpSize, visiting, memo);
  }
  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(parent))
    return isWarpUniformValue(switchOp.getArg(), warpSize, visiting, memo) &&
           hasUniformExecution(switchOp, warpSize, visiting, memo);
  if (auto parallelOp = dyn_cast<scf::ParallelOp>(parent)) {
    for (Value bound : parallelOp.getLowerBound())
      if (!isWarpUniformValue(bound, warpSize, visiting, memo))
        return false;
    for (Value bound : parallelOp.getUpperBound())
      if (!isWarpUniformValue(bound, warpSize, visiting, memo))
        return false;
    for (Value step : parallelOp.getStep())
      if (!isWarpUniformValue(step, warpSize, visiting, memo))
        return false;
    return hasUniformExecution(parallelOp, warpSize, visiting, memo);
  }
  // Warp-specialize partitions distribute whole warps: partition bodies belong
  // to the `warp_specialize.partitions` container op, which never splits a
  // warp, so a barrier inside a partition still executes warp-uniformly and
  // analysis keeps searching above the `warp_specialize`.
  if (isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(parent))
    return hasUniformExecution(parent, warpSize, visiting, memo);
  // `scf.execute_region` runs its region unconditionally. Any other
  // region-holding control is unknown: fail closed.
  if (isa<scf::ExecuteRegionOp>(parent))
    return hasUniformExecution(parent, warpSize, visiting, memo);
  return false;
}

} // namespace detail

bool isWarpUniformValue(Value v) {
  Operation *anchor = v.getDefiningOp();
  if (!anchor) {
    if (Block *block = v.getParentBlock())
      anchor = block->getParentOp();
  }
  int warpSize = anchor ? detail::getUniformityWarpSize(anchor) : 32;
  DenseSet<Value> visiting;
  DenseMap<Value, bool> memo;
  return detail::isWarpUniformValue(v, warpSize, visiting, memo);
}

bool hasUniformExecution(Operation *op) {
  int warpSize = detail::getUniformityWarpSize(op);
  DenseSet<Value> visiting;
  DenseMap<Value, bool> memo;
  return detail::hasUniformExecution(op, warpSize, visiting, memo);
}

} // namespace mlir::triton::NVIDIA
