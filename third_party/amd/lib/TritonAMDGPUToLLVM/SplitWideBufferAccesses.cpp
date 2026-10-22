#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "tritonamdgpu-split-wide-buffer-accesses"

using namespace llvm;

namespace {

// This LLVM's AMDGPU backend can only select raw buffer accesses up to 128
// bits wide. The O3 pipeline (VectorCombine) can merge adjacent narrow
// accesses into wider ones which then fail selection with e.g. "Cannot
// select: ... BUFFER_STORE<(store (s256) ...)>". Split those back down into
// 128-bit halves; halves that are still too wide are re-queued and split
// again. Only accesses with a constant-zero aux operand are split: a nonzero
// aux (cache policy, swizzle, volatile) may give the offset operand
// addressing semantics that plain byte arithmetic does not preserve.
constexpr unsigned kMaxBufferAccessBits = 128;

struct BufferAccessInfo {
  Intrinsic::ID iid;
  bool isStore;
  unsigned offsetOpNo; // Byte-offset operand.
};

std::optional<BufferAccessInfo> matchBufferAccess(CallInst *call) {
  Function *callee = call->getCalledFunction();
  if (!callee || !callee->isIntrinsic())
    return std::nullopt;
  switch (callee->getIntrinsicID()) {
  case Intrinsic::amdgcn_raw_ptr_buffer_store:
  case Intrinsic::amdgcn_raw_buffer_store:
    // (vdata, rsrc, offset, soffset, aux).
    return BufferAccessInfo{callee->getIntrinsicID(), /*isStore=*/true, 2};
  case Intrinsic::amdgcn_raw_ptr_buffer_load:
  case Intrinsic::amdgcn_raw_buffer_load:
    // (rsrc, offset, soffset, aux).
    return BufferAccessInfo{callee->getIntrinsicID(), /*isStore=*/false, 1};
  default:
    return std::nullopt;
  }
}

void copyCallMetadata(CallInst *dst, CallInst *src) {
  dst->setCallingConv(src->getCallingConv());
  dst->setTailCallKind(src->getTailCallKind());
  dst->setAttributes(src->getAttributes());
  dst->setDebugLoc(src->getDebugLoc());
  SmallVector<std::pair<unsigned, MDNode *>, 8> allMetadata;
  src->getAllMetadata(allMetadata);
  for (auto &[kind, node] : allMetadata)
    dst->setMetadata(kind, node);
}

// Split one wide access into two halves. Returns the new calls, or empty if
// the access must stay whole.
SmallVector<CallInst *, 2> splitAccess(CallInst *call, BufferAccessInfo info,
                                       IRBuilder<> &builder) {
  Type *dataTy =
      info.isStore ? call->getArgOperand(0)->getType() : call->getType();
  auto *vecTy = dyn_cast<FixedVectorType>(dataTy);
  if (!vecTy)
    return {};
  Type *eltTy = vecTy->getElementType();
  if (!eltTy->isIntegerTy() && !eltTy->isFloatingPointTy())
    return {};
  unsigned numElts = vecTy->getNumElements();
  unsigned eltBits = eltTy->getPrimitiveSizeInBits();
  if (numElts < 2 || numElts % 2 != 0 || eltBits % 8 != 0 ||
      numElts * eltBits <= kMaxBufferAccessBits)
    return {};
  auto *aux = dyn_cast<ConstantInt>(call->getArgOperand(call->arg_size() - 1));
  if (!aux || !aux->isZero())
    return {};

  unsigned half = numElts / 2;
  unsigned halfBytes = half * eltBits / 8;
  auto *halfVecTy = FixedVectorType::get(eltTy, half);

  builder.SetInsertPoint(call);
  Function *splitFn = Intrinsic::getOrInsertDeclaration(
      call->getModule(), info.iid, {halfVecTy});
  Value *offset = call->getArgOperand(info.offsetOpNo);
  Value *offsetHi = builder.CreateAdd(offset, builder.getInt32(halfBytes));

  SmallVector<Value *, 5> argsLo(call->args()), argsHi(call->args());
  argsHi[info.offsetOpNo] = offsetHi;
  if (info.isStore) {
    Value *vdata = call->getArgOperand(0);
    auto *poison = PoisonValue::get(vecTy);
    SmallVector<int, 16> maskLo(half), maskHi(half);
    std::iota(maskLo.begin(), maskLo.end(), 0);
    std::iota(maskHi.begin(), maskHi.end(), half);
    argsLo[0] = builder.CreateShuffleVector(vdata, poison, maskLo);
    argsHi[0] = builder.CreateShuffleVector(vdata, poison, maskHi);
  }
  CallInst *callLo = builder.CreateCall(splitFn, argsLo);
  CallInst *callHi = builder.CreateCall(splitFn, argsHi);
  copyCallMetadata(callLo, call);
  copyCallMetadata(callHi, call);

  if (!info.isStore) {
    SmallVector<int, 32> mask(numElts);
    std::iota(mask.begin(), mask.end(), 0);
    Value *combined = builder.CreateShuffleVector(callLo, callHi, mask);
    call->replaceAllUsesWith(combined);
  }
  return {callLo, callHi};
}

struct SplitWideBufferAccesses : FunctionPass {
  SplitWideBufferAccesses() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    IRBuilder<> builder(F.getContext());
    SmallVector<CallInst *, 16> worklist;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *call = dyn_cast<CallInst>(&I))
          if (!call->isInlineAsm() && matchBufferAccess(call))
            worklist.push_back(call);

    bool changed = false;
    SmallVector<Instruction *> toErase;
    while (!worklist.empty()) {
      CallInst *call = worklist.pop_back_val();
      auto info = matchBufferAccess(call);
      if (!info)
        continue;
      SmallVector<CallInst *, 2> halves = splitAccess(call, *info, builder);
      if (halves.empty())
        continue;
      worklist.append(halves);
      toErase.push_back(call);
      changed = true;
    }
    for (Instruction *I : toErase)
      I->eraseFromParent();
    return changed;
  }

  static char ID;
};

} // end anonymous namespace

char SplitWideBufferAccesses::ID = 0;

namespace mlir::triton::AMD {
void runSplitWideBufferAccessesPass(Function &F) {
  SplitWideBufferAccesses pass;
  pass.runOnFunction(F);
  // If there are no errors, the function returns false.
  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
}
} // namespace mlir::triton::AMD
