#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/LogicalResult.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::tlx {

#define GEN_PASS_DEF_TRITONTLXFIXUP
#include "tlx/dialect/include/Transforms/Passes.h.inc"

// ---------------------------------------------------------------------------
// Placeholder (no_verify) layout propagation across encoding-uniform ops.
//
// tlx.local_load(layout=...) pins a #tlx.no_verify_layout<#linear> encoding on
// its result already in TTIR. Triton ops (broadcast/expand_dims/reduce/reshape)
// verify layouts through the DialectInferLayoutInterface, which honors
// no_verify (see verifyLayoutsAreEqual). But upstream MLIR arith/math
// elementwise ops (addf, mulf, select, cmpf, truncf, fma, ...) use the generic
// SameOperandsAndResultType-style verifier that compares tensor encodings
// literally and ignores no_verify. So when a pinned tensor meets a
// null-encoded, same-shape sibling at one of those ops, make_ttir's verifier
// would reject the module -- before the real layout resolution
// (tlx-propagate-layout / tlx-resolve-placeholder-layouts) runs in make_ttgir.
//
// This runs first in make_ttir (before the verifier) and stamps the placeholder
// encoding onto every same-shape operand/result of such ops so the module stays
// verifiable. The concrete layout is still resolved later.
static bool isEncodingUniformArithOp(Operation *op) {
  if (isa<arith::ConstantOp>(op))
    return false; // handled as a leaf when retyped, not as a consumer
  // Propagate the placeholder across ops whose generic MLIR verifier compares
  // operand/result *types* (ignoring the TLX wrapper) and would otherwise
  // reject a placeholder meeting a null/concrete sibling: same-type elementwise
  // (SameOperandsAndResultType), select/cmp (whose condition / i1 result differ
  // in type), and the arith cast ops (which change the element type but
  // preserve shape/layout). Keyed on the trait + explicit op list rather than a
  // hard-coded dialect name. NB: deliberately NOT the broad Elementwise trait
  // -- it also matches ops that legitimately mix a pinned and an unpinned
  // operand, which would over-propagate the pin. The element type may differ
  // across the op (casts); only the encoding is propagated.
  if (!op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() &&
      !isa<arith::SelectOp, arith::CmpFOp, arith::CmpIOp, arith::ExtFOp,
           arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
           arith::SIToFPOp, arith::FPToSIOp, arith::BitcastOp>(op))
    return false;
  // Encoding-uniform == every ranked-tensor operand/result shares one shape
  // (true for elementwise / select / cmp). Scalars are ignored.
  ArrayRef<int64_t> shape;
  bool haveShape = false;
  auto sameShape = [&](Type ty) -> bool {
    if (auto t = dyn_cast<RankedTensorType>(ty)) {
      if (!haveShape) {
        shape = t.getShape();
        haveShape = true;
      } else if (t.getShape() != shape) {
        return false;
      }
    }
    return true;
  };
  for (Type ty : op->getOperandTypes())
    if (!sameShape(ty))
      return false;
  for (Type ty : op->getResultTypes())
    if (!sameShape(ty))
      return false;
  return haveShape;
}

static bool hasSameOperandsEncodingTrait(Operation *op) {
  return op->hasTrait<mlir::OpTrait::SameOperandsEncoding>() ||
         op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsEncoding>();
}

static bool hasSameOperandsAndResultEncodingTrait(Operation *op) {
  return op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
         op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>();
}

static bool isPlaceholderEncoding(Attribute enc) {
  if (!enc)
    return false;
  // The deferred pin is marked by #tlx.no_verify_layout, which may be the top
  // wrapper (no_verify<user_layout<L>>), nested under user_layout (SMEM pin:
  // user_layout<no_verify<L>>), or nested inside a ttg encoding produced by
  // inference -- e.g. a reduce result slice<parent=no_verify<...>>. Scan the
  // whole attribute tree for no_verify. Deliberately keyed on no_verify (not a
  // bare user_layout) so a user_layout-only pin (which is not a deferred
  // placeholder) is left untouched by this fixup.
  if (hasNoVerifyLayout(enc))
    return true;
  bool found = false;
  enc.walkImmediateSubElements(
      [&](Attribute sub) { found |= isPlaceholderEncoding(sub); }, [](Type) {});
  return found;
}

static bool retypeWithEncoding(Value v, Attribute enc) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t || t.getEncoding() == enc)
    return false;
  auto newTy = RankedTensorType::get(t.getShape(), t.getElementType(), enc);
  // A tensor arith.constant carries its type in the value attr too; rebuild it.
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      if (dense.isSplat())
        cst.setValueAttr(
            DenseElementsAttr::get(newTy, dense.getSplatValue<Attribute>()));
  }
  v.setType(newTy);
  return true;
}

// Give one call site a private copy of a shared @triton.jit helper before its
// tensor ABI is specialized.  Both concrete and deferred layout propagation
// use this path; keeping the cloning mechanics in one place also guarantees
// that their generated symbols cannot collide.
static void privatizeHelperForCall(::mlir::triton::CallOp call,
                                   ::mlir::triton::FuncOp callee,
                                   StringRef suffix) {
  OpBuilder b(callee);
  auto clone = cast<::mlir::triton::FuncOp>(b.clone(*callee.getOperation()));
  std::string base = (callee.getSymName() + suffix).str();
  std::string name = base;
  unsigned n = 0;
  while (SymbolTable::lookupNearestSymbolFrom(
      call, StringAttr::get(callee.getContext(), name)))
    name = base + "_" + std::to_string(n++);
  clone.setSymName(name);
  SymbolTable::setSymbolVisibility(clone, SymbolTable::Visibility::Private);
  call.setCalleeAttr(FlatSymbolRefAttr::get(callee.getContext(), name));
}

// Triton's Python frontend monomorphizes @triton.jit helpers with
// encoding-free tensor argument and result types. TLX primitives can
// nevertheless pass or produce concrete distributed values at a helper
// boundary (for example dot operands loaded from LDS or AMD MFMA
// accumulators). Repair that temporary ABI before the module verifier runs:
// infer helper inputs from concrete call operands, infer results from concrete
// return operands, update unreachable poison returns, mirror the repaired
// types on calls, and keep any scf.for ABI crossed by those values concrete.
static LogicalResult synchronizeConcreteHelperABI(ModuleOp mod) {
  auto isConcreteDistributed = [](Type ty) {
    auto tensorTy = dyn_cast<RankedTensorType>(ty);
    Attribute enc = tensorTy ? tensorTy.getEncoding() : Attribute();
    return enc && !hasNoVerifyLayout(enc) &&
           isa<ttg::DistributedEncodingTrait>(enc);
  };

  bool changed = true;
  unsigned iteration = 0;
  constexpr unsigned kMaxIterations = 64;
  while (changed) {
    changed = false;
    if (++iteration > kMaxIterations)
      return mod.emitError(
          "TLX concrete helper-ABI synchronization did not converge");

    // A frontend-monomorphized helper can be shared by call sites carrying
    // different concrete layouts.  Privatize every concrete call while the
    // original ABI is still encoding-free; otherwise the first call would
    // specialize the shared function and make a later, equally valid layout
    // look like a conflict.  Encoding-free callers retain the original.
    SmallVector<std::pair<::mlir::triton::CallOp, ::mlir::triton::FuncOp>>
        concreteClones;
    mod.walk([&](::mlir::triton::CallOp call) {
      if (!llvm::any_of(call.getOperandTypes(), isConcreteDistributed))
        return;
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.getBody().empty())
        return;
      auto uses = SymbolTable::getSymbolUses(callee, mod);
      if (!uses)
        return;
      unsigned useCount = 0;
      for (const auto &use : *uses) {
        (void)use;
        ++useCount;
      }
      if (useCount > 1)
        concreteClones.emplace_back(call, callee);
    });
    for (auto [call, callee] : concreteClones) {
      privatizeHelperForCall(call, callee, "_tlxabi");
      changed = true;
    }

    bool inputConflict = false;
    mod.walk([&](::mlir::triton::CallOp call) {
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.getBody().empty())
        return;

      SmallVector<Type> inputTypes(callee.getFunctionType().getInputs().begin(),
                                   callee.getFunctionType().getInputs().end());
      Block &entry = callee.getBody().front();
      bool signatureChanged = false;
      for (unsigned i = 0; i < call.getNumOperands() && i < inputTypes.size() &&
                           i < entry.getNumArguments();
           ++i) {
        Type actual = call.getOperand(i).getType();
        Type expected = inputTypes[i];
        Type target =
            isConcreteDistributed(actual)
                ? actual
                : (isConcreteDistributed(expected) ? expected : Type());
        if (!target)
          continue;

        auto targetTy = cast<RankedTensorType>(target);
        auto actualTy = dyn_cast<RankedTensorType>(actual);
        auto expectedTy = dyn_cast<RankedTensorType>(expected);
        if (!actualTy || !expectedTy ||
            actualTy.getShape() != targetTy.getShape() ||
            actualTy.getElementType() != targetTy.getElementType() ||
            expectedTy.getShape() != targetTy.getShape() ||
            expectedTy.getElementType() != targetTy.getElementType()) {
          call.emitError() << "helper argument " << i
                           << " has inconsistent tensor payload type";
          inputConflict = true;
          return;
        }
        if ((isConcreteDistributed(actual) && actual != target) ||
            (isConcreteDistributed(expected) && expected != target)) {
          call.emitError() << "conflicting concrete layouts for helper "
                              "argument "
                           << i;
          inputConflict = true;
          return;
        }

        // A second call may still carry an encoding-free value after another
        // call specialized the shared callee. Bridge only this use instead of
        // changing the producer's type out from under its other users.
        if (actual != target) {
          OpBuilder b(call);
          Value converted = RequireLayoutOp::create(b, call.getLoc(), target,
                                                    call.getOperand(i));
          call.setOperand(i, converted);
          changed = true;
        }
        if (entry.getArgument(i).getType() != target) {
          entry.getArgument(i).setType(target);
          changed = true;
        }
        if (inputTypes[i] != target) {
          inputTypes[i] = target;
          signatureChanged = true;
        }
      }
      if (signatureChanged) {
        callee.setType(
            FunctionType::get(callee.getContext(), inputTypes,
                              callee.getFunctionType().getResults()));
        changed = true;
      }
    });
    if (inputConflict)
      return failure();

    SmallVector<::mlir::triton::FuncOp> funcs;
    mod.walk([&](::mlir::triton::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      SmallVector<::mlir::triton::ReturnOp> returns;
      func.walk([&](::mlir::triton::ReturnOp ret) { returns.push_back(ret); });
      if (returns.empty() || func.getFunctionType().getNumResults() == 0)
        continue;

      SmallVector<Type> resultTypes(func.getFunctionType().getResults().begin(),
                                    func.getFunctionType().getResults().end());
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        Type target =
            isConcreteDistributed(resultTypes[i]) ? resultTypes[i] : Type();
        for (auto ret : returns) {
          if (i >= ret.getNumOperands())
            continue;
          Type candidate = ret.getOperand(i).getType();
          if (!isConcreteDistributed(candidate))
            continue;
          if (target && target != candidate)
            return ret.emitError()
                   << "conflicting concrete layouts for helper result " << i;
          target = candidate;
        }
        if (!target)
          continue;

        if (resultTypes[i] != target) {
          resultTypes[i] = target;
          changed = true;
        }
        auto targetTy = cast<RankedTensorType>(target);
        for (auto ret : returns) {
          if (i >= ret.getNumOperands())
            continue;
          Value operand = ret.getOperand(i);
          auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTy)
            continue;
          if (operandTy.getShape() != targetTy.getShape() ||
              operandTy.getElementType() != targetTy.getElementType())
            return ret.emitError() << "helper result " << i
                                   << " has inconsistent shape or element type";
          if (isConcreteDistributed(operandTy) && operandTy != targetTy)
            return ret.emitError()
                   << "conflicting concrete layouts for helper result " << i;
          if (operandTy != targetTy) {
            // Specialize this return edge without retyping its producer.  A
            // value may also feed another result that intentionally remains
            // encoding-free (or deferred), so mutating the SSA value here can
            // silently change an unrelated result's ABI.
            OpBuilder b(ret);
            Value converted =
                RequireLayoutOp::create(b, ret.getLoc(), targetTy, operand);
            ret.setOperand(i, converted);
            changed = true;
          }
        }
      }

      if (func.getFunctionType().getResults() != ArrayRef<Type>(resultTypes)) {
        func.setType(FunctionType::get(func.getContext(),
                                       func.getFunctionType().getInputs(),
                                       resultTypes));
        changed = true;
      }
    }

    mod.walk([&](::mlir::triton::CallOp call) {
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee)
        return;
      auto calleeResults = callee.getFunctionType().getResults();
      if (calleeResults.size() != call.getNumResults())
        return;
      for (unsigned i = 0; i < calleeResults.size(); ++i) {
        if (call.getResult(i).getType() != calleeResults[i]) {
          call.getResult(i).setType(calleeResults[i]);
          changed = true;
        }
      }
    });

    // A helper argument can acquire its concrete layout above after the
    // Python frontend has already built a loop with encoding-free region
    // iter-arguments/results. Likewise, specializing a helper return can
    // retype an scf.for result through the enclosing tt.return without
    // updating the loop body. Repair all four pieces of each loop-carried
    // value together: init, iter-arg, yield, and result.
    bool loopConflict = false;
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        SmallVector<Value> linked{
            forOp.getInitArgs()[i],
            forOp.getRegionIterArg(i),
            yield.getOperand(i),
            forOp.getResult(i),
        };
        Type target;
        for (Value value : linked) {
          Type candidate = value.getType();
          if (!isConcreteDistributed(candidate))
            continue;
          if (target && target != candidate) {
            forOp.emitError()
                << "conflicting concrete layouts for loop-carried value " << i;
            loopConflict = true;
            break;
          }
          target = candidate;
        }
        if (loopConflict || !target)
          continue;

        auto targetTy = cast<RankedTensorType>(target);
        for (Value value : linked) {
          auto valueTy = dyn_cast<RankedTensorType>(value.getType());
          if (!valueTy || valueTy.getShape() != targetTy.getShape() ||
              valueTy.getElementType() != targetTy.getElementType()) {
            forOp.emitError() << "loop-carried value " << i
                              << " has inconsistent tensor payload type";
            loopConflict = true;
            break;
          }
        }
        if (loopConflict)
          continue;

        Value init = forOp.getInitArgs()[i];
        if (init.getType() != target) {
          OpBuilder b(forOp);
          Value converted =
              RequireLayoutOp::create(b, forOp.getLoc(), target, init);
          forOp.setOperand(forOp.getNumControlOperands() + i, converted);
          changed = true;
        }
        if (forOp.getRegionIterArg(i).getType() != target) {
          forOp.getRegionIterArg(i).setType(target);
          changed = true;
        }
        Value yielded = yield.getOperand(i);
        if (yielded.getType() != target) {
          OpBuilder b(yield);
          Value converted =
              RequireLayoutOp::create(b, yield.getLoc(), target, yielded);
          yield.setOperand(i, converted);
          changed = true;
        }
        if (forOp.getResult(i).getType() != target) {
          forOp.getResult(i).setType(target);
          changed = true;
        }
      }
    }
    if (loopConflict)
      return failure();
  }
  return success();
}

static void appendForwardedValues(OpOperand &use,
                                  SmallVectorImpl<Value> &worklist) {
  Operation *user = use.getOwner();
  if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    unsigned index = use.getOperandNumber();
    Operation *parent = yield->getParentOp();
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      if (index < ifOp.getNumResults())
        worklist.push_back(ifOp.getResult(index));
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (index < forOp.getNumResults()) {
        worklist.push_back(forOp.getRegionIterArg(index));
        worklist.push_back(forOp.getResult(index));
      }
    }
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    unsigned index = use.getOperandNumber();
    if (index >= forOp.getNumControlOperands()) {
      index -= forOp.getNumControlOperands();
      if (index < forOp.getNumRegionIterArgs()) {
        worklist.push_back(forOp.getRegionIterArg(index));
        worklist.push_back(forOp.getResult(index));
      }
    }
    return;
  }
  // An scf.if operand is its condition, not a value forwarded to its results.
  // An explicit release ends the pinned value's provenance by definition.
  if (isa<scf::IfOp, ReleaseLayoutOp>(user) || !isMemoryEffectFree(user))
    return;
  for (Value result : user->getResults())
    if (isa<RankedTensorType>(result.getType()))
      worklist.push_back(result);
}

static SmallVector<Value> collectForwardedValues(Value root) {
  SmallVector<Value> worklist{root};
  SmallVector<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (llvm::is_contained(visited, value))
      continue;
    visited.push_back(value);
    for (OpOperand &use : value.getUses())
      appendForwardedValues(use, worklist);
  }
  return visited;
}

static bool flowsPreservingEncodingToValue(Value root, Value target) {
  SmallVector<Value> worklist{root};
  SmallVector<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (llvm::is_contained(visited, value))
      continue;
    visited.push_back(value);
    if (value == target)
      return true;

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<scf::YieldOp, scf::ForOp>(user)) {
        appendForwardedValues(use, worklist);
        continue;
      }
      if (!isEncodingUniformArithOp(user) &&
          !hasSameOperandsAndResultEncodingTrait(user))
        continue;
      for (Value result : user->getResults())
        if (isa<RankedTensorType>(result.getType()))
          worklist.push_back(result);
    }
  }
  return false;
}

static LogicalResult reconcileVerifierLayouts(ModuleOp mod) {
  bool conflict = false;
  // Concrete tlx.require_layout results are explicit local ownership anchors.
  // Track only those anchors and the encoding-uniform results derived from
  // them: an arbitrary concrete tensor type selected elsewhere in TTIR is not
  // authority to retag its producer graph.
  llvm::DenseSet<Value> concreteLayoutAnchors;
  mod.walk([&](RequireLayoutOp op) {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (type && type.getEncoding() &&
        !isPlaceholderEncoding(type.getEncoding()))
      concreteLayoutAnchors.insert(op.getResult());
  });

  // Find a placeholder encoding among a set of linked values (values an op
  // requires to share one encoding). If two of them carry *different* pinned
  // placeholders, that is a user error (two conflicting tlx.local_load(layout=)
  // pins meeting on one op) -- report it instead of silently retagging one.
  auto findPlaceholder = [&](ArrayRef<Value> vs, Operation *op) -> Attribute {
    Attribute enc;
    for (Value v : vs)
      if (auto t = dyn_cast<RankedTensorType>(v.getType()))
        if (isPlaceholderEncoding(t.getEncoding())) {
          if (enc && enc != t.getEncoding()) {
            op->emitError("conflicting user-pinned layouts meet on this "
                          "operation; insert tlx.release_layout to reconcile "
                          "them before combining");
            conflict = true;
            return enc;
          }
          if (!enc)
            enc = t.getEncoding();
        }
    return enc;
  };

  // Find the concrete layout required by an explicitly anchored value in an
  // encoding-uniform operation. Two distinct anchors are an ambiguous source
  // contract: neither one may silently override the other.
  auto findConcreteAnchor = [&](ArrayRef<Value> vs,
                                Operation *op) -> Attribute {
    Attribute enc;
    for (Value v : vs) {
      if (!concreteLayoutAnchors.contains(v))
        continue;
      auto type = cast<RankedTensorType>(v.getType());
      Attribute candidate = type.getEncoding();
      if (enc && enc != candidate) {
        op->emitError("conflicting explicit layouts meet on this operation; "
                      "insert tlx.release_layout before combining them");
        conflict = true;
        return {};
      }
      enc = candidate;
    }
    return enc;
  };

  bool changed = true;
  unsigned iter = 0;
  constexpr unsigned kMaxIter = 1000;
  // Give a linked value the placeholder encoding. Values defined by
  // arith.constant or tt.call cannot be retyped in place (rebuilding a
  // constant's value attr, or a call result bound to the callee signature,
  // leaves the IR inconsistent once resolve strips the wrapper), nor can a
  // select condition's producer -- bridge those to the consumer with a
  // require_layout convert; everything else is retyped directly. `user`/
  // `operandIdx` identify the use to rewrite in the bridge case.
  auto bridgeOrRetype = [&](Value v, Attribute enc, Operation *user,
                            unsigned operandIdx, bool forceBridge = false) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getEncoding() == enc)
      return;
    bool isConst = v.getDefiningOp<arith::ConstantOp>() != nullptr;
    bool isCall = v.getDefiningOp<::mlir::triton::CallOp>() != nullptr;
    if (forceBridge || isConst || isCall) {
      OpBuilder b(user);
      auto convTy =
          RankedTensorType::get(t.getShape(), t.getElementType(), enc);
      Value conv = RequireLayoutOp::create(b, user->getLoc(), convTy, v);
      user->setOperand(operandIdx, conv);
      changed = true;
      return;
    }
    changed |= retypeWithEncoding(v, enc);
  };
  while (changed) {
    changed = false;
    if (++iter > kMaxIter) {
      mod.emitError("TLX verifier layout reconciliation exceeded iteration "
                    "limit");
      return failure();
    }

    // A helper can establish a pin internally and return it without receiving
    // any pinned argument.  The frontend still gives that helper and its call
    // sites encoding-free result types.  Synchronize those result ABIs from
    // the return operands before walking calls, including flattened aggregate
    // results and unreachable poison returns.  Input-driven call
    // specialization below cannot discover this case because there is no
    // placeholder on the call operands.
    SmallVector<::mlir::triton::FuncOp> funcs;
    mod.walk([&](::mlir::triton::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      SmallVector<::mlir::triton::ReturnOp> returns;
      func.walk([&](::mlir::triton::ReturnOp ret) { returns.push_back(ret); });
      if (returns.empty() || func.getFunctionType().getNumResults() == 0)
        continue;

      SmallVector<Type> resultTypes(func.getFunctionType().getResults().begin(),
                                    func.getFunctionType().getResults().end());
      bool signatureChanged = false;
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        SmallVector<Value> candidates;
        for (auto ret : returns)
          if (i < ret.getNumOperands())
            candidates.push_back(ret.getOperand(i));
        Attribute enc = findPlaceholder(candidates, func);
        if (!enc)
          continue;

        auto declared = dyn_cast<RankedTensorType>(resultTypes[i]);
        if (!declared) {
          func.emitError() << "pinned helper result " << i
                           << " is not a ranked tensor";
          conflict = true;
          continue;
        }
        Type target = RankedTensorType::get(declared.getShape(),
                                            declared.getElementType(), enc);
        for (auto ret : returns) {
          if (i >= ret.getNumOperands())
            continue;
          Value operand = ret.getOperand(i);
          auto type = dyn_cast<RankedTensorType>(operand.getType());
          if (!type || type.getShape() != declared.getShape() ||
              type.getElementType() != declared.getElementType()) {
            ret.emitError() << "pinned helper result " << i
                            << " has inconsistent tensor payload type";
            conflict = true;
            continue;
          }
          if (type.getEncoding() && type.getEncoding() != enc) {
            ret.emitError()
                << "conflicting layout for pinned helper result " << i;
            conflict = true;
            continue;
          }
          if (type != target) {
            OpBuilder b(ret);
            Value converted =
                RequireLayoutOp::create(b, ret.getLoc(), target, operand);
            ret.setOperand(i, converted);
            changed = true;
          }
        }
        if (resultTypes[i] != target) {
          resultTypes[i] = target;
          signatureChanged = true;
        }
      }
      if (signatureChanged) {
        func.setType(FunctionType::get(func.getContext(),
                                       func.getFunctionType().getInputs(),
                                       resultTypes));
        changed = true;
      }
    }
    if (conflict)
      return failure();

    // Mirror repaired helper result types onto every call before result users
    // participate in this fixpoint iteration.
    mod.walk([&](::mlir::triton::CallOp call) {
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee)
        return;
      auto results = callee.getFunctionType().getResults();
      if (results.size() != call.getNumResults())
        return;
      for (unsigned i = 0; i < results.size(); ++i)
        if (call.getResult(i).getType() != results[i]) {
          call.getResult(i).setType(results[i]);
          changed = true;
        }
    });

    // tt.calls whose shared helper callee must be privatized (cloned) before it
    // can be specialized. Collected during the walk and applied afterwards, so
    // we never insert into the module while traversing it.
    SmallVector<::mlir::triton::CallOp> pendingClones;
    mod.walk([&](Operation *op) {
      // arith/math elementwise, select, cmp: all same-shape tensor operands and
      // results must share the encoding.
      if (isEncodingUniformArithOp(op)) {
        SmallVector<Value> linked(op->getOperands());
        linked.append(op->getResults().begin(), op->getResults().end());
        Attribute enc = findPlaceholder(linked, op);
        if (enc) {
          for (Value v : op->getResults())
            changed |= retypeWithEncoding(v, enc);
          bool isSelect = isa<arith::SelectOp>(op);
          for (auto en : llvm::enumerate(op->getOperands())) {
            // arith.select's condition (operand 0) must carry the encoding too,
            // but its producer may be un-retypeable (e.g. a tt.call mask), so
            // force a require_layout bridge for it.
            bool isSelectCond = isSelect && en.index() == 0;
            bridgeOrRetype(en.value(), enc, op, en.index(), isSelectCond);
          }
          return;
        }

        enc = findConcreteAnchor(linked, op);
        if (!enc)
          return;

        // The operation owns its result types, so carrying the anchor forward
        // is safe. Operand producers remain untouched: bridge only this use so
        // helper calls, broadcasts, and other shared values keep their own
        // contracts and normal layout propagation can later fold a free
        // conversion or materialize a real one.
        for (Value v : op->getResults())
          changed |= retypeWithEncoding(v, enc);
        for (Value v : op->getResults())
          if (isa<RankedTensorType>(v.getType()))
            concreteLayoutAnchors.insert(v);
        for (auto en : llvm::enumerate(op->getOperands()))
          bridgeOrRetype(en.value(), enc, op, en.index(),
                         /*forceBridge=*/true);
        return;
      }
      // Ops declaring a same-encoding trait keep their ranked tensor values in
      // one layout domain. Use the trait rather than an op name so helper
      // specialization works for all current and future load/store operations.
      bool sameOperands = hasSameOperandsEncodingTrait(op);
      bool sameOperandsAndResults = hasSameOperandsAndResultEncodingTrait(op);
      if (sameOperands || sameOperandsAndResults) {
        SmallVector<Value> linked;
        for (Value operand : op->getOperands())
          if (isa<RankedTensorType>(operand.getType()))
            linked.push_back(operand);
        if (sameOperandsAndResults)
          for (Value result : op->getResults())
            if (isa<RankedTensorType>(result.getType()))
              linked.push_back(result);
        Attribute enc = findPlaceholder(linked, op);
        if (enc) {
          if (sameOperandsAndResults)
            for (Value result : op->getResults())
              changed |= retypeWithEncoding(result, enc);
          for (auto indexed : llvm::enumerate(op->getOperands()))
            if (isa<RankedTensorType>(indexed.value().getType()))
              bridgeOrRetype(indexed.value(), enc, op, indexed.index(),
                             /*forceBridge=*/true);
        }
        // Operand-only traits (e.g. tt.reduce) do not describe result layouts;
        // continue to the op's inference interface below.
        if (sameOperandsAndResults)
          return;
      }
      // scf.for: init / region iter-arg / yield operand / result of each
      // loop-carried value must share the encoding.
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          Value initV = forOp.getInitArgs()[i];
          Value iterArg = forOp.getRegionIterArg(i);
          Value yieldV = yield.getOperand(i);
          Value resV = forOp.getResult(i);
          Attribute enc =
              findPlaceholder({initV, iterArg, yieldV, resV}, forOp);
          if (!enc)
            continue;
          // init is an external operand and the yield operand is loop-body
          // produced -- either may be an arith.constant (tl.zeros) or a tt.call
          // result, so bridge those; the iter-arg and result are the loop's own
          // SSA values and are retyped in place.
          bridgeOrRetype(initV, enc, forOp, forOp.getNumControlOperands() + i);
          changed |= retypeWithEncoding(iterArg, enc);
          bridgeOrRetype(yieldV, enc, yield, i);
          changed |= retypeWithEncoding(resV, enc);
        }
        return;
      }
      // scf.if: result / then-yield / else-yield of each value must share it.
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto thenY = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        scf::YieldOp elseY =
            ifOp.elseBlock()
                ? cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator())
                : nullptr;
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
          SmallVector<Value> linked{ifOp.getResult(i), thenY.getOperand(i)};
          if (elseY)
            linked.push_back(elseY.getOperand(i));
          Attribute enc = findPlaceholder(linked, ifOp);
          if (!enc)
            continue;
          // The if result is the op's own SSA value (retype in place); the
          // branch yields are produced values that may be arith.constant /
          // tt.call results, so bridge them.
          changed |= retypeWithEncoding(ifOp.getResult(i), enc);
          bridgeOrRetype(thenY.getOperand(i), enc, thenY, i);
          if (elseY)
            bridgeOrRetype(elseY.getOperand(i), enc, elseY, i);
        }
        return;
      }
      // Re-infer any op whose operands acquired a placeholder after frontend
      // construction. This covers trans/split/reduce/expand-dims and future ops
      // implementing InferTypeOpInterface.
      Attribute operandEnc;
      for (Value operand : op->getOperands())
        if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
          if (isPlaceholderEncoding(type.getEncoding())) {
            operandEnc = type.getEncoding();
            break;
          }
      if (operandEnc && !isa<::mlir::triton::CallOp>(op)) {
        if (auto iface = dyn_cast<InferTypeOpInterface>(op)) {
          SmallVector<Type> inferred;
          if (succeeded(iface.inferReturnTypes(
                  op->getContext(), op->getLoc(), op->getOperands(),
                  op->getAttrDictionary(), op->getPropertiesStorage(),
                  op->getRegions(), inferred)) &&
              inferred.size() == op->getNumResults()) {
            for (unsigned i = 0; i < inferred.size(); ++i)
              if (op->getResult(i).getType() != inferred[i]) {
                op->getResult(i).setType(inferred[i]);
                changed = true;
              }
          }
          return;
        }
        if (auto reshape = dyn_cast<::mlir::triton::ReshapeOp>(op)) {
          auto srcTy = reshape.getSrc().getType();
          auto dstTy = reshape.getType();
          Attribute dstEnc = dstTy.getEncoding();
          auto *layout = cast<::mlir::triton::DialectInferLayoutInterface>(
              &operandEnc.getDialect());
          if (succeeded(layout->inferReshapeOpEncoding(
                  srcTy.getShape(), operandEnc, dstTy.getShape(), dstEnc,
                  reshape.getAllowReorder(), reshape.getLoc())))
            changed |= retypeWithEncoding(reshape.getResult(), dstEnc);
          return;
        }
        if (auto join = dyn_cast<::mlir::triton::JoinOp>(op)) {
          SmallVector<Value> linked{join.getLhs(), join.getRhs()};
          Attribute srcEnc = findPlaceholder(linked, op);
          if (!srcEnc)
            return;
          bridgeOrRetype(join.getLhs(), srcEnc, op, 0);
          bridgeOrRetype(join.getRhs(), srcEnc, op, 1);
          auto srcTy = join.getLhs().getType();
          Attribute dstEnc;
          auto *layout = cast<::mlir::triton::DialectInferLayoutInterface>(
              &srcEnc.getDialect());
          if (succeeded(layout->inferDefaultJoinOpEncoding(
                  srcEnc, dstEnc, srcTy.getShape(), join.getLoc())))
            changed |= retypeWithEncoding(join.getResult(), dstEnc);
          return;
        }
      }
      // tt.call to a @triton.jit helper carrying a pinned placeholder argument.
      // Specialize the encoding-stripped callee signature, then let the
      // fixpoint re-infer its body and synchronize its return/call result
      // types. The original pin remains authoritative across the function
      // boundary.
      if (auto callOp = dyn_cast<::mlir::triton::CallOp>(op)) {
        Attribute enc;
        for (Value a : callOp.getOperands())
          if (auto t = dyn_cast<RankedTensorType>(a.getType()))
            if (isPlaceholderEncoding(t.getEncoding())) {
              enc = t.getEncoding();
              break;
            }
        auto callee =
            enc ? SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
                      callOp, callOp.getCalleeAttr())
                : ::mlir::triton::FuncOp();
        if (!callee)
          return;

        Block &entry = callee.getBody().front();

        // If the monomorphized helper is shared with other call sites (possibly
        // unpinned or pinned to a different layout), defer privatizing it:
        // clone a private copy after the walk and point this call at it, so
        // specializing never changes types out from under other callers or
        // toggles a shared callee between two pins across fixpoint iterations.
        if (auto symUses = SymbolTable::getSymbolUses(callee, mod)) {
          unsigned useCount = 0;
          for (const auto &u : *symUses) {
            (void)u;
            ++useCount;
          }
          if (useCount > 1) {
            pendingClones.push_back(callOp);
            return;
          }
        }
        // Triton monomorphizes @triton.jit helpers with encoding-stripped
        // (null) signatures, so specialize params (entry block args +
        // FunctionType inputs) to the placeholder; nested helper calls and the
        // callee body (e.g. the reduction's tt.reduce) are re-inferred by the
        // fixpoint once their params are set.
        SmallVector<Type> newInputs(
            callee.getFunctionType().getInputs().begin(),
            callee.getFunctionType().getInputs().end());
        bool inputsChanged = false;
        for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
          Type at = callOp.getOperand(i).getType();
          auto rt = dyn_cast<RankedTensorType>(at);
          if (!rt || !isPlaceholderEncoding(rt.getEncoding()))
            continue;
          if (i < newInputs.size() && newInputs[i] != at) {
            newInputs[i] = at;
            inputsChanged = true;
          }
          if (i < entry.getNumArguments() &&
              entry.getArgument(i).getType() != at) {
            entry.getArgument(i).setType(at);
            changed = true;
          }
        }
        if (inputsChanged)
          changed = true;

        SmallVector<::mlir::triton::ReturnOp> rets;
        callee.walk([&](::mlir::triton::ReturnOp r) { rets.push_back(r); });
        SmallVector<Type> newResults(
            callee.getFunctionType().getResults().begin(),
            callee.getFunctionType().getResults().end());
        bool sigChanged = inputsChanged;
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          auto callRt =
              dyn_cast<RankedTensorType>(callOp.getResult(i).getType());
          // Reduction: mirror the callee's return operand once the fixpoint has
          // re-inferred it to a placeholder (slice of the pin).
          RankedTensorType retT;
          if (!rets.empty() && i < rets[0].getNumOperands())
            retT = dyn_cast<RankedTensorType>(rets[0].getOperand(i).getType());
          Type target;
          if (retT && isPlaceholderEncoding(retT.getEncoding())) {
            target = retT;
          } else if (callRt && !rets.empty() && i < rets[0].getNumOperands()) {
            // Elementwise: specialize only when the return value actually
            // descends from a surviving pinned argument with the same shape.
            // This avoids re-pinning a result derived from another argument
            // whose layout was released before a restructure.
            Value retV = rets[0].getOperand(i);
            for (unsigned argIdx = 0; argIdx < callOp.getNumOperands();
                 ++argIdx) {
              auto argT = dyn_cast<RankedTensorType>(
                  callOp.getOperand(argIdx).getType());
              if (!argT || !isPlaceholderEncoding(argT.getEncoding()) ||
                  argT.getShape() != callRt.getShape() ||
                  argIdx >= entry.getNumArguments() ||
                  !flowsPreservingEncodingToValue(entry.getArgument(argIdx),
                                                  retV))
                continue;
              target = RankedTensorType::get(callRt.getShape(),
                                             callRt.getElementType(),
                                             argT.getEncoding());
              break;
            }
          }
          if (!target)
            continue;
          if (callOp.getResult(i).getType() != target) {
            callOp.getResult(i).setType(target);
            changed = true;
          }
          if (i < newResults.size() && newResults[i] != target) {
            newResults[i] = target;
            sigChanged = true;
          }
          for (auto r : rets) {
            if (i < r.getNumOperands() && r.getOperand(i).getType() != target) {
              // Specialize only this return edge. The returned value may also
              // feed another result or an unrelated use, and constants cannot
              // be retyped without rebuilding their value attribute.
              OpBuilder b(r);
              Value converted = RequireLayoutOp::create(b, r.getLoc(), target,
                                                        r.getOperand(i));
              r.setOperand(i, converted);
              changed = true;
            }
          }
        }
        if (sigChanged) {
          callee.setType(
              FunctionType::get(callee.getContext(), newInputs, newResults));
          changed = true;
        }
        return;
      }
    });
    // Privatize shared helper callees collected during the walk. Cloning gives
    // each pinned call site its own copy of the helper, so specializing it
    // cannot change types out from under other callers.
    for (auto callOp : pendingClones) {
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
              callOp, callOp.getCalleeAttr());
      if (!callee)
        continue;
      privatizeHelperForCall(callOp, callee, "_tlxpin");
      changed = true;
    }
    if (conflict)
      return failure();
  }
  return success();
}

class TritonTLXFixupPass : public impl::TritonTLXFixupBase<TritonTLXFixupPass> {
  using impl::TritonTLXFixupBase<TritonTLXFixupPass>::TritonTLXFixupBase;

public:
  // validate the module and error early for unsupported cases
  LogicalResult verifyModule(ModuleOp &mod, bool tlx_2cta) {
    // ws should not capture RankedTensorType
    ttg::WarpSpecializeOp invalidWSOp = nullptr;
    auto result = mod.walk([&](ttg::WarpSpecializeOp op) {
      for (auto argType : op.getPartitionOp().getOperandTypes()) {
        if (isa<RankedTensorType>(argType)) {
          invalidWSOp = op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return invalidWSOp.emitError() << "WarpSpecializeOp should not capture "
                                        "RankedTensorType. Try moving tensor "
                                        "computation into specific async task.";
    }

    if (tlx_2cta) {
      if (numCTAs > 1) {
        return mod.emitError()
               << "num_ctas should not be set for TLX 2cta mode";
      }

      // all the async_dot ops need to be either 1cta or 2cta together
      auto walkResult = mod.walk([&](ttng::TCGen5MMAOp tcgen05MMAOp) {
        if (!tcgen05MMAOp.getTwoCtas()) {
          tcgen05MMAOp.emitError()
              << "Expecting all dot ops to be 2cta together or 1cta together";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        return failure();
      }
    } else {
      bool isClustered = false;
      if (!clusterDims.empty()) {
        // Ensure we have exactly 3 dimensions (X, Y, Z)
        if (clusterDims.size() != 3) {
          return mod.emitError()
                 << "Expected 3 cluster dimensions, got " << clusterDims.size();
        }
        isClustered = (clusterDims[0] * clusterDims[1] * clusterDims[2]) > 1;
      }
      // There should not be a mapa in unclustered mode
      if (!isClustered) {
        if (mod.walk([&](ttng::MapToRemoteBufferOp mapaOp) {
                 mapaOp.emitError()
                     << "Unexpected buffer remote view in 1cta mode";
                 return WalkResult::interrupt();
               })
                .wasInterrupted()) {
          return failure();
        }
      }
    }
    return success();
  }

  bool isAMD() const {
    // target is set up as f"hip:{options.arch}"
    return (target.getValue().find("hip:") == 0);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto hasTLXTwoCTAs =
        mod.walk([&](Operation *op) {
             if (auto tcgen05MMAOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
               if (tcgen05MMAOp.getTwoCtas()) {
                 return WalkResult::interrupt();
               }
             } else if (auto tcgen05MMAScaledOp =
                            dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
               if (tcgen05MMAScaledOp.getTwoCtas()) {
                 return WalkResult::interrupt();
               }
             }
             return WalkResult::advance();
           })
            .wasInterrupted();

    if (failed(verifyModule(mod, hasTLXTwoCTAs))) {
      return signalPassFailure();
    }

    // First check if there is any TLX related op in the module. If not, do
    // nothing.
    auto tlxDialectName = TLXDialect::getDialectNamespace();
    WalkResult result = mod.walk([&](Operation *op) {
      // Ops directly in TLX Dialect
      if (op->getDialect()->getNamespace() == tlxDialectName) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    auto hasTLXOps = result.wasInterrupted();

    auto hasExplicitLocalMemAccess =
        mod.walk([&](Operation *op) {
             // Ops that should not be in TTIR unless introduced by TLX
             if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp,
                     ttng::AsyncTMACopyGlobalToLocalOp,
                     ttng::AsyncTMACopyLocalToGlobalOp, ttng::TMEMAllocOp,
                     ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TCGen5MMAOp>(
                     op)) {
               return WalkResult::interrupt();
             }
             return WalkResult::advance();
           })
            .wasInterrupted();

    auto hasWarpSpecOps = mod.walk([&](Operation *op) {
                               if (isa<ttg::WarpSpecializeOp, ttg::WarpYieldOp,
                                       ttg::WarpReturnOp>(op)) {
                                 return WalkResult::interrupt();
                               }
                               return WalkResult::advance();
                             })
                              .wasInterrupted();

    if (!hasTLXOps && !hasExplicitLocalMemAccess && !hasWarpSpecOps &&
        !hasTLXTwoCTAs) {
      return;
    }

    // Attach metadata to the module.
    Builder b(&getContext());
    mod->setAttr(kSkipGenericPipelineAttrName, b.getUnitAttr());
    mod->setAttr(ttg::AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(ttg::AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(ttg::AttrTargetName, b.getStringAttr(this->target.getValue()));
    if (hasTLXOps)
      mod->setAttr(AttrHasTLXOpsName, b.getBoolAttr(true));
    if (hasExplicitLocalMemAccess)
      mod->setAttr(AttrHasExplicitLocalMemAccessName, b.getBoolAttr(true));
    if (hasWarpSpecOps)
      mod->setAttr(AttrHasWarpSpecOpsName, b.getBoolAttr(true));
    if (hasTLXTwoCTAs) {
      mod->setAttr(AttrTLXEnablePairedCTAMMAName, b.getBoolAttr(true));
    }

    // Propagate the `exclusive` marker from `tlx.async_tasks(exclusive=True)`:
    // it enables the single-warp-specialize lowering, which requires the module
    // to contain exactly one warp_specialize op. Only NVIDIA consumes this; on
    // AMD `exclusive` is ignored.
    if (!isAMD()) {
      int numWarpSpecializeOps = 0;
      bool hasExclusiveWS = false;
      bool hasNoEndingClusterSync = false;
      std::optional<int32_t> mbarrierTryWaitSuspendNs;
      mod.walk([&](ttg::WarpSpecializeOp op) {
        ++numWarpSpecializeOps;
        if (op->hasAttr("tlx.exclusive"))
          hasExclusiveWS = true;
        if (op->hasAttr("tlx.no_ending_cluster_sync"))
          hasNoEndingClusterSync = true;
        if (auto attr = op->getAttrOfType<IntegerAttr>(
                "tlx.mbarrier_try_wait_suspend_ns")) {
          int32_t value = attr.getInt();
          if (!mbarrierTryWaitSuspendNs || value < *mbarrierTryWaitSuspendNs)
            mbarrierTryWaitSuspendNs = value;
        }
      });
      if (mbarrierTryWaitSuspendNs)
        mod->setAttr("tlx.mbarrier_try_wait_suspend_ns",
                     b.getI32IntegerAttr(*mbarrierTryWaitSuspendNs));
      if (hasExclusiveWS) {
        if (numWarpSpecializeOps != 1) {
          mod.emitError()
              << "tlx.async_tasks(exclusive=True) requires exactly one "
                 "warp_specialize op in the module, but found "
              << numWarpSpecializeOps;
          return signalPassFailure();
        }
        ttg::setHasSingleWarpSpecialize(mod, /*value=*/true);
      }
      // `no_ending_cluster_sync`: the user handles the post-warp-specialize
      // sync, so mark the module to skip the compiler's cluster arrive/wait
      // before TMEM dealloc.
      if (hasNoEndingClusterSync)
        setUserPostWsSyncOnMod(mod, /*value=*/true);
    }

    // First specialize helper ABIs carrying concrete distributed values, then
    // reconcile explicit layouts across verifier-uniform operations so
    // make_ttir's verifier accepts the module. Placeholder (no_verify) layouts
    // pinned by tlx.local_load(layout=...) propagate through linked producer
    // graphs; concrete require_layout anchors constrain only their local
    // arithmetic chain and bridge mismatched inputs at the consuming use. Runs
    // after the ttg.num-warps metadata above is set, since validating a pinned
    // #linear layout needs it. Concrete conversions resolve in make_ttgir.
    if (failed(synchronizeConcreteHelperABI(mod)))
      return signalPassFailure();
    if (failed(reconcileVerifierLayouts(mod)))
      return signalPassFailure();
  }
};

} // namespace mlir::triton::tlx
