#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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

static LogicalResult propagatePlaceholderLayouts(ModuleOp mod) {
  bool conflict = false;
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
      mod.emitError(
          "TLX placeholder layout propagation exceeded iteration limit");
      return failure();
    }
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
        if (!enc)
          return;
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
      // tt.store: value / ptr / mask must share one encoding
      // (SameLoadStoreOperandsEncoding). If the value carries a user pin,
      // propagate it to ptr/mask so make_ttir's verifier accepts the store
      // without a per-op tolerance; concrete layouts resolve later in
      // make_ttgir. Bridge (require_layout convert) the ptr/mask for this use
      // only, leaving their producers (addptr / mask cmp, with their own
      // same-encoding verifiers) untouched.
      if (auto store = dyn_cast<::mlir::triton::StoreOp>(op)) {
        SmallVector<Value> linked{store.getValue(), store.getPtr()};
        if (store.getMask())
          linked.push_back(store.getMask());
        Attribute enc = findPlaceholder(linked, op);
        if (enc) {
          bridgeOrRetype(store.getPtr(), enc, op, /*operandIdx=*/0,
                         /*forceBridge=*/true);
          if (store.getMask())
            bridgeOrRetype(store.getMask(), enc, op, /*operandIdx=*/2,
                           /*forceBridge=*/true);
        }
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
      // tt.expand_dims / tt.broadcast: their result type was fixed at build
      // time, but the operand may only receive a placeholder via the arith/scf
      // propagation above. Re-infer the result via the op's
      // InferTypeOpInterface (broadcast preserves the encoding; expand_dims
      // maps slice<->parent through the TLX infer interface) so operand and
      // result stay consistent.
      if (isa<::mlir::triton::ExpandDimsOp, ::mlir::triton::BroadcastOp,
              ::mlir::triton::ReduceOp>(op)) {
        auto ot = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        if (!ot || !isPlaceholderEncoding(ot.getEncoding()))
          return;
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
        }
        return;
      }
      // tt.call to a @triton.jit helper carrying a pinned (placeholder) arg.
      // Three cases, keyed on what the (monomorphized, encoding-stripped)
      // callee does:
      //   (A) it restructures the tensor (reshape / split / join / trans, e.g.
      //       subtile_ops._split_n_2D): the row-per-thread pin has no meaning
      //       across the restructure, so auto-release the placeholder args
      //       before the call (make_ttgir picks the tail layout) -- the kernel
      //       does not need an explicit tlx.release_layout.
      //   (B/C) it is elementwise (fast_fma) or a reduction (standard.max/sum):
      //       specialize the callee params, then sync the call result to the
      //       callee return -- forcing the placeholder for an elementwise
      //       result (shape matches the arg) or mirroring the fixpoint-inferred
      //       return for a reduction result (shape differs, slice of the pin).
      //       This keeps the kernel on natural tl.max/tl.sum and fast_fma.
      if (auto callOp = dyn_cast<::mlir::triton::CallOp>(op)) {
        Attribute enc;
        ArrayRef<int64_t> encShape;
        for (Value a : callOp.getOperands())
          if (auto t = dyn_cast<RankedTensorType>(a.getType()))
            if (isPlaceholderEncoding(t.getEncoding())) {
              enc = t.getEncoding();
              encShape = t.getShape();
              break;
            }
        auto callee =
            enc ? SymbolTable::lookupNearestSymbolFrom<::mlir::triton::FuncOp>(
                      callOp, callOp.getCalleeAttr())
                : ::mlir::triton::FuncOp();
        if (!callee)
          return;

        // (A) restructuring callee -> release the placeholder args.
        bool restructures = false;
        callee.walk([&](Operation *o) {
          if (isa<::mlir::triton::ReshapeOp, ::mlir::triton::SplitOp,
                  ::mlir::triton::JoinOp, ::mlir::triton::TransOp>(o))
            restructures = true;
        });
        if (restructures) {
          OpBuilder b(callOp);
          for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
            auto t = dyn_cast<RankedTensorType>(callOp.getOperand(i).getType());
            if (!t || !isPlaceholderEncoding(t.getEncoding()))
              continue;
            auto relTy =
                RankedTensorType::get(t.getShape(), t.getElementType());
            Value rel = ReleaseLayoutOp::create(b, callOp.getLoc(), relTy,
                                                callOp.getOperand(i));
            callOp.setOperand(i, rel);
            changed = true;
          }
          return;
        }

        // (B/C) elementwise / reduction: specialize the callee.
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
        Block &entry = callee.getBody().front();
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
          if (retT && isPlaceholderEncoding(retT.getEncoding()))
            target = retT;
          else if (callRt && callRt.getShape() == encShape)
            // Elementwise: result shares the arg shape -> force the
            // placeholder.
            target = RankedTensorType::get(callRt.getShape(),
                                           callRt.getElementType(), enc);
          else
            continue;
          if (callOp.getResult(i).getType() != target) {
            callOp.getResult(i).setType(target);
            changed = true;
          }
          if (i < newResults.size() && newResults[i] != target) {
            newResults[i] = target;
            sigChanged = true;
          }
          for (auto r : rets)
            if (i < r.getNumOperands() && r.getOperand(i).getType() != target) {
              r.getOperand(i).setType(target);
              changed = true;
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
      OpBuilder b(callee);
      auto cloneOp =
          cast<::mlir::triton::FuncOp>(b.clone(*callee.getOperation()));
      std::string base = (callee.getSymName() + "_tlxpin").str();
      std::string name = base;
      unsigned n = 0;
      while (SymbolTable::lookupNearestSymbolFrom(
          callOp, StringAttr::get(callee.getContext(), name)))
        name = base + "_" + std::to_string(n++);
      cloneOp.setSymName(name);
      SymbolTable::setSymbolVisibility(cloneOp,
                                       SymbolTable::Visibility::Private);
      callOp.setCalleeAttr(FlatSymbolRefAttr::get(callee.getContext(), name));
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

    // Make placeholder (no_verify) layouts pinned by tlx.local_load(layout=...)
    // consistent across arith/math elementwise, select and scf region-carried
    // values, so make_ttir's verifier (arith verifiers don't honor no_verify)
    // accepts the module. Runs after the ttg.num-warps metadata above is set,
    // since validating the pinned #linear layout needs it. Concrete layouts are
    // resolved later in make_ttgir.
    if (failed(propagatePlaceholderLayouts(mod)))
      return signalPassFailure();
  }
};

} // namespace mlir::triton::tlx
