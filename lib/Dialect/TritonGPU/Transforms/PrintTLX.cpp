#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/SaveAndRestore.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUPRINTTLX
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class TLXPrinter {
public:
  /// The stream for the current function. This is populated when we start
  /// emitting the current function, and cleared when we finish.
  mlir::raw_indented_ostream *os;

  /// The ID to use for the next value we need a variable for. Variables are
  /// generated for SSA values in the form v_<id>.
  unsigned curId = 0;

  /// Map from SSA values we have encountered to their assigned IDs.
  llvm::DenseMap<Value, unsigned> valIds{};

  /// The ID used to use for the next function we need to emit. This is appended
  /// even to functions that have a name, to ensure uniqueness.
  unsigned funcId = 0;

  /// Handle a value by printing its variable name, assigning it an ID if
  /// needed.
  void handleValue(const Value &v) {
    auto [it, inserted] = valIds.try_emplace(v, curId);
    if (inserted)
      ++curId;
    *os << "v_" << it->second;
  }

  /// Print the TLX representation for the given module \p module to stdout.
  void handleModule(ModuleOp module) {
    for (auto op : module.getOps<FuncOp>())
      createFn(op.getBody(), Twine(op.getSymName()) + "_" + Twine(funcId++));
  }

  /// Create a function from the given region \p funcRegion with the given \p
  /// name. The arguments to the first block in the region are treated as the
  /// function arguments.
  void createFn(Region &funcRegion, llvm::Twine name) {
    // Write the contents of this function to a string first, then dump it to
    // stdout once we are done. This ensures that the function will be printed
    // after any nested functions.
    std::string s;
    llvm::raw_string_ostream rso(s);
    {
      mlir::raw_indented_ostream indentedOS(rso);
      llvm::SaveAndRestore osSaver(os, &indentedOS);

      *os << "@triton.jit\ndef " << name << "(";
      bool first = true;
      for (auto arg : funcRegion.getArguments()) {
        if (!first)
          *os << ", ";
        first = false;
        handleValue(arg);
      }
      *os << "):\n";
      raw_indented_ostream::DelimitedScope scope(*os);

      // Print the contents of each block in the function.
      for (Block &block : funcRegion.getBlocks())
        for (Operation &op : block.getOperations())
          handleOperation(&op);
    }
    llvm::outs() << rso.str();
  }

  /// Print the TLX representation for the given operation \p op to \p os.
  void handleOperation(Operation *op) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      handleValue(constOp.getResult());
      *os << " = ";
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        *os << intAttr.getInt() << "\n";
      } else if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        *os << floatAttr.getValueAsDouble() << "\n";
      } else {
        *os << "unknown_const\n";
      }
    } else if (auto gpiOp = dyn_cast<GetProgramIdOp>(op)) {
      handleValue(gpiOp.getResult());
      *os << " = tl.program_id(" << gpiOp.getAxisAsInt() << ")\n";
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      valIds[loadOp.getResult()] = curId++;
      handleValue(loadOp.getResult());
      *os << " = tl.load(";
      handleValue(loadOp.getPtr());
      if (auto mask = loadOp.getMask()) {
        *os << ", ";
        handleValue(mask);
      }
      if (auto other = loadOp.getOther()) {
        *os << ", ";
        handleValue(other);
      }
      *os << ")\n";
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      *os << "tl.store(";
      handleValue(storeOp.getPtr());
      *os << ", ";
      handleValue(storeOp.getValue());
      if (auto mask = storeOp.getMask()) {
        *os << ", ";
        handleValue(mask);
      }
      *os << ")\n";
    } else if (auto addptrOp = dyn_cast<AddPtrOp>(op)) {
      handleValue(addptrOp.getResult());
      *os << " = ";
      handleValue(addptrOp.getPtr());
      *os << " + ";
      handleValue(addptrOp.getOffset());
      *os << "\n";
    } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)) {
      handleValue(addfOp.getResult());
      *os << " = ";
      handleValue(addfOp.getLhs());
      *os << " + ";
      handleValue(addfOp.getRhs());
      *os << "\n";
    } else if (auto addiOp = dyn_cast<arith::AddIOp>(op)) {
      handleValue(addiOp.getResult());
      *os << " = ";
      handleValue(addiOp.getLhs());
      *os << " + ";
      handleValue(addiOp.getRhs());
      *os << "\n";
    } else if (auto muliOp = dyn_cast<arith::MulIOp>(op)) {
      handleValue(muliOp.getResult());
      *os << " = ";
      handleValue(muliOp.getLhs());
      *os << " * ";
      handleValue(muliOp.getRhs());
      *os << "\n";
    } else if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
      handleValue(cmpiOp.getResult());
      *os << " = ";
      handleValue(cmpiOp.getLhs());
      *os << " ";
      auto pred = cmpiOp.getPredicate();
      switch (pred) {
      case arith::CmpIPredicate::eq:
        *os << "==";
        break;
      case arith::CmpIPredicate::ne:
        *os << "!=";
        break;
      case arith::CmpIPredicate::ult:
      case arith::CmpIPredicate::slt:
        *os << "<";
        break;
      case arith::CmpIPredicate::ule:
      case arith::CmpIPredicate::sle:
        *os << "<=";
        break;
      case arith::CmpIPredicate::ugt:
      case arith::CmpIPredicate::sgt:
        *os << ">";
      case arith::CmpIPredicate::uge:
      case arith::CmpIPredicate::sge:
        *os << ">=";
        break;
      }
      *os << " ";
      handleValue(cmpiOp.getRhs());
      *os << "\n";
    } else if (auto rangeOp = dyn_cast<MakeRangeOp>(op)) {
      handleValue(rangeOp.getResult());
      *os << " = tl.arange(";
      if (auto startAttr = op->getAttrOfType<IntegerAttr>("start")) {
        *os << startAttr.getInt();
      } else {
        *os << "0";
      }
      *os << ", ";
      if (auto endAttr = op->getAttrOfType<IntegerAttr>("end")) {
        *os << endAttr.getInt();
      }
      *os << ")\n";
    } else if (auto splatOp = dyn_cast<SplatOp>(op)) {
      handleValue(splatOp.getResult());
      *os << " = tl.full(";
      auto ty = cast<TensorType>(splatOp.getResult().getType());
      *os << "(";
      for (auto dim : ty.getShape()) {
        *os << dim << ", ";
      }
      *os << "), ";
      handleValue(splatOp.getSrc());
      *os << ", ";
      printType(ty.getElementType());
      *os << ")\n";
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      *os << "(";
      for (auto res : reduceOp.getResults()) {
        handleValue(res);
        *os << ", ";
      }
      *os << ") = tl.reduce(";
      *os << "(";
      for (auto src : reduceOp.getSrcs()) {
        handleValue(src);
        *os << ", ";
      }
      *os << "), ";
      *os << reduceOp.getAxis();
      *os << ", ";
      *os << "reduce_fn_" << funcId;
      createFn(reduceOp.getRegion(), "reduce_fn_" + Twine(funcId++));
      *os << ")\n";
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      handleValue(reshapeOp.getResult());
      *os << " = ";
      handleValue(reshapeOp.getSrc());
      *os << ".reshape((";
      for (auto dim : reshapeOp.getResult().getType().getShape())
        *os << dim << ", ";
      *os << "), can_reorder=" << (reshapeOp.getAllowReorder() ? "True" : "False") << ")\n";
    }
    else if (isa<ReturnOp>(op) || isa<ReduceReturnOp>(op)) {
      *os << "return (";
      for (auto res: op->getOperands()) {
        handleValue(res);
        *os << ", ";
      }
        *os << ")\n";
    } else {
      *os << "unknown " << op->getName().getStringRef() << "\n";
    }
  }

  void printType(Type ty) {
    if (auto floatTy = dyn_cast<FloatType>(ty)) {
      *os << "tl.float" << floatTy.getWidth();
      return;
    }
    if (auto iTy = dyn_cast<IntegerType>(ty)) {
      *os << "tl.int" << iTy.getWidth();
      return;
    }
    if (auto ptrTy = dyn_cast<PointerType>(ty)) {
      *os << "tl.pointer_type(";
      printType(ptrTy.getPointeeType());
      *os << ")";
      return;
    }
    *os << "UNKNOWN TYPE " << ty;
  }
};

class TritonGPUPrintTLXPass
    : public impl::TritonGPUPrintTLXBase<TritonGPUPrintTLXPass> {
public:
  void runOnOperation() override { TLXPrinter{}.handleModule(getOperation()); }
};

} // namespace

} // namespace mlir::triton::gpu
