#include <memory>
#include <stack>
#include <iostream>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

class TritonTTIRHelloWorldPass : public TritonTTIRHelloWorldBase<TritonTTIRHelloWorldPass> {
private:

public:
  void runOnOperation() override {
    Operation* op = getOperation();
    llvm::outs() << "[HelloWorld] Hello! Entering the TritonTTIRHelloWorld Pass: \n";
    llvm::outs() << "[HelloWorld] op->getName() returns: " << op->getName() << "\n";
    llvm::outs() << "[HelloWorld] op->getLoc() returns: " << op->getLoc() << "\n";
    llvm::outs() << "[HelloWorld] op->dump() returns: " << "\n" ;
    op->dump();
  }
};

std::unique_ptr<Pass> triton::createTritonTTIRHelloWorldPass() {
  return std::make_unique<TritonTTIRHelloWorldPass>();
}
