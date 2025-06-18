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

class TritonTTIRChangeVarNamePass : public TritonTTIRChangeVarNameBase<TritonTTIRChangeVarNamePass> {
private:

  int visitDepth = 0;
  void visitOperation(Operation& op) {
    visitDepth++;
    llvm::outs() << "---";
    llvm::outs() << "visitOperation() of " << op.getName() << "at depth " << visitDepth << "\n";
    llvm::outs() << "op.getLoc(): " << op.getLoc() << "\n";
    // llvm::outs() << "op.getOperand(0): " << op.getOperand(0) << "\n";

    llvm::outs() << "op.dump(): " << "\n";
    op.dump();

    for(Value operand: op.getOperands()){
      // llvm ::outs() << "operand.dump() " << "\n";
      // llvm::outs() << "operand.getName(): " << operand.getName() << "\n";
      // llvm::outs() << "operand.getLoc(): " << operand.getLoc() << "\n";
      // llvm::outs() << "operand.dump(): \n";
      // operand.dump();
    }
    llvm::outs() << "---";

    for (Region& region : op.getRegions()){
      visitRegion(region);
    }
    visitDepth --;
  }

  void visitRegion(Region& region) {
    llvm::outs() << "visitRegion() " << "at depth " << visitDepth << "\n";
    visitDepth++;
    // region.dump();
    for (Block& block : region.getBlocks()){
      visitBlock(block);
    }
    visitDepth--;
  }

  void visitBlock(Block& block) {
    llvm::outs() << "visitBlock(), block.dump() of " << "at depth " << visitDepth << "\n";
    visitDepth++;
    llvm::outs() << "block.dump(): " << "\n";
    block.dump();
    for (Operation& op : block.getOperations()){
      visitOperation(op);
    }
    visitDepth--;
  }

public:
  void runOnOperation() override {
    Operation* op = getOperation();
    llvm::outs() << "***** Entering [TritonTTIRHelloWorld.cpp] *****\n";
    llvm::outs() << "Hello! Entering the TritonTTIRHelloWorld Pass: \n";
    llvm::outs() << "op->getName() returns: " << op->getName() << "\n";
    llvm::outs() << "op->getLoc() returns: " << op->getLoc() << "\n";
    llvm::outs() << "op->dump() returns: " << "\n" ;
    /*
      - dive into op->getReagions(),
      - get reference of each instructions
      - read doc, find where name is stored, see what methods can change name
    */


    op->dump();

    // visitOperation(*op);



    // for (Region &region : op->getRegions()){
    //   llvm::outs() << "Region with " << region.getBlocks().size() << " blocks:\n";
    //   for (Block &block : region.getBlocks()){
    //     for (Operation &op1 : block.getOperations()){
    //       llvm::outs() << "\top1->getName(): " << op1.getName() << "\n";
    //       llvm::outs() << "op1.dump(): \n";
    //       for (Region &region1: op1.getRegions()){
    //         llvm::outs() << "\t\rregion1.getBlocks().size(): " << region1.getBlocks().size() << " blocks:\n";
    //         for(Block &block1 : region1.getBlocks()){
    //           for (Operation &op2 : block.getOperations()){
    //             llvm::outs() << "\top2->getName(): " << op2.getName() << "\n";
    //             llvm::outs() << "op2.dump(): \n";
    //             for (Region &region2: op2.getRegions()){
    //               llvm::outs() << "\t\rregion1.getBlocks().size(): " << region2.getBlocks().size() << " blocks:\n";
    //               for(Block &block2 : region2.getBlocks()){
    //                 for (Operation &op3 : block.getOperations()){
    //                   llvm::outs() << "\top2->getName(): " << op3.getName() << "\n";
    //                   llvm::outs() << "op3.dump(): \n";
    //                   op3.dump();
    //                 }
    //               }
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    llvm::outs() << "***** Leaving [TritonTTIRHelloWorld.cpp] *****\n";
  }
};

std::unique_ptr<Pass> triton::createTritonTTIRChangeVarNamePass() {
  return std::make_unique<TritonTTIRChangeVarNamePass>();
}
