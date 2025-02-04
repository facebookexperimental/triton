#include "Utility.h"

namespace mlir::LLVM::CPU {
Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0 && axis < 3);

  // Kinda detouring. Get the kernel function from module.
  FunctionOpInterface funcOp;
  moduleOp.walk([&](FunctionOpInterface f) {
    if (LLVM::isKernel(f)) {
      funcOp = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  assert(funcOp);
  auto args = funcOp.getArguments();
  assert(args.size() >= 6);

  // The first three of the last 6 args are x, y, z program ids.
  auto argIdx = args.size() - 6 + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}
} // namespace mlir::LLVM::CPU
