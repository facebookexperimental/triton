#include "triton/Dialect/TritonGPU/Transforms/PtxInterpUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

using namespace triton;

// Is this PTX instruction only moving data around, rather than computing?
// Register declarations and moves cannot change a zero into a non-zero.
static bool isPtxDataMovement(StringRef instr) {
  // Match "mov." rather than "mov" so a future mnemonic that merely starts
  // with those three letters is treated as arithmetic (and so rejected) rather
  // than silently assumed zero-preserving. Every real PTX mov variant is
  // dotted: mov.b64, mov.u32, ...
  return instr.starts_with(".reg") || instr.starts_with("mov.");
}

// Recognize a packed mul.f32x2 written as elementwise inline PTX. The kernels
// emit it as a small block, e.g.
//
//   { .reg .b64 ra, rb, rc;
//     mov.b64 ra, { $2, $3 };
//     mov.b64 rb, { $4, $5 };
//     mul.f32x2 rc, ra, rb;
//     mov.b64 { $0, $1 }, rc; }
//
// so a single-instruction match is too strict (it would reject every real
// producer) and a substring match is too loose: a block that multiplies and
// then adds still contains "mul.f32x2" but does not propagate zero. Split the
// block into instructions and require that the multiply is the ONLY arithmetic
// performed; everything else must be a declaration or a data move.
static bool isPackedF32Mul(Operation *op) {
  auto inlineAsm = dyn_cast<ElementwiseInlineAsmOp>(op);
  if (!inlineAsm || !inlineAsm.getPure() || inlineAsm.getPackedElement() != 2 ||
      op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;

  SmallVector<StringRef> instrs;
  inlineAsm.getAsmString().split(instrs, ';');
  bool sawMul = false;
  for (StringRef instr : instrs) {
    // Strip the block braces and surrounding whitespace/newlines so the
    // matching is insensitive to how the asm was indented in Python.
    instr = instr.trim(" \t\n\r{}");
    if (instr.empty())
      continue;
    if (isPtxDataMovement(instr))
      continue;
    if (!instr.starts_with("mul.f32x2") || sawMul)
      return false;
    sawMul = true;
  }
  return sawMul;
}

bool isMulOp(Operation *op) {
  return isa<arith::MulFOp>(op) || isPackedF32Mul(op);
}

} // namespace mlir
