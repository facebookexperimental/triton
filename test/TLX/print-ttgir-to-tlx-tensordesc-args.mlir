// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file | FileCheck %s

// A host-side TensorDescriptor reaches the kernel as a !tt.tensordesc followed
// by its shape and strides, named desc.shape.N / desc.stride.N. Value names are
// sanitized into Python identifiers before the signature is printed, so the
// expansion has to be recognized in that spelling; left in, every launch of the
// regenerated kernel is missing its arguments.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def dot_qualified_expansion(sm_scale, desc_q, desc_k, N):
  tt.func public @dot_qualified_expansion(%sm_scale: f32 loc("sm_scale"), %desc_q: !tt.tensordesc<128x128xbf16, #shared> loc("desc_q"), %desc_q.shape.0: i32 loc("desc_q.shape.0"), %desc_q.shape.1: i32 loc("desc_q.shape.1"), %desc_q.stride.0: i64 loc("desc_q.stride.0"), %desc_q.stride.1: i64 loc("desc_q.stride.1"), %desc_k: !tt.tensordesc<128x128xbf16, #shared> loc("desc_k"), %desc_k.shape.0: i32 loc("desc_k.shape.0"), %desc_k.shape.1: i32 loc("desc_k.shape.1"), %desc_k.stride.0: i64 loc("desc_k.stride.0"), %desc_k.stride.1: i64 loc("desc_k.stride.1"), %N: i32 loc("N")) attributes {noinline = false} {
    tt.return
  }

  // A scalar that merely shares the descriptor's prefix is a real argument.
  // CHECK-LABEL: def prefix_is_not_expansion(desc_q, desc_q_shape_extra):
  tt.func public @prefix_is_not_expansion(%desc_q: !tt.tensordesc<128x128xbf16, #shared> loc("desc_q"), %desc_q_shape_extra: i32 loc("desc_q_shape_extra")) attributes {noinline = false} {
    tt.return
  }
}
