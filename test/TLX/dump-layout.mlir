// RUN: triton-opt -split-input-file --tlx-dump-layout %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG
// RUN: triton-opt -split-input-file --tlx-dump-layout %s 2>/dev/null | FileCheck %s --check-prefix=IR

// tlx.dump_layout renders the resolved layout in CuTe Shape:Stride notation,
// then erases the op.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Register tensor: CuTe thread-value layout ((thread),(value)):((...),(...)).
  // DIAG: tlx.dump_layout @
  // DIAG: type: tensor<64xf32, #ttg.blocked
  // DIAG: cute: ((_32,_2,_2),_1):((_1,_32,_0),_0)

  // Non-swizzled SMEM buffer: a single strided layout (offset -> element).
  // DIAG: tlx.dump_layout @
  // DIAG: type: !ttg.memdesc<64xf32, #ttg.swizzled_shared
  // DIAG: cute: _64:_1

  // The dump_layout ops are consumed (erased); the rest of the IR is preserved.
  // IR-LABEL: @dump_layout_kernel
  // IR: ttg.local_load
  // IR-NOT: tlx.dump_layout
  tt.func public @dump_layout_kernel(%arg0: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %0 = ttg.local_load %arg0 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>
    tlx.dump_layout %0 : tensor<64xf32, #blocked>
    tlx.dump_layout %arg0 : !ttg.memdesc<64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A real XOR swizzle is rendered as a CuTe swizzle functor composed over the
// base layout: Swizzle<B,M,S> o (base):(stride), derived from the encoding and
// verified against the linear layout.
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // DIAG: tlx.dump_layout @
  // DIAG: cute: Swizzle<2,1,5> o (_32,_32):(_32,_1)

  // IR-LABEL: @swizzled
  // IR-NOT: tlx.dump_layout
  tt.func public @swizzled(%arg0: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    tlx.dump_layout %arg0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}
