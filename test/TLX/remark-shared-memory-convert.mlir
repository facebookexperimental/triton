// RUN: env MLIR_ENABLE_DIAGNOSTICS=remarks triton-opt -split-input-file --tlx-dump-layout -verify-diagnostics %s -o /dev/null
// RUN: env MLIR_ENABLE_DIAGNOSTICS=warnings,remarks triton-opt -split-input-file --tlx-dump-layout -verify-diagnostics %s -o /dev/null
// RUN: env -u MLIR_ENABLE_DIAGNOSTICS triton-opt -split-input-file --tlx-dump-layout %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
// RUN: env MLIR_ENABLE_DIAGNOSTICS=warnings triton-opt -split-input-file --tlx-dump-layout %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
// RUN: env MLIR_ENABLE_DIAGNOSTICS=notremarks triton-opt -split-input-file --tlx-dump-layout %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty

// QUIET-NOT: remark:
// QUIET-NOT: note:

// convert_layout through shared memory gets a remark. When a shared-memory load
// feeds the conversion, a note suggests a candidate layout.

// One row per thread: the TMEM 32x32b register layout of a 128x32 tile.
#row = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // A value in the TMEM layout is converted to meet an SMEM-loaded operand.
  // The candidate load layout may move the conversion to the function return,
  // so the note must not promise to eliminate it.
  tt.func public @peer_local_load(%g: tensor<128x32xf32, #row>, %buf: !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>) -> tensor<128x32xf32, #blocked> {
    // expected-note @below {{loading this shared-memory value in layout ((_32,_4),_32):((_32,_1024),_1) may remove the conversion; in TLX: tlx.local_load(..., layout=tlx.layout(shape=((32, 4), (32,)), stride=((32, 1024), (1,))))}}
    %x = ttg.local_load %buf : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> tensor<128x32xbf16, #blocked>
    %xf = arith.extf %x : tensor<128x32xbf16, #blocked> to tensor<128x32xf32, #blocked>
    // expected-remark @below {{convert_layout through shared memory for 128x32xf32: from ((_32,_4),_32):((_32,_1024),_1) to}}
    %gc = ttg.convert_layout %g : tensor<128x32xf32, #row> -> tensor<128x32xf32, #blocked>
    %r = arith.mulf %gc, %xf : tensor<128x32xf32, #blocked>
    tt.return %r : tensor<128x32xf32, #blocked>
  }
}

// -----

#row = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // The converted value itself comes from shared memory, inside a loop: load it
  // in the destination layout.
  tt.func public @converted_local_load(%buf: !ttg.memdesc<128x32xf32, #shared, #smem, mutable>, %out: tensor<128x32x!tt.ptr<f32>, #row>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      // expected-note @below {{loading this shared-memory value in layout ((_32,_4),_32):((_32,_1024),_1) may remove the conversion}}
      %l = ttg.local_load %buf : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> tensor<128x32xf32, #blocked>
      // expected-remark @below {{convert_layout through shared memory inside a loop for 128x32xf32}}
      %c = ttg.convert_layout %l : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #row>
      tt.store %out, %c : tensor<128x32x!tt.ptr<f32>, #row>
    }
    tt.return
  }

  // Changing the load layout can require a conversion for its other user.
  tt.func public @multi_use_local_load(%buf: !ttg.memdesc<128x32xf32, #shared, #smem, mutable>, %row_out: tensor<128x32x!tt.ptr<f32>, #row>, %blocked_out: tensor<128x32x!tt.ptr<f32>, #blocked>) {
    // expected-note @below {{may remove the conversion; in TLX: tlx.local_load}}
    %l = ttg.local_load %buf : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> tensor<128x32xf32, #blocked>
    tt.store %blocked_out, %l : tensor<128x32x!tt.ptr<f32>, #blocked>
    // expected-remark @below {{convert_layout through shared memory for 128x32xf32}}
    %c = ttg.convert_layout %l : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #row>
    tt.store %row_out, %c : tensor<128x32x!tt.ptr<f32>, #row>
    tt.return
  }
}

// -----

// A register-only reorder needs no shared memory: no remark.
#row = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#row_swapped = #ttg.linear<{register = [[0, 2], [0, 1], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @register_reorder(%g: tensor<128x32xf32, #row>) -> tensor<128x32xf32, #row_swapped> {
    %c = ttg.convert_layout %g : tensor<128x32xf32, #row> -> tensor<128x32xf32, #row_swapped>
    tt.return %c : tensor<128x32xf32, #row_swapped>
  }
}
