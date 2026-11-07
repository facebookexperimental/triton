// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory -verify-diagnostics | FileCheck %s

// An NPOT tile that its pow2 cover's layout maps densely is allocated exactly.

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @n96
// CHECK-SAME: ttg.shared = 24576 : i32
module @n96 attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @n96() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x96xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @n192
// CHECK-SAME: ttg.shared = 49152 : i32
module @n192 attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @n192() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x192xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @m96_one_panel
// CHECK-SAME: ttg.shared = 12288 : i32
module @m96_one_panel attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @m96_one_panel() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<96x64xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @n96_three_stages
// CHECK-SAME: ttg.shared = 73728 : i32
module @n96_three_stages attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @n96_three_stages() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x128x96xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// NPOT rows across two panels put the second panel after 96 rows.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @m96_two_panels
// CHECK-SAME: ttg.shared = 24576 : i32
module @m96_two_panels attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @m96_two_panels() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<96x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// A panel of 100 rows would not start on a swizzle atom, so it keeps the pow2
// size.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @m100_two_panels
// CHECK-SAME: ttg.shared = 32768 : i32
module @m100_two_panels attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @m100_two_panels() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<100x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @pow2_control
// CHECK-SAME: ttg.shared = 32768 : i32
module @pow2_control attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @pow2_control() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// Fewer than 8 rows are not dense, and their 4x128 cover is smaller than the
// NVMMA core tile.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module @fewer_than_8_rows attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @fewer_than_8_rows() {
    // expected-error @+1 {{NPOT memdesc allocation that does not match the encoding's pow2 alloc shape is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x96xf16, #nvmma, #smem, mutable>
    tt.return
  }
}
