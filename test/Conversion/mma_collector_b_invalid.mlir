// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' -verify-diagnostics | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @valid_shared_b
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::b::fill
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::b::lastuse
  // CHECK-NOT: collector::a
  tt.func @valid_shared_b(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                          %a2: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                          %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                          %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                          %d2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                          %pred: i1) {
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_b = "fill"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a2, %b, %d2, %pred, %pred {is_async, ttng.collector_b = "lastuse"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Two M instructions already take the automatic B-collector path.
  tt.func @multiple_m(%a: !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory>,
                      %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                      %d: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %pred: i1) {
    // expected-error@+2 {{B-collector reuse requires a single M=128 MMA instruction on a target with B reuse}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_b = "fill"} : !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @m64(%a: !ttg.memdesc<64x16xf16, #shared, #ttg.shared_memory>,
               %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
               %d: !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %pred: i1) {
    // expected-error@+2 {{B-collector reuse requires a single M=128 MMA instruction on a target with B reuse}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_b = "fill"} : !ttg.memdesc<64x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @bad_action(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                      %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                      %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %pred: i1) {
    // expected-error@+2 {{unsupported B-collector reuse action}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_b = "discard"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
