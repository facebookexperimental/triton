// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' -verify-diagnostics | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // A mistagged K tile must fail in both assertion-enabled and release builds.
  tt.func @multiple_k(%a: !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>,
                      %b: !ttg.memdesc<32x128xf16, #shared_t, #ttg.shared_memory>,
                      %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %pred: i1) {
    // expected-error@+2 {{A-collector reuse requires a single MMA instruction with operand A in shared memory}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_a = "fill"} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<32x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Multiple M instructions would also enter the B-collector path.
  tt.func @multiple_m(%a: !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory>,
                      %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                      %d: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %pred: i1) {
    // expected-error@+2 {{A-collector reuse requires a single MMA instruction with operand A in shared memory}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_a = "use"} : !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @multiple_n(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                      %b: !ttg.memdesc<16x256xf16, #shared_t, #ttg.shared_memory>,
                      %d: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %pred: i1) {
    // expected-error@+2 {{A-collector reuse requires a single MMA instruction with operand A in shared memory}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_a = "lastuse"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x256xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem_a = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @a_in_tmem(%a: !ttg.memdesc<128x16xf16, #tmem_a, #ttng.tensor_memory>,
                     %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                     %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                     %pred: i1) {
    // expected-error@+2 {{A-collector reuse requires a single MMA instruction with operand A in shared memory}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_a = "fill"} : !ttg.memdesc<128x16xf16, #tmem_a, #ttng.tensor_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @valid_shared_a
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::a::fill
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::a::lastuse
  tt.func @valid_shared_a(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                         %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                         %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                         %d2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                         %pred: i1) {
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_a = "fill"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %d2, %pred, %pred {is_async, ttng.collector_a = "lastuse"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
