// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=103 ptx-version=94' -verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // The B collector is Rubin-only.
  tt.func @pre_rubin(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                     %b: !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>,
                     %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                     %pred: i1) {
    // expected-error@+2 {{B-collector reuse requires a single M=128 MMA instruction on a target with B reuse}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %pred, %pred {is_async, ttng.collector_b = "fill"} : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared_t, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
