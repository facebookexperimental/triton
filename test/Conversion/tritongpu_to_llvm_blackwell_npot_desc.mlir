// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' | FileCheck %s

// B is MN-major with 64-column swizzle atoms. N96 spans two atoms, so its
// descriptor needs the next-atom stride (LBO 512 << 16), as N128 does.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @mn_major_b_n96
  // CHECK-DAG: llvm.mlir.constant(32767 : i32)
  // CHECK-DAG: llvm.mlir.constant(33554432 : i32)
  // CHECK: llvm.bitcast %{{.*}} : vector<2xi32> to i64
  // CHECK: llvm.and
  // CHECK: tcgen05.mma.cta_group::1.kind::f16
  tt.func @mn_major_b_n96(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                          %b: !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>,
                          %c: !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>, %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>, !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }

  // CHECK-LABEL: @mn_major_b_n128
  // CHECK-DAG: llvm.mlir.constant(32767 : i32)
  // CHECK-DAG: llvm.mlir.constant(33554432 : i32)
  // CHECK: llvm.bitcast %{{.*}} : vector<2xi32> to i64
  // CHECK: llvm.and
  // CHECK: tcgen05.mma.cta_group::1.kind::f16
  tt.func @mn_major_b_n128(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                           %b: !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>,
                           %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
