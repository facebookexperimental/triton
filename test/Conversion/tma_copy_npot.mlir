// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=84" | FileCheck %s
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file 2>&1 | FileCheck %s --check-prefix=OFF --implicit-check-not="op NPOT"

// With TRITON_ALLOW_NPOT=1, TMA copies an NPOT tile as whole boxes: an NPOT box
// spans its extent and an odd box count adds the message index as a mixed-radix
// digit. Flag off, no NPOT TMA error; an NPOT inner dim fails the shape check.

// 48 rows: one 48-row box.
// CHECK-LABEL: llvm.func @tma_load_48x64
// CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: llvm.icmp "ult" %{{.*}}, %[[N]] : i32
// CHECK-NOT: llvm.urem
// CHECK-COUNT-1: cp.async.bulk.tensor.2d.shared::cluster.global
// CHECK-NOT: cp.async.bulk.tensor
// OFF-LABEL: tt.func @tma_load_48x64
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_48x64(%desc: !tt.tensordesc<48x64xf16, #nvmma>, %alloc: !ttg.memdesc<48x64xf16, #nvmma, #smem, mutable, 64x64>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<48x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<48x64xf16, #nvmma, #smem, mutable, 64x64>
    tt.return
  }
}

// -----

// 96 columns: three 32-column sw64 boxes.
// CHECK-LABEL: llvm.func @tma_load_64x96
// CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(96 : i32) : i32
// CHECK-DAG: %[[R:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG: %[[S:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: llvm.icmp "ult" %{{.*}}, %[[N]] : i32
// CHECK: %[[D:.*]] = llvm.urem %{{.*}}, %[[R]] : i32
// CHECK-NEXT: llvm.shl %[[D]], %[[S]] : i32
// CHECK-COUNT-1: cp.async.bulk.tensor.2d.shared::cluster.global
// CHECK-NOT: cp.async.bulk.tensor
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_64x96(%desc: !tt.tensordesc<64x96xf16, #nvmma>, %alloc: !ttg.memdesc<64x96xf16, #nvmma, #smem, mutable, 64x128>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<64x96xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<64x96xf16, #nvmma, #smem, mutable, 64x128>
    tt.return
  }
}

// -----

// 768 rows: three 256-row boxes.
// CHECK-LABEL: llvm.func @tma_store_768x64
// CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(96 : i32) : i32
// CHECK-DAG: %[[R:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG: %[[S:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: llvm.icmp "ult" %{{.*}}, %[[N]] : i32
// CHECK: %[[D:.*]] = llvm.urem %{{.*}}, %[[R]] : i32
// CHECK-NEXT: llvm.shl %[[D]], %[[S]] : i32
// CHECK-COUNT-1: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
// CHECK-NOT: cp.async.bulk.tensor
// OFF-LABEL: tt.func @tma_store_768x64
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_768x64(%desc: !tt.tensordesc<768x64xf16, #nvmma>, %alloc: !ttg.memdesc<768x64xf16, #nvmma, #smem, 1024x64>, %x: i32) {
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<768x64xf16, #nvmma>, !ttg.memdesc<768x64xf16, #nvmma, #smem, 1024x64>
    tt.return
  }
}

// -----

// 384 rows: a 256-row box doesn't tile them.
// OFF-LABEL: tt.func @tma_load_384x64
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_384x64(%desc: !tt.tensordesc<384x64xf16, #nvmma>, %alloc: !ttg.memdesc<384x64xf16, #nvmma, #smem, mutable, 512x64>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA box 256 does not divide extent 384}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<384x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<384x64xf16, #nvmma, #smem, mutable, 512x64>
    tt.return
  }
}

// -----

// Unswizzled 48-byte rows: the box pitch is narrower than the 64-byte pow2
// alloc shape's.
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_sw0_128x48(%desc: !tt.tensordesc<128x48xf8E4M3FN, #nvmma>, %alloc: !ttg.memdesc<128x48xf8E4M3FN, #nvmma, #smem, mutable, 128x64>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA box pitch differs from its shared layout}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<128x48xf8E4M3FN, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<128x48xf8E4M3FN, #nvmma, #smem, mutable, 128x64>
    tt.return
  }
}

// -----

// 8-byte inner box.
// OFF-LABEL: tt.func @tma_store_48x4
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_48x4(%desc: !tt.tensordesc<48x4xf16, #nvmma>, %alloc: !ttg.memdesc<48x4xf16, #nvmma, #smem, 64x4>, %x: i32) {
    // expected-error @+1 {{NPOT TMA box inner dimension is not a multiple of 16 bytes}}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<48x4xf16, #nvmma>, !ttg.memdesc<48x4xf16, #nvmma, #smem, 64x4>
    tt.return
  }
}

// -----

// fp4-padded tiles keep the padded plan.
// OFF-LABEL: tt.func @tma_load_fp4_padded_48x128
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_fp4_padded_48x128(%desc: !tt.tensordesc<48x128xi8, #nvmma>, %alloc: !ttg.memdesc<48x128xi8, #nvmma, #smem, mutable, 64x128>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA tile needs a tiled, untransposed, unpadded NVMMA layout}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<48x128xi8, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<48x128xi8, #nvmma, #smem, mutable, 64x128>
    tt.return
  }
}

// -----

// Two CTAs split the 96 rows.
// OFF-LABEL: tt.func @tma_store_2cta_96x64
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_2cta_96x64(%desc: !tt.tensordesc<96x64xf16, #nvmma>, %alloc: !ttg.memdesc<96x64xf16, #nvmma, #smem, 128x64>, %x: i32) {
    // expected-error @+1 {{NPOT TMA tile across CTAs is not yet supported}}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<96x64xf16, #nvmma>, !ttg.memdesc<96x64xf16, #nvmma, #smem, 128x64>
    tt.return
  }
}

// -----

// Other encodings keep the existing TMA error.
#sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_swizzled_48x64(%desc: !tt.tensordesc<48x64xf16, #nvmma>, %alloc: !ttg.memdesc<48x64xf16, #sw, #smem, 64x64>, %x: i32) {
    // expected-error @+1 {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<48x64xf16, #nvmma>, !ttg.memdesc<48x64xf16, #sw, #smem, 64x64>
    tt.return
  }
}

// -----

// Power-of-two control: 512 rows are two 256-row boxes and lower.
// CHECK-LABEL: llvm.func @tma_load_512x64
// CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK: llvm.icmp "ult" %{{.*}}, %[[N]] : i32
// CHECK-NOT: llvm.urem
// CHECK-COUNT-1: cp.async.bulk.tensor.2d.shared::cluster.global
// CHECK-NOT: cp.async.bulk.tensor
// OFF-LABEL: tt.func @tma_load_512x64
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_512x64(%desc: !tt.tensordesc<512x64xf16, #nvmma>, %alloc: !ttg.memdesc<512x64xf16, #nvmma, #smem, mutable>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<512x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<512x64xf16, #nvmma, #smem, mutable>
    tt.return
  }
}
